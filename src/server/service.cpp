#include "service.hpp"

#include "log.hpp"
#include "images.hpp"
#include "keepawake.hpp"
#include "telemetry.hpp"
#include "vision.hpp"
#include "src/unicode.h"   // llama.cpp's: code point classes and lower case (the loop check's words)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <unordered_map>

namespace lamina {

double now_s() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

double mono_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

namespace {

/// Python's repr of a str, as the log lines quote text
std::string py_repr_str(const std::string& s) {
    const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
    const char q = dq ? '"' : '\'';
    std::string out(1, q);
    char buf[16];
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const uint32_t c = utf8_at(s, i, &n);
        if (c == '\\') out += "\\\\";
        else if (c == (uint32_t) q) out += std::string("\\") + q;
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20 || c == 0x7F) {
            std::snprintf(buf, sizeof buf, "\\x%02x", c);
            out += buf;
        } else {
            out.append(s, i, n);
        }
        i += n;
    }
    return out + q;
}

std::string py_list_repr(const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) out += (i ? ", " : "") + py_repr_str(items[i]);
    return out + "]";
}

const std::map<std::string, std::string> kEffortText = {
    {"low", "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
            "without unnecessary elaboration."},
    {"medium", "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary elaboration."},
    // the template's own xhigh sentence, which the loop recovery swaps for the low one
    {"xhigh", "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
              "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
              "answer."}};

}  // namespace

// ---------------------------------------------------------------------------------------------- helpers
std::vector<std::string> stop_strings(const json& req) {
    json stop;
    if (!jget(req, "stop").is_null()) {
        stop = req["stop"];
        if (stop.is_string()) stop = json::array({stop});
        bool ok = stop.is_array();
        if (ok)
            for (const auto& x : stop) ok = ok && x.is_string();
        if (!ok) throw RequestError("stop must be a string or a list of strings");
        if (stop.size() > 4) throw RequestError("stop takes at most 4 strings");
    } else if (!jget(req, "stop_sequences").is_null()) {
        stop = req["stop_sequences"];
        bool ok = stop.is_array();
        if (ok)
            for (const auto& x : stop) ok = ok && x.is_string();
        if (!ok) throw RequestError("stop_sequences must be a list of strings");
    } else {
        return {};
    }
    std::vector<std::string> out;
    for (const auto& x : stop)
        if (!x.get_ref<const std::string&>().empty()) out.push_back(x.get<std::string>());
    return out;
}

std::string StopMatcher::push(const std::string& text) {
    if (hit.has_value()) return "";
    const std::string buf = held_ + text;
    std::optional<std::pair<size_t, std::string>> first;
    for (const auto& s : stops_) {
        const size_t i = buf.find(s);
        if (i != std::string::npos && (!first || i < first->first)) first = std::make_pair(i, s);
    }
    if (first) {
        held_.clear();
        hit = first->second;
        return buf.substr(0, first->first);
    }
    size_t keep = 0;   // the longest end of buf that a stop string starts with
    for (const auto& s : stops_)
        for (size_t k = std::min(s.size() - 1, buf.size()); k > keep; --k)
            if (ends_with(buf, std::string_view(s).substr(0, k))) {
                keep = k;
                break;
            }
    held_ = keep ? buf.substr(buf.size() - keep) : "";
    return buf.substr(0, buf.size() - keep);
}

std::string StopMatcher::flush() {
    std::string h;
    h.swap(held_);
    return h;
}

void check_request_sampling(json& req) {
    if (req.is_object()) {   // #1819: llama.cpp's names; the OpenAI-style field wins when both are sent
        if (!jget(req, "repeat_penalty").is_null() && jget(req, "repetition_penalty").is_null())
            req["repetition_penalty"] = req["repeat_penalty"];
        const json& rl = jget(req, "repeat_last_n");
        if (jget(req, "penalty_last_n").is_null() && is_int(rl) && rl.get<int64_t>() > 0) req["penalty_last_n"] = rl;
    }
    const std::pair<const char*, bool> keys[] = {{"temperature", false},  {"top_p", false},           {"min_p", false},
                                                 {"presence_penalty", false}, {"frequency_penalty", false},
                                                 {"repetition_penalty", false}, {"top_k", true}, {"penalty_last_n", true},
                                                 {"seed", true}};
    for (const auto& [key, integer] : keys) {
        const json value = jget(req, key);
        if (value.is_null()) continue;
        const std::string what = std::string(key) + ": " + (integer ? "an integer" : "a number");
        if (value.is_string() && !strip(value.get_ref<const std::string&>()).empty()) {
            // some clients send numbers as strings: read and used
            const std::string s(strip(value.get<std::string>()));
            char* end = nullptr;
            const double num = std::strtod(s.c_str(), &end);
            if (end == s.c_str() || *end != '\0' || std::isnan(num) || std::isinf(num)) throw RequestError(what);
            const bool whole = num == std::floor(num) && std::fabs(num) < 9.2e18;
            if (integer) {
                if (!whole) throw RequestError(what);
                req[key] = (int64_t) num;
            } else {
                bool int_text = !s.empty();
                for (size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0; i < s.size(); ++i) int_text = int_text && std::isdigit((unsigned char) s[i]);
                req[key] = int_text && whole ? json((int64_t) num) : json(num);
            }
            continue;
        }
        if (value.is_boolean() || !(integer ? value.is_number_integer() : value.is_number())) throw RequestError(what);
    }
    // the sampler divides the repeated tokens' logits by it: 0 made them infinite
    if (const json& rp = jget(req, "repetition_penalty"); rp.is_number() && !(rp.get<double>() > 0))
        throw RequestError("repetition_penalty: a number greater than 0");
}

json clean_shared_defaults(const json& d) {
    if (d.is_null()) return json::object();
    if (!d.is_object()) throw RequestError("defaults must be an object");
    json out = json::object();
    for (auto it = d.begin(); it != d.end(); ++it) {
        const std::string& key = it.key();
        json value = it.value();
        if (value.is_null() || (value.is_string() && value.get_ref<const std::string&>().empty())) continue;
        const bool number = value.is_number();
        const double v = number ? value.get<double>() : 0.0;
        if (key == "reasoning_effort") {
            const std::string e = value.is_string() ? value.get<std::string>() : "";
            if (e != "none" && e != "low" && e != "medium" && e != "high")
                throw RequestError("reasoning_effort: none, low, medium or high");
        } else if (key == "temperature") {
            if (!number || !(v >= 0 && v <= 2)) throw RequestError("temperature: 0..2");
        } else if (key == "top_p") {
            if (!number || !(v > 0 && v <= 1)) throw RequestError("top_p: 0 < top_p <= 1");
        } else if (key == "top_k") {
            if (!number || v != std::floor(v) || !(v >= 1 && v <= 64)) throw RequestError("top_k: an integer 1..64");
            value = (int64_t) v;
        } else if (key == "seed" || key == "max_tokens") {
            if (!number || v != std::floor(v) || v <= 0) throw RequestError(key + ": a positive integer");
            value = (int64_t) v;
        } else if (key == "reasoning_budget_tokens") {   // #123: 0 means no budget, so 0 is allowed here
            if (!number || v != std::floor(v) || v < 0)
                throw RequestError("reasoning_budget_tokens: a whole number of tokens (0: no budget)");
            value = (int64_t) v;
        } else {
            throw RequestError("unknown setting '" + key + "'");
        }
        out[key] = key == "temperature" || key == "top_p" ? json(v) : value;
    }
    return out;
}

json sampling_defaults_from_config(const json& cfg) {
    json out = json::object();
    json block = jget(cfg, "sampling").is_object() ? cfg["sampling"] : json::object();
    for (const auto& [alias, canon] : {std::pair<const char*, const char*>{"repeat_penalty", "repetition_penalty"},
                                       {"repeat_last_n", "penalty_last_n"}}) {
        if (!jget(block, alias).is_null()) {
            if (!jget(block, canon).is_null())
                throw std::runtime_error(std::string("config sampling: both '") + alias + "' and '" + canon +
                                         "' are set; they are the same setting, keep one");
            block[canon] = block[alias];
            block.erase(alias);
        }
    }
    for (auto it = block.begin(); it != block.end(); ++it) {
        const std::string& key = it.key();
        const json& value = it.value();
        if (value.is_null()) continue;
        const bool number = value.is_number();
        const double v = number ? value.get<double>() : 0.0;
        const std::string shown = "config sampling." + key + "=" + py_repr_value(value);
        auto bad = [&](const std::string& want) { throw std::runtime_error(shown + ": " + want); };
        if (key == "temperature") {
            if (!number || v < 0) bad("expected a number >= 0 (0 = greedy)");
            out[key] = v;
        } else if (key == "top_p") {
            if (!number || !(v > 0 && v <= 1)) bad("expected 0 < top_p <= 1");
            out[key] = v;
        } else if (key == "min_p") {
            if (!number || !(v >= 0 && v <= 1)) bad("expected 0 <= min_p <= 1");
            out[key] = v;
        } else if (key == "top_k") {
            if (!number || v != std::floor(v) || !(v >= 1 && v <= 64)) bad("the sampled path takes an integer 1..64");
            out[key] = (int64_t) v;
        } else if (key == "presence_penalty" || key == "frequency_penalty") {
            if (!number || v < 0) bad("expected a number >= 0");
            out[key] = v;
        } else if (key == "repetition_penalty") {
            if (!number || v <= 0) bad("expected a number > 0 (1 = off)");
            out[key] = v;
        } else if (key == "penalty_last_n") {
            if (!number || v != std::floor(v) || v < 0) bad("expected a non-negative integer");
            out[key] = (int64_t) v;
        } else if (key == "seed") {
            if (!number || v != std::floor(v) || v <= 0) bad("expected a positive integer");
            out[key] = (int64_t) v;
        } else {
            say("[lamina] " + shown + ": unknown key, ignored");
        }
    }
    return out;
}

json model_card_sampling(const std::string& arch, bool thinking) {
    // the model cards' sets: Qwen3.8-Flash-Next and Qwen3.8-27B; Qwen3.6-35B-A3B's general-task sets
    if (arch == "qwen4exp" || arch == "qwen35")
        return thinking ? json{{"temperature", 1.0}, {"top_p", 0.95}, {"top_k", 20}}
                        : json{{"temperature", 0.7}, {"top_p", 0.8}, {"top_k", 20}, {"presence_penalty", 1.5}};
    if (arch == "qwen35moe")
        return thinking ? json{{"temperature", 1.0}, {"top_p", 0.95}, {"top_k", 20}, {"presence_penalty", 1.5}}
                        : json{{"temperature", 0.7}, {"top_p", 0.8}, {"top_k", 20}, {"presence_penalty", 1.5}};
    return json::object();
}

std::string default_model_name(const json& cfg, const std::string& arch) {
    if (!jstr(cfg, "model_name").empty()) return jstr(cfg, "model_name");
    if (arch == "qwen35moe") return "qwen3.6-35b-a3b";
    if (arch == "qwen35") return "qwen3.8-27b";
    return "qwen3.8-flash-next";
}

double reasoning_repeat_coverage(const std::string& text) {
    // coverage of the recent words by 12-word passages seen at least three times; #1753: a tail of at most 64
    // repeating characters (one long word cycling a pattern) is full coverage
    constexpr size_t kTailChars = 2048, kTailPeriod = 64, kHistoryWords = 30000, kWidth = 12, kWindow = 2000;
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < text.size();) {
        size_t n = 1;
        cps.push_back(unicode_tolower(utf8_at(text, i, &n)));
        i += n;
    }
    if (cps.size() >= kTailChars) {
        const std::vector<uint32_t> tail(cps.end() - (ptrdiff_t) kTailChars, cps.end());
        for (size_t p = 1; p <= kTailPeriod; ++p)
            if (std::equal(tail.begin() + (ptrdiff_t) p, tail.end(), tail.begin())) return 1.0;
    }
    // re.findall(r"\w+|[^\w\s]"): runs of word characters, else one character that is not whitespace
    std::vector<std::string> words;
    std::string cur;
    auto word_char = [](uint32_t c) {
        if (c == '_') return true;
        const unicode_cpt_flags f = unicode_cpt_flags_from_cpt(c);
        return (bool) f.is_letter || (bool) f.is_number;
    };
    auto flush = [&] {
        if (!cur.empty()) words.push_back(std::move(cur));
        cur.clear();
    };
    for (const uint32_t c : cps) {
        if (word_char(c)) {
            utf8_append(cur, c);
        } else {
            flush();
            if (!py_isspace(c)) {
                std::string one;
                utf8_append(one, c);
                words.push_back(std::move(one));
            }
        }
    }
    flush();
    if (words.size() > kHistoryWords) words.erase(words.begin(), words.end() - (ptrdiff_t) kHistoryWords);
    if (words.size() < kWindow) return 0.0;
    std::unordered_map<std::string, int32_t> word_id;
    std::vector<int32_t> ids;
    for (const auto& w : words) ids.push_back(word_id.emplace(w, (int32_t) word_id.size()).first->second);
    const size_t ng = ids.size() - kWidth + 1;
    std::unordered_map<std::string, int> counts;
    std::vector<std::string> grams(ng);
    for (size_t i = 0; i < ng; ++i) {
        grams[i].assign((const char*) &ids[i], kWidth * sizeof(int32_t));
        ++counts[grams[i]];
    }
    const size_t start = ids.size() > kWindow ? ids.size() - kWindow : 0;
    std::vector<uint8_t> covered(ids.size(), 0);
    for (size_t i = start; i < ng; ++i)
        if (counts[grams[i]] >= 3)
            for (size_t k = i; k < i + kWidth; ++k) covered[k] = 1;
    size_t c = 0;
    for (size_t k = 0; k < covered.size(); ++k) c += covered[k];
    return (double) c / (double) (ids.size() - start);
}

std::optional<std::vector<int32_t>> focused_recovery_prompt(const Tokenizer& tok, const std::vector<int32_t>& ids,
                                                            const std::vector<int32_t>& generated) {
    // the prompt with the xhigh effort sentence of the first system message replaced by the low one, then every
    // token generated so far: a splice of ids, so the prompt is never decoded and encoded again
    const std::vector<int32_t> end = tok.encode("<|im_end|>", true);
    if (end.size() != 1) return std::nullopt;
    const auto at = std::find(ids.begin(), ids.end(), end[0]);
    if (at == ids.end()) return std::nullopt;
    const std::vector<int32_t> head(ids.begin(), at);
    if (!starts_with(tok.decode(head), "<|im_start|>system\n")) return std::nullopt;
    const std::vector<int32_t> old = tok.encode(kEffortText.at("xhigh")), neu = tok.encode(kEffortText.at("low"));
    for (size_t i = 0; i + old.size() <= head.size(); ++i)
        if (std::equal(old.begin(), old.end(), head.begin() + (ptrdiff_t) i)) {
            std::vector<int32_t> out(ids.begin(), ids.begin() + (ptrdiff_t) i);
            out.insert(out.end(), neu.begin(), neu.end());
            out.insert(out.end(), ids.begin() + (ptrdiff_t) (i + old.size()), ids.end());
            out.insert(out.end(), generated.begin(), generated.end());
            return out;
        }
    return std::nullopt;
}

std::optional<std::string> held_departure(const Tokenizer& tok, const Held& held, const std::vector<int32_t>& ids,
                                          int64_t reused) {
    // a prompt that leaves what the engine held at least 256 tokens after where it resumed: those were read again
    const size_t plen = held.prompt_len;
    const std::vector<int32_t>& h = held.ids;
    const size_t lim = std::min(h.size(), ids.size());
    size_t m = 0;
    while (m < lim && h[m] == ids[m]) ++m;
    if (m + 1 >= h.size() || (int64_t) m - reused < 256) return std::nullopt;
    const std::string where = m >= plen ? std::to_string(m - plen) + " tokens into its reply of " + std::to_string(h.size() - plen)
                                        : std::to_string(plen - m) + " before its end";
    auto pieces = [&](const std::vector<int32_t>& v, size_t from) {
        std::vector<std::string> out;
        for (size_t k = from; k < std::min(v.size(), from + 8); ++k) out.push_back(tok.decode({v[k]}));
        return py_list_repr(out);
    };
    const std::vector<int32_t> before(h.begin() + (ptrdiff_t) (m > 24 ? m - 24 : 0), h.begin() + (ptrdiff_t) m);
    return "[lamina] prompt cache: this prompt leaves the engine's last request at token " + std::to_string(m) + " (" +
           where + "); it resumed at " + std::to_string(reused) + ", so " + std::to_string((int64_t) m - reused) +
           " tokens were read again. Before: " + py_repr_str(tok.decode(before)) + "; held " + pieces(h, m) +
           "; prompt " + pieces(ids, m);
}

int64_t prompt_tokens_seen(int64_t prompt_tokens, const json& last) {
    const json& read = jget(last, "prompt_read");
    if (read.is_null() || jstr(last, "finish") != "cancel") return prompt_tokens;
    return std::min(prompt_tokens, (is_int(jget(last, "reused")) ? last["reused"].get<int64_t>() : 0) + read.get<int64_t>());
}

json request_timings(int64_t prompt_tokens, int64_t generated, const json& last) {
    if (!jget(last, "prompt_ms").is_number()) return nullptr;
    int64_t cache_n = is_int(jget(last, "reused")) ? last["reused"].get<int64_t>() : 0;
    if (is_int(jget(last, "prompt_read")) && jstr(last, "finish") != "cancel")
        cache_n = std::max(cache_n, std::max<int64_t>(0, prompt_tokens - last["prompt_read"].get<int64_t>()));
    const int64_t prompt_n = std::max<int64_t>(0, prompt_tokens - cache_n);
    const double prompt_ms = last["prompt_ms"].get<double>();
    const double decode_ms = jget(last, "decode_ms").is_number() ? last["decode_ms"].get<double>() : 0.0;
    const int64_t decoded = is_int(jget(last, "generated")) && last["generated"].get<int64_t>() ? last["generated"].get<int64_t>()
                                                                                                 : generated;
    json t = {{"cache_n", cache_n}, {"prompt_n", prompt_n}, {"prompt_ms", py_round(prompt_ms, 1)},
              {"prompt_per_token_ms", prompt_n ? json(py_round(prompt_ms / (double) prompt_n, 3)) : json(nullptr)},
              {"prompt_per_second", prompt_n && prompt_ms > 0 ? json(py_round((double) prompt_n / (prompt_ms / 1000), 1)) : json(nullptr)},
              {"predicted_n", generated}, {"predicted_ms", py_round(decode_ms, 1)},
              {"predicted_per_token_ms", decoded ? json(py_round(decode_ms / (double) decoded, 3)) : json(nullptr)},
              {"predicted_per_second", decoded && decode_ms > 0 ? json(py_round((double) decoded / (decode_ms / 1000), 1)) : json(nullptr)}};
    if (!jget(last, "drafts_offered").is_null()) {
        t["draft_n"] = last["drafts_offered"];
        t["draft_n_accepted"] = jget(last, "drafts_accepted");
    }
    if (truthy(jget(last, "expert_tiers"))) t["expert_tiers"] = last["expert_tiers"];
    return t;
}

json request_stats(const std::vector<json>& segments) {
    // input and cache accounting at the API request's boundary, the native work summed across continuations
    if (segments.empty()) return json::object();
    json result = segments.back();
    for (const char* key : {"reused", "prompt_read", "chunk"}) {
        if (segments.front().contains(key)) result[key] = segments.front()[key];
        else result.erase(key);
    }
    for (const char* key : {"prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted", "hits", "lookups",
                            "offloaded", "ram_blobs", "file_blobs", "file_mb", "experts_streamed"}) {
        bool any = false, is_float = false;
        double sum = 0;
        for (const auto& s : segments) {
            const json& v = jget(s, key);
            if (v.is_null()) continue;
            any = true;
            is_float = is_float || v.is_number_float();
            sum += v.is_number() ? v.get<double>() : 0.0;
        }
        if (any) result[key] = is_float ? json(sum) : json((int64_t) sum);
    }
    std::vector<json> tiers;
    for (const auto& s : segments)
        if (truthy(jget(s, "expert_tiers"))) tiers.push_back(s["expert_tiers"]);
    if (tiers.size() > 1) {
        json t = json::object();
        for (auto it = tiers.back().begin(); it != tiers.back().end(); ++it) {
            if (it.key() == "second_empty_slots" || it.key() == "deterministic_adapt") {
                t[it.key()] = it.value();
            } else {
                int64_t sum = 0;
                for (const auto& x : tiers) sum += is_int(jget(x, it.key())) ? x[it.key()].get<int64_t>() : 0;
                t[it.key()] = sum;
            }
        }
        result["expert_tiers"] = t;
    }
    return result;
}

// ---------------------------------------------------------------------------------------------- Service
thread_local Service::Embeddings Service::embeddings;
thread_local json* Service::trace = nullptr;

Service::Service(std::shared_ptr<Engine> engine_, const Tokenizer& tok_, std::shared_ptr<ChatTemplate> tpl_,
                 std::string model_, std::shared_ptr<Vision> vision_, json sampling_defaults_, bool fit_max_tokens_)
    : fit_max_tokens(fit_max_tokens_), sampling_defaults(std::move(sampling_defaults_)), engine(std::move(engine_)),
      tok(tok_), tpl(std::move(tpl_)), model(std::move(model_)), vision(std::move(vision_)) {
    literals = literal_tags(tok.control_tokens());
    for (const char* s : {"<|im_end|>", "<|endoftext|>"})
        for (const int32_t t : tok.encode(s, true)) stop_ids.insert(t);
    prompts = std::make_unique<PromptEncoder>(tok);
    started_at = now_s();
    instance_id = random_hex(32);
    totals = {{"since", started_at}, {"requests", 0}, {"prompt_tokens", 0}, {"reused", 0}, {"output_tokens", 0},
              {"prompt_ms", 0.0}, {"decode_ms", 0.0}, {"drafts_offered", 0}, {"drafts_accepted", 0}};
    host_names = {"localhost", "127.0.0.1", "::1"};
}

Service::~Service() {
    if (telemetry != nullptr) telemetry->close();   // its extra() reads this Service
    stopping_ = true;
    if (idle_thread_.joinable()) idle_thread_.join();
}

bool Service::loaded() { return engine->alive(); }

std::optional<int64_t> Service::admission_limit() {
    if (!max_queue) return std::nullopt;
    return std::max(1, engine->batch()) + *max_queue;
}

std::optional<bool> Service::reserve_inference() {
    std::lock_guard<std::mutex> lk(status_mu);
    if (draining) return std::nullopt;
    const auto limit = admission_limit();
    if (limit && admission_reserved >= *limit) {
        ++admission_rejected;
        return false;
    }
    ++admission_reserved;
    return true;
}

void Service::release_inference() {
    std::lock_guard<std::mutex> lk(status_mu);
    --admission_reserved;
    admission_changed.notify_all();
}

int64_t Service::begin_drain() {
    // new inference is refused from here on; -> the calls still to drain
    std::lock_guard<std::mutex> lk(status_mu);
    draining = true;
    admission_changed.notify_all();
    return admission_reserved;
}

int64_t Service::wait_for_drain(double timeout_s) {
    std::unique_lock<std::mutex> lk(status_mu);
    admission_changed.wait_for(lk, std::chrono::duration<double>(std::max(0.0, timeout_s)), [&] { return admission_reserved == 0; });
    return admission_reserved;
}

int64_t Service::cancel_owned_requests() {
    // only the requests this server admitted, never other GPU work
    std::vector<Cancel*> active;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        active.assign(active_cancels.begin(), active_cancels.end());
    }
    for (Cancel* c : active) c->set();
    return (int64_t) active.size();
}

void Service::set_aliases(const json& value) {
    // #297: the config's aliases, a list of names (or one comma-separated string), like llama-server's --alias
    std::vector<std::string> items;
    if (value.is_string()) {
        items = split(value.get<std::string>(), ",");
    } else if (value.is_array() && std::all_of(value.begin(), value.end(), [](const json& x) { return x.is_string(); })) {
        for (const auto& x : value) items.push_back(x.get<std::string>());
    } else if (!value.is_null()) {
        throw std::invalid_argument("aliases=" + py_repr_value(value) + ": expected a list of model names");
    }
    aliases.clear();
    for (const auto& raw : items) {
        const std::string x(strip(raw));
        if (!x.empty() && x != model && std::find(aliases.begin(), aliases.end(), x) == aliases.end()) aliases.push_back(x);
    }
}

void Service::note_client_disconnect(Cancel& cancel) {
    // cancel once; count only work that was still queued or active when the client left
    std::lock_guard<std::mutex> lk(status_mu);
    if (!cancel.is_set()) {
        const bool active = terminal_requests.count(cancel.request_id) > 0;
        cancel.set();
        if (active) ++client_disconnect_cancels;
    }
}

std::vector<std::string> Service::model_names() const {
    std::vector<std::string> out{model};
    out.insert(out.end(), aliases.begin(), aliases.end());
    return out;
}

std::string Service::model_for(const json& req) const {
    // the request's own name when it is an alias (#297), else the model's; other names are still served
    const std::string asked = jstr(req, "model");
    return std::find(aliases.begin(), aliases.end(), asked) != aliases.end() ? asked : model;
}

int64_t Service::reported_ctx() {
    // the engine's context, or the last one it reported while it restarts (#351)
    const int64_t c = engine->max_context();
    return c > 0 ? c : engine->known_ctx();
}

std::optional<int64_t> Service::reasoning_budget(const json& req) const {
    json value = req.is_object() ? jget(req, "reasoning_budget_tokens") : json(nullptr);
    if (value.is_null()) value = reasoning_budget_tokens;
    if (value.is_number_float() && value.get<double>() == std::floor(value.get<double>())) value = (int64_t) value.get<double>();
    if (!is_int(value))
        throw RequestError("reasoning_budget_tokens=" + py_repr_value(value) + ": expected a whole number of tokens (0: no budget)");
    const int64_t v = value.get<int64_t>();
    return v > 0 ? std::optional<int64_t>(v) : std::nullopt;
}

std::optional<int64_t> Service::free_vram_mib() {
    const int64_t f = Telemetry::free_vram_mib(gpu_index);
    return f < 0 ? std::nullopt : std::optional<int64_t>(f);
}

std::string Service::say_died(const std::exception& e) {
    const std::string note = engine->death_note();
    const std::string line = "[lamina] " + std::string(e.what()) + ". " + note + " The next request starts the engine again.";
    say(line);
    return line;
}

void Service::ensure_loaded() {
    // start the engine if it is not running (unloaded, or it died), after the before_load hook and the free-VRAM check;
    // the caller holds the turn
    const bool vision_down = vision != nullptr && !vision->alive();
    if (loaded() && !vision_down) return;
    std::string hook_failed;
    if (truthy(before_load)) {
        std::string cmd;
        if (before_load.is_string()) {
            cmd = before_load.get<std::string>();
        } else {
            for (const auto& x : before_load) cmd += (cmd.empty() ? "" : " ") + (x.is_string() ? x.get<std::string>() : py_dumps(x));
        }
        say("[lamina] before loading: " + cmd);
        std::string why;
        const int64_t rc = run_command(before_load, before_load_timeout_s, why);
        if (why.empty() && rc != 0) why = "it exited with code " + std::to_string(rc);
        if (!why.empty()) {
            // a failing hook may mean the GPU was not given back; the free VRAM decides (min_free_vram_mib below),
            // "strict" always refuses, "ignore" always loads
            if (before_load_strict && !before_load_ignore_failure) {
                say("[lamina] the before_load command failed (" + why + "); the model stays unloaded and this request "
                    "is answered 503 - try again (before_load_strict)");
                throw GpuBusy("the before_load command failed (" + why + "): the GPU was not given back, so the model "
                              "was not loaded; try again in a moment", 15);
            }
            hook_failed = why;
            if (before_load_ignore_failure || !min_free_vram_mib) {
                say("[lamina] the before_load command failed (" + why + "); loading anyway" +
                    (before_load_ignore_failure ? "" : " (set min_free_vram_mib to have the free VRAM checked, or "
                                                       "before_load_strict to refuse)"));
            } else {
                say("[lamina] the before_load command failed (" + why + "); checking the free VRAM");
            }
        }
    }
    if (min_free_vram_mib && !(!hook_failed.empty() && before_load_ignore_failure)) {
        auto free = free_vram_mib();
        const double deadline = now_s() + vram_free_wait_s;   // memory another process gave back can take a moment
        while (free && *free < min_free_vram_mib && now_s() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            free = free_vram_mib();
        }
        if (free && *free < min_free_vram_mib) {
            if (!hook_failed.empty() && !before_load_ignore_failure)
                say("[lamina] the before_load command failed (" + hook_failed + ") and only " + std::to_string(*free) +
                    " MiB of VRAM is free (the model needs " + std::to_string(min_free_vram_mib) + "): not loading; 503");
            throw GpuBusy("the GPU is in use by another program: " + std::to_string(*free) + " MiB of VRAM free, the model "
                          "needs " + std::to_string(min_free_vram_mib) + " (min_free_vram_mib) - it stays unloaded until "
                          "that is free" + (hook_failed.empty() ? "" : "; the before_load command failed (" + hook_failed + ")"), 15);
        }
        if (!hook_failed.empty() && !before_load_ignore_failure)
            say("[lamina] the before_load command failed (" + hook_failed + "), but " + std::to_string(free.value_or(0)) +
                " MiB of VRAM is free: loading");
    }
    bool vision_started = false;
    if (vision != nullptr && !vision->alive()) {
        say("[lamina] starting the vision encoder ...");
        try {
            vision->restart();
        } catch (const std::exception& e) {
            throw EngineDied(std::string("the vision encoder failed to start: ") + e.what());
        }
        vision_started = true;
    }
    if (loaded()) return;
    if (engine->unloaded()) {
        say("[lamina] loading the model again (it was unloaded) ...");
    } else {
        say("[lamina] the engine had stopped; starting it again (a minute or two) ...");
    }
    try {
        engine->load();
    } catch (...) {
        if (vision_started) vision->unload();
        throw;
    }
    say("[lamina] the engine is running again");
}

void Service::load() {
    // POST /load and every generation request: start the engine now if it is unloaded (GpuBusy)
    {
        std::lock_guard<std::mutex> lk(status_mu);
        last_request_at = now_s();   // the idle thread must not unload between this and the request's own start
    }
    if (loaded() && !(vision != nullptr && !vision->alive())) return;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        status["queued"] = status["queued"].get<int64_t>() + 1;
    }
    bool counted = true;
    try {
        std::lock_guard<std::timed_mutex> turn(fifo_);
        {
            std::lock_guard<std::mutex> lk(status_mu);
            status["queued"] = status["queued"].get<int64_t>() - 1;
        }
        counted = false;
        ensure_loaded();
    } catch (...) {
        if (counted) {
            std::lock_guard<std::mutex> lk(status_mu);
            status["queued"] = status["queued"].get<int64_t>() - 1;
        }
        throw;
    }
}

std::string Service::unload(std::optional<double> idle_for) {
    // stop the engine between requests: "unloaded", "not loaded" or "busy"
    std::unique_lock<std::timed_mutex> turn(fifo_, std::try_to_lock);
    if (!turn.owns_lock()) return "busy";
    if (!engine->alive()) return "not loaded";
    {
        std::lock_guard<std::mutex> lk(status_mu);
        if (truthy(jget(status, "busy")) || truthy(jget(status, "queued"))) return "busy";
        if (idle_for && now_s() - last_request_at.value_or(started_at) < *idle_for) return "busy";
    }
    engine->unload();
    if (vision != nullptr) vision->unload();
    say(std::string("[lamina] model unloaded") +
        (idle_for ? " after " + std::to_string((int) *idle_for) + " s idle" : "") + "; the next request loads it again");
    return "unloaded";
}

std::string Service::unload_waiting(double wait_s) {
    // #1997: a running or queued request is waited for (up to wait_s) instead of answering "busy"
    const double deadline = now_s() + std::max(0.0, wait_s);
    while (true) {
        const std::string r = unload();
        if (r != "busy" || now_s() >= deadline) return r;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
}

void Service::start_idle_unload() {
    if (idle_unload_s <= 0) return;
    say("[lamina] the model unloads after " + std::to_string((int) idle_unload_s) + " s without requests");
    idle_thread_ = std::thread([this] {
        const double every = std::max(1.0, std::min(30.0, idle_unload_s / 4));
        while (!stopping_.load()) {
            for (double t = 0; t < every && !stopping_.load(); t += 0.25)
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (stopping_.load()) break;
            try {
                unload(idle_unload_s);
            } catch (const std::exception& e) {
                say(std::string("[lamina] idle unload: ") + e.what());
            }
        }
    });
}

json Service::shared() const {
    std::lock_guard<std::mutex> lk(shared_mu_);
    return shared_;
}

json Service::set_shared(const json& defaults, bool save) {
    // the Chat settings every client gets for what it leaves out; {} / null: clients use their own again
    const json shared = clean_shared_defaults(defaults);
    {
        std::lock_guard<std::mutex> lk(shared_mu_);
        shared_ = shared;
    }
    if (save && !shared_path.empty()) {
        try {
            if (!shared.empty()) {
                std::ofstream f(shared_path, std::ios::binary);
                f << py_dumps(shared, true, false, 1);
            } else {
                std::remove(shared_path.c_str());
            }
        } catch (const std::exception& e) {
            say(std::string("[lamina] could not save the shared settings: ") + e.what());
        }
    }
    return shared;
}

std::string Service::effort_default() const {
    // the thinking level of a request that names none: the shared Chat settings, then the config's (#1641)
    const std::string s = jstr(shared(), "reasoning_effort");
    return !s.empty() ? s : config_effort;
}

json Service::with_shared(const json& req0, const std::string& api) const {
    const std::string effort = effort_default();
    const json shared = this->shared();
    if (shared.empty() && effort.empty()) return req0;
    json req = req0;
    if (shared.contains("max_tokens") && !truthy(jget(req, "max_tokens")) && !truthy(jget(req, "max_completion_tokens")))
        req["max_tokens"] = shared["max_tokens"];
    if (!effort.empty()) {
        if (api == "openai") {
            const json& ctk = jget(req, "chat_template_kwargs");
            const json& re = jget(req, "reasoning_effort");
            if ((re.is_null() || (re.is_string() && re.get_ref<const std::string&>().empty())) &&
                !truthy(jget(req, "reasoning")) && !ctk.contains("enable_thinking") && !ctk.contains("reasoning_effort"))
                req["reasoning_effort"] = effort;
        } else if (!truthy(jget(req, "thinking")) && !truthy(jget(req, "output_config"))) {
            if (effort == "none") req["thinking"] = {{"type", "disabled"}};
            else req["output_config"] = {{"effort", effort}};
        }
    }
    return req;
}

double Service::rate_of(const json& s, const std::deque<std::pair<double, int64_t>>& rt) {
    if (!truthy(jget(s, "busy")) || !jget(s, "first_token").is_number()) return 0.0;
    const double now = now_s();
    if (!rt.empty()) {
        const auto& newest = rt.back();
        for (const auto& r : rt)
            if (now - r.first <= 2.0) {   // RATE_WINDOW_S
                if (newest.first - r.first >= 0.25) return std::max(0.0, (double) (newest.second - r.second) / (newest.first - r.first));
                break;
            }
    }
    return (double) s["generated"].get<int64_t>() / std::max(0.25, now - s["first_token"].get<double>());
}

double Service::tok_s() {
    std::lock_guard<std::mutex> lk(status_mu);
    if (!live_reqs_.empty()) {
        double sum = 0;
        for (const auto& [k, v] : live_reqs_) sum += rate_of(*v.first, *v.second);
        return sum;
    }
    return rate_of(status, rate);
}

double Service::tok_s_mean() {
    std::lock_guard<std::mutex> lk(status_mu);
    double sum = 0;
    auto one = [&](const json& s) {
        if (truthy(jget(s, "busy")) && jget(s, "first_token").is_number())
            sum += (double) s["generated"].get<int64_t>() / std::max(1e-6, now_s() - s["first_token"].get<double>());
    };
    if (!live_reqs_.empty())
        for (const auto& [k, v] : live_reqs_) one(*v.first);
    else
        one(status);
    return sum;
}

json Service::prefill_tok_s_mean() {
    bool reading;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        reading = truthy(jget(status, "busy")) && jget(status, "first_token").is_null();
    }
    if (!reading) return 0.0;
    const double mean = engine->progress().tok_s_mean;
    return mean >= 0 ? json(mean) : json(nullptr);
}

std::shared_ptr<json> Service::begin_request(const std::string& path, const json& req) {
    // #332: a monitor record for this request, or null when the monitor is off (nothing is kept then)
    if (!api_monitor) return nullptr;
    const std::string raw = py_dumps(req, false, false, 2);
    const json& rf = jget(req, "response_format");
    json record = {{"id", random_hex(12)}, {"path", path}, {"model", truthy(jget(req, "model")) ? req["model"] : json(model)},
                   {"started_at", now_s()}, {"state", "queued"}, {"stream", truthy(jget(req, "stream"))},
                   {"response_format", rf.is_object() ? jget(rf, "type") : json(nullptr)},
                   {"input", raw.substr(0, std::min<size_t>(raw.size(), 262144))}, {"input_truncated", raw.size() > 262144},
                   {"output", ""}, {"reasoning", ""}, {"output_truncated", false}, {"reasoning_truncated", false},
                   {"queue_s", 0.0}, {"load_s", 0.0}, {"first_token_s", nullptr}, {"_clock", mono_s()}};
    auto kept = std::make_shared<json>(std::move(record));
    std::lock_guard<std::mutex> lk(status_mu);
    api_requests.push_back(kept);
    while (api_requests.size() > 100) api_requests.pop_front();
    trace = kept.get();
    return kept;
}

void Service::end_request() { trace = nullptr; }

json Service::request_records(const std::string& request_id) {
    std::lock_guard<std::mutex> lk(status_mu);
    auto visible = [](const json& r, bool all) {
        json out = json::object();
        for (auto it = r.begin(); it != r.end(); ++it) {
            if (starts_with(it.key(), "_")) continue;
            if (!all && (it.key() == "input" || it.key() == "output" || it.key() == "reasoning" || it.key() == "response")) continue;
            out[it.key()] = it.value();
        }
        return out;
    };
    if (!request_id.empty()) {
        for (const auto& r : api_requests)
            if (jstr(*r, "id") == request_id) return visible(*r, true);
        return nullptr;
    }
    json out = json::array();
    for (auto it = api_requests.rbegin(); it != api_requests.rend(); ++it) {
        const json& r = **it;
        json v = visible(r, false);
        v["wallclock_s"] = r.contains("wallclock_s") ? r["wallclock_s"] : json(py_round(mono_s() - r["_clock"].get<double>(), 3));
        out.push_back(v);
    }
    return out;
}

json Service::metrics(bool all_requests) {
    // GET /metrics: what the Monitor shows - the engine's facts, what it is doing, the last requests, the hardware
    json s, hist = json::array(), tot;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        s = status;
        if (!live_reqs_.empty()) {
            const json q = jget(s, "queued");
            for (auto it = live_reqs_.rbegin()->second.first->begin(); it != live_reqs_.rbegin()->second.first->end(); ++it)
                s[it.key()] = it.value();
            s["queued"] = q;
        }
        for (auto it = history.rbegin(); it != history.rend(); ++it) {
            if (!all_requests && hist.size() >= 12) break;
            hist.push_back(*it);
        }
        tot = totals;
    }
    const double now = now_s();
    const bool busy = truthy(jget(s, "busy"));
    const std::string state = busy && jget(s, "first_token").is_null() ? "reading" : busy ? "generating"
                              : !loaded() ? "unloaded" : "idle";
    const Engine::Progress pr = engine->progress();
    json live = {{"state", state}, {"queued", jget(s, "queued")}, {"phase", busy ? jget(s, "phase") : json(nullptr)},
                 {"prompt_tokens", busy ? jget(s, "prompt_tokens") : json(nullptr)}, {"prompt_read", nullptr},
                 {"prompt_total", nullptr}, {"generated", busy ? jget(s, "generated") : json(nullptr)},
                 {"max_tokens", busy ? jget(s, "max_tokens") : json(nullptr)},
                 {"elapsed_s", busy && jget(s, "started").is_number() ? json(py_round(now - s["started"].get<double>(), 1)) : json(nullptr)},
                 {"tok_s", state == "generating" ? json(py_round(tok_s(), 1)) : json(nullptr)},
                 {"tok_s_mean", state == "generating" ? json(py_round(tok_s_mean(), 1)) : json(nullptr)},
                 {"prefill_tok_s_mean", busy && pr.tok_s_mean >= 0 ? json(pr.tok_s_mean) : json(nullptr)},
                 {"tok_s_window_s", state == "generating" ? json(2.0) : json(nullptr)}};
    if (state == "reading" && pr.read) {
        live["prompt_read"] = pr.read->first;
        live["prompt_total"] = pr.read->second;
    }
    if (const int par = engine->batch()) {
        size_t running;
        {
            std::lock_guard<std::mutex> lk(status_mu);
            running = live_reqs_.size();
        }
        // #465: the requests running together, slot by slot; the ones not in a slot run alone on the solo path or
        // wait their turn
        json slots = engine->slots_view();
        if (!slots.is_array()) slots = json::array();
        const size_t in_slots = (size_t) std::count_if(slots.begin(), slots.end(), [](const json& x) { return jstr(x, "state") != "idle"; });
        const auto [waiting, admitting] = engine->admission();
        live["parallel"] = par;
        live["running"] = running;
        live["slots"] = slots;
        live["outside_slots"] = running > in_slots ? running - in_slots : 0;
        live["waiting"] = waiting;
        live["admitting"] = admitting;
        if (running && state == "idle") live["state"] = "generating";
    }
    if (const json rv = engine->replica_view(); !rv.is_null()) live["replicas"] = rv;   // each replica's state
    if (const json cv = engine->cache_view(); !cv.is_null()) live["cache"] = cv;   // the Conversation cache card
    json eng = {{"model", model}, {"max_context", reported_ctx()}, {"images", vision != nullptr}};
    const json facts = engine->info();
    for (auto it = facts.begin(); it != facts.end(); ++it) eng[it.key()] = it.value();
    const json tel = telemetry != nullptr ? telemetry->snapshot()
                                          : json{{"now", json::object()}, {"history", json::object()}, {"static", json::object()}};
    size_t kept;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        kept = history.size();
    }
    return {{"engine", eng}, {"live", live}, {"requests", hist}, {"requests_kept", kept}, {"totals", tot},
            {"hardware", tel["now"]}, {"hardware_static", tel["static"]}, {"history", tel["history"]}, {"time", now}};
}

std::string Service::prometheus() {
    // GET /metrics/prometheus: the totals and what runs under llama-server's names (llamacpp:*), the drafts and
    // requests as lamina:*; the two throughput gauges over the window between two scrapes, as llama.cpp keeps them
    json s, t;
    size_t running;
    double d_prompt_n, d_prompt_s, d_pred_n, d_decode_s;
    int64_t prompt_n;
    double prompt_s, decode_s;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        s = status;
        t = totals;
        running = live_reqs_.size() ? live_reqs_.size() : (size_t) truthy(jget(s, "busy"));
        prompt_n = t["prompt_tokens"].get<int64_t>() - t["reused"].get<int64_t>();
        prompt_s = t["prompt_ms"].get<double>() / 1000;
        decode_s = t["decode_ms"].get<double>() / 1000;
        const auto [wpn, wps, wpred, wds] = scrape_bucket;
        scrape_bucket = {prompt_n, prompt_s, t["output_tokens"].get<int64_t>(), decode_s};
        d_prompt_n = (double) (prompt_n - wpn);
        d_prompt_s = prompt_s - wps;
        d_pred_n = (double) (t["output_tokens"].get<int64_t>() - wpred);
        d_decode_s = decode_s - wds;
    }
    auto num = [](double v) { return py_repr(v); };
    const std::vector<std::tuple<std::string, std::string, std::string, std::string>> rows = {
        {"llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", std::to_string(prompt_n)},
        {"llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the conversation cache already held",
         std::to_string(t["reused"].get<int64_t>())},
        {"llamacpp:prompt_seconds_total", "counter", "Prompt read time", num(prompt_s)},
        {"llamacpp:tokens_predicted_total", "counter", "Tokens generated", std::to_string(t["output_tokens"].get<int64_t>())},
        {"llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", num(decode_s)},
        {"llamacpp:prompt_tokens_seconds", "gauge", "Average prompt read throughput in tokens/s, since the last scrape",
         d_prompt_s > 0 ? num(d_prompt_n / d_prompt_s) : "0"},
        {"llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s, since the last scrape",
         d_decode_s > 0 ? num(d_pred_n / d_decode_s) : "0"},
        {"llamacpp:requests_processing", "gauge", "Requests running", std::to_string(running)},
        {"llamacpp:requests_deferred", "gauge", "Requests waiting their turn", std::to_string(jget(s, "queued").get<int64_t>())},
        {"lamina:requests_total", "counter", "Requests finished", std::to_string(t["requests"].get<int64_t>())},
        {"lamina:drafts_offered_total", "counter", "Speculative draft tokens offered", std::to_string(t["drafts_offered"].get<int64_t>())},
        {"lamina:drafts_accepted_total", "counter", "Speculative draft tokens accepted", std::to_string(t["drafts_accepted"].get<int64_t>())},
        {"lamina:model_loaded", "gauge", "1 while the model is loaded", loaded() ? "1" : "0"},
    };
    std::string out;
    for (const auto& [name, kind, help, value] : rows)
        out += "# HELP " + name + " " + help + "\n# TYPE " + name + " " + kind + "\n" + name + " " + value + "\n";
    return out;
}

json Service::v1_status() {
    // GET /v1/status: what this server is and does, for a client that would rather ask than guess
    json s, tot, last_t;
    std::optional<double> last_at;
    size_t running;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        s = status;
        tot = totals;
        last_t = last_timings;
        last_at = last_request_at;
        running = live_reqs_.size() ? live_reqs_.size() : (size_t) truthy(jget(s, "busy"));
    }
    const json tel = telemetry != nullptr ? telemetry->snapshot() : json{{"now", json::object()}, {"static", json::object()}};
    const json& hw = tel["now"];
    const json& st = tel["static"];
    auto scaled = [](const json& v, double unit, int digits) -> json {
        return v.is_number() ? json(py_round(v.get<double>() / unit, digits)) : json(nullptr);
    };
    const int64_t ctx = reported_ctx();
    const bool img = vision != nullptr;
    json machine = {{"at", (int64_t) now_s()}};
    machine["gpu"] = truthy(jget(st, "gpu_name"))
                         ? json{{"name", st["gpu_name"]}, {"used_mib", scaled(jget(hw, "gpu_mem_used"), 1048576.0, 0)},
                                {"total_mib", scaled(jget(hw, "gpu_mem_total"), 1048576.0, 0)}, {"util_pct", jget(hw, "gpu_util")},
                                {"temp_c", jget(hw, "gpu_temp")}, {"power_w", jget(hw, "gpu_power")}}
                         : json(nullptr);
    machine["ram"] = truthy(jget(hw, "ram_total")) ? json{{"used_gib", scaled(jget(hw, "ram_used"), 1073741824.0, 1)},
                                                          {"total_gib", scaled(jget(hw, "ram_total"), 1073741824.0, 1)}}
                                                   : json(nullptr);
    const int serving = std::max(1, engine->batch());
    return {{"service", "lamina"}, {"model", model}, {"loaded", loaded()}, {"auto_load", engine->auto_load()},
            {"structured_output", {{"formats", {"json_object", "json_schema"}}, {"method", "prompt_and_validate"},
                                   {"constrained_decoding", false}, {"stream_buffered", true}}},
            {"engine", jget(engine->info(), "version")}, {"started", (int64_t) started_at},
            {"uptime_s", (int64_t) (now_s() - started_at)}, {"cache_max_tokens", ctx},
            {"context", {{"native", ctx}, {"max_positions", ctx}}},
            {"concurrency", {{"serving", serving}, {"requested", requested_slots > 0 ? requested_slots : (int64_t) serving}}},
            {"dialects", {"/v1/chat/completions", "/v1/messages", "/v1/responses"}},
            {"vision", {{"enabled", img}, {"available", img}, {"error", nullptr}}},
            {"image_generation", images != nullptr ? images->status() : json{{"enabled", false}}},
            {"activity", {{"requests", tot["requests"].get<int64_t>() + (int64_t) running},
                          {"in_flight", (int64_t) running + jget(s, "queued").get<int64_t>()},
                          {"last_request_at", last_at ? json((int64_t) *last_at) : json(nullptr)}}},
            {"last_timings", last_t}, {"machine", machine}};
}

bool Service::short_opening(const json& messages) const {
    // the only user message so far, of text alone, too short to hold a task
    const json* user = nullptr;
    for (const auto& m : messages) {
        const std::string role = jstr(m, "role");
        if (role == "system") continue;
        if (user != nullptr || role != "user") return false;
        user = &m;
    }
    if (user == nullptr) return false;
    const json& content = jget(*user, "content");
    std::string text;
    if (content.is_string()) text = content.get<std::string>();
    else if (content.is_array())
        for (const auto& part : content) {
            if (jstr(part, "type") != "text") return false;
            text += jstr(part, "text");
        }
    return (int64_t) tok.encode(text, false).size() <= short_opening_tokens;
}

std::string Service::render_prompt(const json& messages, const json& tools, const json& kwargs) {
    // #458: with effort_end, a request with a non-default effort is rendered as a default one up to the answer (the
    // same prompt start, so the conversation cache keeps it) and its effort follows in a short system turn
    const json& effort = jget(kwargs, "reasoning_effort");
    const bool off = jget(kwargs, "enable_thinking").is_boolean() && !kwargs["enable_thinking"].get<bool>();
    const std::string e = effort.is_string() ? effort.get<std::string>() : "";
    const bool top = !off && (effort.is_null() || e.empty() || e == "xhigh" || e == "high");
    if (top && short_opening_tokens > 0 && !effort_end && short_opening(messages)) {
        // the top effort's sentence on a message with no task: the model goes looking for one and invents instructions
        json medium = kwargs;
        medium["reasoning_effort"] = "medium";
        return tpl->render(messages, tools, true, medium);
    }
    if (!effort_end || top)
        return tpl->render(messages, tools, true, kwargs);
    json base = json::object();
    for (auto it = kwargs.begin(); it != kwargs.end(); ++it)
        if (it.key() != "reasoning_effort" && it.key() != "enable_thinking") base[it.key()] = it.value();
    const std::string history_text = tpl->render(messages, tools, false, base);
    const std::string full = tpl->render(messages, tools, true, kwargs);
    const std::string mine = tpl->render(messages, tools, false, kwargs);
    if (!starts_with(full, mine)) return full;   // a template this cannot take apart: rendered as asked
    const auto text = kEffortText.find(e);
    const std::string tail = off ? "" : "<|im_start|>system\n" + (text != kEffortText.end() ? text->second : kEffortText.at("medium")) +
                                            "<|im_end|>\n";
    return history_text + tail + full.substr(mine.size());
}

std::vector<int32_t> Service::encode_prompt(const json& messages, const json& tools, const json& kwargs, bool store) {
    // #537: a literal tag (and a control token's text) inside a message is encoded as the text it is
    const Marked m = mark_think_literals(messages, tools, literals);
    std::string prompt = render_prompt(m.messages, m.tools, kwargs);
    std::vector<Span> plain;
    if (m.changed) {
        auto [p, spans] = unmark_think_literals(prompt, literals);
        prompt = std::move(p);
        for (const auto& [a, b] : spans) plain.push_back({a, b});
    }
    return prompts->encode(prompt, plain, store);   // #567: only what follows the last literal shared with a recent prompt
}

void Service::forget_prompt(const std::vector<int32_t>& ids) { prompts->forget(ids); }

void Service::drop_embeddings() {
    // the combined image file prepare() wrote when no run() took it over
    if (!embeddings.path.empty()) std::remove(embeddings.path.c_str());
    embeddings.path.clear();
    embeddings.keys.clear();
}

void Service::note(int64_t n, const std::vector<Event>& evs, json& st, std::deque<std::pair<double, int64_t>>& rt) {
    std::lock_guard<std::mutex> lk(status_mu);
    st["generated"] = n;
    if (jget(st, "first_token").is_null()) st["first_token"] = now_s();
    rt.emplace_back(now_s(), n);
    while (rt.size() > 32) rt.pop_front();
    for (const Event& ev : evs) {
        if (ev.kind == "reasoning") st["phase"] = "thinking";
        else if (ev.kind == "content") st["phase"] = "answering";
        else if (ev.kind == "tool_start") {
            st["phase"] = "writing a tool call: " + ev.call->name;
            st["tool"] = ev.call->name;
        } else if (ev.kind == "tool_call") {
            st["phase"] = "tool call complete";
        }
        std::string tail = jstr(st, "tail") + ev.text;
        if (tail.size() > 600) tail = tail.substr(tail.size() - 600);
        // a character cut at the start is dropped (the Python tail counted characters)
        size_t i = 0;
        while (i < tail.size() && ((unsigned char) tail[i] & 0xC0) == 0x80) ++i;
        st["tail"] = tail.substr(i);
    }
}

double Service::progress_line(double last_print, json& st, double every) {
    // a progress line in the server window every `every` seconds while a request runs
    const double now = now_s();
    if (now - last_print < every) return last_print;
    json s;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        s = st;
    }
    const double el = now - (jget(s, "started").is_number() ? s["started"].get<double>() : now);
    char buf[256];
    if (jget(s, "first_token").is_null()) {
        auto with_commas = [](int64_t v) {
            std::string d = std::to_string(v), out;
            for (size_t i = 0; i < d.size(); ++i) {
                if (i && (d.size() - i) % 3 == 0) out += ',';
                out += d[i];
            }
            return out;
        };
        const auto pr = engine->progress().read;
        const std::string done = pr && pr->second ? with_commas(pr->first) + " of " + with_commas(pr->second)
                                                  : with_commas(jget(s, "prompt_tokens").is_number() ? s["prompt_tokens"].get<int64_t>() : 0);
        std::snprintf(buf, sizeof buf, "[lamina] reading the prompt: %s tokens, %.0f s so far", done.c_str(), el);
    } else {
        const int64_t gen = s["generated"].get<int64_t>();
        // the span floored as the live rate floors it: at the first token the rate is not a rate yet
        const double r = (double) gen / std::max(0.25, now - s["first_token"].get<double>());
        const std::string limit = jstr(s, "phase") == "thinking" && truthy(jget(s, "thinking_budget"))
                                      ? "budget " + py_repr_value(s["thinking_budget"]) + " (max " + py_repr_value(jget(s, "max_tokens")) + ")"
                                      : "max " + py_repr_value(jget(s, "max_tokens"));
        std::snprintf(buf, sizeof buf, "[lamina] %s: %lld of %s tokens, %.1f tok/s, %.0f s", jstr(s, "phase").c_str(),
                      (long long) gen, limit.c_str(), r, el);
    }
    say(buf);
    return now;
}

}  // namespace lamina
