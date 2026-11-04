// src/server/service_run.cpp - Service::prepare, choose and run: a request's prompt, and its generation as events.
#include "log.hpp"
#include "vision.hpp"
#include "keepawake.hpp"
#include "service.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace lamina {
namespace {

constexpr double kQueueBeatS = 10.0;     // a request waiting for the single turn sends a keep-alive this often
constexpr double kKeepaliveS = 10.0;     // a held-back tool argument leaves the stream quiet: ping at least this often
constexpr int64_t kAnswerRoomMinCap = 1024, kAnswerReserveMin = 512;
constexpr int64_t kLoopCheckEvery = 512;
constexpr double kLoopCoverage = 0.25;
const char* kReasoningClose = "\n</think>\n\n";
const char* kReasoningWrapUp = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n";
const char* kPrefixEnd = "\x01lamina-prefix-end\x01";   // where a message's shared part ends in the rendered prompt
constexpr int kMarkerNames = 64;         // the stop markers a reply may name in its text before one ends it

// Whether a stop token drawn now names the marker in the reply's text rather than ending the turn: the vocabulary has
// the marker as that one token, so a model quoting it draws the id that ends a turn. `text`: the reply so far. A turn
// does not end inside a code span or fence, after an opening quote, in a chat turn the text opened, in a call's open
// parameter, nor, while thinking, anywhere but after a sentence or a line.
bool names_stop_marker(std::string_view text, const std::string& state) {
    constexpr size_t none = std::string_view::npos;
    const auto opened = [&](std::string_view open, std::string_view close) {
        const size_t i = text.rfind(open), j = text.rfind(close);
        return i != none && (j == none || j < i);
    };
    if (state == "call" || state == "rcall") return opened("<parameter=", "</parameter>");
    if (state != "reasoning" && state != "content") return false;
    const bool thinking = state == "reasoning";
    if (const size_t e = text.rfind(kThinkEnd); !thinking && e != none) text = text.substr(e + kThinkEnd.size());
    bool fence = false;
    size_t line = 0;
    for (size_t nl; (nl = text.find('\n', line)) != none; line = nl + 1) {
        const size_t c = text.find_first_not_of(" \t", line);
        if (c < nl && text.compare(c, 3, "```") == 0) fence = !fence;
    }
    const std::string_view last = text.substr(line);
    const size_t first = last.find_first_not_of(" \t");
    if (first != none && last.compare(first, 3, "```") == 0) return false;   // the line of a fence: a real end can follow
    if (opened("<|im_start|>", "<|im_end|>") || (fence && (thinking || !last.empty()))) return true;
    if (std::count(last.begin(), last.end(), '`') % 2 == 1) return true;
    if (const std::string_view q[] = {"(", "[", "\xE2\x80\x9C", "\xE2\x80\x98", "\xC2\xAB"};   // ( [ left quotes
        std::any_of(std::begin(q), std::end(q), [&](std::string_view o) { return ends_with(last, o); }))
        return true;
    if (ends_with(last, "\"") && std::count(last.begin(), last.end(), '"') % 2 == 1) return true;
    if (ends_with(last, "'") && (last.size() == 1 || !std::isalnum((unsigned char) last[last.size() - 2]))) return true;
    const size_t end = last.find_last_not_of(" \t");
    if (!thinking || end == none) return false;
    const std::string_view written = last.substr(0, end + 1);
    const std::string_view sentence[] = {".", "!", "?", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F", "\xE2\x80\xA6"};   // . ! ? and CJK, ...
    return std::none_of(std::begin(sentence), std::end(sentence), [&](std::string_view o) { return ends_with(written, o); });
}

bool is_url(const std::string& s) { return starts_with(s, "http://") || starts_with(s, "https://"); }

int64_t jint(const json& v, int64_t dflt = 0) { return v.is_number() ? v.get<int64_t>() : dflt; }

}  // namespace

std::unique_lock<std::timed_mutex> Service::turn(Cancel* cancel) {
    std::unique_lock<std::timed_mutex> held(fifo_, std::defer_lock);
    if (cancel == nullptr) {
        held.lock();
        return held;
    }
    while (!held.try_lock_for(std::chrono::milliseconds(250)))
        if (cancel->is_set()) throw ClientGone();
    return held;
}

bool Service::slots_busy() {
    std::lock_guard<std::mutex> lk(status_mu);
    return !live_reqs_.empty();
}

json Service::note_unreadable_tool_images(json& messages, Cancel* cancel) {
    // a picture a tool returned that this server cannot read becomes a short note in its place instead of a 400:
    // clients resend the whole history every turn, so one unreadable tool picture would fail every later request.
    // A picture that encodes stays in the encoder's cache, so the prompt's own encode below finds it.
    json fetched = json::object();   // a URL's bytes, so prepare() does not download the same picture again
    for (auto& m : messages) {
        if (jstr(m, "role") != "tool" || !jget(m, "content").is_array()) continue;
        for (auto& item : m["content"]) {
            if (!item.is_object() || jstr(item, "type") != "image") continue;
            std::string why = vision == nullptr ? "this server has no image encoder" : "";
            if (why.empty()) {
                const std::string src = jstr(item, "source");
                try {
                    if (cancel != nullptr && cancel->is_set()) throw ClientGone();
                    if (offline && is_url(src)) throw std::invalid_argument("offline mode cannot download an image URL");
                    std::string bytes;
                    bool raw = false;
                    if (is_url(src)) {
                        bytes = Vision::download(src);   // outside the turn, as in prepare()
                        fetched[src] = bytes;
                        raw = true;
                    }
                    const auto held = turn(cancel);
                    vision->encode(raw ? bytes : src, raw);
                } catch (const ClientGone&) {
                    throw;
                } catch (const std::exception& e) {
                    why = e.what();
                }
            }
            if (!why.empty()) {
                say("[lamina] a picture a tool returned was left out of the prompt: " + why);
                item = {{"type", "text"}, {"text", "[image omitted: " + why + "]"}};
            }
        }
    }
    return fetched;
}

Prepared Service::prepare(json& messages, const json& tools, const json& kwargs, std::optional<int64_t> max_new0,
                          const std::optional<std::string>& force, json* req, Cancel* cancel) {
    // -> (ids, thinking, max_new); an unset or non-positive max_new means the rest of the context. `force`: without
    // thinking the reply starts with the forced call's opening, so it ends the prompt
    const json fetched = note_unreadable_tool_images(messages, cancel);
    const std::vector<int32_t> encoded_ids = encode_prompt(messages, tools, kwargs);
    std::vector<int32_t> ids = encoded_ids;
    // the prompt's end says, where a template decides for itself (a <|think_off|> or <|think_low|> in a message)
    const std::string prompt_end = ids.size() >= 2 ? tok.decode(std::vector<int32_t>(ids.end() - 2, ids.end())) : "";
    const bool thinking_off = prompt_end == "</think>\n\n" || (prompt_end != "<think>\n" && jget(kwargs, "enable_thinking").is_boolean() &&
                                                               !kwargs["enable_thinking"].get<bool>());
    if (force && thinking_off) {
        const std::vector<int32_t> f = tok.encode(*force, true);
        ids.insert(ids.end(), f.begin(), f.end());
    }
    embeddings = Embeddings{};
    const std::vector<std::string> pics = images_of(messages);
    std::vector<std::pair<std::string, int64_t>> encoded;
    if (!pics.empty()) {
        if (offline && std::any_of(pics.begin(), pics.end(), is_url))
            throw RequestError("offline mode cannot download an image URL; send a data: image instead");
        if (vision == nullptr)
            throw RequestError("this server was started without the vision encoder (run setup again with --vision on), "
                               "so it cannot read images");
        const int32_t pad = tok.encode("<|image_pad|>", true)[0], start = tok.encode("<|vision_start|>", true)[0];
        // an image URL is downloaded first, outside the turn: under it, a slow server held every other request
        std::vector<std::pair<std::string, bool>> sources;
        for (const auto& src : pics) {
            if (cancel != nullptr && cancel->is_set()) throw ClientGone();
            if (fetched.contains(src)) sources.emplace_back(fetched[src].get<std::string>(), true);
            else if (is_url(src)) sources.emplace_back(Vision::download(src), true);
            else sources.emplace_back(src, false);
        }
        {
            // the engine and the image encoder must not work at the same time: encoding takes its turn as requests do
            const auto held = turn(cancel);
            encoded = vision->encode_all(sources, cancel);
        }
        // one <|image_pad|> per image -> one per image token; only the markers the template writes for an image (right
        // after <|vision_start|>) are images, the same text inside a message stays text
        const std::vector<int32_t> literal = tok.encode("<|image_pad|>", false);
        std::vector<int32_t> out;
        size_t k = 0;
        for (size_t j = 0; j < ids.size(); ++j) {
            const int32_t t = ids[j];
            if (t == pad && j > 0 && ids[j - 1] == start) {
                if (k == encoded.size()) throw RequestError("the prompt and its images do not match");
                out.insert(out.end(), (size_t) encoded[k].second, pad);
                ++k;
            } else if (t == pad) {
                out.insert(out.end(), literal.begin(), literal.end());
            } else {
                out.push_back(t);
            }
        }
        if (k != encoded.size()) throw RequestError("the prompt and its images do not match");
        ids = std::move(out);
    }
    int64_t ctx = engine->max_context();
    if (ctx <= 0) {
        if (engine->starting()) throw EngineStarting("the engine is starting (a minute or two); try again shortly");
        // a failed restart leaves max_context 0: the last known context, so the request reaches run(), which starts
        // the engine again
        ctx = engine->known_ctx();
        if (ctx <= 0) throw EngineStarting("the engine is starting (a minute or two); try again shortly");
    }
    const int64_t room = ctx - kCtxSlack - (int64_t) ids.size();
    int64_t max_new = max_new0.value_or(0);
    if (max_new <= 0 || (fit_max_tokens && room < 1)) {
        if (room < 1) {
            forget_prompt(encoded_ids);   // a prompt refused for the context is not kept (#567)
            throw RequestError("request exceeds the context window: prompt (" + std::to_string(ids.size()) +
                               " tokens) leaves no room to answer in the context (" + std::to_string(ctx) +
                               "); requests are never truncated");
        }
        max_new = room;
    } else if (max_new > room) {
        if (!fit_max_tokens) {
            forget_prompt(encoded_ids);
            throw RequestError("request exceeds the context window: prompt (" + std::to_string(ids.size()) +
                               " tokens) + max tokens (" + std::to_string(max_new) + ") exceeds the context (" +
                               std::to_string(ctx) + "); requests are never truncated. Send a smaller max_tokens (at "
                               "most " + std::to_string(std::max<int64_t>(0, room)) + " here), or add \"fit_max_tokens\": "
                               "true to the model's lamina-<model>.json to shorten it to the room left (#545)");
        }
        say("[lamina] max_tokens " + std::to_string(max_new) + " shortened to " + std::to_string(std::max<int64_t>(1, room)) +
            ": the prompt (" + std::to_string(ids.size()) + " tokens) leaves that much of the context (" + std::to_string(ctx) +
            "); \"fit_max_tokens\": false answers 400 instead");
        max_new = std::max<int64_t>(1, room);   // a shorter completion beats a 400 (the default)
    }
    if (!encoded.empty()) {
        // the request's pictures in one file for GENI, written once nothing above refuses the request
        uintmax_t nbytes = 0;
        std::vector<std::string> parts;
        for (const auto& [p, n] : encoded) {
            std::error_code ec;
            nbytes += std::filesystem::file_size(p, ec);
            parts.push_back(p);
        }
        embeddings.path = combined_embeddings_path(vision->dir, nbytes);
        // #2228: each picture's key is its normalized bytes' hash (the encoder's file name)
        for (const auto& p : parts) embeddings.keys.push_back(std::filesystem::path(p).stem().string().substr(0, 16));
        write_temporary(embeddings.path, parts);
    }
    if (req != nullptr && !jget(*req, "lamina_prefix").is_null())
        (*req)["lamina_prefix"] = resolve_prefix((*req)["lamina_prefix"], messages, tools, kwargs, ids, !encoded.empty());
    return {ids, !thinking_off, max_new};
}

json Service::resolve_prefix(const json& spec, const json& messages, const json& tools, const json& kwargs,
                             const std::vector<int32_t>& ids, bool has_images) {
    // "lamina_prefix" marks the start of the prompt as a shared read-only prefix (a long document many short
    // questions follow): {"messages": K}, {"message": I, "chars": N} or {"tokens": N} -> {"tokens": N}, or {} when the
    // prefix cannot be marked (said in the log); a malformed field is a 400
    const std::set<std::string> keys = {"messages", "message", "chars", "tokens"};
    std::set<std::string> given;
    bool ok = spec.is_object() && !spec.empty();
    if (ok)
        for (auto it = spec.begin(); it != spec.end(); ++it) {
            ok = ok && keys.count(it.key());
            given.insert(it.key());
        }
    if (!ok || !(given == std::set<std::string>{"messages"} || given == std::set<std::string>{"tokens"} ||
                 given == std::set<std::string>{"message", "chars"}))
        throw RequestError("lamina_prefix: {\"messages\": K} (the first K messages), {\"message\": I, \"chars\": N} (the "
                           "first N characters of message I) or {\"tokens\": N}");
    for (auto it = spec.begin(); it != spec.end(); ++it) {
        const int64_t lo = it.key() == "message" ? 0 : 1;
        if (!is_int(it.value()) || it.value().get<int64_t>() < lo)
            throw RequestError("lamina_prefix." + it.key() + ": a whole number (" + std::to_string(lo) + " or more)");
    }
    if (has_images) {
        say("[lamina] lamina_prefix ignored: the request has images");
        return json::object();
    }
    std::string what;
    for (auto it = spec.begin(); it != spec.end(); ++it)
        what += (what.empty() ? "" : ",") + it.key() + "=" + std::to_string(it.value().get<int64_t>());
    auto lcp = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return (int64_t) n;
    };
    auto head_ids = [&](const json& msgs) {
        const Marked m = mark_think_literals(msgs, tools, literals);
        const std::string effort = jstr(kwargs, "reasoning_effort");
        const bool off = jget(kwargs, "enable_thinking").is_boolean() && !kwargs["enable_thinking"].get<bool>();
        json kw = json::object();
        const bool strip_effort = effort_end && (off || !(effort.empty() || effort == "xhigh" || effort == "high"));
        for (auto it = kwargs.begin(); it != kwargs.end(); ++it)
            if (!strip_effort || (it.key() != "reasoning_effort" && it.key() != "enable_thinking")) kw[it.key()] = it.value();
        std::string text = tpl->render(m.messages, m.tools, false, kw);
        const size_t cut = text.find(kPrefixEnd);
        if (cut != std::string::npos) text = text.substr(0, cut);
        if (m.changed) {
            auto [t, spans] = unmark_think_literals(text, literals);
            std::vector<Span> plain;
            for (const auto& [a, b] : spans) plain.push_back({a, b});
            return tok.encode(t, true, plain);
        }
        return tok.encode(text, true);
    };
    int64_t n;
    if (spec.contains("tokens")) {
        n = spec["tokens"].get<int64_t>();
    } else if (spec.contains("messages")) {
        const int64_t k = spec["messages"].get<int64_t>();
        if (k >= (int64_t) messages.size()) {
            say("[lamina] lamina_prefix.messages=" + std::to_string(k) + " ignored: the request has " +
                std::to_string(messages.size()) + " message(s) and the prefix must leave at least the last one");
            return json::object();
        }
        n = lcp(head_ids(json(std::vector<json>(messages.begin(), messages.begin() + k))), ids);
    } else {
        const int64_t i = spec["message"].get<int64_t>(), chars = spec["chars"].get<int64_t>();
        const json text = i >= 0 && i < (int64_t) messages.size() ? jget(messages[(size_t) i], "content") : json(nullptr);
        if (!text.is_string() || chars > (int64_t) utf8_len(text.get_ref<const std::string&>())) {
            say("[lamina] lamina_prefix ignored: message " + std::to_string(i) + " has no text of " + std::to_string(chars) +
                " characters");
            return json::object();
        }
        json cut(std::vector<json>(messages.begin(), messages.begin() + i));
        json last = messages[(size_t) i];
        const std::string& s = text.get_ref<const std::string&>();
        size_t b = 0;
        for (int64_t c = 0; c < chars && b < s.size(); ++c) {
            size_t k = 1;
            utf8_at(s, b, &k);
            b += k;
        }
        last["content"] = s.substr(0, b) + kPrefixEnd;
        cut.push_back(last);
        n = lcp(head_ids(cut), ids);
    }
    if (n < 1 || n >= (int64_t) ids.size() - 1) {
        say("[lamina] lamina_prefix " + what + " ignored: it leaves no usable prefix of the " + std::to_string(ids.size()) +
            "-token prompt (common start " + std::to_string(n) + ")");
        return json::object();
    }
    return {{"tokens", n}};
}

json Service::choose(const json& req, Cancel& cancel) {
    // POST /v1/choose: one decision in one forward pass, nothing generated; the chat rendered as for chat completions
    // with thinking off, `answer_prefix` after the assistant header, each option's log-probability from the head
    const json& options = jget(req, "options");
    bool ok = options.is_array() && !options.empty();
    if (ok)
        for (const auto& o : options) ok = ok && o.is_string() && !o.get_ref<const std::string&>().empty();
    if (!ok) throw RequestError("choose: `options` must be a non-empty list of strings");
    std::set<std::string> uniq;
    for (const auto& o : options) uniq.insert(o.get<std::string>());
    if (uniq.size() != options.size()) throw RequestError("choose: `options` repeat");
    std::vector<int32_t> cand;
    for (const auto& o : options) {
        const std::vector<int32_t> t = tok.encode(o.get<std::string>(), false);
        if (t.size() != 1)
            throw RequestError("choose: option '" + o.get<std::string>() + "' is " + std::to_string(t.size()) +
                               " tokens; options must be single tokens (A, B, ...)");
        cand.push_back(t[0]);
    }
    const json& prefix_v = jget(req, "answer_prefix");
    if (!prefix_v.is_null() && !prefix_v.is_string()) throw RequestError("choose: `answer_prefix` must be a string");
    const std::string prefix = prefix_v.is_string() ? prefix_v.get<std::string>() : "";
    Converted c = openai_to_messages(req);
    load();
    Prepared p = prepare(c.messages, c.tools, json{{"enable_thinking", false}}, 1);
    const std::vector<int32_t> pre = tok.encode(prefix, false);
    p.ids.insert(p.ids.end(), pre.begin(), pre.end());
    if ((int64_t) p.ids.size() + 1 + kCtxSlack > engine->max_context()) {
        drop_embeddings();
        throw RequestError("choose: the prompt (" + std::to_string(p.ids.size()) + " tokens) does not fit the context (" +
                           std::to_string(engine->max_context()) + ")");
    }
    const double t0 = now_s();
    json done = json::object();
    json score = json::array();
    for (const int32_t t : cand) score.push_back(t);
    run(p.ids, false, nullptr, 1, json{{"score", score}}, cancel, std::nullopt,
        [&](const std::string& kind, const json& value, const Event*) {
            if (kind == "done") done = value;
            return true;
        });
    const json& lp = jget(done, "scores");
    bool all = lp.is_object();
    for (const int32_t t : cand) all = all && lp.contains(std::to_string(t));
    if (jstr(done, "finish") == "cancel" || !all)
        throw RequestError(std::string("choose: the engine returned no log-probabilities for the options") +
                           (jstr(done, "finish") == "cancel" ? " (cancelled)" : ""));
    json logprobs = json::object(), probs = json::object();
    double top = -INFINITY;
    for (size_t k = 0; k < cand.size(); ++k) {
        const double v = lp[std::to_string(cand[k])].get<double>();
        logprobs[options[k].get<std::string>()] = v;
        top = std::max(top, v);
    }
    double total = 0, mass = 0;
    for (auto it = logprobs.begin(); it != logprobs.end(); ++it) {
        total += std::exp(it.value().get<double>() - top);
        mass += std::exp(it.value().get<double>());
    }
    std::string choice;
    double best = -1;
    for (auto it = logprobs.begin(); it != logprobs.end(); ++it) {
        const double pr = std::exp(it.value().get<double>() - top) / total;
        probs[it.key()] = pr;
        if (pr > best) {
            best = pr;
            choice = it.key();
        }
    }
    return {{"probs", probs}, {"logprobs", logprobs}, {"mass", mass}, {"choice", choice}, {"confidence", best},
            {"prompt_tokens", p.ids.size()}, {"reused", jget(done, "reused")}, {"seconds", py_round(now_s() - t0, 3)}};
}

void Service::run(const std::vector<int32_t>& ids, bool thinking, const json& tools, int64_t max_new, json sampling,
                  Cancel& cancel, std::optional<std::string> force, const Sink& sink) {
    // the request's events; with "prevent_sleep" the PC stays awake while it runs or waits
    if (keep_awake != nullptr) keep_awake->acquire();
    struct Release {
        std::shared_ptr<KeepAwake> k;
        ~Release() {
            if (k != nullptr) k->release();
        }
    } release{keep_awake};
    run_inner(ids, thinking, tools, max_new, std::move(sampling), cancel, std::move(force), sink);
}

void Service::run_inner(const std::vector<int32_t>& ids, bool thinking, const json& tools, int64_t max_new,
                        json sampling, Cancel& cancel, std::optional<std::string> force, const Sink& sink) {
    // the model card's sampling for this thinking state, then the config's block, then the Chat settings shared with
    // apps; the request's own fields win over all of them (a scored one, /v1/choose, reads the model's own logits)
    json defaults = model_card_sampling(model_arch, thinking);
    for (auto it = sampling_defaults.begin(); it != sampling_defaults.end(); ++it) defaults[it.key()] = it.value();
    const json shared = this->shared();
    for (auto it = shared.begin(); it != shared.end(); ++it) defaults[it.key()] = it.value();
    if (!defaults.empty() && !truthy(jget(sampling, "score"))) {
        json merged = defaults;
        if (sampling.is_object())
            for (auto it = sampling.begin(); it != sampling.end(); ++it)
                if (!it.value().is_null()) merged[it.key()] = it.value();
        sampling = std::move(merged);
    }
    std::optional<int64_t> budget = thinking ? reasoning_budget(sampling) : std::nullopt;
    if (thinking && !budget && answer_room && max_new && max_new >= kAnswerRoomMinCap) {
        // max_tokens counts the thinking too: thinking is closed early enough to leave the answer its quarter of the cap
        budget = max_new - std::min(std::max(kAnswerReserveMin, max_new / 4), max_new / 2);
        if (max_new <= 32768)
            say("[lamina] thinking limited to " + std::to_string(*budget) + " of max_tokens " + std::to_string(max_new) +
                " so the answer has room (\"answer_room\": false in the config turns this off)");
    }
    if (budget && max_new > 0) {
        // #984: a budget at or near max_tokens never fires before the reply is cut off while still thinking
        const int64_t reserve = std::min(std::max(kAnswerReserveMin, max_new / 4), max_new / 2);
        if (*budget > max_new - reserve) {
            say("[lamina] thinking capped at " + std::to_string(max_new - reserve) + " of max_tokens " + std::to_string(max_new) +
                " to leave room for the answer (budget " + std::to_string(*budget) + ")");
            budget = std::max<int64_t>(1, max_new - reserve);
        }
    }
    OutputParser parser(thinking, tools, true, tool_call_recovery);
    Detokenizer detok(tok);
    int64_t n = 0;
    std::string finish = "length";
    int32_t run_tok = -1;
    int64_t run_len = 0;
    bool repeated = false;
    int64_t thinking_n = 0;
    const std::vector<std::string> stop_list = stop_strings(sampling);
    std::optional<StopMatcher> stops;
    if (!stop_list.empty()) stops.emplace(stop_list);
    auto cut = [&](std::vector<Event> evs) {
        // the answer's text cut at a stop string; nothing after it is sent
        if (!stops) return evs;
        std::vector<Event> out;
        for (auto& ev : evs) {
            if (stops->hit) break;
            if (ev.kind == "content") {
                const std::string text = stops->push(ev.text);
                if (!text.empty()) out.push_back({"content", text, nullptr});
                continue;
            }
            const std::string held = stops->flush();   // a stop string does not run across a tool call or thinking
            if (!held.empty()) out.push_back({"content", held, nullptr});
            out.push_back(std::move(ev));
        }
        return out;
    };
    std::vector<Event> opening;   // without thinking the prompt ends with the forced opening: parsed as the model's
    if (force && !thinking) {
        opening = parser.feed(*force);
        force.reset();
    }
    std::string tail;   // the last characters written (the newlines before a call)
    bool answered = false, close_retried = false, nf_retried = false, leaving = false, timed_out = false;
    json timings = nullptr;
    int recovery_count = 0, marker_names = 0;
    std::string reasoning_text;
    double repeat_coverage = 0.0;
    bool looped = false;
    int64_t next_loop_check = kLoopCheckEvery;
    std::vector<int32_t> raw_ids;
    const std::string emb = embeddings.path;
    const std::vector<std::string> img_keys = embeddings.keys;
    embeddings = Embeddings{};
    if (!emb.empty() && !img_keys.empty()) {
        if (!sampling.is_object()) sampling = json::object();
        sampling["lamina_img_keys"] = img_keys;   // #2228: sent beside the embeddings file
    }
    std::vector<json> segments;   // each pass's own DONE (a reasoning-budget continuation is another pass)
    json* tr = trace;
    const double waiting = mono_s();
    std::optional<double> first_engine_token_s;
    bool first_token_noted = false;   // the record's first_token_s: read and written under status_mu, once
    const bool par = engine->batch() > 0;
    json own_st = json::object();
    std::deque<std::pair<double, int64_t>> own_rate;
    json& st = par ? own_st : status;
    std::deque<std::pair<double, int64_t>>& rt = par ? own_rate : rate;
    const std::string request_id = random_hex(12);
    const std::optional<double> deadline = queue_timeout_s ? std::optional<double>(mono_s() + *queue_timeout_s) : std::nullopt;
    cancel.queue_deadline = deadline;   // with batch slots the engine waits for its turn, and keeps the deadline
    cancel.queue_started = [this, request_id] {
        std::lock_guard<std::mutex> lk(status_mu);
        const auto it = terminal_requests.find(request_id);
        if (it != terminal_requests.end() && jstr(it->second, "state") != "reading") it->second["state"] = "reading";
    };
    cancel.replica_selected = [this, request_id](int index) {
        std::lock_guard<std::mutex> lk(status_mu);
        const auto it = terminal_requests.find(request_id);
        if (it != terminal_requests.end()) it->second["replica"] = index;
    };
    bool queued_counted = true;
    {
        std::lock_guard<std::mutex> lk(status_mu);
        status["queued"] = jint(status["queued"]) + 1;
        cancel.request_id = request_id;   // (the client watch reads it under the lock)
        terminal_requests[request_id] = {{"id", request_id}, {"state", "queued"}, {"queued_at", now_s()},
                                         {"admitted_at", nullptr}, {"prompt_tokens", ids.size()}};
        active_cancels.insert(&cancel);
    }
    auto outer_finally = [&] {
        std::lock_guard<std::mutex> lk(status_mu);
        if (queued_counted) status["queued"] = jint(status["queued"]) - 1;
        terminal_requests.erase(request_id);
        active_cancels.erase(&cancel);
        if (!emb.empty()) std::remove(emb.c_str());
    };
    std::unique_lock<std::timed_mutex> turn(fifo_, std::defer_lock);
    try {
        if (!par) {
            // #1619: a request waiting for the single turn sends a keep-alive every QUEUE_BEAT_S; a client that left
            // while queued gives its place up
            while (!turn.owns_lock()) {
                if (cancel.is_set()) throw QueueCancelled("the client disconnected while waiting for the engine");
                const double left = deadline ? *deadline - mono_s() : kQueueBeatS;
                if (left <= 0) throw QueueWaitExpired("the inference queue wait deadline expired; retry later");
                if (turn.try_lock_for(std::chrono::duration<double>(std::min(kQueueBeatS, left)))) {
                    if (cancel.is_set()) throw QueueCancelled("the client disconnected while waiting for the engine");
                    if (deadline && mono_s() >= *deadline)
                        throw QueueWaitExpired("the inference queue wait deadline expired; retry later");
                } else if (!sink("ping", nullptr, nullptr)) {
                    leaving = true;
                    outer_finally();
                    return;
                }
            }
        }
    } catch (const QueueCancelled&) {
        if (turn.owns_lock()) turn.unlock();
        outer_finally();
        sink("done", json{{"finish", "cancel"}, {"completion_tokens", 0}, {"reused", 0}, {"timings", nullptr},
                          {"reasoning_tokens", 0}, {"reasoning_recoveries", 0}}, nullptr);
        return;
    } catch (const QueueWaitExpired&) {
        if (turn.owns_lock()) turn.unlock();
        {
            std::lock_guard<std::mutex> lk(status_mu);
            ++queue_timeouts;
        }
        outer_finally();
        throw;
    }
    // ---- the turn is held (one at a time), or not needed (batch slots)
    auto bookkeeping = [&] {
        // #266: this request's status, history and totals while still holding the turn
        std::lock_guard<std::mutex> lk(status_mu);
        if (truthy(jget(st, "busy")) && !timed_out) {
            const json last = request_stats(segments);   // only this request's own DONEs count
            const double started = jget(st, "started").is_number() ? st["started"].get<double>() : now_s();
            const json hit_rate = jint(jget(last, "lookups")) ? json(py_round((double) jint(last["hits"]) / (double) jint(last["lookups"]), 3))
                                                              : json(nullptr);
            // #588: the share another GPU computed or the GPU read over PCIe (not in the hit rate's lookups)
            const int64_t routed = jint(jget(last, "lookups")) + jint(jget(last, "offloaded"));
            const json pcie_share = !jget(last, "offloaded").is_null() && routed
                                        ? json(py_round((double) jint(last["offloaded"]) / (double) routed, 3)) : json(nullptr);
            const int64_t seen = prompt_tokens_seen((int64_t) ids.size(), last);
            json h = {{"id", request_id}, {"time", started}, {"duration_s", py_round(now_s() - started, 1)},
                      {"finish", finish}, {"prompt_tokens", seen}, {"reused", jget(last, "reused")}, {"output_tokens", n},
                      {"first_engine_token_s", first_engine_token_s ? json(*first_engine_token_s) : json(nullptr)},
                      {"prompt_total", ids.size()}, {"prompt_read", jget(last, "prompt_read")},
                      {"reasoning_recoveries", recovery_count}, {"reasoning_repeat_coverage", py_round(repeat_coverage, 3)},
                      {"engine_generated", jget(last, "generated")}, {"prompt_ms", jget(last, "prompt_ms")},
                      {"decode_ms", jget(last, "decode_ms")},
                      {"decode_tok_s", n && jint(jget(last, "generated")) && jget(last, "decode_ms").is_number() &&
                                               last["decode_ms"].get<double>() > 0
                                           ? json(py_round((double) jint(last["generated"]) / (last["decode_ms"].get<double>() / 1000), 1))
                                           : json(nullptr)},
                      {"hit_rate", hit_rate}, {"pcie_share", pcie_share}, {"ram_blobs", jget(last, "ram_blobs")},
                      {"file_blobs", jget(last, "file_blobs")}, {"file_mb", jget(last, "file_mb")},
                      {"drafts_offered", jget(last, "drafts_offered")}, {"drafts_accepted", jget(last, "drafts_accepted")}};
            if (truthy(jget(last, "expert_tiers"))) h["expert_tiers"] = last["expert_tiers"];
            history.push_back(h);
            while (history.size() > 500) history.pop_front();
            json& t = totals;
            t["requests"] = jint(t["requests"]) + 1;
            t["prompt_tokens"] = jint(t["prompt_tokens"]) + seen;
            t["reused"] = jint(t["reused"]) + jint(jget(last, "reused"));
            t["output_tokens"] = jint(t["output_tokens"]) + n;
            t["prompt_ms"] = t["prompt_ms"].get<double>() + (jget(last, "prompt_ms").is_number() ? last["prompt_ms"].get<double>() : 0.0);
            t["decode_ms"] = t["decode_ms"].get<double>() + (jget(last, "decode_ms").is_number() ? last["decode_ms"].get<double>() : 0.0);
            t["drafts_offered"] = jint(t["drafts_offered"]) + jint(jget(last, "drafts_offered"));
            t["drafts_accepted"] = jint(t["drafts_accepted"]) + jint(jget(last, "drafts_accepted"));
            const std::optional<double> ft0 = jget(st, "first_token").is_number() ? std::optional<double>(st["first_token"].get<double>())
                                                                                  : std::nullopt;
            if (observe_latencies)
                observe_latencies(ft0 ? std::optional<double>(*ft0 - started) : std::nullopt, n,
                                  (jget(last, "decode_ms").is_number() ? last["decode_ms"].get<double>() : 0.0) / 1000,
                                  now_s() - started);
            if (!segments.empty()) {   // the engine's clock for THIS request
                timings = request_timings(seen, n, last);
                if (timings.is_object()) {
                    last_timings = timings;
                    last_timings["at"] = (int64_t) now_s();
                } else {
                    last_timings = nullptr;
                }
            }
            last_request_at = now_s();
            const double now = now_s();
            const double el = now - started;
            const double r = ft0 ? (double) n / std::max(0.25, now - *ft0) : 0.0;   // floored as the live rate is
            std::string hit_msg;
            if (hit_rate.is_number()) {
                char hb[96];
                std::snprintf(hb, sizeof hb, ", expert cache %.1f%% hit", hit_rate.get<double>() * 100);
                hit_msg = hb;
                if (pcie_share.is_number() && pcie_share.get<double>() > 0) {
                    std::snprintf(hb, sizeof hb, " (+%.1f%% of the routed experts over PCIe)", pcie_share.get<double>() * 100);
                    hit_msg += hb;
                }
            }
            char buf[256];
            std::snprintf(buf, sizeof buf, "[lamina] done: %lld tokens in %.0f s (%.1f tok/s) (%s, cancel=%s)", (long long) n, el, r,
                          finish.c_str(), cancel.is_set() ? "True" : "False");
            say(std::string(buf) + hit_msg);
            if (finish == "length" && (parser.state() == "reasoning" || parser.state() == "rcall"))
                say("[lamina] the reply reached max tokens while still thinking, so it has no answer: a thinking budget "
                    "(reasoning_budget_tokens, in the request or in lamina-<model>.json for every request) leaves room to answer");
            if (std::getenv("LAMINA_DEBUG") != nullptr && !raw_ids.empty()) say("[lamina] raw: " + py_repr_value(json(tok.decode(raw_ids))));
        }
        st["busy"] = false;
        st.erase("tail");   // #212: the answer's end is not kept once it is done
        st.erase("tool");
        if (par) {
            live_reqs_.erase(&st);
            if (!live_reqs_.empty()) {
                for (auto it = live_reqs_.rbegin()->second.first->begin(); it != live_reqs_.rbegin()->second.first->end(); ++it)
                    status[it.key()] = it.value();
            } else {
                status["busy"] = false;
                status.erase("tail");
                status.erase("tool");
            }
        }
    };
    try {
        {
            std::lock_guard<std::mutex> lk(status_mu);
            if (tr != nullptr) {
                (*tr)["queue_s"] = py_round(jget(*tr, "queue_s").get<double>() + mono_s() - waiting, 3);
                (*tr)["state"] = "generating";
            }
            if (!par) {
                status["queued"] = jint(status["queued"]) - 1;
                queued_counted = false;
                terminal_requests[request_id]["state"] = "admitted";
                terminal_requests[request_id]["admitted_at"] = now_s();
            }
        }
        // issue #27: it died in an earlier request (or was unloaded): start it again instead of failing
        if (!par) {
            ensure_loaded();
        } else if (!(loaded() && !(vision != nullptr && !vision->alive()))) {
            std::lock_guard<std::timed_mutex> load_turn(fifo_);
            ensure_loaded();
        }
        {
            std::lock_guard<std::mutex> lk(status_mu);
            if (par) {
                status["queued"] = jint(status["queued"]) - 1;
                queued_counted = false;
                terminal_requests[request_id]["state"] = "waiting";
                terminal_requests[request_id]["admitted_at"] = now_s();
            }
            st["busy"] = true;
            st["phase"] = "reading the prompt";
            st["prompt_tokens"] = ids.size();
            st["generated"] = 0;
            st["started"] = now_s();
            st["first_token"] = nullptr;
            st["tool"] = nullptr;
            st["tail"] = "";
            st["max_tokens"] = max_new;
            st["reasoning_recoveries"] = 0;
            st["thinking_budget"] = budget ? json(*budget) : json(nullptr);
            if (par) {
                live_reqs_[&st] = {&st, &rt};
                for (auto it = st.begin(); it != st.end(); ++it) status[it.key()] = it.value();
            }
            last_request_at = now_s();
            rt.clear();   // the previous request's samples must not leak into this one
        }
        double last_print = now_s();
        std::vector<int32_t> prompt = ids;
        int64_t thought = 0;   // the reasoning tokens so far (the budget's count)
        std::optional<Held> held_before = par || !emb.empty() ? std::nullopt : engine->held;
        for (const Event& ev : opening)
            if (!sink("event", nullptr, &ev)) {
                leaving = true;
                break;
            }
        while (!leaving) {
            const uint64_t dones_before = engine->done_count();
            const size_t segments_before = segments.size();
            std::optional<std::vector<int32_t>> recover_prompt;
            std::optional<int32_t> named_marker;
            std::vector<int32_t> seg;
            bool wrap = false, nonfinite = false, opens = false;
            double last_out = mono_s();
            auto on_token = [&](int32_t t) -> bool {
                ++n;
                if (!first_engine_token_s) {
                    first_engine_token_s = py_round(mono_s() - waiting, 3);
                    std::lock_guard<std::mutex> lk(status_mu);
                    const auto it = terminal_requests.find(request_id);
                    if (it != terminal_requests.end()) it->second["first_engine_token_s"] = *first_engine_token_s;
                }
                if (tr != nullptr && !first_token_noted) {   // (the client watch writes the record meanwhile)
                    first_token_noted = true;
                    std::lock_guard<std::mutex> lk(status_mu);
                    if (jget(*tr, "first_token_s").is_null())
                        (*tr)["first_token_s"] = py_round(mono_s() - (*tr)["_clock"].get<double>(), 3);
                }
                if (stop_ids.count(t)) {
                    if (force && parser.state() == "reasoning" && parser.buf() == kThinkEnd) {
                        opens = true;   // #537: the held </think> was the end: the call opens there
                        return false;
                    }
                    if (literal_markers && marker_names < kMarkerNames && parser.buf() != kThinkEnd &&
                        names_stop_marker(tok.decode(raw_ids), parser.state())) {
                        --n;   // its text comes in plain tokens
                        named_marker = t;
                        return false;
                    }
                    finish = "stop";
                    raw_ids.push_back(t);
                    return false;
                }
                raw_ids.push_back(t);
                seg.push_back(t);
                thinking_n += parser.state() == "reasoning" || parser.state() == "rcall";
                run_len = t == run_tok ? run_len + 1 : 1;
                run_tok = t;
                if (repeat_stop_tokens && run_len >= repeat_stop_tokens) {
                    repeated = true;   // #606: a degenerate output, not an answer: end it here
                    return false;
                }
                const std::string piece = detok.push(t);
                tail = (tail + piece);
                if (tail.size() > 2) tail = tail.substr(tail.size() - 2);
                std::vector<Event> evs = cut(parser.feed(piece));
                note(n, evs, st, rt);
                last_print = progress_line(last_print, st);
                for (const Event& ev : evs) {
                    if (!reasoning_loop_recovery.empty() && ev.kind == "reasoning") reasoning_text += ev.text;
                    if (ev.kind == "content" || ev.kind == "tool_start" || ev.kind == "tool_call") answered = true;
                    if (!sink("event", nullptr, &ev)) {
                        leaving = true;
                        return false;
                    }
                }
                if (!evs.empty()) {
                    last_out = mono_s();
                } else if (mono_s() - last_out >= kKeepaliveS) {
                    last_out = mono_s();
                    if (!sink("ping", nullptr, nullptr)) {
                        leaving = true;
                        return false;
                    }
                }
                if (stops && stops->hit) {
                    finish = "stop";   // the engine is stopped, as for a stop token
                    return false;
                }
                if (!reasoning_loop_recovery.empty() && !recovery_count && parser.state() == "reasoning" &&
                    n >= next_loop_check && parser.buf().empty() && !parser.has_pending() && !detok.pending()) {
                    next_loop_check = n + kLoopCheckEvery;
                    repeat_coverage = reasoning_repeat_coverage(reasoning_text);
                    if (repeat_coverage >= kLoopCoverage) {
                        if (reasoning_loop_recovery == "stop") {
                            looped = true;   // #728: end the reply here, as for a repeated token
                            return false;
                        }
                        if (emb.empty()) {   // "recover": not for a request with pictures
                            recover_prompt = focused_recovery_prompt(tok, ids, raw_ids);
                            if (recover_prompt) return false;
                        }
                    }
                }
                if (budget && parser.state() == "reasoning") {
                    ++thought;
                    // at a clean point: no tag held back, no character split across tokens
                    if (thought >= *budget && parser.buf().empty() && !detok.pending()) {
                        wrap = true;
                        return false;
                    }
                }
                if (force) {
                    if (std::any_of(evs.begin(), evs.end(), [](const Event& e) { return e.kind == "tool_start" || e.kind == "tool_call"; })) {
                        force.reset();   // the model is writing a call of its own
                    } else if (parser.state() == "content" && parser.buf().empty() && !detok.pending()) {
                        opens = true;
                        return false;
                    }
                }
                return true;
            };
            auto on_beat = [&]() -> bool {
                last_print = progress_line(last_print, st);
                last_out = mono_s();
                if (!sink("ping", nullptr, nullptr)) {
                    leaving = true;
                    return false;
                }
                return true;
            };
            auto after_pass = [&] {
                if (engine->done_count() != dones_before) segments.push_back(engine->last_figures());
                if (held_before && !segments.empty()) {
                    const auto note = held_departure(tok, *held_before, prompt, jint(jget(segments[0], "reused")));
                    held_before.reset();
                    if (note) say(*note);
                }
            };
            // tokens the engine did not draw this pass join the reply, and the next pass's prompt, as if it had
            auto join_tokens = [&](const std::vector<int32_t>& tokens) {
                for (const int32_t t : tokens) {
                    ++n;
                    raw_ids.push_back(t);
                    seg.push_back(t);
                    thinking_n += parser.state() == "reasoning" || parser.state() == "rcall";
                    const std::string piece = detok.push(t);
                    tail = tail + piece;
                    if (tail.size() > 2) tail = tail.substr(tail.size() - 2);
                    std::vector<Event> evs = cut(parser.feed(piece));
                    note(n, evs, st, rt);
                    for (const Event& ev : evs) {
                        if (!reasoning_loop_recovery.empty() && ev.kind == "reasoning") reasoning_text += ev.text;
                        if (ev.kind == "content" || ev.kind == "tool_start" || ev.kind == "tool_call") answered = true;
                        if (!sink("event", nullptr, &ev)) leaving = true;
                    }
                }
            };
            try {
                engine->generate(prompt, max_new - n, sampling, cancel, emb, on_token, on_beat);
            } catch (const EngineNonFinite&) {
                nonfinite = true;
            } catch (const EngineSilent& e) {   // #481: the STOP was never acknowledged: the engine was given up
                finish = "error";
                say_died(e);
                after_pass();
                if (!leaving && !cancel.is_set()) throw;
            } catch (const EngineDied& e) {
                finish = "error";
                say_died(e);
                after_pass();
                throw;
            } catch (const EngineError& e) {   // the engine's ERR line
                finish = "error";
                say(std::string("[lamina] the engine reported an error: ") + e.what());
                after_pass();
                throw;
            }
            after_pass();
            if (leaving) break;
            if (nonfinite && !cancel.is_set()) {
                // #879: the engine emitted none of the poisoned window's tokens and reads its next prompt from token 0
                if (n == 0 && !nf_retried) {
                    nf_retried = true;
                    segments.resize(segments_before);   // the retry's DONE, not this one's, says what was reused
                    say("[lamina] the engine's logits became non-finite before the first token (#879); it dropped its "
                        "cached state: sending the request again once");
                    continue;
                }
                if (n == 0) {
                    finish = "error";
                    say("[lamina] the engine's logits became non-finite again on the retry (#879): the request ends "
                        "with an error");
                    throw EngineNonFinite("the model's output became non-finite (NaN) twice in a row (#879); the engine "
                                          "has dropped its cached state - send the request again");
                }
                say("[lamina] the engine's logits became non-finite after " + std::to_string(n) + " tokens (#879): the "
                    "reply ends here as \"length\"; the engine dropped its cached state, so the next request reads its "
                    "prompt from the start");
                finish = "length";
                break;
            }
            if (recover_prompt && !cancel.is_set() && n < max_new) {
                ++recovery_count;
                // only the two settings that keep the same words coming are raised: temperature to at least 1.0,
                // presence penalty to at least 1.5
                const double t0 = jget(sampling, "temperature").is_number() ? sampling["temperature"].get<double>() : 0.0;
                const double p0 = jget(sampling, "presence_penalty").is_number() ? sampling["presence_penalty"].get<double>() : 0.0;
                if (std::max(1.0, t0) != t0 || std::max(1.5, p0) != p0) {
                    sampling["temperature"] = std::max(1.0, t0);
                    sampling["presence_penalty"] = std::max(1.5, p0);
                }
                prompt = *recover_prompt;
                {
                    std::lock_guard<std::mutex> lk(status_mu);
                    st["reasoning_recoveries"] = recovery_count;
                    if (tr != nullptr) {
                        (*tr)["reasoning_recoveries"] = recovery_count;
                        (*tr)["reasoning_repeat_coverage"] = py_round(repeat_coverage, 3);
                    }
                }
                char buf[200];
                std::snprintf(buf, sizeof buf, "[lamina] repeated reasoning detected at %lld tokens (coverage=%.3f); "
                              "resuming the same output with the low-effort instruction (reasoning_loop_recovery)",
                              (long long) n, repeat_coverage);
                say(buf);
                continue;
            }
            if (named_marker && !cancel.is_set()) {
                // the marker's text in plain tokens, as a message's text reads it; the engine continues from what it holds
                const std::vector<int32_t> extra = tok.encode(tok.decode({*named_marker}), false);
                if (max_new - n - (int64_t) extra.size() < 1) break;   // no room left: "length"
                ++marker_names;
                say("[lamina] the reply named " + tok.decode({*named_marker}) + " in its text: continuing past it (literal_markers)");
                join_tokens(extra);
                if (stops && stops->hit) {
                    finish = "stop";
                    break;
                }
                prompt.insert(prompt.end(), seg.begin(), seg.end());
                continue;
            }
            if (reasoning_close_retry && thinking && finish == "stop" && !close_retried && !answered && !wrap && !opens &&
                parser.state() == "reasoning" && parser.buf() != kThinkEnd && !(stops && stops->hit) && !cancel.is_set()) {
                // #1053: the model wrote its reasoning and stopped before </think>: close the thinking once and let it answer
                close_retried = true;
                const std::vector<int32_t> extra = tok.encode(kReasoningClose, true);
                if (max_new - n - (int64_t) extra.size() >= 1) {
                    say("[lamina] the reply ended inside its thinking with no answer: closing the thinking once and "
                        "continuing (reasoning_close_retry)");
                    for (const int32_t t : extra) {
                        ++n;
                        raw_ids.push_back(t);
                        thinking_n += parser.state() == "reasoning" || parser.state() == "rcall";
                        std::vector<Event> evs = cut(parser.feed(detok.push(t)));
                        note(n, evs, st, rt);
                        for (const Event& ev : evs)
                            if (!sink("event", nullptr, &ev)) leaving = true;
                    }
                    prompt.insert(prompt.end(), seg.begin(), seg.end());
                    prompt.insert(prompt.end(), extra.begin(), extra.end());
                    finish = "length";
                    continue;
                }
            }
            if (!(wrap || opens) || cancel.is_set()) break;
            // #123: the thinking reached its budget: close it the way the model would (a short wrap-up and </think>)
            // and let it answer; the next pass's prompt is this one plus what was generated plus the wrap-up, so the
            // engine continues from the prefix it already holds. A forced call opens the same way.
            const std::vector<int32_t> spill = wrap ? engine->drained_tokens() : std::vector<int32_t>{};
            const std::string spill_text = spill.empty() ? "" : tok.decode(spill);
            if (!spill.empty() && std::none_of(spill.begin(), spill.end(), [&](int32_t t) { return stop_ids.count(t) > 0; }) &&
                spill_text.find('<') == std::string::npos && spill_text.find("\xEF\xBF\xBD") == std::string::npos &&
                max_new - n - (int64_t) spill.size() > 1) {
                // the engine generated these past the budget's stop before the STOP reached it, and its session holds
                // them: they join the thinking, so the wrap-up continuation extends the engine's state
                join_tokens(spill);
            }
            std::string text;
            if (wrap) {
                budget.reset();
                {
                    std::lock_guard<std::mutex> lk(status_mu);
                    st["thinking_budget"] = nullptr;
                }
                text = std::string(kReasoningWrapUp) + force.value_or("");
            } else {
                size_t nl = 0;
                for (auto it = tail.rbegin(); it != tail.rend() && *it == '\n'; ++it) ++nl;
                text = std::string(2 - std::min<size_t>(2, nl), '\n') + force.value_or("");
            }
            force.reset();
            const std::vector<int32_t> extra = tok.encode(text, true);
            if (max_new - n - (int64_t) extra.size() < 1) break;   // no room left to answer: "length"
            if (wrap) say("[lamina] thinking budget reached (" + std::to_string(thought) + " tokens): wrapping up the thinking");
            for (const int32_t t : extra) {
                ++n;
                raw_ids.push_back(t);
                thinking_n += parser.state() == "reasoning" || parser.state() == "rcall";
                std::vector<Event> evs = cut(parser.feed(detok.push(t)));
                note(n, evs, st, rt);
                for (const Event& ev : evs)
                    if (!sink("event", nullptr, &ev)) leaving = true;
            }
            if (stops && stops->hit) {
                finish = "stop";
                break;
            }
            prompt.insert(prompt.end(), seg.begin(), seg.end());
            prompt.insert(prompt.end(), extra.begin(), extra.end());
        }
        // the engine holds this prompt and every token it sent: the next turn's prompt takes those ids up to where its
        // text differs (PromptEncoder.remember_reply)
        if (!par && emb.empty() && engine->held && engine->held->ids.size() >= ids.size() &&
            std::equal(ids.begin(), ids.end(), engine->held->ids.begin()))
            prompts->remember_reply(ids, std::vector<int32_t>(engine->held->ids.begin() + (ptrdiff_t) ids.size(),
                                                              engine->held->ids.end()));
        if (leaving) {
            finish = "disconnect";   // the client disconnected mid-stream
        } else if (cancel.is_set()) {
            finish = "cancel";
        } else if (looped) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "[lamina] the thinking repeated the same passages (coverage=%.3f) at %lld tokens: "
                          "ended as \"length\" (reasoning_loop_recovery: \"stop\" in lamina-<model>.json; remove it to turn "
                          "this off)", repeat_coverage, (long long) n);
            say(buf);
        } else if (repeated) {
            say("[lamina] the reply repeated one token (" + py_repr_value(json(tok.decode({run_tok}))) + ") " + std::to_string(run_len) +
                " times in a row: ended as \"length\" (repeat_stop_tokens in lamina-<model>.json; 0 turns this off). If a "
                "new request with a short prompt does the same, restart the server and report it (#606)");
        }
    } catch (...) {
        if (finish == "length") finish = "error";
        bookkeeping();
        if (turn.owns_lock()) turn.unlock();
        outer_finally();
        throw;
    }
    bookkeeping();
    if (turn.owns_lock()) turn.unlock();
    outer_finally();
    if (leaving) return;   // nothing more is sent to a client that left
    for (const Event& ev : cut(parser.finish(finish)))
        if (!sink("event", nullptr, &ev)) return;
    if (parser.rescued || parser.refused) {
        {
            std::lock_guard<std::mutex> lk(status_mu);
            totals["tool_calls_from_reasoning"] = jint(jget(totals, "tool_calls_from_reasoning")) + parser.rescued;
        }
        say("[lamina] tool calls inside the thinking: " + std::to_string(parser.rescued) + " read as calls, " +
            std::to_string(parser.refused) + " kept as reasoning (quoted, or the turn did not end on a stop)");
    }
    if (stops && !stops->hit && !stops->held().empty()) {
        const Event ev{"content", stops->flush(), nullptr};   // the held tail was not a stop string after all
        if (!sink("event", nullptr, &ev)) return;
    }
    json done = {{"finish", finish}, {"completion_tokens", n}, {"reused", jint(jget(timings, "cache_n"))},
                 {"timings", timings}, {"reasoning_tokens", thinking_n}, {"reasoning_recoveries", recovery_count}};
    if (!segments.empty() && segments.back().contains("scores")) done["scores"] = segments.back()["scores"];
    if (stops && stops->hit) done["stop_sequence"] = *stops->hit;
    sink("done", done, nullptr);
}

}  // namespace lamina
