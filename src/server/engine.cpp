#include "engine.hpp"

#include "log.hpp"
#include "lamina/program/engine.hpp"
#include "lamina/program/serve_io.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace lamina {
namespace {

using Clock = std::chrono::steady_clock;

// the engine's own words for a verifier that cannot go on: it may even print DONE, but must not take another request
const char* kFatalPrefixes[] = {"ERR verify: timed out at layer ", "ERR verify batch: timed out at layer ",
                                "ERR verify: an earlier window never finished",
                                "ERR verify batch: an earlier window never finished"};

constexpr int64_t kPpChunkMax = 32768;   // the longest prompt chunk before the first PP line
constexpr double kPpFloorTokS = 50.0;    // ... read at the slowest PC's rate
constexpr double kPpSlack = 3.0;         // a chunk may take this many times the last one's time
constexpr int kQuitWaitS = 120;          // an engine's QUIT: the time to free its memory and save what it learned

bool digits(const std::string& s, bool sign = false) {
    if (s.empty()) return false;
    size_t i = sign && s[0] == '-' ? 1 : 0;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i)
        if (s[i] < '0' || s[i] > '9') return false;
    return true;
}

/// Python's repr of a float, as the keys of a GEN line write it
std::string fr(double v) { return py_repr(v); }

}  // namespace

// ---------------------------------------------------------------------------------------------- LaminaEngine
double LaminaEngine::ready_timeout_s = 900.0;

LaminaEngine::LaminaEngine(std::vector<std::string> args, std::string log, bool lazy, EngineProcess process)
    : args_(std::move(args)), log_(std::move(log)), process_(std::move(process)) {
    for (size_t i = 0; i + 1 < args_.size(); ++i) {
        if (args_[i] == "--native") model_path = args_[i + 1];
        if (args_[i] == "--max-context") max_context_ = std::atoll(args_[i + 1].c_str());
    }
    if (max_context_ <= 0) max_context_ = 4096;
    if (!lazy) start();
}

LaminaEngine::~LaminaEngine() {
    try {
        close();
    } catch (const EngineStuck&) {   // the process is ending: the engine thread ends with it
        if (thread_.joinable()) thread_.detach();
        if (pump_.joinable()) pump_.detach();
    }
}

void LaminaEngine::start() {
    std::lock_guard<std::mutex> lk(start_mu_);
    unloaded_ = false;
    max_context_ = 0;   // before this start is visible: never its predecessor's
    {
        const std::lock_guard<std::mutex> g(cache_mu_);
        cache_ = json::object();   // nor its cache's state: this one prints its own before READY
    }
    in_ = std::make_shared<LineChannel>();
    out_ = std::make_shared<LineChannel>();
    auto partial = std::make_shared<std::string>();
    std::shared_ptr<LineChannel> in = in_, out = out_;
    lamina::program::ServeIo io;
    io.read_line = [in](std::string& line) { return in->pop(line, -1) == 1; };
    io.write = [out, partial](std::string_view text) {
        partial->append(text);
        size_t at;
        while ((at = partial->find('\n')) != std::string::npos) {
            std::string line = partial->substr(0, at);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            out->push(std::move(line));
            partial->erase(0, at + 1);
        }
    };
    io.in_process = true;
    io.end_input = [in] { in->close(); };
    lamina::program::set_serve_io(std::move(io));
    if (!log_.empty()) {
        // a crash report has the time: when an engine started
        std::ofstream f(log_, std::ios::app);
        char when[32];
        const std::time_t t = std::time(nullptr);
        std::strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
        std::string cmd;
        for (const auto& a : args_) cmd += " " + a;
        f << "[lamina] " << when << " engine started: lamina --serve" << cmd << "\n";
    }
    std::vector<std::string> argv{"lamina", "--serve"};
    argv.insert(argv.end(), args_.begin(), args_.end());
    rc_ = -1;
    ended_ = true;   // not alive until READY
    finished_ = false;
    if (!process_.exe.empty()) {
        argv[0] = process_.exe;
        start_process(argv);
    } else thread_ = std::thread([this, argv, out] {
        std::vector<char*> ptrs;
        std::vector<std::string> own(argv);
        for (auto& a : own) ptrs.push_back(a.data());
        ptrs.push_back(nullptr);
        try {   // an error the engine throws ends it, not the server
            rc_ = lamina::program::run((int) own.size(), ptrs.data());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "lamina: the engine failed: %s\n", e.what());
            rc_ = 1;
        } catch (...) {
            std::fprintf(stderr, "lamina: the engine failed\n");
            rc_ = 1;
        }
        ended_ = true;
        out->close();
        finished_ = true;
    });
    const auto t0 = Clock::now();
    std::string line;
    bool timed_out = false;
    while (true) {
        double left = ready_timeout_s > 0 ? ready_timeout_s - since(t0) : -1;
        if (ready_timeout_s > 0 && left <= 0) {
            timed_out = true;
            break;
        }
        const int got = out_->pop(line, left < 0 ? -1.0 : std::min(left, 1.0));
        if (got == 0) continue;
        if (got < 0) break;
        if (starts_with(line, "INFO ")) {
            for (size_t i = 1; const auto& kv : split_ws(line)) {
                if (i++ == 1) continue;
                const size_t eq = kv.find('=');
                const std::string k = kv.substr(0, eq), v = eq == std::string::npos ? "" : kv.substr(eq + 1);
                info[k] = digits(v, true) ? json(std::atoll(v.c_str())) : json(v);
            }
        } else if (starts_with(line, "CACHE ")) {   // the prompt cache at rest
            cache_line(line);
        } else if (starts_with(line, "READY")) {
            const std::vector<std::string> f = split_ws(line);
            max_context_ = f.size() > 1 ? std::atoll(f[1].c_str()) : 0;
            can_stop = std::find(f.begin() + std::min<size_t>(2, f.size()), f.end(), "stop") != f.end();
            break;
        }
    }
    if (max_context_ <= 0) {
        if (proc_ != nullptr) {   // a child is ended, and nothing of it is left half-started
            proc_->kill();
            in_->close();
            if (reader_.joinable()) reader_.join();
            if (thread_.joinable()) thread_.join();
            proc_.reset();
        }
        if (timed_out) {
            // an engine thread cannot be ended from here: it keeps its GPU until it returns
            throw EngineDied("the engine did not report READY within " + std::to_string((int) ready_timeout_s) +
                             " s" + (log_.empty() ? "" : " (see " + log_ + ")"));
        }
        if (thread_.joinable()) thread_.join();
        throw EngineDied("the engine exited before it was ready (exit code " + std::to_string(rc_.load()) + ")" +
                         (log_.empty() ? "" : " (see " + log_ + ")") + "\n" + death_note());
    }
    known_ctx_ = max_context_;
    if (info.contains("engine")) info["version"] = info["engine"].is_string() ? info["engine"] : json(py_dumps(info["engine"]));
    batch_ = info.contains("batch_slots") && info["batch_slots"].is_number_integer() ? info["batch_slots"].get<int>() : 0;
    for (size_t i = 0; i + 1 < args_.size(); ++i)
        if ((args_[i] == "--batch" || args_[i] == "--slots") && std::atoi(args_[i + 1].c_str()) != batch_)
            say("[lamina] parallel requests: " + args_[i + 1] + " asked, the engine runs " +
                (batch_ ? std::to_string(batch_) : std::string("one at a time")) + " (its log says why)");
    ended_ = false;
    if (batch_) start_batch();
}

bool LaminaEngine::alive() { return !ended_.load() && thread_.joinable() && !unloaded_; }

void LaminaEngine::close() {
    // STOP first: a request still running ends at its next step instead of being waited for; then QUIT, and the
    // engine frees its memory itself (unpinning tens of GB takes a while). An engine thread cannot be ended from
    // outside: one that does not return is left running, and EngineStuck says the server must be restarted.
    std::lock_guard<std::mutex> lk(start_mu_);
    if (!thread_.joinable()) return;
    if (in_ != nullptr) {
        if (!ended_.load()) {
            in_->push("STOP");
            in_->push("QUIT");
        }
        in_->close();
    }
    if (proc_ != nullptr) {
        close_process();
    } else {
        for (int i = 0; i < kQuitWaitS * 20 && !finished_.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (!finished_.load())
            throw EngineStuck("the engine did not stop within " + std::to_string(kQuitWaitS) + " s of QUIT: it still holds "
                              "its GPU and RAM; restart the server");
    }
    thread_.join();
    ended_ = true;
    if (pump_.joinable()) pump_.join();   // its output closed with the engine thread
    for (int i = 0; i < 100 && bg_.load() > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

void LaminaEngine::start_process(const std::vector<std::string>& argv) {
    // a replica's engine: a child of its own, the same line protocol through its stdin and stdout
    SpawnOptions o;
    o.args = argv;
    o.cwd = process_.cwd;
    o.env = process_.env;
    o.cpus = process_.cpus;
    o.in = SpawnOptions::pipe;
    o.out = SpawnOptions::pipe;
    if (log_.empty()) {
        o.err = SpawnOptions::null;
    } else {
        o.err = SpawnOptions::file;
        o.err_file = log_;
    }
    o.own_group = true;   // Ctrl+C reaches the server only, which ends the engine the clean way
    try {
        proc_ = std::make_unique<Process>(o);
    } catch (const std::exception& e) {
        finished_ = true;
        throw EngineDied(std::string("the engine could not start: ") + e.what());
    }
    std::shared_ptr<LineChannel> in = in_, out = out_;
    Process* p = proc_.get();
    reader_ = std::thread([this, p, out] {
        std::string line;
        while (p->read_line(line, -1) == 1) out->push(line);
        out->close();
        rc_ = p->wait(-1).value_or(-1);
        ended_ = true;
        finished_ = true;
    });
    thread_ = std::thread([p, in] {   // the lines to the engine, in order, from every request thread
        std::string line;
        while (in->pop(line, -1) == 1)
            if (!p->write(line + "\n")) break;
        p->close_stdin();   // QUIT, or the end of the input: the engine ends
    });
}

void LaminaEngine::close_process() {
    // QUIT's time to free its memory, then the child is ended
    for (int i = 0; i < 20 * 20 && !finished_.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (!finished_.load()) {
        proc_->kill();
        for (int i = 0; i < 20 * 20 && !finished_.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!finished_.load()) throw EngineStuck("Lamina is still releasing GPU/RAM; retry unloading after it exits");
    in_->close();
    if (reader_.joinable()) reader_.join();
    proc_.reset();
}

void LaminaEngine::unload() {
    close();
    unloaded_ = true;
}

void LaminaEngine::load() {
    if (!alive()) {
        close();
        starting_ = true;
        try {
            start();
        } catch (...) {
            starting_ = false;
            throw;
        }
        starting_ = false;
    }
}

void LaminaEngine::send(const std::string& line) {
    if (in_ == nullptr || ended_.load()) throw EngineDied("the engine stopped unexpectedly (exit code " +
                                                          std::to_string(rc_.load()) + ")");
    in_->push(line);
}

std::string LaminaEngine::death_note() {
    if (!silent_note_.empty()) return silent_note_;
    std::string wide;
    if (!log_.empty()) {
        std::ifstream f(log_, std::ios::binary);
        if (f) {
            f.seekg(0, std::ios::end);
            const std::streamoff size = f.tellg();
            f.seekg(std::max<std::streamoff>(0, size - 262144));
            wide.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
    }
    const std::vector<std::string> lines = split(wide, "\n");
    for (auto it = lines.rbegin(); it != lines.rend(); ++it)
        if (it->find("issue #29") != std::string::npos)
            return "The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                   "line: " + std::string(strip(*it)) + " - please report it at github.com/eddoursul/Lamina/issues.";
    for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
        const std::string_view s = strip(*it);
        if (starts_with(s, "lamina") || starts_with(s, "ERR"))
            return "The engine ended (code " + std::to_string(rc_.load()) + "). Its last log line: " + std::string(s) +
                   " - if that does not explain it, please report it at github.com/eddoursul/Lamina/issues with the log.";
    }
    return "The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | grep -i -E "
           "'killed process|out of memory'); Windows slows down instead. Close other programs or use a smaller model.";
}

EngineSilent LaminaEngine::silent(const std::string& what) {
    // an engine thread cannot be ended from here: it is left running, the server reports it and takes no request
    silent_note_ = what + "; the engine was given up (restart the server: an engine that lost step cannot be ended "
                          "from inside the process)";
    ended_ = true;
    return EngineSilent(silent_note_);
}

bool LaminaEngine::note_line(const std::string& line) {
    const std::vector<std::string> f = split_ws(line);
    if (f.empty()) return false;
    if (f[0] == "LP") {
        json scores = json::object();
        for (size_t i = 1; i < f.size(); ++i) {
            const size_t c = f[i].find(':');
            if (c == std::string::npos) continue;
            scores[f[i].substr(0, c)] = std::atof(f[i].c_str() + c + 1);
        }
        pending_["scores"] = scores;
        return true;
    }
    if (f[0] == "TIERS") {
        json tiers = json::object();
        for (size_t i = 1; i < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq == std::string::npos) continue;
            const std::string v = f[i].substr(eq + 1);
            if (digits(v, true)) tiers[f[i].substr(0, eq)] = std::atoll(v.c_str());   // a malformed field is left out
        }
        pending_["expert_tiers"] = tiers;
        return true;
    }
    return false;
}

void LaminaEngine::cache_line(const std::string& line) {
    json out = json::object();
    for (const auto& kv : split_ws(line)) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string val = kv.substr(eq + 1);
        if (!digits(val, true)) continue;   // a malformed field is left out
        out[kv.substr(0, eq)] = std::atoll(val.c_str());
    }
    const std::lock_guard<std::mutex> g(cache_mu_);
    cache_ = std::move(out);
}

json LaminaEngine::cache_view() {
    if (!alive()) return nullptr;   // an engine that is not running holds no cache
    const std::lock_guard<std::mutex> g(cache_mu_);
    if (cache_.empty()) return nullptr;   // no CACHE line: the cache is off
    json v = cache_;
    v["active"] = v.value("live", (int64_t) 0) > 0 ? 1 : 0;   // a conversation in the session
    return v;
}

void LaminaEngine::parse_done(const std::string& line) {
    const std::vector<std::string> f = split_ws(line);
    auto i = [&](size_t k) { return k < f.size() ? (int64_t) std::atoll(f[k].c_str()) : 0; };
    auto d = [&](size_t k) { return k < f.size() ? std::atof(f[k].c_str()) : 0.0; };
    json& fig = figures();
    fig = {{"generated", i(1)}, {"prompt_tokens", i(2)}, {"prompt_ms", d(3)}, {"decode_ms", d(4)},
           {"finish", f.size() > 5 ? f[5] : ""}};
    for (auto it = pending_.begin(); it != pending_.end(); ++it) fig[it.key()] = it.value();
    pending_ = json::object();
    count_done();
    if (f.size() >= 9) {
        fig["drafts_accepted"] = i(6);
        fig["drafts_offered"] = i(7);
        fig["reused"] = i(8);
    }
    if (f.size() >= 11) {
        fig["hits"] = i(9);
        fig["lookups"] = i(10);
    }
    if (f.size() >= 14) {
        fig["ram_blobs"] = i(11);
        fig["file_blobs"] = i(12);
        fig["file_mb"] = d(13);
    }
    if (f.size() >= 15) fig["prompt_read"] = i(14);
    if (f.size() >= 16) fig["offloaded"] = i(15);
    for (size_t k = 16; k < f.size(); ++k) {
        const size_t eq = f[k].find('=');
        if (eq == std::string::npos) continue;
        const std::string key = f[k].substr(0, eq), v = f[k].substr(eq + 1);
        if ((key == "chunk" || key == "experts_streamed") && digits(v, true)) fig[key] = std::atoll(v.c_str());
    }
}

std::string LaminaEngine::sampling_keys(const json& s) {
    std::string keys;
    auto num = [&](const char* k) -> std::optional<double> {
        const json& v = jget(s, k);
        if (v.is_number()) return v.get<double>();
        return std::nullopt;
    };
    if (auto t = num("temperature"); t && *t > 0.0) keys += " temperature=" + fr(*t);
    if (auto tp = num("top_p"); tp && *tp < 1.0) keys += " top_p=" + fr(*tp);
    if (const json& tk = jget(s, "top_k"); is_int(tk) && tk.get<int64_t>() >= 0) {
        // the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
        const int64_t k = tk.get<int64_t>();
        keys += " top_k=" + std::to_string(k >= 1 && k <= 64 ? k : 64);
    }
    if (auto mp = num("min_p"); mp && *mp > 0.0 && *mp <= 1.0) keys += " min_p=" + fr(*mp);
    const auto rp = num("repetition_penalty"), pf = num("frequency_penalty"), pp = num("presence_penalty");
    const bool rp_on = rp && *rp != 1.0, pf_on = pf && *pf != 0.0, pp_on = pp && *pp != 0.0;
    if (rp_on) keys += " penalty_repeat=" + fr(*rp);
    if (pf_on) keys += " penalty_freq=" + fr(*pf);
    if (pp_on) keys += " penalty_present=" + fr(*pp);
    if (rp_on || pf_on || pp_on) {
        // a penalty without a window counts over nothing: the engine's default is the last 64 tokens
        const json& pln = jget(s, "penalty_last_n");
        keys += " penalty_last_n=" + (is_int(pln) && pln.get<int64_t>() > 0 ? std::to_string(pln.get<int64_t>()) : "64");
    }
    if (const json& seed = jget(s, "seed"); is_int(seed) && seed.get<int64_t>() > 0)
        keys += " seed=" + std::to_string(seed.get<int64_t>());
    if (const json& score = jget(s, "score"); truthy(score)) {   // POST /v1/choose: the engine's LP line for these ids
        keys += " score=";
        for (size_t k = 0; k < score.size(); ++k) keys += (k ? ":" : "") + std::to_string(score[k].get<int64_t>());
    }
    // "lamina_checkpoint": false - a one-shot call no later request extends: no conversation checkpoint for it (#830)
    if (const json& c = jget(s, "lamina_checkpoint"); c.is_boolean() && !c.get<bool>()) keys += " ckpt=0";
    // #2228: the pictures' keys (hex, in prompt order): the engine matches a picture by them
    if (const json& ik = jget(s, "lamina_img_keys"); ik.is_array() && !ik.empty()) {
        bool ok = true;
        for (const auto& k : ik)
            ok = ok && k.is_string() && !k.get_ref<const std::string&>().empty() &&
                 k.get_ref<const std::string&>().find_first_not_of("0123456789abcdef") == std::string::npos;
        if (ok) {
            keys += " img_keys=";
            for (size_t k = 0; k < ik.size(); ++k) keys += (k ? "," : "") + ik[k].get<std::string>();
        }
    }
    // "lamina_prefix" (resolved to {"tokens": N}): the engine pins the checkpoint after the first N prompt tokens
    if (const json& prefix = jget(s, "lamina_prefix"); prefix.is_object())
        if (const json& pn = jget(prefix, "tokens"); is_int(pn) && pn.get<int64_t>() > 0)
            keys += " pin=" + std::to_string(pn.get<int64_t>());
    return keys;
}

void LaminaEngine::generate(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                            const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                            const std::function<bool()>& on_beat) {
    if (!alive()) throw EngineDied("the engine is unavailable; this request was not sent");
    held.reset();
    drained.clear();
    if (batch_) {
        generate_batched(ids, max_new, sampling, cancel, embeddings, on_token, on_beat);
        return;
    }
    std::vector<int32_t> sent;   // every token of this request, the drained ones too
    progress.reset();
    progress_ms = 0;
    reused = 0;
    prefill_tok_s_mean = -1;
    pending_ = json::object();
    // an image request takes the same sampling keys as text
    std::string head = embeddings.empty() ? "GEN " + std::to_string(max_new) + sampling_keys(sampling)
                                          : "GENI " + std::to_string(max_new) + sampling_keys(sampling) + " " + embeddings;
    std::string line = head + " ";
    for (size_t k = 0; k < ids.size(); ++k) {
        if (k) line += ',';
        line += std::to_string(ids[k]);
    }
    out_->clear();   // nothing of an earlier request is left to read
    send(line);
    bool done = false, stopped = false;
    // #481: how long the engine may stay silent: until the first line, the request's first prompt chunk at the slowest
    // prompt reading on top of silence_s; a PP line sets it to its own chunk's time
    const double silence = silence_s;
    double allow = silence > 0 ? silence + (double) std::min<int64_t>((int64_t) ids.size(), kPpChunkMax) / kPpFloorTokS : 0;
    auto heard = Clock::now(), beat = heard;
    int64_t read_to = 0;
    auto stop_engine = [&] {
        if (stopped || done) return;
        stopped = true;
        if (can_stop) {
            try {
                send("STOP");
            } catch (const EngineDied&) {
            }
        }
    };
    auto drain = [&] {
        // the consumer stopped early: read the rest of this request to its DONE (never an untimed wait)
        heard = Clock::now();
        while (true) {
            const double left = allow > 0 ? allow - since(heard) : -1;
            if (allow > 0 && left <= 0)
                throw silent("the engine did not finish the request after it was stopped (STOP) within " +
                             std::to_string((int) allow) + " s");
            std::string l;
            const int got = out_->pop(l, left < 0 ? -1.0 : left);
            if (got == 0) continue;
            if (got < 0 || starts_with(l, "ERR")) break;
            note_line(l);
            if (starts_with(l, "CACHE ")) cache_line(l);   // the card keeps the state a stopped request left
            if (starts_with(l, "T ")) {
                sent.push_back((int32_t) std::atoi(l.c_str() + 2));
                drained.push_back(sent.back());
            }
            if (starts_with(l, "DONE")) {
                parse_done(l);
                if (embeddings.empty() && jstr(last, "finish") != "nonfinite") {
                    Held h{ids.size(), ids};
                    h.ids.insert(h.ids.end(), sent.begin(), sent.end());
                    held = std::move(h);
                }
                break;
            }
            if (!can_stop) heard = Clock::now();   // an engine that runs on to max_new: each line must come in time
        }
        done = true;
    };
    while (true) {
        double wait = 10.0;
        if (allow > 0) {
            wait = std::min(wait, allow - since(heard));
            if (wait <= 0) {
                done = true;   // no STOP and no drain: nothing is listening
                throw silent("the engine said nothing for " + std::to_string((int) since(heard)) + " s during the request");
            }
        }
        std::string l;
        const int got = out_->pop(l, std::min(wait, 0.5));
        if (got == 0) {
            if (cancel.is_set()) {
                stop_engine();
                drain();
                return;
            }
            if (since(beat) >= 10.0) {
                beat = Clock::now();
                if (!on_beat()) {
                    stop_engine();
                    drain();
                    return;
                }
            }
            continue;
        }
        if (got < 0) {
            done = true;
            ended_ = true;
            throw EngineDied("the engine stopped unexpectedly (exit code " + std::to_string(rc_.load()) + ")");
        }
        for (const char* p : kFatalPrefixes)
            if (starts_with(l, p)) {
                silent_note_ = "Unrecoverable native verification failure: " + l.substr(4);
                ended_ = true;
                done = true;
                throw EngineDied(silent_note_);
            }
        heard = beat = Clock::now();   // any line is output: T, PP, INFO ...
        if (starts_with(l, "T ")) {
            allow = silence;
            sent.push_back((int32_t) std::atoi(l.c_str() + 2));
            if (cancel.is_set() || !on_token(sent.back())) {
                stop_engine();
                drain();
                return;
            }
        } else if (starts_with(l, "PP ")) {
            const std::vector<std::string> f = split_ws(l);
            if (f.size() >= 3 && digits(f[1]) && digits(f[2])) {
                progress = std::make_pair((int64_t) std::atoll(f[1].c_str()), (int64_t) std::atoll(f[2].c_str()));
                progress_ms = f.size() >= 4 ? std::atof(f[3].c_str()) : 0.0;
                prefill_tok_s_mean = f.size() >= 5 ? std::atof(f[4].c_str()) : -1;
                const double rate = prefill_tok_s_mean > 0 ? prefill_tok_s_mean : 0.0;
                const int64_t chunk = progress->first - read_to;
                read_to = progress->first;
                if (silence > 0 && rate > 0 && chunk > 0) allow = std::max(silence, kPpSlack * (double) chunk / rate);
            }
            if (cancel.is_set() || !on_beat()) {
                stop_engine();
                drain();
                return;
            }
        } else if (starts_with(l, "RESUME ")) {   // the reused tokens: the first chunk starts after them
            const std::vector<std::string> f = split_ws(l);
            if (f.size() > 1 && digits(f[1])) read_to = reused = std::atoll(f[1].c_str());
        } else if (starts_with(l, "CACHE ")) {   // the prompt cache's state: the Monitor's Conversation cache card
            cache_line(l);
        } else if (note_line(l)) {
        } else if (starts_with(l, "DONE")) {
            parse_done(l);
            done = true;
            if (jstr(last, "finish") == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
            if (embeddings.empty()) {
                Held h{ids.size(), ids};
                h.ids.insert(h.ids.end(), sent.begin(), sent.end());
                held = std::move(h);
            }
            return;
        } else if (starts_with(l, "ERR")) {
            done = true;
            throw EngineError(std::string(strip(l.substr(3))));
        }
    }
}

// ---------------------------------------------------------------------------------------------- MockEngine
MockEngine::MockEngine(const Tokenizer& tok, std::vector<std::string> scripts, int64_t max_context, double delay_s,
                       bool cycle)
    : delay_(delay_s), cycle_(cycle) {
    max_context_ = known_ctx_ = max_context;
    const std::vector<int32_t> end = tok.encode("<|im_end|>", true);
    for (const auto& s : scripts) {
        std::vector<int32_t> ids = tok.encode(s, true);
        ids.insert(ids.end(), end.begin(), end.end());
        scripts_.push_back(std::move(ids));
    }
    if (scripts_.empty()) scripts_.push_back(end);
}

void MockEngine::generate(const std::vector<int32_t>& ids, int64_t max_new, const json&, Cancel& cancel,
                          const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                          const std::function<bool()>&) {
    std::vector<int32_t> script;
    {
        std::lock_guard<std::mutex> lk(mu_);
        last_prompt = ids;
        last_embeddings = embeddings;
        script = scripts_.size() > 1 ? scripts_[cycle_ ? turns_ % scripts_.size() : std::min(turns_, scripts_.size() - 1)]
                                     : scripts_[0];
        if (scripts_.size() > 1) ++turns_;
    }
    held.reset();
    drained.clear();
    for (int64_t k = 0; k < max_new && k < (int64_t) script.size(); ++k) {
        if (cancel.is_set()) return;
        if (delay_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(delay_));
        if (!on_token(script[(size_t) k])) return;
    }
}

}  // namespace lamina
