// src/server/engine_batch.cpp - --batch: several requests at once in the engine's batch slots
// (LaminaEngine::generate_batched and its helpers).
//
// Alone (no slot busy, nobody waiting) a request takes the solo path (GEN, with MTP drafts: the fastest stream); when
// another arrives meanwhile it is STOPped and continues in a slot (BGEN with its prompt and what it generated: the
// engine reuses that prefix). Otherwise BGEN admits it into a free slot, whose own BT lines follow until BDONE. The
// control lines (prompt reading, admission) are one request's at a time, and a long prompt read gives way at a chunk
// boundary to a waiting request with a much shorter prompt (#656: BYIELD; the part read waits in its slot). A consumer
// that stops early leaves the engine in step: a solo request is STOPped and read to its DONE, an admission to its BADM,
// a slot is BSTOPped and freed at its BDONE.
#include "engine.hpp"

#include "log.hpp"
#include "vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace lamina {
namespace {

using Clock = std::chrono::steady_clock;

double wall() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

bool digits(const std::string& s) { return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos; }

const std::set<int32_t> kEosIds = {248044, 248046};   // <|endoftext|>, <|im_end|>: the engine's default --eos-ids
constexpr int32_t kTurnId = 248045;                   // <|im_start|>: the engine's turn checkpoint sits at the last one
constexpr int kYieldsMax = 2;                         // #656: how often one request's read gives way
constexpr int kSoloAgainMax = 2;                      // how often a request left alone in a slot goes back to solo
constexpr int64_t kSoloAgainMinLeft = 32;
constexpr int kBgenRefusalsMax = 6;                   // #1994: tries of other slots after the engine refused a BGEN
constexpr double kSlotParkS = 2.0;
constexpr double kOrphanSlotS = 3.0;                  // #1603: a busy flag nobody owns this long is a leak

const std::optional<double> kEngineStallS = timeout_env("LAMINA_ENGINE_STALL_S", 90.0);
const std::optional<double> kAdmitMaxWaitS = timeout_env("LAMINA_ADMIT_MAX_WAIT_S", 30.0);

int64_t admit_short_first() {
    const char* v = std::getenv("LAMINA_ADMIT_SHORT_FIRST");
    try {
        return v != nullptr && *v ? std::max<int64_t>(0, std::stoll(v)) : 0;
    } catch (const std::exception&) {
        return 0;
    }
}

/// the consumer stopped reading (a stop token, a stop string, a client that left): the request ends early
struct ConsumerLeft {};

thread_local json tl_last = json::object();
thread_local uint64_t tl_dones = 0;

std::string join_ids(const std::vector<int32_t>& v) {
    std::string out;
    for (size_t k = 0; k < v.size(); ++k) {
        if (k) out += ',';
        out += std::to_string(v[k]);
    }
    return out;
}

/// #1869: a request that decoded in a slot and went back to the solo path ends with the solo DONE, which replaces the
/// figures: the slot leg's decode time, tokens and prompt figures are kept to add to it
void carry_leg(json& carry, const json& leg) {
    auto num = [&](const char* k) { return jget(leg, k).is_number() ? leg[k].get<double>() : 0.0; };
    carry["decode_ms"] = (carry.contains("decode_ms") ? carry["decode_ms"].get<double>() : 0.0) + num("decode_ms");
    carry["generated"] = (carry.contains("generated") ? carry["generated"].get<int64_t>() : 0) + (int64_t) num("generated");
    if (!carry.contains("prompt_ms")) carry["prompt_ms"] = num("prompt_ms");
    if (!jget(leg, "prompt_read").is_null() && !carry.contains("prompt_read")) carry["prompt_read"] = leg["prompt_read"];
}

json add_carry(const json& carry, json last) {
    auto num = [&](const char* k) { return jget(last, k).is_number() ? last[k].get<double>() : 0.0; };
    last["decode_ms"] = num("decode_ms") + jget(carry, "decode_ms").get<double>();
    last["generated"] = (int64_t) num("generated") + jget(carry, "generated").get<int64_t>();
    last["prompt_ms"] = std::max(num("prompt_ms"), jget(carry, "prompt_ms").get<double>());
    if (carry.contains("prompt_read")) last["prompt_read"] = carry["prompt_read"];
    return last;
}

}  // namespace

json& LaminaEngine::figures() { return batch_ ? tl_last : last; }
void LaminaEngine::count_done() {
    if (batch_) ++tl_dones;
    else ++dones;
}
json LaminaEngine::last_figures() const { return batch_ ? tl_last : last; }
uint64_t LaminaEngine::done_count() const { return batch_ ? tl_dones : dones.load(); }

void LaminaEngine::start_batch() {
    slot_order_.clear();
    for (int b = 0; b < batch_; ++b) slot_order_.push_back(b);
    slot_q_.clear();
    for (int b = 0; b < batch_; ++b) slot_q_.push_back(std::make_shared<LineChannel>());
    slot_busy_ = std::make_shared<std::vector<bool>>((size_t) batch_, false);
    // #1603: who owns a busy flag: slot_claims_ (picked or reserved, not given back), slot_releasing_ (the background
    // BDONE wait); a busy slot in neither has leaked
    slot_claims_.clear();
    slot_releasing_ = std::make_shared<std::set<int>>();
    orphan_seen_.clear();
    slot_held_ = std::make_shared<std::vector<std::vector<int32_t>>>((size_t) batch_);
    slot_used_.assign((size_t) batch_, 0.0);
    slot_live_.assign((size_t) batch_, nullptr);
    ++gen_;   // which engine run this is (a request notes its own)
    ctl_erred_ = false;
    yielded_.reset();
    ctl_lines_ = std::make_shared<LineChannel>();
    pump_ = std::thread([this] { pump(); });
}

void LaminaEngine::pump() {
    // the engine's lines: a slot's own (BT, BDONE) to its queue, the rest to the control lines
    std::shared_ptr<LineChannel> out = out_, ctl = ctl_lines_;
    std::vector<std::shared_ptr<LineChannel>> qs = slot_q_;
    std::string line;
    while (out->pop(line, -1) == 1) {
        if (starts_with(line, "BT ") || starts_with(line, "BDONE ")) {
            const std::vector<std::string> f = split_ws(line);
            if (f.size() > 1 && digits(f[1]) && std::stoul(f[1]) < qs.size()) {
                qs[std::stoul(f[1])]->push(line);
                continue;
            }
        }
        ctl->push(line);
    }
    ctl->close();
    for (auto& q : qs) q->close();
}

json LaminaEngine::slots_view() {
    if (!batch_) return nullptr;
    const double now = wall();
    json view = json::array();
    std::lock_guard<std::mutex> lk(slot_mu_);
    for (int b = 0; b < batch_; ++b) {
        const json& r = slot_live_[(size_t) b];
        if (r.is_null()) {
            view.push_back({{"slot", b}, {"state", "idle"}, {"held_tokens", (*slot_held_)[(size_t) b].size()}});
            continue;
        }
        const json& ft = r["first_token"];
        view.push_back({{"slot", b}, {"state", r["state"]}, {"prompt_tokens", r["prompt_tokens"]},
                        {"prompt_read", r["state"] == "reading" ? r["prompt_read"] : json(nullptr)}, {"generated", r["generated"]},
                        {"elapsed_s", std::round((now - r["started"].get<double>()) * 10) / 10},
                        {"tok_s", ft.is_number() ? json(std::round(r["generated"].get<double>() / std::max(1e-6, now - ft.get<double>()) * 10) / 10)
                                                 : json(nullptr)}});
    }
    return view;
}

std::pair<int, int> LaminaEngine::admission() {
    std::lock_guard<std::mutex> lk(slot_mu_);
    return {waiting_, admitting_};
}

bool LaminaEngine::take_control(Cancel& cancel, size_t plen, std::optional<int64_t> after_epoch, bool is_new,
                                const std::function<void()>& beat) {
    // the control lines (one prompt read at a time), with heartbeats; false when cancelled. `after_epoch`: a read that
    // gave way lets the requests waiting then go first; `is_new`: LAMINA_ADMIT_SHORT_FIRST orders only these
    const int64_t short_first = admit_short_first();
    WaitEntry entry{plen, mono(), is_new};
    {
        std::lock_guard<std::mutex> lk(slot_mu_);
        ++waiting_;
        wait_lens_.push_back(&entry);
    }
    auto admit_next = [&]() -> const WaitEntry* {   // the caller holds slot_mu_
        const double now = mono();
        const WaitEntry* best = nullptr;
        auto key = [&](const WaitEntry* e) {
            const bool shortish = (int64_t) e->len <= short_first || (kAdmitMaxWaitS && now - e->since >= *kAdmitMaxWaitS);
            return std::make_pair(shortish ? 0 : 1, e->since);
        };
        for (const WaitEntry* e : wait_lens_)
            if (e->ordered && (best == nullptr || key(e) < key(best))) best = e;
        return best;
    };
    struct Leave {
        LaminaEngine& e;
        WaitEntry* entry;
        ~Leave() {
            std::lock_guard<std::mutex> lk(e.slot_mu_);
            e.waiting_ = std::max(0, e.waiting_ - 1);
            e.wait_lens_.remove(entry);
        }
    };
    double beat_at = mono();
    {
        Leave leave{*this, &entry};
        for (;;) {
            const std::optional<double> deadline = cancel.queue_deadline;
            if (deadline && mono() >= *deadline) throw QueueWaitExpired("the inference queue wait deadline expired; retry later");
            if (!alive() && !starting())
                // #1012: the engine is gone and nobody starts it again: this request was not sent, so it ends now
                throw EngineDied("the engine stopped while this request waited; it was not sent");
            bool turn;
            {
                std::lock_guard<std::mutex> lk(slot_mu_);
                turn = !after_epoch || ctl_epoch_ > *after_epoch || waiting_ <= 1;
                if (turn && is_new && short_first > 0) turn = admit_next() == &entry;
            }
            const double left = deadline ? *deadline - mono() : 0.5;
            if (turn && left > 0 && ctl_.try_lock_for(std::chrono::duration<double>(std::min(0.5, left)))) {
                bool first = true;
                if (is_new && short_first > 0) {
                    std::lock_guard<std::mutex> lk(slot_mu_);   // a shorter one came while this one waited
                    first = admit_next() == &entry;
                }
                if (!first) {
                    ctl_.unlock();
                    slot_cv_.notify_all();
                    continue;
                }
                if (alive()) {
                    if (deadline && mono() >= *deadline) {
                        ctl_.unlock();
                        throw QueueWaitExpired("the inference queue wait deadline expired; retry later");
                    }
                    break;
                }
                ctl_.unlock();   // it died (or is being restarted) while this one got the lines
                turn = false;
            }
            if (!turn) {
                std::unique_lock<std::mutex> lk(slot_mu_);
                slot_cv_.wait_for(lk, std::chrono::duration<double>(std::min(0.5, std::max(0.0, left))));
            }
            if (cancel.is_set()) return false;
            if (mono() - beat_at >= 10.0) {   // a heartbeat every 10 s, as the engine's own wait
                beat_at = mono();
                beat();
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(slot_mu_);
        ++ctl_epoch_;
    }
    slot_cv_.notify_all();
    return true;
}

void LaminaEngine::control(Cancel& cancel, const std::function<void(int32_t)>& on_token,
                           const std::function<bool()>& stop_when, const std::function<void()>& on_quiet,
                           const std::function<void()>& beat) {
    // the control lines of the request on them (GEN / BGEN) up to its DONE (solo) or BADM (an admission); `stop_when`
    // true sends STOP once; `on_quiet` whenever the engine has been quiet for half a second
    bool stopped = false;
    double beat_at = mono();
    for (;;) {
        std::string line;
        const int got = ctl_lines_->pop(line, 0.5);
        if (got == 0) {
            // a client can leave before the first PP or token: its cancel is polled as often as take_control's
            if (cancel.is_set() && !stopped) {
                send("STOP");
                stopped = true;
            }
            if (on_quiet) on_quiet();
            if (mono() - beat_at >= 10.0) {
                beat_at = mono();
                beat();
            }
            continue;
        }
        beat_at = mono();
        if (got < 0) {
            ended_ = true;
            throw EngineDied("the engine stopped unexpectedly (exit code " + std::to_string(rc_.load()) + ")");
        }
        if (starts_with(line, "T ")) {
            on_token((int32_t) std::atoi(line.c_str() + 2));
            if (!stopped && (cancel.is_set() || (stop_when && stop_when()))) {
                send("STOP");
                stopped = true;
            }
        } else if (starts_with(line, "PP ")) {
            const std::vector<std::string> f = split_ws(line);
            if (f.size() >= 3 && digits(f[1]) && digits(f[2])) {
                progress = std::make_pair((int64_t) std::atoll(f[1].c_str()), (int64_t) std::atoll(f[2].c_str()));
                progress_ms = f.size() >= 4 ? std::atof(f[3].c_str()) : 0.0;
                prefill_tok_s_mean = f.size() >= 5 ? std::atof(f[4].c_str()) : -1;
            }
            if (!stopped && cancel.is_set()) {   // #879: a Stop during the prompt read goes out at once
                send("STOP");
                stopped = true;
            }
            beat();
        } else if (starts_with(line, "DONE")) {
            parse_done(line);
            last_done_ = line;
        } else if (starts_with(line, "BADM ")) {
            const std::vector<std::string> f = split_ws(line);
            ctl_result_ = std::make_pair(std::string("badm"), f.size() >= 3 && f[2] == "1");
            return;
        } else if (starts_with(line, "YIELDED ")) {   // the read gave way (BYIELD): <slot> <tokens read>
            const std::vector<std::string> f = split_ws(line);
            if (f.size() >= 3 && digits(f[2]) && (digits(f[1]) || (f[1].size() > 1 && f[1][0] == '-' && digits(f[1].substr(1)))))
                yielded_ = std::make_pair(std::stoi(f[1]), (int64_t) std::stoll(f[2]));
        } else if (starts_with(line, "CACHE ")) {   // the prompt cache's state: the Monitor's Conversation cache card
            cache_line(line);
        } else if (starts_with(line, "ERR")) {
            ctl_erred_ = true;   // #1059: the engine refused it and is idle: no DONE follows
            throw EngineError(std::string(strip(line.substr(std::min<size_t>(4, line.size())))));
        } else {
            note_line(line);
        }
        if (starts_with(line, "DONE") && ctl_mode_ == "solo") {
            ctl_result_ = std::make_pair(std::string("done"), false);
            return;
        }
    }
}

std::string LaminaEngine::done_finish() const {
    const std::vector<std::string> f = split_ws(last_done_);
    return f.size() > 5 ? f[5] : "";
}

std::string LaminaEngine::drain_control(const std::string& until, int born, double timeout_s) {
    // after a consumer left early: the control lines up to the next `until` line, so the next request does not read
    // this one's leftovers; "" when the engine ended (or is another one: #1012) or never answered
    const double end = mono() + timeout_s;
    while (mono() < end) {
        if (gen_ != born || !alive()) return "";
        std::string line;
        const int got = ctl_lines_->pop(line, std::max(0.1, std::min(1.0, end - mono())));
        if (got == 0) continue;
        if (got < 0) return "";
        if (starts_with(line, "DONE")) parse_done(line);
        if (starts_with(line, "CACHE ")) cache_line(line);   // the card keeps the state a left-early request left
        if (starts_with(line, until) || starts_with(line, "ERR")) return line;
    }
    return "";
}

void LaminaEngine::apply_bdone(const std::vector<std::string>& f) {
    // BDONE <slot> <produced> <finish> <ms> into this request's figures
    if (f.size() < 5) return;
    json& l = figures();
    l["finish"] = f[3];
    l["decode_ms"] = std::atof(f[4].c_str());
    if (digits(f[2])) l["generated"] = std::stoll(f[2]);
}

std::string LaminaEngine::bdone_after_eos(int slot, double wait_s) {
    // #1869: a slot whose consumer left at the stop token has its BDONE right behind it: read here, so the figures are
    // the slot leg's; "" when it does not come in wait_s (the background wait takes over)
    if (slot >= (int) slot_q_.size()) return "";
    const double end = mono() + wait_s;
    while (mono() < end) {
        std::string line;
        const int got = slot_q_[(size_t) slot]->pop(line, std::max(0.05, end - mono()));
        if (got <= 0) return "";
        if (starts_with(line, "BDONE ")) return line;
    }
    return "";
}

void LaminaEngine::release_slot_when_done(int slot, std::optional<std::vector<int32_t>> stream) {
    // a slot whose consumer left: BSTOP it, and free it at its BDONE (in the background); `stream` (the prompt and its
    // tokens so far) with the ones still to come, all but the last, are what the slot holds then
    if (slot >= (int) slot_q_.size()) return;   // the engine restarted with fewer slots: this one is gone
    try {
        send("BSTOP " + std::to_string(slot));
    } catch (const EngineDied&) {
    }
    auto q = slot_q_[(size_t) slot];
    auto busy = slot_busy_;
    auto holds = slot_held_;
    auto releasing = slot_releasing_;
    {
        std::lock_guard<std::mutex> lk(slot_mu_);
        releasing->insert(slot);
    }
    ++bg_;
    std::thread([this, slot, q, busy, holds, releasing, stream] {
        const double end = mono() + 600.0;
        std::vector<int32_t> tail = stream.value_or(std::vector<int32_t>{});
        while (mono() < end) {
            std::string line;
            const int got = q->pop(line, 5.0);
            if (got == 0) continue;
            if (got < 0) {
                tail.clear();
                break;
            }
            if (starts_with(line, "BT ")) {
                const std::vector<std::string> f = split_ws(line);
                if (f.size() > 2 && digits(f[2])) tail.push_back((int32_t) std::stol(f[2]));
                else tail.clear();
            }
            if (starts_with(line, "BDONE ")) {
                const std::vector<std::string> f = split_ws(line);
                if (f.size() > 3 && f[3] == "nonfinite") tail.clear();   // #879: the engine dropped the slot's state
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lk(slot_mu_);
            (*holds)[(size_t) slot] = stream && !tail.empty() ? std::vector<int32_t>(tail.begin(), tail.end() - 1) : std::vector<int32_t>{};
            (*busy)[(size_t) slot] = false;
            releasing->erase(slot);
        }
        slot_cv_.notify_all();
        --bg_;
    }).detach();
}

void LaminaEngine::park_slot(int slot) {
    // #1994: the engine refused a BGEN for this slot: busy here (owned by nobody) for SLOT_PARK_S, then given back
    auto busy = slot_busy_;
    auto releasing = slot_releasing_;
    {
        std::lock_guard<std::mutex> lk(slot_mu_);
        slot_claims_.erase(slot);
        (*busy)[(size_t) slot] = true;
    }
    ++bg_;
    std::thread([this, slot, busy, releasing] {
        std::this_thread::sleep_for(std::chrono::duration<double>(kSlotParkS));
        {
            std::lock_guard<std::mutex> lk(slot_mu_);
            if (!slot_claims_.count(slot) && !releasing->count(slot)) (*busy)[(size_t) slot] = false;
        }
        slot_cv_.notify_all();
        --bg_;
    }).detach();
}

std::vector<int> LaminaEngine::reap_orphan_slots() {
    // #1603 (the caller holds slot_mu_): a busy flag nobody owns for ORPHAN_SLOT_S is given back
    const double now = mono();
    std::vector<int> freed;
    for (int b = 0; b < batch_; ++b) {
        if ((*slot_busy_)[(size_t) b] && !slot_claims_.count(b) && !slot_releasing_->count(b)) {
            const double first = orphan_seen_.emplace(b, now).first->second;
            if (now - first >= kOrphanSlotS) {
                (*slot_busy_)[(size_t) b] = false;
                orphan_seen_.erase(b);
                freed.push_back(b);
            }
        } else {
            orphan_seen_.erase(b);
        }
    }
    if (!freed.empty()) {
        std::string list;
        for (const int b : freed) list += (list.empty() ? "" : ", ") + std::to_string(b);
        say("[lamina] parallel: slot(s) " + list + " were marked busy with no request using them (issue #1603); freed, the "
            "waiting request goes on");
        slot_cv_.notify_all();
    }
    return freed;
}

std::optional<int> LaminaEngine::pick_slot(const std::vector<int32_t>& prompt) {
    // a free slot (the caller holds slot_mu_): the one whose held tokens are the longest start of the prompt, else an
    // empty one, else the one used longest ago - the conversations other slots hold stay for their next turns
    std::vector<int> free;
    for (const int b : slot_order_)
        if (!(*slot_busy_)[(size_t) b]) free.push_back(b);
    if (free.empty()) return std::nullopt;
    auto held_prefix = [&](int b) -> size_t {
        // what of the prompt the slot's cache serves: its whole held stream when this prompt continues it, else its turn
        // checkpoint (the last <|im_start|>: the chat's next turn agrees with it up to there, #2518)
        const std::vector<int32_t>& h = (*slot_held_)[(size_t) b];
        if (h.empty()) return 0;
        if (h.size() < prompt.size() && std::equal(h.begin(), h.end(), prompt.begin())) return h.size();
        const auto rit = std::find(h.rbegin(), h.rend(), kTurnId);
        const size_t k = rit == h.rend() ? 0 : (size_t) (h.rend() - rit) - 1;
        return k > 0 && k < prompt.size() && std::equal(h.begin(), h.begin() + (ptrdiff_t) k, prompt.begin()) ? k : 0;
    };
    int best = free[0];
    for (const int b : free)
        if (held_prefix(b) > held_prefix(best)) best = b;
    if (held_prefix(best) >= 1) return best;
    auto key = [&](int b) { return std::make_pair(!(*slot_held_)[(size_t) b].empty(), slot_used_[(size_t) b]); };   // empty, then least recent
    return *std::min_element(free.begin(), free.end(), [&](int a, int b) { return key(a) < key(b); });
}

bool LaminaEngine::may_go_solo(int64_t left, int times, const std::string& embeddings) {
    // a request decoding in a slot that is alone now goes back to the solo path with its MTP drafts (the engine continues
    // it from the slot's sessions: INFO slot_cache=1); LAMINA_PARALLEL_SOLO=0 keeps it in the slot
    const char* off = std::getenv("LAMINA_PARALLEL_SOLO");
    if (!embeddings.empty() || times >= kSoloAgainMax || left < kSoloAgainMinLeft || !truthy(jget(info, "slot_cache")) ||
        (off != nullptr && std::string(off) == "0"))
        return false;
    std::lock_guard<std::mutex> lk(slot_mu_);
    return std::count(slot_busy_->begin(), slot_busy_->end(), true) == 1 && waiting_ == 0;
}

bool LaminaEngine::shorter_waiting(size_t plen) {
    // #656: someone waits for the control lines with a prompt under half this one's: worth giving way to
    std::lock_guard<std::mutex> lk(slot_mu_);
    for (const WaitEntry* e : wait_lens_)
        if (e->len * 2 <= plen) return true;
    return false;
}

void LaminaEngine::generate_batched(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                                    const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                                    const std::function<bool()>& on_beat) {
    if (!alive()) throw EngineDied("the engine is unavailable; this request was not sent");
    progress.reset();
    progress_ms = 0;
    reused = 0;
    tl_last = json::object();
    int born = gen_;
    const std::string keys = sampling_keys(sampling);
    std::vector<int32_t> out, prompt = ids;
    int64_t left = max_new;
    json reused0;   // the first read's reused tokens (a continued leg's count includes this request's own output)
    json carry = json::object();   // #1869: the legs that ended in a hand-back to the solo path
    auto beat = [&] {
        if (!on_beat()) throw ConsumerLeft();
    };
    auto emit = [&](int32_t t) {
        out.push_back(t);
        if (!on_token(t)) throw ConsumerLeft();
    };
    try {
        if (!take_control(cancel, prompt.size(), std::nullopt, true, beat)) return;
    } catch (const ConsumerLeft&) {
        return;
    }
    bool holding = true;
    born = gen_;
    std::optional<int> slot, reserved;
    std::optional<size_t> gen0;
    std::string phase = "none";   // solo -> (admit -> slot); "none" once the engine has finished with this request
    bool stop_sent = false, admit_wait = false, admit_said = false, cont = false;
    int yields = 0, solo_again = 0, refused = 0;
    double admit_since = 0;
    auto mark_sent = [&] {
        cancel.queue_deadline.reset();
        if (cancel.queue_started) cancel.queue_started();
    };
    auto check_deadline = [&] {
        if (cancel.queue_deadline && mono() >= *cancel.queue_deadline)
            throw QueueWaitExpired("the inference queue wait deadline expired; retry later");
    };
    auto free_slot = [&](int b) {
        {
            std::lock_guard<std::mutex> lk(slot_mu_);
            (*slot_busy_)[(size_t) b] = false;
            slot_claims_.erase(b);
        }
        slot_cv_.notify_all();
    };
    auto cleanup = [&] {
        // a consumer that left early (or an error): the engine and this server stay in step
        try {
            if ((phase == "solo" || phase == "admit") && (gen_ != born || !alive())) phase = "none";   // #1012: gone
            if ((phase == "solo" || phase == "admit") && ctl_erred_) phase = "none";   // #1059: no DONE follows an ERR
            if (phase == "solo") {
                send("STOP");
                drain_control("DONE", born);
            } else if (phase == "admit") {
                const std::string line = drain_control("BADM", born);
                const std::vector<std::string> f = split_ws(line);
                if (starts_with(line, "BADM ") && f.size() > 2 && f[2] == "1") phase = "slot";
            }
        } catch (const EngineDied&) {
        }
        bool bdone_in = false;
        if (phase == "slot" && slot && !out.empty() && kEosIds.count(out.back()) && gen_ == born) {
            const std::string bd = bdone_after_eos(*slot);   // (#1869) the slot's figures, not the admission's
            if (!bd.empty()) {
                apply_bdone(split_ws(bd));
                bdone_in = true;
            }
        }
        if (!reused0.is_null()) tl_last["reused"] = reused0;
        if (!carry.empty()) tl_last = add_carry(carry, tl_last);
        if (holding) {
            ctl_.unlock();
            holding = false;
        }
        if (admit_wait) {
            std::lock_guard<std::mutex> lk(slot_mu_);
            admitting_ = std::max(0, admitting_ - 1);
        }
        if (reserved) free_slot(*reserved);
        if (slot) {
            std::optional<std::vector<int32_t>> stream;
            if (gen0 && out.size() > *gen0) {
                std::vector<int32_t> s = prompt;
                s.insert(s.end(), out.begin() + (ptrdiff_t) *gen0, out.end());
                stream = std::move(s);
            }
            {
                std::lock_guard<std::mutex> lk(slot_mu_);
                slot_live_[(size_t) *slot] = nullptr;
                slot_used_[(size_t) *slot] = wall();
            }
            if (phase == "slot" && bdone_in) {
                {
                    std::lock_guard<std::mutex> lk(slot_mu_);   // as the BDONE wait leaves it
                    (*slot_held_)[(size_t) *slot] = stream && !stream->empty() ? std::vector<int32_t>(stream->begin(), stream->end() - 1)
                                                                               : std::vector<int32_t>{};
                }
                free_slot(*slot);
            } else if (phase == "slot") {
                {
                    std::lock_guard<std::mutex> lk(slot_mu_);
                    (*slot_held_)[(size_t) *slot].clear();
                    slot_claims_.erase(*slot);   // the BDONE wait owns it now
                }
                release_slot_when_done(*slot, stream);
            } else {
                free_slot(*slot);
            }
        }
    };
    auto body = [&] {
        for (;;) {   // a request in a slot that is left alone goes back to the solo path
            if (!holding) {
                if (!take_control(cancel, prompt.size(), std::nullopt, false, beat)) return;
                holding = true;
                born = gen_;
            }
            bool alone;
            {
                std::lock_guard<std::mutex> lk(slot_mu_);
                alone = std::none_of(slot_busy_->begin(), slot_busy_->end(), [](bool b) { return b; }) && waiting_ == 0;
            }
            if (alone && left > 1) {
                const std::string head = embeddings.empty() ? "GEN " + std::to_string(left) + keys
                                                            : "GENI " + std::to_string(left) + keys + " " + embeddings;
                last_done_.clear();   // (#879: this request's DONE decides below)
                check_deadline();
                send(head + " " + join_ids(prompt));
                mark_sent();
                phase = "solo";
                ctl_mode_ = "solo";
                ctl_result_.reset();
                yielded_.reset();
                ctl_erred_ = false;
                auto others = [&] {
                    std::lock_guard<std::mutex> lk(slot_mu_);
                    return waiting_ > 0;
                };
                auto give_way = [&] {   // at a PP line, and between two far apart (taken at the engine's next boundary)
                    if (reserved || !out.empty() || !embeddings.empty() || yields >= kYieldsMax || !shorter_waiting(prompt.size()))
                        return;
                    {
                        std::lock_guard<std::mutex> lk(slot_mu_);   // the slot the part read will wait in
                        reserved = pick_slot(prompt);
                        if (reserved) {
                            (*slot_busy_)[(size_t) *reserved] = true;
                            slot_claims_.insert(*reserved);
                        }
                    }
                    if (reserved) send("BYIELD " + std::to_string(*reserved));
                };
                control(cancel, emit, others, give_way, [&] {
                    give_way();
                    beat();
                });
                phase = "none";   // its DONE is read
                if (done_finish() == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
                if (yielded_ && reserved && yielded_->first == *reserved) {
                    slot = reserved;   // gave way: the read goes on in that slot (below)
                    reserved.reset();
                } else if (reserved) {   // it did not give way: the slot is free again
                    free_slot(*reserved);
                    reserved.reset();
                }
                if (!slot) {
                    const std::string finish = jstr(tl_last, "finish");
                    left = max_new - (int64_t) out.size();
                    if (cancel.is_set() || left <= 0 || finish == "stop" || finish == "length" ||
                        (!out.empty() && kEosIds.count(out.back())))
                        return;
                    if (reused0.is_null()) reused0 = jget(tl_last, "reused");
                    carry_leg(carry, tl_last);   // #1869: the slot's admission and BDONE replace these figures
                    prompt = ids;                // promoted: it continues in a batch slot from here
                    prompt.insert(prompt.end(), out.begin(), out.end());
                }
            }
            json* live = nullptr;
            for (;;) {
                if (!slot) {
                    // a free slot (they free themselves at BDONE, which needs no control lines): the one that holds this
                    // prompt's start, else the one used longest ago. With none free the control lines are given back
                    // while waiting: a slot can be held by a read that gave way and needs them to go on
                    for (;;) {
                        check_deadline();
                        {
                            std::lock_guard<std::mutex> lk(slot_mu_);
                            slot = pick_slot(prompt);
                            if (slot) {
                                (*slot_busy_)[(size_t) *slot] = true;
                                slot_claims_.insert(*slot);
                                if (admit_wait) {
                                    admit_wait = false;
                                    admitting_ = std::max(0, admitting_ - 1);
                                }
                                break;
                            }
                            if (!admit_wait) {   // #1603: counted while it waits for a slot (it holds no control lines)
                                admit_wait = true;
                                admit_since = mono();
                                ++admitting_;
                            }
                        }
                        if (holding) {
                            ctl_.unlock();
                            holding = false;
                        }
                        {
                            std::unique_lock<std::mutex> lk(slot_mu_);
                            if (!pick_slot(prompt)) {
                                if (kEngineStallS && mono() - admit_since >= *kEngineStallS && !admit_said) {
                                    admit_said = true;
                                    std::string busy, claims, ending;
                                    for (int b = 0; b < batch_; ++b)
                                        if ((*slot_busy_)[(size_t) b]) busy += (busy.empty() ? "" : ", ") + std::to_string(b);
                                    for (const int b : slot_claims_) claims += (claims.empty() ? "" : ", ") + std::to_string(b);
                                    for (const int b : *slot_releasing_) ending += (ending.empty() ? "" : ", ") + std::to_string(b);
                                    char buf[64];
                                    std::snprintf(buf, sizeof buf, "%.0f", mono() - admit_since);
                                    say(std::string("[lamina] parallel: a request has waited ") + buf + " s for a free slot (busy: [" +
                                        busy + "], in use: [" + claims + "], ending: [" + ending + "]) (issue #1603)");
                                }
                                reap_orphan_slots();
                            }
                            if (!pick_slot(prompt)) {
                                const double remaining = cancel.queue_deadline ? *cancel.queue_deadline - mono() : 1.0;
                                slot_cv_.wait_for(lk, std::chrono::duration<double>(std::min(1.0, std::max(0.0, remaining))));
                            }
                        }
                        if (cancel.is_set()) return;
                        if (!holding) {
                            if (!take_control(cancel, prompt.size(), std::nullopt, false, beat)) return;
                            holding = true;
                            born = gen_;
                        }
                    }
                }
                if (yielded_) {   // it gave way: the others waiting then go first
                    {
                        std::lock_guard<std::mutex> lk(slot_mu_);
                        (*slot_held_)[(size_t) *slot] = std::vector<int32_t>(prompt.begin(), prompt.begin() + (ptrdiff_t) std::min<int64_t>(yielded_->second, (int64_t) prompt.size()));
                    }
                    yielded_.reset();
                    ++yields;
                    ctl_.unlock();
                    holding = false;
                    int64_t epoch;
                    {
                        std::lock_guard<std::mutex> lk(slot_mu_);
                        epoch = ctl_epoch_;
                    }
                    if (!take_control(cancel, prompt.size(), epoch, false, beat)) return;
                    holding = true;
                    born = gen_;
                }
                slot_q_[(size_t) *slot]->clear();
                const std::string head = embeddings.empty()
                                             ? "BGEN " + std::to_string(*slot) + " " + std::to_string(left) + keys
                                             : "BGENI " + std::to_string(*slot) + " " + std::to_string(left) + keys + " " + embeddings;
                check_deadline();
                {
                    std::lock_guard<std::mutex> lk(slot_mu_);
                    slot_live_[(size_t) *slot] = {{"slot", *slot}, {"request_id", cancel.request_id}, {"state", "reading"},
                                                  {"prompt_tokens", prompt.size()}, {"prompt_read", 0}, {"generated", 0},
                                                  {"started", wall()}, {"first_token", nullptr}};
                    live = &slot_live_[(size_t) *slot];
                }
                last_done_.clear();   // (#879: this admission's DONE decides below)
                send(head + " " + join_ids(prompt));
                mark_sent();
                {
                    std::lock_guard<std::mutex> lk(slot_mu_);
                    (*slot_held_)[(size_t) *slot].clear();   // the admission overwrites what the slot held
                }
                phase = "admit";
                ctl_mode_ = "batch";
                ctl_result_.reset();
                yielded_.reset();
                ctl_erred_ = false;
                bool asked = false;
                auto ask = [&] {   // as give_way above, for the slot this admission reads into
                    if (!asked && embeddings.empty() && yields < kYieldsMax && shorter_waiting(prompt.size())) {
                        send("BYIELD " + std::to_string(*slot));
                        asked = true;
                    }
                };
                auto note_progress = [&] {   // the live prompt counter of the admission
                    std::lock_guard<std::mutex> lk(slot_mu_);
                    if (progress && live != nullptr && (*live)["state"] == "reading") (*live)["prompt_read"] = progress->first;
                };
                try {
                    control(cancel, emit, nullptr, [&] {
                        ask();
                        note_progress();
                    }, [&] {
                        ask();
                        note_progress();
                        beat();
                    });
                } catch (const EngineError& e) {
                    // #1994: the engine says the slot is not free (its view and ours differ for a moment): nothing was
                    // read, so the slot is parked and another one tried - never a 400 for the conversation
                    if (!(ctl_erred_ && starts_with(e.what(), "BGEN:") && left >= 1 && !cancel.is_set())) throw;
                    ++refused;
                    say("[lamina] parallel: the engine refused slot " + std::to_string(*slot) + " (" + e.what() +
                        "); trying another slot (" + std::to_string(refused) + "/" + std::to_string(kBgenRefusalsMax) +
                        ") (issue #1994)");
                    if (refused > kBgenRefusalsMax)
                        throw GpuBusy("the engine's batch slots are not free right now; try again in a moment", 5);
                    {
                        std::lock_guard<std::mutex> lk(slot_mu_);
                        slot_live_[(size_t) *slot] = nullptr;
                        (*slot_held_)[(size_t) *slot].clear();
                    }
                    live = nullptr;
                    phase = "none";
                    ctl_erred_ = false;
                    park_slot(*slot);
                    slot.reset();
                    continue;
                }
                cont = ctl_result_ && ctl_result_->second;
                phase = cont ? "slot" : "none";
                if (!cont && done_finish() == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
                if (!cont && yielded_ && yielded_->first == *slot && !cancel.is_set()) continue;   // gave way: again later
                break;
            }
            ctl_.unlock();
            holding = false;
            if (!cont) return;
            {
                std::lock_guard<std::mutex> lk(slot_mu_);
                if (live != nullptr) {
                    (*live)["state"] = "decoding";
                    (*live)["first_token"] = wall();
                    (*live)["generated"] = out.size();
                }
            }
            gen0 = out.empty() ? 0 : out.size() - 1;   // the admission's own token: the slot feeds it first
            bool going_solo = false;
            for (;;) {
                std::string line;
                const int got = slot_q_[(size_t) *slot]->pop(line, 10.0);
                if (got == 0) {
                    if (cancel.is_set() && !stop_sent) {
                        send("BSTOP " + std::to_string(*slot));
                        stop_sent = true;
                    }
                    beat();
                    continue;
                }
                if (got < 0) {
                    phase = "none";
                    ended_ = true;
                    throw EngineDied("the engine stopped unexpectedly (exit code " + std::to_string(rc_.load()) + ")");
                }
                if (starts_with(line, "BT ")) {
                    const std::vector<std::string> f = split_ws(line);
                    const int32_t t = f.size() > 2 ? (int32_t) std::atoi(f[2].c_str()) : 0;
                    {
                        std::lock_guard<std::mutex> lk(slot_mu_);
                        if (live != nullptr) (*live)["generated"] = out.size() + 1;
                    }
                    if (cancel.is_set()) {
                        out.push_back(t);
                        if (!stop_sent) {
                            send("BSTOP " + std::to_string(*slot));
                            stop_sent = true;
                        }
                        continue;
                    }
                    emit(t);
                    if (!going_solo && !stop_sent && may_go_solo(max_new - (int64_t) out.size(), solo_again, embeddings)) {
                        send("BSTOP " + std::to_string(*slot));   // alone now: on with the drafts (below)
                        going_solo = true;
                    }
                } else if (starts_with(line, "BDONE ")) {
                    phase = "none";
                    const std::vector<std::string> f = split_ws(line);
                    apply_bdone(f);
                    {
                        // what the slot's sessions hold now: the prompt and every token fed (all but the last one)
                        std::lock_guard<std::mutex> lk(slot_mu_);
                        std::vector<int32_t> h;
                        if (out.size() > *gen0) {
                            h = prompt;
                            h.insert(h.end(), out.begin() + (ptrdiff_t) *gen0, out.end() - 1);
                        }
                        (*slot_held_)[(size_t) *slot] = f.size() > 3 && f[3] == "nonfinite" ? std::vector<int32_t>{} : h;
                    }
                    if (f.size() > 3 && f[3] == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
                    if (going_solo && f.size() > 3 && f[3] == "cancel" && !cancel.is_set() && (int64_t) out.size() < max_new &&
                        !(!out.empty() && kEosIds.count(out.back()))) {
                        // the solo path continues it: the engine copies the slot's sessions back and decodes with drafts
                        carry_leg(carry, tl_last);   // #1869: the solo DONE replaces these figures
                        {
                            std::lock_guard<std::mutex> lk(slot_mu_);
                            slot_live_[(size_t) *slot] = nullptr;
                            slot_used_[(size_t) *slot] = wall();
                        }
                        live = nullptr;
                        free_slot(*slot);
                        slot.reset();
                        gen0.reset();
                        if (reused0.is_null()) reused0 = jget(tl_last, "reused");
                        prompt = ids;
                        prompt.insert(prompt.end(), out.begin(), out.end());
                        left = max_new - (int64_t) out.size();
                        ++solo_again;
                        break;
                    }
                    return;
                }
            }
        }
    };
    try {
        body();
    } catch (const ConsumerLeft&) {
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    held.reset();
}

}  // namespace lamina
