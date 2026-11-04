// src/server/engine.hpp - the engine as the server sees it (LaminaEngine and MockEngine).
//
// LaminaEngine runs the engine (lamina::program::run) on a thread of this process and speaks its --serve line
// protocol through in-process channels: GEN / GENI requests (sampling keys between max_new and the ids), STOP, QUIT;
// T tokens, PP prompt progress, LP / TIERS notes, DONE figures, ERR, READY and INFO. A request runs through generate(),
// whose callbacks take each token and the heartbeats; a callback that returns false ends the request (STOP, then
// the rest of it is read to its DONE).
#pragma once

#include "json.hpp"
#include "process.hpp"
#include "sync.hpp"
#include "tokenizer.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lamina {

/// a client left while its request waited for the engine
struct QueueCancelled : std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// a request waited longer than the queue's deadline for its first turn
struct QueueWaitExpired : std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// the client hung up while its request was prepared
struct ClientGone : std::runtime_error {
    ClientGone() : std::runtime_error("the client left") {}
};
/// the GPU is in use by another program: a 503 with Retry-After
struct GpuBusy : std::runtime_error {
    GpuBusy(const std::string& m, int retry) : std::runtime_error(m), retry_after(retry) {}
    int retry_after;
};

/// the engine did not end when it was asked to (an unload or a restart): it keeps its GPU and RAM
struct EngineStuck : std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// the engine ended in the middle of a request, or is unavailable
struct EngineDied : std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// the engine said nothing for too long, or did not finish after a STOP: it was ended
struct EngineSilent : EngineDied {
    using EngineDied::EngineDied;
};
/// the engine is (re)starting and has not said READY yet: a 503
struct EngineStarting : std::runtime_error {
    using std::runtime_error::runtime_error;
};
/// the engine's logits became non-finite (#879)
struct EngineNonFinite : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
/// the engine refused the request (its ERR line): a 400 with its words
struct EngineError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/// what the engine held after the last text request: its prompt's length, and the prompt with every token it sent
struct Held {
    size_t prompt_len = 0;
    std::vector<int32_t> ids;
};

class Engine {
public:
    virtual ~Engine() = default;
    /// tokens one at a time through `on_token`; `on_beat` while the engine is quiet (every 10 s, and on each PP line);
    /// either returning false ends the request early. `embeddings`: GENI's file (empty: GEN).
    virtual void generate(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                          const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                          const std::function<bool()>& on_beat) = 0;
    virtual bool alive() { return true; }
    /// it was unloaded (and loads again on use)
    virtual bool unloaded() const { return false; }
    /// it starts again by itself after an unload or a death (the scripted engine does not)
    virtual bool auto_load() const { return false; }
    virtual void unload() {}
    virtual void load() {}
    virtual void close() {}
    virtual int64_t max_context() const { return max_context_; }
    virtual int64_t known_ctx() const { return known_ctx_; }
    virtual bool starting() const { return starting_.load(); }
    /// the requests that run at once (the engine's batch slots; 0: one at a time, behind the service's turn)
    virtual int batch() const { return batch_; }
    /// the engine's own words on why it ended
    virtual std::string death_note() { return ""; }
    /// the tokens this thread's request got from the engine after a STOP (the engine holds them)
    virtual std::vector<int32_t> drained_tokens() const { return drained; }

    /// the figures of this thread's request's last DONE (with batch slots each request has its own)
    virtual json last_figures() const { return last; }
    /// DONE lines this thread's request read so far (a request's own came when this moved)
    virtual uint64_t done_count() const { return dones.load(); }
    /// --batch: each slot - idle (the tokens it holds), reading or decoding (null without batch slots)
    virtual json slots_view() { return nullptr; }
    /// --batch: requests waiting for the control lines, and those that gave them back to wait for a slot
    virtual std::pair<int, int> admission() { return {0, 0}; }
    /// replicas: each one's state (null without them)
    virtual json replica_view() { return nullptr; }
    /// the Monitor's Conversation cache card: the running engine's last CACHE line and `active` (null: the cache is
    /// off, or no engine runs)
    virtual json cache_view() { return nullptr; }

    // what the last request left (read by its own thread after generate returns)
    json last = json::object();            ///< the DONE line's figures (and its LP / TIERS notes)
    std::atomic<uint64_t> dones{0};
    json info = json::object();            ///< INFO key=value facts (the Monitor)
    std::optional<Held> held;              ///< after a text request that ended with DONE
    std::vector<int32_t> drained;          ///< the tokens the engine sent after a STOP (it holds them)
    /// the running request's prompt progress, from its PP and RESUME lines: its thread writes it while HTTP threads
    /// read it (/metrics, the keep-alives' prompt_progress), so a copy under a lock
    struct Progress {
        std::optional<std::pair<int64_t, int64_t>> read;   ///< (read, total) prompt tokens
        double ms = 0;                                    ///< the prompt's time so far
        double tok_s_mean = -1;                           ///< the engine's mean prompt rate (-1: none)
        int64_t reused = 0;                               ///< the prompt tokens it continues from
    };
    Progress progress() const {
        std::lock_guard<std::mutex> lk(progress_mu_);
        return progress_;
    }
    template <class F>
    void update_progress(F&& f) {
        std::lock_guard<std::mutex> lk(progress_mu_);
        f(progress_);
    }
    double silence_s = 300.0;              ///< the config's engine_silence_s (0: wait for ever)
    std::string model_path;
    bool can_stop = true;

protected:
    int64_t max_context_ = 0, known_ctx_ = 0;
    std::atomic<bool> starting_{false};
    int batch_ = 0;

private:
    mutable std::mutex progress_mu_;
    Progress progress_;
};

/// how a LaminaEngine runs: on a thread of this process (the default), or as a `lamina --serve` child of its own
/// (a replica: its own cards, environment and CPU cores)
struct EngineProcess {
    std::string exe;                   ///< "": in this process
    std::string cwd;
    std::optional<Environment> env;
    std::vector<int> cpus;             ///< Linux: the cores it may run on (empty: any)
};

class LaminaEngine : public Engine {
public:
    /// the engine's command line (without --serve), its log file (empty: none); starts it unless `lazy`
    LaminaEngine(std::vector<std::string> args, std::string log, bool lazy = false, EngineProcess process = {});
    ~LaminaEngine() override;
    void generate(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                  const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                  const std::function<bool()>& on_beat) override;
    bool alive() override;
    void unload() override;
    void load() override;
    void close() override;
    bool unloaded() const override { return unloaded_; }
    bool auto_load() const override { return true; }
    /// the engine's own words on why it ended (its log's tail)
    std::string death_note() override;
    static std::string sampling_keys(const json& sampling);
    /// the engine's start: READY within this many seconds (0: no limit)
    static double ready_timeout_s;
    json last_figures() const override;
    uint64_t done_count() const override;
    json slots_view() override;
    std::pair<int, int> admission() override;
    json cache_view() override;

private:
    void start();
    void send(const std::string& line);
    bool note_line(const std::string& line);
    /// a CACHE line: the prompt cache's state at READY and after each request (the Conversation cache card)
    void cache_line(const std::string& line);
    void parse_done(const std::string& line);
    EngineSilent silent(const std::string& what);
    json& figures();   ///< where parse_done writes: `last`, or this thread's with batch slots
    void count_done();

    // ---- --batch (engine_batch.cpp): several requests at once in the engine's batch slots
    void generate_batched(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                          const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                          const std::function<bool()>& on_beat);
    void start_batch();   ///< after READY: the slots' state and the line pump
    void pump();
    bool take_control(Cancel& cancel, size_t plen, std::optional<int64_t> after_epoch, bool is_new,
                      const std::function<void()>& beat);
    void control(Cancel& cancel, const std::function<void(int32_t)>& on_token, const std::function<bool()>& stop_when,
                 const std::function<void()>& on_quiet, const std::function<void()>& beat);
    std::string drain_control(const std::string& until, int born, double timeout_s = 300.0);
    void apply_bdone(const std::vector<std::string>& f);
    std::string bdone_after_eos(int slot, double wait_s = 2.0);
    void release_slot_when_done(int slot, std::optional<std::vector<int32_t>> stream);
    void park_slot(int slot);
    std::vector<int> reap_orphan_slots();
    std::optional<int> pick_slot(const std::vector<int32_t>& prompt);
    bool may_go_solo(int64_t left, int times, const std::string& embeddings);
    bool shorter_waiting(size_t plen);
    std::string done_finish() const;

    struct WaitEntry {
        size_t len;
        double since;
        bool ordered;
    };
    int gen_ = 0;
    std::vector<int> slot_order_;
    std::vector<std::shared_ptr<LineChannel>> slot_q_;
    std::shared_ptr<std::vector<bool>> slot_busy_;
    std::set<int> slot_claims_;
    std::shared_ptr<std::set<int>> slot_releasing_;
    std::map<int, double> orphan_seen_;
    std::shared_ptr<std::vector<std::vector<int32_t>>> slot_held_;
    std::vector<double> slot_used_;
    std::vector<json> slot_live_;          ///< /metrics: the request in each slot (null: none)
    // the admission state is the server's, kept across restarts (#1012: waiters hold it)
    std::mutex slot_mu_;
    std::condition_variable slot_cv_;
    int waiting_ = 0, admitting_ = 0;
    std::list<WaitEntry*> wait_lens_;
    int64_t ctl_epoch_ = 0;
    std::timed_mutex ctl_;                 ///< one admission or solo request on the control lines at a time
    std::shared_ptr<LineChannel> ctl_lines_;
    std::thread pump_;
    std::atomic<int> bg_{0};               ///< the slots' background waits still running (close() waits for them)
    bool ctl_erred_ = false;
    std::optional<std::pair<int, int64_t>> yielded_;
    std::string ctl_mode_, last_done_;
    std::optional<std::pair<std::string, bool>> ctl_result_;

    void start_process(const std::vector<std::string>& argv);
    void close_process();

    std::vector<std::string> args_;
    std::string log_;
    EngineProcess process_;
    std::unique_ptr<Process> proc_;          ///< the child (EngineProcess::exe)
    std::thread reader_;                     ///< the child's output to out_
    std::thread thread_;
    std::shared_ptr<LineChannel> in_, out_;   // to the engine, from it
    std::atomic<bool> ended_{true};
    std::atomic<bool> finished_{true};        ///< the engine thread has returned
    std::atomic<int> rc_{-1};
    bool unloaded_ = true;
    json pending_ = json::object();           ///< the request's LP / TIERS notes until its DONE
    std::string silent_note_;
    std::mutex cache_mu_;
    json cache_ = json::object();             ///< the last CACHE line (the Monitor's Conversation cache card)
    std::mutex start_mu_;
};

/// replays scripted completions as token ids, one per step, then the end-of-turn token (the tests' engine); with
/// several scripts each request gets the next and the last repeats (or `cycle`: by position)
class MockEngine : public Engine {
public:
    MockEngine(const Tokenizer& tok, std::vector<std::string> scripts, int64_t max_context = 32768, double delay_s = 0,
               bool cycle = false);
    void generate(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                  const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                  const std::function<bool()>& on_beat) override;
    std::vector<int32_t> last_prompt;
    std::string last_embeddings;

private:
    std::vector<std::vector<int32_t>> scripts_;
    double delay_;
    bool cycle_;
    size_t turns_ = 0;
    std::mutex mu_;
};

}  // namespace lamina
