// src/server/replicas.hpp - data-parallel replicas: N engines, each a `lamina --serve` child on its own group of cards,
// behind one server (serve/replicas.py).
//
// Opt-in, in the run config: "replicas": 2 cuts the "gpu" list into 2 equal groups ([0,1] and [2,3]);
// "replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}] names them (an entry may add "env" and "args"). Each request goes
// to the replica with the lowest cost: the prompt tokens it would read again (it holds the start of a conversation)
// plus LOAD_TOKENS for every request already running there, so a conversation's next turn goes back to its replica
// until that one is much busier. On Linux the replicas get disjoint physical cores ("replica_cpus": false: off).
#pragma once

#include "engine.hpp"
#include "json.hpp"

#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lamina {

/// a "replicas" entry that cannot be used: the server does not start
struct ReplicaConfigError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/// the cards of each replica; empty when there are none or one
std::vector<std::vector<int>> gpu_groups(const json& cfg);
/// replica i's config: its own cards, log and learned-profile file, the entry's "env" and "args" added
json replica_cfg(const json& cfg, size_t i, const std::vector<int>& gpus);
/// n disjoint sets of CPUs (whole physical cores, SMT siblings together); nullopt off Linux, or when the topology
/// cannot be read (the engines then share the cores)
std::optional<std::vector<std::vector<int>>> cpu_sets(size_t n, bool enabled);

/// which replicas hold the start of a prompt: chained hashes of its 64-token blocks, each with a bit mask of replicas
class Router {
public:
    explicit Router(size_t n) : n_(n) {}
    /// per replica: how many leading tokens of `ids` it holds (a multiple of the block)
    std::vector<int64_t> matched(const std::vector<int32_t>& ids);
    void note(size_t r, const std::vector<int32_t>& ids);
    /// replica r lost its cache (it restarted)
    void forget(size_t r);

private:
    static std::vector<uint64_t> chain(const std::vector<int32_t>& ids);
    size_t n_;
    std::mutex mu_;
    std::list<uint64_t> order_;   ///< the hashes, the least recently noted first
    std::unordered_map<uint64_t, std::pair<uint64_t, std::list<uint64_t>::iterator>> blocks_;   ///< hash -> (mask, place)
};

/// stands in for one engine (what the Service uses), over several
class ReplicaEngine : public Engine {
public:
    explicit ReplicaEngine(std::vector<std::shared_ptr<LaminaEngine>> engines, int64_t load_tokens);
    void generate(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                  const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                  const std::function<bool()>& on_beat) override;
    bool alive() override;
    void unload() override;
    /// starts the replicas that are not running (one that fails is said; the others serve)
    void load() override;
    void close() override;
    int64_t max_context() const override;
    int64_t known_ctx() const override;
    bool starting() const override;
    /// the slots in total: the Service lets this many requests run at once (an engine without slots counts one)
    int batch() const override;
    bool unloaded() const override;
    bool auto_load() const override { return true; }
    std::string death_note() override;
    json last_figures() const override;
    uint64_t done_count() const override;
    std::vector<int32_t> drained_tokens() const override;
    json slots_view() override;
    std::pair<int, int> admission() override;
    /// for /metrics: each replica's state, requests running and requests routed to it so far
    json replica_view() override;
    /// the Conversation cache card: the replicas' caches together (each field summed)
    json cache_view() override;

private:
    int route(const std::vector<int32_t>& ids);
    void heal();
    void refresh_info();

    std::vector<std::shared_ptr<LaminaEngine>> engines_;
    Router router_;
    int64_t load_tokens_;
    std::mutex mu_;
    std::vector<int64_t> inflight_, routed_;
    std::vector<std::unique_ptr<std::timed_mutex>> run_locks_;   ///< an engine without slots: one request at a time
    std::vector<bool> healing_;
    std::vector<double> heal_at_;
};

}  // namespace lamina
