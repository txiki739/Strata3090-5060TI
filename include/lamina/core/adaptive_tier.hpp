// include/lamina/core/adaptive_tier.hpp - plan v0.3 P6: a VRAM expert tier follows the conversation.
//
// Every few rounds the missing experts routed most often (decayed counts, at least twice) move into the tier: into
// their layer's empty slots first, then into the slots of the layer's least-routed residents when they were routed
// clearly more often.  The copies run on their own stream while the rounds go on; an evicted expert is a miss at once,
// a new one resident once its copy has landed (`apply_pending`).  Both decode loops (--serve and the speculative loop)
// use it, for the first GPU's cache and for a second GPU's.
#pragma once

#include "lamina/core/expert_cache.hpp"
#include "lamina/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lamina::core {

class AdaptiveTier {
public:
    AdaptiveTier() = default;
    ~AdaptiveTier();
    AdaptiveTier(const AdaptiveTier&) = delete;
    AdaptiveTier& operator=(const AdaptiveTier&) = delete;

    /// `host_res` is the residency table (n_layers x n_expert, slot or kNotResident) the dispatch reads (the verify
    /// windows take a snapshot of it; a device copy of it is its owner's to refresh); up to `max_moves` experts move
    /// per call.  `device`: the cache's GPU (its stream and copies), `main_device` current again afterwards.
    /// `deterministic_adapt` settles the prior epoch, submits the next one as a full batch, and orders equal gains
    /// by tuple; false preserves the asynchronous gain-only policy.
    bool init(ExpertCache& cache, ExpertSource& src,
              std::vector<int32_t>& host_res, int64_t n_layers, int64_t n_expert,
              int max_moves, std::string& err, int device = -1, int main_device = 0,
              bool deterministic_adapt = false);
    bool on() const { return res_ != nullptr; }
    /// The source has layers in pageable memory: their copies go through a pinned staging ring, at most kMaxFlying
    /// batches fly, and the mid-window pump waits for the second GPU's rows (`Verifier::set_rows_in`).  A copy from a
    /// pageable source makes cudaMemcpyBatchAsync block the host thread that launches the second GPU's graphs, while
    /// the main GPU spins on that GPU's flag.
    bool guarded() const { return guard_; }
    /// An empty slot sized for `layer`'s experts.
    void add_free(int64_t layer, int32_t slot) { free_[(size_t) layer].push_back(slot); }
    /// The elastic K/V (--kv-grow) took the slots [lo, hi): no move may target them, empty or not.
    void drop_free_slots(int64_t lo, int64_t hi) {
        for (auto& f : free_)
            for (size_t i = 0; i < f.size();)
                if (f[i] >= lo && f[i] < hi) { f[i] = f.back(); f.pop_back(); } else ++i;
    }
    int64_t free_slots() const;
    /// The slots [0, n) hold experts that live only in VRAM (--vram-pin-gib): never evicted.  And an expert the RAM
    /// does not hold is never a candidate (its copy would have no source; it lives in another GPU's frozen slots).
    void set_frozen(int64_t n) { frozen_ = n; }
    /// A tier in front of this one: the experts it holds or is loading are not candidates here, and this tier's
    /// copies of them are evicted first.  Each call adds one.
    void add_upper(const AdaptiveTier* upper) { uppers_.push_back(upper); }
    /// A tier behind this one: a candidate it holds counts half its use here (moving it only shifts work from that
    /// tier's GPU to this one's).  Each call adds one.
    void add_lower(const AdaptiveTier* lower) { lowers_.push_back(lower); }
    /// An update waits while the previous one's moves are under way, instead of dropping the ones not staged yet (a
    /// tier whose moves all go out before the next window).
    void set_wait(bool on) { wait_ = on; }
    /// The share of the routing counts kept after each update (--adapt-decay, upstream's c38dc71; 0.7 by default).
    void set_decay(float f) { decay_ = f; }

    /// Ranks and queues this call's moves, then decays `usage` (n_layers x n_expert) unless `decay` is false (a tier
    /// ranked before another on the same counts).  The queued moves whose residents are still in place give way to
    /// this call's (ranked again if still worth it); the experts still under way are not candidates.
    bool adapt(std::vector<float>& usage, std::string& err, bool decay = true);

    /// By default the copies go out in pieces (the first GPU's tier: its copies cross the link that the verify
    /// window's own transfers need, and slowed them while they ran).  `stage` evicts the next queued moves' residents
    /// until `bytes` are staged and not sent: called before a window, whose residency snapshot then leaves their slots
    /// alone.  `pump_bytes` sends the staged moves' next `bytes`, a blob in as many calls as it takes.
    void stage(uint64_t bytes);
    bool pump_bytes(uint64_t bytes, std::string& err);

    /// Paced (the second GPU's tier, its copies between windows): whole moves, each resident evicted as its copy goes
    /// out.  `pump` submits them in order while they fit `budget` bytes, `sent` the bytes submitted, after `ev` (on
    /// the cache's GPU: the last work that may read the slots they overwrite).
    void set_paced(cudaEvent_t ev) { paced_ = true; after_ = ev; }
    /// Unpaced: the copies still wait for `ev` (the work that may read their slots), as paced ones do.
    void set_after(cudaEvent_t ev) { after_ = ev; }
    bool pump(uint64_t budget, uint64_t& sent, std::string& err);

    /// Moves into slots outside its free lists (the prompt path's lent slots, given back): (layer, expert, slot)
    /// each, sent at once in batches of about `batch` bytes; each is admitted by apply_pending once its batch lands.
    bool refill(const std::vector<std::array<int32_t, 3>>& moves, uint64_t batch, std::string& err);

    /// A running average of the copies' measured rate (bytes per ms), 0 until the first timed ones have landed.
    double copy_rate() const { return rate_; }
    /// Admits the moves whose copies have landed into `host_res`; `wait` sends the queued ones and
    /// blocks until they have landed.
    void apply_pending(bool wait);

    int64_t swaps = 0, fills = 0;   ///< experts moved into an occupied / an empty slot
    int64_t dropped = 0;            ///< moves the next update dropped before their residents left
    uint64_t sent_bytes = 0;        ///< bytes copied
    double ms = 0;                  ///< host time in `adapt`

private:
    struct Move { int32_t layer, in, out, slot; };   // out: the evicted expert, < 0 for an empty slot
    ExpertCache* cache_ = nullptr;
    ExpertSource* src_ = nullptr;
    std::vector<int32_t>* res_ = nullptr;
    int64_t n_layers_ = 0, n_expert_ = 0;
    int max_moves_ = 0;
    int dev_ = -1, main_ = 0;
    bool paced_ = false;
    cudaEvent_t after_ = nullptr;
    std::vector<const AdaptiveTier*> uppers_, lowers_;
    bool wait_ = false;
    bool deterministic_ = false;
    int64_t frozen_ = 0;
    std::vector<uint8_t> blocked_;                        // per (layer, expert): not a candidate, rebuilt each call
    std::vector<uint8_t> upper_has_;                      // per (layer, expert), rebuilt each call
    std::vector<uint8_t> lower_has_;                      // per (layer, expert), rebuilt each call
    std::vector<std::vector<int32_t>> free_;              // per layer
    // The queued moves in order and their (residency index, slot) once landed: [0, admitted_) admitted, up to sent_
    // sent (and off_ bytes of the next), up to staged_ with their residents evicted.
    std::vector<Move> queued_;
    std::vector<std::pair<int32_t, int32_t>> pending_;
    size_t admitted_ = 0, sent_ = 0, staged_ = 0;
    // The moves whose batch has landed, [0, landed_): retire() runs inside the pumps too (they retire batches to free
    // staging blocks and flight slots), so what it learns is kept here for apply_pending rather than returned and lost.
    size_t landed_ = 0;
    float decay_ = 0.7f;
    uint64_t off_ = 0;
    cudaEvent_t t0_ = nullptr, t1_ = nullptr;   // around the last timed copies, whose `timed_` bytes have not landed
    uint64_t timed_ = 0;
    double rate_ = 0;
    std::string failed_;                        // a copy apply_pending could not submit
    cudaStream_t stream_ = nullptr;
    // the batches in flight, oldest first: a ring of events, each with the moves sent when it was recorded
    static constexpr int kBatches = 64;
    // bounds outstanding copy batches so the enqueue can never block the host thread that launches the second GPU's graphs
    static constexpr size_t kMaxFlying = 2;
    cudaEvent_t evs_[kBatches] = {};
    size_t batch_sent_[kBatches] = {};
    int first_ = 0, flying_ = 0;
    // a call's copies, sent as one batch (copy_blobs)
    std::vector<void*> cp_dst_;
    std::vector<const void*> cp_src_;
    std::vector<size_t> cp_bytes_;
    // pinned staging ring (guard_): a pageable source slice is staged into one of these pinned blocks so the batch's
    // sources are always pinned.  A block is held by the batch whose copy reads it and returns to the free list when
    // that batch retires.
    bool guard_ = false;
    uint8_t* stage_arena_ = nullptr;
    static constexpr size_t kStageBlock = 8u << 20;
    static constexpr size_t kStageBlocks = 8;
    std::vector<size_t> stage_free_, stage_used_;
    std::vector<std::vector<size_t>> batch_stage_;   // per ring slot: the blocks its batch reads
    bool stage_blocked_ = false;                     // copy() could not stage: the pump stops early
    size_t staged_now_ = 0;                          // slices staged into the batch being built
    uint64_t bytes_of(const Move& m) const;
    void evict(const Move& m);
    bool copy(const Move& m, uint64_t off, uint64_t n, std::string& err);
    bool send(std::string& err);
    bool time_begin(bool worth);
    bool end_batch(bool timed, uint64_t bytes, std::string& err);
    void retire();
};

}  // namespace lamina::core
