// include/lamina/core/expert_source.hpp - P2.S3/P2.S5: the expert pool's adapter to the host loop.
//
// `session_loop` publishes, per layer, the NORMED activation `x_f`, the ten routed expert ids and their router
// weights, and expects `k x n_embd` floats back.  This is the piece that turns those four things into an
// `ExpertPool::run` call.  Nothing here is clever, and that is the point: the pool, the kernel and the loop are
// each already verified, so this file has exactly one job - get the CONTRACT between them right.
//
// THE CONTRACT, and each clause is a way to be wrong:
//
//   1. **`x_f` IS THE SAME PINNED BUFFER EVERY LAYER.**  Its ADDRESS never changes, so the `ActQ` cannot be
//      cached by pointer - a cache keyed on `x_f` would quantize layer 0 and reuse it for all 47 remaining
//      layers, which is a finite, plausible, completely wrong token.  There is no cache here at all: the
//      conversion is one 2560-element pass against a 0.3 ms/layer budget, and a correct answer is worth more
//      than the microseconds.
//   2. **THE POOL DOES NOT APPLY THE ROUTER WEIGHT.**  `moe_combine` (`src/core/layer.cpp:479`) sums
//      `w[i] * parts[i]` on the device.  `ExpertJob::weight` is a diagnostic field; setting it here would
//      apply the weight twice, which is invisible in a single layer and compounds over 48.
//   3. **THE KERNEL'S GEOMETRY IS FIXED BY THE ARTIFACT** (`H = 2560`, `FF = 640`, `BLOB = 1,382,400`).  A
//      different `n_embd` or a different expert width is REFUSED rather than mis-indexed, because the blob's
//      internal offsets are compile-time constants and reading a 640-wide expert as a 2560-wide one walks off
//      the end into the next expert's bytes without faulting.
#pragma once

#include "lamina/core/expert_cache.hpp"
#include "lamina/kernels/cpu/pool.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace lamina::core {

class SecondGpu;

/// GPUs that hold experts beside the main GPU's cache (--second-gpu A[,B,C]): up to three, each computing the experts
/// it holds of a verify window's layer beside the CPU pool.
constexpr int kMaxGpuTiers = 3;

/// Where one routed expert's bytes come from: the resident arena (ArenaExpertSource) in the engine.
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    /// The 1,382,400-byte blob for `(layer, expert)`, or nullptr if it cannot be produced.
    ///
    /// The pointer only has to stay valid until the next `blob()` call: with `h = 0` every expert is computed
    /// immediately and nothing is retained.  A CACHING source must return pointers into the cache, not into a
    /// reused staging buffer - otherwise the pool would read the next expert's bytes while computing this one.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    /// Blobs touched, for the driver to report.  A source that does not count returns 0.
    virtual int64_t reads() const { return 0; }

    /// Plan v0.3 P5: whether `blob(layer, expert)` lies in page-locked, CUDA-registered memory, so an
    /// asynchronous host-to-device copy can DMA it directly (no staging copy on the CPU).
    virtual bool pinned(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return false; }

    /// Called once before the first expert of a layer.  A source that reads from disk wants to start the read
    /// here so it overlaps the quantisation, and a prefetching source in Phase 3 wants the ids.
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) { (void) layer; (void) ids; (void) k; }
    /// Plan v0.3 P6: the DEVICE address of a pinned, mapped blob (the GPU can read it over PCIe), or null.
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return nullptr; }
    /// Whether `layer`'s blobs in RAM have device aliases (the PCIe share can read them).  Not device_alias(layer, 0):
    /// that expert may live only in VRAM.
    virtual bool layer_mapped(int64_t layer) const { return device_alias(layer, 0) != nullptr; }

    /// Whether `blob(layer, expert)` exists.  False for an expert that lives only in VRAM (--vram-pin-gib): a GPU
    /// holds it for good and computes it, so no path may ask the RAM for it.
    virtual bool in_ram(int64_t layer, int64_t expert) const { (void) layer; (void) expert; return true; }
};

/// Plan v0.3 P6: the PCIe share of a verify window's layer - missed experts the main GPU reads over PCIe - written by
/// the pool (mapped host memory) right after the ring and published before the CPU starts its own share.  (The
/// experts in its VRAM tier the window decides itself, from the residency the pool reads too.)  Groups of entries
/// that share a blob: `ptr2[q]` is staging slot q (DMA) or the arena's device alias, its entries dst/tok
/// [start2[q], start2[q + 1]).  counts[1] = entries, counts[2] = groups.
struct GpuPlanSink {
    int32_t* counts = nullptr;
    int32_t* dst = nullptr;                ///< cap: the entry's row of parts (token * k + j)
    int32_t* tok = nullptr;                ///< cap: the entry's token
    unsigned long long* ptr2 = nullptr;    ///< cap
    int32_t* start2 = nullptr;             ///< cap + 1
    unsigned long long staging = 0;
    int64_t staging_cap = 0;
    int64_t cap = 0;
    /// Whether the window takes a PCIe share (`--pcie-frac` > 0): the pool publishes one every layer, else none.
    bool pcie = false;
    void (*publish)(void* ctx) = nullptr;
    /// Starts the DMA copies of `n` host blobs (pinned) into staging slots 0..n-1 and signals the GPU when they land.
    void (*fetch)(void* ctx, const uint8_t* const* src, int n, size_t bytes) = nullptr;
    void* ctx = nullptr;
    /// Plan v0.3 P6: how a PCIe group reaches the GPU.  0 = the copy engine stages it (DMA, `fetch`); 1 = the grouped
    /// kernel reads the mapped arena directly; 2 = a copy kernel stages it inside the graph.  For 1 and 2 `ptr2`
    /// holds the arena's device alias.
    int pcie_mode = 0;
    /// The GPU reads only the rows it did not compute, so the pool leaves the GPU's rows of `out` unwritten.
    bool host_rows_only = false;
    /// The GPU tiers' shares of this token group (mapped; null without a tier): their entries, tier by tier,
    /// [n, end of tier 0's, of tier 1's, of tier 2's, entries], and per tier the flag its graph raises to `ring` once
    /// its rows are in `out`, which the main GPU waits for.  The pool raises a tier's flag itself when that tier takes
    /// no share, and sets its `tier_ring` to `ring` once the flag is raised or a share is submitted.
    int32_t* gpu2_list = nullptr;
    uint32_t* tier_flag[kMaxGpuTiers] = {};
    uint32_t ring = 0, tier_ring[kMaxGpuTiers] = {};
    /// This token group's activations as the window's router wrote them (mapped): the rows the pool takes, `xk_stride`
    /// bytes apart, when it takes them so: ggml's Q8_K rows (`pool_takes_q8k`) or the q8_1 rows (`pool_takes_q8_1`), else
    /// null (the pool quantizes the floats); and the q8_1 rows (n_embd / 32 blocks each) the GPU tiers take.
    const uint8_t* xk = nullptr;
    int64_t xk_stride = 0;
    const uint8_t* x1 = nullptr;
};

/// Whether the pool takes a native layer's gate/up activation as ggml's Q8_K rows (`GpuPlanSink::xk`): its format's
/// vec_dot_type (the i-quants', the K-quants'), and not the Q2_0 kernels' own.
inline bool pool_takes_q8k(const lamina::kernels::cpu::NativeFmt& f) { return f.gu_type != 42 && f.gu_act == 15; }
/// Whether it takes the q8_1 rows: a Hadamard-folded layer's, rotated on the GPU for its own experts.
inline bool pool_takes_q8_1(const lamina::kernels::cpu::NativeFmt& f) { return f.had_gu; }

/// The adapter's own state.  One per session, reused every layer so the token path allocates nothing (P2.T10).
struct ExpertDispatch {
    lamina::kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    int64_t n_expert = lamina::kernels::cpu::NE;

    /// Counters, for the driver to report rather than for control flow.
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    /// The verify windows' routed experts a GPU held and those none held (the stats line).
    int64_t cache_hits = 0;
    int64_t cache_refused = 0;
    /// The residency table (`n_layers x n_expert`, slot or -1): the host's copy of what the device hit path reads
    /// (`Verifier::residency`).  The pool leaves a resident expert's rows to the GPU.
    const int32_t* host_res = nullptr;
    /// Plan v0.3 P6 verify window: one quantized activation per token, the multi-token jobs, and the expert ->
    /// job map (reset after every layer).
    std::vector<lamina::kernels::cpu::ActQ> act_multi;
    /// Plan v0.3 P6: a native pack's per-token activations (MAXT x kNativeActBytes).
    std::vector<uint8_t> nact_multi;
    std::vector<lamina::kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;
    /// Plan v0.3 P6: the verify window's PCIe share; `pcie_num`/256 of each layer's distinct missed experts (the last
    /// ones in routing order) are read by the GPU over PCIe.
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;      ///< distinct experts the GPU read over PCIe in verify windows
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;   ///< verify-window dispatch sections
    /// Plan v0.3 P6: decayed routing counts per (layer, expert) during decode (sized by the caller; empty = off),
    /// which the driver uses to swap the most-routed missing experts into the VRAM tier between rounds.
    std::vector<float> usage;
    int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
    int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
    /// Routed (token, expert) entries per (layer, expert) in verify windows, when the caller sizes it: how many of
    /// the entries a cache of the N most-routed experts would serve.
    std::vector<uint32_t> routed;
    /// The GPU tiers past the main GPU (verify windows), in order: an expert the main GPU's cache does not hold is
    /// computed on the first tier that holds it (`res`, n_layers x n_expert, slot or -1) or has prefetched it, beside
    /// the CPU pool.
    struct GpuTier {
        SecondGpu* gpu = nullptr;
        const int32_t* res = nullptr;
        uint64_t prefetch_bytes = 0;   ///< per layer, its likeliest experts no GPU holds, copied there ahead
        int64_t entries = 0;           ///< routed entries it served
    };
    GpuTier tiers[kMaxGpuTiers];
    int n_tiers = 0;
    int64_t tier_entries = 0;      ///< routed entries the tiers served, all of them
    int64_t tier_skipped = 0;      ///< layers the CPU took the tiers' shares of (it was quicker)
    bool tier_used = false;        ///< this layer's decision
    uint64_t tier_min_bytes = 0;   ///< a layer's miss bytes from which the tiers take their shares
    std::string tier_err;          ///< `fail` points here when a tier's GPU failed
    /// Set when `dispatch` could not produce an answer.  The loop itself has no error channel, so this is
    /// where a source failure surfaces: the driver checks it after `session_loop` returns rather than the
    /// engine computing from a half-filled `parts`.
    ///
    /// **`failed` LATCHES AND `fail` IS ONLY THE MESSAGE.**  Clearing the message must not re-arm the adapter:
    /// a session that failed at layer 5 has already fed `moe_combine` whatever `parts` held, so every layer
    /// after it is built on a hole - and resuming into a plausible-looking token is the exact outcome this
    /// whole mechanism exists to prevent.  `failed` is what the short-circuit reads.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

/// Plan v0.3 P6: the pool for a verify window of `n_tok` tokens.  `x_f` is (n_tok, n_embd), `ids` (n_tok, k) and
/// `out` (n_tok * k, n_embd).  Each distinct missed expert is computed once for all the tokens routed to it;
/// resident experts' rows are the main GPU's (zeroed, or left unwritten with `GpuPlanSink::host_rows_only`), and each
/// GPU tier writes its share's rows itself, after this returns (`GpuPlanSink::tier_flag`).
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out);
/// Copies `layer`'s likeliest experts that no GPU holds to the tiers' prefetch slots (each tier's `prefetch_bytes`,
/// the first tier's filled first), where that layer's dispatch finds them; `ids`/`w` (n_tok, k) are the layer's
/// predicted routing (`Verifier::set_predict`).
void expert_prefetch_multi(ExpertDispatch& d, int64_t layer, const int32_t* ids, const float* w, int64_t n_tok,
                           int64_t k);


// ================================ THE RESIDENT ARENA (R2.1) ================================
//
// A mapping of the experts' file ran the pool at ~19 GB/s against 42.8 GB/s from resident memory: mapped,
// file-backed pages are the ones the OS drops from the standby list when it wants memory, and the expert stream then
// re-faults from disk.  So the arena reads the experts from the model's GGUF once at startup into anonymous memory
// the OS has no cheaper reason to evict: one pinned block per layer (cudaHostAlloc, portable and mapped), which the
// GPUs need for cache fills, the prompt path and the PCIe share of the misses.  Registering one reserved range
// instead (cudaHostRegister) is limited to ~42 GiB on Windows once a second GPU's context exists, and past that limit
// every CUDA allocation fails; allocated blocks reach ~75 GiB with one or two GPUs.  The layers past the pinning
// budget share one locked, pageable range; the note says how many layers were pinned.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    /// Allocates the arena and reads the experts into it from set_gguf's shards.  Prints nothing; the caller reports
    /// `note()` and the load rate, because those are the two numbers that say whether the arena is the one that was
    /// asked for.
    bool open(int64_t n_layers, int64_t n_expert, int threads, std::string& err);
    /// Before open(): the model's GGUF shards (lamina::gguf_split_paths), whose expert tensors are found by name.
    void set_gguf(const std::vector<std::string>& shards) { gguf_ = shards; }
    /// Before open(): the experts (n_layers x n_expert, non-zero) that live only in VRAM.  The arena leaves them
    /// out (a 119.5 GiB Q8_0 then fits a 125.7 GiB machine), blob() has none for them, and `read_file_blob` reads
    /// one from the model for the GPU that keeps it.
    void set_vram_only(std::vector<uint8_t> mask) { vram_only_ = std::move(mask); }
    /// Before open(), --no-file-cache: the experts read from the GGUF past the OS file cache (Windows: unbuffered
    /// reads; Linux: the pages read dropped from the page cache), which then keeps no second copy of them.  Where
    /// the RAM holds the arena and the cached file together, the cache settles on pages the later starts' arenas
    /// would have had (on this PC the faster ones).  Each start reads the experts from the drive.
    void set_uncached(bool on) { uncached_ = on; }
    void close();

    bool mapped() const { return !layer_base_.empty(); }
    int64_t blobs() const { return blobs_; }
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const { return reads_.load(std::memory_order_relaxed); }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;
    bool layer_mapped(int64_t layer) const override {
        return layer >= 0 && layer < pinned_layers_ && layer_dev_[(size_t) layer] != nullptr;
    }
    bool in_ram(int64_t layer, int64_t expert) const override {
        return idx_.empty() || idx_[(size_t) (layer * n_expert_ + expert)] >= 0;
    }
    /// One expert's blob read from the model into `dst` (blob_bytes(layer) bytes; any expert, thread-safe).
    bool read_file_blob(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) const;
    /// The bytes the arena left out (set_vram_only).
    uint64_t vram_only_bytes() const { return vram_only_bytes_; }
    /// --arena-place (off by default: for RAM whose regions read at different rates, as unpaired DIMMs make):
    /// times each blob's place in the pinned blocks (every physical core reading it; the slow ones again), then swaps
    /// the experts on no GPU out of the slow places with GPU-resident ones in fast places of the same size, in any
    /// pinned layer: the GPU-resident ones the hottest first (the last to move to the CPU as the tiers adapt), the
    /// others the coldest first.  `rank` and `on_gpu` per (layer, expert): the profile's rank (lower: routed more),
    /// whether it stays on a GPU.  Before generation (nothing may read the arena meanwhile).  `line`: what it found
    /// and did, for the startup print; false: LAMINA_LOAD_VERIFY found a moved blob unlike the model's.
    bool place(const std::vector<int32_t>& rank, const std::vector<uint8_t>& on_gpu, std::string& line);
    /// Between requests, after place(): the swaps for `rank` and `on_gpu` as they are now, on the places it timed
    /// (none with LAMINA_ARENA_PLACE_DRY), the CPU's hottest experts' first; returns how many.  replace_some() makes
    /// up to `max` of them on the calling thread (nothing may read the arena meanwhile) and returns how many are left.
    size_t replan(const std::vector<int32_t>& rank, const std::vector<uint8_t>& on_gpu);
    size_t replace_some(size_t max);

    /// What backing was obtained and why, for the startup print.  "The engine adapts to the machine it is on" is
    /// only true if the engine says what it got.
    const std::string& note() const { return note_; }
    double load_gib_per_second() const { return gib_per_s_; }

private:
    std::vector<uint8_t*> layer_base_;       ///< each layer's experts: a pinned block, or a part of `tail_`
    std::vector<const uint8_t*> layer_dev_;  ///< the device alias of each pinned layer (null for the others)
    std::vector<uint8_t*> blocks_;           ///< the pinned blocks, owned (layers [0, blocks_.size()))
    void* tail_ = nullptr;                   ///< the layers past the pinning budget, locked resident, owned
    uint64_t tail_bytes_ = 0;
    uint64_t tail_locked_ = 0;   // of it, what lock_resident locked
    /// Linux (LAMINA_PIN_AFTER_COPY, default on): the arena as one anonymous range the load's threads write into,
    /// each layer page-locked with cudaHostRegister after the load (cudaHostAlloc zeroes and locks it on one thread
    /// first: ~16 s for 55 GiB).  The layers past the first refused registration are locked resident instead.
    void* anon_ = nullptr;
    uint64_t anon_bytes_ = 0;
    std::vector<std::pair<uint8_t*, uint64_t>> registered_;
    uint8_t* anon_lock_ = nullptr;
    uint64_t anon_lock_bytes_ = 0;
    int64_t pinned_layers_ = 0;              ///< layers [0, pinned_layers_) are page-locked and mapped
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    std::atomic<int64_t> reads_{0};   ///< (a counter only; two threads can read blobs at once)
    std::string note_;
    double gib_per_s_ = 0.0;
    std::vector<std::string> gguf_;
    std::vector<uint8_t> vram_only_;
    bool uncached_ = false;
    /// per (layer, expert): its place among its layer's blobs in the arena, -1 for one only in VRAM (empty: all
    /// in the arena, at their expert index)
    std::vector<int32_t> idx_;
    /// per (layer, expert): the pinned block its blob is in once place() moved some (empty: its layer's)
    std::vector<int32_t> blk_;
    /// place()'s places (the mapped blocks' blobs): each block's first, whether each is slow, the (layer, expert) in
    /// each; the swaps replan() left to make
    std::vector<int64_t> place_off_;
    std::vector<uint8_t> place_slow_;
    std::vector<int32_t> place_occ_;
    std::vector<std::pair<size_t, size_t>> pending_;
    size_t pending_at_ = 0;
    std::vector<uint8_t> swap_buf_;
    bool place_dry_ = false;
    int64_t place_block(size_t q) const;
    uint8_t* place_addr(size_t q) const;
    void note_swap(size_t q1, size_t q2);
    std::vector<std::pair<size_t, size_t>> plan_swaps(const std::vector<int32_t>& rank,
                                                      const std::vector<uint8_t>& on_gpu) const;
    uint64_t vram_only_bytes_ = 0;
    std::vector<std::pair<size_t, uint64_t>> spans_;   ///< per (layer, role): its shard and the tensor's offset
    std::vector<int> fds_;                              ///< the shards, open for read_file_blob
};

}  // namespace lamina::core
