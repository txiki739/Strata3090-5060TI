// include/lamina/prefill/experts.hpp - the prompt path's routed experts on one GPU.
//
// A prompt chunk uses nearly every expert of every layer.  Its experts go through in batches of consecutive ones in the
// order run_layer gets them (Prefill: by row count with a second GPU or an x16 link, else by id): per batch one
// submission of staging copies, one for each run of adjacent arena blobs among the experts the GPU's cache does not
// hold, one dequantization launch, one gather of the batch's rows, the two GEMMs of each expert with a SwiGLU between
// them, and one ordered add of every row, times its routing weight, into its token's sum.  A token's experts add up in
// the order of the batches.  A chunk uses many of the experts a layer streams, so they are copied ahead
// (`prefetch_*`), the likeliest first, while the prompt path's dense work runs and before the routing is known; the
// ones the routing picks that did not make it stream batch by batch.
//
// Local: on the prompt path's own GPU and stream, reading the chunk's MoE input and adding into its MoE output there.
// Remote: on a second GPU.  The first GPU (the 3090) sits on a PCIe 4.0 x4 link, over which an expert its cache lacks
// streams at 6.4 GB/s (~0.65 GB (IQ3_XXS) to ~1.4 GB (UD-Q4_K_XL) a layer); the second GPU sits on x16.  The first GPU
// copies the chunk's MoE input (FP16) to pinned host memory; the second copies it in, computes the experts the first
// one's cache lacks - from its own cache when it holds them - and returns their sums (FP16: they stay below ~20) the
// same way, while the first computes the experts its cache holds.  Neither GPU reads the other's memory: Windows gives GeForce cards no
// peer access.
#pragma once

#include "lamina/core/expert_cache.hpp"
#include "lamina/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace lamina::prefill {

class ExpertRunner {
public:
    ExpertRunner();
    ~ExpertRunner();
    ExpertRunner(const ExpertRunner&) = delete;
    ExpertRunner& operator=(const ExpertRunner&) = delete;

    /// For chunks of up to `max_chunk` tokens on `device`.  `stream`: the prompt path's stream on that device (local),
    /// or null (remote: its own streams; `main_device` is made current again after every call).  `cache`/`res`: the
    /// device's expert cache and its residency (n_layers x n_expert slots, -1 when not held), or null.  `streams`:
    /// whether experts its cache does not hold come its way (staging buffers).
    bool init(int device, int main_device, void* stream, core::ExpertSource* src, const core::ExpertCache* cache,
              const int32_t* res, int64_t n_expert, int64_t max_chunk, bool streams, std::string& err);
    /// Device bytes of the buffers for chunks of `chunk` tokens.
    static uint64_t bytes_needed(int64_t chunk, int64_t n_expert, bool remote, bool streams);
    /// The same with its prefetch area (prefetch_area).
    uint64_t bytes_for(int64_t chunk, const int32_t* skip) const;
    /// The prefetch area for chunks of `chunk` tokens, for the residency `skip` (see prefetch_need) and its own as
    /// they are now: at most what the chunk's dense steps leave time to copy ahead (area_cap), at most prefetch_need.
    /// A larger area lends more cache slots, refilled after the prompt.
    uint64_t prefetch_area(int64_t chunk, const int32_t* skip) const;
    /// ~0.3 MB a token on a GPU with an x16 link (`fast`: a second GPU, or set_fast_link; its link takes ~50 GB/s
    /// while the main GPU spends ~5.6 us a token on a layer's dense steps and the transfers of its input and sums),
    /// ~0.03 MB on the main GPU (its x4 link, ~6 GB/s).
    static uint64_t area_cap(int64_t chunk, bool fast);
    /// A local runner on a GPU with an x16 link (the split prompt path's second stage, prefill/split.hpp): its copies
    /// ahead go in pieces as a second GPU's do, into an area sized as a second GPU's.  Before init.
    void set_fast_link(bool on);
    /// The buffers for chunks of up to `chunk` tokens, carved from `region` (`bytes` long, e.g. lent cache slots), or
    /// with a null region from its own allocation (made once, for `max_chunk` tokens).  Before the first layer of
    /// every prompt that uses a region, and of the next one.  `area`: device memory for the prefetched blobs; null: what the region has left.
    bool bind(void* region, uint64_t bytes, int64_t chunk, std::string& err, void* area = nullptr,
              uint64_t area_bytes = 0);
    /// Bytes of prefetch area that hold a layer's streamed experts: the largest layer's experts that neither its
    /// cache nor `skip` (another GPU's residency, n_layers x n_expert, or null) holds, and a margin.
    uint64_t prefetch_need(const int32_t* skip) const;
    /// The same before a runner exists: `res` its cache's residency, `streams` as init's, `shift` set_shift's.
    static uint64_t prefetch_need(const int32_t* res, const int32_t* skip, int64_t n_expert, bool streams,
                                  double shift = 0.0);
    /// --prompt-shift (Prefill::set_shift): a layer's prefetch area also holds that share of the experts `skip` holds,
    /// for chunks of kShiftMinTokens and more.
    void set_shift(double fraction);
    /// The shortest chunk --prompt-shift applies to: below it (IQ3_S, both GPUs: +3.3% time at 2K tokens, +1.4% at 4K,
    /// -1.1% at 8K) copying an expert costs the second GPU more than computing it saves the first.
    static constexpr int64_t kShiftMinTokens = 8192;
    /// Copies ahead while the dense steps run.  `prefetch_begin`, before the routing of a layer of a chunk of `chunk`
    /// tokens: the experts to copy, the likeliest first (those its cache holds are skipped), up to the area's bytes
    /// and the chunk's area_cap.  A second GPU's runner then takes them a piece at a time in that order:
    /// `prefetch_next` queues the next piece (~1 ms of copies) into the prefetch area on its copy stream,
    /// `prefetch_pending` counts the pieces that have not landed, `prefetch_left` whether experts remain that fit.
    /// The main GPU's queues the ones that fit at once, in id order (its host thread can sit in a full launch queue
    /// through the dense steps; its batches run in id order and wait only for their own copies).  run_layer of that
    /// layer reads them there and streams the rest.
    bool prefetch_begin(int64_t layer, int64_t chunk, const std::vector<int32_t>& experts, std::string& err);
    bool prefetch_next(std::string& err);
    int prefetch_pending() const;
    bool prefetch_left() const;

    /// Remote: pinned [max_chunk, n_embd]: the MoE input goes in here, the sums come out here (both FP16).
    uint16_t* host_input() const;
    uint16_t* host_sum() const;
    /// Remote, before run_layer of the same layer: rows [t0, t0 + rows) of the input, copied in once `ready` (an
    /// event of the first GPU, recorded once host_input holds them) has passed; a chunk's rows in pieces, as they
    /// reach host_input.
    void stage_input(int64_t t0, int64_t rows, cudaEvent_t ready);

    /// Layer `layer` of a chunk of T tokens, the rows of tokens [t_begin, t_end): expert experts[j] serves rows
    /// [off[j], off[j + 1]), row r being token src[r] with routing weight w[r]; each row times its weight adds to its
    /// token's sum.  Local: reads `input` [T, n_embd] (FP16) and adds into `sum` [T, n_embd], queued on the prompt
    /// path's stream.  Remote: `input` and `sum` are null, the input came through stage_input, and the sums of
    /// [t_begin, t_end) reach host_sum in pieces of `piece` rows (t_begin a multiple of it), piece_done(k) recorded as
    /// piece k lands; a layer's calls cover [0, T) in order.  Returns once the work is queued.
    bool run_layer(int64_t layer, int64_t T, int64_t t_begin, int64_t t_end, const std::vector<int32_t>& experts,
                   const std::vector<int32_t>& off, const std::vector<int32_t>& src, const std::vector<float>& w,
                   int64_t piece, const uint16_t* input, float* sum, std::string& err);
    cudaEvent_t piece_done(int64_t k) const;
    /// Called after each batch's work is queued (Prefill::set_pacing; false: stop, the caller's error), or null.
    void set_pace(std::function<bool()> f);

    int64_t experts_resident = 0;   ///< expert-layer groups served from its cache
    int64_t experts_streamed = 0;   ///< expert blobs copied from the arena
    int64_t experts_prefetched = 0; ///< ...of which ahead of the routing
    double ms_host = 0;             ///< host time queuing its work
    double ms_gpu = 0;              ///< remote: its GPU time from the input's arrival (all layers but the last)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lamina::prefill
