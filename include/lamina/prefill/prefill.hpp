// include/lamina/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks through all 48 layers, leaving the session state
// (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path would have left it; the
// decode loop then continues with the next token.  Per layer: the projections are tensor-core GEMMs (quantized
// weights dequantized to FP16 on the fly, BF16 weights as they are) and the recurrences walk the tokens inside one
// kernel, in sub-chunks of up to 2048 tokens whose buffers are reused; the routed experts then run over the whole
// chunk (lamina/prefill/experts.hpp), so an expert the cache does not hold is streamed once per chunk.  Long chunks
// cost little VRAM besides the residual stream (~55 KB a token).
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "lamina/core/expert_cache.hpp"
#include "lamina/core/expert_source.hpp"
#include "lamina/core/layer.hpp"
#include "lamina/core/mtp.hpp"
#include "lamina/core/session.hpp"
#include "lamina/core/weights.hpp"
#include "lamina/prefill/experts.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace lamina::prefill {

/// --hc-q8: the hyper-connection matrices have no BF16 copy, and a chunk's buffers hold a layer's in BF16 (26 MiB).
/// Before any sizing.
void set_hc_q8(bool on);

/// The sections `Prefill::profile` times, in the order a layer runs them.
enum PrefillSection {
    kPsPle, kPsHcRead, kPsGdn, kPsGdnConv, kPsGdnScan, kPsGdnNorm, kPsGdnOut, kPsQsaProj, kPsQsaIndexer, kPsQsaScores, kPsQsaAttn, kPsQsaOut, kPsHcFfn, kPsRouter,
    kPsExperts, kPsCombine, kPsDraft, kPsCount
};
const char* prefill_section_name(int section);

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time queuing this GPU's experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> this GPU
    int64_t experts_prefetched = 0; ///< ...of which ahead of the routing
    int64_t experts_resident = 0;   ///< expert-layer groups served from its VRAM tier
    double ms_ple = 0;
    double ms_section[kPsCount] = {};   ///< GPU ms per section, with `Prefill::profile`
};

/// The first stage's side of a split prompt path (Prefill::set_layers, lamina/prefill/split.hpp): where a chunk's
/// residual after the stage's last layer goes.
struct SplitOut {
    virtual ~SplitOut() = default;
    /// Before a chunk's residual goes out: pinned host memory for it ([T, hc * n_embd] floats), once the second
    /// stage has read the previous chunk's out of it (waited for on the host; null: an error).
    virtual float* out_begin(std::string& err) = 0;
    /// The chunk's residual is on its way: piece k (the sub-chunk of `piece` tokens) lands at `landed[k]`.  `tokens`:
    /// the chunk's and the one after it (the draft layer's), alive until the second stage is done; `run_end`: the
    /// chunk ends the run() call (the prompt cache may checkpoint after it).
    virtual bool out_end(const int64_t* tokens, int64_t T, int64_t pos0, const cudaEvent_t* landed, int64_t piece,
                         bool run_end, std::string& err) = 0;
};

class Prefill {
public:
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// For chunks of up to `max_chunk` tokens.  `host_res`: the static residency table (n_layers x n_expert, slot
    /// or -1) or null; `cache` its slots.  `offload`: a runner on a second GPU (initialized, bound by the caller)
    /// that computes the experts this GPU's cache does not hold; null: they stream here.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t max_chunk,
              void* stream, ExpertRunner* offload, std::string& err);

    /// Device bytes of the buffers for chunks of `chunk` tokens.
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk,
                                 bool offload);
    /// The same with the area its experts are prefetched into (without a second GPU), for the residency as it is now.
    uint64_t bytes_for(int64_t chunk) const;
    /// The buffers for chunks of up to `chunk` tokens, carved from `region` (`bytes` long: lent expert-cache
    /// slots; what is left over takes prefetched experts), or with a null region from its own allocation (made once,
    /// for the longest chunk).  Before the first chunk of every prompt that uses a region, and of the next one.
    bool bind(void* region, uint64_t bytes, int64_t chunk, std::string& err);

    /// The order the experts a layer streams are copied ahead in: per layer (n_layers x n_expert) its experts, the
    /// likeliest to be routed first (the expert profile's ranking).  Empty: by id.
    void set_order(std::vector<int32_t> order);
    /// --prompt-shift: with the second GPU's share, `fraction` of the experts this GPU holds (per layer, the likeliest
    /// in set_order's order that the expert source has in RAM) go to the second GPU too, copied ahead with the ones it
    /// streams (0: none).
    void set_shift(double fraction);

    /// Positions [pos0, pos0 + n) holding `tokens`, in chunks of the bound length; `ss.ple_prev` must be the two
    /// tokens before pos0 (oldest first, -1 for none) and is advanced to the last two of these.  With a `draft`
    /// layer tokens[n], the token after the last position, is read too.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    /// Positions the last `run` completed: n, or fewer after `should_stop` (the session state holds them).
    int64_t processed() const { return processed_; }

    const PrefillStats& stats() const { return stats_; }
    /// The prompt's routing: per layer (n_layers x n_expert) how often the tokens run since `clear_routed` chose each
    /// expert.
    const std::vector<uint32_t>& routed() const;
    void clear_routed();
    /// Adds another instance's counts (the second stage's layers).
    void add_routed(const std::vector<uint32_t>& counts);

    /// Item bf, the prompt path in two stages on two GPUs (lamina/prefill/split.hpp): this instance runs layers
    /// [first, last) only.  first > 0: no embedding and no PLE block; a chunk's residual reaches the buffers' R
    /// (`residual()`) a sub-chunk at a time, each piece before the event `set_in_events` names for it; the session
    /// state holds the layers from `first` on (its QSA states and GDN states indexed from there).  last < n_layers:
    /// the residual after layer last - 1 goes to `out`, and the draft layer's K/V are the last stage's.
    void set_layers(int64_t first, int64_t last, SplitOut* out);
    /// Its own GPU's link to the arena is x16 (the second stage): ExpertRunner::set_fast_link.  Before init.
    void set_fast_link(bool on);
    /// A stage on a GPU that runs another stage's work too: at each section of a layer (and each batch of its experts)
    /// the host waits until the GPU has reached the section `sections` back, so that a GPU queue of this stage's
    /// work never holds the other's up for long (0: off).  Before init.
    void set_pacing(int sections);
    float* residual() const;
    /// The second stage, before each run() of one chunk: piece k of its residual is in `residual()` at ev[k].
    void set_in_events(const cudaEvent_t* ev, int64_t n);
    /// The draft layer's weights and cells in place of `draft->prompt_kv()` (the second stage's copies), or null.
    const core::MtpPromptKv* draft_kv = nullptr;
    /// Called after each chunk's last layer (with every layer here) with its final residual rows on the device
    /// ([T, hc * n_embd] floats from position pos0), queued on `stream` and read before the draft layer's K/V pass
    /// overwrites them (--dump-draft-data).  Null: none.
    std::function<void(const float* R, int64_t T, int64_t pos0, void* stream)> on_final;

    /// The MTP draft layer (or null): after every chunk, its K/V over the chunk's cells in the chunk's batched
    /// arithmetic (cell t pairs the chunk's final residual at t with the token at t + 1).
    core::MtpDrafter* draft = nullptr;

    /// Checked before every chunk: true stops the prompt there (`run` returns false with err "cancelled", and the
    /// state and `ss.ple_prev` hold the chunks before it, `processed()` positions).
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// GPU time per section into `stats().ms_section` (events between the sections; the host waits for the last
    /// one after every chunk).
    bool profile = false;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
    int64_t processed_ = 0;
};

}  // namespace lamina::prefill
