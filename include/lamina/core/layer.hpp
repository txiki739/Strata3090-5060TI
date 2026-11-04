// include/lamina/core/layer.hpp - what the verify windows, the drafter and the prompt path share per layer: the
// router, the QSA layers' K/V state and its formats (with the elastic K/V pools), the embedding rows, and the PLE's
// run.
#pragma once

#include "lamina/core/layout.hpp"
#include "lamina/core/vmm.hpp"
#include "lamina/core/weights.hpp"

#include "lamina/kernels/gr.hpp"
#include "lamina/kernels/ngram.hpp"
#include "lamina/kernels/ple.hpp"
#include "lamina/kernels/qsa_decode_attn.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lamina::core {


// ================================ the MoE block ================================






// ================================ the QSA mixer ================================

/// The K/V storage of a QSA layer (`--kv`): FP16; INT8 codes with an FP16 scale per 64 values (kv_q8.hpp); 4-bit K
/// and V of rotated rows (q4_0) or INT8 K with 4-bit V (k8v4; kv_q4.hpp).
enum class KvFormat : int { F16 = 0, Int8 = 1, Q4 = 2, K8V4 = 3 };

/// THE PERSISTENT STATE OF ONE QSA LAYER FOR ONE SEQUENCE: the KV cache, the indexer's pooled keys, the rope
/// tables, and the per-token step buffer.  None of it is scratch - it survives every token.
///
/// `max_cells` is the sequence capacity the arena was sized for, and it fixes the rope table's length: a table
/// shorter than the sequence would have `rope_neox_apply` read past it, which is a wrong rotation rather than a
/// fault.
struct QsaState {
    KvFormat kv = KvFormat::F16;
    uint16_t* k_pool = nullptr;      ///< [page][kv_head][page_size][head_dim] fp16 (KvFormat::F16)
    uint16_t* v_pool = nullptr;
    /// Plan v0.3 P7: INT8 KV. Codes [page][kv_head][page_size][head_dim], one FP16 scale per 64 (K in K8V4 too).
    bool kv_int8 = false;            ///< KvFormat::Int8: K and V
    int8_t* k_q = nullptr;
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;
    uint16_t* v_scale = nullptr;
    /// 4-bit KV (kv_q4.hpp): codes [page][kv_head][page_size][head_dim / 2], scales [...][head_dim / 32]; K in Q4,
    /// V in Q4 and K8V4
    uint8_t* k_q4 = nullptr;
    uint8_t* v_q4 = nullptr;
    uint16_t* k_q4s = nullptr;
    uint16_t* v_q4s = nullptr;
    int32_t* page_table = nullptr;   ///< (n_pages,) logical page -> physical page
    int64_t n_pages = 0;
    int64_t max_cells = 0;
    /// The elastic K/V (qsa_set_kv_elastic): this state's pools are in a VMM range mapped only as far as the context
    /// needs - its index in layer.cpp's registry; -1: carved from the arena as before.
    int32_t kv_elastic = -1;

    float* idx_tail = nullptr;       ///< (idx_block - 1, idx_dim): the raw tail of the block being filled
    float* idx_dead = nullptr;       ///< (idx_dim,): the spare slot's key, CONSTANT for the sequence
    float* idx_pooled = nullptr;     ///< (max_cells/idx_block + 2, idx_dim); elastic: in the K/V's range
    int32_t* idx_block_pos = nullptr;

    float* cos_tab = nullptr;        ///< (max_cells, n_rot/2), built on the HOST in float64; elastic: mapped as the K/V
    float* sin_tab = nullptr;

    /// THE PER-TOKEN COUNTS, IN DEVICE MEMORY - the whole reason this layer can be a graph.  `qsa_step_fill`
    /// computes all four from the position, so one H2D per token updates every count every kernel needs.
    int32_t* step = nullptr;
    int32_t* attention_status = nullptr; ///< per-layer device status, captured D2H to host_step[kStepCount]
    /// The rope positions, (n_head,) int32, all equal to the current position.  `rope_neox_apply` reads its
    /// positions from the DEVICE, so a host scalar here would be an illegal access (and uncapturable).
    int32_t* pos_dev = nullptr;

    /// PINNED HOST STAGING AT FIXED ADDRESSES, and they are not an optimisation.  The uploads in `qsa_layer`
    /// copy FROM these, so a CUDA graph captures the SOURCE POINTER - and the first version of the layer filled
    /// a stack array and a local `std::vector` instead.  Those addresses are dead by the time the graph is
    /// replayed, which is this project's own recorded pitfall ("graph inputs live at fixed addresses") walked
    /// straight into: the replay would read whatever the stack held, and the step counts would be wrong in a way
    /// that looks like a working token.
    ///
    /// Pinned as well as fixed, because a pageable source would make `cudaMemcpyAsync` a synchronous staged
    /// copy and give back the overlap the doorbell protocol exists for.
    int32_t* host_step = nullptr;    ///< (kStepCount + 1,) pinned; final element is attention status
    int32_t* host_pos = nullptr;     ///< (n_head,) pinned
};

/// The format of the model's QSA layers (`--kv`): set before sizing and initializing the session.
void qsa_set_kv_format(KvFormat f);
KvFormat qsa_kv_format();
/// The MTP draft layer's: INT8 under the 4-bit formats (its K/V is one layer's, its drafts the speed).
KvFormat qsa_draft_kv_format();

/// Plan v0.3 P7: the RoPE cos/sin table (max_cells x n_rot/2 x 2 floats, 64 MiB at 262K) is identical in every
/// QSA layer. `with_rope = false` sizes a state that borrows it; `share_rope` points `st` at another state's table
/// instead of building a copy (the session builds it once, in the first QSA layer).  `kv`: the K/V format.
uint64_t qsa_state_bytes(const ModelGeometry& g, int64_t max_cells, bool with_rope = true,
                         KvFormat kv = qsa_kv_format());
uint64_t qsa_state_init(const ModelGeometry& g, int64_t max_cells, void* base, QsaState& st,
                        const QsaState* share_rope = nullptr, KvFormat kv = qsa_kv_format());
/// Frees what qsa_state_init took outside `base`: the pinned step and position staging (an elastic state's range goes
/// with qsa_kv_elastic_release).
void qsa_state_free(QsaState& st);
/// Zeroes the pools AND the indexer, so a fresh sequence matches the reference's own `zeros()`.  The KV pool
/// matters even for cells that are never attended, because `kv_gather` reads whatever the selection names.
void qsa_state_zero(const QsaState& st, const ModelGeometry& g, void* stream);
/// Item bf: a state for the batched prompt path on another GPU (prefill/split.hpp), carved from `base`: the K/V pools
/// for cells [first_page * page_size, max_cells) (the page table maps logical page p >= first_page to p - first_page,
/// the pages before it to the first), with `indexer` the indexer's tail, spare key, pooled keys (for all max_cells)
/// and block position.  Never elastic, no rotary tables, no per-token staging; the pools and the indexer zeroed on
/// `stream`.
uint64_t qsa_state_prompt_bytes(const ModelGeometry& g, int64_t max_cells, int64_t first_page, bool indexer,
                                KvFormat kv);
uint64_t qsa_state_prompt_init(const ModelGeometry& g, int64_t max_cells, int64_t first_page, bool indexer, KvFormat kv,
                               void* base, QsaState& st, void* stream);
/// The attention kernels' view of the state's K/V pools (qsa_decode_attn.hpp).
lamina::kernels::QsaAttnPools qsa_attn_pools(const QsaState& st);
/// Appends n_tok cells in the state's format (graph-capturable): token t's step record at step + t * step_stride,
/// its K and V rows at kcur/vcur + t * n_head_kv * head_dim (4-bit sides rotated and quantized here).
void qsa_kv_append_steps(const QsaState& st, const ModelGeometry& g, const int32_t* step, int step_stride,
                         const float* kcur, const float* vcur, int n_tok, void* stream);
/// The state's K/V arrays, laid out [page][kv_head][page_size][...] (with an identity page table the first cells
/// are one prefix of each), and their bytes per cell and KV head.
struct KvArray {
    void* data;
    uint64_t row_bytes;
};
std::vector<KvArray> qsa_kv_arrays(const QsaState& st, const ModelGeometry& g);
/// A cell's bytes in the K/V pools of format `kv` (every array, every KV head).
uint64_t qsa_kv_cell_bytes(const ModelGeometry& g, KvFormat kv);

/// THE ELASTIC K/V (--kv-grow; vmm.hpp; after Niko1221/Strata 0.1.40).  A state's K/V pools and its indexer's pooled
/// keys get addresses for every cell of the context but physical memory only for the first `init_cells`, and so do the
/// rotary tables (a chunk's rows built on the host as it is mapped); `qsa_kv_elastic_grow` maps more as the context
/// grows (from chunks the expert cache gives up) and `qsa_kv_elastic_shrink` hands them back.  The addresses never
/// move, so the captured graphs stay valid.  Set before sizing and initializing the session and the drafter.
void qsa_set_kv_elastic(bool enabled, int64_t init_cells);
bool qsa_kv_elastic();
/// Cells every elastic state can hold now (INT64_MAX when none is elastic).
int64_t qsa_kv_elastic_cells();
/// Chunks still to map for every elastic state to hold `cells` cells.
int64_t qsa_kv_elastic_need(int64_t cells);
/// Maps them, each from `take()` (0: a new chunk), the new memory zeroed.  Synchronous; nothing may be running on
/// the device.  false: out of memory.
bool qsa_kv_elastic_grow(int64_t cells, const std::function<VmmChunk()>& take);
/// Unmaps the chunks past `cells` cells, each handed to `give`.  Returns how many.
int64_t qsa_kv_elastic_shrink(int64_t cells, const std::function<void(VmmChunk)>& give);
/// Physical bytes the elastic pools hold, and what all of them would at the full context.
uint64_t qsa_kv_elastic_mapped_bytes();
uint64_t qsa_kv_elastic_full_bytes();
/// Unmaps and frees every elastic range (the engine's end: no state may use them after; a server starts the engine
/// again in its process).
void qsa_kv_elastic_release();






// ================================ THE TWO ENDS OF A TOKEN ================================

/// The embeddings of `n` token ids in DEVICE memory (checked by the caller) into out_dev [n, n_embd], one launch:
/// a column dequant of `token_embd.weight` (`[n_embd, n_vocab]`, so `out[i] = W[i][token]`), not a GEMV.
bool embed_rows(const WeightTable& tables, const ModelGeometry& g, const int32_t* tokens_dev, int64_t n, float* out_dev,
                void* stream, std::string& err);



// ================================ THE PLE (LAYER 1'S PER-LAYER EMBEDDING) ================================
//
//
// Layer 1's per-layer embedding.  Its table (`per_layer_token_embd.weight`, 28.8 GB of IQ4_NL in the configs' files)
// stays in its GGUF and is read by row: sixteen rows a token (90 bytes each as IQ4_NL).
struct PleRun {
    /// Null when the PLE is not wired (no table, or a pack without `blk.1.ple_*`).
    lamina::kernels::PleTable* table = nullptr;
    lamina::kernels::PleWeights w;
    float* hist = nullptr;             ///< the session's NG_HIST x hc_dim NORMALIZED history, row-fastest
    lamina::kernels::PleConsts consts;
    bool ready() const {
        return table != nullptr && table->is_open() && hist != nullptr &&
               (w.key_bf16 != nullptr || w.key_native_data != nullptr) &&
               w.value_bf16 != nullptr &&
               w.norm_key != nullptr && w.norm_query != nullptr && w.norm_conv != nullptr &&
               w.conv1d_f16 != nullptr;
    }
};




}  // namespace lamina::core
