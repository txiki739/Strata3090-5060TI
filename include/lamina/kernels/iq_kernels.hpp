// include/lamina/kernels/iq_kernels.hpp - the GGUF expert and embedding formats on the GPU: the i-quants (IQ1_M,
// IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL) and Q2_0 of the IQ2_XS / IQ3_XXS model files, Q4_K, Q5_K,
// Q5_1 and Q8_0 of the UD-Q4_K_XL file, and Q5_0 of unsloth's Dynamic 3.1 quants.
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);
/// iq_mmvq for 2..8 columns from their interleaved copy (native_quantize_q8_1_il), or one from its q8_1 blocks,
/// bitwise, a warp taking `rows` rows (1, 2 or 4): IQ3_S.
bool iq_mmvq_il_supported(int ggml_type) noexcept;
void iq_mmvq_il(int ggml_type, const void* w, const void* x_q8_1, const void* x_il, float* y, int n_in, int n_out,
                int ncols, void* stream, int rows);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table of `n_rows` rows (`row_bytes` each; the table may
/// be mapped host memory) dequantized to fp32, `n_embd` per row (a multiple of 256); an id outside the table (-1:
/// a pick of no finite logit) gives a row of zeros.
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, int64_t n_rows, const int32_t* tokens,
                   int64_t n_tok, int64_t n_embd, float* out, void* stream);
/// iq_dequant_gu_f16 and iq_dequant_f16 (the down matrix) of n experts in one launch: expert i's blob at blobs[i]
/// (a DEVICE array), its gate/up into gu + i * 2 * n_ff * n_embd and its down into dn + i * n_embd * n_ff.
void iq_dequant_experts_f16(int gu_type, int d_type, const uint8_t* const* blobs, int n, size_t up_off, size_t down_off,
                            int64_t n_ff, int64_t n_embd, uint16_t* gu, uint16_t* dn, void* stream);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int gu_act = 9, d_act = 9;             // ggml vec_dot activation types
    bool q5k_fma = true;                   // exact Q5_K dots: the mins' sum as one FMA (set_exact_q5_K_fma)
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
    int had_block = 0;                  // Hadamard-folded down weights: h is rotated before it is quantized (hadamard.hpp)
    const float* d_signs = nullptr;     // their sign vector, n_ff floats on this device, or null
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
                                        int gu_act = -1, int d_act = -1);

/// How the GPU's native expert kernels quantize and sum (`--expert-math`).  The CPU pool always runs ggml's own contract
/// (each layer's vec_dot types); this decides what the GPU does with the experts it holds.
///  - Fast (the default): llama.cpp's CUDA contract - q8_1 activations for gate/up and the hidden, lane-strided warp
///    sums.  A GPU hit agrees with the CPU computing the same expert to rounding (~1e-2 relative at the expert output,
///    the same as the CPU's own distance from fp32), not bit for bit, so greedy output can change with which experts
///    happen to be resident.
///  - Exact: the CPU pool's ggml arithmetic - Q8_K / Q8_0 / Q8_1 per the layer's vec_dot types, the accumulation and
///    reduction order of ggml's x86 dots (AVX2 code: an AVX-512 build of the pinned ggml has no other branches for these
///    formats), and `expf_contract`, which the pool's SwiGLU takes too under Exact (`cpu::set_native_exact_swiglu`) -
///    so a GPU hit is bit-identical to the CPU's result and the output does not depend on residency.  Slower; for
///    regression tests, A/B runs and reproducible output.  Another CPU back end would need its own contract.
enum class ExpertMath { Fast, Exact };
void set_expert_math(ExpertMath m) noexcept;   ///< once, before any expert work
/// Exact's Q5_K dots round the mins' sum as ggml's compiled C does: one FMA (GCC, Clang) or a product then a sum
/// (MSVC).  `cpu::ggml_q5_K_mins_rounding()` probes it; set before any layout is built.
void set_exact_q5_K_fma(bool fused) noexcept;
ExpertMath expert_math() noexcept;
/// The activation type the GPU's kernels take for a layer whose CPU vec_dot type is `cpu_act`: `cpu_act` under Exact,
/// q8_1 (9) under Fast.
int gpu_expert_act(int cpu_act) noexcept;
/// Whether Exact has bit-exact GPU kernels for these weight and CPU activation types (Q4_K/Q5_K gate/up with Q8_K, and
/// Q5_1 + Q8_1, Q8_0 + Q8_0 or Q4_K/Q5_K + Q8_K down).  Other formats only have the Fast kernels.
bool native_expert_exact_supported(int gu_type, int d_type, int gu_act, int d_act) noexcept;
/// Whether the GPU has the dot products (native_expert_grouped) and the dequantizers (the prompt path) for a layer's
/// gate/up and down formats at this geometry.
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;
/// Exact's dot products alone, for their test: `n_rows` rows of `type` (Q4_K, Q5_K, Q5_1, Q8_0; `row_bytes` apart,
/// device) against one activation row in the CPU's vec_dot type (`n` values, device) -> out[r] (device).
void native_exact_dot_rows(int type, const void* rows, size_t row_bytes, int n_rows, const void* x, int64_t n,
                           float* out, void* stream);
/// Exact's Q8_0 quantizer of the hidden rows, for its test: n floats (device) -> n / 32 blocks of Q8_0 (device).
void native_q8_0_contract_probe(const float* x, int64_t n, void* y, void* stream);

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.  The largest supported hidden
/// activation format is selected automatically (Q8_K, Q8_0 or Q8_1).
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff, bool rotated = false);
/// y = H_128 (signs * x), the Hadamard-folded experts' rotation (hadamard.hpp), per 128 values of each of `rows` rows of `width`
/// floats, bitwise hadamard_rotate; x and y may be the same.  `signs`: `width` floats on this device, or null.
void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, const float* signs, void* stream);
/// The same in place on FP16 rows: FP32 arithmetic, rounded once on the store.
void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, const float* signs, void* stream);
/// The current device's copy of a host sign vector (made once; kept for the process).
const float* hadamard_device_signs(const float* host, int width);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s activation in `L.gu_act` format and write row ent_dst[e]
/// of `out` (n_embd floats).  The hidden activation between GU and DOWN is quantized in `L.d_act`.  Counts are
/// read on the device.  The input and scratch pointers are byte-addressed because Q8_K and Q8_0 have different
/// block sizes from q8_1. `x_stride` is the byte distance between token rows; it may include alignment padding
/// (the verifier's mapped Q8_K rows are 2,928 bytes apart although their payload is 2,920 bytes).
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, int64_t x_stride, void* scratch, float* out,
                           void* stream);
/// Entries [grp_start[0], grp_start[*n_groups]): entry e's row ent_dst[e] of `rows` (n_embd floats each,
/// n_embd % 4 == 0) to the same row of `*out`, at most `max_rows` rows, in 16-byte stores: rows bound for mapped host
/// memory cross PCIe in full transactions (the grouped kernels' one-float stores there cost ~0.2 ms a layer).  Then,
/// unless `*flag` is null, `**flag = *ring` once every row is visible to the host.  Counts and pointers are read on
/// the device; `counter` (device, zero) is left zero.
void native_expert_rows_out(const float* rows, const int32_t* n_groups, const int32_t* grp_start,
                            const int32_t* ent_dst, int64_t n_embd, int64_t max_rows, float* const* out,
                            uint32_t* const* flag, const int32_t* ring, unsigned* counter, void* stream);

}  // namespace lamina::kernels
