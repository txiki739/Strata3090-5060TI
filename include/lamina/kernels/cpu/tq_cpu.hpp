// include/lamina/kernels/cpu/tq_cpu.hpp - Gyro-L's pair-trellis experts (TQP5, TQP6, TQP7: third_party/ggml-tq) on the CPU,
// against the q8_1 activations the GPUs take (block_q8_1: fp16 scale, fp16 sum, 32 x int8).
//
// THE DECODE: a 32768-entry table maps a state straight to its two weights as int16 at scale 2^13, rounded half away from
// zero as the fork's ggml-cpu table is: exact for codebook values of 2^-3 and up, within 2^-14 below.
//
// THE ARITHMETIC, the same on every ISA, so every path gives the same bits at any token-group size:
//   S[i][k] = sum_j w16[i][32k + j] * q8[i][k][j]        exact in int32
//   acc[k]  = fma((float) S[i][k], d_w[i] * d_a[i][k], acc[k])      for blocks i in order, k = 0..3
//   dot     = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * 2^-13
// It differs from the fork's ggml-cpu vec_dot (the same integers, Q8_0 activations, another float order) by rounding.
//
// The kernels: AVX-512 or AVX2 where the CPU runs them (cpu_avx512_ok, cpu_avx2_ok), else the scalar reference;
// LAMINA_TQ_ISA=scalar|avx2|avx512 caps the choice (tests, A/B).
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::kernels::cpu {

inline bool tq_supported(int ggml_type) noexcept { return ggml_type >= 147 && ggml_type <= 149; }
/// Bits per step: 5, 6, 7 (0 for any other type).
inline int tq_kbits(int ggml_type) noexcept { return tq_supported(ggml_type) ? ggml_type - 142 : 0; }
/// Values per block of the trellis types, and bytes per block (2 + 8 * bits: 42, 50, 58).
inline constexpr int kTqBlock = 128;
inline size_t tq_block_bytes(int ggml_type) noexcept { return (size_t) (2 + 8 * tq_kbits(ggml_type)); }
inline constexpr int kTqMaxTokens = 8;

/// Rows [r0, r1) of a trellis matrix (`row_bytes` per row, `n` weights per row) against `nt` (<= 8) q8_1 activations of n
/// values: out[t][r] = w_r . a[t].
void tq_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
             int r0, int r1);
/// Gate and up rows [r0, r1) (gate at blob, up at blob + up_off): ff[t][r] = silu(g) * u, native_gu_rows' SwiGLU.
void tq_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                float* const* ff, int r0, int r1);
/// One row's `n` weights as floats (d * the fp16 codebook, what the fork's ggml reads the type as).
void tq_dequant_row(int ggml_type, const uint8_t* row, int n, float* out);

namespace tq_detail {
/// A token's q8_1 activation as the kernels read it: the int8 values, and the scales as floats.
struct TqAct {
    const uint8_t* blocks;       ///< block_q8_1 rows
    const float* d;              ///< one scale per block of 32 values
};
/// The 32768-entry state table: weight 0 in the low 16 bits, weight 1 in the high ones.
const uint32_t* tq_state_table();
void tq_rows_scalar(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0,
                    int r1);
void tq_rows_avx2(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0,
                  int r1);
void tq_rows_avx512(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0,
                    int r1);
}  // namespace tq_detail

}  // namespace lamina::kernels::cpu
