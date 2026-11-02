// src/kernels/cuda/iq_kernels.cu - see include/lamina/kernels/iq_kernels.hpp.
//
// The dequantizers and the q8_1 quantizer are transcribed from llama.cpp (ggml/src/ggml-cuda/dequantize.cuh,
// quantize.cu at the commit in third_party/ggml/VERSION.txt; MIT license, third_party/ggml/LICENSE), the dot products
// in iq_dot.cuh.
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/q8_1_finite.hpp"
#include "iq_dot.cuh"
#include "fwht.cuh"
#include "q8_1_il.cuh"
#include "lamina/kernels/expf_contract.hpp"
#include "lamina/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>

namespace lamina::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// mmvq_kernel for NC columns from their interleaved (position-major) copy, or one from its q8_1 blocks: a warp takes R
// rows, decodes each call's weights once and keeps row_dot's lane-strided sum per (row, column), so every value is
// bitwise mmvq_kernel's.
template <int NC, int R>
__global__ void __launch_bounds__(128) mmvq_il_iq3_s_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                            const void* __restrict__ xq, const float* __restrict__ xd,
                                                            float* __restrict__ y, int n_in, int n_out) {
    using F = Fmt<21>;
    const int lane = threadIdx.x & 31;
    const auto x = q8_1_cols<NC, true>(xq, xd, n_in / 32);
    const int nb = n_in / F::qk;
    // a grid-stride loop, though the grid gives each warp one group: native_mmvq_il_kernel's form, which its
    // rows-a-warp table is measured on
    for (int grp = blockIdx.x * 4 + (threadIdx.x >> 5); grp * R < n_out; grp += gridDim.x * 4) {
        const int row0 = grp * R;
        float acc[R][NC];
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) acc[i][c] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = min(row0 + i, n_out - 1);   // a partial last group recomputes its last row
                IQ3SCols wv;
                wv.load(w + (size_t) row * row_bytes, kbx, iqs);
                float o[NC];
                wv.apply<NC>(x, kbx * (F::qk / 32) + iqs / 2, o);
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[i][c] += o[c];
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const float s = warp_sum(acc[i][c]);
                if (lane == 0 && row0 + i < n_out) y[(size_t) c * n_out + row0 + i] = s;
            }
    }
}

// ---------------------------------------------------------------- grouped native experts
// Two kernels a call.  The first computes a group's gate and up rows for one chunk of 32 rows of n_ff - one q8_1 block
// of h - SwiGLU and that block for each entry; the second the down rows.  A warp takes several rows at once, one
// lane-strided accumulator each, so every value's sums and their order are `row_dot`'s.
constexpr int GUS_ROWS = 2, GUS_WARPS = 16;   // gate and up rows per warp; 8 warps x 4 rows = one q8_1 block
constexpr int GUS_BATCH = 8;                 // entries whose block of h is staged at once
constexpr int DOWN_ROWS = 4, DOWN_WARPS = 8;

// The K-quants' gate/up rows need more registers than the i-quants': with one block per SM they do not spill (Q4_K, Q5_K
// 15-17% faster at 2-4 tokens), while the i-quants run best at the occupancy the compiler picks (0: no bound).
template<int T> constexpr int gus_min_blocks() { return T == 12 || T == 13 || T == 14 ? 1 : 0; }

template<int TG>
__global__ void __launch_bounds__(GUS_WARPS * 32, gus_min_blocks<TG>()) native_gus_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                    const int32_t* __restrict__ grp_start,
                                                                    const int32_t* __restrict__ n_groups,
                                                                    const int32_t* __restrict__ ent_tok,
                                                                    const block_q8_1* __restrict__ xq,
                                                                    NativeExpertLayout L, block_q8_1* __restrict__ hq) {
    using F = Fmt<TG>;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    __shared__ float sh[GUS_BATCH][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.x;
    const uint8_t* wg = (const uint8_t*) grp_ptr[g] + (size_t) (32 * c + GUS_ROWS * warp) * L.gu_row;
    const uint8_t* wu = wg + L.up_off;
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int b0 = e0; b0 < e1; b0 += GUS_BATCH) {
        const int b1 = min(e1, b0 + GUS_BATCH);
        for (int e = b0; e < b1; ++e) {
            const block_q8_1* x = xq + (size_t) ent_tok[e] * xb;
            float sg[GUS_ROWS], su[GUS_ROWS];
#pragma unroll
            for (int i = 0; i < GUS_ROWS; ++i) sg[i] = su[i] = 0.0f;
            for (int k = lane; k < nb * F::ipb; k += 32) {
                const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
                const block_q8_1* xk = x + kbx * (F::qk / 32);
#pragma unroll
                for (int i = 0; i < GUS_ROWS; ++i) {
                    sg[i] += F::dot(wg + (size_t) i * L.gu_row, xk, kbx, iqs);
                    su[i] += F::dot(wu + (size_t) i * L.gu_row, xk, kbx, iqs);
                }
            }
#pragma unroll
            for (int i = 0; i < GUS_ROWS; ++i) {
                const float gv = warp_sum(sg[i]), uv = warp_sum(su[i]);
                if (lane == 0) sh[e - b0][GUS_ROWS * warp + i] = (gv / (1.0f + __expf(-gv))) * uv;
            }
        }
        __syncthreads();
        for (int e = b0 + warp; e < b1; e += GUS_WARPS) {   // an entry's block of h, as quantize_q8_1_kernel
            const float xi = sh[e - b0][lane];
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp
            const int8_t q = q8_1_quant(xi, d, amax);
            block_q8_1* y = hq + (size_t) e * hb + c;
            y->qs[lane] = q;
            if (lane == 0) y->ds = q8_1_ds(d, sum);
        }
        __syncthreads();
    }
}

template<int TD>
__global__ void __launch_bounds__(DOWN_WARPS * 32) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                      const int32_t* __restrict__ grp_start,
                                                                      const int32_t* __restrict__ n_groups,
                                                                      const int32_t* __restrict__ ent_dst,
                                                                      const block_q8_1* __restrict__ hq,
                                                                      NativeExpertLayout L, float* __restrict__ out) {
    using F = Fmt<TD>;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * DOWN_WARPS + warp) * DOWN_ROWS;
    const uint8_t* wr = (const uint8_t*) grp_ptr[g] + L.down_off + (size_t) r0 * L.d_row;
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; ++e) {
        const block_q8_1* x = hq + (size_t) e * hb;
        float sd[DOWN_ROWS];
#pragma unroll
        for (int i = 0; i < DOWN_ROWS; ++i) sd[i] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const block_q8_1* xk = x + kbx * (F::qk / 32);
#pragma unroll
            for (int i = 0; i < DOWN_ROWS; ++i) sd[i] += F::dot(wr + (size_t) i * L.d_row, xk, kbx, iqs);
        }
        float v[DOWN_ROWS];
#pragma unroll
        for (int i = 0; i < DOWN_ROWS; ++i) v[i] = warp_sum(sd[i]);
        if (lane == 0) {
            float* o = out + (size_t) ent_dst[e] * L.n_embd + r0;
            if constexpr (DOWN_ROWS == 4) *(float4*) o = make_float4(v[0], v[1], v[2], v[3]);
            else if constexpr (DOWN_ROWS == 2) *(float2*) o = make_float2(v[0], v[1]);
            else
#pragma unroll
                for (int i = 0; i < DOWN_ROWS; ++i) o[i] = v[i];
        }
    }
}

// ---------------------------------------------------------------- the same, decoded once
// The two kernels above for the formats with a Split (iq_dot.cuh) and rows of NS calls: a row takes G lanes instead of
// the warp, with the same values bitwise.  Lane t of a G-lane group stands for the warp's lanes t + G j: it sums each
// one's calls t + G j, t + G j + 32, ... in row_dot's order and folds those sums in warp_sum's order (its xor 16 .. G
// steps in registers, the rest by shuffles within the group).  A call's weights are decoded once for NE entries x[c],
// and an entry's activation block is fetched once for the R rows w[r]; `lut` is the format's codebook.
template<int TY, int G, int NS, int R, int NE>
__device__ __forceinline__ void rows_sub(const uint8_t* const (&w)[R], const block_q8_1* const (&x)[NE], int t,
                                         const void* lut, float (&out)[R][NE]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
    constexpr int J = 32 / G;
    float acc[J][R][NE];
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < NE; ++c) acc[j][r][c] = 0.0f;
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
        for (int k0 = G * j; k0 < NS; k0 += 32) {
            const int k = k0 + t;
            if (NS % G != 0 && k >= NS) continue;
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            typename S::W wv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) wv[r] = S::load(w[r], kbx, iqs, lut);
#pragma unroll
            for (int c = 0; c < NE; ++c) {
                const typename S::A a = S::fetch(x[c] + kbx * (F::qk / 32), iqs);
#pragma unroll
                for (int r = 0; r < R; ++r) acc[j][r][c] += S::apply(wv[r], a);
            }
        }
#pragma unroll
    for (int h = J / 2; h >= 1; h /= 2)
#pragma unroll
        for (int j = 0; j < h; ++j)
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int c = 0; c < NE; ++c) acc[j][r][c] = __fadd_rn(acc[j][r][c], acc[j + h][r][c]);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NE; ++c) {
            float v = acc[0][r][c];
#pragma unroll
            for (int o = G / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            out[r][c] = v;
        }
}

// The gate and up rows of h's column `col` for the NE entries from `tok` on, SwiGLU into sh[c][col]
template<int TG, int NE>
__device__ __forceinline__ void gus80_pass(const uint8_t* wg, const uint8_t* wu, const block_q8_1* __restrict__ xq,
                                           const int32_t* __restrict__ tok, int t, const void* lut, float (*sh)[32],
                                           int col) {
    const uint8_t* const w[2] = {wg, wu};
    const block_q8_1* x[NE];
#pragma unroll
    for (int c = 0; c < NE; ++c) x[c] = xq + (size_t) tok[c] * 80;
    float v[2][NE];
    rows_sub<TG, 16, 80, 2, NE>(w, x, t, lut, v);
    if (t == 0)
#pragma unroll
        for (int c = 0; c < NE; ++c) sh[c][col] = (v[0][c] / (1.0f + __expf(-v[0][c]))) * v[1][c];
}

// native_gus_kernel for gate/up rows of 80 calls (n_embd 2560): a half-warp takes a column of h (its gate and up rows),
// two entries a pass at two blocks an SM (64 registers; four entries took 128, one block an SM, 2-17% slower at 1-2
// tokens), the codebook copied to shared memory (its scattered lookups 5-15% faster there).
template<int TG>
__global__ void __launch_bounds__(GUS_WARPS * 32, 2) native_gus80_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                         const int32_t* __restrict__ grp_start,
                                                                         const int32_t* __restrict__ n_groups,
                                                                         const int32_t* __restrict__ ent_tok,
                                                                         const block_q8_1* __restrict__ xq,
                                                                         NativeExpertLayout L,
                                                                         block_q8_1* __restrict__ hq) {
    using S = Split<TG>;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    __shared__ float sh[GUS_BATCH][32];
    __shared__ __align__(16) uint32_t s_grid[S::grid_bytes > 0 ? S::grid_bytes / 4 : 1];
    const void* lut = nullptr;
    if constexpr (S::grid_bytes > 0) {
        for (int i = threadIdx.x; i < S::grid_bytes / 4; i += blockDim.x) s_grid[i] = ((const uint32_t*) S::grid())[i];
        lut = s_grid;
        __syncthreads();
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.x;
    const int col = GUS_ROWS * warp + (lane >> 4), t = lane & 15;
    const uint8_t* wg = (const uint8_t*) grp_ptr[g] + (size_t) (32 * c + col) * L.gu_row;
    const uint8_t* wu = wg + L.up_off;
    const int hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int b0 = e0; b0 < e1; b0 += GUS_BATCH) {
        const int b1 = min(e1, b0 + GUS_BATCH);
        for (int e = b0; e < b1; e += 2) {
            if (b1 - e == 1) gus80_pass<TG, 1>(wg, wu, xq, ent_tok + e, t, lut, sh + (e - b0), col);
            else gus80_pass<TG, 2>(wg, wu, xq, ent_tok + e, t, lut, sh + (e - b0), col);
        }
        __syncthreads();
        for (int e = b0 + warp; e < b1; e += GUS_WARPS) {   // as native_gus_kernel
            const float xi = sh[e - b0][lane];
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = q8_1_finite(amax / 127.0f);
            const int8_t q = q8_1_quant(xi, d, amax);
            block_q8_1* y = hq + (size_t) e * hb + c;
            y->qs[lane] = q;
            if (lane == 0) y->ds = q8_1_ds(d, sum);
        }
        __syncthreads();
    }
}

// The down row r for the NE entries from `e` on
template<int TD, int G, int NS, int NE>
__device__ __forceinline__ void down_pass(const uint8_t* wr, const block_q8_1* __restrict__ hq, int hb, int e, int t,
                                          const int32_t* __restrict__ ent_dst, int64_t n_embd, int r, const void* lut,
                                          float* __restrict__ out) {
    const uint8_t* const w[1] = {wr};
    const block_q8_1* x[NE];
#pragma unroll
    for (int c = 0; c < NE; ++c) x[c] = hq + (size_t) (e + c) * hb;
    float v[1][NE];
    rows_sub<TD, G, NS, 1, NE>(w, x, t, lut, v);
    if (t == 0)
#pragma unroll
        for (int c = 0; c < NE; ++c) out[(size_t) ent_dst[e + c] * n_embd + r] = v[0][c];
}

// native_down_kernel for down rows of NS calls (n_ff 640): G lanes a row, DOWN_WARPS * 32 / G rows a block, up to four
// entries a pass (two or one: 1-11% slower)
template<int TD, int G, int NS>
__global__ void __launch_bounds__(DOWN_WARPS * 32) native_down_sub_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                          const int32_t* __restrict__ grp_start,
                                                                          const int32_t* __restrict__ n_groups,
                                                                          const int32_t* __restrict__ ent_dst,
                                                                          const block_q8_1* __restrict__ hq,
                                                                          NativeExpertLayout L, float* __restrict__ out) {
    using S = Split<TD>;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    __shared__ __align__(16) uint32_t s_grid[S::grid_bytes > 0 ? S::grid_bytes / 4 : 1];
    const void* lut = nullptr;
    if constexpr (S::grid_bytes > 0) {   // the codebook in shared memory, as native_gus80_kernel
        for (int i = threadIdx.x; i < S::grid_bytes / 4; i += blockDim.x) s_grid[i] = ((const uint32_t*) S::grid())[i];
        lut = s_grid;
        __syncthreads();
    }
    const int lane = threadIdx.x & 31, t = lane & (G - 1);
    const int r = (blockIdx.x * DOWN_WARPS + (threadIdx.x >> 5)) * (32 / G) + lane / G;
    const uint8_t* wr = (const uint8_t*) grp_ptr[g] + L.down_off + (size_t) r * L.d_row;
    const int hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += 4) {
        switch (min(4, e1 - e)) {
            case 1: down_pass<TD, G, NS, 1>(wr, hq, hb, e, t, ent_dst, L.n_embd, r, lut, out); break;
            case 2: down_pass<TD, G, NS, 2>(wr, hq, hb, e, t, ent_dst, L.n_embd, r, lut, out); break;
            case 3: down_pass<TD, G, NS, 3>(wr, hq, hb, e, t, ent_dst, L.n_embd, r, lut, out); break;
            default: down_pass<TD, G, NS, 4>(wr, hq, hb, e, t, ent_dst, L.n_embd, r, lut, out); break;
        }
    }
}

// ---------------------------------------------------------------- Hadamard-folded experts (hadamard.hpp)
// h is rotated over 128 of its n_ff values at a time, more than a block of the kernels above holds, so a layer takes three
// kernels: the gate and up rows of 160 calls (n_embd 2560), h's floats by column; h rotated and quantized, an entry's 128 values
// a warp; the down rows (native_down_sub_kernel).

// The gate and up rows of h's column `col` for the NE entries from `e` on, SwiGLU into their rows of h
template<int TG, int NE>
__device__ __forceinline__ void gu_h_pass(const uint8_t* wg, const uint8_t* wu, const block_q8_1* __restrict__ xq,
                                          const int32_t* __restrict__ tok, int t, const void* lut, float* h, int n_ff) {
    const uint8_t* const w[2] = {wg, wu};
    const block_q8_1* x[NE];
#pragma unroll
    for (int c = 0; c < NE; ++c) x[c] = xq + (size_t) tok[c] * 80;
    float v[2][NE];
    rows_sub<TG, 16, 160, 2, NE>(w, x, t, lut, v);
    if (t == 0)
#pragma unroll
        for (int c = 0; c < NE; ++c) h[(size_t) c * n_ff] = (v[0][c] / (1.0f + __expf(-v[0][c]))) * v[1][c];
}

template<int TG>
__global__ void __launch_bounds__(GUS_WARPS * 32, 2) native_gu_h_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                        const int32_t* __restrict__ grp_start,
                                                                        const int32_t* __restrict__ n_groups,
                                                                        const int32_t* __restrict__ ent_tok,
                                                                        const block_q8_1* __restrict__ xq,
                                                                        NativeExpertLayout L, float* __restrict__ hf) {
    using S = Split<TG>;
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    __shared__ __align__(16) uint32_t s_grid[S::grid_bytes / 4];
    for (int i = threadIdx.x; i < S::grid_bytes / 4; i += blockDim.x) s_grid[i] = ((const uint32_t*) S::grid())[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.x;
    const int col = 32 * c + GUS_ROWS * warp + (lane >> 4), t = lane & 15;
    const uint8_t* wg = (const uint8_t*) grp_ptr[g] + (size_t) col * L.gu_row;
    const uint8_t* wu = wg + L.up_off;
    const int n_ff = (int) L.n_ff;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += 2) {
        float* h = hf + (size_t) e * n_ff + col;
        if (e1 - e == 1) gu_h_pass<TG, 1>(wg, wu, xq, ent_tok + e, t, s_grid, h, n_ff);
        else gu_h_pass<TG, 2>(wg, wu, xq, ent_tok + e, t, s_grid, h, n_ff);
    }
}

// The plan's entries' h: 128 values a warp, signs and rotation in registers, each register a q8_1 block
__global__ void __launch_bounds__(256) h_rotate_q8_1_kernel(const float* __restrict__ hf, block_q8_1* __restrict__ hq,
                                                            const float* __restrict__ signs, float scale,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups, int n_ff) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, parts = n_ff / 128;
    const int i = blockIdx.x * 8 + warp, e = grp_start[0] + i / parts, b = i % parts;
    if (e >= grp_start[*n_groups]) return;
    const float* h = hf + (size_t) e * n_ff + b * 128;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int at = j * 32 + lane;
        v[j] = signs != nullptr ? h[at] * signs[b * 128 + at] : h[at];
    }
    fwht128_warp(v, lane, scale);
    block_q8_1* y = hq + (size_t) e * (n_ff / 32) + b * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float amax = fabsf(v[j]), sum = v[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
        }
        const float d = q8_1_finite(amax / 127.0f);
        y[j].qs[lane] = q8_1_quant(v[j], d, amax);
        if (lane == 0) y[j].ds = q8_1_ds(d, sum);
    }
}

// hadamard_rows: a warp a row's 128 values
template<typename T> __device__ __forceinline__ float had_ld(const T* p) { return *p; }
template<> __device__ __forceinline__ float had_ld<__half>(const __half* p) { return __half2float(*p); }
template<typename T> __device__ __forceinline__ void had_st(T* p, float v) { *p = v; }
template<> __device__ __forceinline__ void had_st<__half>(__half* p, float v) { *p = __float2half(v); }
template<typename T>
__global__ void __launch_bounds__(256) hadamard_rows_kernel(const T* x, T* y, const float* __restrict__ signs, float scale,
                                                            int64_t rows, int64_t width) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t i = (int64_t) blockIdx.x * 8 + warp, parts = width / 128, row = i / parts;
    if (row >= rows) return;
    const int b = (int) (i % parts);
    const int64_t r0 = row * width + b * 128;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int at = j * 32 + lane;
        v[j] = signs != nullptr ? had_ld<T>(x + r0 + at) * signs[b * 128 + at] : had_ld<T>(x + r0 + at);
    }
    fwht128_warp(v, lane, scale);
#pragma unroll
    for (int j = 0; j < 4; ++j) had_st<T>(y + r0 + j * 32 + lane, v[j]);
}

// The decoded-once kernels' formats: gate/up rows of 80 calls at n_embd 2560; down rows (n_ff 640) of IQ4_NL (40 calls,
// 8 lanes a row; 4 or 16: 3-6% slower) and Q2_0 (20 calls, 4 lanes; 8 or 16: up to 14% slower at 8 tokens)
#define LAMINA_GU80_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(42)
bool gu80(const NativeExpertLayout& L) {
    switch (L.gu_type) {
#define LAMINA_GU80(T) case T:
        LAMINA_GU80_FMTS(LAMINA_GU80)
#undef LAMINA_GU80
            return L.n_embd == 2560;
        default: return false;
    }
}
constexpr int DOWN_G_IQ4_NL = 8, DOWN_G_Q2_0 = 4;
int down_sub_g(const NativeExpertLayout& L) {
    const int G = L.n_ff != 640 ? 0 : L.d_type == 20 ? DOWN_G_IQ4_NL : L.d_type == 42 ? DOWN_G_Q2_0 : 0;
    return G > 0 && L.n_embd % (DOWN_WARPS * 32 / G) == 0 ? G : 0;
}

__global__ void rows_out_kernel(const float4* __restrict__ src, const int32_t* __restrict__ n_groups,
                                const int32_t* __restrict__ grp_start, const int32_t* __restrict__ ent_dst,
                                long long n4, float4* const* out, uint32_t* const* flag, const int32_t* ring,
                                unsigned* counter) {
    const int e0 = grp_start[0];
    const long long total = (long long) (grp_start[*n_groups] - e0) * n4, step = (long long) gridDim.x * blockDim.x;
    float4* o = *out;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += step) {
        const long long at = (long long) ent_dst[e0 + i / n4] * n4 + i % n4;
        o[at] = src[at];
    }
    uint32_t* f = *flag;
    if (f == nullptr) return;
    __shared__ bool last;
    __threadfence_system();   // this thread's rows before the block counts as done
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (last && threadIdx.x == 0) {
        *counter = 0u;   // for the next launch
        *(volatile uint32_t*) f = (uint32_t) *ring;
    }
}

// Activation-contract-aware native expert kernels: --expert-math exact.  The kernels above are --expert-math fast
// (llama.cpp's q8_1 contract, every format).  NativeFmt's Q4_K/Q5_K gate/up layers use Q8_K rows, while Q8_0 down
// layers use Q8_0 hidden rows; these kernels keep those formats all the way through the grouped hit path.
__device__ __forceinline__ int native_q8_0_contract_code(float xi, float amax) {
    const float inv = amax == 0.0f ? 0.0f : __fdiv_rn(127.0f, amax);
    const float z = amax == 0.0f ? 0.0f : __fmul_rn(xi, inv);
    const int q = amax == 0.0f ? 0 : __float2int_rn(z);
    return max(-127, min(127, q));
}

template<int TG>
__global__ void __launch_bounds__(GUS_WARPS * 32) native_gus_contract_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_tok, const void* __restrict__ xq,
    int64_t x_stride, NativeExpertLayout L, void* __restrict__ hq) {
    using F = Fmt<TG>;
    static_assert(TG == 12 || TG == 13, "native Q8_K GU is only defined for K-quants");
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    __shared__ float sh[GUS_BATCH][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = blockIdx.x;
    const uint8_t* wg = (const uint8_t*) grp_ptr[g] + (size_t) (32 * c + GUS_ROWS * warp) * L.gu_row;
    const uint8_t* wu = wg + L.up_off;
    const int nb = (int) (L.n_embd / F::qk), hb = (int) (L.n_ff / 32);
    const size_t block_stride = L.gu_row / (size_t) nb;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int b0 = e0; b0 < e1; b0 += GUS_BATCH) {
        const int b1 = min(e1, b0 + GUS_BATCH);
        for (int e = b0; e < b1; ++e) {
            const block_q8_K* x = (const block_q8_K*) ((const uint8_t*) xq + (size_t) ent_tok[e] * (size_t) x_stride);
            // The warp's four dots at once, one per eight-lane group: group q takes row q >> 1 of the gate (q even)
            // or the up (q odd) matrix, each in the CPU's own order (row_dot_k_q8_k_cpu_avx2), so the values are
            // bitwise those of the four dots in turn.
            static_assert(GUS_ROWS == 2, "four eight-lane groups: the gate and up dots of two rows");
            const int q = lane >> 3;
            const float dv = row_dot_k_q8_k_stride<TG == 13>((q & 1 ? wu : wg) + (size_t) (q >> 1) * L.gu_row,
                                                             block_stride, x, nb, lane, L.q5k_fma);
            float sg[GUS_ROWS], su[GUS_ROWS];
            sg[0] = __shfl_sync(0xffffffffu, dv, 0);
            su[0] = __shfl_sync(0xffffffffu, dv, 8);
            sg[1] = __shfl_sync(0xffffffffu, dv, 16);
            su[1] = __shfl_sync(0xffffffffu, dv, 24);
            if (lane == 0) {
#pragma unroll
                for (int i = 0; i < GUS_ROWS; ++i) {
                    const float gd = sg[i], ud = su[i];
                    const float ee = expf_contract(-gd);
                    const float den = __fadd_rn(1.0f, ee);
                    const float sil = __fdiv_rn(gd, den);
                    sh[e - b0][GUS_ROWS * warp + i] = __fmul_rn(sil, ud);
                }
            }
        }
        __syncthreads();
        for (int e = b0 + warp; e < b1; e += GUS_WARPS) {
            const float xi = sh[e - b0][lane];
            float amax = fabsf(xi);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            if (L.d_act == 8) {
                uint8_t* y = (uint8_t*) hq + (size_t) e * (size_t) hb * 34 + (size_t) c * 34;
                const float d32 = amax / 127.0f;
                if (lane == 0) {
                    const uint16_t bits = f16_from_f32(d32);
                    y[0] = (uint8_t) bits; y[1] = (uint8_t) (bits >> 8);
                }
                const int q = native_q8_0_contract_code(xi, amax);
                y[2 + lane] = (uint8_t) (int8_t) q;
            } else {
                uint8_t* y = (uint8_t*) hq + (size_t) e * (size_t) hb * 36 + (size_t) c * 36;
                const float d32 = __fdiv_rn(amax, 127.0f);
                // ggml's AVX2 quantize_row_q8_1 computes the reciprocal once,
                // multiplies by 127/max, and converts with _MM_ROUND_NEAREST
                // (ties-to-even).  Division plus roundf is a different contract
                // at half-integer thresholds and changes the hidden bytes.
                const float id = amax == 0.0f ? 0.0f : __fdiv_rn(127.0f, amax);
                const float z = amax == 0.0f ? 0.0f : __fmul_rn(xi, id);
                const int q = amax == 0.0f ? 0 : __float2int_rn(z);
                int sumq = q;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) sumq += __shfl_xor_sync(0xffffffffu, sumq, o);
                if (lane == 0)
                    *reinterpret_cast<half2*>(y) = __halves2half2(__float2half(d32), __float2half(__fmul_rn((float) sumq, d32)));
                y[4 + lane] = (uint8_t) (int8_t) q;
            }
        }
        __syncthreads();
    }
}

// native_exact_dot_rows: one row per eight-lane group, as the exact kernels take them
__global__ void exact_dot_rows_kernel(int type, const uint8_t* __restrict__ rows, size_t row_bytes, int n_rows,
                                      const void* __restrict__ x, int nb, bool fma_mins, float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int r = (int) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 4 + (lane >> 3);
    const uint8_t* row = rows + (size_t) min(r, n_rows - 1) * row_bytes;   // every lane takes part in the shuffles
    const size_t bs = row_bytes / (size_t) nb;
    float v = 0.0f;
    if (type == 12) v = row_dot_k_q8_k_stride<false>(row, bs, (const block_q8_K*) x, nb, lane, fma_mins);
    else if (type == 13) v = row_dot_k_q8_k_stride<true>(row, bs, (const block_q8_K*) x, nb, lane, fma_mins);
    else if (type == 8) v = row_dot_q8_0_q8_0(row, bs, (const block_q8_0*) x, nb, lane);
    else if (type == 7) v = row_dot_q5_1_q8_1_cpu_avx2(row, (const block_q8_1*) x, nb, lane);
    if ((lane & 7) == 0 && r < n_rows) out[r] = v;
}

__global__ void native_q8_0_contract_probe_kernel(const float* x, uint8_t* y, int nblocks) {
    const int b = blockIdx.x;
    if (b >= nblocks || threadIdx.x != 0) return;
    const float* xb = x + (size_t) b * 32;
    uint8_t* out = y + (size_t) b * 34;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
    const float d32 = amax / 127.0f;
    const uint16_t bits = f16_from_f32(d32);
    out[0] = (uint8_t) bits; out[1] = (uint8_t) (bits >> 8);
    for (int i = 0; i < 32; ++i)
        out[2 + i] = (uint8_t) (int8_t) native_q8_0_contract_code(xb[i], amax);
}

template<int TD>
__global__ void __launch_bounds__(DOWN_WARPS * 32) native_down_contract_kernel(
    const unsigned long long* __restrict__ grp_ptr, const int32_t* __restrict__ grp_start,
    const int32_t* __restrict__ n_groups, const int32_t* __restrict__ ent_dst, const void* __restrict__ hq,
    NativeExpertLayout L, float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * DOWN_WARPS + warp) * DOWN_ROWS;
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nb = L.d_act == 8 ? (int) (L.n_ff / 32) : L.d_act == 15 ? (int) (L.n_ff / 256)
                                                                         : (int) (L.n_ff / 32);
    const size_t block_stride = L.d_row / (size_t) nb;
    const int hb = L.d_act == 8 ? (int) (L.n_ff / 32) : L.d_act == 15 ? (int) (L.n_ff / 256)
                                                                        : (int) (L.n_ff / 32);
    for (int e = e0; e < e1; ++e) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g] + L.down_off;
        float v[DOWN_ROWS];
        if (L.d_act == 9 && TD != 7) {
            // a q8_1 hidden with no CPU-order dot: llama.cpp's lane-strided one, the whole warp on each row
            const block_q8_1* x = (const block_q8_1*) ((const uint8_t*) hq + (size_t) e * hb * 36);
#pragma unroll
            for (int i = 0; i < DOWN_ROWS; ++i) v[i] = row_dot<TD>(blob + (size_t) (r0 + i) * L.d_row, x, nb, lane);
        } else {
            // The CPU-order dots, the warp's four rows at once: eight-lane group q takes row r0 + q, so each row's
            // value is bitwise the one-row-per-warp result (see row_dot_q5_1_q8_1_cpu_avx2).
            static_assert(DOWN_ROWS == 4, "four eight-lane groups: one row each");
            const uint8_t* wrow = blob + (size_t) (r0 + (lane >> 3)) * L.d_row;
            float vq = 0.0f;
            if (L.d_act == 8) {
                const block_q8_0* x = (const block_q8_0*) ((const uint8_t*) hq + (size_t) e * hb * 34);
                vq = row_dot_q8_0_q8_0(wrow, block_stride, x, nb, lane);
            } else if (L.d_act == 15) {
                const block_q8_K* x = (const block_q8_K*) ((const uint8_t*) hq + (size_t) e * hb * 292);
                if constexpr (TD == 12 || TD == 13)
                    vq = row_dot_k_q8_k_stride<TD == 13>(wrow, block_stride, x, nb, lane, L.q5k_fma);
            } else if constexpr (TD == 7) {
                const block_q8_1* x = (const block_q8_1*) ((const uint8_t*) hq + (size_t) e * hb * 36);
                vq = row_dot_q5_1_q8_1_cpu_avx2(wrow, x, nb, lane);
            }
#pragma unroll
            for (int i = 0; i < DOWN_ROWS; ++i) v[i] = __shfl_sync(0xffffffffu, vq, 8 * i);
        }
        if (lane == 0) {
            float* o = out + (size_t) ent_dst[e] * L.n_embd + r0;
            if constexpr (DOWN_ROWS == 4) *(float4*) o = make_float4(v[0], v[1], v[2], v[3]);
            else if constexpr (DOWN_ROWS == 2) *(float2*) o = make_float2(v[0], v[1]);
            else
#pragma unroll
                for (int i = 0; i < DOWN_ROWS; ++i) o[i] = v[i];
        }
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xi = x[i];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = __low2half(x[ibs].dm);
    const float dmin = __high2half(x[ibs].dm);
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
// llama.cpp's dequantize_q5_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q5_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = __low2half(x[ibs].dm);
        const float dmin = __high2half(x[ibs].dm);
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
// 32-value blocks: a "superblock" is 8 of them; thread tid writes 8 values of block tid % 8
template<typename dst_t>
__device__ void dq_q5_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float2 dm = __half22float2(x[ib].dm);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x + dm.y);
        y[iqs + 16] = cvt<dst_t>((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dm.x + dm.y);
    }
}
template<typename dst_t>
__device__ void dq_q8_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    dst_t* y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>((float) x[ib].qs[8 * il + j] * d);
}
// llama.cpp's dequantize_q6_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q6_K* x = (const block_q6_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int ip = tt / 32, il = tt - 32 * ip, is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d = __half2float(x[ibs].d);
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0] = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}
template<typename dst_t>
__device__ void dq_iq1_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_s* x = (const block_iq1_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float) x[ibs].d * (2 * ((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[ib] >> 3 * il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}

// TQP: one 256-value call = two 128-blocks; thread tid decodes block tid / 16, lane (tid >> 1) & 7, its steps 4 * half .. + 3
// (8 values, 2 a step), through the fp16 codebook, what the fork's ggml reads the type as
template<int K, typename dst_t>
__device__ void dq_tqp(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const int blk = tid >> 4, l = (tid >> 1) & 7, half = tid & 1;
    const uint8_t* b = (const uint8_t*) vx + (size_t) (2 * ibs + blk) * (2 + 8 * K);
    const float d = __half2float(__ushort_as_half(*(const uint16_t*) b));
    uint32_t s[8];
    tqp_states8<K>(b + 2, l, s);
    dst_t* y = yy + blk * 128 + 16 * l + 8 * half;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t pa = ((const uint32_t*) tq_lut_f16)[(s[4 * half + i] * 0x9e3779b1u) >> 21];
        y[2 * i + 0] = cvt<dst_t>(d * __half2float(__ushort_as_half((unsigned short) (pa & 0xFFFFu))));
        y[2 * i + 1] = cvt<dst_t>(d * __half2float(__ushort_as_half((unsigned short) (pa >> 16))));
    }
}

template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 147: dq_tqp<5>(vx, ibs, y, tid); break;
        case 148: dq_tqp<6>(vx, ibs, y, tid); break;
        case 149: dq_tqp<7>(vx, ibs, y, tid); break;
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 14: dq_q6_k(vx, ibs, y, tid); break;
        case 19: dq_iq1_s(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
__global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y) {
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
}

// several experts at once, expert z's blob at blobs[z]: first its gate/up superblocks (as dequant_gu_kernel), then
// its down superblocks (as dequant_flat_kernel)
__global__ void dequant_experts_kernel(int gu_ty, int d_ty, const uint8_t* const* __restrict__ blobs, size_t up_off,
                                       size_t down_off, int64_t per_row, int64_t gu_blocks, __half* __restrict__ gu,
                                       __half* __restrict__ dn, int64_t gu_elems, int64_t dn_elems) {
    const uint8_t* b = blobs[blockIdx.z];
    const int64_t i = blockIdx.x;
    if (i < 2 * gu_blocks) {
        const int parity = (int) (i / gu_blocks);
        const int64_t j = i % gu_blocks, r = j / per_row, c = j % per_row;
        dq_dispatch<__half>(gu_ty, parity ? b + up_off : b, j,
                            gu + blockIdx.z * gu_elems + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x);
    } else {
        const int64_t j = i - 2 * gu_blocks;
        dq_dispatch<__half>(d_ty, b + down_off, j, dn + blockIdx.z * dn_elems + j * QK_K, threadIdx.x);
    }
}

// the formats dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 14 || t == 19 || t == 7 || t == 8 || (t >= 147 && t <= 149);
}

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 19: return (size_t) (n / 256) * sizeof(block_iq1_s);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 147: return (size_t) (n / 128) * sizeof(block_tqp5);
        case 148: return (size_t) (n / 128) * sizeof(block_tqp6);
        case 149: return (size_t) (n / 128) * sizeof(block_tqp7);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (block_q8_1*) y, n);
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const dim3 grid((unsigned) ((n_out + 3) / 4)), block(32, 4);
    const size_t rb = iq_row_bytes(t, n_in);
    cudaStream_t s = (cudaStream_t) stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define LAMINA_MMVQ(T) case T: mmvq_kernel<T><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols); break;
        LAMINA_FMTS(LAMINA_MMVQ)
#undef LAMINA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

bool iq_mmvq_il_supported(int t) noexcept { return t == 21; }

namespace {
template <int R>
void launch_iq3_s_il(const uint8_t* w, size_t rb, const void* x_q8_1, const void* x_il, float* y, int n_in,
                     int n_out, int ncols, cudaStream_t s) {
    const unsigned blocks = (unsigned) (((n_out + R - 1) / R + 3) / 4);
    if (ncols == 1) {
        mmvq_il_iq3_s_kernel<1, R><<<blocks, 128, 0, s>>>(w, rb, x_q8_1, nullptr, y, n_in, n_out);
        return;
    }
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    switch (ncols) {
#define LAMINA_IL(N) \
    case N: mmvq_il_iq3_s_kernel<N, R><<<blocks, 128, 0, s>>>(w, rb, parts.pm, parts.d, y, n_in, n_out); break;
        LAMINA_IL(2) LAMINA_IL(3) LAMINA_IL(4) LAMINA_IL(5) LAMINA_IL(6) LAMINA_IL(7) LAMINA_IL(8)
#undef LAMINA_IL
    }
}
}  // namespace

void iq_mmvq_il(int t, const void* w, const void* x_q8_1, const void* x_il, float* y, int n_in, int n_out, int ncols,
                void* stream, int rows) {
    if (t != 21 || ncols < 1 || ncols > 8 || n_in % 256 != 0) {
        std::fprintf(stderr, "iq_mmvq_il: type %d, %d columns are not supported\n", t, ncols);
        std::exit(1);
    }
    const size_t rb = iq_row_bytes(t, n_in);
    const auto* W = (const uint8_t*) w;
    cudaStream_t s = (cudaStream_t) stream;
    switch (rows) {
    case 1: launch_iq3_s_il<1>(W, rb, x_q8_1, x_il, y, n_in, n_out, ncols, s); break;
    case 4: launch_iq3_s_il<4>(W, rb, x_q8_1, x_il, y, n_in, n_out, ncols, s); break;
    default: launch_iq3_s_il<2>(W, rb, x_q8_1, x_il, y, n_in, n_out, ncols, s); break;
    }
    check("iq_mmvq_il");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<__half><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, (__half*) dst);
    check("iq_dequant_f16");
}

namespace {
__global__ void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, threadIdx.x);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    embed_rows_kernel<<<dim3((unsigned) (n_embd / 256), (unsigned) n_tok), 32, 0, (cudaStream_t) stream>>>(
        t, (const uint8_t*) table, row_bytes, tokens, n_embd, out);
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<float><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, dst);
    check("iq_dequant_f32");
}

void iq_dequant_experts_f16(int gu_type, int d_type, const uint8_t* const* blobs, int n, size_t up_off, size_t down_off,
                            int64_t n_ff, int64_t n_embd, uint16_t* gu, uint16_t* dn, void* stream) {
    if (n <= 0) return;
    if (n_embd % 256 != 0 || (n_embd * n_ff) % 256 != 0 || !is_iq(gu_type) || !is_iq(d_type) || n > 65535) {
        std::fprintf(stderr, "iq_dequant_experts_f16: bad arguments\n");
        std::exit(1);
    }
    const int64_t per_row = n_embd / 256, gu_blocks = n_ff * per_row, d_blocks = n_embd * n_ff / 256;
    dequant_experts_kernel<<<dim3((unsigned) (2 * gu_blocks + d_blocks), 1, (unsigned) n), 32, 0, (cudaStream_t) stream>>>(
        gu_type, d_type, blobs, up_off, down_off, per_row, gu_blocks, (__half*) gu, (__half*) dn, 2 * n_ff * n_embd,
        n_embd * n_ff);
    check("iq_dequant_experts_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = fmt_qk(gu_type), qd = fmt_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0;
}
namespace {
ExpertMath g_expert_math = ExpertMath::Fast;
bool g_q5k_fma = true;
}  // namespace

void native_exact_dot_rows(int type, const void* rows, size_t row_bytes, int n_rows, const void* x, int64_t n,
                           float* out, void* stream) {
    const int qk = type == 12 || type == 13 ? 256 : 32;
    if (n_rows <= 0 || n % qk != 0 || (type != 7 && type != 8 && type != 12 && type != 13)) {
        std::fprintf(stderr, "native_exact_dot_rows: type %d, %lld values\n", type, (long long) n);
        std::exit(1);
    }
    const unsigned warps = (unsigned) ((n_rows + 3) / 4);
    exact_dot_rows_kernel<<<(warps + 7) / 8, 256, 0, (cudaStream_t) stream>>>(type, (const uint8_t*) rows, row_bytes,
                                                                               n_rows, x, (int) (n / qk), g_q5k_fma, out);
    check("native_exact_dot_rows");
}

void native_q8_0_contract_probe(const float* x, int64_t n, void* y, void* stream) {
    if (n <= 0 || n % 32 != 0) { std::fprintf(stderr, "native_q8_0_contract_probe: n must be a positive multiple of 32\n"); std::exit(1); }
    native_q8_0_contract_probe_kernel<<<(unsigned) (n / 32), 1, 0, (cudaStream_t) stream>>>(x, (uint8_t*) y, (int) (n / 32));
    check("native_q8_0_contract_probe");
}


void set_expert_math(ExpertMath m) noexcept { g_expert_math = m; }
void set_exact_q5_K_fma(bool fused) noexcept { g_q5k_fma = fused; }
ExpertMath expert_math() noexcept { return g_expert_math; }
int gpu_expert_act(int cpu_act) noexcept { return g_expert_math == ExpertMath::Exact ? cpu_act : 9; }

bool native_expert_exact_supported(int gu_type, int d_type, int gu_act, int d_act) noexcept {
    // native_expert_grouped's contract dispatch: Q8_K gate/up kernels, then the contract down kernels
    const bool gu = gu_act == 15 && (gu_type == 12 || gu_type == 13);
    const bool dn = (d_act == 8 && d_type == 8) || (d_act == 9 && d_type == 7) ||
                    (d_act == 15 && (d_type == 12 || d_type == 13));
    return gu && dn;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, int gu_act, int d_act) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    // A negative activation type means an intentional legacy q8_1 caller.
    // Native production paths pass NativeFmt.gu_act/d_act explicitly; do not
    // infer those values here because tests and future formats may use the
    // same weight types with a different contract.
    if (gu_act < 0) gu_act = 9;
    if (d_act < 0) d_act = 9;
    L.gu_act = gu_act;
    L.d_act = d_act;
    L.q5k_fma = g_q5k_fma;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff, bool rotated) {
    const size_t q81 = (size_t) (n_ff / 32) * sizeof(block_q8_1);
    const size_t q80 = (size_t) (n_ff / 32) * 34;
    const size_t q8k = (size_t) (n_ff / 256) * 292;
    const size_t quant = ((size_t) cap * (std::max)(q81, (std::max)(q80, q8k)) + 255) & ~(size_t) 255;
    return rotated ? quant + (size_t) cap * (size_t) n_ff * sizeof(float) : quant;   // the rotated layers' h in floats, after it
}

void hadamard_rows(const float* x, float* y, int64_t rows, int64_t width, const float* signs, void* stream) {
    if (rows <= 0) return;
    const unsigned blocks = (unsigned) ((rows * (width / 128) + 7) / 8);
    hadamard_rows_kernel<float><<<blocks, 256, 0, (cudaStream_t) stream>>>(x, y, signs, 1.0f / std::sqrt(128.0f), rows, width);
    check("hadamard_rows");
}

void hadamard_rows_f16(uint16_t* x, int64_t rows, int64_t width, const float* signs, void* stream) {
    if (rows <= 0) return;
    const unsigned blocks = (unsigned) ((rows * (width / 128) + 7) / 8);
    hadamard_rows_kernel<__half><<<blocks, 256, 0, (cudaStream_t) stream>>>((const __half*) x, (__half*) x, signs,
                                                                            1.0f / std::sqrt(128.0f), rows, width);
    check("hadamard_rows_f16");
}

const float* hadamard_device_signs(const float* host, int width) {
    if (host == nullptr || width <= 0) return nullptr;
    static std::mutex mu;
    static std::map<std::tuple<int, const float*, int>, float*> table;   // (device, host vector, width)
    int dev = 0;
    cudaGetDevice(&dev);
    std::lock_guard<std::mutex> lock(mu);
    float*& d = table[std::make_tuple(dev, host, width)];
    if (d == nullptr && (cudaMalloc((void**) &d, (size_t) width * sizeof(float)) != cudaSuccess ||
                         cudaMemcpy(d, host, (size_t) width * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)) {
        std::fprintf(stderr, "hadamard_device_signs: %s\n", cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
    return d;
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, int64_t x_stride, void* scratch, float* out,
                           void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0 || L.n_embd % (DOWN_WARPS * DOWN_ROWS) != 0) {
        std::fprintf(stderr, "native_expert_grouped: n_ff %lld / n_embd %lld do not divide into the kernels' rows\n",
                     (long long) L.n_ff, (long long) L.n_embd);
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    auto* hq = (block_q8_1*) scratch;
    const auto* X = (const block_q8_1*) x_q8_1;
    const dim3 ggu((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
    if (L.had_block > 0) {   // Hadamard-folded: the trellis formats, at the geometry their kernels take
        if (L.had_block != 128 || L.n_embd != 2560 || L.n_ff != 640) {
            std::fprintf(stderr, "native_expert_grouped: a rotation of %d over %lld x %lld\n", L.had_block, (long long) L.n_embd,
                         (long long) L.n_ff);
            std::exit(1);
        }
        auto* hf = (float*) ((uint8_t*) scratch + native_expert_scratch_bytes(cap_entries, L.n_ff));
        const unsigned parts = (unsigned) (L.n_ff / 128);
        const dim3 gs((unsigned) (L.n_embd / (DOWN_WARPS * 32 / 8)), (unsigned) cap_groups);
#define LAMINA_GU_H(T) case T: native_gu_h_kernel<T><<<ggu, GUS_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hf); break;
        switch (L.gu_type) { LAMINA_TQP_FMTS(LAMINA_GU_H) default: std::fprintf(stderr, "native_expert_grouped: folded gate/up type %d\n", L.gu_type); std::exit(1); }
#undef LAMINA_GU_H
        check("native_expert_grouped/gu");
        h_rotate_q8_1_kernel<<<(unsigned) (((size_t) cap_entries * parts + 7) / 8), 256, 0, s>>>(hf, hq, L.d_signs, 1.0f / std::sqrt(128.0f),
                                                                                                grp_start, n_groups, (int) L.n_ff);
        check("native_expert_grouped/rotate");
#define LAMINA_DOWN_H(T) case T: native_down_sub_kernel<T, 8, 40><<<gs, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        switch (L.d_type) { LAMINA_TQP_FMTS(LAMINA_DOWN_H) default: std::fprintf(stderr, "native_expert_grouped: folded down type %d\n", L.d_type); std::exit(1); }
#undef LAMINA_DOWN_H
        check("native_expert_grouped/down");
        return;
    }
    if (L.gu_act == 15) {
        switch (L.gu_type) {
        case 12: native_gus_contract_kernel<12><<<ggu, GUS_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok,
                                                                                x_q8_1, x_stride, L, scratch); break;
        case 13: native_gus_contract_kernel<13><<<ggu, GUS_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok,
                                                                                x_q8_1, x_stride, L, scratch); break;
        default: std::fprintf(stderr, "native_expert_grouped: Q8_K gate/up type %d\n", L.gu_type); std::exit(1);
        }
    } else if (gu80(L)) {
        switch (L.gu_type) {
#define LAMINA_GU80(T) case T: native_gus80_kernel<T><<<ggu, GUS_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            LAMINA_GU80_FMTS(LAMINA_GU80)
#undef LAMINA_GU80
        }
    } else switch (L.gu_type) {
#define LAMINA_GU(T) case T: native_gus_kernel<T><<<ggu, GUS_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        LAMINA_FMTS(LAMINA_GU)
#undef LAMINA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const dim3 gd((unsigned) (L.n_embd / (DOWN_WARPS * DOWN_ROWS)), (unsigned) cap_groups);
    if (L.d_act == 8 && L.d_type == 8) {
        native_down_contract_kernel<8><<<gd, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst,
                                                                       scratch, L, out);
    } else if (L.gu_act == 15 && L.d_act == 9 && L.d_type == 7) {
        native_down_contract_kernel<7><<<gd, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst,
                                                                       scratch, L, out);
    } else if (L.d_act == 15 && L.d_type == 12) {
        native_down_contract_kernel<12><<<gd, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst,
                                                                        scratch, L, out);
    } else if (L.d_act == 15 && L.d_type == 13) {
        native_down_contract_kernel<13><<<gd, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst,
                                                                        scratch, L, out);
    } else if (const int G = down_sub_g(L); G > 0) {
        const dim3 gs((unsigned) (L.n_embd / (DOWN_WARPS * 32 / G)), (unsigned) cap_groups);
        if (L.d_type == 20)
            native_down_sub_kernel<20, DOWN_G_IQ4_NL, 40><<<gs, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups,
                                                                                       ent_dst, hq, L, out);
        else
            native_down_sub_kernel<42, DOWN_G_Q2_0, 20><<<gs, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups,
                                                                                     ent_dst, hq, L, out);
    } else {
    switch (L.d_type) {
#define LAMINA_DOWN(T) case T: native_down_kernel<T><<<gd, DOWN_WARPS * 32, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        LAMINA_FMTS(LAMINA_DOWN)
#undef LAMINA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    }
    check("native_expert_grouped/down");
}

void native_expert_rows_out(const float* rows, const int32_t* n_groups, const int32_t* grp_start,
                            const int32_t* ent_dst, int64_t n_embd, int64_t max_rows, float* const* out,
                            uint32_t* const* flag, const int32_t* ring, unsigned* counter, void* stream) {
    const long long n4 = n_embd / 4, most = max_rows * n4;
    const unsigned blocks = (unsigned) std::min<long long>((most + 255) / 256, 256);
    rows_out_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((const float4*) rows, n_groups, grp_start, ent_dst, n4,
                                                               (float4* const*) out, flag, ring, counter);
    check("native_expert_rows_out");
}

}  // namespace lamina::kernels
