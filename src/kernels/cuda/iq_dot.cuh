// src/kernels/cuda/iq_dot.cuh - the GGUF formats' dot products against q8_1 activations (one call covers a slice of a
// block), for iq_kernels.cu and the parity test that keeps its earlier expert kernels.
//
// Transcribed from llama.cpp (ggml/src/ggml-cuda/vecdotq.cuh at the commit in third_party/ggml/VERSION.txt; MIT
// license, third_party/ggml/LICENSE); the i-quants apply their signs another way (`sign_masks`), with the same
// integer sums.  The block structs and codebook grids come from its ggml-common.h, included unchanged.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#include "ggml-common-tq.h"
#include "tq_lut_i8.cuh"
#include "lamina/kernels/dp4a.hpp"
#include "lamina/kernels/warp.cuh"
#include <type_traits>

namespace lamina::kernels {
namespace {

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}
#define ggml_cuda_dp4a(a, b, c) __dp4a((a), (b), (c))

// The i-quants' signs as byte masks (0xff where a value is negative), four values a word: the multiply puts sign bit
// i on bit 7 of byte i, and prmt's sign-replicate mode (bit 3 of a selector nibble, which __byte_perm drops) fills
// the byte with it.  Their dot products take sum(s g u) = sum(g u) - 2 sum((g & m) u) over the grid's unsigned
// values: the integers llama.cpp's sign-applied grid gives, in fewer instructions than __vcmpne4 and __vsub4.
__device__ __forceinline__ uint32_t msb_bytes(uint32_t x) {
    uint32_t d;
    asm("prmt.b32 %0, %1, 0, 0xBA98;" : "=r"(d) : "r"(x));
    return d;
}
__device__ __forceinline__ uint2 sign_masks(uint32_t s8) {    // bit i of s8 -> byte i of (x, y)
    return make_uint2(msb_bytes((s8 & 0x0F) * 0x10204080u), msb_bytes((s8 >> 4 & 0x0F) * 0x10204080u));
}
__device__ __forceinline__ uint2 ksign_masks(uint32_t v7) {   // 7 sign bits, the eighth their parity (ksigns_iq2xs)
    return sign_masks(v7 | (__popc(v7) & 1) << 7);
}

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sp = 0, sn = 0;   // sum(g u), and over the negative values
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint2 m = ksign_masks((aux32 >> (7 * k0 / 2)) & 0x7F);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sp = ggml_cuda_dp4a((int) grid_pos.x, u0, sp);
        sn = ggml_cuda_dp4a((int) (grid_pos.x & m.x), u0, sn);
        sp = ggml_cuda_dp4a((int) grid_pos.y, u1, sp);
        sn = ggml_cuda_dp4a((int) (grid_pos.y & m.y), u1, sn);
    }
    int sumi = sp - 2 * sn;
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sp[2] = {0, 0}, sn[2] = {0, 0};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint2 m = ksign_masks(q2[l0 / 2] >> 9);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        const int h = l0 / 4;
        sp[h] = ggml_cuda_dp4a((int) grid_pos.x, u0, sp[h]);
        sn[h] = ggml_cuda_dp4a((int) (grid_pos.x & m.x), u0, sn[h]);
        sp[h] = ggml_cuda_dp4a((int) grid_pos.y, u1, sp[h]);
        sn[h] = ggml_cuda_dp4a((int) (grid_pos.y & m.y), u1, sn[h]);
    }
    const int sumi0 = sp[0] - 2 * sn[0], sumi1 = sp[1] - 2 * sn[1];
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sp[2] = {0, 0}, sn[2] = {0, 0};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2s_grid)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const uint2 m = sign_masks(signs_packed_8[l0 / 2]);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        const int h = l0 / 4;
        sp[h] = ggml_cuda_dp4a((int) grid_pos.x, u0, sp[h]);
        sn[h] = ggml_cuda_dp4a((int) (grid_pos.x & m.x), u0, sn[h]);
        sp[h] = ggml_cuda_dp4a((int) grid_pos.y, u1, sp[h]);
        sn[h] = ggml_cuda_dp4a((int) (grid_pos.y & m.y), u1, sn[h]);
    }
    const int sumi0 = sp[0] - 2 * sn[0], sumi1 = sp[1] - 2 * sn[1];
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sp = 0, sn = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = iq3xxs_grid[q3[l0 + 0]], gy = iq3xxs_grid[q3[l0 + 1]];
        const uint2 m = ksign_masks((aux32 >> (7 * l0 / 2)) & 0x7F);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sp = ggml_cuda_dp4a((int) gx, u0, sp);
        sn = ggml_cuda_dp4a((int) (gx & m.x), u0, sn);
        sp = ggml_cuda_dp4a((int) gy, u1, sp);
        sn = ggml_cuda_dp4a((int) (gy & m.y), u1, sn);
    }
    int sumi = sp - 2 * sn;
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sp = 0, sn = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t gy = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
        const uint2 m = sign_masks(signs_packed_8[l0 / 2]);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sp = ggml_cuda_dp4a((int) gx, u0, sp);
        sn = ggml_cuda_dp4a((int) (gx & m.x), u0, sn);
        sp = ggml_cuda_dp4a((int) gy, u1, sp);
        sn = ggml_cuda_dp4a((int) (gy & m.y), u1, sn);
    }
    int sumi = sp - 2 * sn;
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

// vec_dot_iq3_s_q8_1 for several activation columns: the grid values, masks and scale decoded once (`load`), each
// column's integer sums and float expression as above (`apply`; X::u(b, p, u) gives int p of q8_1 block b of every
// column, X::scales(b, d) their __low2float scales).
struct IQ3SCols {
    uint32_t g[8], gm[8];   // grid words, and their negative values' bytes
    int sc;
    float dw;
    __device__ __forceinline__ void load(const void* __restrict__ vbq, const int& kbx, const int& iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t gx = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
            const uint32_t gy = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
            const uint2 m = sign_masks(signs_packed_8[l0 / 2]);
            g[l0] = gx;
            gm[l0] = gx & m.x;
            g[l0 + 1] = gy;
            gm[l0 + 1] = gy & m.y;
        }
        sc = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        dw = __half2float(bq3->d);
    }
    template <int NC, class X>
    __device__ __forceinline__ void apply(const X& x, const int& b, float (&out)[NC]) const {
        int u[8][NC];   // every position's activations first, so that their loads overlap
#pragma unroll
        for (int l = 0; l < 8; ++l) x.u(b, l, u[l]);
        int sp[NC], sn[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) sp[c] = sn[c] = 0;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                sp[c] = ggml_cuda_dp4a((int) g[l], u[l][c], sp[c]);
                sn[c] = ggml_cuda_dp4a((int) gm[l], u[l][c], sn[c]);
            }
        }
        float d8[NC];
        x.scales(b, d8);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int sumi = sp[c] - 2 * sn[c];
            sumi *= sc;
            const float d = dw * d8[c];
            out[c] = d * sumi;
        }
    }
};

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down (the unsloth UD-Q4_K_XL file's experts)
__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(const int* __restrict__ v, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
                                                             const half2& dm4, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// the 6-bit scale and min of the 32-value group pair `bq8_offset / 2` (vecdotq.cuh, shared by Q4_K and Q5_K)
__device__ __forceinline__ void k_scales(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

__device__ __forceinline__ float vec_dot_q4_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scales(bq4_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

__device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(const int* __restrict__ vl, const int* __restrict__ vh,
                                                             const int* __restrict__ u, const uint8_t* __restrict__ sc,
                                                             const uint8_t* __restrict__ m, const half2& dm5,
                                                             const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float vec_dot_q5_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scales(bq5_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

constexpr int VDR_Q5_1 = 2, VDR_Q8_0 = 2;   // VDR_Q5_1_Q8_1_MMVQ, VDR_Q8_0_Q8_1_MMVQ

// The weights are llama.cpp's; the min term is not.  llama.cpp multiplies the Q5_1 min by the q8_1 block's sum of
// the ORIGINAL activations while the scaled term uses the quantized ones, which adds min * (sum x - sum x_q) to
// every block: 1.9% instead of 1.2% relative error per expert (native_expert_parity).  Like ggml-cpu, and like
// llama.cpp's own Q4_K / Q5_K dots, the min multiplies the quantized sum here.
__device__ __forceinline__ float vec_dot_q5_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm5 = __half22float2(bq5_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
}

// Q6_K gate/up (UD-Q5_K_XL, UD-Q6_K_XL), VDR_Q6_K_Q8_1_MMVQ = 1
__device__ __forceinline__ float vec_dot_q6_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q6_K* bq6_K = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = bq6_K->scales + scale_offset;
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int u = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        const float d8 = __low2float(bq8_1[bq8_offset + 2 * i].ds);
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = lamina_q6_quants(vil, vih);   // (vil | vih) - 32
        sumf += d8 * (ggml_cuda_dp4a(vi, u, 0) * sc);
    }
    return __half2float(bq6_K->d) * sumf;
}

// IQ1_S gate/up (UD-IQ1_S), VDR 1.  As for Q5_1, the delta term takes the quantized activation sum (ggml-cpu's
// bsums) rather than llama.cpp's sum of the original values.
__device__ __forceinline__ float vec_dot_iq1_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_s* bq1 = (const block_iq1_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq1->qh[iqs];
    int sumi = 0, sumy = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | (((qh >> 3 * (l0 / 2)) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
        sumy = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumy));
    }
    const float d1q = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
    return d1q * __low2float(bq8_1[iqs].ds) * (sumi + sumy * delta);
}

// IQ4_XS gate/up (UD-IQ4_XS, UD-Q3_K_XL), VDR 4
__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    const float d8_0 = __half2float(bq8_0->d), d8_1 = __low2float(bq8_1->ds);
    return d8_0 * d8_1 * ((float) sumi);
}

// ---------------------------------------------------------------- the dots split, to decode a weight once for many entries
// `load` takes what depends on the weight (grid words and their negative values' bytes, integer scales, the block scale
// as a float; the codebook `grid()`, `grid_bytes` long, from `lut`: the table or a copy of it), `fetch` an activation's
// ints and scale, `apply` the dot's integer chain and float expression, so that apply(load(w), fetch(x)) is the dot's
// value bitwise (expert_grouped_parity checks it through the grouped kernels).
template<int TY> struct Split;
template<int N> struct SplitAct { int u[N]; float d8; };
// the 8 ints and the scale of q8_1 block b
__device__ __forceinline__ SplitAct<8> fetch8(const block_q8_1* __restrict__ b) {
    SplitAct<8> a;
#pragma unroll
    for (int l = 0; l < 8; ++l) a.u[l] = get_int_b4(b->qs, l);
    a.d8 = __low2float(b->ds);
    return a;
}
// the i-quants' grid words and their negative values' bytes; sp - 2 sn over a block's 8 ints (h: two halves' sums)
struct SplitGrid {
    uint32_t g[8], gm[8];
    __device__ __forceinline__ void put(int l0, uint32_t gx, uint32_t gy, uint2 m) {
        g[l0] = gx; gm[l0] = gx & m.x; g[l0 + 1] = gy; gm[l0 + 1] = gy & m.y;
    }
    __device__ __forceinline__ int sum(const SplitAct<8>& a) const {
        int sp = 0, sn = 0;
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            sp = ggml_cuda_dp4a((int) g[l], a.u[l], sp);
            sn = ggml_cuda_dp4a((int) gm[l], a.u[l], sn);
        }
        return sp - 2 * sn;
    }
    __device__ __forceinline__ int sum_half(const SplitAct<8>& a, int h) const {
        int sp = 0, sn = 0;
#pragma unroll
        for (int l = 4 * h; l < 4 * h + 4; ++l) {
            sp = ggml_cuda_dp4a((int) g[l], a.u[l], sp);
            sn = ggml_cuda_dp4a((int) gm[l], a.u[l], sn);
        }
        return sp - 2 * sn;
    }
};
struct SplitNoGrid {
    static constexpr int grid_bytes = 0;
    __device__ static const void* grid() { return nullptr; }
};
// the i-quants of 8 calls a block: call iqs takes q8_1 block iqs / 2 of the weight block's 8
struct SplitIq8 {
    using A = SplitAct<8>;
    __device__ static A fetch(const block_q8_1* __restrict__ bq8_1, int iqs) { return fetch8(bq8_1 + iqs / 2); }
};
template<> struct Split<16> : SplitIq8 {   // IQ2_XXS
    struct W : SplitGrid { int ls; float dw; };
    static constexpr int grid_bytes = sizeof(iq2xxs_grid);
    __device__ static const void* grid() { return iq2xxs_grid; }
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void* __restrict__ lut) {
        W w;
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = ((const uint2*) lut)[aux8[k0 / 2]];
            w.put(k0, grid_pos.x, grid_pos.y, ksign_masks((aux32 >> (7 * k0 / 2)) & 0x7F));
        }
        w.ls = aux32 >> 27 | 1;
        w.dw = __half2float(bq2->d);
        return w;
    }
    __device__ static float apply(const W& w, const A& a) {
        int sumi = w.sum(a);
        sumi = sumi * w.ls / 8;
        const float d = w.dw * a.d8;
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S: two halves with a 4-bit scale each
struct SplitLs2 : SplitIq8 {
    struct W : SplitGrid { int ls0, ls1; float dw; };
    __device__ static float apply(const W& w, const A& a) {
        const int sumi0 = w.sum_half(a, 0), sumi1 = w.sum_half(a, 1);
        const int sumi = (sumi0 * w.ls0 + sumi1 * w.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = w.dw * a.d8;
        return d * sumi;
    }
};
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    static constexpr int grid_bytes = sizeof(iq2xs_grid);
    __device__ static const void* grid() { return iq2xs_grid; }
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void* __restrict__ lut) {
        W w;
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t* q2 = (const uint16_t*) &q2_packed;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = ((const uint2*) lut)[q2[l0 / 2] & 0x1FF];
            w.put(l0, grid_pos.x, grid_pos.y, ksign_masks(q2[l0 / 2] >> 9));
        }
        w.ls0 = bq2->scales[iqs / 2] & 0x0F;
        w.ls1 = bq2->scales[iqs / 2] >> 4;
        w.dw = __half2float(bq2->d);
        return w;
    }
};
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    static constexpr int grid_bytes = sizeof(iq2s_grid);
    __device__ static const void* grid() { return iq2s_grid; }
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void* __restrict__ lut) {
        W w;
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = ((const uint2*) lut)[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
            w.put(l0, grid_pos.x, grid_pos.y, sign_masks(signs_packed_8[l0 / 2]));
        }
        w.ls0 = bq2->scales[iqs / 2] & 0x0F;
        w.ls1 = bq2->scales[iqs / 2] >> 4;
        w.dw = __half2float(bq2->d);
        return w;
    }
};
template<> struct Split<18> : SplitIq8 {   // IQ3_XXS
    struct W : SplitGrid { int ls; float dw; };
    static constexpr int grid_bytes = sizeof(iq3xxs_grid);
    __device__ static const void* grid() { return iq3xxs_grid; }
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void* __restrict__ lut) {
        const uint32_t* grid32 = (const uint32_t*) lut;
        W w;
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
            w.put(l0, grid32[q3[l0 + 0]], grid32[q3[l0 + 1]], ksign_masks((aux32 >> (7 * l0 / 2)) & 0x7F));
        w.ls = aux32 >> 28;
        w.dw = __half2float(bq3->d);
        return w;
    }
    __device__ static float apply(const W& w, const A& a) {
        int sumi = w.sum(a);
        sumi = (w.ls * sumi + sumi / 2) / 2;
        const float d = w.dw * a.d8;
        return d * sumi;
    }
};
template<> struct Split<21> : SplitIq8 {   // IQ3_S
    struct W : SplitGrid { int sc; float dw; };
    static constexpr int grid_bytes = sizeof(iq3s_grid);
    __device__ static const void* grid() { return iq3s_grid; }
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void* __restrict__ lut) {
        const uint32_t* grid32 = (const uint32_t*) lut;
        W w;
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
            w.put(l0, grid32[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)], grid32[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)],
                  sign_masks(signs_packed_8[l0 / 2]));
        w.sc = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        w.dw = __half2float(bq3->d);
        return w;
    }
    __device__ static float apply(const W& w, const A& a) {
        int sumi = w.sum(a);
        sumi *= w.sc;
        const float d = w.dw * a.d8;
        return d * sumi;
    }
};
template<> struct Split<23> : SplitNoGrid {   // IQ4_XS
    using A = SplitAct<8>;
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void*) {
        W w;
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
#pragma unroll
        for (int j = 0; j < 4; ++j) w.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        w.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        w.dw = __half2float(bq4->d);
        return w;
    }
    __device__ static A fetch(const block_q8_1* __restrict__ bq8_1, int iqs) { return fetch8(bq8_1 + iqs / 4); }
    __device__ static float apply(const W& w, const A& a) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = ggml_cuda_dp4a(w.v[j].x, a.u[j + 0], sumi);
            sumi = ggml_cuda_dp4a(w.v[j].y, a.u[j + 4], sumi);
        }
        sumi *= w.ls - 32;
        const float d = w.dw * a.d8;
        return d * sumi;
    }
};
template<> struct Split<20> : SplitNoGrid {   // IQ4_NL: a call takes ints iqs, iqs + 1, iqs + 4, iqs + 5 of the q8_1 block
    using A = SplitAct<4>;
    struct W { int2 v[2]; float dw; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void*) {
        W w;
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
#pragma unroll
        for (int l = 0; l < 2; ++l) w.v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        w.dw = __half2float(bq4->d);
        return w;
    }
    __device__ static A fetch(const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        A a;
        a.u[0] = q8[0]; a.u[1] = q8[4]; a.u[2] = q8[1]; a.u[3] = q8[5];
        a.d8 = __low2float(bq8_1->ds);
        return a;
    }
    __device__ static float apply(const W& w, const A& a) {
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(w.v[l].x, a.u[2 * l + 0], sumi);
            sumi = ggml_cuda_dp4a(w.v[l].y, a.u[2 * l + 1], sumi);
        }
        const float d = w.dw * a.d8;
        return d * sumi;
    }
};
template<> struct Split<42> : SplitNoGrid {   // Q2_0
    using A = SplitAct<8>;
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void*) {
        W w;
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        w.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            w.qx[j] = __byte_perm(qe, qo, 0x5140);
            w.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return w;
    }
    __device__ static A fetch(const block_q8_1* __restrict__ bq8_1, int iqs) { return fetch8(bq8_1 + iqs); }
    __device__ static float apply(const W& w, const A& a) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = ggml_cuda_dp4a(a.u[2 * j + 0], w.qx[j], sumi);
            sumi = ggml_cuda_dp4a(a.u[2 * j + 1], w.qy[j], sumi);
        }
        return w.d2 * a.d8 * sumi;
    }
};
// Gyro-L's pair trellis (TQP5, TQP6, TQP7: third_party/ggml-tq): a call is a lane of 8 steps, 16 weights.  The steps' 15-bit states
// lie in the 64 bits from stream bit S = (56 - 8l) * K, a whole number of bytes, so r = S & 15 is 0 or 8 and five u16 words wrapped
// over the block's 4 * K cover them (the fifth only when K > 5); qs is 2-byte aligned, every trellis block being of even size.  A
// state hashes to a point of the int8 codebook (`grid`, one global scale): two weights, so the lane's 16 are 4 ints of int8 and
// the dot is 4 dp4a.  The lane reads half of q8_1 block l / 2.
template<int K>
__device__ __forceinline__ void tqp_states8(const uint8_t* __restrict__ qs, int l, uint32_t (&s)[8]) {
    constexpr int nwords = 4 * K;
    const uint16_t* qs16 = (const uint16_t*) qs;
    const int S = (56 - 8 * l) * K, wb = S >> 4, r = S & 15;
    const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
    const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
    const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
    uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) | ((uint64_t) qs16[wi2] << 32) |
                 ((uint64_t) qs16[wi3] << 48);
    v >>= r;
    if constexpr (8 + 7 * K + TQ_STATE_BITS > 64) {
        const int wi4 = wb + 4 >= nwords ? wb + 4 - nwords : wb + 4;
        v |= r ? (uint64_t) qs16[wi4] << (64 - r) : 0;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = (uint32_t) (v >> ((7 - j) * K)) & TQ_STATE_MASK;
}
template<int K>
struct SplitTqp {
    using A = SplitAct<4>;
    struct W { int q[4]; float dw; };
    static constexpr int grid_bytes = sizeof(tq_lut_i8);
    __device__ static const void* grid() { return tq_lut_i8; }
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const void* lut) {
        const uint8_t* b = (const uint8_t*) vbq + (size_t) kbx * (2 + 8 * K);
        uint32_t s[8];
        tqp_states8<K>(b + 2, iqs / 4, s);
        const uint16_t* t = (const uint16_t*) lut;
        W w;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            w.q[i] = (int) ((uint32_t) t[(s[2 * i] * 0x9e3779b1u) >> 21] | ((uint32_t) t[(s[2 * i + 1] * 0x9e3779b1u) >> 21] << 16));
        w.dw = __half2float(__ushort_as_half(*(const uint16_t*) b));
        return w;
    }
    __device__ static A fetch(const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int l = iqs / 4;
        const block_q8_1* b8 = bq8_1 + l / 2;
        const int* q8 = (const int*) b8->qs + 4 * (l & 1);
        A a;
#pragma unroll
        for (int i = 0; i < 4; ++i) a.u[i] = q8[i];
        a.d8 = __low2float(b8->ds);
        return a;
    }
    __device__ static float apply(const W& w, const A& a) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) sumi = ggml_cuda_dp4a(w.q[i], a.u[i], sumi);
        return w.dw * (a.d8 * TQ_LUT_I8_SCALE) * (float) sumi;
    }
};
template<> struct Split<147> : SplitTqp<5> {};
template<> struct Split<148> : SplitTqp<6> {};
template<> struct Split<149> : SplitTqp<7> {};

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = QI6_K, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<19> { static constexpr int qk = 256, ipb = QI1_S, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = QI4_XS / 4, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };

#define LAMINA_TQP_FMT(T) template<> struct Fmt<T> { static constexpr int qk = 128, ipb = 8, step = 4; \
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { \
        return Split<T>::apply(Split<T>::load(v, kbx, iqs, tq_lut_i8), Split<T>::fetch(y, iqs)); } };
LAMINA_TQP_FMT(147)
LAMINA_TQP_FMT(148)
LAMINA_TQP_FMT(149)
#undef LAMINA_TQP_FMT

// Every format above, for the dispatch switches: X(type) once per Fmt<type>.
#define LAMINA_TQP_FMTS(X) X(147) X(148) X(149)
#define LAMINA_FMTS(X) X(7) X(8) X(12) X(13) X(14) X(16) X(17) X(18) X(19) X(20) X(21) X(22) X(23) X(29) X(42) LAMINA_TQP_FMTS(X)

inline int fmt_qk(int t) {
    switch (t) {
#define LAMINA_QK(T) case T: return Fmt<T>::qk;
        LAMINA_FMTS(LAMINA_QK)
#undef LAMINA_QK
        default: return 0;
    }
}

// The --expert-math exact dots below each compute ONE ROW PER EIGHT-LANE GROUP: lane l = lane & 7 of a group owns
// AVX2 accumulator l, and the group reduces its eight accumulators with width-8 shuffles.  A warp's four groups
// therefore take four rows at once (the callers pass each group its own row), and every group performs exactly the
// operations of the one-row-per-warp form, so the values are bitwise unchanged.  Every lane receives its own group's
// result.  All 32 lanes must call together (the shuffles are warp-wide); a caller with one row passes it to all four.

// Match ggml's x86 implementation (its AVX2 branch).  ggml keeps
// eight independent float accumulators, with each accumulator receiving the
// integer dot for four bytes in the qx/qy vectors.  The qx vector is ordered
// as q5_1 low nibbles [0..15], then high nibbles [0..15].  It applies the
// d*d product with a fused multiply-add, horizontally reduces the eight
// accumulators in hsum_float_8 order, and adds the separately accumulated
// m*s term afterward.
__device__ __forceinline__ float row_dot_q5_1_q8_1_cpu_avx2(const uint8_t* row, const block_q8_1* x,
                                                             int nb, int lane) {
    const int l = lane & 7;
    float acc = 0.0f;
    float summs = 0.0f;
    {
        const block_q5_1* w = (const block_q5_1*) row;
        const int qbase = l * 4;
        for (int ib = 0; ib < nb; ++ib) {
            const uint32_t qh = (uint32_t) get_int_b4(w[ib].qh, 0);
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int p = qbase + j;
                const int qbyte = p < 16 ? p : p - 16;
                const int qshift = p < 16 ? 0 : 4;
                const int q = ((int) w[ib].qs[qbyte] >> qshift & 0x0f) | (((qh >> p) & 1) << 4);
                sumi += q * (int) ((const int8_t *) x[ib].qs)[p];
            }
            const float2 dm5 = __half22float2(w[ib].dm);
            const float2 ds8 = __half22float2(x[ib].ds);
            const float d5d8 = __fmul_rn(dm5.x, ds8.x);
            acc = __fmaf_rn((float) sumi, d5d8, acc);
            if (l == 0) {
                summs = __fadd_rn(summs, __fmul_rn(dm5.y, ds8.y));
            }
        }
    }
    const unsigned mask = 0xffffffffu;
    // Every lane must execute the same shuffle instructions.  Having only
    // lane 0 issue these shuffles would read undefined values because lanes
    // 1..7 would be at a different shuffle instruction.
    const float a0 = __shfl_sync(mask, acc, 0, 8);
    const float a1 = __shfl_sync(mask, acc, 1, 8);
    const float a2 = __shfl_sync(mask, acc, 2, 8);
    const float a3 = __shfl_sync(mask, acc, 3, 8);
    const float a4 = __shfl_sync(mask, acc, 4, 8);
    const float a5 = __shfl_sync(mask, acc, 5, 8);
    const float a6 = __shfl_sync(mask, acc, 6, 8);
    const float a7 = __shfl_sync(mask, acc, 7, 8);
    float result = 0.0f;
    if (l == 0) {
        const float lo0 = __fadd_rn(a0, a4);
        const float lo1 = __fadd_rn(a1, a5);
        const float lo2 = __fadd_rn(a2, a6);
        const float lo3 = __fadd_rn(a3, a7);
        const float left = __fadd_rn(__fadd_rn(lo0, lo2), __fadd_rn(lo1, lo3));
        result = __fadd_rn(left, summs);
    }
    return __shfl_sync(mask, result, 0, 8);
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

// ---------------------------------------------------------------- native activation-contract dots
// The native CPU path uses ggml's vec_dot_type, not the CUDA q8_1 convenience format.  UD-Q4_K_XL has
// Q4_K/Q5_K -> Q8_K for gate/up and Q5_1 -> Q8_1 or Q8_0 -> Q8_0 for down.  These helpers mirror ggml's
// generic integer formulas.  Each lane owns eight values of every Q8_K superblock; the warp then sums the
// integer partials in FP32.  This keeps the quantized codes and scale/min terms identical to the CPU contract.
__device__ __forceinline__ void unpack_k_scales_mins(const uint8_t* src, uint8_t (&sc)[8], uint8_t (&mn)[8]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    constexpr uint32_t kmask3 = 0x03030303;
    const uint32_t* p = (const uint32_t*) src;
    uint32_t u[4] = {p[0], p[1], p[2], 0};
    u[3] = ((u[2] >> 4) & kmask2) | (((u[1] >> 6) & kmask3) << 4);
    const uint32_t aux = u[1] & kmask1;
    u[1] = (u[2] & kmask2) | (((u[0] >> 6) & kmask3) << 4);
    u[2] = aux;
    u[0] &= kmask1;
    const uint8_t* b = (const uint8_t*) u;
#pragma unroll
    for (int i = 0; i < 8; ++i) { sc[i] = b[i]; mn[i] = b[8 + i]; }
}

// ggml's x86 Q4_K/Q5_K x Q8_K dots (their AVX2 branch).  A lane-strided form, mathematically
// equivalent, sums every block's floats in another order.  So: eight persistent main accumulators and four persistent min accumulators, exactly
// as ggml_vec_dot_q4_K_q8_K / q5_K_q8_K do, then reproduce hsum_float_8 and its
// final min reduction.  Lanes 0..7 of each eight-lane group own the main
// accumulators and its lanes 0..3 the min accumulators; the result is broadcast
// to the group so the caller can still consume the group's lane zero.  Q5_K's
// `summs += dmin * total` is C, not an intrinsic: one FMA where GCC or Clang
// compiled ggml, a product then a sum where MSVC did (`fma_mins`, probed by
// cpu::ggml_q5_K_mins_rounding).
template<bool Q5>
__device__ __forceinline__ float row_dot_k_q8_k_cpu_avx2(const uint8_t* row, size_t block_stride,
                                                         const block_q8_K* x, int nb, int lane, bool fma_mins) {
    using W = typename std::conditional<Q5, block_q5_K, block_q4_K>::type;
    const int l = lane & 7;
    float acc = 0.0f;
    float acc_m = 0.0f;
    float summs = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const W* w = (const W*) (row + (size_t) kbx * block_stride);
        const block_q8_K* a = x + kbx;
        uint8_t sc[8], mn[8];
        unpack_k_scales_mins(w->scales, sc, mn);

        // This is the integer grouping produced by AVX2 maddubs/madd in ggml:
        // each output lane sums four adjacent q4/q5 bytes, with the low and high
        // nibbles occupying the two 32-value halves of the super-block.
        {
            int32_t si = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int32_t lo = 0, hi = 0;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int qi = 4 * l + p;
                    int qlo = (int) (w->qs[j * 32 + qi] & 0x0f);
                    int qhi = (int) (w->qs[j * 32 + qi] >> 4);
                    if constexpr (Q5) {
                        qlo |= ((int) (w->qh[qi] >> (2 * j + 0)) & 1) << 4;
                        qhi |= ((int) (w->qh[qi] >> (2 * j + 1)) & 1) << 4;
                    }
                    lo += qlo * (int) a->qs[j * 64 + qi];
                    hi += qhi * (int) a->qs[j * 64 + 32 + qi];
                }
                si += (int32_t) sc[2 * j + 0] * lo + (int32_t) sc[2 * j + 1] * hi;
            }
            const float2 dm = __half22float2(w->dm);
            const float d = __fmul_rn(a->d, dm.x);
            acc = fmaf(d, (float) si, acc);
        }

        // ggml's q8sums/hadd/madd path forms four signed-int32 min products.
        int32_t mi = 0;
        if (l < 4) {
            const int group = l * 4;
            const int qsum0 = (int) a->bsums[group + 0] + (int) a->bsums[group + 1];
            const int qsum1 = (int) a->bsums[group + 2] + (int) a->bsums[group + 3];
            mi = (int32_t) mn[2 * l + 0] * qsum0 +
                 (int32_t) mn[2 * l + 1] * qsum1;
        }
        if constexpr (Q5) {
            // AVX2 Q5_K reduces the four signed integer min products inside
            // each 256-element block, then performs one scalar FP32 FMA into
            // a persistent summs value. Q4_K intentionally keeps its existing
            // four-lane packed min accumulator below.
            const int32_t mi0 = __shfl_sync(0xffffffffu, mi, 0, 8);
            const int32_t mi1 = __shfl_sync(0xffffffffu, mi, 1, 8);
            const int32_t mi2 = __shfl_sync(0xffffffffu, mi, 2, 8);
            const int32_t mi3 = __shfl_sync(0xffffffffu, mi, 3, 8);
            if (l == 0) {
                const int32_t t01 = mi0 + mi1;
                const int32_t t23 = mi2 + mi3;
                const int32_t total = t01 + t23;
                const float2 dm = __half22float2(w->dm);
                const float dmin = -__fmul_rn(a->d, dm.y);
                summs = fma_mins ? fmaf(dmin, (float) total, summs)
                                 : __fadd_rn(summs, __fmul_rn(dmin, (float) total));
            }
        } else if (l < 4) {
            const float2 dm = __half22float2(w->dm);
            const float dmin = -__fmul_rn(a->d, dm.y);
            acc_m = fmaf(dmin, (float) mi, acc_m);
        }
    }
    const unsigned mask = 0xffffffffu;
    const float v = acc;
    // Every lane executes every shuffle.  Keeping the shuffles out of the
    // lane-zero branch is required by CUDA's warp-synchronous contract.
    const float a0 = __shfl_sync(mask, v, 0, 8);
    const float a1 = __shfl_sync(mask, v, 1, 8);
    const float a2 = __shfl_sync(mask, v, 2, 8);
    const float a3 = __shfl_sync(mask, v, 3, 8);
    const float a4 = __shfl_sync(mask, v, 4, 8);
    const float a5 = __shfl_sync(mask, v, 5, 8);
    const float a6 = __shfl_sync(mask, v, 6, 8);
    const float a7 = __shfl_sync(mask, v, 7, 8);
    const float m0 = __shfl_sync(mask, l < 4 ? acc_m : 0.0f, 0, 8);
    const float m1 = __shfl_sync(mask, l < 4 ? acc_m : 0.0f, 1, 8);
    const float m2 = __shfl_sync(mask, l < 4 ? acc_m : 0.0f, 2, 8);
    const float m3 = __shfl_sync(mask, l < 4 ? acc_m : 0.0f, 3, 8);
    float result = 0.0f;
    if (l == 0) {
        // Same sequence as hsum_float_8: high+low, movehl pair, then scalar add.
        const float r0 = __fadd_rn(a0, a4);
        const float r1 = __fadd_rn(a1, a5);
        const float r2 = __fadd_rn(a2, a6);
        const float r3 = __fadd_rn(a3, a7);
        const float main = __fadd_rn(__fadd_rn(r0, r2), __fadd_rn(r1, r3));
        if constexpr (Q5) result = __fadd_rn(main, summs);
        else {
            const float minsum = __fadd_rn(__fadd_rn(m0, m2), __fadd_rn(m1, m3));
            result = __fadd_rn(main, minsum);
        }
    }
    return __shfl_sync(mask, result, 0, 8);
}

// Row-stride-aware versions.  The caller supplies the block stride so this remains valid if ggml changes padding.
template<bool Q5>
__device__ __forceinline__ float row_dot_k_q8_k_stride(const uint8_t* row, size_t block_stride,
                                                       const block_q8_K* x, int nb, int lane, bool fma_mins) {
    return row_dot_k_q8_k_cpu_avx2<Q5>(row, block_stride, x, nb, lane, fma_mins);
}

__device__ __forceinline__ float row_dot_q8_0_q8_0(const uint8_t* row, size_t row_stride,
                                                   const block_q8_0* x, int nb, int lane) {
    const int l = lane & 7;
    float acc = 0.0f;
    for (int kbx = 0; kbx < nb; ++kbx) {
        const auto* w = (const block_q8_0*) (row + (size_t) kbx * row_stride);
        const block_q8_0* a = x + kbx;
        int32_t si = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            si += (int32_t) w->qs[4 * l + j] * (int32_t) a->qs[4 * l + j];
        const float d = __fmul_rn(__half2float(w->d), __half2float(a->d));
        acc = fmaf(d, (float) si, acc);
    }
    const unsigned mask = 0xffffffffu;
    const float v = acc;
    const float a0 = __shfl_sync(mask, v, 0, 8), a1 = __shfl_sync(mask, v, 1, 8);
    const float a2 = __shfl_sync(mask, v, 2, 8), a3 = __shfl_sync(mask, v, 3, 8);
    const float a4 = __shfl_sync(mask, v, 4, 8), a5 = __shfl_sync(mask, v, 5, 8);
    const float a6 = __shfl_sync(mask, v, 6, 8), a7 = __shfl_sync(mask, v, 7, 8);
    float result = 0.0f;
    if (l == 0) {
        const float r0 = __fadd_rn(a0, a4), r1 = __fadd_rn(a1, a5);
        const float r2 = __fadd_rn(a2, a6), r3 = __fadd_rn(a3, a7);
        result = __fadd_rn(__fadd_rn(r0, r2), __fadd_rn(r1, r3));
    }
    return __shfl_sync(mask, result, 0, 8);
}

}  // namespace
}  // namespace lamina::kernels
