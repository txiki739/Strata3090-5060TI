// third_party/ggml-tq/tq.cuh - Gyro-L's pair-trellis expert types (TQP5, TQP6, TQP7) for ggml-cuda's MMQ tile loader.
// Derived from agentionai/llama.cpp ggml/src/ggml-cuda/tq.cuh (MIT, the ggml authors; Laurent Zuijdwijk): only what the MMQ path
// needs, reading the engine's copies of the codebook tables (ggml-common-tq.h, tq_lut_i8.cuh).
//
// A 128-weight block is 64 trellis steps of 2 weights, split into 8 lanes of 8 steps (16 weights).  A step's state s gives its
// weights d * lut[(s * 0x9e3779b1) >> 21].{lo,hi}.
#pragma once

#include "common.cuh"
#include "ggml-common-tq.h"
#include "tq_lut_i8.cuh"

#define TQ_LANES      8   // lanes per 128-weight block
#define TQ_STEPS_LANE 8   // trellis steps per lane
#define TQ_INTS_LANE  4   // the lane's 16 int8 weights as ints

#define QR_TQ 4
#define QI_TQ (QK_TQP/QR_TQ)

template<> struct ggml_cuda_type_traits<GGML_TYPE_TQP5> {
    static constexpr int qk = QK_TQP; static constexpr int qr = QR_TQ; static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp5); };
template<> struct ggml_cuda_type_traits<GGML_TYPE_TQP6> {
    static constexpr int qk = QK_TQP; static constexpr int qr = QR_TQ; static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp6); };
template<> struct ggml_cuda_type_traits<GGML_TYPE_TQP7> {
    static constexpr int qk = QK_TQP; static constexpr int qr = QR_TQ; static constexpr int qi = QI_TQ;
    static constexpr int bs = sizeof(block_tqp7); };

static constexpr __host__ __device__ bool ggml_cuda_type_is_tq(ggml_type type) {
    return type == GGML_TYPE_TQP5 || type == GGML_TYPE_TQP6 || type == GGML_TYPE_TQP7;
}

template <ggml_type type> struct tq_block_info;
template <> struct tq_block_info<GGML_TYPE_TQP5>  { typedef block_tqp5  block_t; static constexpr int K = 5; };
template <> struct tq_block_info<GGML_TYPE_TQP6>  { typedef block_tqp6  block_t; static constexpr int K = 6; };
template <> struct tq_block_info<GGML_TYPE_TQP7>  { typedef block_tqp7  block_t; static constexpr int K = 7; };

// The 15-bit states of steps 8l..8l+7 (s[j] = step 8l + j).  Step t's window starts at stream bit (63 - t)*K, so the lane's steps
// lie in the bits from S = (56 - 8l)*K: step 8l + j at S + (7 - j)*K.  S = 8*(7 - l)*K, so r = S & 15 is 0 or 8 and
// r + 7*K + 15 <= 72: five u16 words (wrapped modulo the 4*K words of the stream) cover it, the fifth only when
// r + 7*K + 15 > 64 (never for K = 5).  qs is 2-byte aligned: every trellis block has an even size.
template <ggml_type type>
static __device__ __forceinline__ void tqp_states8(const uint8_t * qs, const int l, uint32_t s[TQ_STEPS_LANE]) {
    constexpr int K      = tq_block_info<type>::K;
    constexpr int nwords = 4*K;
    const uint16_t * qs16 = (const uint16_t *) qs;
    const int S   = (56 - 8*l)*K;
    const int wb  = S >> 4;
    const int r   = S & 15;
    const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
    const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
    const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
    uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) |
                 ((uint64_t) qs16[wi2] << 32) | ((uint64_t) qs16[wi3] << 48);
    v >>= r;
    if constexpr (8 + 7*K + TQ_STATE_BITS > 64) {
        const int wi4 = wb + 4 >= nwords ? wb + 4 - nwords : wb + 4;
        v |= r ? (uint64_t) qs16[wi4] << (64 - r) : 0;
    }
#pragma unroll
    for (int j = 0; j < TQ_STEPS_LANE; ++j) {
        s[j] = (uint32_t) (v >> ((7 - j)*K)) & TQ_STATE_MASK;
    }
}

// Lane l of block x as 16 int8 values (4 packed ints, weight order) with one float scale per lane, i.e. weights ~= scale * q.
// The int8 codebook (one global scale) is already int8: no decode to float.
template <ggml_type type>
static __device__ __forceinline__ float tq_lane_to_q8(const void * x, const int l, int q[TQ_INTS_LANE]) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) x;
    uint32_t s[TQ_STEPS_LANE];
    tqp_states8<type>(b->qs, l, s);
#pragma unroll
    for (int i = 0; i < TQ_INTS_LANE; ++i) {
        const uint32_t p0 = tq_lut_i8[(s[2 * i] * 0x9e3779b1u) >> 21];
        const uint32_t p1 = tq_lut_i8[(s[2 * i + 1] * 0x9e3779b1u) >> 21];
        q[i] = (int) (p0 | (p1 << 16));
    }
    // the block structs here hold the fp16 scale as uint16_t: reinterpret, do not convert
    return __half2float(__ushort_as_half(b->d)) * TQ_LUT_I8_SCALE;
}
