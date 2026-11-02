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
#include "tq_states8.cuh"

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

// Lane l of block x as 16 int8 values (4 packed ints, weight order) with one float scale per lane, i.e. weights ~= scale * q.
// The int8 codebook (one global scale) is already int8: no decode to float.
template <ggml_type type>
static __device__ __forceinline__ float tq_lane_to_q8(const void * x, const int l, int q[TQ_INTS_LANE]) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) x;
    uint32_t s[TQ_STEPS_LANE];
    tqp_states8<tq_block_info<type>::K>(b->qs, l, s);
#pragma unroll
    for (int i = 0; i < TQ_INTS_LANE; ++i) {
        const uint32_t p0 = tq_lut_i8[(s[2 * i] * 0x9e3779b1u) >> 21];
        const uint32_t p1 = tq_lut_i8[(s[2 * i + 1] * 0x9e3779b1u) >> 21];
        q[i] = (int) (p0 | (p1 << 16));
    }
    // the block structs here hold the fp16 scale as uint16_t: reinterpret, do not convert
    return __half2float(__ushort_as_half(b->d)) * TQ_LUT_I8_SCALE;
}
