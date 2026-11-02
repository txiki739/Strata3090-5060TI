// third_party/ggml-tq/tq_states8.cuh - a trellis lane's 8 states, for the verify kernels (iq_dot.cuh) and the MMQ tile loader (tq.cuh).
//
// qs: a block's 8 * K stream bytes, 2-byte aligned (every trellis block has an even size).  Step t's state is the 15 stream bits from
// bit (63 - t) * K, circular; lane l's steps 8l .. 8l+7 lie in the bits from S = (56 - 8l) * K, step 8l + j at S + (7 - j) * K.  S is a
// whole number of bytes, so r = S & 15 is 0 or 8, and five u16 words cover them (the fifth only when K > 5).
#pragma once

#include "ggml-common-tq.h"

template<int K>
__device__ __forceinline__ void tqp_states8(const uint8_t* __restrict__ qs, int l, uint32_t (&s)[8]) {
    constexpr int nwords = 4 * K;
    const int S = (56 - 8 * l) * K, wb = S >> 4, r = S & 15;
    const uint16_t* w = (const uint16_t*) qs + wb;
    // only lane 0's window passes the block's end: K = 5, 6 its fourth word, K = 7 its fifth (its fourth is the last of the block)
    const int wrap = l == 0 ? -nwords : 0;
    // 32-bit windows: a 64-bit shift takes three instructions a state (S is a multiple of 16 for even K, r only for odd K)
    uint32_t lo = (uint32_t) w[0] | ((uint32_t) w[1] << 16);
    uint32_t hi = (uint32_t) w[2] | ((uint32_t) w[K > 6 ? 3 : 3 + wrap] << 16);
    if constexpr (K % 2 != 0) {
        lo = (lo >> r) | ((hi << 8) << (24 - r));   // r is 0 or 8: never a shift by 32
        if constexpr (K > 5) hi = (hi >> r) | (((uint32_t) w[4 + wrap] << 8) << (24 - r));
        else hi >>= r;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int sh = (7 - j) * K;
        const uint32_t v = sh >= 32 ? hi >> (sh - 32) : sh + TQ_STATE_BITS <= 32 ? lo >> sh : (lo >> sh) | (hi << (32 - sh));
        s[j] = v & TQ_STATE_MASK;
    }
}
