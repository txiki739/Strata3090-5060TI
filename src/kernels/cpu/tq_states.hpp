// src/kernels/cpu/tq_states.hpp - the 64 trellis states of one TQP block, internal to src/kernels/cpu/tq_*.cpp.
//
// Step t's state is the 15 stream bits from bit (63 - t) * K, LSB first, circular over 64 * K bits.  The eight steps 8l .. 8l+7
// lie in the 64 bits from byte (7 - l) * K, step 8l + j at bit (7 - j) * K of that word (7 * K + 15 <= 64), and only lane 0's
// word runs past the block's 8 * K bytes, into its first ones.
#pragma once

#include <cstdint>
#include <cstring>

namespace lamina::kernels::cpu::tq_detail {
// internal linkage: each object (tq_cpu, tq_avx2, tq_avx512, built for their own ISAs) keeps its own copies, where one
// inline copy the linker kept for all could be the AVX-512 object's, run on an AVX2 CPU
namespace {

inline constexpr uint32_t kTqStateMask = 0x7fffu;

inline uint64_t tq_ld64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;   // little-endian: every target this engine builds for
}

/// The lanes' words: word[l] holds the states of steps 8l .. 8l+7.
template <int KB>
inline void tq_lane_words(const uint8_t* qs, uint64_t* word) {
    const int tail = 8 * (8 - KB);   // lane 0: the block's last KB bytes, then its first 8 - KB
    word[0] = (tq_ld64(qs + 7 * KB - (8 - KB)) >> tail) | (tq_ld64(qs) << (8 * KB));
    for (int l = 1; l < 8; ++l) word[l] = tq_ld64(qs + (7 - l) * KB);
}

/// fp16 bits -> float, exact (the block scales and the activation scales).
inline float tq_half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (man << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    } else if (man == 0) {
        bits = sign;
    } else {   // subnormal: man * 2^-24, exact in float
        const float m = (float) man * (1.0f / 16777216.0f);
        return sign ? -m : m;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t tq_ld16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

}  // namespace
}  // namespace lamina::kernels::cpu::tq_detail
