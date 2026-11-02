// src/kernels/cpu/tq_avx512.cpp - Gyro-L's expert rows on AVX-512 F/BW/VL/VNNI (tq_cpu.hpp), reached behind cpu_avx512_ok() only.
//
// Per 128-weight block the AVX2 kernel's decode, a ZMM of 32 int16 weights per q8_1 block; per token four VPDPWSSD against the
// sign-extended int8 activations, an unpack/add tree to the four exact sums, and the scalar kernel's float steps lane for lane.
#include "lamina/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

namespace lamina::kernels::cpu::tq_detail {
namespace {

// four ZMM of 16 int32 partial sums -> [S0, S1, S2, S3] (exact integer adds)
inline __m128i reduce4(__m512i p0, __m512i p1, __m512i p2, __m512i p3) {
    const __m512i a = _mm512_add_epi32(_mm512_unpacklo_epi32(p0, p1), _mm512_unpackhi_epi32(p0, p1));
    const __m512i b = _mm512_add_epi32(_mm512_unpacklo_epi32(p2, p3), _mm512_unpackhi_epi32(p2, p3));
    const __m512i c = _mm512_add_epi32(_mm512_unpacklo_epi64(a, b), _mm512_unpackhi_epi64(a, b));
    // c: per 128-bit lane [p0, p1, p2, p3] partial sums; add the four lanes
    const __m256i d = _mm256_add_epi32(_mm512_castsi512_si256(c), _mm512_extracti64x4_epi64(c, 1));
    return _mm_add_epi32(_mm256_castsi256_si128(d), _mm256_extracti128_si256(d, 1));
}

template <int KB>
inline __m256i lane_weights(const uint32_t* tab, uint64_t word) {
    const auto at = [&](int j) { return tab[(uint32_t) (word >> ((7 - j) * KB)) & kTqStateMask]; };
    return _mm256_set_epi32((int) at(7), (int) at(6), (int) at(5), (int) at(4), (int) at(3), (int) at(2), (int) at(1),
                            (int) at(0));
}

template <int KB>
void rows_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0, int r1) {
    const int nb = n / kTqBlock;
    constexpr size_t bsz = 2 + 8 * KB;
    const uint32_t* tab = tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        __m128 acc[kTqMaxTokens];
        for (int t = 0; t < nt; ++t) acc[t] = _mm_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            uint64_t word[8];
            tq_lane_words<KB>(blk + 2, word);
            __m512i wv[4];
            for (int k = 0; k < 4; ++k)
                wv[k] = _mm512_inserti64x4(_mm512_castsi256_si512(lane_weights<KB>(tab, word[2 * k])),
                                           lane_weights<KB>(tab, word[2 * k + 1]), 1);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            const __m512i z = _mm512_setzero_si512();
            for (int t = 0; t < nt; ++t) {
                const uint8_t* q = a[t].blocks + (size_t) (4 * i) * 36 + 4;
                __m512i p[4];
                for (int k = 0; k < 4; ++k)
                    p[k] = _mm512_dpwssd_epi32(z, wv[k],
                                               _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) (q + (size_t) k * 36))));
                const __m128i s = reduce4(p[0], p[1], p[2], p[3]);
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t].d + 4 * i));
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s), sc, acc[t]);
            }
        }
        for (int t = 0; t < nt; ++t) {
            alignas(16) float v[4];
            _mm_store_ps(v, acc[t]);
            out[t][r] = ((v[0] + v[1]) + (v[2] + v[3])) * (1.0f / 8192.0f);
        }
    }
}

}  // namespace

void tq_rows_avx512(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0,
                    int r1) {
    if (kb == 5) rows_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace lamina::kernels::cpu::tq_detail
