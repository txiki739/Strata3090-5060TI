// src/kernels/cpu/tq_avx2.cpp - Gyro-L's expert rows on AVX2 + FMA (tq_cpu.hpp), reached behind cpu_avx2_ok() only.
//
// Per 128-weight block: the 64 states from the lanes' words in scalar code, their table entries into YMMs (eight states make
// the 16 int16 weights of a register, two registers a q8_1 block's 32), then per token VPMADDWD against the sign-extended
// int8 activations, one exact int32 sum per q8_1 block, and the scalar kernel's float steps lane for lane.  The weights are
// decoded once per block for all the tokens of the group.
#include "lamina/kernels/cpu/tq_cpu.hpp"

#include "tq_states.hpp"

#include <immintrin.h>

namespace lamina::kernels::cpu::tq_detail {
namespace {

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
            __m256i wv[8];
            for (int l = 0; l < 8; ++l) wv[l] = lane_weights<KB>(tab, word[l]);
            const __m128 dx = _mm_set1_ps(tq_half_to_float(tq_ld16(blk)));
            for (int t = 0; t < nt; ++t) {
                const uint8_t* q = a[t].blocks + (size_t) (4 * i) * 36 + 4;
                int32_t sum[4];
                for (int k = 0; k < 4; ++k) {
                    const __m128i lo = _mm_loadu_si128((const __m128i*) (q + (size_t) k * 36));
                    const __m128i hi = _mm_loadu_si128((const __m128i*) (q + (size_t) k * 36 + 16));
                    const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(wv[2 * k], _mm256_cvtepi8_epi16(lo)),
                                                       _mm256_madd_epi16(wv[2 * k + 1], _mm256_cvtepi8_epi16(hi)));
                    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
                    s = _mm_hadd_epi32(s, s);
                    s = _mm_hadd_epi32(s, s);
                    sum[k] = _mm_cvtsi128_si32(s);
                }
                const __m128 sc = _mm_mul_ps(dx, _mm_loadu_ps(a[t].d + 4 * i));
                const __m128i s4 = _mm_set_epi32(sum[3], sum[2], sum[1], sum[0]);
                acc[t] = _mm_fmadd_ps(_mm_cvtepi32_ps(s4), sc, acc[t]);
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

void tq_rows_avx2(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0,
                  int r1) {
    if (kb == 5) rows_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace lamina::kernels::cpu::tq_detail
