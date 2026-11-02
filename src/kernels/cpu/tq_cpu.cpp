// src/kernels/cpu/tq_cpu.cpp - see include/lamina/kernels/cpu/tq_cpu.hpp: the state table, the scalar reference kernel, the
// activation scales and the run-time ISA choice.  Compiled for the baseline x86-64, so everything here runs on any CPU; the
// AVX2 and AVX-512 kernels are in their own translation units.
#include "lamina/kernels/cpu/tq_cpu.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"

#include "tq_states.hpp"
#include "ggml-common-tq.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace lamina::kernels::cpu {
namespace {

using tq_detail::TqAct;

alignas(64) uint32_t g_table[tq_detail::kTqStateMask + 1];

void fill_table() {
    for (uint32_t s = 0; s <= tq_detail::kTqStateMask; ++s) {
        const uint32_t pt = (s * 0x9e3779b1u) >> 21;   // the state's codebook point
        uint32_t row = 0;
        for (int j = 0; j < 2; ++j) {
            // fp16 * 2^13 is exact in float; rounded half away from zero, as the fork's ggml-cpu table is
            const float v = tq_detail::tq_half_to_float(tq_lut_f16[2 * pt + j]) * 8192.0f;
            row |= (uint32_t) (uint16_t) (int16_t) (v < 0.0f ? v - 0.5f : v + 0.5f) << (16 * j);
        }
        g_table[s] = row;
    }
}

enum class Isa { scalar, avx2, avx512 };

Isa isa() {
    static const Isa v = [] {
        const char* e = std::getenv("LAMINA_TQ_ISA");
        const std::string cap = e != nullptr ? e : "";
        if (cap != "scalar" && cap != "avx2" && cpu_avx512_ok()) return Isa::avx512;
        if (cap != "scalar" && cpu_avx2_ok()) return Isa::avx2;
        return Isa::scalar;
    }();
    return v;
}

template <int KB>
void rows_scalar_t(const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0, int r1) {
    const int nb = n / kTqBlock;
    constexpr size_t bsz = 2 + 8 * KB;
    const uint32_t* tab = tq_detail::tq_state_table();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* x = w + (size_t) r * row_bytes;
        float acc[kTqMaxTokens][4] = {};
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = x + (size_t) i * bsz;
            const float dx = tq_detail::tq_half_to_float(tq_detail::tq_ld16(blk));
            uint64_t word[8];
            tq_detail::tq_lane_words<KB>(blk + 2, word);
            int16_t wv[128];
            for (int l = 0; l < 8; ++l)
                for (int j = 0; j < 8; ++j) {
                    const uint32_t row = tab[(uint32_t) (word[l] >> ((7 - j) * KB)) & tq_detail::kTqStateMask];
                    wv[16 * l + 2 * j] = (int16_t) (row & 0xffffu);
                    wv[16 * l + 2 * j + 1] = (int16_t) (row >> 16);
                }
            for (int t = 0; t < nt; ++t)
                for (int k = 0; k < 4; ++k) {
                    const int8_t* q = (const int8_t*) a[t].blocks + (size_t) (4 * i + k) * 36 + 4;
                    int32_t s = 0;
                    for (int j = 0; j < 32; ++j) s += (int32_t) wv[32 * k + j] * q[j];
                    acc[t][k] = std::fma((float) s, dx * a[t].d[4 * i + k], acc[t][k]);
                }
        }
        for (int t = 0; t < nt; ++t)
            out[t][r] = ((acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3])) * (1.0f / 8192.0f);
    }
}

// A call's activation scales, once for all its rows (the q8_1 rows themselves are read in place)
struct ActSet {
    TqAct act[kTqMaxTokens];
    alignas(64) float d[kTqMaxTokens][2560 / 32];
};

ActSet& unpack_all(const void* const* act, int n, int nt) {
    thread_local ActSet set;
    for (int t = 0; t < nt; ++t) {
        const uint8_t* p = (const uint8_t*) act[t];
        for (int b = 0; b < n / 32; ++b) set.d[t][b] = tq_detail::tq_half_to_float(tq_detail::tq_ld16(p + (size_t) b * 36));
        set.act[t].blocks = p;
        set.act[t].d = set.d[t];
    }
    return set;
}

void rows_on(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0, int r1) {
    switch (isa()) {
        case Isa::avx512: tq_detail::tq_rows_avx512(kb, w, row_bytes, n, a, nt, out, r0, r1); break;
        case Isa::avx2: tq_detail::tq_rows_avx2(kb, w, row_bytes, n, a, nt, out, r0, r1); break;
        default: tq_detail::tq_rows_scalar(kb, w, row_bytes, n, a, nt, out, r0, r1); break;
    }
}

}  // namespace

namespace tq_detail {

const uint32_t* tq_state_table() {
    static std::once_flag once;
    std::call_once(once, fill_table);
    return g_table;
}

void tq_rows_scalar(int kb, const uint8_t* w, size_t row_bytes, int n, const TqAct* a, int nt, float* const* out, int r0,
                    int r1) {
    if (kb == 5) rows_scalar_t<5>(w, row_bytes, n, a, nt, out, r0, r1);
    else if (kb == 6) rows_scalar_t<6>(w, row_bytes, n, a, nt, out, r0, r1);
    else rows_scalar_t<7>(w, row_bytes, n, a, nt, out, r0, r1);
}

}  // namespace tq_detail

void tq_rows(int ggml_type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
             int r0, int r1) {
    const ActSet& s = unpack_all(act, n, nt);
    rows_on(tq_kbits(ggml_type), w, row_bytes, n, s.act, nt, out, r0, r1);
}

void tq_gu_rows(int ggml_type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                float* const* ff, int r0, int r1) {
    const ActSet& s = unpack_all(act, n, nt);
    thread_local std::vector<float> gbuf, ubuf;
    const size_t nr = (size_t) r1;   // row r of token t at [t * r1 + r]
    if (gbuf.size() < nr * kTqMaxTokens) {
        gbuf.resize(nr * kTqMaxTokens);
        ubuf.resize(nr * kTqMaxTokens);
    }
    float* gp[kTqMaxTokens];
    float* up[kTqMaxTokens];
    for (int t = 0; t < nt; ++t) {
        gp[t] = gbuf.data() + (size_t) t * nr;
        up[t] = ubuf.data() + (size_t) t * nr;
    }
    const int kb = tq_kbits(ggml_type);
    rows_on(kb, blob, gu_row, n, s.act, nt, gp, r0, r1);
    rows_on(kb, blob + up_off, gu_row, n, s.act, nt, up, r0, r1);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) {
            const float g = gp[t][r], u = up[t][r];
            ff[t][r] = (g / (1.f + std::exp(-g))) * u;   // native_gu_rows' SwiGLU expression
        }
}

void tq_dequant_row(int ggml_type, const uint8_t* row, int n, float* out) {
    const int kb = tq_kbits(ggml_type);
    const size_t bsz = tq_block_bytes(ggml_type);
    for (int i = 0; i < n / kTqBlock; ++i) {
        const uint8_t* blk = row + (size_t) i * bsz;
        const float d = tq_detail::tq_half_to_float(tq_detail::tq_ld16(blk));
        const uint8_t* qs = blk + 2;
        const uint32_t nbytes = 8u * (uint32_t) kb;
        for (int t = 0; t < 64; ++t) {
            const uint32_t off = (uint32_t) (63 - t) * (uint32_t) kb, b0 = off >> 3;
            const uint32_t w = (uint32_t) qs[b0] | ((uint32_t) qs[(b0 + 1) % nbytes] << 8) |
                               ((uint32_t) qs[(b0 + 2) % nbytes] << 16);
            const uint32_t s = (w >> (off & 7)) & tq_detail::kTqStateMask;
            const uint32_t pt = (s * 0x9e3779b1u) >> 21;
            out[kTqBlock * i + 2 * t] = d * tq_detail::tq_half_to_float(tq_lut_f16[2 * pt]);
            out[kTqBlock * i + 2 * t + 1] = d * tq_detail::tq_half_to_float(tq_lut_f16[2 * pt + 1]);
        }
    }
}

}  // namespace lamina::kernels::cpu
