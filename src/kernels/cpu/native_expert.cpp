// src/kernels/cpu/native_expert.cpp - plan v0.3 P6: native (GGUF-form) experts on the CPU through ggml-cpu.
// See the header.  Nothing here is Lamina arithmetic: the activation quantizers and the row dot products are
// ggml-cpu's, so an IQ expert computes what llama.cpp's CPU backend computes for it.
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/expert.hpp"
#include "lamina/kernels/cpu/iq_avx512.hpp"
#include "lamina/kernels/cpu/iq_avx2.hpp"
#include "lamina/kernels/cpu/iq_avx1.hpp"
#include "lamina/kernels/cpu/kq_avx2.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"
#include "lamina/kernels/cpu/hadamard.hpp"
#include "lamina/kernels/cpu/tq_cpu.hpp"
#include "lamina/kernels/expf_contract.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace lamina::kernels::cpu {
namespace {

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

}  // namespace

bool native_experts_available() noexcept { return true; }

// The "is this format handled by that kernel" questions, answered here and not next to the kernels: iq_avx512.cpp is
// compiled for AVX-512 and iq_avx2.cpp / kq_avx2.cpp for AVX2, and native_gu_rows asks these on every CPU before it
// has checked what the CPU can run.  A function in a wide-ISA file may use that ISA anywhere in its body, so an
// answer that is only a comparison must not come from one (#795).
bool iq512_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22;
}

bool iq256_supported(int type) noexcept {
    return type == 16 || type == 17 || type == 18 || type == 21 || type == 22 || type == 23;
}

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    init_once();
    if (tq_supported(gu_type) || tq_supported(d_type)) {   // not ggml types here: tq_cpu.hpp's kernels, q8_1 activations
        if (!tq_supported(gu_type) || !tq_supported(d_type)) {
            err = "native experts: a layer of trellis experts with a down or gate/up of another format";
            return false;
        }
        if (n_embd > H || n_ff > FF || n_embd % kTqBlock || n_ff % kTqBlock) {
            err = "native experts: the trellis experts' geometry is not whole blocks of the pool's rows";
            return false;
        }
        f.gu_type = gu_type;
        f.d_type = d_type;
        f.gu_act = f.d_act = (int) GGML_TYPE_Q8_1;
        f.n_embd = n_embd;
        f.n_ff = n_ff;
        f.gu_row = (size_t) (n_embd / kTqBlock) * tq_block_bytes(gu_type);
        f.d_row = (size_t) (n_ff / kTqBlock) * tq_block_bytes(d_type);
        f.up_off = f.gu_row * (size_t) n_ff;
        f.down_off = 2 * f.up_off;
        f.bytes = f.down_off + f.d_row * (size_t) n_embd;
        f.act_bytes = (size_t) (n_embd / 32) * 36;
        f.h_bytes = (size_t) (n_ff / 32) * 36;
        if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
            err = "native experts: activation larger than the pool's buffers";
            return false;
        }
        return true;
    }
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    if (tg == nullptr || tg->vec_dot == nullptr || td == nullptr || td->vec_dot == nullptr) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(tg && tg->vec_dot ? d_type : gu_type);
        return false;
    }
    const ggml_type_traits_cpu* ag = traits(tg->vec_dot_type);
    const ggml_type_traits_cpu* ad = traits(td->vec_dot_type);
    if (ag == nullptr || ag->from_float == nullptr || ad == nullptr || ad->from_float == nullptr) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    if (n_embd > H || n_ff > FF) {   // the pool's per-token rows (ExpertPool's SplitBufMulti) are Flash-Next's size
        err = "native experts: an expert wider than the pool's buffers (at most " + std::to_string(FF) + " x " +
              std::to_string(H) + ")";
        return false;
    }
    if (n_embd % ggml_blck_size((ggml_type) gu_type) || n_ff % ggml_blck_size((ggml_type) d_type) ||
        n_embd % ggml_blck_size(tg->vec_dot_type) || n_ff % ggml_blck_size(td->vec_dot_type)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = (int) tg->vec_dot_type;
    f.d_act = (int) td->vec_dot_type;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    f.d_row = ggml_row_size((ggml_type) d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    f.h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

namespace {
// Q8_K (the activations of every i-quant row): ggml-cpu's x86 quantizer is the scalar reference, ~3 us per token
// and layer on the host before the pool can start; q8k_quant_avx2 writes the same bytes.  cpu_avx2_ok() too:
// iq_avx2.cpp is compiled for AVX2 (an AVX-only CPU, or LAMINA_FORCE_ISA=avx, keeps ggml's).  LAMINA_NO_Q8K_AVX2=1:
// ggml's on any CPU.
bool q8k_avx2(int type) {
    static const bool on = cpu_avx2_ok() && std::getenv("LAMINA_NO_Q8K_AVX2") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}
// the older CPUs' copy (iq_avx1.cpp): byte-identical like the AVX-2 one, and only where AVX2 is absent (same
// policy as LAMINA_NO_IQ128 above; LAMINA_NO_Q8K_AVX1=1 keeps ggml's scalar reference on those CPUs).
bool q8k_avx1(int type) {
    static const bool on = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("LAMINA_NO_Q8K_AVX1") == nullptr;
    return on && type == (int) GGML_TYPE_Q8_K;
}

bool g_exact_swiglu = false;
}  // namespace

void set_native_exact_swiglu(bool on) noexcept { g_exact_swiglu = on; }
bool native_exact_swiglu() noexcept { return g_exact_swiglu; }
// the GPU's exact kernels' operations in their order: expf_contract(-g), 1 + it, g over that, times u
float native_swiglu_exact(float g, float u) noexcept { return (g / (1.f + expf_contract(-g))) * u; }

int ggml_q5_K_mins_rounding() {
    init_once();
    // block_q5_K: d, dmin (fp16), 12 bytes of 6-bit scales and mins, qh[32], qs[128]; block_q8_K: d (float), qs[256],
    // bsums[16].  Quants 0, scale bytes 0 but byte 4 (block min 0 = m): the dot is the blocks' dmin * m * bsums[0].
    constexpr size_t QB = 176, YB = 292;
    const auto* tr = traits((int) GGML_TYPE_Q5_K);
    const float yd[] = {0.7310586f, 1.2345678f, 0.913f, 1.0571f, 0.3318f, 1.7779f, 0.6211f, 1.4142135f};
    for (int k = 0; k + 1 < (int) (sizeof yd / sizeof yd[0]); ++k) {
        uint8_t x[2 * QB] = {}, y[2 * YB] = {};
        const float dm[2] = {0.0137f, 0.00931f};
        const int m[2] = {37, 53}, bsum[2] = {1500, -1789};
        float dmin[2];
        for (int b = 0; b < 2; ++b) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(dm[b]);
            std::memcpy(x + b * QB + 2, &h, 2);
            x[b * QB + 4 + 4] = (uint8_t) m[b];
            std::memcpy(y + b * YB, &yd[k + b], 4);
            const int16_t s = (int16_t) bsum[b];
            std::memcpy(y + b * YB + 4 + 256, &s, 2);
            dmin[b] = -(yd[k + b] * ggml_fp16_to_fp32(h));
        }
        const float t0 = (float) (m[0] * bsum[0]), t1 = (float) (m[1] * bsum[1]);
        const float s0 = dmin[0] * t0;
        const float fused = std::fma(dmin[1], t1, s0);
        const float p1 = dmin[1] * t1;
        const float separate = s0 + p1;
        if (fused == separate) continue;   // these values round alike either way: the next pair
        float got = 0.f;
        tr->vec_dot(512, &got, 0, x, 0, y, 0, 1);
        return got == fused ? 1 : got == separate ? 0 : -1;
    }
    return -1;
}

void native_quant_act(const NativeFmt& f, const float* x, void* dst) {
    if (f.had_gu) {
        float r[H];
        std::memcpy(r, x, (size_t) f.n_embd * sizeof(float));
        hadamard_rotate(r, (int) f.n_embd, f.had_block, f.gu_signs);
        traits(f.gu_act)->from_float(r, dst, f.n_embd);
        return;
    }
    if (q8k_avx2(f.gu_act)) { q8k_quant_avx2(x, dst, f.n_embd); return; }
    if (q8k_avx1(f.gu_act)) { q8k_quant_avx1(x, dst, f.n_embd); return; }
    traits(f.gu_act)->from_float(x, dst, f.n_embd);
}

void native_quant_h(const NativeFmt& f, const float* h, void* dst) {
    if (f.had_d) {
        float r[FF];
        std::memcpy(r, h, (size_t) f.n_ff * sizeof(float));
        hadamard_rotate(r, (int) f.n_ff, f.had_block, f.d_signs);
        traits(f.d_act)->from_float(r, dst, f.n_ff);
        return;
    }
    if (q8k_avx2(f.d_act)) { q8k_quant_avx2(h, dst, f.n_ff); return; }
    if (q8k_avx1(f.d_act)) { q8k_quant_avx1(h, dst, f.n_ff); return; }
    traits(f.d_act)->from_float(h, dst, f.n_ff);
}

int native_gu_mt_min(int gu_type) {
    // #152: from how many tokens the multi-token kernels run (ggml's vec_dot below that).  The default 2 is the
    // measured-fastest rule, but a token's expert rows then round differently alone than in a group, so greedy output
    // can depend on how many drafts a verify window held.  LAMINA_IQ_MT_MIN=1 (opt-in, 0.1.30) uses the multi-token
    // kernels for every group: output independent of the drafting, at a measured -1..-3% decode on IQ3_S (AVX-512).
    static const char* env = std::getenv("LAMINA_IQ_MT_MIN");
    static const int mt_min = env ? std::atoi(env) : 2;
    // IQ3_S where the AVX-2 kernel gathers its grid (cpu_gather_fast, unless LAMINA_IQ256_GATHER=0) on a CPU without
    // AVX-512: there that kernel beats ggml's dot for ONE token too (14900KF P-core, an expert's gate/up rows: 0.34 ->
    // 0.19 ms; its E-cores, which keep the scalar decode: 0.71 -> 0.68), and in decode most CPU experts serve one token
    // of the window.  So IQ3_S takes it for every group, and those rows no longer depend on the drafting.  A machine-
    // wide rule, not a per-core one: every core rounds an expert the same.  Every probe here is in expert_layout.cpp
    // and cpu_avx2_ok() comes first, so no AVX2 code runs before the check.  LAMINA_IQ_MT_MIN set keeps its rule.
    // Opt-in (LAMINA_IQ3S_MT1=1): it changes a lone token's IQ3_S rounding on those CPUs, so the default stays 0.1.39's.
    static const bool iq3s_one = env == nullptr && std::getenv("LAMINA_IQ3S_MT1") != nullptr && cpu_avx2_ok() && std::getenv("LAMINA_NO_IQ256") == nullptr &&
                                 !cpu_avx512_ok() && iq256_gather_setting() != 0 && cpu_gather_fast();
    return iq3s_one && gu_type == 21 ? 1 : mt_min;
}

void native_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act, int nt, float* const* ff,
                    int r0, int r1) {
    if (tq_supported(f.gu_type)) {
        tq_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // the multi-token kernels decode the weights once for all tokens: 2.0-2.4x ggml-cpu at three tokens, no faster
    // at one (all are bound by the codebook lookups, ~5 GB/s per core), measured by native_expert_parity.  AVX-512
    // first, then the AVX-2 one (Zen 2/3, Intel 12th-14th gen).  LAMINA_NO_IQ512 drops an AVX-512 CPU to the
    // AVX-2 kernel, LAMINA_NO_IQ256 drops the AVX-2 kernel; ggml-cpu's single-token vec_dot is reached only with
    // both set (and on a CPU without AVX-512, LAMINA_NO_IQ512 changes nothing).
    static const bool avx512 = cpu_avx512_ok() && std::getenv("LAMINA_NO_IQ512") == nullptr;
    // LAMINA_NO_IQ256 drops the AVX-2 kernel.  cpu_avx2_ok() is ALSO required: iq_avx2.cpp and kq_avx2.cpp
    // are compiled /arch:AVX2 and use AVX2 and FMA3, so calling them on a CPU without either is an
    // illegal instruction, not a slow path.  The gate sat next to `avx512` above, which does test
    // cpu_avx512_ok(), and that asymmetry is what let a pre-Haswell CPU reach iq256_gu_rows.
    // Without it this falls through to ggml-cpu's vec_dot below, which is compiled for whatever
    // baseline the build selected (AVX1 here) and covers the same types - IQ2_XXS and IQ2_S among
    // them.  That path loops over tokens itself, so it is correct for any `nt`, not just one.
    static const bool avx2 = cpu_avx2_ok() && std::getenv("LAMINA_NO_IQ256") == nullptr;
    // The older-CPU tier (LAMINA_ISA_FLOOR=avx builds): without it every expert on such a CPU runs on ggml-cpu's
    // scalar _generic dots.  Only where AVX2 is absent, so a modern CPU's rounding never moves (and
    // LAMINA_NO_IQ256=1 there still means ggml, as before).  iq_avx1.cpp is compiled for AVX and gated on
    // cpu_avx1_ok().  Measured on an E5-2470 v2 (iq_avx1_parity --bench): 1.2-3.1x ggml's dot from two tokens on,
    // at one token ggml's auto-vectorized generic loop matches the 128-bit kernel - so the same #152 mt_min rule
    // as the AVX-2 kernels, and LAMINA_IQ_MT_MIN=1 takes the kernel for one token too (its rows are then
    // independent of the drafting: iq_avx1_parity checks every width bit for bit).
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("LAMINA_NO_IQ128") == nullptr;
    const int mt_min = native_gu_mt_min(f.gu_type);   // #152
    // Unsloth UD-Q4_K_XL's Q4_K gate/up: the multi-token kernel is bit-exact against ggml's per-token dot (any group
    // size, no #152 rule).  Opt-in, LAMINA_KQ256=1: measured no faster in the engine (a window's expert groups hold
    // ~1.4 tokens and the weights stay in L1 across ggml's per-token calls; 1.01-1.13x in native_expert_parity).
    static const bool kq = [] { const char* v = std::getenv("LAMINA_KQ256"); return cpu_avx2_ok() && v != nullptr && std::atoi(v) != 0; }();
    if (kq && f.gu_type == 12 && nt >= 2) {   // one token: ggml's own dot below (the same bits, less overhead)
        kq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    // A format with only an AVX-2 kernel (IQ4_XS, #415) takes it on AVX-2 CPUs only: an AVX-512 CPU keeps ggml-cpu for
    // it, as before (its rows would round differently).  Each kernel only for the formats it implements: falling
    // through an empty switch would leave ff unwritten instead of falling back to ggml-cpu.
    static const bool cpu512 = cpu_avx512_ok();
    if (nt >= mt_min && (iq512_supported(f.gu_type) || (!cpu512 && iq256_supported(f.gu_type)))) {
        if (avx512 && iq512_supported(f.gu_type)) {
            iq512_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
        if (avx2 && iq256_supported(f.gu_type)) {
            iq256_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
            return;
        }
    }
    if (avx1 && nt >= mt_min && iq128_supported(f.gu_type)) {
        iq128_gu_rows(f.gu_type, blob, f.gu_row, f.up_off, (int) f.n_embd, act, nt, ff, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.gu_type)->vec_dot;
    const int n = (int) f.n_embd;
    const bool exact = g_exact_swiglu;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = blob + (size_t) r * f.gu_row;
        const uint8_t* ur = blob + f.up_off + (size_t) r * f.gu_row;
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, gr, 0, act[t], 0, 1);
            dot(n, &u, 0, ur, 0, act[t], 0, 1);
            ff[t][r] = exact ? native_swiglu_exact(g, u) : (g / (1.f + std::exp(-g))) * u;
        }
    }
}

void native_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* hq, int nt, float* const* out,
                      int r0, int r1) {
    if (tq_supported(f.d_type)) {
        tq_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // IQ4_NL down rows: the AVX-2 multi-token kernel decodes the nibbles and absolutises the weights once per
    // block instead of once per token; ggml-cpu's dot is single-token.  LAMINA_NO_IQ4NL falls back to it.
    static const bool iq4nl_mt = std::getenv("LAMINA_NO_IQ4NL") == nullptr;
    static const int mt_min = [] { const char* e = std::getenv("LAMINA_IQ_MT_MIN"); return e ? std::atoi(e) : 2; }();
    static const bool kq = [] { const char* v = std::getenv("LAMINA_KQ256"); return v != nullptr && std::atoi(v) != 0; }();
    // Both multi-token kernels below are /arch:AVX2 translation units (kq_avx2.cpp and iq_avx2.cpp),
    // so a CPU without AVX2 has to reach ggml-cpu's vec_dot instead - same reasoning as the gate/up
    // rows above, where `avx512` tested cpu_avx512_ok() and `avx2` did not.
    if (cpu_avx2_ok() && kq && nt >= 2 && (f.d_type == 7 || f.d_type == 8)) {   // Q5_1 / Q8_0 down: bit-exact, any group size
        kq256_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    if (cpu_avx2_ok() && nt >= mt_min && f.d_type == 20 && iq4nl_mt) {   // #152: the same rule as the gate/up rows
        iq4nl256_down_rows(blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    // the older-CPU tier: IQ4_NL and Q2_0 down rows in 128-bit (ggml-cpu's dots for both are scalar here)
    static const bool avx1 = !cpu_avx2_ok() && cpu_avx1_ok() && std::getenv("LAMINA_NO_IQ128") == nullptr;
    if (avx1 && iq128_down_supported(f.d_type)) {
        iq128_down_rows(f.d_type, blob + f.down_off, f.d_row, (int) f.n_ff, hq, nt, out, r0, r1);
        return;
    }
    const ggml_vec_dot_t dot = traits(f.d_type)->vec_dot;
    const int n = (int) f.n_ff;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = blob + f.down_off + (size_t) r * f.d_row;
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, dr, 0, hq[t], 0, 1);
            out[t][r] = s;
        }
    }
}

}  // namespace lamina::kernels::cpu
