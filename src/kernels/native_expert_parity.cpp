// src/kernels/native_expert_parity.cpp - plan v0.3 P6: one native expert three ways.
//
//     build/native_expert_parity <shard1.gguf> [layer ...]      real rows (a split model's shards are found by name)
//     build/native_expert_parity --synthetic GU/DOWN ...        random weights quantized by ggml, e.g. iq1_s/iq4_nl
//
// (a) float reference: ggml's own dequantizer (`to_float`) and a float SwiGLU expert, (b) the CPU path
// (ggml-cpu vec_dot with its quantized activations), (c) the GPU path (`native_expert_grouped`, q8_1
// activations: --expert-math fast).  (b) and (c) each differ from (a) by their activation rounding only (about 1e-2
// relative).  (c2) where --expert-math exact has kernels (the K-quant formats): the GPU in the CPU's activation formats
// against the CPU with its exact SwiGLU, bit for bit, the hidden rows' bytes too.  (d) the GPU dequantizers of the
// prompt path and the embedding, against `to_float`: exact.
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/cpu/hadamard.hpp"
#include "lamina/kernels/cpu/tq_cpu.hpp"
#include "lamina/core/pack_image.hpp"
#include "lamina/kernels/cpu/expert.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"
#include "lamina/kernels/cpu/iq_avx512.hpp"
#include "ggml-cpu.h"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/quantize_act.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <cstring>
#include <vector>

namespace cpu = lamina::kernels::cpu;

namespace {

constexpr int NT = 3;
constexpr int64_t H = 2560, FF = 640;

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

// --expert-math exact's Q8_0 hidden rows at a rounding tie: 0.0196850393 * (127 / 1) rounds to 2.5 in float, which
// ggml's x86 quantizer (a multiply by 127 / max, then round half to even) makes 2
int q8_0_tie_fixture(cudaStream_t s) {
    std::vector<float> x(32, 0.0f);
    x[0] = 1.0f;
    x[1] = 0.019685039296746254f;
    float* dx = nullptr;
    uint8_t* dy = nullptr;
    cudaMalloc((void**) &dx, x.size() * sizeof(float));
    cudaMalloc((void**) &dy, 34);
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    lamina::kernels::native_q8_0_contract_probe(dx, 32, dy, s);
    cudaStreamSynchronize(s);
    std::vector<uint8_t> got(34);
    cudaMemcpy(got.data(), dy, got.size(), cudaMemcpyDeviceToHost);
    int failures = (got[2] != 127 || got[3] != 2);
    for (size_t i = 4; i < got.size(); ++i) failures += got[i] != 0;
    std::printf("q8_0_tie_fixture: q0=%u q1=%u expected=127,2 %s\n",
                (unsigned) got[2], (unsigned) got[3], failures ? "FAIL" : "PASS");
    cudaFree(dx);
    cudaFree(dy);
    return failures;
}

double us_since(std::chrono::steady_clock::time_point t0, int it) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
}

// The three ways and the dequantizers for one expert blob; returns the number of failed checks.
int check_blob(const cpu::NativeFmt& f, const std::vector<uint8_t>& blob, int seed, const std::string& label,
               cudaStream_t s) {
    int failures = 0;
    // (a) the float reference
    const bool tq = cpu::tq_supported(f.gu_type);
    const auto to_float = [&](int type, const uint8_t* row, float* out, int64_t n) {
        if (tq) cpu::tq_dequant_row(type, row, (int) n, out);
        else ggml_get_type_traits((ggml_type) type)->to_float(row, out, n);
    };
    const auto type_name = [&](int type) { return tq ? lamina::ggml_type_name((uint32_t) type) : ggml_type_name((ggml_type) type); };
    std::vector<float> G((size_t) FF * H), U((size_t) FF * H), D((size_t) H * FF);
    for (int64_t r = 0; r < FF; ++r) {
        to_float(f.gu_type, blob.data() + r * f.gu_row, G.data() + r * H, H);
        to_float(f.gu_type, blob.data() + f.up_off + r * f.gu_row, U.data() + r * H, H);
    }
    for (int64_t r = 0; r < H; ++r) to_float(f.d_type, blob.data() + f.down_off + r * f.d_row, D.data() + r * FF, FF);
    std::mt19937 rng(11 + seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) NT * H);
    for (auto& v : x) v = nd(rng);
    std::vector<float> ref((size_t) NT * H), got_c((size_t) NT * H), got_g((size_t) NT * H);
    const bool exact = lamina::kernels::native_expert_exact_supported(f.gu_type, f.d_type, f.gu_act, f.d_act);
    std::vector<float> got_cx((size_t) NT * H), got_gx((size_t) NT * H);
    std::vector<std::vector<uint8_t>> cpu_hqx(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
    std::vector<float> xr = x;   // the inputs the folded weights take (hadamard.hpp)
    if (f.had_gu)
        for (int k = 0; k < NT; ++k) cpu::hadamard_rotate(xr.data() + k * H, (int) H, f.had_block, f.gu_signs);
    for (int k = 0; k < NT; ++k) {
        std::vector<float> h(FF);
        for (int64_t r = 0; r < FF; ++r) {
            double g = 0, u = 0;
            for (int64_t i = 0; i < H; ++i) { g += (double) G[r * H + i] * xr[k * H + i]; u += (double) U[r * H + i] * xr[k * H + i]; }
            h[r] = (float) (g / (1.0 + std::exp(-g)) * u);
        }
        if (f.had_d) cpu::hadamard_rotate(h.data(), (int) FF, f.had_block, f.d_signs);
        for (int64_t r = 0; r < H; ++r) {
            double o = 0;
            for (int64_t i = 0; i < FF; ++i) o += (double) D[r * FF + i] * h[i];
            ref[k * H + r] = (float) o;
        }
    }
    // (b) the CPU
    {
        std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
        std::vector<std::vector<float>> ff(NT, std::vector<float>(FF));
        const void* a[NT];
        float* ffp[NT];
        const void* hp[NT];
        float* op[NT];
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_act(f, x.data() + k * H, act[k].data());
            a[k] = act[k].data();
            ffp[k] = ff[k].data();
        }
        cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
        const int it = 50;
        if (tq) {   // the kernels' times on one thread, and every ISA against the scalar reference, bit for bit
            float* f1[1] = {ffp[0]};
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) cpu::native_gu_rows(f, blob.data(), a, 1, f1, 0, (int) FF);
            const double us1 = us_since(t0, it);
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) cpu::native_gu_rows(f, blob.data(), a, NT, ffp, 0, (int) FF);
            std::printf("          gate+up one thread, weights cache-resident: %.0f us one token (%.2f GB/s), %.0f us %d tokens\n",
                        us1, 2.0 * f.up_off / us1 / 1e3, us_since(t0, it), NT);
            size_t ndiff = 0;
            std::vector<std::vector<float>> refs(NT, std::vector<float>(FF)), gots(NT, std::vector<float>(FF));
            float* rp[NT];
            float* gp[NT];
            std::vector<float> dg(NT * 80);
            std::vector<cpu::tq_detail::TqAct> ta(NT);
            for (int k = 0; k < NT; ++k) {
                for (int b = 0; b < 80; ++b) {
                    uint16_t h16;
                    std::memcpy(&h16, act[k].data() + (size_t) b * 36, 2);
                    dg[(size_t) k * 80 + b] = ggml_fp16_to_fp32(h16);
                }
                ta[k].blocks = act[k].data();
                ta[k].d = dg.data() + (size_t) k * 80;
                rp[k] = refs[k].data();
                gp[k] = gots[k].data();
            }
            const int kb = cpu::tq_kbits(f.gu_type);
            cpu::tq_detail::tq_rows_scalar(kb, blob.data(), f.gu_row, (int) H, ta.data(), NT, rp, 0, (int) FF);
            if (cpu::cpu_avx2_ok()) {
                cpu::tq_detail::tq_rows_avx2(kb, blob.data(), f.gu_row, (int) H, ta.data(), NT, gp, 0, (int) FF);
                for (int k = 0; k < NT; ++k) ndiff += std::memcmp(refs[k].data(), gots[k].data(), (size_t) FF * 4) != 0;
            }
            if (cpu::cpu_avx512_ok()) {
                cpu::tq_detail::tq_rows_avx512(kb, blob.data(), f.gu_row, (int) H, ta.data(), NT, gp, 0, (int) FF);
                for (int k = 0; k < NT; ++k) ndiff += std::memcmp(refs[k].data(), gots[k].data(), (size_t) FF * 4) != 0;
            }
            std::printf("          trellis rows, AVX2 and AVX-512 against scalar: %zu differ  %s\n", ndiff, ndiff == 0 ? "ok" : "FAIL");
            failures += ndiff != 0;
        }
        const auto* tc = tq ? nullptr : ggml_get_type_traits_cpu((ggml_type) f.gu_type);
        if (cpu::iq512_supported(f.gu_type) && cpu::cpu_avx512_ok()) {   // its kernel only where the CPU runs it
            // the AVX-512 rows against ggml's own vec_dot, same Q8_K activations: float-order differences only
            std::vector<float> g512((size_t) NT * FF), gref((size_t) NT * FF);
            float* gp[NT];
            for (int k = 0; k < NT; ++k) gp[k] = g512.data() + k * FF;
            cpu::iq512_rows(f.gu_type, blob.data(), f.gu_row, (int) H, a, NT, gp, 0, (int) FF);
            for (int k = 0; k < NT; ++k)
                for (int64_t r = 0; r < FF; ++r)
                    tc->vec_dot((int) H, &gref[k * FF + r], 0, blob.data() + r * f.gu_row, 0, a[k], 0, 1);
            std::printf("          %s AVX-512 gate rows vs ggml vec_dot: rel %.2e\n",
                        ggml_type_name((ggml_type) f.gu_type), rel(g512, gref));
            float* f1[1] = {ffp[0]};
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) cpu::iq512_gu_rows(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, 1, f1, 0, (int) FF);
            const double us1 = us_since(t0, it);
            t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) cpu::iq512_gu_rows(f.gu_type, blob.data(), f.gu_row, f.up_off, (int) H, a, NT, ffp, 0, (int) FF);
            std::printf("          gate+up one thread, weights cache-resident: avx512 %.0f us one token (%.2f GB/s), "
                        "%.0f us %d tokens\n", us1, 2.0 * f.up_off / us1 / 1e3, us_since(t0, it), NT);
        }
        if (!tq) {   // ggml-cpu's own rows, one thread (what the pool runs for formats without an AVX-512 kernel)
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i)
                for (int64_t r = 0; r < FF; ++r) {
                    float gg, uu;
                    tc->vec_dot((int) H, &gg, 0, blob.data() + r * f.gu_row, 0, a[0], 0, 1);
                    tc->vec_dot((int) H, &uu, 0, blob.data() + f.up_off + r * f.gu_row, 0, a[0], 0, 1);
                }
            const double usg = us_since(t0, it);
            std::printf("          gate+up one thread, weights cache-resident: ggml %.0f us one token (%.2f GB/s)\n", usg,
                        2.0 * f.up_off / usg / 1e3);
        }
        std::vector<std::vector<uint8_t>> hq(NT, std::vector<uint8_t>(cpu::kNativeHBytes));
        for (int k = 0; k < NT; ++k) {
            cpu::native_quant_h(f, ff[k].data(), hq[k].data());
            hp[k] = hq[k].data();
            op[k] = got_c.data() + k * H;
        }
        cpu::native_down_rows(f, blob.data(), hp, NT, op, 0, (int) H);
        if (exact) {   // the pool under --expert-math exact
            std::vector<std::vector<float>> ffx(NT, std::vector<float>(FF));
            float* ffxp[NT];
            const void* hxp[NT];
            float* oxp[NT];
            for (int k = 0; k < NT; ++k) ffxp[k] = ffx[k].data();
            cpu::set_native_exact_swiglu(true);
            cpu::native_gu_rows(f, blob.data(), a, NT, ffxp, 0, (int) FF);
            cpu::set_native_exact_swiglu(false);
            for (int k = 0; k < NT; ++k) {
                cpu::native_quant_h(f, ffx[k].data(), cpu_hqx[k].data());
                hxp[k] = cpu_hqx[k].data();
                oxp[k] = got_cx.data() + k * H;
            }
            cpu::native_down_rows(f, blob.data(), hxp, NT, oxp, 0, (int) H);
        }
        if (f.d_type == 42) {
            // (b2) the AVX-512 GGUF-layout Q2_0 kernel the pool uses for Q2_0 down projections
            std::vector<cpu::ActQ> a2(NT);
            const cpu::ActQ* ap[NT];
            std::vector<float> alt((size_t) NT * H);
            float* altp[NT];
            for (int k = 0; k < NT; ++k) {
                cpu::act_quant_q8_1(ff[k].data(), (int) FF, a2[k]);
                ap[k] = &a2[k];
                altp[k] = alt.data() + k * H;
            }
            cpu::q2_0_gguf_rows_multi(blob.data() + f.down_off, f.d_row, (int) (FF / 64), ap, NT, altp, 0, (int) H);
            std::printf("          q2_0 AVX-512 down vs ggml down: rel %.2e\n", rel(alt, got_c));
        }
    }
    // (c) the GPU: one group holding the NT entries, fast and (c2) exact; (d) its dequantizers
    {
        void *dblob, *dx, *dxq, *dscr;
        float* dout;
        unsigned long long* dptr;
        int32_t *dstart, *dn, *ddst, *dtok;
        cudaMalloc(&dblob, blob.size());
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dxq, (size_t) NT * (std::max)(H / 32 * 36, H / 256 * 292));
        cudaMalloc(&dscr, lamina::kernels::native_expert_scratch_bytes(NT, FF, f.had_d));
        cudaMalloc((void**) &dout, (size_t) NT * H * 4);
        cudaMalloc((void**) &dptr, 8);
        cudaMalloc((void**) &dstart, 8);
        cudaMalloc((void**) &dn, 4);
        cudaMalloc((void**) &ddst, NT * 4);
        cudaMalloc((void**) &dtok, NT * 4);
        cudaMemcpy(dblob, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        const unsigned long long p = (unsigned long long) dblob;
        const int32_t st[2] = {0, NT}, one = 1, idx[NT] = {0, 1, 2};
        cudaMemcpy(dptr, &p, 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dstart, st, 8, cudaMemcpyHostToDevice);
        cudaMemcpy(dn, &one, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(ddst, idx, NT * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dtok, idx, NT * 4, cudaMemcpyHostToDevice);
        auto run = [&](bool contract, std::vector<float>& out) {
            auto L = contract ? lamina::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF, f.gu_act, f.d_act)
                              : lamina::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
            if (f.had_d) {
                L.had_block = f.had_block;
                L.d_signs = lamina::kernels::hadamard_device_signs(f.d_signs, (int) FF);
            }
            if (f.had_gu)
                lamina::kernels::hadamard_rows((const float*) dx, (float*) dx, NT, H,
                                               lamina::kernels::hadamard_device_signs(f.gu_signs, (int) H), s);
            if (L.gu_act == 15)
                lamina::kernels::quantize_q8_K((const float*) dx, (uint8_t*) dxq, (int64_t) NT * H, s);
            else
                lamina::kernels::quantize_q8_1_rows((const float*) dx, NT, H, dxq, s);
            const int64_t xstride = L.gu_act == 15 ? (H / 256) * 292 : (H / 32) * 36;
            lamina::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, 1, NT, dxq, xstride, dscr, dout, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
        };
        run(false, got_g);
        if (f.had_gu) {   // run rotated dx in place: the kernel's floats against hadamard_rotate's
            std::vector<float> dr(x.size());
            cudaMemcpy(dr.data(), dx, dr.size() * 4, cudaMemcpyDeviceToHost);
            size_t nd = 0;
            for (size_t i = 0; i < dr.size(); ++i) nd += std::memcmp(&dr[i], &xr[i], 4) != 0;
            std::printf("          GPU rotation vs the CPU's: %zu of %zu floats differ  %s\n", nd, dr.size(), nd == 0 ? "ok" : "FAIL");
            failures += nd != 0;
        }
        if (exact) {
            run(true, got_gx);
            // the hidden rows (the down projection's input, in the CPU's d_act format) and the outputs, bit for bit
            size_t hdiff = 0, odiff = 0;
            for (int k = 0; k < NT; ++k) {
                std::vector<uint8_t> ghq(f.h_bytes);
                cudaMemcpy(ghq.data(), (const uint8_t*) dscr + (size_t) k * f.h_bytes, f.h_bytes, cudaMemcpyDeviceToHost);
                for (size_t j = 0; j < f.h_bytes; ++j) hdiff += ghq[j] != cpu_hqx[k][j];
            }
            for (size_t i = 0; i < got_gx.size(); ++i) odiff += std::memcmp(&got_gx[i], &got_cx[i], 4) != 0;
            std::printf("          exact: hidden bytes differing %zu/%zu, outputs differing %zu/%zu  %s\n", hdiff,
                        (size_t) NT * f.h_bytes, odiff, got_gx.size(), hdiff == 0 && odiff == 0 ? "ok" : "FAIL");
            if (hdiff != 0 || odiff != 0) ++failures;
            // every row's dot alone against ggml's vec_dot, on the CPU's activation bytes
            size_t ddiff = 0, drows = 0;
            float* drow = nullptr;
            void* dact = nullptr;
            cudaMalloc((void**) &drow, (size_t) (std::max)(FF, H) * 4);
            cudaMalloc(&dact, cpu::kNativeActBytes);
            std::vector<float> gd((size_t) (std::max)(FF, H));
            for (int k = 0; k < NT; ++k) {
                std::vector<uint8_t> act(cpu::kNativeActBytes);
                cpu::native_quant_act(f, x.data() + k * H, act.data());
                const struct { int type; size_t off, row_bytes; int64_t rows; const uint8_t* a; int64_t n; size_t ab; } sets[3] = {
                    {f.gu_type, 0, f.gu_row, FF, act.data(), H, f.act_bytes},
                    {f.gu_type, f.up_off, f.gu_row, FF, act.data(), H, f.act_bytes},
                    {f.d_type, f.down_off, f.d_row, H, cpu_hqx[k].data(), FF, f.h_bytes}};
                for (const auto& set : sets) {
                    cudaMemcpy(dact, set.a, set.ab, cudaMemcpyHostToDevice);
                    lamina::kernels::native_exact_dot_rows(set.type, (const uint8_t*) dblob + set.off, set.row_bytes,
                                                           (int) set.rows, dact, set.n, drow, s);
                    cudaStreamSynchronize(s);
                    cudaMemcpy(gd.data(), drow, (size_t) set.rows * 4, cudaMemcpyDeviceToHost);
                    const auto* tc = ggml_get_type_traits_cpu((ggml_type) set.type);
                    for (int64_t r = 0; r < set.rows; ++r) {
                        float v = 0.f;
                        tc->vec_dot((int) set.n, &v, 0, blob.data() + set.off + (size_t) r * set.row_bytes, 0, set.a, 0, 1);
                        ddiff += std::memcmp(&v, &gd[(size_t) r], 4) != 0;
                        ++drows;
                    }
                }
            }
            std::printf("          exact: dots against ggml's vec_dot, %zu of %zu rows differing  %s\n", ddiff, drows,
                        ddiff == 0 ? "ok" : "FAIL");
            if (ddiff != 0) ++failures;
            cudaFree(drow);
            cudaFree(dact);
        }
        float* dq = nullptr;
        cudaMalloc((void**) &dq, (size_t) FF * H * 4);
        std::vector<float> got_dq((size_t) FF * H);
        double dq_err = 0.0;
        const struct { const uint8_t* src; int type; const std::vector<float>* want; } mats[3] = {
            {(const uint8_t*) dblob, f.gu_type, &G}, {(const uint8_t*) dblob + f.up_off, f.gu_type, &U},
            {(const uint8_t*) dblob + f.down_off, f.d_type, &D}};
        for (const auto& m : mats) {
            lamina::kernels::iq_dequant_f32(m.type, m.src, FF * H, dq, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(got_dq.data(), dq, got_dq.size() * 4, cudaMemcpyDeviceToHost);
            dq_err = (std::max)(dq_err, rel(got_dq, *m.want));
        }
        std::printf("          GPU dequant vs ggml to_float: rel %.2e\n", dq_err);
        if (!(dq_err < 1e-6)) ++failures;
        cudaFree(dq);
        cudaFree(dblob); cudaFree(dx); cudaFree(dxq); cudaFree(dscr); cudaFree(dout); cudaFree(dptr);
        cudaFree(dstart); cudaFree(dn); cudaFree(ddst); cudaFree(dtok);
    }
    const double ec = rel(got_c, ref), eg = rel(got_g, ref), ecg = rel(got_c, got_g);
    const bool ok = ec < 3e-2 && eg < 3e-2 && std::isfinite(ec) && std::isfinite(eg);
    std::printf("%-9s %-8s/%-7s blob %8zu  cpu rel %.2e  gpu rel %.2e  cpu-gpu %.2e  %s\n", label.c_str(),
                type_name(f.gu_type), type_name(f.d_type), f.bytes, ec, eg, ecg,
                ok ? "ok" : "FAIL");
    return failures + (ok ? 0 : 1);
}

int type_of(std::string name) {
    for (auto& c : name) c = (char) std::tolower((unsigned char) c);
    if (name == "tqp5" || name == "tqp6" || name == "tqp7") return 147 + (name[3] - '5');   // not ggml's
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* n = ggml_type_name((ggml_type) t);
        if (n == nullptr) continue;
        std::string m(n);
        for (auto& c : m) c = (char) std::tolower((unsigned char) c);
        if (m == name) return t;
    }
    return -1;
}

// Random weights (rows of different magnitude) quantized by ggml into the blob layout.
std::vector<uint8_t> synthetic_blob(const cpu::NativeFmt& f, int seed) {
    std::vector<uint8_t> blob(f.bytes);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    if (cpu::tq_supported(f.gu_type)) {   // any trellis stream is a valid one: random steps, a scale a block
        std::uniform_int_distribution<int> byte(0, 255);
        const size_t bb = cpu::tq_block_bytes(f.gu_type);
        for (size_t at = 0; at + bb <= blob.size(); at += bb) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(0.01f * (0.5f + (float) (byte(rng) % 7) / 7.0f));
            std::memcpy(blob.data() + at, &d, 2);
            for (size_t i = 2; i < bb; ++i) blob[at + i] = (uint8_t) byte(rng);
        }
        return blob;
    }
    auto quant = [&](int type, int64_t rows, int64_t cols, uint8_t* dst) {
        std::vector<float> w((size_t) (rows * cols));
        for (int64_t r = 0; r < rows; ++r) {
            const float scale = 0.02f * (0.5f + (float) (r % 7) / 7.0f);
            for (int64_t c = 0; c < cols; ++c) w[(size_t) (r * cols + c)] = scale * nd(rng);
        }
        const std::vector<float> imatrix((size_t) cols, 1.0f);
        ggml_quantize_chunk((ggml_type) type, w.data(), dst, 0, rows, cols,
                            ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr);
    };
    quant(f.gu_type, FF, H, blob.data());
    quant(f.gu_type, FF, H, blob.data() + f.up_off);
    quant(f.d_type, H, FF, blob.data() + f.down_off);
    return blob;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: native_expert_parity <shard1.gguf> [layer ...]\n"
                             "       native_expert_parity --synthetic GU/DOWN ...   (ggml type names, e.g. q6_K/q8_0)\n");
        return 2;
    }
    int failures = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int q5k = cpu::ggml_q5_K_mins_rounding();   // the exact Q5_K dots follow this build's ggml, as the engine's
    lamina::kernels::set_exact_q5_K_fma(q5k == 1);
    std::printf("ggml's Q5_K mins: %s\n", q5k == 1 ? "one FMA" : q5k == 0 ? "a product, then a sum" : "UNKNOWN");
    if (q5k < 0) ++failures;
    failures += q8_0_tie_fixture(s);
    if (std::string(argv[1]) == "--synthetic") {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            const size_t slash = arg.find('/');
            const int gu = slash == std::string::npos ? -1 : type_of(arg.substr(0, slash));
            const int dn = slash == std::string::npos ? -1 : type_of(arg.substr(slash + 1));
            cpu::NativeFmt f;
            std::string err;
            if (gu < 0 || dn < 0 || !cpu::native_fmt(gu, dn, H, FF, f, err) ||
                !lamina::kernels::native_expert_supported(gu, dn, H, FF)) {
                std::printf("%s: %s\n", arg.c_str(), err.empty() ? "not a supported pair of GGML types" : err.c_str());
                ++failures;
                continue;
            }
            if (cpu::tq_supported(gu)) {   // folded with identity signs
                f.had_block = 128;
                f.had_gu = f.had_d = true;
            }
            failures += check_blob(f, synthetic_blob(f, i), i, "synthetic", s);
        }
    } else {
        const lamina::GgufModel model(lamina::gguf_split_paths(argv[1]));
        lamina::core::PackImage pack;
        {
            std::string err;
            if (!lamina::core::PackImage::build(argv[1], pack, err)) {
                std::printf("%s\n", err.c_str());
                return 1;
            }
            cpu::hadamard_set(pack.hadamard());
        }
        std::vector<int> layers;
        for (int i = 2; i < argc; ++i) layers.push_back(std::atoi(argv[i]));
        if (layers.empty()) layers = {0, 1, 2, 3, 20, 47};
        const int E = 7;
        for (int l : layers) {
            const lamina::TensorInfo* t[3] = {};
            const uint8_t* data[3] = {};
            const char* roles[3] = {"gate", "up", "down"};
            for (int r = 0; r < 3; ++r) {
                size_t at = 0;
                t[r] = model.find("blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight", &at);
                if (t[r]) data[r] = model.shard(at).tensor_data(*t[r]);
            }
            if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); ++failures; continue; }
            cpu::NativeFmt f;
            std::string err;
            if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) {
                std::printf("layer %d: %s\n", l, err.c_str()); ++failures; continue;
            }
            if (const cpu::HadamardSpec& hs = cpu::hadamard_spec(); hs.any()) {
                f.had_block = hs.block;
                f.had_gu = hs.gu(l);
                f.had_d = hs.down(l);
                f.gu_signs = f.had_gu ? cpu::hadamard_signs((int) H) : nullptr;
                f.d_signs = f.had_d ? cpu::hadamard_signs((int) FF) : nullptr;
            }
            std::vector<uint8_t> blob(f.bytes);
            std::memcpy(blob.data(), data[0] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.up_off, data[1] + (size_t) E * f.up_off, f.up_off);
            std::memcpy(blob.data() + f.down_off, data[2] + (size_t) E * (f.bytes - f.down_off), f.bytes - f.down_off);
            failures += check_blob(f, blob, l, "layer " + std::to_string(l), s);
        }
    }
    std::printf("native_expert_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
