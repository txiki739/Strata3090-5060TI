// mmq_quant_parity - the prompt experts' q8_1 inputs, each in one kernel, byte for byte the pair it replaces:
//   mmq::quantize_f16 with row ids (the routed rows from the FP16 input) against gather_rows16_f32 + mmq::quantize,
//   mmq::swiglu_quantize (H) against mmq::swiglu + mmq::quantize (gate/up rows split and interleaved),
//   and both with a Hadamard rotation in them (`signs`) against hadamard_rows between the pair,
// for every q8_1 layout MMQ reads (D4: Q8_0 IQ4_NL Q2_0 IQ3_S, DS4: Q5_1 Q4_K, D2S6: Q2_K), on rows routed as the prompt
// path routes them (10 of 512 experts a token, grouped by expert), with an all-zero token and row counts that are
// multiples of nothing.  --bench: both forms at a batch of 8192 rows (the engine's), ms and the rows' bytes moved.
#include "lamina/prefill/kernels.hpp"
#include "lamina/prefill/moe_mmq.hpp"
#include "lamina/kernels/iq_kernels.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = lamina::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <class T> T* as() const { return (T*) p; }
};

constexpr int64_t N = 2560, FF = 640;
constexpr int kExperts = 512, kUsed = 10;

struct Layout {
    const char* name;
    ggml_type type;
};
const Layout kTypes[] = {{"Q8_0 (D4)", GGML_TYPE_Q8_0},   {"IQ4_NL (D4)", GGML_TYPE_IQ4_NL}, {"Q2_0 (D4)", GGML_TYPE_Q2_0},
                         {"IQ3_S (D4)", GGML_TYPE_IQ3_S}, {"Q5_1 (DS4)", GGML_TYPE_Q5_1},    {"Q4_K (DS4)", GGML_TYPE_Q4_K},
                         {"Q2_K (D2S6)", GGML_TYPE_Q2_K}};

// the prompt path's rows: each token's 10 experts, rows grouped by expert; src[row] = its token
std::vector<int32_t> routed_rows(int64_t T, std::mt19937& rng) {
    std::vector<std::vector<int32_t>> by(kExperts);
    std::uniform_int_distribution<int> pick(0, kExperts - 1);
    for (int64_t t = 0; t < T; ++t) {
        std::vector<int> used;
        while ((int) used.size() < kUsed) {
            const int e = pick(rng);
            if (std::find(used.begin(), used.end(), e) == used.end()) used.push_back(e);
        }
        for (int e : used) by[(size_t) e].push_back((int32_t) t);
    }
    std::vector<int32_t> src;
    for (const auto& v : by) src.insert(src.end(), v.begin(), v.end());
    return src;
}

size_t diff_bytes(const Dev& a, const Dev& b, size_t n) {
    std::vector<uint8_t> x(n), y(n);
    ck(cudaMemcpy(x.data(), a.p, n, cudaMemcpyDeviceToHost), "download");
    ck(cudaMemcpy(y.data(), b.p, n, cudaMemcpyDeviceToHost), "download");
    size_t d = 0;
    for (size_t i = 0; i < n; ++i) d += x[i] != y[i];
    return d;
}

int g_fail = 0;
void report(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

// the MoE input: T tokens' FP16 rows (token 1 all zero, a few large values), the routed rows by id
void input_case(int64_t T, std::mt19937& rng, cudaStream_t s) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__half> x((size_t) (T * N));
    for (int64_t i = 0; i < T * N; ++i) {
        float v = nd(rng);
        if (i % 977 == 0) v *= 300.0f;
        if (T > 1 && i / N == 1) v = 0.0f;
        x[(size_t) i] = __float2half_rn(v);
    }
    const std::vector<int32_t> src = routed_rows(T, rng);
    const int64_t rows = (int64_t) src.size();
    Dev xd(x.size() * 2), sd(src.size() * 4), f32((size_t) rows * N * 4);
    ck(cudaMemcpy(xd.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(sd.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    const size_t qb = mmq::q8_bytes(rows, N);
    Dev qa(qb), qf(qb);
    for (const Layout& l : kTypes) {
        ck(cudaMemset(qa.p, 0xA5, qb), "fill");
        ck(cudaMemset(qf.p, 0xA5, qb), "fill");
        lamina::prefill::gather_rows16_f32(xd.as<uint16_t>(), sd.as<int32_t>(), f32.as<float>(), rows, N, s);
        mmq::quantize(f32.as<float>(), nullptr, qa.p, (int) l.type, N, N, rows, s);
        mmq::quantize_f16(xd.as<uint16_t>(), sd.as<int32_t>(), qf.p, (int) l.type, N, N, rows, s);
        ck(cudaStreamSynchronize(s), "sync");
        const size_t d = diff_bytes(qa, qf, qb);
        report(d == 0, std::string("input ") + l.name + ", " + std::to_string(T) + " tokens (" + std::to_string(rows) +
                           " rows): " + std::to_string(d) + " of " + std::to_string(qb) + " bytes differ");
    }
}

// H: `rows` gate/up rows (FP32, the MMQ product's output), row 0 all zero, gate values over SwiGLU's whole bend
void h_case(int64_t rows, bool interleaved, std::mt19937& rng, cudaStream_t s) {
    std::normal_distribution<float> gate(0.0f, 4.0f), up(0.0f, 2.0f);
    std::vector<float> gu((size_t) (rows * 2 * FF));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < FF; ++k) {
            const int64_t gi = interleaved ? r * 2 * FF + 2 * k : r * 2 * FF + k;
            const int64_t ui = interleaved ? gi + 1 : r * 2 * FF + FF + k;
            gu[(size_t) gi] = r == 0 ? 0.0f : gate(rng) * ((r * FF + k) % 811 == 0 ? 40.0f : 1.0f);
            gu[(size_t) ui] = r == 0 ? 0.0f : up(rng);
        }
    Dev gd(gu.size() * 4), h((size_t) rows * FF * 4);
    ck(cudaMemcpy(gd.p, gu.data(), gu.size() * 4, cudaMemcpyHostToDevice), "gu");
    const size_t qb = mmq::q8_bytes(rows, FF);
    Dev qa(qb), qf(qb);
    for (const Layout& l : kTypes) {
        ck(cudaMemset(qa.p, 0xA5, qb), "fill");
        ck(cudaMemset(qf.p, 0xA5, qb), "fill");
        mmq::swiglu(gd.as<float>(), h.as<float>(), rows, FF, interleaved, s);
        mmq::quantize(h.as<float>(), nullptr, qa.p, (int) l.type, FF, FF, rows, s);
        mmq::swiglu_quantize(gd.as<float>(), qf.p, (int) l.type, rows, FF, interleaved, s);
        ck(cudaStreamSynchronize(s), "sync");
        const size_t d = diff_bytes(qa, qf, qb);
        report(d == 0, std::string("H ") + l.name + (interleaved ? ", interleaved" : ", split") + ", " +
                           std::to_string(rows) + " rows: " + std::to_string(d) + " of " + std::to_string(qb) +
                           " bytes differ");
    }
}

// The folded experts' inputs: a sign vector of the width, rows rotated inside the quantizer against rotated between the two
std::vector<float> random_signs(int64_t n, std::mt19937& rng) {
    std::vector<float> v((size_t) n);
    for (auto& x : v) x = (rng() & 1) ? 1.0f : -1.0f;
    return v;
}

void rotated_case(int64_t T, std::mt19937& rng, cudaStream_t s) {
    const auto xsign = random_signs(N, rng), hsign = random_signs(FF, rng);
    Dev sx(N * 4), sh(FF * 4);
    ck(cudaMemcpy(sx.p, xsign.data(), N * 4, cudaMemcpyHostToDevice), "signs");
    ck(cudaMemcpy(sh.p, hsign.data(), FF * 4, cudaMemcpyHostToDevice), "signs");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__half> x((size_t) (T * N));
    for (auto& v : x) v = __float2half_rn(nd(rng));
    const std::vector<int32_t> src = routed_rows(T, rng);
    const int64_t rows = (int64_t) src.size();
    Dev xd(x.size() * 2), sd(src.size() * 4), f32((size_t) rows * N * 4);
    ck(cudaMemcpy(xd.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(sd.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    const size_t qb = mmq::q8_bytes(rows, N);
    Dev qa(qb), qf(qb);
    ck(cudaMemset(qa.p, 0xA5, qb), "fill");
    ck(cudaMemset(qf.p, 0xA5, qb), "fill");
    lamina::prefill::gather_rows16_f32(xd.as<uint16_t>(), sd.as<int32_t>(), f32.as<float>(), rows, N, s);
    lamina::kernels::hadamard_rows(f32.as<float>(), f32.as<float>(), rows, N, sx.as<float>(), s);
    mmq::quantize(f32.as<float>(), nullptr, qa.p, (int) GGML_TYPE_IQ3_S, N, N, rows, s);
    mmq::quantize_f16(xd.as<uint16_t>(), sd.as<int32_t>(), qf.p, (int) GGML_TYPE_IQ3_S, N, N, rows, s, sx.as<float>());
    ck(cudaStreamSynchronize(s), "sync");
    size_t d = diff_bytes(qa, qf, qb);
    report(d == 0, "input rotated, " + std::to_string(rows) + " rows: " + std::to_string(d) + " of " + std::to_string(qb) + " bytes differ");

    std::normal_distribution<float> gate(0.0f, 4.0f), up(0.0f, 2.0f);
    std::vector<float> gu((size_t) (rows * 2 * FF));
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < FF; ++k) {
            gu[(size_t) (r * 2 * FF + k)] = gate(rng);
            gu[(size_t) (r * 2 * FF + FF + k)] = up(rng);
        }
    Dev gd(gu.size() * 4), h((size_t) rows * FF * 4);
    ck(cudaMemcpy(gd.p, gu.data(), gu.size() * 4, cudaMemcpyHostToDevice), "gu");
    const size_t qh = mmq::q8_bytes(rows, FF);
    Dev ha(qh), hf(qh);
    ck(cudaMemset(ha.p, 0xA5, qh), "fill");
    ck(cudaMemset(hf.p, 0xA5, qh), "fill");
    mmq::swiglu(gd.as<float>(), h.as<float>(), rows, FF, false, s);
    lamina::kernels::hadamard_rows(h.as<float>(), h.as<float>(), rows, FF, sh.as<float>(), s);
    mmq::quantize(h.as<float>(), nullptr, ha.p, (int) GGML_TYPE_IQ3_S, FF, FF, rows, s);
    mmq::swiglu_quantize(gd.as<float>(), hf.p, (int) GGML_TYPE_IQ3_S, rows, FF, false, s, sh.as<float>());
    ck(cudaStreamSynchronize(s), "sync");
    d = diff_bytes(ha, hf, qh);
    report(d == 0, "H rotated, " + std::to_string(rows) + " rows: " + std::to_string(d) + " of " + std::to_string(qh) + " bytes differ");
}

// both forms at the engine's batch of 8192 rows: each call's median over 50 after a warm-up, in ms
void bench(cudaStream_t s) {
    std::mt19937 rng(7);
    const int64_t T = 819, rows = 8190;   // 819 tokens x 10
    std::vector<__half> x((size_t) (T * N));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = __float2half_rn(nd(rng));
    std::vector<int32_t> src = routed_rows(T, rng);
    Dev xd(x.size() * 2), sd(src.size() * 4), f32((size_t) rows * N * 4), gu((size_t) rows * 2 * FF * 4),
        h((size_t) rows * FF * 4), q(mmq::q8_bytes(rows, N));
    ck(cudaMemcpy(xd.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    ck(cudaMemcpy(sd.p, src.data(), src.size() * 4, cudaMemcpyHostToDevice), "src");
    ck(cudaMemset(gu.p, 0x3C, (size_t) rows * 2 * FF * 4), "gu");
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    auto time = [&](auto&& f) {
        for (int i = 0; i < 20; ++i) f();
        std::vector<float> t;
        for (int i = 0; i < 50; ++i) {
            ck(cudaEventRecord(a, s), "record");
            f();
            ck(cudaEventRecord(b, s), "record");
            ck(cudaEventSynchronize(b), "sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
            t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    const int ty = GGML_TYPE_IQ3_S, td = GGML_TYPE_IQ4_NL;
    const float in2 = time([&] {
        lamina::prefill::gather_rows16_f32(xd.as<uint16_t>(), sd.as<int32_t>(), f32.as<float>(), rows, N, s);
        mmq::quantize(f32.as<float>(), nullptr, q.p, ty, N, N, rows, s);
    });
    const float in1 = time([&] { mmq::quantize_f16(xd.as<uint16_t>(), sd.as<int32_t>(), q.p, ty, N, N, rows, s); });
    const float h2 = time([&] {
        mmq::swiglu(gu.as<float>(), h.as<float>(), rows, FF, false, s);
        mmq::quantize(h.as<float>(), nullptr, q.p, td, FF, FF, rows, s);
    });
    const float h1 = time([&] { mmq::swiglu_quantize(gu.as<float>(), q.p, td, rows, FF, false, s); });
    std::printf("bench, %lld rows: input gather + quantize %.3f ms, quantize_f16 by id %.3f ms (%.2fx); "
                "H swiglu + quantize %.3f ms, swiglu_quantize %.3f ms (%.2fx)\n",
                (long long) rows, in2, in1, in2 / in1, h2, h1, h2 / h1);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0), "device");
        std::printf("mmq_quant_parity on %s\n", prop.name);
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
            bench(s);
            return 0;
        }
        std::mt19937 rng(1525);
        for (int64_t T : {1, 2, 7, 333, 2099}) input_case(T, rng, s);
        for (int64_t T : {1, 7, 333}) rotated_case(T, rng, s);
        for (int64_t rows : {1, 3, 31, 1000, 8190})
            for (bool il : {false, true}) h_case(rows, il, rng, s);
        std::printf(g_fail ? "FAIL\n" : "PASS\n");
        return g_fail ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mmq_quant_parity: %s\n", e.what());
        return 1;
    }
}
