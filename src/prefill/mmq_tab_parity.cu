// mmq_tab_parity - the prompt experts' products read in place (mmq::Product::tab: each expert's matrix where its blob
// lies) byte for byte against the gathered products they replace (the experts copied `expert_bytes` apart, a zeroed
// 4096-byte tail after the last), for the configs' files' gate/up types (IQ2_S IQ3_XXS IQ3_S IQ4_XS Q4_K Q5_K) at
// their shape [1280, 2560] and down types (Q2_0 IQ4_NL Q5_0 Q5_1 Q8_0) at [2560, 640].  In place the matrices lie at
// scattered 256-byte-aligned addresses of an arena filled with an fp16 NaN pattern (0xFF 0x7C), each followed by the
// tab_overread bytes the k loop reads past it, zeroed by zero_after, then by the pattern: a read further past a matrix
// would put a non-finite value into the output.  Groups of 1-16 experts with mixed row counts (0 to 1,000), and
// max_rows from 1 to 1,000 so that every tile width MMQ picks (8-128) is taken, few-tile launches (the stream-k grid
// with its fixup) and many-tile ones.  Both outputs are pre-filled with the NaN pattern and compared whole.
#include "lamina/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = lamina::prefill::mmq;

constexpr int G = mmq::kGatherGroupMax;
constexpr size_t TAIL = 4096;
constexpr int64_t kIdsPad = 2048 + 128;        // MMQ reads a whole tile's ids from an expert's first row on

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t bytes) : n(bytes) { ck(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    uint8_t* b() const { return (uint8_t*) p; }
};

void upload(void* dst, const void* src, size_t n) {
    ck(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice), "upload");
    ck(cudaDeviceSynchronize(), "upload");
}
void fill_nan(const Dev& d) {
    std::vector<uint8_t> h(d.n);
    for (size_t i = 0; i < d.n; ++i) h[i] = (i & 1) ? 0x7C : 0xFF;
    upload(d.p, h.data(), d.n);
}

// random blocks with sane fp16 scales where each format keeps them (d first; the K-quants' dmin, Q5_1's m after it)
void fill_matrix(uint8_t* m, ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(t), nb = (size_t) rows * (size_t) (cols / ggml_blck_size(t));
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> sc(0.0004f, 0.0024f);
    for (size_t i = 0; i < nb * bs; ++i) m[i] = (uint8_t) byte(rng);
    const bool dmin = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q5_1;
    for (size_t b = 0; b < nb; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(sc(rng)), dm = ggml_fp32_to_fp16(sc(rng));
        std::memcpy(m + b * bs, &d, 2);
        if (dmin) std::memcpy(m + b * bs + 2, &dm, 2);
    }
}

int g_cases = 0, g_fail = 0;

// one product of n experts of [ROWS, COLS] with cnt[e] rows each
void product_case(mmq::Context& ctx, ggml_type t, int64_t ROWS, int64_t COLS, const std::vector<int>& cnt,
                  std::mt19937& rng, cudaStream_t s) {
    const int n = (int) cnt.size();
    int64_t total = 0, max_rows = 0;
    std::vector<int32_t> bounds(cnt.size() + 1, 0);
    for (size_t i = 0; i < cnt.size(); ++i) {
        total += cnt[i];
        max_rows = std::max<int64_t>(max_rows, cnt[i]);
        bounds[i + 1] = bounds[i] + cnt[i];
    }
    const size_t eb = mmq::matrix_bytes((int) t, ROWS, COLS);
    // activations, the identity row map, the bounds
    const int64_t tr = std::max<int64_t>(1, total);
    Dev x((size_t) tr * COLS * 4), xq(mmq::q8_bytes(tr, COLS)), ids((size_t) (tr + kIdsPad) * 4), bd(bounds.size() * 4);
    std::vector<float> hx((size_t) tr * COLS);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : hx) v = nd(rng);
    upload(x.p, hx.data(), hx.size() * 4);
    upload(bd.p, bounds.data(), bounds.size() * 4);
    mmq::iota((int32_t*) ids.p, tr + kIdsPad, s);
    if (total > 0) mmq::quantize((const float*) x.p, nullptr, xq.p, (int) t, COLS, COLS, total, s);
    ck(cudaStreamSynchronize(s), "acts");
    std::vector<std::vector<uint8_t>> w((size_t) n, std::vector<uint8_t>(eb));
    for (auto& m : w) fill_matrix(m.data(), t, ROWS, COLS, rng);
    // gathered: back to back, a zeroed tail
    Dev grp((size_t) n * eb + TAIL + 4096);
    fill_nan(grp);
    for (int z = 0; z < n; ++z) upload(grp.b() + (size_t) z * eb, w[(size_t) z].data(), eb);
    ck(cudaMemset(grp.b() + (size_t) n * eb, 0, TAIL), "tail");
    // in place: shuffled 256-aligned places with random gaps, the over-read zeroed after each matrix, then NaN
    const size_t over = mmq::tab_overread((int) t, COLS), stride = (eb + over + 255) / 256 * 256;
    std::vector<size_t> order((size_t) n), where((size_t) n);
    for (int z = 0; z < n; ++z) order[(size_t) z] = (size_t) z;
    std::shuffle(order.begin(), order.end(), rng);
    std::uniform_int_distribution<int> gap(0, 8);
    Dev arena((size_t) n * (stride + 8 * 256) + 4096);
    fill_nan(arena);
    size_t at = 0;
    for (int k = 0; k < n; ++k) {
        at += (size_t) gap(rng) * 256;
        where[order[(size_t) k]] = at;
        at += stride;
    }
    std::vector<const void*> tab((size_t) n);
    std::vector<uint8_t*> after((size_t) n);
    for (int z = 0; z < n; ++z) {
        upload(arena.b() + where[(size_t) z], w[(size_t) z].data(), eb);
        tab[(size_t) z] = arena.b() + where[(size_t) z];
        after[(size_t) z] = arena.b() + where[(size_t) z] + eb;
    }
    mmq::zero_after(after.data(), n, over, s);
    const size_t out = (size_t) tr * ROWS * 4;
    Dev ref(out), got(out);
    fill_nan(ref);
    fill_nan(got);
    mmq::Product p;
    p.type = (int) t; p.w_rows = ROWS; p.w_cols = COLS; p.n = n; p.xq = xq.p; p.bounds = (const int32_t*) bd.p;
    p.ids = (const int32_t*) ids.p; p.total_rows = total; p.max_rows = max_rows; p.ld_dst = ROWS;
    p.w = grp.p; p.expert_bytes = eb; p.dst = (float*) ref.p;
    ctx.run(p, s);
    p.w = nullptr; p.expert_bytes = 0; p.tab = tab.data(); p.dst = (float*) got.p;
    ctx.run(p, s);
    ck(cudaStreamSynchronize(s), "run");
    std::vector<uint8_t> ha(out), hb(out);
    ck(cudaMemcpy(ha.data(), ref.p, out, cudaMemcpyDeviceToHost), "dtoh");
    ck(cudaMemcpy(hb.data(), got.p, out, cudaMemcpyDeviceToHost), "dtoh");
    ++g_cases;
    bool finite = true;
    for (int64_t i = 0; i < total * ROWS && finite; ++i) {
        float v;
        std::memcpy(&v, ha.data() + i * 4, 4);
        finite = v == v && v - v == 0.0f;
    }
    if (std::memcmp(ha.data(), hb.data(), out) == 0 && finite) return;
    size_t i = 0;
    while (i < out && ha[i] == hb[i]) ++i;
    std::printf("  FAIL %s n=%d rows=%lld max_rows=%lld: %s\n", ggml_type_name(t), n, (long long) total,
                (long long) max_rows, finite ? ("first differing byte " + std::to_string(i)).c_str()
                                              : "the gathered product is not finite");
    ++g_fail;
}

}  // namespace

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        std::printf("mmq_tab_parity: no CUDA device, skipped\n");
        return 77;
    }
    std::printf("mmq_tab_parity on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    try {
        cudaStream_t s = nullptr;
        ck(cudaStreamCreate(&s), "stream");
        mmq::Context ctx;
        std::mt19937 rng(20261018);
        static const int kRows[] = {0, 1, 7, 8, 9, 31, 64, 100, 512, 1000};
        static const int kMax[] = {1, 7, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 56, 64, 65, 72, 80, 88, 96, 100,
                                   104, 112, 120, 128, 129, 200, 256, 512, 1000};
        for (const ggml_type t : {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K,
                                  GGML_TYPE_Q5_K, GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                                  GGML_TYPE_Q8_0}) {
            const bool down = t == GGML_TYPE_Q2_0 || t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 ||
                              t == GGML_TYPE_Q8_0;
            const int64_t ROWS = down ? 2560 : 1280, COLS = down ? 640 : 2560;
            if (!mmq::tab_takes((int) t, ROWS, COLS)) {
                std::printf("FAIL %s: no pointer-table product in this build\n", ggml_type_name(t));
                ++g_fail;
                continue;
            }
            const int f0 = g_fail, c0 = g_cases;
            std::uniform_int_distribution<int> pick(0, 9);
            for (int n = 1; n <= G; ++n) {   // mixed counts
                std::vector<int> c((size_t) n);
                for (auto& v : c) v = kRows[pick(rng)];
                if (*std::max_element(c.begin(), c.end()) == 0) c[0] = 1;
                product_case(ctx, t, ROWS, COLS, c, rng, s);
            }
            for (const int mx : kMax) {      // every tile width: one expert with mx rows, the others fewer
                std::uniform_int_distribution<int> fewer(0, mx);
                std::vector<int> c(5);
                for (auto& v : c) v = fewer(rng);
                c[2] = mx;
                product_case(ctx, t, ROWS, COLS, c, rng, s);
            }
            std::printf("%s %-8s [%lld, %lld] %d products, %zu bytes read past each matrix\n",
                        g_fail == f0 ? "ok  " : "FAIL", ggml_type_name(t), (long long) ROWS, (long long) COLS,
                        g_cases - c0, mmq::tab_overread((int) t, COLS));
        }
        cudaStreamDestroy(s);
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("mmq_tab_parity: %d of %d products differ\n", g_fail, g_cases);
    return g_fail ? 1 : 0;
}
