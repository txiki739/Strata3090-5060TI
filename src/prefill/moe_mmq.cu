// src/prefill/moe_mmq.cu - see include/lamina/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "lamina/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#if defined(LAMINA_MMQ_TAB)
#include "moe_mmq_tab.hpp"   // declarations: the kernels are in the per-type mmq-tab-*.cu TUs
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace lamina::prefill::mmq {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// ggml's tables stop at GGML_TYPE_COUNT: the trellis types' block sizes are here
bool is_tq(int t) { return t >= 147 && t <= 149; }
int64_t type_blck(int t) { return is_tq(t) ? 128 : (int64_t) ggml_blck_size((ggml_type) t); }
size_t type_bytes(int t) { return is_tq(t) ? (size_t) (2 + 8 * (t - 142)) : ggml_type_size((ggml_type) t); }
const char* type_str(int t) { return t == 147 ? "tqp5" : t == 148 ? "tqp6" : t == 149 ? "tqp7" : ggml_type_name((ggml_type) t); }

// A warp's 128 values, four a lane, as H_128 (signs * x) (hadamard.hpp): hadamard_rotate's butterflies in its order
__device__ __forceinline__ float4 rotate128(float4 v, const float* __restrict__ signs, int64_t i0, int64_t ne00, float scale) {
    if (signs != nullptr && i0 < ne00) {
        const float4 s = *(const float4*) (signs + i0);
        v.x *= s.x; v.y *= s.y; v.z *= s.z; v.w *= s.w;
    }
    const float a = v.x + v.y, b = v.x - v.y, c = v.z + v.w, d = v.z - v.w;
    float r[4] = {a + c, b + d, a - c, b - d};
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int h = 1; h < 32; h <<= 1)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float o = __shfl_xor_sync(0xFFFFFFFF, r[k], h, WARP_SIZE);
            r[k] = (lane & h) ? o - r[k] : r[k] + o;
        }
    return make_float4(r[0] * scale, r[1] * scale, r[2] * scale, r[3] * scale);
}

__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}
// copy16_kernel for a batch of experts: blockIdx.y is the expert
struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride;   // in uint4
};
__global__ void copy16_group_kernel(GroupArgs ga, int64_t na, int64_t nc, uint4* __restrict__ gu_dst,
                                    uint4* __restrict__ d_dst) {
    const int q = (int) blockIdx.y;
    const uint4* src = (const uint4*) ga.blob[q];
    uint4* ab = gu_dst + (int64_t) q * ga.gu_stride;
    uint4* cd = d_dst + (int64_t) q * ga.d_stride;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < 2 * na + nc) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

// llama.cpp's quantize_mmq_q8_1 (quantize.cu, MIT) for FP16 rows: one channel, row i of the output from row ids[i]
// (or i), the same arithmetic on the values the FP16 rows hold
template <mmq_q8_1_ds_layout ds_layout, bool rotated>
__global__ void quantize_mmq_q8_1_f16(const half* __restrict__ x, const int32_t* __restrict__ ids, void* __restrict__ vy,
                                      const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1,
                                      const float* __restrict__ signs, const float scale) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    block_q8_1_mmq* y = (block_q8_1_mmq*) vy;
    const int64_t k_block = i0 / QK8_1_MMQ, iqs = i0 % QK8_1_MMQ;
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < ne00) {
        const half2* h = (const half2*) (x + (int64_t) (ids ? ids[blockIdx.x] : (int32_t) blockIdx.x) * s01 + i0);
        const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
        xi = make_float4(a.x, a.y, b.x, b.y);
    }
    if constexpr (rotated) xi = rotate128(xi, signs, i0, ne00, scale);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    const int64_t ib = k_block * ne1 + blockIdx.x;
    ((char4*) y[ib].qs)[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) y[ib].d2s6[iqs / 64] = d;
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) y[ib].ds4[iqs / 32] = make_half2(d, sum);
        else y[ib].d4[iqs / 32] = d;
    }
}

// SwiGLU's h (swiglu_kernel's arithmetic) as the values of quantize_mmq_q8_1: H's q8_1 rows without its FP32 rows
template <mmq_q8_1_ds_layout ds_layout, bool interleaved, bool rotated>
__global__ void swiglu_quantize_mmq_q8_1(const float* __restrict__ gu, void* __restrict__ vy, const int64_t n_ff,
                                         const int64_t ne0, const int ne1, const float* __restrict__ signs,
                                         const float scale) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    block_q8_1_mmq* y = (block_q8_1_mmq*) vy;
    const int64_t k_block = i0 / QK8_1_MMQ, iqs = i0 % QK8_1_MMQ;
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < n_ff) {
        const float* row = gu + (int64_t) blockIdx.x * 2 * n_ff;
        float g[4], u[4], h[4];
        if (interleaved) {
            const float4 a = *(const float4*) (row + 2 * i0), b = *(const float4*) (row + 2 * i0 + 4);
            g[0] = a.x; u[0] = a.y; g[1] = a.z; u[1] = a.w; g[2] = b.x; u[2] = b.y; g[3] = b.z; u[3] = b.w;
        } else {
            const float4 a = *(const float4*) (row + i0), b = *(const float4*) (row + n_ff + i0);
            g[0] = a.x; g[1] = a.y; g[2] = a.z; g[3] = a.w; u[0] = b.x; u[1] = b.y; u[2] = b.z; u[3] = b.w;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) h[k] = g[k] / (1.0f + __expf(-g[k])) * u[k];
        xi = make_float4(h[0], h[1], h[2], h[3]);
    }
    if constexpr (rotated) xi = rotate128(xi, signs, i0, n_ff, scale);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    const int64_t ib = k_block * ne1 + blockIdx.x;
    ((char4*) y[ib].qs)[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) y[ib].d2s6[iqs / 64] = d;
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) y[ib].ds4[iqs / 32] = make_half2(d, sum);
        else y[ib].d4[iqs / 32] = d;
    }
}

}  // namespace

bool built() { return true; }

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_1:   // unsloth's files
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
#if defined(LAMINA_MMQ_TQP)
        case GGML_TYPE_TQP5: case GGML_TYPE_TQP6: case GGML_TYPE_TQP7:   // Gyro-L (cmake/lamina_ggml_tq_overlay.cmake)
#endif
            return true;
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    if (!supported(t)) return false;
    // mul_mat_q_case's choice: the "fallback" configs when the rows are not a multiple of 128; then
    // mul_mat_q_switch_J's loop - a tile size whose config exists for this card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (is_tq(t) || (t >= 0 && t < GGML_TYPE_COUNT && !said[t])) {
                if (!is_tq(t)) said[t] = true;
                std::fprintf(stderr, "lamina: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path\n",
                             type_str(t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / type_blck(t)) * type_bytes(t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

void quantize_f16(const uint16_t* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows,
                  void* stream, const float* signs) {
    if (rows <= 0) return;
    const int64_t ne0 = pad512(cols);
    constexpr int B = CUDA_QUANTIZE_BLOCK_SIZE_MMQ;
    const dim3 grid((unsigned) rows, (unsigned) ((ne0 + 4 * B - 1) / (4 * B)));
    const cudaStream_t s = (cudaStream_t) stream;
    const half* h = (const half*) x;
    const float scale = 1.0f / sqrtf(128.0f);
#define LAMINA_QF16(L) (signs != nullptr ? quantize_mmq_q8_1_f16<L, true><<<grid, B, 0, s>>>(h, ids, xq, cols, ld, ne0, (int) rows, signs, scale) \
                                         : quantize_mmq_q8_1_f16<L, false><<<grid, B, 0, s>>>(h, ids, xq, cols, ld, ne0, (int) rows, nullptr, 1.0f))
    switch (mmq_get_q8_1_ds_layout((ggml_type) t)) {
        case MMQ_Q8_1_DS_LAYOUT_D4: LAMINA_QF16(MMQ_Q8_1_DS_LAYOUT_D4); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: LAMINA_QF16(MMQ_Q8_1_DS_LAYOUT_DS4); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: LAMINA_QF16(MMQ_Q8_1_DS_LAYOUT_D2S6); break;
        default:
            std::fprintf(stderr, "prefill mmq: no q8_1 layout for type %d\n", t);
            std::exit(1);
    }
#undef LAMINA_QF16
    ck(cudaGetLastError(), "quantize_f16");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

#if defined(LAMINA_MMQ_TAB)
namespace {
// the pointer-table MMQ for the types CMakeLists builds it for; false: none (nothing launched; ctx null: only asked)
bool run_tab(ggml_backend_cuda_context* ctx, ggml_type t, const mmq_args& a, const mmq_x_tab& tab, cudaStream_t s) {
    switch (t) {
#define LAMINA_TAB_CASE(T) case T: return ctx == nullptr || mul_mat_q_tab_case<T>(*ctx, a, tab, s)
        LAMINA_TAB_CASE(GGML_TYPE_IQ2_S);
        LAMINA_TAB_CASE(GGML_TYPE_IQ3_XXS);
        LAMINA_TAB_CASE(GGML_TYPE_IQ3_S);
        LAMINA_TAB_CASE(GGML_TYPE_IQ4_XS);
        LAMINA_TAB_CASE(GGML_TYPE_Q4_K);
        LAMINA_TAB_CASE(GGML_TYPE_Q5_K);
        LAMINA_TAB_CASE(GGML_TYPE_Q2_0);
        LAMINA_TAB_CASE(GGML_TYPE_IQ4_NL);
        LAMINA_TAB_CASE(GGML_TYPE_Q5_0);
        LAMINA_TAB_CASE(GGML_TYPE_Q5_1);
        LAMINA_TAB_CASE(GGML_TYPE_Q8_0);
#if defined(LAMINA_MMQ_TQP)
        LAMINA_TAB_CASE(GGML_TYPE_TQP5);
        LAMINA_TAB_CASE(GGML_TYPE_TQP6);
        LAMINA_TAB_CASE(GGML_TYPE_TQP7);
#endif
#undef LAMINA_TAB_CASE
        default: return false;
    }
}
}  // namespace
#endif

bool tab_takes(int t, int64_t w_rows, int64_t w_cols) {
#if defined(LAMINA_MMQ_TAB)
    return w_rows % 128 == 0 && w_cols % type_blck(t) == 0 &&
           run_tab(nullptr, (ggml_type) t, mmq_args{}, mmq_x_tab{}, nullptr);
#else
    (void) t; (void) w_rows; (void) w_cols;
    return false;
#endif
}

size_t tab_overread(int t, int64_t w_cols) {
    // the k loop's step (K_vram) for every tile width the launch may pick (fallback = false, the tab kernels' only)
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t qk = type_blck(t), bpr = w_cols / qk;
    int64_t most = 0;
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, false, cc);
        if (c.type == GGML_TYPE_COUNT) continue;
        const int64_t bpi = c.K_vram / qk;
        most = std::max<int64_t>(most, (bpr + bpi - 1) / bpi * bpi - bpr);
    }
    return (size_t) most * type_bytes(t);
}

namespace {
struct ZeroArgs {
    uint32_t* at[kGatherGroupMax];
};
// __grid_constant__: the pointers indexed in the parameter bank (a by-value copy went to local memory, 128 bytes a
// thread, past the engine's stack limit)
__global__ void zero_after_kernel(const __grid_constant__ ZeroArgs za, int words) {
    if ((int) threadIdx.x < words) za.at[blockIdx.x][threadIdx.x] = 0;
}
}  // namespace

void zero_after(uint8_t* const* at, int n, size_t bytes, void* stream) {
    if (n <= 0 || bytes == 0) return;
    if (n > kGatherGroupMax || bytes > 4096) {
        std::fprintf(stderr, "prefill mmq: zero_after takes up to %d places of up to 4096 bytes\n", kGatherGroupMax);
        std::exit(1);
    }
    ZeroArgs za{};
    for (int q = 0; q < n; ++q) za.at[q] = (uint32_t*) at[q];
    const int words = (int) ((bytes + 3) / 4);   // a trellis block's bytes are not whole words
    zero_after_kernel<<<(unsigned) n, (unsigned) ((words + 31) / 32 * 32), 0, (cudaStream_t) stream>>>(za, words);
    ck(cudaGetLastError(), "zero_after");
}

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = type_blck(t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, p.tab ? 0 : (int64_t) (p.expert_bytes / type_bytes(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    if (p.tab) {
#if defined(LAMINA_MMQ_TAB)
        mmq_x_tab tab{};
        for (int z = 0; z < p.n && z < kGatherGroupMax; ++z) tab.p[z] = (const char*) p.tab[z];
        if (p.n <= kGatherGroupMax && tab_takes(p.type, p.w_rows, p.w_cols) && run_tab(&ctx, t, a, tab, s)) {
            ck(cudaGetLastError(), "mul_mat_q (pointer table)");
            return;
        }
#endif
        std::fprintf(stderr, "prefill mmq: no pointer-table MMQ for %s [%lld, %lld] (%d experts)\n", type_str(t),
                     (long long) p.w_rows, (long long) p.w_cols, p.n);
        std::exit(1);
    }
    switch (t) {
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
        case GGML_TYPE_IQ1_S: mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, a, s); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
#if defined(LAMINA_MMQ_TQP)
        case GGML_TYPE_TQP5: mul_mat_q_case<GGML_TYPE_TQP5>(ctx, a, s); break;
        case GGML_TYPE_TQP6: mul_mat_q_case<GGML_TYPE_TQP6>(ctx, a, s); break;
        case GGML_TYPE_TQP7: mul_mat_q_case<GGML_TYPE_TQP7>(ctx, a, s); break;
#endif
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

void Context::run_dense(const void* w, int type, int64_t rows, int64_t cols, const void* xq, int64_t T, float* dst,
                        int64_t ld_dst, void* stream) {
    if (T <= 0) return;
    const ggml_type t = (ggml_type) type;
    const int64_t bpr = cols / ggml_blck_size(t);
    const int64_t s12 = T * pad512(cols) * (int64_t) sizeof(block_q8_1) / (QK8_1 * (int64_t) sizeof(int));
    const mmq_args a = {(const char*) w, t, (const int*) xq, nullptr, nullptr, dst, nullptr,
                        cols, rows, T, bpr, T, ld_dst,
                        1, 1, rows * bpr, s12, T * ld_dst,
                        1, 1, 0, 0, 0,
                        T, T};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
        case GGML_TYPE_IQ1_S: mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, a, s); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q (dense)");
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      gu_half_bytes | d_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst,
                                                          (const uint4*) down, nc, (uint4*) d_dst);
    } else {
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

bool gather_native_group(const uint8_t* const* blob, int n, size_t up_off, size_t gu_half_bytes, size_t down_off,
                         size_t d_bytes, void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (n <= 0 || n > kGatherGroupMax) return false;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride |
                  d_stride;
    for (int q = 0; q < n; ++q) a |= (uintptr_t) blob[q];
    if (a % 16 != 0) return false;
    GroupArgs ga{};
    for (int q = 0; q < n; ++q) ga.blob[q] = blob[q];
    ga.up_off = (int64_t) up_off / 16;
    ga.down_off = (int64_t) down_off / 16;
    ga.gu_stride = (int64_t) gu_stride / 16;
    ga.d_stride = (int64_t) d_stride / 16;
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
    copy16_group_kernel<<<dim3(blocks(2 * na + nc), (unsigned) n), 256, 0, (cudaStream_t) stream>>>(ga, na, nc,
                                                                                                  (uint4*) gu_dst,
                                                                                                  (uint4*) d_dst);
    ck(cudaGetLastError(), "gather_native_group");
    return true;
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

void swiglu_quantize(const float* gu, void* xq, int t, int64_t rows, int64_t n_ff, bool interleaved, void* stream,
                     const float* signs) {
    if (rows <= 0) return;
    if (n_ff % 4 != 0) {
        std::fprintf(stderr, "prefill mmq: swiglu_quantize needs a multiple of 4 values a row (%lld)\n", (long long) n_ff);
        std::exit(1);
    }
    const int64_t ne0 = pad512(n_ff);
    constexpr int B = CUDA_QUANTIZE_BLOCK_SIZE_MMQ;
    const dim3 grid((unsigned) rows, (unsigned) ((ne0 + 4 * B - 1) / (4 * B)));
    const cudaStream_t s = (cudaStream_t) stream;
    const float scale = 1.0f / sqrtf(128.0f);
#define LAMINA_SWQ(L) (signs != nullptr ? swiglu_quantize_mmq_q8_1<L, false, true><<<grid, B, 0, s>>>(gu, xq, n_ff, ne0, (int) rows, signs, scale) \
                       : interleaved ? swiglu_quantize_mmq_q8_1<L, true, false><<<grid, B, 0, s>>>(gu, xq, n_ff, ne0, (int) rows, nullptr, 1.0f) \
                                     : swiglu_quantize_mmq_q8_1<L, false, false><<<grid, B, 0, s>>>(gu, xq, n_ff, ne0, (int) rows, nullptr, 1.0f))
    switch (mmq_get_q8_1_ds_layout((ggml_type) t)) {
        case MMQ_Q8_1_DS_LAYOUT_D4: LAMINA_SWQ(MMQ_Q8_1_DS_LAYOUT_D4); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: LAMINA_SWQ(MMQ_Q8_1_DS_LAYOUT_DS4); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: LAMINA_SWQ(MMQ_Q8_1_DS_LAYOUT_D2S6); break;
        default:
            std::fprintf(stderr, "prefill mmq: no q8_1 layout for type %d\n", t);
            std::exit(1);
    }
#undef LAMINA_SWQ
    ck(cudaGetLastError(), "swiglu_quantize");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

}  // namespace lamina::prefill::mmq
