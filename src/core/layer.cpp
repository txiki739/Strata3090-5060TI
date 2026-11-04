
// src/core/layer.cpp - the GDN layer, composed.  See the header for the operation order and its traps.
#include "lamina/core/layer.hpp"
#include "lamina/core/native_head.hpp"
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/bf16_gemv.hpp"
#include "lamina/kernels/elementwise.hpp"
#include "lamina/kernels/gdn.hpp"
#include "lamina/kernels/mrope.hpp"
#include "lamina/kernels/qsa.hpp"
#include "lamina/kernels/kv_q4.hpp"
#include "lamina/kernels/kv_q8.hpp"
#include "lamina/kernels/quantize_act.hpp"
#include "lamina/kernels/rope.hpp"
#include "lamina/kernels/s2_gemv_q8.hpp"
#include "lamina/kernels/s_gemv.hpp"
#include "lamina/kernels/shared_expert.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/native_moe.hpp"
#include "lamina/kernels/native_router.hpp"
#include "lamina/kernels/native_qsa.hpp"
#include "lamina/kernels/native_qsa_indexer.hpp"
#include "lamina/kernels/native_rope.hpp"
#include "lamina/kernels/fused_gr.hpp"
#include "lamina/kernels/qsa_decode_attn.hpp"
#include "lamina/kernels/fused_gdn.hpp"
#include "lamina/kernels/qsa_select.hpp"
#include "lamina/kernels/verify_kernels.hpp"
#include <exception>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
namespace lamina::core {
namespace {
constexpr int Q8K_BYTES_PER_BLOCK = 292;
constexpr int Q8K_ELEMS_PER_BLOCK = 256;
const std::string EMBEDDING_NAME = "token_embd.weight";
/// A Q8_K buffer needs `n` a multiple of 256 and `n/256` blocks of 292 bytes.
uint64_t q8k_bytes(int64_t n) { return (uint64_t) (n / Q8K_ELEMS_PER_BLOCK) * Q8K_BYTES_PER_BLOCK; }

}
// namespace
namespace { KvFormat g_kv_format = KvFormat::F16; }
void qsa_set_kv_format(KvFormat f) { g_kv_format = f; }
KvFormat qsa_kv_format() { return g_kv_format; }
KvFormat qsa_draft_kv_format() { return g_kv_format == KvFormat::F16 ? KvFormat::F16 : KvFormat::Int8; }
// ================================ the QSA layers' state ================================
namespace {
using lamina::kernels::QsaIndexerBuffers;
using lamina::kernels::QsaShapes;
uint64_t align_up16(uint64_t n) { return (n + 15) & ~15ull; }
/// One cursor over an arena, so every region is 16-byte aligned without a list of hand-added offsets.
struct Cursor {
    uint8_t* p;
    uint64_t used = 0;
    template <typename T>
    T* take(uint64_t count) {
        T* r = (T*) (p + used);
        used = align_up16(used + count * sizeof(T));
        return r;
    }
    uint8_t* take_bytes(uint64_t n) {
        uint8_t* r = p + used;
        used = align_up16(used + n);
        return r;
    }
};
}
// namespace
namespace {
// a format's K/V arrays in the arena's order, as bytes per cell and KV head
std::vector<uint64_t> kv_row_bytes(KvFormat kv, uint64_t hd) {
    const uint64_t s8 = hd / lamina::kernels::KV_Q8_GROUP * 2, s4 = hd / lamina::kernels::KV_Q4_GROUP * 2;
    switch (kv) {
    case KvFormat::Int8: return {hd, hd, s8, s8};
    case KvFormat::Q4: return {hd / 2, hd / 2, s4, s4};
    case KvFormat::K8V4: return {hd, s8, hd / 2, s4};
    default: return {hd * 2, hd * 2};
    }
}

// ---- the elastic K/V (--kv-grow; after Niko1221/Strata 0.1.40): one VMM range per state, each pool array at a chunk
// boundary in it, mapped in whole pages from the first cell on (the page table is the identity).  A state's range holds
// its K/V arrays and the indexer's pooled keys; one more range holds the rotary tables, filled as they are mapped.
bool g_kv_elastic = false;
int64_t g_kv_elastic_init = 16384;
struct ElasticPool {
    VmmRange range;
    // each array's offset in the range (whole chunks), bytes per page, and bytes past its pages (the pooled keys' spare
    // rows: the row after the last block)
    std::vector<uint64_t> off, per_page, extra;
    int64_t n_pages = 0, page_size = 0;
    /// Writes array a's bytes [lo, hi) to `dst` as they are mapped; none: they are zeroed.
    std::function<bool(size_t a, uint64_t lo, uint64_t hi, uint8_t* dst)> fill;
    /// Appends an array (after n_pages is set).
    void add(uint64_t page_bytes, uint64_t extra_bytes = 0) {
        off.push_back(bytes());
        per_page.push_back(page_bytes);
        extra.push_back(extra_bytes);
    }
    /// The range's size: every array in whole chunks.
    uint64_t bytes() const {
        const uint64_t G = vmm_granularity();
        return off.empty() ? 0 : off.back() + ((uint64_t) n_pages * per_page.back() + extra.back() + G - 1) / G * G;
    }
};
std::vector<std::unique_ptr<ElasticPool>> g_pools;

int64_t pool_chunks(const ElasticPool& p, size_t a, int64_t pages) {
    const uint64_t G = vmm_granularity();
    return (int64_t) (((uint64_t) pages * p.per_page[a] + p.extra[a] + G - 1) / G);
}
/// Array a's bytes mapped from its start.
uint64_t pool_mapped_bytes(const ElasticPool& p, size_t a) {
    const uint64_t G = vmm_granularity();
    const int64_t c0 = (int64_t) (p.off[a] / G);
    const int64_t end = a + 1 < p.off.size() ? (int64_t) (p.off[a + 1] / G) : p.range.chunks();
    int64_t c = 0;
    while (c0 + c < end && p.range.mapped(c0 + c)) ++c;
    return std::min<uint64_t>((uint64_t) c * G, (uint64_t) p.n_pages * p.per_page[a] + p.extra[a]);
}
int64_t pool_pages(const ElasticPool& p, int64_t cells) {
    return std::min<int64_t>(p.n_pages, std::max<int64_t>((cells + p.page_size - 1) / p.page_size, 1));
}
/// Whole pages every array of the pool has mapped.
int64_t pool_pages_mapped(const ElasticPool& p) {
    int64_t pages = p.n_pages;
    for (size_t a = 0; a < p.off.size(); ++a) {
        const uint64_t m = pool_mapped_bytes(p, a);
        pages = std::min<int64_t>(pages, m < p.extra[a] ? 0 : (int64_t) ((m - p.extra[a]) / p.per_page[a]));
    }
    return pages;
}
bool pool_grow(ElasticPool& p, int64_t cells, const std::function<VmmChunk()>& take) {
    const uint64_t G = vmm_granularity();
    const int64_t pages = pool_pages(p, cells);
    std::vector<std::vector<int64_t>> fresh(p.off.size());
    size_t most = 0;
    bool mapped_all = true;
    for (size_t a = 0; a < p.off.size() && mapped_all; ++a) {
        const int64_t c0 = (int64_t) (p.off[a] / G), c1 = c0 + pool_chunks(p, a, pages);
        for (int64_t c = c0; c < c1; ++c)
            if (!p.range.mapped(c)) fresh[a].push_back(c);
        if (fresh[a].empty()) continue;
        most = std::max(most, fresh[a].size());
        if (!p.range.map_range(c0, c1, take)) {
            std::fprintf(stderr, "lamina: the elastic K/V could not map %zu chunks (the driver has no VRAM left)\n",
                         fresh[a].size());
            mapped_all = false;   // what was mapped stays mapped, and a later growth skips it: cleared below too
        }
    }
    // what a chunk held before (another pool's cells, an expert) is not K/V: zeroed, as a new session is.  Or filled
    // (the rotary tables): the k-th new chunk of every array in turn, so cos's and sin's of one range follow each other
    for (size_t k = 0; k < most; ++k)
        for (size_t a = 0; a < p.off.size(); ++a) {
            if (k >= fresh[a].size() || !p.range.mapped(fresh[a][k])) continue;
            uint8_t* dst = p.range.base() + (uint64_t) fresh[a][k] * G;
            const uint64_t lo = (uint64_t) fresh[a][k] * G - p.off[a];
            if (p.fill ? !p.fill(a, lo, lo + G, dst) : cudaMemset(dst, 0, (size_t) G) != cudaSuccess) {
                std::fprintf(stderr, "lamina: the elastic K/V could not clear or fill a chunk: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return false;
            }
        }
    return mapped_all;
}
}  // namespace

void qsa_set_kv_elastic(bool enabled, int64_t init_cells) {
    g_kv_elastic = enabled && vmm_available();
    if (init_cells > 0) g_kv_elastic_init = init_cells;
}
bool qsa_kv_elastic() { return g_kv_elastic; }
int64_t qsa_kv_elastic_cells() {
    int64_t cells = std::numeric_limits<int64_t>::max();
    for (const auto& p : g_pools) cells = std::min<int64_t>(cells, pool_pages_mapped(*p) * p->page_size);
    return cells;
}
int64_t qsa_kv_elastic_need(int64_t cells) {
    const uint64_t G = vmm_granularity();
    int64_t n = 0;
    for (const auto& p : g_pools) {
        const int64_t pages = pool_pages(*p, cells);
        for (size_t a = 0; a < p->off.size(); ++a) {
            const int64_t c0 = (int64_t) (p->off[a] / G), c1 = c0 + pool_chunks(*p, a, pages);
            for (int64_t c = c0; c < c1; ++c) n += !p->range.mapped(c);
        }
    }
    return n;
}
bool qsa_kv_elastic_grow(int64_t cells, const std::function<VmmChunk()>& take) {
    for (auto& p : g_pools)
        if (!pool_grow(*p, cells, take)) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
}
int64_t qsa_kv_elastic_shrink(int64_t cells, const std::function<void(VmmChunk)>& give) {
    const uint64_t G = vmm_granularity();
    int64_t n = 0;
    for (auto& p : g_pools) {
        const int64_t pages = pool_pages(*p, cells);
        for (size_t a = 0; a < p->off.size(); ++a) {
            const int64_t c0 = (int64_t) (p->off[a] / G), c1 = c0 + pool_chunks(*p, a, pages);
            const int64_t end = a + 1 < p->off.size() ? (int64_t) (p->off[a + 1] / G) : p->range.chunks();
            for (int64_t c = c1; c < end; ++c)
                if (const VmmChunk h = p->range.unmap(c)) { give(h); ++n; }
        }
    }
    return n;
}
uint64_t qsa_kv_elastic_mapped_bytes() {
    uint64_t n = 0;
    for (const auto& p : g_pools) n += (uint64_t) p->range.mapped_count() * vmm_granularity();
    return n;
}
uint64_t qsa_kv_elastic_full_bytes() {
    uint64_t n = 0;
    for (const auto& p : g_pools) n += (uint64_t) p->range.chunks() * vmm_granularity();
    return n;
}
void qsa_kv_elastic_release() { g_pools.clear(); }
uint64_t qsa_state_bytes(const ModelGeometry& g, int64_t max_cells, bool with_rope, KvFormat kv) {
    const QsaShapes s = qsa_shapes(g);
    const int64_t pages = (max_cells + s.page_size - 1) / s.page_size;
    uint64_t n = 0;
    if (!g_kv_elastic)   // the K/V pools (the elastic ones are in their own VMM range)
        for (const uint64_t b : kv_row_bytes(kv, (uint64_t) s.head_dim))
            n += align_up16((uint64_t) pages * s.n_head_kv * s.page_size * b);
n += (uint64_t) pages * 4;
// page_table
n += (uint64_t) (s.idx_block - 1) * s.idx_dim * 4;
// tail
n += (uint64_t) s.idx_dim * 4;
// dead
if (!g_kv_elastic) n += (uint64_t) ((max_cells / s.idx_block) + 2) * s.idx_dim * 4;
// pooled (the elastic keys are in the K/V's range)
n += 16;
// block_pos
if (with_rope && !g_kv_elastic) n += (uint64_t) max_cells * (s.n_rot / 2) * 4 * 2;
// cos + sin tables (the elastic ones are in their own range)
n += lamina::kernels::qsa_step_bytes() + 16; // counts and aligned attention status
// the per-token counts
n += (uint64_t) s.n_head * 4;
// pos_dev
return align_up16(n) + 256;}
uint64_t qsa_state_init(const ModelGeometry& g, int64_t max_cells, void* base, QsaState& st,
                        const QsaState* share_rope, KvFormat kv) {
    const QsaShapes s = qsa_shapes(g);
    const int64_t pages = (max_cells + s.page_size - 1) / s.page_size;
    Cursor c{(uint8_t*) base};
    st.kv = kv;
    st.kv_int8 = kv == KvFormat::Int8;
    std::vector<uint8_t*> a;
    st.kv_elastic = -1;
    if (g_kv_elastic) {
        // the elastic K/V: each array at a chunk boundary of the state's own range, the first cells mapped
        // (then the indexer's pooled keys: a row per idx_block cells, and the spare's past the last)
        if (s.page_size % s.idx_block != 0) {
            std::fprintf(stderr, "lamina: the elastic K/V needs whole indexer blocks in a page\n");
            return 0;
        }
        auto pool = std::make_unique<ElasticPool>();
        pool->n_pages = pages;
        pool->page_size = s.page_size;
        for (const uint64_t b : kv_row_bytes(kv, (uint64_t) s.head_dim)) pool->add((uint64_t) s.n_head_kv * s.page_size * b);
        const uint64_t row = (uint64_t) s.idx_dim * 4;
        pool->add((uint64_t) (s.page_size / s.idx_block) * row, 2 * row);
        if (!pool->range.reserve(pool->bytes()) || !pool_grow(*pool, g_kv_elastic_init, [] { return (VmmChunk) 0; })) {
            std::fprintf(stderr, "lamina: the elastic K/V could not reserve or map its pools\n");
            return 0;
        }
        for (const uint64_t o : pool->off) a.push_back(pool->range.base() + o);
        st.kv_elastic = (int32_t) g_pools.size();
        g_pools.push_back(std::move(pool));
    } else {
        for (const uint64_t b : kv_row_bytes(kv, (uint64_t) s.head_dim))
            a.push_back(c.take_bytes((uint64_t) pages * s.n_head_kv * s.page_size * b));
        a.push_back((uint8_t*) c.take<float>((uint64_t) ((max_cells / s.idx_block) + 2) * s.idx_dim));
    }
    st.idx_pooled = (float*) a.back();
    a.pop_back();
    switch (kv) {
    case KvFormat::Int8:
        st.k_q = (int8_t*) a[0]; st.v_q = (int8_t*) a[1]; st.k_scale = (uint16_t*) a[2]; st.v_scale = (uint16_t*) a[3];
        break;
    case KvFormat::Q4:
        st.k_q4 = a[0]; st.v_q4 = a[1]; st.k_q4s = (uint16_t*) a[2]; st.v_q4s = (uint16_t*) a[3];
        break;
    case KvFormat::K8V4:
        st.k_q = (int8_t*) a[0]; st.k_scale = (uint16_t*) a[1]; st.v_q4 = a[2]; st.v_q4s = (uint16_t*) a[3];
        break;
    default: st.k_pool = (uint16_t*) a[0]; st.v_pool = (uint16_t*) a[1];
    }
    st.page_table = c.take<int32_t>((uint64_t) pages);
    st.n_pages = pages;
    st.max_cells = max_cells;
    st.idx_tail = c.take<float>((uint64_t) (s.idx_block - 1) * s.idx_dim);
    st.idx_dead = c.take<float>((uint64_t) s.idx_dim);
    st.idx_block_pos = c.take<int32_t>(1);
    if (share_rope != nullptr) {
        st.cos_tab = share_rope->cos_tab;
        st.sin_tab = share_rope->sin_tab;
    } else if (g_kv_elastic) {
        // the elastic rotary tables, cos and sin: a chunk's positions built on the host as it is mapped
        auto pool = std::make_unique<ElasticPool>();
        pool->n_pages = pages;
        pool->page_size = s.page_size;
        const uint64_t row = (uint64_t) (s.n_rot / 2) * 4;
        pool->add((uint64_t) s.page_size * row);
        pool->add((uint64_t) s.page_size * row);
        // a range's positions built once: pool_grow fills cos's chunk, then sin's of the same range
        pool->fill = [n_rot = (int) s.n_rot, sc = lamina::kernels::rope_scaling(), row, hc = std::vector<float>(),
                      hs = std::vector<float>(), at = ~0ull](size_t a, uint64_t lo, uint64_t hi, uint8_t* dst) mutable {
            if (lo != at) {
                hc.resize((size_t) ((hi - lo) / 4));
                hs.resize(hc.size());
                lamina::kernels::build_rope_table(n_rot, sc, (int) (hi / row), hc.data(), hs.data(), (int) (lo / row));
                at = lo;
            }
            return cudaMemcpy(dst, (a == 0 ? hc : hs).data(), (size_t) (hi - lo), cudaMemcpyHostToDevice) == cudaSuccess;
        };
        if (!pool->range.reserve(pool->bytes()) || !pool_grow(*pool, g_kv_elastic_init, [] { return (VmmChunk) 0; })) {
            std::fprintf(stderr, "lamina: the elastic rotary tables could not reserve or map their range\n");
            return 0;
        }
        st.cos_tab = (float*) (pool->range.base() + pool->off[0]);
        st.sin_tab = (float*) (pool->range.base() + pool->off[1]);
        g_pools.push_back(std::move(pool));
    } else {
        st.cos_tab = c.take<float>((uint64_t) max_cells * (s.n_rot / 2));
        st.sin_tab = c.take<float>((uint64_t) max_cells * (s.n_rot / 2));
    }
    st.step = c.take<int32_t>((uint64_t) lamina::kernels::kStepCount);
    st.attention_status = c.take<int32_t>(1);
    st.pos_dev = c.take<int32_t>((uint64_t) s.n_head);
    // the pinned staging the uploads copy FROM - see the note on `host_step` in the header
    if (cudaHostAlloc((void**) &st.host_step, lamina::kernels::qsa_step_bytes() + sizeof(int32_t),
                      cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void**) &st.host_pos, (size_t) s.n_head * 4, cudaHostAllocMapped | cudaHostAllocPortable) !=
            cudaSuccess) {
        return 0;
        // the caller sees a zero byte count; a half-built state is worse than none
    }
st.host_step[lamina::kernels::kStepCount] = 0;
// THE ROPE TABLE IS BUILT ON THE HOST IN FLOAT64 and uploaded once, because the reference computes its
// frequencies in float64 and reproducing that on device means double-precision `pow`/`cos` that need not
// agree with the host's libm.  A table shorter than the sequence would have the rotation read past it.
if (share_rope == nullptr) {
    if (!g_kv_elastic) {   // (the elastic tables are built as they are mapped)
        std::vector<float> hc((size_t) max_cells * (s.n_rot / 2)), hs((size_t) max_cells * (s.n_rot / 2));
        lamina::kernels::build_rope_table((int) s.n_rot, lamina::kernels::rope_scaling(), (int) max_cells, hc.data(),
                                          hs.data());
        cudaMemcpy(st.cos_tab, hc.data(), hc.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(st.sin_tab, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    }
    // the native and prompt-path rope kernels read it with LAMINA_ROPE_TABLE=1 (mrope.hpp)
    if (s.n_rot == 64)
        lamina::kernels::rope_table_set(st.cos_tab, st.sin_tab, (int) max_cells, lamina::kernels::rope_scaling());
}
// the page table starts as the IDENTITY, which is the simplest legal mapping and what a caller that does
// not page at all wants.  A real allocator re-points it; nothing outside `kv_append`/`kv_gather` may care.
std::vector<int32_t> tab((size_t) pages);
for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
cudaMemcpy(st.page_table, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice);
return c.used;
}
void qsa_state_free(QsaState& st) {
    if (st.host_step != nullptr) cudaFreeHost(st.host_step);
    if (st.host_pos != nullptr) cudaFreeHost(st.host_pos);
    st.host_step = st.host_pos = nullptr;
}
std::vector<KvArray> qsa_kv_arrays(const QsaState& st, const ModelGeometry& g) {
    const uint64_t hd = (uint64_t) g.head_dim;
    const uint64_t s8 = hd / lamina::kernels::KV_Q8_GROUP * 2, s4 = hd / lamina::kernels::KV_Q4_GROUP * 2;
    std::vector<KvArray> a;
    auto add = [&](void* p, uint64_t row) {
        if (p != nullptr) a.push_back({p, row});
    };
    add(st.k_pool, hd * 2); add(st.v_pool, hd * 2);
    add(st.k_q, hd); add(st.v_q, hd); add(st.k_scale, s8); add(st.v_scale, s8);
    add(st.k_q4, hd / 2); add(st.v_q4, hd / 2); add(st.k_q4s, s4); add(st.v_q4s, s4);
    return a;
}
lamina::kernels::QsaAttnPools qsa_attn_pools(const QsaState& st) {
    lamina::kernels::QsaAttnPools p;
    p.page_table = st.page_table;
    switch (st.kv) {
    case KvFormat::Int8: p.k_q = st.k_q; p.v_q = st.v_q; p.k_scale = st.k_scale; p.v_scale = st.v_scale; break;
    case KvFormat::Q4: p.k_q4 = st.k_q4; p.v_q4 = st.v_q4; p.k_q4s = st.k_q4s; p.v_q4s = st.v_q4s; break;
    case KvFormat::K8V4: p.k_q = st.k_q; p.k_scale = st.k_scale; p.v_q4 = st.v_q4; p.v_q4s = st.v_q4s; break;
    default: p.k_pool = st.k_pool; p.v_pool = st.v_pool;
    }
    return p;
}
void qsa_kv_append_steps(const QsaState& st, const ModelGeometry& g, const int32_t* step, int step_stride,
                         const float* kcur, const float* vcur, int n_tok, void* stream) {
    using namespace lamina::kernels;
    const QsaShapes s = qsa_shapes(g);
    switch (st.kv) {
    case KvFormat::Int8:
        kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step, step_stride, kcur, vcur, n_tok,
                           s, stream);
        break;
    case KvFormat::Q4:
        kv_append_q4_steps(st.k_q4, st.k_q4s, st.v_q4, st.v_q4s, st.page_table, step, step_stride, kcur, vcur, n_tok,
                           s, stream);
        break;
    case KvFormat::K8V4:
        kv_append_q8_steps(st.k_q, nullptr, st.k_scale, nullptr, st.page_table, step, step_stride, kcur, vcur, n_tok, s,
                           stream, 1);
        kv_append_q4_steps(nullptr, nullptr, st.v_q4, st.v_q4s, st.page_table, step, step_stride, kcur, vcur, n_tok, s,
                           stream);
        break;
    default:
        for (int t = 0; t < n_tok; ++t)
            kv_append_step(st.k_pool, st.v_pool, st.page_table, step + (size_t) t * step_stride,
                           kcur + (size_t) t * g.n_head_kv * g.head_dim, vcur + (size_t) t * g.n_head_kv * g.head_dim, s,
                           stream);
    }
}
void qsa_state_zero(const QsaState& st, const ModelGeometry& g, void* stream) {
    const QsaShapes s = qsa_shapes(g);
    cudaStream_t cs = (cudaStream_t) stream;
    if (st.kv_elastic >= 0) {
        // an elastic state zeroes what is mapped, the K/V and the pooled keys (the rest is zeroed when it is mapped)
        const ElasticPool& p = *g_pools[(size_t) st.kv_elastic];
        for (size_t a = 0; a < p.off.size(); ++a)
            cudaMemsetAsync(p.range.base() + p.off[a], 0, (size_t) pool_mapped_bytes(p, a), cs);
    } else {
        const size_t rows = (size_t) st.n_pages * s.n_head_kv * s.page_size;
        for (const KvArray& a : qsa_kv_arrays(st, g)) cudaMemsetAsync(a.data, 0, rows * a.row_bytes, cs);
        cudaMemsetAsync(st.idx_pooled, 0, (size_t) ((st.max_cells / s.idx_block) + 2) * s.idx_dim * 4, cs);
    }
    cudaMemsetAsync(st.idx_tail, 0, (size_t) (s.idx_block - 1) * s.idx_dim * 4, cs);
    cudaMemsetAsync(st.idx_dead, 0, (size_t) s.idx_dim * 4, cs);
}

uint64_t qsa_kv_cell_bytes(const ModelGeometry& g, KvFormat kv) {
    uint64_t n = 0;
    for (const uint64_t b : kv_row_bytes(kv, (uint64_t) g.head_dim)) n += b;
    return n * (uint64_t) g.n_head_kv;
}

uint64_t qsa_state_prompt_bytes(const ModelGeometry& g, int64_t max_cells, int64_t first_page, bool indexer,
                                KvFormat kv) {
    const QsaShapes s = qsa_shapes(g);
    const int64_t pages = (max_cells + s.page_size - 1) / s.page_size, held = std::max<int64_t>(pages - first_page, 1);
    uint64_t n = 0;
    for (const uint64_t b : kv_row_bytes(kv, (uint64_t) s.head_dim))
        n += align_up16((uint64_t) held * s.n_head_kv * s.page_size * b);
    n += align_up16((uint64_t) pages * 4);
    if (indexer)
        n += align_up16((uint64_t) (s.idx_block - 1) * s.idx_dim * 4) + align_up16((uint64_t) s.idx_dim * 4) +
             align_up16((uint64_t) ((max_cells / s.idx_block) + 2) * s.idx_dim * 4) + 16;
    return n;
}

uint64_t qsa_state_prompt_init(const ModelGeometry& g, int64_t max_cells, int64_t first_page, bool indexer, KvFormat kv,
                               void* base, QsaState& st, void* stream) {
    const QsaShapes s = qsa_shapes(g);
    const int64_t pages = (max_cells + s.page_size - 1) / s.page_size, held = std::max<int64_t>(pages - first_page, 1);
    cudaStream_t cs = (cudaStream_t) stream;
    Cursor c{(uint8_t*) base};
    st = QsaState{};
    st.kv = kv;
    st.kv_int8 = kv == KvFormat::Int8;
    std::vector<uint8_t*> a;
    for (const uint64_t b : kv_row_bytes(kv, (uint64_t) s.head_dim)) {
        const uint64_t bytes = (uint64_t) held * s.n_head_kv * s.page_size * b;
        a.push_back(c.take_bytes(bytes));
        cudaMemsetAsync(a.back(), 0, (size_t) bytes, cs);
    }
    switch (kv) {
    case KvFormat::Int8: st.k_q = (int8_t*) a[0]; st.v_q = (int8_t*) a[1]; st.k_scale = (uint16_t*) a[2]; st.v_scale = (uint16_t*) a[3]; break;
    case KvFormat::Q4: st.k_q4 = a[0]; st.v_q4 = a[1]; st.k_q4s = (uint16_t*) a[2]; st.v_q4s = (uint16_t*) a[3]; break;
    case KvFormat::K8V4: st.k_q = (int8_t*) a[0]; st.k_scale = (uint16_t*) a[1]; st.v_q4 = a[2]; st.v_q4s = (uint16_t*) a[3]; break;
    default: st.k_pool = (uint16_t*) a[0]; st.v_pool = (uint16_t*) a[1];
    }
    st.page_table = c.take<int32_t>((uint64_t) pages);
    st.n_pages = held;
    st.max_cells = max_cells;
    if (indexer) {
        st.idx_tail = c.take<float>((uint64_t) (s.idx_block - 1) * s.idx_dim);
        st.idx_dead = c.take<float>((uint64_t) s.idx_dim);
        st.idx_pooled = c.take<float>((uint64_t) ((max_cells / s.idx_block) + 2) * s.idx_dim);
        st.idx_block_pos = c.take<int32_t>(1);
        cudaMemsetAsync(st.idx_tail, 0, (size_t) (s.idx_block - 1) * s.idx_dim * 4, cs);
        cudaMemsetAsync(st.idx_dead, 0, (size_t) s.idx_dim * 4, cs);
        cudaMemsetAsync(st.idx_pooled, 0, (size_t) ((max_cells / s.idx_block) + 2) * s.idx_dim * 4, cs);
        cudaMemsetAsync(st.idx_block_pos, 0, 4, cs);
    }
    std::vector<int32_t> tab((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) std::max<int64_t>(i - first_page, 0);
    if (cudaMemcpy(st.page_table, tab.data(), tab.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) return 0;
    return c.used;
}

// ================================ PER-STAGE TIMING, DEBUG ONLY ================================
//
// **THE ENGINE SPENDS 1.047 ms PER LAYER WITH THE EXPERTS OFF, AND EVERY COST MODEL IN `bench/` PREDICTS LESS
// THAN HALF OF THAT.**  `dense_pass.exe` puts all 302 dense GEMVs at 11.64 ms/token and the 1.377 GiB of
// BF16/F32 weights they exclude at another ~5.9, so ~17.5 ms of the 50.25 ms captured no-pool floor is
// accounted for and ~33 ms is not.  Arithmetic has already discounted the obvious candidates - the GDN
// recurrence moves 6.2 MB of state per token, the QSA attention at 1,024 tokens is 75M MACs over twelve
// layers, and `gr_norm_kernel`'s 4-block launch moves 40 KB - so the attribution has to come from the engine.
//
// Event pairs are recorded per (layer, stage) and read once at the end, so **NOTHING IS SYNCHRONISED DURING
// THE RUN** and the measurement does not perturb what it measures.  That matters here more than usual: a
// per-stage `cudaEventSynchronize` would serialise the pipeline and report the serialised time.
//
// **IT CANNOT BE USED WITH CAPTURED GRAPHS.**  An event record inside a stream capture is silently dropped
// (round 199: 2 nodes for two kernels plus an event record, and the event never completed across a 41 ms
// graph), so this is gated on the `--no-capture` path where the layers are ordinary launches.
constexpr int STAGE_SLOTS = 20;

// ================================ THE TWO ENDS OF A TOKEN ================================
namespace {
// The pack's S2/S4/S8 embedding table, checked against its planes; null with `err` when it cannot be read.
const WeightRef* s_embedding(const WeightTable& tables, const ModelGeometry& g, std::string& err) {
    const WeightRef* w = tables.find(EMBEDDING_NAME);
    if (w == nullptr) { err = "token_embd.weight is missing"; return nullptr; }
    if (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8) {
        err = "embed_row: token_embd.weight is not an S2/S4/S8 tensor";
        return nullptr;
    }
    if (w->codebook_iq4nl) {
        err = "embed_row: an IQ4NL embedding is not supported";
        return nullptr;
    }
    if (w->data == nullptr || w->ne0 <= 0 || w->ne1 <= 0 ||
        w->ne0 != g.n_embd || w->group_elems <= 0 ||
        w->ne0 % w->group_elems != 0 || w->ne0 % (8 / w->code_bits) != 0) {
        err = "embed_row: invalid embedding pointers, dimensions or group size";
        return nullptr;
    }
    const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
    const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
    const uint64_t rows = (uint64_t) w->ne1;
    // Division-based checks avoid overflow in rows * row_bytes and reject partial
    // code/group rows before any device pointer arithmetic.
    if (w->codes_bytes > w->bytes || w->scales_bytes > w->bytes - w->codes_bytes ||
        w->offset_bytes != w->bytes - w->codes_bytes - w->scales_bytes ||
        w->codes_bytes % row_codes != 0 || w->codes_bytes / row_codes != rows ||
        w->codes_bytes % alignof(float) != 0 || w->scales_bytes % sizeof(float) != 0 ||
        (w->scales_bytes / sizeof(float)) % row_groups != 0 ||
        (w->scales_bytes / sizeof(float)) / row_groups != rows ||
        w->has_offset != (w->offset_bytes != 0) ||
        (w->has_offset && w->offset_bytes != w->scales_bytes)) {
        err = "embed_row: embedding planes do not match the loaded tensor shape";
        return nullptr;
    }
    return w;
}
}  // namespace

bool embed_rows(const WeightTable& tables, const ModelGeometry& g, const int32_t* tokens_dev, int64_t n, float* out_dev,
                void* stream, std::string& err) {
    if (n <= 0) return true;
    if (tokens_dev == nullptr || out_dev == nullptr) { err = "embed_rows: invalid tokens or output"; return false; }
    if (const NativeEmbed* ne = native_embed()) {
        ne->gather_dev(tokens_dev, n, out_dev, stream);
        return true;
    }
    const WeightRef* w = s_embedding(tables, g, err);
    if (w == nullptr) return false;
    const auto* codes = static_cast<const uint8_t*>(w->data);
    const auto* scales = reinterpret_cast<const float*>(codes + w->codes_bytes);
    const auto* offsets = w->has_offset ? reinterpret_cast<const float*>(codes + w->codes_bytes + w->scales_bytes)
                                       : nullptr;
    lamina::kernels::embedding_gather_dev(codes, scales, offsets, tokens_dev, (int) n, w->ne0, w->code_bits, w->code_bias,
                                          w->group_elems, (uint64_t) (w->ne0 / (8 / w->code_bits)),
                                          (uint64_t) (w->ne0 / w->group_elems), out_dev, stream);
    return true;
}


}  // namespace lamina::core
