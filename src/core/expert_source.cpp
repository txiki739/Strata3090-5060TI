// src/core/expert_source.cpp - the adapter.  See the header for the three clauses of the contract.
#include "lamina/core/expert_source.hpp"
#include "lamina/core/second_gpu.hpp"
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"

#include "lamina/core/pinned.hpp"
#include "lamina/platform/memory.hpp"
#include "lamina/kernels/elementwise.hpp"
#include "lamina/kernels/quantize_act.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/s2_expert_grouped.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace lamina::core {

// ================================ THE ADAPTER ================================


void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k,
                                float* out) {
    using namespace lamina::kernels::cpu;
    if (d.failed) return;
    if (n_tok < 1 || n_tok > MAXT) {
        d.failed = true;
        d.fail = "a verify window has more tokens than the multi-token expert kernel takes";
        d.fail_layer = d.layers;
        return;
    }
    if ((int64_t) d.act_multi.size() < n_tok) d.act_multi.resize((size_t) MAXT);
    const ExpertLayout& lay = expert_layout();
    if (d.nact_multi.size() < (size_t) MAXT * kNativeActBytes) d.nact_multi.resize((size_t) MAXT * kNativeActBytes);
    if (d.job_of.size() != (size_t) d.n_expert) d.job_of.assign((size_t) d.n_expert, (int16_t) -1);
    if (d.jobs_multi.size() < (size_t) (n_tok * k)) d.jobs_multi.resize((size_t) (MAXT * k));
    static const bool ptrace = std::getenv("LAMINA_POOL_TRACE") != nullptr;
    auto pt = [&](const char* what, long long a = -1) {
        if (ptrace) { std::fprintf(stderr, "pool trace: layer %lld %s %lld\n", (long long) d.layers, what, a); std::fflush(stderr); }
    };
    const auto c0 = std::chrono::steady_clock::now();
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
    if (!d.usage.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
    if (!d.routed.empty())
        for (int64_t i = 0; i < n_tok * k; ++i)
            if (ids[i] >= 0 && ids[i] < d.n_expert) ++d.routed[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]];
    // ---- plan v0.3 P6: who computes each routed entry.  Distinct experts in routing order: the ones the main GPU's
    // VRAM tier holds are its own (the window decides them itself, from the same residency), a GPU tier's when the
    // tiers take this layer, the last pcie_num/256 of the rest the PCIe share (published FIRST, so the GPU reads them
    // while the CPU works), the others the CPU's.
    const int64_t n = n_tok * k;
    if (n > 128) {
        d.failed = true;
        d.fail = "a verify window routes more than 128 entries";
        d.fail_layer = d.layers;
        return;
    }
    const int32_t* res = d.host_res != nullptr ? d.host_res + (size_t) d.layers * (size_t) d.n_expert : nullptr;
    auto in_vram = [&](int32_t e) { return res != nullptr && res[e] >= 0; };   // e in range
    int32_t kind[128];                     // per entry: -1 CPU, 0 VRAM, 1 PCIe, 2 + t GPU tier t
    // tier t's share: group g = the expert in its slot g2_slot[t][g], entries g2_ent[t][g2_start[t][g] ..)
    const int NT = d.n_tiers;
    int n2g[kMaxGpuTiers] = {}, n2e[kMaxGpuTiers] = {};
    int32_t g2_slot[kMaxGpuTiers][128], g2_start[kMaxGpuTiers][129], g2_ent[kMaxGpuTiers][128];
    // the first tier that holds `e` or has prefetched it, and its slot there (-2 - p for prefetch slot p); -1: none
    auto tier_of = [&](int32_t e, int32_t& slot) -> int {
        const size_t at = (size_t) d.layers * (size_t) d.n_expert + (size_t) e;
        for (int t = 0; t < NT; ++t) {
            if (d.tiers[t].res[at] >= 0) { slot = d.tiers[t].res[at]; return t; }
            const int p = d.tiers[t].gpu->prefetched(d.layers, e);
            if (p >= 0) { slot = -2 - p; return t; }
        }
        return -1;
    };
    int64_t distinct[128], first_of[128];
    int nd = 0;
    for (int64_t i = 0; i < n; ++i) {
        first_of[i] = i;
        for (int64_t j = 0; j < i; ++j)
            if (ids[j] == ids[i]) { first_of[i] = first_of[j]; break; }
        if (first_of[i] == i) distinct[nd++] = i;
    }
    // The tiers take this layer's shares only when the layer's misses are big enough that the CPU would need longer
    // for all of them than their round trip (~100 us: ~4 MB of experts at the pool's ~45 GB/s); otherwise they would
    // only add their latency.  Their rows reach the main GPU through the sink.
    GpuPlanSink* const S2 = NT > 0 && d.plan != nullptr && d.plan->tier_flag[0] != nullptr ? d.plan : nullptr;
    d.tier_used = false;
    int8_t tier_q[128];      // per distinct expert: its tier, or -1
    int32_t slot_q[128];     // and its slot there
    for (int q = 0; q < nd; ++q) tier_q[q] = -1;
    if (S2 != nullptr) {
        uint64_t miss_bytes = 0, tier_bytes = 0;
        bool only_there = false;   // an expert only in a tier's VRAM (--second-gpu-pin-gib): no other device has it
        for (int q = 0; q < nd; ++q) {
            const int32_t e = ids[distinct[q]];
            if (e < 0 || e >= d.n_expert || in_vram(e)) continue;
            miss_bytes += lay.blob_bytes(d.layers);
            tier_q[q] = (int8_t) tier_of(e, slot_q[q]);
            if (tier_q[q] >= 0) {
                tier_bytes += lay.blob_bytes(d.layers);
                only_there = only_there || !d.src->in_ram(d.layers, e);
            }
        }
        d.tier_used = tier_bytes > 0 && (only_there || miss_bytes >= d.tier_min_bytes);
        if (tier_bytes > 0 && !d.tier_used) ++d.tier_skipped;
    }
    auto on_tier_now = [&](int q) { return d.tier_used && tier_q[q] >= 0; };
    GpuPlanSink* P = d.plan != nullptr && d.plan->pcie && n <= d.plan->cap ? d.plan : nullptr;
    int nmiss = 0;
    for (int q = 0; q < nd; ++q) {
        const int32_t e = ids[distinct[q]];
        if (e >= 0 && e < d.n_expert && !in_vram(e) && !on_tier_now(q)) ++nmiss;
    }
    const bool pcie_ok = P != nullptr && d.pcie_num > 0 && d.src->layer_mapped(d.layers);
    const int m = pcie_ok ? (nmiss * d.pcie_num) >> 8 : 0;
    int miss_rank = 0, fetches = 0;
    const uint8_t* dma_src[64];
    int64_t pcie_i0[64];
    for (int q = 0; q < nd; ++q) {
        const int64_t i0 = distinct[q];
        const int32_t e = ids[i0];
        int kd = -1;
        if (e >= 0 && e < d.n_expert) {
            if (in_vram(e)) {
                kd = 0;
            } else if (on_tier_now(q)) {
                const int t = tier_q[q];
                kd = 2 + t;
                g2_slot[t][n2g[t]] = slot_q[q];
                g2_start[t][n2g[t]++] = n2e[t];
                for (int64_t i = i0; i < n; ++i)
                    if (first_of[i] == i0) g2_ent[t][n2e[t]++] = (int32_t) i;
            } else {
                if (pcie_ok && miss_rank >= nmiss - m && fetches < P->staging_cap && fetches < 64) {
                    const uint8_t* src = d.src->blob(d.layers, e);
                    if (src != nullptr && d.src->pinned(d.layers, e)) {
                        kd = 1;
                        dma_src[fetches] = src;
                        pcie_i0[fetches] = i0;
                        ++fetches;
                    }
                }
                ++miss_rank;
            }
        }
        for (int64_t i = i0; i < n; ++i)
            if (first_of[i] == i0) kind[i] = kd;
    }
    for (int t = 0; t < NT; ++t) g2_start[t][n2g[t]] = n2e[t];
    if (S2 != nullptr) {   // the main GPU takes each tier's rows once its flag rises: at once without a share
        int32_t* L = S2->gpu2_list;
        int at = 0;
        for (int t = 0; t < NT; ++t) {
            for (int i = 0; i < n2e[t]; ++i) L[4 + at + i] = g2_ent[t][i];
            at += n2e[t];
            L[1 + t] = at;
        }
        L[0] = at;
        std::atomic_thread_fence(std::memory_order_seq_cst);   // every tier's list before any flag
        for (int t = 0; t < NT; ++t)
            if (n2g[t] == 0) {
                *(volatile uint32_t*) S2->tier_flag[t] = S2->ring;
                S2->tier_ring[t] = S2->ring;
            }
    }
    if (P != nullptr) {                          // the PCIe groups: staging slot q, their entries in routing order
        const uint64_t bb = lay.blob_bytes(d.layers);
        int entries = 0;
        for (int q = 0; q < fetches; ++q) {
            const int64_t i0 = pcie_i0[q];
            P->ptr2[q] = P->pcie_mode != 0 ? (unsigned long long) d.src->device_alias(d.layers, ids[i0])
                                           : P->staging + (unsigned long long) q * (unsigned long long) bb;
            P->start2[q] = entries;
            for (int64_t i = i0; i < n; ++i)
                if (first_of[i] == i0) {
                    P->dst[entries] = (int32_t) i;
                    P->tok[entries] = (int32_t) (i / k);
                    ++entries;
                }
            ++d.pcie_experts;
        }
        P->start2[fetches] = entries;
        P->counts[0] = 0;
        P->counts[1] = entries;
        P->counts[2] = fetches;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        pt("publish", fetches);
        if (P->publish) P->publish(P->ctx);
        pt("fetch", fetches);
        if (P->fetch) P->fetch(P->ctx, dma_src, P->pcie_mode != 0 ? 0 : fetches, (size_t) bb);   // the copy engine, beside the CPU's work
    }
    const bool skip_gpu_rows = d.plan != nullptr && d.plan->host_rows_only;
    const auto c1 = std::chrono::steady_clock::now();
    // the gate/up activations: the window's router wrote them as Q8_K rows when the layer takes those
    const uint8_t* xk = d.plan != nullptr ? d.plan->xk : nullptr;
    auto nact = [&](int64_t t) -> const void* {
        return xk != nullptr ? xk + (size_t) t * (size_t) d.plan->xk_stride
                             : d.nact_multi.data() + (size_t) t * kNativeActBytes;
    };
    if (lay.fmt[(size_t) d.layers].gu_type == 42)   // a Q2_0 pack: the Q2_0 kernels' activations
        for (int64_t t = 0; t < n_tok; ++t) act_quant_any(x_f + (size_t) t * H, H, d.act_multi[(size_t) t]);
    else if (xk == nullptr)
        for (int64_t t = 0; t < n_tok; ++t)
            native_quant_act(lay.fmt[(size_t) d.layers], x_f + (size_t) t * H, d.nact_multi.data() + (size_t) t * kNativeActBytes);
    const auto c2 = std::chrono::steady_clock::now();
    int njobs = 0;
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            const int64_t e = ids[i];
            float* row = out + (size_t) i * H;
            if (e < 0 || e >= d.n_expert) {
                d.failed = true;
                d.fail = "a routed expert id is out of range";
                d.fail_layer = d.layers;
                d.fail_expert = e;
                return;
            }
            if (kind[i] >= 0) {             // a GPU computes this entry (a VRAM hit, a PCIe read, a GPU tier)
                if (kind[i] == 0) ++d.cache_hits;
                if (kind[i] >= 2) {         // its row is that tier's to write
                    ++d.tier_entries;
                    ++d.tiers[kind[i] - 2].entries;
                } else if (!skip_gpu_rows) {
                    std::memset(row, 0, (size_t) H * sizeof(float));
                }
                continue;
            }
            ++d.cache_refused;
            int16_t& jo = d.job_of[(size_t) e];
            if (jo < 0) {
                const uint8_t* b = d.src->blob(d.layers, e);
                if (b == nullptr) {
                    d.failed = true;
                    d.fail = "the expert source could not produce a blob";
                    d.fail_layer = d.layers;
                    d.fail_expert = e;
                    ++d.missing;
                    return;
                }
                jo = (int16_t) njobs++;
                ExpertJobMulti& nj = d.jobs_multi[(size_t) jo];
                nj.blob = b;
                nj.nt = 0;
            }
            ExpertJobMulti& jb = d.jobs_multi[(size_t) jo];
            jb.act[jb.nt] = &d.act_multi[(size_t) t];
            jb.nact[jb.nt] = nact(t);
            jb.out[jb.nt] = row;
            ++jb.nt;
            ++d.multi_entries;
        }
    const auto c3 = std::chrono::steady_clock::now();
    // the tiers' shares go out once the CPU's workers have started on theirs: a launch takes the host tens of
    // microseconds.  They take the q8_1 rows the window's router wrote, and write their rows into `out` themselves.
    struct Submit2 {
        ExpertDispatch* d;
        int n_tok;
        int64_t k;
        const int* n_groups;
        const int32_t (*slots)[128];
        const int32_t (*starts)[129];
        const int32_t (*entries)[128];
        float* out;
        bool ok;
    } s2{&d, (int) n_tok, k, n2g, g2_slot, g2_start, g2_ent, out, true};
    void (*submit2)(void*) = [](void* p) {
        Submit2& s = *(Submit2*) p;
        GpuPlanSink& S = *s.d->plan;
        // the layer's gate/up activations as the tiers' kernels take them: Q8_K rows under --expert-math exact on a
        // K-quant layer, else q8_1
        const auto& nf = lamina::kernels::cpu::expert_layout().fmt[(size_t) s.d->layers];
        const bool xk = lamina::kernels::gpu_expert_act(nf.gu_act) == 15;
        const uint8_t* xa = xk ? S.xk : S.x1;
        const int64_t xs = xk ? S.xk_stride : (int64_t) (lamina::kernels::cpu::H / 32) * 36;
        for (int t = 0; t < s.d->n_tiers && s.ok; ++t) {   // (the last tier first measured within 1.4% either way)
            if (s.n_groups[t] <= 0) continue;
            s.ok = xa != nullptr && s.d->tiers[t].gpu->submit(s.d->layers, xa, xs, s.n_tok, s.k, s.slots[t],
                                                               s.starts[t], s.entries[t], s.n_groups[t], s.out,
                                                               S.tier_flag[t], S.ring, s.d->tier_err);
            if (s.ok) S.tier_ring[t] = S.ring;
        }
    };
    bool shares = false;
    for (int t = 0; t < NT; ++t) shares = shares || n2g[t] > 0;
    if (!shares) submit2 = nullptr;
    pt("run", njobs);
    // LAMINA_POOL_FUSED=1 (opt-in): the layer's gate/up and down rows as one batch, no barrier between them
    static const bool pool_fused = [] { const char* v = std::getenv("LAMINA_POOL_FUSED"); return v != nullptr && std::atoi(v) != 0; }();
    if (pool_fused)
        d.pool->run_fused_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs, submit2, &s2);
    else
        d.pool->run_split_multi_native(lay.fmt[(size_t) d.layers], d.jobs_multi.data(), njobs, submit2, &s2);
    if (!s2.ok) {
        d.failed = true;
        d.fail = d.tier_err.c_str();
        d.fail_layer = d.layers;
        return;
    }
    const auto c4 = std::chrono::steady_clock::now();
    pt("ran");
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    d.ms_plan += ms(c0, c1);
    d.ms_actq += ms(c1, c2);
    d.ms_jobs += ms(c2, c3);
    d.ms_run += ms(c3, c4);
    for (int64_t i = 0; i < n_tok * k; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < d.n_expert) d.job_of[(size_t) e] = -1;
    }
    d.multi_misses += njobs;
    ++d.layers;
    d.experts += n_tok * k;
}

void expert_prefetch_multi(ExpertDispatch& d, int64_t layer, const int32_t* ids, const float* w, int64_t n_tok,
                           int64_t k) {
    if (d.failed) return;
    const uint64_t bb = lamina::kernels::cpu::expert_layout().blob_bytes(layer);
    int room = 0;   // the prefetch slots the tiers' budgets take
    for (int t = 0; t < d.n_tiers; ++t)
        room += (int) (std::min)((uint64_t) d.tiers[t].gpu->prefetch_slots(), d.tiers[t].prefetch_bytes / bb);
    if (room <= 0) return;
    // the predicted experts no GPU holds, ranked by their summed routing weight over the window's tokens
    int32_t cand[128];
    float score[128];
    int nc = 0;
    for (int64_t i = 0; i < n_tok * k && i < 128; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= d.n_expert) continue;
        const size_t at = (size_t) layer * (size_t) d.n_expert + (size_t) e;
        bool held = d.host_res[at] >= 0 || !d.src->pinned(layer, e);
        for (int t = 0; t < d.n_tiers && !held; ++t) held = d.tiers[t].res[at] >= 0;
        if (held) continue;
        int c = 0;
        while (c < nc && cand[c] != e) ++c;
        if (c == nc) { cand[nc] = e; score[nc++] = 0.0f; }
        score[c] += w[i];
    }
    // the likeliest first, the first tier's slots filled first
    int m[kMaxGpuTiers] = {}, total = 0;
    for (int t = 0; t < d.n_tiers; ++t) {
        m[t] = (int) (std::min)((uint64_t) (std::min)(nc - total, d.tiers[t].gpu->prefetch_slots()),
                                d.tiers[t].prefetch_bytes / bb);
        total += m[t];
    }
    const uint8_t* src[128];
    for (int q = 0; q < total; ++q) {
        int best = q;
        for (int c = q + 1; c < nc; ++c) if (score[c] > score[best]) best = c;
        std::swap(cand[q], cand[best]);
        std::swap(score[q], score[best]);
        src[q] = d.src->blob(layer, cand[q]);
    }
    for (int t = 0, at = 0; t < d.n_tiers; at += m[t++]) {
        if (m[t] > 0 && !d.tiers[t].gpu->prefetch(layer, cand + at, src + at, m[t], bb, d.tier_err)) {
            d.failed = true;
            d.fail = d.tier_err.c_str();
            d.fail_layer = layer;
            return;
        }
    }
}


// ================================ THE RESIDENT ARENA (R2.1) ================================

namespace {
// Plan v0.3 P6: the arena straight from the model's GGUF shards.  Each layer's gate, up and down tensors hold the
// 512 experts one after another; they are read in chunks and each expert's slice lands at its place in the blob
// [gate rows | up rows | down rows] (expert_layout.hpp).
struct GgufSpan {
    size_t shard = 0;
    uint64_t offset = 0;   ///< in the shard's file
};

/// Where each layer's gate, up and down tensors are (3 per layer), found by name in whichever shard holds them
/// and checked against the pack's formats, so a pack and a model that do not belong together are refused.
bool locate_experts_gguf(const std::vector<std::string>& shards, const lamina::kernels::cpu::ExpertLayout& lay,
                         std::vector<GgufSpan>& out, std::string& err) {
    static const char* roles[3] = {"gate", "up", "down"};
    out.assign((size_t) (3 * lay.n_layers), GgufSpan{});
    try {
        const lamina::GgufModel model(shards);
        for (int64_t l = 0; l < lay.n_layers; ++l) {
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, lay.bytes[(size_t) l] - fm.down_off};
            for (int r = 0; r < 3; ++r) {
                const std::string name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                size_t s = 0;
                const lamina::TensorInfo* t = model.find(name, &s);
                int be = 0, bb = 0;
                if (t == nullptr || (int) t->type != (r < 2 ? fm.gu_type : fm.d_type) ||
                    !lamina::block_geometry(t->type, be, bb) ||
                    t->elements() / (uint64_t) be * (uint64_t) bb != per[r] * (uint64_t) lay.n_expert) {
                    err = "ArenaExpertSource: " + name + " is missing from the model or differs from the pack's "
                          "native_experts.txt";
                    return false;
                }
                out[(size_t) (3 * l + r)] = GgufSpan{s, model.shard(s).data_start() + t->offset};
            }
        }
        return true;
    } catch (const std::exception& e) {
        err = std::string("ArenaExpertSource: ") + e.what();
        return false;
    }
}

// `idx` (empty, or per (layer, expert)): an expert's place among its layer's blobs, -1 for one left out (only in VRAM,
// `skipped` bytes in all).  `ready` (or null): called before a layer is read, until its place in `layer_base` exists
// (false: stop, the load fails).
LoadStats load_experts_gguf(const std::vector<std::string>& shards, const std::vector<GgufSpan>& spans,
                            const std::vector<uint8_t*>& layer_base, const lamina::kernels::cpu::ExpertLayout& lay,
                            int threads, const std::function<void(int64_t)>& done = nullptr,
                            const std::vector<int32_t>& idx = {}, uint64_t skipped = 0,
                            const std::function<bool(int64_t)>& ready = nullptr, bool uncached = false) {
    LoadStats st;
    st.layers = (uint64_t) lay.n_layers;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<int64_t> next{0};
    std::atomic<bool> bad{false};
#if !defined(_WIN32)
    // Linux: a role tensor's slices read with preadv straight into their places in the blobs, LAMINA_LOAD_BATCH
    // experts (16) a call - one large forward read, as the staged copy made, without the staging buffer and its copy
    static const int batch = [] {
        const char* v = std::getenv("LAMINA_LOAD_BATCH");
        return v != nullptr && std::atoi(v) > 0 ? std::min(std::atoi(v), 512) : 16;
    }();
    auto worker = [&]() {
        std::vector<int> fds(shards.size(), -1);
        std::vector<struct iovec> iov((size_t) batch);
        // reads `n` iovecs from `at`, a short read resumed where it stopped
        auto read_all = [](int fd, struct iovec* v, int n, uint64_t at) {
            while (n > 0) {
                const ssize_t r = ::preadv(fd, v, n, (off_t) at);
                if (r <= 0) return false;
                at += (uint64_t) r;
                for (size_t left = (size_t) r; left > 0 && n > 0;) {
                    if (left >= v->iov_len) {
                        left -= v->iov_len;
                        ++v;
                        --n;
                    } else {
                        v->iov_base = (uint8_t*) v->iov_base + left;
                        v->iov_len -= left;
                        left = 0;
                    }
                }
            }
            return true;
        };
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            if (ready && !ready(l)) { bad = true; break; }
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            const int32_t* ix = idx.empty() ? nullptr : idx.data() + (size_t) l * (size_t) lay.n_expert;
            for (int r = 0; r < 3 && !bad; ++r) {
                const GgufSpan& sp = spans[(size_t) (3 * l + r)];
                int& fd = fds[sp.shard];
                if (fd < 0) {
                    fd = ::open(shards[sp.shard].c_str(), O_RDONLY | O_CLOEXEC);
                    if (fd >= 0) (void) posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
                }
                if (fd < 0) { bad = true; break; }
                // runs of up to `batch` experts the arena holds, one forward read each
                for (int64_t e0 = 0; e0 < lay.n_expert;) {
                    if (ix != nullptr && ix[e0] < 0) { ++e0; continue; }   // only in VRAM
                    int n = 0;
                    for (; n < batch && e0 + n < lay.n_expert && (ix == nullptr || ix[e0 + n] >= 0); ++n) {
                        const int64_t at_e = ix != nullptr ? ix[e0 + n] : e0 + n;
                        iov[(size_t) n] = {layer_base[(size_t) l] + (uint64_t) at_e * blob + at[r], (size_t) per[r]};
                    }
                    if (!read_all(fd, iov.data(), n, sp.offset + (uint64_t) e0 * per[r])) {
                        bad = true;
                        break;
                    }
                    if (uncached)   // --no-file-cache: the pages just read leave the page cache
                        (void) posix_fadvise(fd, (off_t) (sp.offset + (uint64_t) e0 * per[r]), (off_t) ((uint64_t) n * per[r]),
                                             POSIX_FADV_DONTNEED);
                    e0 += n;
                }
            }
            if (!bad && done) done(l);
        }
        for (int fd : fds)
            if (fd >= 0) ::close(fd);
    };
#else
    // fread on a FILE* (upstream #230: MSVC's std::ifstream::read splits a request into 4095-byte freads, 6.2 GiB/s
    // from the file cache against 9.0, 1.6 from the drive against 4.2-4.4); the copy through the cache leaves the file
    // there for the next start.  --no-file-cache: unbuffered reads (FILE_FLAG_NO_BUFFERING), 4 KB aligned, into an
    // aligned staging buffer (4.3-4.4 GiB/s from this PC's drive)
    auto worker = [&]() {
        std::vector<FILE*> files(shards.size(), nullptr);
        std::vector<HANDLE> handles(shards.size(), INVALID_HANDLE_VALUE);
        std::vector<uint8_t> buf;
        uint8_t* abuf = nullptr;   // the unbuffered reads' staging
        size_t abuf_bytes = 0;
        for (;;) {
            const int64_t l = next.fetch_add(1);
            if (l >= lay.n_layers || bad) break;
            if (ready && !ready(l)) { bad = true; break; }
            const auto& fm = lay.fmt[(size_t) l];
            const uint64_t blob = lay.bytes[(size_t) l];
            const uint64_t per[3] = {fm.up_off, fm.up_off, blob - fm.down_off};
            const uint64_t at[3] = {0, fm.up_off, fm.down_off};
            const int32_t* ix = idx.empty() ? nullptr : idx.data() + (size_t) l * (size_t) lay.n_expert;
            for (int r = 0; r < 3 && !bad; ++r) {
                const GgufSpan& sp = spans[(size_t) (3 * l + r)];
                const uint64_t total = per[r] * (uint64_t) lay.n_expert, chunk = per[r] * 16;   // 16 experts a read
                FILE*& f = files[sp.shard];
                HANDLE& h = handles[sp.shard];
                if (uncached) {
                    if (h == INVALID_HANDLE_VALUE) {
                        const std::string& s = shards[sp.shard];
                        std::wstring w((size_t) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0), L'\0');
                        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), (int) w.size());
                        h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                    }
                    if (h == INVALID_HANDLE_VALUE) { bad = true; break; }
                    const size_t need = (size_t) chunk + 2 * 4096;
                    if (abuf_bytes < need) {
                        if (abuf != nullptr) VirtualFree(abuf, 0, MEM_RELEASE);
                        abuf = (uint8_t*) VirtualAlloc(nullptr, need, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                        abuf_bytes = abuf != nullptr ? need : 0;
                    }
                    if (abuf == nullptr) { bad = true; break; }
                } else {
                    if (f == nullptr) f = std::fopen(shards[sp.shard].c_str(), "rb");
                    if (f == nullptr) { bad = true; break; }
                    buf.resize((size_t) chunk);
                }
                for (uint64_t pos = 0; pos < total; pos += chunk) {
                    const uint64_t n = std::min<uint64_t>(chunk, total - pos);
                    const uint8_t* src = buf.data();
                    if (uncached) {   // from the sector below to the one past the end
                        const uint64_t at0 = (sp.offset + pos) / 4096 * 4096, head = sp.offset + pos - at0;
                        OVERLAPPED ov{};
                        ov.Offset = (DWORD) at0;
                        ov.OffsetHigh = (DWORD) (at0 >> 32);
                        DWORD got = 0;
                        if (!ReadFile(h, abuf, (DWORD) ((head + n + 4095) / 4096 * 4096), &got, &ov) || got < head + n) {
                            bad = true;
                            break;
                        }
                        src = abuf + head;
                    } else if (_fseeki64(f, (long long) (sp.offset + pos), SEEK_SET) != 0 ||
                               std::fread(buf.data(), 1, (size_t) n, f) != (size_t) n) {
                        bad = true;
                        break;
                    }
                    for (uint64_t k = 0; k < n / per[r]; ++k) {
                        const uint64_t e = pos / per[r] + k;
                        if (ix != nullptr && ix[e] < 0) continue;   // only in VRAM
                        const uint64_t at_e = ix != nullptr ? (uint64_t) ix[e] : e;
                        std::memcpy(layer_base[(size_t) l] + at_e * blob + at[r], src + k * per[r], (size_t) per[r]);
                    }
                }
            }
            if (!bad && done) done(l);
        }
        for (FILE* f : files)
            if (f != nullptr) std::fclose(f);
        for (HANDLE h : handles)
            if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        if (abuf != nullptr) VirtualFree(abuf, 0, MEM_RELEASE);
    };
#endif
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    if (bad) {
        st.seconds = -1.0;
        return st;
    }
    st.bytes = lay.total - skipped;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}
// Pageable memory for the arena's layers past the pinning budget.
void* reserve_pageable(uint64_t bytes) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = mmap(nullptr, (size_t) bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
#endif
}

void release_pageable(void* p, uint64_t bytes) {
#if defined(_WIN32)
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, (size_t) bytes);
#endif
}
}  // namespace

ArenaExpertSource::~ArenaExpertSource() { close(); }

bool ArenaExpertSource::open(int64_t n_layers, int64_t n_expert, int threads, std::string& err) {
    close();
    if (gguf_.empty()) {
        err = "ArenaExpertSource: no model GGUF to read the experts from";
        return false;
    }
    // plan v0.3 P6: the layout (the pack's per-layer blobs) was loaded by the driver
    const lamina::kernels::cpu::ExpertLayout& lay = lamina::kernels::cpu::expert_layout();
    if (lay.n_layers != n_layers || lay.n_expert != n_expert) {
        err = "ArenaExpertSource: the expert layout was loaded for a different geometry";
        return false;
    }
    // --vram-pin-gib: the experts only in VRAM are left out, each layer's others packed in expert order
    std::vector<int64_t> kept((size_t) n_layers, n_expert);
    if (!vram_only_.empty()) {
        if (vram_only_.size() != (size_t) (n_layers * n_expert)) {
            err = "ArenaExpertSource: the VRAM-only mask was built for a different geometry";
            return false;
        }
        idx_.assign(vram_only_.size(), -1);
        for (int64_t l = 0; l < n_layers; ++l) {
            int32_t at = 0;
            for (int64_t e = 0; e < n_expert; ++e) {
                const size_t i = (size_t) (l * n_expert + e);
                if (vram_only_[i] == 0) idx_[i] = at++;
                else vram_only_bytes_ += lay.blob_bytes(l);
            }
            kept[(size_t) l] = at;
        }
    }
    const uint64_t want = lay.total - vram_only_bytes_;

    // the expert tensors found before the allocation, not after: a model that does not match the layout is refused
    // before 34 GB and a minute of loading are spent
    std::vector<GgufSpan> spans;
    if (!locate_experts_gguf(gguf_, lay, spans, err)) return false;

    // one pinned block per layer, each followed by room for a whole slot-sized copy (the largest blob) from its last
    // expert; the layers past the pinning budget in one pageable range, locked resident
    const uint64_t maxb = lay.max_blob;
    std::vector<uint64_t> room;
    for (int64_t l = 0; l < n_layers; ++l) {
        const uint64_t bytes = lay.blob_bytes(l) * (uint64_t) kept[(size_t) l];
        // (never empty: a layer whose experts are all in VRAM still registers a page)
        room.push_back(std::max<uint64_t>(bytes + (maxb - lay.blob_bytes(l)), 4096));
    }
    uint64_t pinned_total = 0;
    std::string tail_note;
    bool anon = false;
#if !defined(_WIN32)
    // LAMINA_PIN_AFTER_COPY=0 keeps the cudaHostAlloc blocks (the A/B arm)
    static const bool pin_after = [] {
        const char* v = std::getenv("LAMINA_PIN_AFTER_COPY");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (pin_after) {
        constexpr uint64_t kAlign = 2ull << 20;   // every layer on pages of its own: each one registers alone
        std::vector<uint64_t> at;
        uint64_t total = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            at.push_back(total);
            total += (room[(size_t) l] + kAlign - 1) / kAlign * kAlign;
        }
        void* m = mmap(nullptr, (size_t) total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED) {
            anon_ = m;
            anon_bytes_ = total;
            anon = true;
            for (int64_t l = 0; l < n_layers; ++l) layer_base_.push_back((uint8_t*) m + at[(size_t) l]);
        }
    }
#endif
    // a pinned block per layer, in order; the layers past the first refusal in one pageable range (false: none)
    bool reserve_failed = false;
    auto place_layers = [&](const std::function<void(int64_t)>& placed) {
        for (int64_t l = 0; l < n_layers; ++l) {
            void* p = nullptr;
            if (cudaHostAlloc(&p, (size_t) room[(size_t) l], cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
                (void) cudaGetLastError();
                for (int64_t m = l; m < n_layers; ++m) tail_bytes_ += room[(size_t) m];
                tail_ = reserve_pageable(tail_bytes_);
                if (tail_ == nullptr) {
                    reserve_failed = true;
                    placed(n_layers);
                    return;
                }
                const lamina::platform::LockResult lr = lamina::platform::lock_resident(tail_, tail_bytes_);
                tail_note = lr.note;
                tail_locked_ = lr.locked_bytes;
                uint8_t* at = (uint8_t*) tail_;
                for (int64_t m = l; m < n_layers; ++m) {
                    layer_base_[(size_t) m] = at;
                    at += room[(size_t) m];
                }
                placed(n_layers);
                return;
            }
            blocks_.push_back((uint8_t*) p);
            layer_base_[(size_t) l] = (uint8_t*) p;
            void* d = nullptr;
            layer_dev_[(size_t) l] = cudaHostGetDevicePointer(&d, p, 0) == cudaSuccess ? (const uint8_t*) d : nullptr;
            (void) cudaGetLastError();
            pinned_total += room[(size_t) l];
            placed(l + 1);
        }
    };
    // A layer is read as soon as its block exists: the blocks are allocated on a thread of their own (all of them
    // first, 48 cudaHostAlloc blocks of 1-1.5 GiB, delayed the reads by 3.5 s on IQ3_S, 6 on UD-Q4_K_XL)
    std::mutex placed_mu;
    std::condition_variable placed_cv;
    int64_t n_placed = 0;
    std::thread placer;
    if (!anon) {
        layer_base_.assign((size_t) n_layers, nullptr);
        layer_dev_.assign((size_t) n_layers, nullptr);
        auto publish = [&](int64_t n) {
            {
                std::lock_guard<std::mutex> lk(placed_mu);
                n_placed = n;
            }
            placed_cv.notify_all();
        };
        int dev0 = 0;
        (void) cudaGetDevice(&dev0);
        placer = std::thread([&, dev0] {
            (void) cudaSetDevice(dev0);
            place_layers(publish);
        });
    }
    auto layer_ready = [&](int64_t l) {
        std::unique_lock<std::mutex> lk(placed_mu);
        placed_cv.wait(lk, [&] { return n_placed > l; });
        return !reserve_failed;
    };
    if (const char* v = std::getenv("LAMINA_LOAD_THREADS"); v != nullptr && std::atoi(v) > 0) threads = std::atoi(v);
    // the anonymous arena: each layer page-locked by the thread that read it, as soon as it is in place (the
    // registrations overlap the other layers' reads, and the pages are not left to the swap meanwhile)
    std::vector<char> reg_ok((size_t) n_layers, 0);
    int dev = 0;
    (void) cudaGetDevice(&dev);
    auto on_layer = [&](int64_t l) {
        if (!anon) return;
        (void) cudaSetDevice(dev);
        if (cudaHostRegister(layer_base_[(size_t) l], (size_t) room[(size_t) l],
                             cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess)
            reg_ok[(size_t) l] = 1;
        else
            (void) cudaGetLastError();
    };
    const LoadStats st = load_experts_gguf(gguf_, spans, layer_base_, lay, threads, on_layer, idx_, vram_only_bytes_,
                                           anon ? nullptr : std::function<bool(int64_t)>(layer_ready), uncached_);
    if (placer.joinable()) placer.join();
    if (!anon) pinned_layers_ = (int64_t) blocks_.size();
    if (reserve_failed) {
        close();
        err = "ArenaExpertSource: the arena could not be reserved (" + std::to_string(tail_bytes_) + " B)";
        return false;
    }
    if (st.bytes != want) {
        for (int64_t l = 0; l < n_layers; ++l)
            if (reg_ok[(size_t) l]) (void) cudaHostUnregister(layer_base_[(size_t) l]);
        close();
        err = "ArenaExpertSource: the load read " + std::to_string(st.bytes) + " B of " + std::to_string(want);
        return false;
    }
#if !defined(_WIN32)
    if (anon) {   // the registered layers up to the first refused one; from there on, locked resident instead
        int64_t l = 0;
        while (l < n_layers && reg_ok[(size_t) l]) ++l;
        for (int64_t m = l + 1; m < n_layers; ++m)
            if (reg_ok[(size_t) m]) (void) cudaHostUnregister(layer_base_[(size_t) m]);
        for (int64_t k = 0; k < l; ++k) {
            uint8_t* p = layer_base_[(size_t) k];
            registered_.push_back({p, room[(size_t) k]});
            void* d = nullptr;
            layer_dev_.push_back(cudaHostGetDevicePointer(&d, p, 0) == cudaSuccess ? (const uint8_t*) d : nullptr);
            (void) cudaGetLastError();
            pinned_total += room[(size_t) k];
        }
        pinned_layers_ = l;
        for (int64_t m = l; m < n_layers; ++m) layer_dev_.push_back(nullptr);
        if (l < n_layers) {
            anon_lock_ = layer_base_[(size_t) l];
            anon_lock_bytes_ = (uint64_t) ((uint8_t*) anon_ + anon_bytes_ - anon_lock_);
            tail_note = lamina::platform::lock_resident(anon_lock_, anon_lock_bytes_).note;
        }
    }
#endif
    // LAMINA_LOAD_VERIFY=1: every blob read again through std::ifstream (not the load's own reads) and compared
    // with the arena's bytes - a test, a minute or two
    if (const char* v = std::getenv("LAMINA_LOAD_VERIFY"); v != nullptr && std::atoi(v) != 0) {
        std::vector<uint8_t> want_b;
        int64_t checked = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            const uint64_t b = lay.blob_bytes(l);
            for (int64_t e = 0; e < n_expert; ++e) {
                const int64_t at_e = idx_.empty() ? e : idx_[(size_t) (l * n_expert + e)];
                if (at_e < 0) continue;   // only in VRAM: the fill compares its slot with the file
                want_b.assign((size_t) b, 0);
                const auto& fm = lay.fmt[(size_t) l];
                const uint64_t per[3] = {fm.up_off, fm.up_off, b - fm.down_off};
                const uint64_t at[3] = {0, fm.up_off, fm.down_off};
                for (int r = 0; r < 3; ++r) {
                    const GgufSpan& sp = spans[(size_t) (3 * l + r)];
                    std::ifstream f(gguf_[sp.shard], std::ios::binary);
                    f.seekg((std::streamoff) (sp.offset + (uint64_t) e * per[r]));
                    f.read((char*) want_b.data() + at[r], (std::streamsize) per[r]);
                }
                if (std::memcmp(want_b.data(), layer_base_[(size_t) l] + (uint64_t) at_e * b, (size_t) b) != 0) {
                    close();
                    err = "ArenaExpertSource: LAMINA_LOAD_VERIFY: layer " + std::to_string(l) + " expert " +
                          std::to_string(e) + " differs from the file";
                    return false;
                }
                ++checked;
            }
        }
        std::fprintf(stderr, "ArenaExpertSource: LAMINA_LOAD_VERIFY: %lld blobs equal to the file\n", (long long) checked);
    }
    note_ = std::to_string(pinned_layers_) + " of " + std::to_string(n_layers) + " layers pinned (" +
            std::to_string(pinned_total >> 30) + (anon ? " GiB, page-locked as each layer loaded)" : " GiB, a cudaHostAlloc block each)");
    if (tail_ != nullptr || anon_lock_ != nullptr) note_ += "; the others pageable: " + tail_note;
    if (uncached_) note_ += "; read past the file cache";
    if (vram_only_bytes_ > 0) {
        char buf[160];
        int64_t n = 0;
        for (const uint8_t v : vram_only_) n += v != 0;
        std::snprintf(buf, sizeof buf, "; %lld experts (%.2f GiB) only in VRAM, left out", (long long) n,
                      (double) vram_only_bytes_ / 1073741824.0);
        note_ += buf;
    }
    for (const GgufSpan& sp : spans) spans_.emplace_back(sp.shard, sp.offset);
#if !defined(_WIN32)
    if (!vram_only_.empty())
        for (const auto& path_s : gguf_) {
            fds_.push_back(::open(path_s.c_str(), O_RDONLY | O_CLOEXEC));
            if (fds_.back() < 0) {
                close();
                err = "ArenaExpertSource: cannot open " + path_s;
                return false;
            }
        }
#endif
    blobs_ = n_layers * n_expert;
    n_expert_ = n_expert;
    reads_ = 0;
    gib_per_s_ = st.gib_per_second();
    return true;
}

void ArenaExpertSource::close() {
    for (uint8_t* p : blocks_) cudaFreeHost(p);
    blocks_.clear();
#if !defined(_WIN32)
    for (const auto& r : registered_) (void) cudaHostUnregister(r.first);
    registered_.clear();
    if (anon_lock_ != nullptr) {
        lamina::platform::unlock_resident(anon_lock_, anon_lock_bytes_);
        anon_lock_ = nullptr;
        anon_lock_bytes_ = 0;
    }
    if (anon_ != nullptr) {
        munmap(anon_, (size_t) anon_bytes_);
        anon_ = nullptr;
        anon_bytes_ = 0;
    }
#endif
    pinned_layers_ = 0;
    if (tail_ != nullptr) {
        lamina::platform::unlock_resident(tail_, tail_locked_);
        release_pageable(tail_, tail_bytes_);
        tail_ = nullptr;
        tail_bytes_ = 0;
        tail_locked_ = 0;
    }
    layer_base_.clear();
    layer_dev_.clear();
    blobs_ = 0;
    n_expert_ = 0;
    idx_.clear();
    blk_.clear();
    place_off_.clear();
    place_slow_.clear();
    place_occ_.clear();
    pending_.clear();
    pending_at_ = 0;
    vram_only_bytes_ = 0;
    spans_.clear();
#if !defined(_WIN32)
    for (const int fd : fds_) if (fd >= 0) ::close(fd);
#endif
    fds_.clear();
}

bool ArenaExpertSource::pinned(int64_t layer, int64_t expert) const {
    return layer >= 0 && layer < pinned_layers_ && expert >= 0 && expert < n_expert_ && in_ram(layer, expert);
}

const uint8_t* ArenaExpertSource::device_alias(int64_t layer, int64_t expert) const {
    if (!pinned(layer, expert) || layer_dev_[(size_t) layer] == nullptr) return nullptr;
    const size_t i = (size_t) (layer * n_expert_ + expert);
    const int64_t at = idx_.empty() ? expert : idx_[i];
    // (place() moves blobs between mapped blocks only)
    return layer_dev_[blk_.empty() ? (size_t) layer : (size_t) blk_[i]] +
           (uint64_t) at * lamina::kernels::cpu::expert_layout().blob_bytes(layer);
}

const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= (int64_t) layer_base_.size() || expert < 0 || expert >= n_expert_) return nullptr;
    const size_t i = (size_t) (layer * n_expert_ + expert);
    const int64_t at = idx_.empty() ? expert : idx_[i];
    if (at < 0) return nullptr;   // only in VRAM
    reads_.fetch_add(1, std::memory_order_relaxed);
    // Pointer arithmetic into resident memory: no fault, no copy, no mapping.
    return layer_base_[blk_.empty() ? (size_t) layer : (size_t) blk_[i]] +
           (uint64_t) at * lamina::kernels::cpu::expert_layout().blob_bytes(layer);
}

bool ArenaExpertSource::read_file_blob(int64_t layer, int64_t expert, uint8_t* dst, std::string& err) const {
    const auto& lay = lamina::kernels::cpu::expert_layout();
    if (layer < 0 || layer >= lay.n_layers || expert < 0 || expert >= lay.n_expert ||
        spans_.size() != (size_t) (3 * lay.n_layers)) {
        err = "ArenaExpertSource: no model to read an expert from";
        return false;
    }
    const auto& fm = lay.fmt[(size_t) layer];
    const uint64_t b = lay.blob_bytes(layer);
    const uint64_t per[3] = {fm.up_off, fm.up_off, b - fm.down_off};
    const uint64_t at[3] = {0, fm.up_off, fm.down_off};
    std::memset(dst, 0, (size_t) b);   // as the arena's blobs: any bytes between the roles are zero
    for (int r = 0; r < 3; ++r) {
        const auto& [shard, offset] = spans_[(size_t) (3 * layer + r)];
        const uint64_t from = offset + (uint64_t) expert * per[r];
#if !defined(_WIN32)
        if (shard >= fds_.size()) {
            err = "ArenaExpertSource: the model's files are open only with experts only in VRAM";
            return false;
        }
        for (uint64_t got = 0; got < per[r];) {
            const ssize_t n = ::pread(fds_[shard], dst + at[r] + got, (size_t) (per[r] - got), (off_t) (from + got));
            if (n <= 0) {
                err = "ArenaExpertSource: reading an expert from " + gguf_[shard] + " failed";
                return false;
            }
            got += (uint64_t) n;
        }
#else
        std::ifstream f(gguf_[shard], std::ios::binary);
        f.seekg((std::streamoff) from);
        f.read((char*) dst + at[r], (std::streamsize) per[r]);
        if ((uint64_t) f.gcount() != per[r]) {
            err = "ArenaExpertSource: reading an expert from " + gguf_[shard] + " failed";
            return false;
        }
#endif
    }
    return true;
}

}  // namespace lamina::core
