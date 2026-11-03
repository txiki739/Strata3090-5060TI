// src/core/prompt_cache.cpp - the --serve prompt cache (see include/lamina/core/prompt_cache.hpp).
#include "lamina/core/prompt_cache.hpp"

#include "lamina/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace lamina::core {

PromptCache::PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream,
                         int chain_cap, uint64_t park_budget, size_t park_slots, int64_t min_tokens, bool spread)
    : g_(g), ss_(ss), kv_(std::move(kv)), stream_(stream), chain_cap_(chain_cap), spread_(spread),
      ckpt_bytes_(session_ckpt_bytes(g)), parked_(park_budget, park_slots, min_tokens, ckpt_bytes_) {}

PromptCache::~PromptCache() {
    for (const auto& [p, pinned] : owned_) {
        if (pinned) cudaFreeHost(p);
        else std::free(p);
    }
}

void* PromptCache::take_buffer() {
    if (!free_.empty()) {
        void* h = free_.back();
        free_.pop_back();
        return h;
    }
    void* h = nullptr;
    const bool pinned = cudaMallocHost(&h, (size_t) ckpt_bytes_) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
        h = (void*) std::malloc((size_t) ckpt_bytes_);   // pageable: slower copies, same result
    }
    if (h != nullptr) owned_.emplace_back(h, pinned);
    return h;
}

void PromptCache::give_back(void* h) {
    if (h != nullptr) free_.push_back(h);
}

void PromptCache::free_ckpts(Seq& seq) {
    for (const Ckpt& c : seq.ckpts) give_back(c.host);
    seq.ckpts.clear();
}

void PromptCache::enforce_cap(Seq& seq) {
    while ((int64_t) seq.ckpts.size() > chain_cap_) {
        const size_t n = seq.ckpts.size();
        std::vector<int64_t> lens(n);
        std::vector<uint64_t> stamps(n);
        std::unique_ptr<bool[]> tail(new bool[n]), pinned(new bool[n]);
        for (size_t i = 0; i < n; ++i) {
            lens[i] = seq.ckpts[i].pos;
            stamps[i] = seq.ckpts[i].stamp;
            tail[i] = seq.ckpts[i].tail;
            pinned[i] = seq.ckpts[i].pos == seq.pin;
        }
        const size_t v = spread_ ? conv_cache::spread_victim(lens.data(), stamps.data(), n, chain_cap_, tail.get(),
                                                             pinned.get())
                                 : conv_cache::eviction_victim(stamps.data(), n, chain_cap_, tail.get(),
                                                               pinned.get());
        give_back(seq.ckpts[v].host);
        seq.ckpts.erase(seq.ckpts.begin() + (ptrdiff_t) v);
    }
}

void PromptCache::release(Seq& seq, int64_t keep_upto) {
    for (size_t i = 0; i < seq.ckpts.size();) {
        if (seq.ckpts[i].pos > keep_upto) {
            give_back(seq.ckpts[i].host);
            seq.ckpts.erase(seq.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
}

std::vector<PromptCache::Part> PromptCache::kv_parts(int64_t cells) const {
    // [page][kv_head][page_size][head_dim] with an identity page table: the first cells are one prefix of each
    // pool.  The indexer's pooled rows are one per idx_block cells.
    const lamina::kernels::QsaShapes sh = lamina::kernels::qsa_real_shapes();
    const uint64_t row = (uint64_t) g_.idx_key_dim * sizeof(float);
    std::vector<Part> parts;
    for (QsaState* st : kv_) {
        const uint64_t pages = (uint64_t) std::min<int64_t>(st->n_pages, (cells + sh.page_size - 1) / sh.page_size);
        const uint64_t c = pages * (uint64_t) g_.n_head_kv * (uint64_t) sh.page_size;
        for (const KvArray& a : qsa_kv_arrays(*st, g_)) parts.push_back({a.data, c * a.row_bytes});
        const int64_t rows = std::min<int64_t>(cells / sh.idx_block + 2, st->max_cells / sh.idx_block + 2);
        parts.push_back({st->idx_pooled, (uint64_t) rows * row});
    }
    return parts;
}

bool PromptCache::kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) {
    cudaStream_t cs = (cudaStream_t) stream_;
    const std::vector<Part> parts = kv_parts(cells), layout = kv_parts(held);
    for (size_t i = 0; i < parts.size(); ++i) {
        const Part& p = parts[i];
        const cudaError_t e = to_host ? cudaMemcpyAsync(host, p.dev, (size_t) p.bytes, cudaMemcpyDeviceToHost, cs)
                                      : cudaMemcpyAsync(p.dev, host, (size_t) p.bytes, cudaMemcpyHostToDevice, cs);
        if (e != cudaSuccess) return session_copy_failed(stream_);
        host += layout[i].bytes;
    }
    return cudaStreamSynchronize(cs) == cudaSuccess || session_copy_failed(stream_);
}

void PromptCache::stash_live(int64_t keep_upto) {
    const int64_t L = (int64_t) live_.tokens.size();
    if (!enabled() || !parked_.wants(L)) return;
    if (ensure_kv && !ensure_kv(L)) return;
    uint64_t bytes = 0;
    for (const Part& p : kv_parts(L)) bytes += p.bytes;
    checkpoint(L);   // the end state: the session is still at L
    // What the parked conversation carries: its K/V copy and the chain after `keep_upto`.
    uint64_t incoming = bytes;
    for (const Ckpt& c : live_.ckpts)
        if (c.pos > keep_upto) incoming += ckpt_bytes_;
    std::vector<Seq> evicted;
    if (!parked_.make_room(incoming, evicted)) return;   // parks nothing: the live one keeps going unsaved
    for (Seq& s : evicted) {   // whole conversations left to pay for this one: their buffers are recycled
        std::fprintf(stderr, "lamina serve: prompt cache: parked conversation %zu cells evicted\n", s.tokens.size());
        free_ckpts(s);
    }
    Seq s;
    s.kv.reset(new (std::nothrow) uint8_t[bytes]);
    if (!s.kv || !kv_copy(L, L, s.kv.get(), true)) return;
    for (size_t i = 0; i < live_.ckpts.size();) {
        if (live_.ckpts[i].pos > keep_upto) {
            s.ckpts.push_back(live_.ckpts[i]);
            live_.ckpts.erase(live_.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
    if (s.ckpts.empty()) return;
    s.tokens = live_.tokens;
    for (const ImgKey& image : live_.imgs)
        if (image.start < L) s.imgs.push_back(image);
    s.kv_cells = L;
    s.kv_bytes = bytes;
    parked_.park(std::move(s));
}

bool PromptCache::restore(Seq& seq, int64_t pos, bool prefix) {
    Ckpt* ck = seq.find(pos);
    if (ck == nullptr) return false;
    const int64_t cells = prefix ? pos : seq.kv_cells;
    if (seq.kv && ensure_kv && !ensure_kv(cells)) return false;
    if (seq.kv && !kv_copy(cells, seq.kv_cells, seq.kv.get(), false)) return false;
    ck->stamp = parked_.tick();
    return session_ckpt_load(ss_, g_, pos, ck->host, stream_);
}

int64_t PromptCache::begin(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs) {
    const int64_t L = (int64_t) live_.tokens.size();
    auto zero = [&]() -> int64_t {
        release(live_, 0);
        live_.tokens.clear();
        live_.imgs.clear();
        session_zero(ss_, g_, nullptr, stream_);
        cudaStreamSynchronize((cudaStream_t) stream_);
        return 0;
    };
    if (!enabled()) return zero();
    int64_t best = reusable(live_, ids, imgs, true);
    int from = -1;   // -1: the live sequence, else a parked one
    for (size_t i = 0; i < parked_.size(); ++i) {
        const int64_t r = reusable(parked_.entries()[i], ids, imgs, false);
        if (r > best) {
            best = r;
            from = (int) i;
        }
    }
    if (best <= 0) {
        stash_live(0);
        const int64_t result = zero();
        live_.imgs = imgs;
        return result;
    }
    if (from < 0) {
        // The live sequence continues.  When the prompt drops most of it (another conversation that shares a
        // prefix), the rest goes to the park; an edit near the end just overwrites it.
        const int64_t dropped = L - best;
        if (dropped >= 1024 && dropped > best) {
            live_.pin = best;
            stash_live(best);
            live_.pin = -1;
        }
        if (best < L && !restore(live_, best)) return zero();
        release(live_, best);
        live_.tokens.resize((size_t) best);
        live_.imgs = imgs;
        return best;
    }
    int64_t newest = 0;
    for (const Ckpt& c : parked_.entries()[(size_t) from].ckpts) newest = std::max(newest, c.pos);
    if (best < newest) {
        // Another conversation that shares the parked one's start: it borrows the cells before `best`, and the
        // parked conversation stays whole (pinned while the live one is parked beside it).
        parked_.entries()[(size_t) from].pin = best;
        stash_live(0);
        release(live_, 0);
        live_.tokens.clear();
        auto donor = std::find_if(parked_.entries().begin(), parked_.entries().end(),
                                  [](const Seq& s) { return s.pin >= 0; });
        if (donor == parked_.entries().end()) return zero();
        const bool ok = restore(*donor, best, true);
        donor->pin = -1;
        if (!ok) return zero();
        donor->stamp = parked_.tick();
        live_.tokens.assign(donor->tokens.begin(), donor->tokens.begin() + best);
        live_.imgs = imgs;
        return best;
    }
    // That conversation continues: the live one is parked and the chosen one becomes live.
    Seq chosen = parked_.take((size_t) from);
    stash_live(0);
    release(live_, 0);
    live_.tokens.clear();
    if (!restore(chosen, best)) {
        release(chosen, 0);
        return zero();
    }
    live_.tokens.assign(chosen.tokens.begin(), chosen.tokens.begin() + best);
    live_.ckpts = std::move(chosen.ckpts);
    release(live_, best);
    live_.imgs = imgs;
    return best;
}

void PromptCache::set(int64_t pos, const int64_t* tokens, int64_t n) {
    if (!enabled()) return;
    release(live_, pos);
    live_.tokens.resize((size_t) pos);
    live_.tokens.insert(live_.tokens.end(), tokens, tokens + n);
}

void PromptCache::set(int64_t pos, const int32_t* tokens, int64_t n) {
    if (!enabled()) return;
    release(live_, pos);
    live_.tokens.resize((size_t) pos);
    for (int64_t i = 0; i < n; ++i) live_.tokens.push_back(tokens[i]);
}

void PromptCache::checkpoint(int64_t pos, bool tail) {
    if (!enabled() || pos <= 0 || pos != (int64_t) live_.tokens.size()) return;
    if (Ckpt* ck = live_.find(pos)) {   // a new checkpoint landing on an existing length: mount stamp advances
        ck->stamp = parked_.tick();
        return;
    }
    void* h = take_buffer();
    if (h == nullptr) return;
    cudaDeviceSynchronize();
    const bool split = split_from > 0 && split_save;
    if (!(split ? session_ckpt_save_layers(ss_, g_, pos, h, stream_, 0, split_from, 0)
                : session_ckpt_save(ss_, g_, pos, h, stream_))) {
        give_back(h);
        return;
    }
    if (split) split_save(pos, h);
    live_.ckpts.push_back({pos, h, parked_.tick(), tail});
    live_.sort_ckpts();
    enforce_cap(live_);
}

void PromptCache::clear() {
    release(live_, -1);
    live_.tokens.clear();
    live_.imgs.clear();
    for (Seq& s : parked_.drain()) free_ckpts(s);
}

PromptCache::State PromptCache::state() const {
    State s;
    s.live_cells = (int64_t) live_.tokens.size();
    s.checkpoints = live_.ckpts.size();
    s.parked = parked_.size();
    s.parked_bytes = parked_.bytes();
    s.parked_budget = parked_.budget();
    s.slots = parked_.slots();
    s.evictions = parked_.evictions();
    return s;
}

std::string PromptCache::summary() const {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "live %lld cells; %zu checkpoints (%.0f MiB); parked %zu conversations (%.0f MiB, %zu evictions)",
                  (long long) live_.tokens.size(), live_.ckpts.size(),
                  (double) live_.ckpts.size() * ckpt_bytes_ / 1048576.0, parked_.size(),
                  (double) parked_.bytes() / 1048576.0, parked_.evictions());
    return buf;
}

}  // namespace lamina::core
