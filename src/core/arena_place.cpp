// src/core/arena_place.cpp - --arena-place: the expert arena's blobs placed by the RAM's measured read rate.
//
// Where the pinned blocks land in physical memory is the OS's choice, made from its free lists at the start.  On RAM
// with a slower region (unpaired DIMMs: two 32 GB and two 48 GB modules leave 32 GB read from one channel, ~37 GB/s
// against ~58 with 8 cores) that choice varies with what ran before: 2-25 GiB of UD-Q4_K_XL's 72 GiB arena in
// back-to-back starts, whole layers' blocks at a time.  The CPU reads only the experts no GPU holds; the ones that stay
// on a GPU (the caller's: below the slots its prompt path borrows) take the slow places.  Blocks of the same blob size
// take each other's experts, so a layer whose whole block is slow keeps its CPU experts elsewhere.  The GPU tiers
// still move experts as a conversation goes (on fixed text 8-17% of the CPU's expert reads were from slow places
// afterwards, 18-30% without the moves): between requests the swaps follow them (replan, replace_some).
#include "lamina/core/expert_source.hpp"

#include "lamina/kernels/cpu/expert_layout.hpp"
#include "lamina/kernels/cpu/pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <thread>
#include <vector>

namespace lamina::core {

namespace {

using Clock = std::chrono::steady_clock;

std::atomic<uint64_t> g_sink{0};   // the reads' results, so that they happen

// One thread on each physical core (the caller's on the last), all running the same task on a spin barrier: the
// arena's places are timed one at a time with every core reading a share, as the CPU pool reads experts.
class Crew {
public:
    Crew() {
        for (const auto& core : lamina::kernels::cpu::physical_cores())
            if (!core.empty()) cpus_.push_back(core.front());
        if (cpus_.empty()) cpus_.push_back(-1);
        std::rotate(cpus_.begin(), cpus_.end() - 1, cpus_.end());
        n_ = (int) cpus_.size();
        prev_ = lamina::kernels::cpu::pin_current_thread(cpus_[0]);
        for (int t = 1; t < n_; ++t)
            th_.emplace_back([this, t] {
                (void) lamina::kernels::cpu::pin_current_thread(cpus_[(size_t) t]);
                uint64_t seen = 0;
                for (;;) {
                    const uint64_t g = go_.load(std::memory_order_acquire);
                    if (g == seen) continue;
                    seen = g;
                    if (quit_.load()) return;
                    (*task_)(t);
                    done_.fetch_add(1, std::memory_order_acq_rel);
                }
            });
    }
    ~Crew() {
        quit_.store(true);
        go_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& t : th_) t.join();
        lamina::kernels::cpu::restore_thread_affinity(prev_);
    }
    int size() const { return n_; }
    void run(const std::function<void(int)>& task) {
        task_ = &task;
        done_.store(0, std::memory_order_relaxed);
        go_.fetch_add(1, std::memory_order_acq_rel);
        task(0);
        while (done_.load(std::memory_order_acquire) < n_ - 1) {}
    }

private:
    std::vector<int> cpus_;
    int n_ = 1;
    lamina::kernels::cpu::ThreadAffinity prev_;
    std::vector<std::thread> th_;
    const std::function<void(int)>* task_ = nullptr;
    std::atomic<uint64_t> go_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> quit_{false};
};

void read_words(const uint8_t* p, uint64_t bytes) {
    const uint64_t* w = (const uint64_t*) p;
    const uint64_t n = bytes / 8;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 ^= w[i];
        a1 ^= w[i + 1];
        a2 ^= w[i + 2];
        a3 ^= w[i + 3];
    }
    for (; i < n; ++i) a0 ^= w[i];
    g_sink.fetch_xor(a0 ^ a1 ^ a2 ^ a3, std::memory_order_relaxed);
}

}  // namespace

int64_t ArenaExpertSource::place_block(size_t q) const {
    return (int64_t) (std::upper_bound(place_off_.begin(), place_off_.end(), (int64_t) q) - place_off_.begin()) - 1;
}

uint8_t* ArenaExpertSource::place_addr(size_t q) const {
    const int64_t b = place_block(q);
    return layer_base_[(size_t) b] +
           (uint64_t) ((int64_t) q - place_off_[(size_t) b]) * lamina::kernels::cpu::expert_layout().blob_bytes(b);
}

void ArenaExpertSource::note_swap(size_t q1, size_t q2) {
    const int64_t b1 = place_block(q1), b2 = place_block(q2);
    const int32_t i1 = place_occ_[q1], i2 = place_occ_[q2];
    idx_[(size_t) i1] = (int32_t) ((int64_t) q2 - place_off_[(size_t) b2]);
    blk_[(size_t) i1] = (int32_t) b2;
    idx_[(size_t) i2] = (int32_t) ((int64_t) q1 - place_off_[(size_t) b1]);
    blk_[(size_t) i2] = (int32_t) b1;
    place_occ_[q1] = i2;
    place_occ_[q2] = i1;
}

std::vector<std::pair<size_t, size_t>> ArenaExpertSource::plan_swaps(const std::vector<int32_t>& rank,
                                                                     const std::vector<uint8_t>& on_gpu) const {
    // in each class of blocks with equal blobs, the fast places for the experts that deserve them: the CPU's hottest,
    // its coldest, then the GPUs' coldest; the swaps move the others' out of the slow places
    const auto& lay = lamina::kernels::cpu::expert_layout();
    const int64_t L = (int64_t) layer_base_.size(), E = n_expert_;
    std::map<uint64_t, std::vector<int64_t>> classes;
    for (int64_t b = 0; b < L; ++b)
        if (place_off_[(size_t) b + 1] > place_off_[(size_t) b]) classes[lay.blob_bytes(b)].push_back(b);
    auto key = [&](int32_t i) {
        const int64_t r = rank[(size_t) i] >= 0 ? rank[(size_t) i] : L * E + i;
        return on_gpu[(size_t) i] != 0 ? 4 * L * E - r : r;
    };
    std::vector<std::pair<size_t, size_t>> swaps;   // (slow place, fast place)
    for (const auto& [bb, blocks] : classes) {
        std::vector<size_t> by_key;
        for (const int64_t b : blocks)
            for (int64_t q = place_off_[(size_t) b]; q < place_off_[(size_t) b + 1]; ++q) by_key.push_back((size_t) q);
        std::sort(by_key.begin(), by_key.end(),
                  [&](size_t a, size_t c) { return key(place_occ_[a]) < key(place_occ_[c]); });
        size_t nf = 0;
        for (const size_t q : by_key) nf += place_slow_[q] == 0;
        std::vector<size_t> out, in;   // fast experts in slow places, slow experts in fast places (as many)
        for (size_t k = 0; k < by_key.size(); ++k) {
            if (k < nf && place_slow_[by_key[k]]) out.push_back(by_key[k]);
            if (k >= nf && !place_slow_[by_key[k]]) in.push_back(by_key[k]);
        }
        for (size_t k = 0; k < out.size() && k < in.size(); ++k) swaps.emplace_back(out[k], in[k]);
    }
    // the CPU's hottest expert's first, over all classes (replace_some makes them in this order)
    std::stable_sort(swaps.begin(), swaps.end(), [&](const auto& a, const auto& b) {
        return key(place_occ_[a.first]) < key(place_occ_[b.first]);
    });
    return swaps;
}

bool ArenaExpertSource::place(const std::vector<int32_t>& rank, const std::vector<uint8_t>& on_gpu, std::string& line) {
    const auto& lay = lamina::kernels::cpu::expert_layout();
    const int64_t L = (int64_t) layer_base_.size(), E = n_expert_;
    if (L == 0 || rank.size() != (size_t) (L * E) || on_gpu.size() != (size_t) (L * E)) {
        line = "--arena-place: no arena, or a ranking for a different geometry";
        return true;
    }
    // LAMINA_ARENA_PLACE_DRY=1 (the A/B arm): times and reports, moves nothing, now or between requests
    const char* dry_env = std::getenv("LAMINA_ARENA_PLACE_DRY");
    place_dry_ = dry_env != nullptr && std::atoi(dry_env) != 0;
    if (idx_.empty()) {   // each blob at its expert index
        idx_.resize((size_t) (L * E));
        for (int64_t i = 0; i < L * E; ++i) idx_[(size_t) i] = (int32_t) (i % E);
    }
    if (blk_.empty()) {
        blk_.resize((size_t) (L * E));
        for (int64_t i = 0; i < L * E; ++i) blk_[(size_t) i] = (int32_t) (i / E);
    }
    // the places: the mapped blocks' blobs, each block's first and the (layer, expert) in each
    place_off_.assign((size_t) L + 1, 0);
    for (int64_t i = 0; i < L * E; ++i)
        if (idx_[(size_t) i] >= 0 && layer_mapped(blk_[(size_t) i])) ++place_off_[(size_t) blk_[(size_t) i] + 1];
    for (int64_t b = 0; b < L; ++b) place_off_[(size_t) b + 1] += place_off_[(size_t) b];
    const size_t n = (size_t) place_off_[(size_t) L];
    if (n == 0) {
        place_off_.clear();
        line = "--arena-place: no pinned, mapped blocks";
        return true;
    }
    place_occ_.assign(n, -1);
    for (int64_t i = 0; i < L * E; ++i)
        if (idx_[(size_t) i] >= 0 && layer_mapped(blk_[(size_t) i]))
            place_occ_[(size_t) (place_off_[(size_t) blk_[(size_t) i]] + idx_[(size_t) i])] = (int32_t) i;

    Crew crew;
    const int T = crew.size();
    // ---- each place's read rate: every place once, then the slow ones again (the best of two)
    const auto t0 = Clock::now();
    std::vector<float> gbs(n, 0.0f);
    const uint8_t* p = nullptr;
    uint64_t bytes = 0;
    const std::function<void(int)> read = [&](int t) {
        const uint64_t per = bytes / (uint64_t) T / 64 * 64, from = per * (uint64_t) t;
        read_words(p + from, t == T - 1 ? bytes - from : per);
    };
    auto time_place = [&](size_t q) {
        p = place_addr(q);
        bytes = lay.blob_bytes(place_block(q));
        const auto s = Clock::now();
        crew.run(read);
        const double sec = std::chrono::duration<double>(Clock::now() - s).count();
        gbs[q] = std::max(gbs[q], (float) ((double) bytes / sec / 1e9));
    };
    for (size_t q = 0; q < n; ++q) time_place(q);
    std::vector<float> sorted = gbs;
    std::sort(sorted.begin(), sorted.end());
    const float fast = sorted[n * 9 / 10], cut = 0.85f * fast;
    for (size_t q = 0; q < n; ++q)
        if (gbs[q] < cut) time_place(q);
    place_slow_.assign(n, 0);
    std::vector<float> slow_gbs;
    uint64_t slow_bytes = 0;
    for (size_t q = 0; q < n; ++q) {
        place_slow_[q] = gbs[q] < cut;
        if (place_slow_[q]) {
            slow_gbs.push_back(gbs[q]);
            slow_bytes += lay.blob_bytes(place_block(q));
        }
    }
    std::sort(slow_gbs.begin(), slow_gbs.end());
    const double t_time = std::chrono::duration<double>(Clock::now() - t0).count();
    auto cpu_in_slow = [&] {
        int64_t k = 0;
        for (size_t q = 0; q < n; ++q) k += place_slow_[q] && on_gpu[(size_t) place_occ_[q]] == 0;
        return k;
    };
    int64_t cpu_n = 0;
    for (size_t q = 0; q < n; ++q) cpu_n += on_gpu[(size_t) place_occ_[q]] == 0;
    const int64_t cpu_slow = cpu_in_slow();
    const std::vector<std::pair<size_t, size_t>> swaps = plan_swaps(rank, on_gpu);

    // ---- the swaps, through a buffer per thread
    const auto t1 = Clock::now();
    uint64_t moved = 0;
    if (!place_dry_) {
        std::atomic<size_t> next{0};
        std::vector<std::vector<uint8_t>> tmp((size_t) T, std::vector<uint8_t>(lay.max_blob));
        crew.run([&](int t) {
            for (size_t k; (k = next++) < swaps.size();) {
                uint8_t* p1 = place_addr(swaps[k].first);
                uint8_t* p2 = place_addr(swaps[k].second);
                const size_t bb = (size_t) lay.blob_bytes(place_block(swaps[k].first));
                std::memcpy(tmp[(size_t) t].data(), p1, bb);
                std::memcpy(p1, p2, bb);
                std::memcpy(p2, tmp[(size_t) t].data(), bb);
            }
        });
        for (const auto& [q1, q2] : swaps) {
            note_swap(q1, q2);
            moved += 2 * lay.blob_bytes(place_block(q1));
        }
    }
    char buf[512];
    int len = std::snprintf(buf, sizeof buf, "--arena-place: %zu places timed in %.1f s on %d cores, the fast ones ~%.1f "
                                             "GB/s (the median %.1f, the 10th percentile %.1f); %zu slow", n, t_time, T,
                            (double) fast, (double) sorted[n / 2], (double) sorted[n / 10], slow_gbs.size());
    if (!slow_gbs.empty())
        len += std::snprintf(buf + len, sizeof buf - (size_t) len, " (%.2f GiB, ~%.1f GB/s)",
                             (double) slow_bytes / 1073741824.0, (double) slow_gbs[slow_gbs.size() / 2]);
    std::snprintf(buf + len, sizeof buf - (size_t) len,
                  "; of the %lld experts on no GPU, %lld were in them, %lld are; %zu pairs %s (%.2f GiB in %.1f s)",
                  (long long) cpu_n, (long long) cpu_slow, (long long) cpu_in_slow(), swaps.size(),
                  place_dry_ ? "to swap (LAMINA_ARENA_PLACE_DRY: none moved)" : "swapped",
                  (double) moved / 1073741824.0, std::chrono::duration<double>(Clock::now() - t1).count());
    line = buf;
    // LAMINA_LOAD_VERIFY=1: every moved blob read again from the model and compared (a test)
    if (const char* v = std::getenv("LAMINA_LOAD_VERIFY"); v != nullptr && std::atoi(v) != 0 && !place_dry_) {
        std::vector<uint8_t> want(lay.max_blob);
        std::string err;
        int64_t checked = 0;
        for (const auto& [q1, q2] : swaps)
            for (const int32_t i : {place_occ_[q1], place_occ_[q2]}) {
                const int64_t l = i / E, e = i % E;
                if (!read_file_blob(l, e, want.data(), err)) {
                    line += "; LAMINA_LOAD_VERIFY: " + err;
                    return true;
                }
                if (std::memcmp(want.data(), blob(l, e), (size_t) lay.blob_bytes(l)) != 0) {
                    line += "; LAMINA_LOAD_VERIFY: layer " + std::to_string(l) + " expert " + std::to_string(e) +
                            " differs from the model after the move";
                    return false;
                }
                ++checked;
            }
        line += "; LAMINA_LOAD_VERIFY: " + std::to_string(checked) + " moved blobs equal to the model's";
    }
    return true;
}

size_t ArenaExpertSource::replan(const std::vector<int32_t>& rank, const std::vector<uint8_t>& on_gpu) {
    pending_.clear();
    pending_at_ = 0;
    if (place_dry_ || place_occ_.empty() || rank.size() != idx_.size() || on_gpu.size() != idx_.size()) return 0;
    pending_ = plan_swaps(rank, on_gpu);
    return pending_.size();
}

size_t ArenaExpertSource::replace_some(size_t max) {
    const auto& lay = lamina::kernels::cpu::expert_layout();
    if (swap_buf_.size() < lay.max_blob) swap_buf_.resize(lay.max_blob);
    // LAMINA_LOAD_VERIFY=1: both moved blobs read again from the model and compared (a test)
    static const bool verify = [] {
        const char* v = std::getenv("LAMINA_LOAD_VERIFY");
        return v != nullptr && std::atoi(v) != 0;
    }();
    for (size_t k = 0; k < max && pending_at_ < pending_.size(); ++k) {
        const auto [q1, q2] = pending_[pending_at_++];
        uint8_t* p1 = place_addr(q1);
        uint8_t* p2 = place_addr(q2);
        const size_t bb = (size_t) lay.blob_bytes(place_block(q1));
        std::memcpy(swap_buf_.data(), p1, bb);
        std::memcpy(p1, p2, bb);
        std::memcpy(p2, swap_buf_.data(), bb);
        note_swap(q1, q2);
        for (const int32_t i : {place_occ_[q1], place_occ_[q2]}) {
            if (!verify) break;
            const int64_t l = i / n_expert_, e = i % n_expert_;
            std::string err;
            if (!read_file_blob(l, e, swap_buf_.data(), err) || std::memcmp(swap_buf_.data(), blob(l, e), bb) != 0) {
                std::fprintf(stderr, "ArenaExpertSource: LAMINA_LOAD_VERIFY: layer %lld expert %lld differs from the "
                                     "model after a swap between requests %s\n", (long long) l, (long long) e, err.c_str());
                std::abort();
            }
        }
    }
    return pending_.size() - pending_at_;
}

}  // namespace lamina::core
