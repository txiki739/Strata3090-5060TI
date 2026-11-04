// src/core/adaptive_tier.cpp - see include/lamina/core/adaptive_tier.hpp.
#include "lamina/core/adaptive_tier.hpp"

#include "lamina/core/adaptive_policy.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace lamina::core {
AdaptiveTier::~AdaptiveTier() {
    DeviceScope on(dev_, main_, dev_ >= 0);
    if (stream_) cudaStreamSynchronize(stream_);
    for (cudaEvent_t e : evs_) if (e) cudaEventDestroy(e);
    cudaEvent_t evs[] = {t0_, t1_};
    for (cudaEvent_t e : evs) if (e) cudaEventDestroy(e);
    if (stream_) cudaStreamDestroy(stream_);
    if (stage_arena_ != nullptr) cudaFreeHost(stage_arena_);
}

bool AdaptiveTier::init(ExpertCache& cache, ExpertSource& src,
                        std::vector<int32_t>& host_res, int64_t n_layers, int64_t n_expert,
                        int max_moves, std::string& err, int device, int main_device,
                        bool deterministic_adapt) {
    cache_ = &cache;
    src_ = &src;
    res_ = &host_res;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    max_moves_ = max_moves;
    dev_ = device;
    main_ = main_device;
    deterministic_ = deterministic_adapt;
    free_.assign((size_t) n_layers, {});
    DeviceScope on(dev_, main_, dev_ >= 0);
    bool ok = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreate(&t0_) == cudaSuccess && cudaEventCreate(&t1_) == cudaSuccess;
    for (cudaEvent_t& e : evs_) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    batch_stage_.resize(kBatches);
    for (int64_t l = 0; l < n_layers && !guard_; ++l) guard_ = !src.pinned(l, 0);
    if (!guard_) {
    } else if (cudaHostAlloc((void**) &stage_arena_, kStageBlocks * kStageBlock, cudaHostAllocDefault) == cudaSuccess) {
        for (size_t i = 0; i < kStageBlocks; ++i) stage_free_.push_back(i);
    } else {
        cudaGetLastError();   // fall back to copying pageable sources directly, as before
        stage_arena_ = nullptr;
        std::fprintf(stderr, "lamina: adaptive tier staging arena unavailable, copying pageable sources directly\n");
        std::fflush(stderr);
    }
    if (!ok) {
        res_ = nullptr;   // off
        err = "adaptive tier: cannot create the refill stream";
    }
    return ok;
}

int64_t AdaptiveTier::free_slots() const {
    int64_t n = 0;
    for (const auto& f : free_) n += (int64_t) f.size();
    return n;
}

uint64_t AdaptiveTier::bytes_of(const Move& m) const {
    return lamina::kernels::cpu::expert_layout().blob_bytes(m.layer);
}

void AdaptiveTier::evict(const Move& m) {
    if (m.out >= 0) (*res_)[(size_t) (m.layer * n_expert_ + m.out)] = kNotResident;   // a miss from now on
}

bool AdaptiveTier::copy(const Move& m, uint64_t off, uint64_t n, std::string& err) {
    const uint8_t* b = src_->blob(m.layer, m.in);
    if (b == nullptr) {
        err = "adaptive tier: a refill copy failed";
        return false;
    }
    const void* src = b + off;
    if (stage_arena_ != nullptr && n <= kStageBlock && !src_->pinned(m.layer, m.in)) {
        if (stage_free_.empty()) {   // cannot stage now: the pump stops, blocks return as batches retire
            stage_blocked_ = true;
            return true;
        }
        const size_t blk = stage_free_.back();
        stage_free_.pop_back();
        uint8_t* s = stage_arena_ + blk * kStageBlock;
        std::memcpy(s, b + off, (size_t) n);   // the slice goes whole into the block's start
        stage_used_.push_back(blk);
        ++staged_now_;
        src = s;
    }
    cp_dst_.push_back(cache_->device_slot(m.slot) + off);
    cp_src_.push_back(src);
    cp_bytes_.push_back((size_t) n);
    sent_bytes += n;
    return true;
}

bool AdaptiveTier::send(std::string& err) {
    const bool ok = copy_blobs(cp_dst_.data(), cp_src_.data(), cp_bytes_.data(), cp_dst_.size(), stream_);
    cp_dst_.clear();
    cp_src_.clear();
    cp_bytes_.clear();
    if (!ok) err = "adaptive tier: a refill copy failed";
    return ok;
}

// Takes the last timed copies' rate once they have landed; then starts timing these when `worth` (long enough for
// their start and end not to dominate) and none are being timed.
bool AdaptiveTier::time_begin(bool worth) {
    float took = 0;
    if (timed_ > 0 && cudaEventQuery(t1_) == cudaSuccess && cudaEventElapsedTime(&took, t0_, t1_) == cudaSuccess &&
        took > 0) {
        const double r = (double) timed_ / (double) took;
        rate_ = rate_ > 0 ? 0.8 * rate_ + 0.2 * r : r;
        timed_ = 0;
    }
    return worth && timed_ == 0 && cudaEventRecord(t0_, stream_) == cudaSuccess;
}

// After a batch of copies: its timing, and its event in the ring, with the moves sent so far (a full ring's newest
// event takes this batch too).
bool AdaptiveTier::end_batch(bool timed, uint64_t bytes, std::string& err) {
    if (timed) {
        if (cudaEventRecord(t1_, stream_) != cudaSuccess) { err = "adaptive tier: a refill copy failed"; return false; }
        timed_ = bytes;
    }
    // the pump-side kMaxFlying cap makes this ring-full overwrite path unreachable
    if (flying_ < kBatches) ++flying_;
    const int i = (first_ + flying_ - 1) % kBatches;
    batch_sent_[i] = sent_;
    batch_stage_[i] = std::move(stage_used_);   // the batch owns its staging blocks until it retires
    stage_used_.clear();
    staged_now_ = 0;
    if (cudaEventRecord(evs_[i], stream_) != cudaSuccess) { err = "adaptive tier: a refill copy failed"; return false; }
    return true;
}

void AdaptiveTier::stage(uint64_t bytes) {
    uint64_t have = 0;
    for (size_t i = sent_; i < staged_; ++i) have += bytes_of(queued_[i]);
    have -= off_;
    for (; staged_ < queued_.size() && have < bytes; ++staged_) {
        evict(queued_[staged_]);
        have += bytes_of(queued_[staged_]);
    }
}

bool AdaptiveTier::pump_bytes(uint64_t bytes, std::string& err) {
    if (guard_) {   // never block the host thread that launches the second GPU's graphs
        retire();
        if (flying_ >= (int) kMaxFlying) return true;
    }
    if (sent_ >= staged_ || bytes < (64u << 10)) return true;
    DeviceScope on(dev_, main_, dev_ >= 0);
    if (after_ != nullptr && cudaStreamWaitEvent(stream_, after_, 0) != cudaSuccess) {   // set_after
        err = "adaptive tier: a refill copy failed";
        return false;
    }
    const bool timed = time_begin(bytes >= (256u << 10));
    uint64_t done = 0;
    while (sent_ < staged_ && done < bytes) {
        stage_blocked_ = false;
        const Move& m = queued_[sent_];
        const uint64_t blob = bytes_of(m);
        uint64_t n = std::min(bytes - done, blob - off_);
        if (off_ + n < blob) n &= ~(uint64_t) 4095;   // whole pages, but the blob's end
        if (n == 0) break;
        if (!copy(m, off_, n, err)) return false;
        if (stage_blocked_) break;   // nothing staged for this slice: break BEFORE advancing done/off_/sent_
        done += n;
        off_ += n;
        if (off_ == blob) {
            off_ = 0;
            ++sent_;
        }
    }
    if (done == 0) return true;
    if (!send(err)) return false;
    return end_batch(timed, done, err);
}

bool AdaptiveTier::pump(uint64_t budget, uint64_t& sent, std::string& err) {
    sent = 0;
    if (guard_) {   // never block the host thread that launches the second GPU's graphs
        retire();
        if (flying_ >= (int) kMaxFlying) return true;
    }
    size_t n = 0;
    for (; sent_ + n < queued_.size(); ++n) {
        const uint64_t b = bytes_of(queued_[sent_ + n]);
        if (sent + b > budget) break;
        sent += b;
    }
    if (n == 0) return true;
    DeviceScope on(dev_, main_, dev_ >= 0);
    // timed from when the work that may read its slots is done
    if (after_ != nullptr && cudaStreamWaitEvent(stream_, after_, 0) != cudaSuccess) {
        err = "adaptive tier: a refill copy failed";
        return false;
    }
    const bool timed = time_begin(n >= 4);   // a batch of a few copies: its start and end dominate
    // the copies are only submitted by send() below, so evicting after queueing one keeps the resident out of the
    // table before its slot is written; a blocked copy breaks BEFORE evict/sent_, leaving the move queued
    uint64_t sub = 0;
    for (size_t i = 0; i < n; ++i) {
        stage_blocked_ = false;
        const uint64_t mb = bytes_of(queued_[sent_]);
        if (!copy(queued_[sent_], 0, mb, err)) return false;
        if (stage_blocked_) break;
        evict(queued_[sent_]);
        ++sent_;
        sub += mb;
    }
    sent = sub;
    if (sub == 0) return true;
    staged_ = sent_;
    return send(err) && end_batch(timed, sent, err);
}

bool AdaptiveTier::refill(const std::vector<std::array<int32_t, 3>>& moves, uint64_t batch, std::string& err) {
    if (moves.empty()) return true;
    for (const auto& m : moves) {
        queued_.push_back({m[0], m[1], -1, m[2]});
        pending_.emplace_back(m[0] * (int32_t) n_expert_ + m[1], m[2]);
        ++fills;
    }
    if (paced_) {
        for (uint64_t sent = 1; sent > 0 && sent_ < queued_.size();)
            if (!pump(batch, sent, err)) return false;
    } else {
        stage(UINT64_MAX);
        while (sent_ < staged_) {
            const size_t s0 = sent_;
            const uint64_t o0 = off_;
            if (!pump_bytes(batch, err)) return false;
            if (sent_ == s0 && off_ == o0) break;   // a guarded pump at its flight cap: the windows' pumps go on
        }
    }
    return true;
}

bool AdaptiveTier::adapt(std::vector<float>& usage, std::string& err, bool decay) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!failed_.empty()) { err = failed_; return false; }
    apply_pending(deterministic_);
    if (!failed_.empty()) { err = failed_; return false; }
    if (wait_ && admitted_ < queued_.size()) return true;
    // the moves whose residents are still in place give way to this call's (ranked again if still worth it)
    for (size_t i = staged_; i < queued_.size(); ++i) {
        const Move& m = queued_[i];
        if (m.out >= 0) {
            --swaps;
        } else {
            free_[(size_t) m.layer].push_back(m.slot);
            --fills;
        }
    }
    dropped += (int64_t) (queued_.size() - staged_);
    queued_.resize(staged_);
    pending_.resize(staged_);
    queued_.erase(queued_.begin(), queued_.begin() + (ptrdiff_t) admitted_);
    pending_.erase(pending_.begin(), pending_.begin() + (ptrdiff_t) admitted_);
    for (int b = 0; b < flying_; ++b) batch_sent_[(first_ + b) % kBatches] -= admitted_;
    landed_ -= admitted_;   // apply_pending above admitted every landed move, so landed_ == admitted_ here
    sent_ -= admitted_;
    staged_ -= admitted_;
    admitted_ = 0;
    blocked_.assign((size_t) (n_layers_ * n_expert_), 0);   // the experts still under way
    for (const auto& pr : pending_) blocked_[(size_t) pr.first] = 1;
    if (!uppers_.empty()) {
        upper_has_.assign((size_t) (n_layers_ * n_expert_), 0);
        for (const AdaptiveTier* up : uppers_) {
            for (size_t i = 0; i < upper_has_.size(); ++i) upper_has_[i] |= (*up->res_)[i] >= 0 ? 1 : 0;
            for (size_t i = 0; i < up->staged_; ++i) upper_has_[(size_t) up->pending_[i].first] = 1;
        }
    }
    if (!lowers_.empty()) {
        lower_has_.assign((size_t) (n_layers_ * n_expert_), 0);
        for (const AdaptiveTier* lo : lowers_)
            for (size_t i = 0; i < lower_has_.size(); ++i) lower_has_[i] |= (*lo->res_)[i] >= 0 ? 1 : 0;
    }
    using Ranked = AdaptiveMoveOrder;   // out < 0: an empty slot
    std::vector<Ranked> moves;
    std::vector<std::pair<float, int32_t>> cand, vict;
    std::vector<int32_t>& res = *res_;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + l * n_expert_;
        const int32_t* r = res.data() + l * n_expert_;
        const uint8_t* bl = blocked_.data() + l * n_expert_;
        const uint8_t* up = !uppers_.empty() ? upper_has_.data() + l * n_expert_ : nullptr;
        const uint8_t* lo = !lowers_.empty() ? lower_has_.data() + l * n_expert_ : nullptr;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            const float ue = lo != nullptr && lo[e] ? 0.5f * u[e] : u[e];
            if (r[e] < 0) {
                if (ue >= 2.0f && !bl[e] && (up == nullptr || !up[e]) && src_->in_ram(l, e))
                    cand.emplace_back(ue, e);
            } else if (r[e] >= frozen_) {
                vict.emplace_back(up != nullptr && up[e] ? -1.0f : u[e], e);   // the upper tier's copy goes first
            }
        }
        if (cand.empty()) continue;
        std::sort(cand.begin(), cand.end(), [this](auto& a, auto& b) {
            if (a.first > b.first) return true;
            if (b.first > a.first) return false;
            return deterministic_ && a.second < b.second;
        });
        const std::vector<int32_t>& fr = free_[(size_t) l];
        size_t c = 0;
        for (; c < cand.size() && c < fr.size(); ++c)
            moves.push_back({cand[c].first, (int32_t) l, cand[c].second, -1, fr[fr.size() - 1 - c]});
        const size_t nv = std::min(cand.size() - c, vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nv, vict.end(),
                          [this](auto& a, auto& b) {
                              if (a.first < b.first) return true;
                              if (b.first < a.first) return false;
                              return deterministic_ && a.second < b.second;
                          });
        for (size_t i = 0; i < nv; ++i, ++c) {
            if (cand[c].first < vict[i].first + 1.5f) break;
            moves.push_back({cand[c].first - vict[i].first, (int32_t) l, cand[c].second, vict[i].second,
                             r[vict[i].second]});
        }
    }
    std::sort(moves.begin(), moves.end(), [this](const Ranked& a, const Ranked& b) {
        return adaptive_move_less(a, b, deterministic_);
    });
    if ((int) moves.size() > max_moves_) moves.resize((size_t) max_moves_);
    // a move counts once queued; its resident leaves when it is staged (or sent, paced), its expert comes once landed
    for (const Ranked& m : moves) {
        if (m.out >= 0) {
            ++swaps;
        } else {
            std::vector<int32_t>& fr = free_[(size_t) m.layer];
            fr.erase(std::find(fr.begin(), fr.end(), m.slot));
            ++fills;
        }
        queued_.push_back({m.layer, m.in, m.out, m.slot});
        pending_.emplace_back((int32_t) (m.layer * n_expert_ + m.in), m.slot);
    }
    if (decay)
        for (float& v : usage) v *= decay_;
    if (deterministic_) {
        std::string submit_err;
        bool submitted = false;
        if (!paced_) {
            stage(UINT64_MAX);
            submitted = pump_bytes(UINT64_MAX, submit_err);
        } else {
            uint64_t sent = 0;
            submitted = pump(UINT64_MAX, sent, submit_err);
        }
        if (!submitted) {
            failed_ = submit_err;
            err = failed_;
            return false;
        }
    }
    ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// Retires the batches whose events have landed: advances the ring and raises `landed_` to the highest moves-sent index
// among them.  **IT DOES NOT RETURN IT**: the pumps retire batches too (for their staging blocks and the kMaxFlying cap),
// and when they discarded the index a batch landing just before a pump was never admitted.  Its 96 evicted residents
// stayed empty and, with an update waiting for the previous one's (burst), the tier never ranked again: a session past
// that point decoded on a frozen cache (edit 69 tok/s at 35% hits instead of 136 at 79%, Ryzen 9 5950X + RTX 3090, 12 pool workers).
void AdaptiveTier::retire() {
    for (; flying_ > 0 && cudaEventQuery(evs_[first_]) == cudaSuccess; first_ = (first_ + 1) % kBatches, --flying_) {
        for (const size_t blk : batch_stage_[first_]) stage_free_.push_back(blk);   // its copies have read the blocks
        batch_stage_[first_].clear();
        landed_ = std::max(landed_, batch_sent_[first_]);
    }
}

void AdaptiveTier::apply_pending(bool wait) {
    if (deterministic_ && !wait) return;
    if (wait && admitted_ < queued_.size()) {   // send the rest and wait for it
        std::string err;
        uint64_t sent = 0;
        if (!paced_) stage(UINT64_MAX);
        // a guarded pump stops at the flight cap or with the staging ring empty: then the stream is waited for, which
        // retires its batches and frees their blocks, until every move is sent (none may land in a slot lent later)
        DeviceScope on(dev_, main_, dev_ >= 0);
        for (;;) {
            const size_t before = sent_;
            if (!(paced_ ? pump(UINT64_MAX, sent, err) : pump_bytes(UINT64_MAX, err))) {
                failed_ = err;
                return;
            }
            if (sent_ >= queued_.size()) break;
            if (sent_ == before) {
                cudaStreamSynchronize(stream_);
                retire();
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            failed_ = "adaptive tier: refill synchronization failed";
            return;
        }
    }
    retire();   // the batches landed, oldest first (or already retired by a pump: landed_ remembers them)
    if (landed_ <= admitted_) return;
    std::vector<int32_t>& res = *res_;
    for (; admitted_ < landed_; ++admitted_) res[(size_t) pending_[admitted_].first] = pending_[admitted_].second;
}

}  // namespace lamina::core
