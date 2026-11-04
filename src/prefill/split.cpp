// src/prefill/split.cpp - see include/lamina/prefill/split.hpp.
#include "lamina/prefill/split.hpp"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/core/cuda_host.hpp"
#include "lamina/core/geometry.hpp"
#include "lamina/core/layer.hpp"
#include "lamina/core/layout.hpp"
#include "lamina/core/native_head.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace lamina::prefill {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t SUB = 2048;   // the prompt path's sub-chunk (prefill.cpp): the residual goes across in these
uint64_t up(uint64_t n) { return (n + 255) & ~255ull; }
/// NativeDense's zeroed tail after each native matrix: the prompt path's MMQ reads past the last row of one whose rows
/// are not whole 256-value steps (the shared expert's down projection, 640)
constexpr uint64_t kNativeTail = 4096;

/// A copy of `bytes` from `src` to `dst`, each a device pointer on its own GPU.
struct Run {
    const uint8_t* src;
    uint8_t* dst;
    uint64_t bytes;
};

/// The runs that copy cells [c0, c1) of one K/V array, laid out [page][kv_head][page_size][row] in both states, from
/// one whose physical pages start at logical page `src_first` to one whose start at `dst_first`: whole pages in one
/// run, a page's part one run a head.
void cell_runs(int64_t c0, int64_t c1, int64_t ps, int64_t nkv, uint64_t rb, const void* src, int64_t src_first,
               void* dst, int64_t dst_first, std::vector<Run>& out) {
    const uint64_t page = (uint64_t) nkv * (uint64_t) ps * rb;
    auto at = [&](const void* base, int64_t first, int64_t p, uint64_t in) {
        return (uint8_t*) base + (uint64_t) (p - first) * page + in;
    };
    for (int64_t c = c0; c < c1;) {
        const int64_t p = c / ps;
        if (c == p * ps && c1 >= (p + 1) * ps) {
            const int64_t q = c1 / ps;   // pages [p, q) whole
            out.push_back({at(src, src_first, p, 0), at(dst, dst_first, p, 0), (uint64_t) (q - p) * page});
            c = q * ps;
            continue;
        }
        const int64_t e = std::min(c1, (p + 1) * ps);
        for (int64_t h = 0; h < nkv; ++h) {
            const uint64_t in = ((uint64_t) h * (uint64_t) ps + (uint64_t) (c - p * ps)) * rb;
            out.push_back({at(src, src_first, p, in), at(dst, dst_first, p, in), (uint64_t) (e - c) * rb});
        }
        c = e;
    }
}

/// A tensor of the second stage's layers: the main GPU's row, and its buffers' bytes and offsets in the copies.
struct Tensor {
    std::string name;
    core::WeightRef ref;
    uint64_t off[4] = {}, bytes[4] = {};   // data, native_data, hc_codes, hc_scales
};

struct Job {
    bool chunk = false;   // false: the session state at the prompt's start, and the weights, in
    const int64_t* tokens = nullptr;
    int64_t T = 0, p0 = 0;
    bool run_end = false;
    std::vector<cudaEvent_t> landed;
};

}  // namespace

struct PromptSplit::Impl {
    PromptSplit* self = nullptr;
    Prefill* p1 = nullptr;
    Prefill p2;
    const core::WeightTable* wt = nullptr;
    const core::ModelGeometry* g = nullptr;
    core::SessionState* ss = nullptr;
    core::MtpDrafter* mtp = nullptr;
    int64_t at = 0, max_chunk = 0;
    int dev = -1, main = 0;
    lamina::kernels::QsaShapes s;
    // the dense weights of the layers from `at` on and the draft layer's prompt-K/V weights: the main GPU's rows, the
    // copies' layout, and the second GPU's table of its copies
    std::vector<Tensor> tensors;
    uint64_t dkv_off[10] = {}, dkv_bytes[10] = {};   // MtpPromptKv's weights, in its field order
    uint64_t wbytes = 0;
    struct WSrc { const void* dev; uint64_t off, bytes; };
    std::vector<WSrc> wsrc;   // the main GPU's buffers and their place in the copies
    core::WeightTable wt2;
    // the second stage's session state, indexed from `at`: its QSA layers' copies, its GDN layers' (the main session's
    // GDN states [gdn_first, gdn_first + gdn_count)), and the draft layer's cells of one chunk at a time
    core::SessionState ss2;
    std::vector<core::QsaState> qsa2;
    std::vector<int64_t> qsa_main;   // their index among the main session's QSA states
    int64_t gdn_first = 0, gdn_count = 0;
    core::QsaState draft2;
    core::MtpPromptKv dkv2;
    uint8_t *w_base = nullptr, *draft_base = nullptr;
    int64_t pos0 = 0, first_cell = 0;
    // the second GPU's stage stream and copy stream; the main GPU's stream for the copies to and from its session
    cudaStream_t s2 = nullptr, cp2 = nullptr, back = nullptr;
    std::vector<cudaEvent_t> ev_in;    // the second GPU: a piece of the residual is in the stage's buffers
    cudaEvent_t ev_free = nullptr;     // the second GPU: the pinned residual has been read
    cudaEvent_t ev_a = nullptr;        // the second GPU's last use of the staging buffer
    cudaEvent_t ev_m = nullptr;        // the main GPU's last use of the staging buffer
    float* host_R = nullptr;           // pinned, portable: [max_chunk, hc * n_embd]
    uint8_t* stage = nullptr;          // pinned, portable
    uint64_t stage_bytes = 0;
    // the second stage's thread and its queue
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Job> jobs;
    bool quit = false, busy = false, failed = false, on = false, ckpts_on = false;
    std::string err;
    int64_t pushed = 0, taken = 0;   // chunks handed on, and those whose residual the second stage has read
    int64_t ran_pos = -1;            // ran(): the main thread's checkpoint decisions up to there are made
    std::vector<std::pair<int64_t, void*>> ckpts;

    const core::QsaState& main_qsa(size_t i) const { return ss->qsa_states[qsa_main[i]]; }

    /// `runs` through the staging buffer, a buffer's worth at a time: from the main GPU's memory to the second's
    /// (`in`) or back.  Each side's copies go out once the buffer is theirs, waited for on the host: a copy queued
    /// behind a wait holds up every copy queued after it on its GPU's copy engine (the other stage's too).  The last
    /// batch's copies to the reading side are left running (`last_read`).  The caller may have either device current.
    cudaEvent_t last_read = nullptr;
    bool stage_copy(const std::vector<Run>& runs, bool in, std::string& e) {
        int cur = 0;
        cudaGetDevice(&cur);
        std::vector<Run> batch;
        uint64_t used = 0;
        auto flush = [&]() -> bool {
            if (batch.empty()) return true;
            // the side that writes the staging buffer, then the side that reads it
            const int wdev = in ? main : dev, rdev = in ? dev : main;
            cudaStream_t ws = in ? back : cp2, rs = in ? cp2 : back;
            cudaEvent_t wev = in ? ev_m : ev_a, rev = in ? ev_a : ev_m;
            if (last_read != nullptr && cudaEventSynchronize(last_read) != cudaSuccess) {
                e = std::string("prompt split: copies: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            cudaSetDevice(wdev);
            uint64_t o = 0;
            for (const Run& r : batch) {
                cudaMemcpyAsync(stage + o, r.src, (size_t) r.bytes, cudaMemcpyDeviceToHost, ws);
                o += up(r.bytes);
            }
            cudaEventRecord(wev, ws);
            if (cudaEventSynchronize(wev) != cudaSuccess) {
                e = std::string("prompt split: copies: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            cudaSetDevice(rdev);
            o = 0;
            for (const Run& r : batch) {
                cudaMemcpyAsync(r.dst, stage + o, (size_t) r.bytes, cudaMemcpyHostToDevice, rs);
                o += up(r.bytes);
            }
            cudaEventRecord(rev, rs);
            last_read = rev;
            batch.clear();
            used = 0;
            const cudaError_t x = cudaGetLastError();
            if (x != cudaSuccess) { e = std::string("prompt split: copies: ") + cudaGetErrorString(x); return false; }
            return true;
        };
        for (Run r : runs)
            while (r.bytes > 0) {
                if (used >= stage_bytes && !flush()) return false;
                const uint64_t n = std::min(r.bytes, stage_bytes - used);
                batch.push_back({r.src, r.dst, n});
                used += up(n);
                r.src += n;
                r.dst += n;
                r.bytes -= n;
            }
        const bool ok = flush();
        cudaSetDevice(cur);
        return ok;
    }

    /// The runs of cells [c0, c1) of every QSA layer here (and the pooled keys of their blocks), from the second
    /// stage's copies to the main session's (`back`) or the other way.
    void qsa_runs(int64_t c0, int64_t c1, bool back_to_main, std::vector<Run>& runs) const {
        if (c0 >= c1) return;
        const uint64_t row = (uint64_t) s.idx_dim * 4;
        const int64_t r0 = c0 / s.idx_block, r1 = c1 / s.idx_block + 1;   // the open block's row too
        for (size_t i = 0; i < qsa2.size(); ++i) {
            const auto a2 = core::qsa_kv_arrays(qsa2[i], *g), am = core::qsa_kv_arrays(main_qsa(i), *g);
            for (size_t k = 0; k < a2.size(); ++k)
                if (back_to_main) cell_runs(c0, c1, s.page_size, s.n_head_kv, a2[k].row_bytes, a2[k].data, 0, am[k].data, 0, runs);
                else cell_runs(c0, c1, s.page_size, s.n_head_kv, am[k].row_bytes, am[k].data, 0, a2[k].data, 0, runs);
            const uint8_t* px = (const uint8_t*) qsa2[i].idx_pooled + (uint64_t) r0 * row;
            const uint8_t* pm = (const uint8_t*) main_qsa(i).idx_pooled + (uint64_t) r0 * row;
            const uint64_t n = (uint64_t) (r1 - r0) * row;
            runs.push_back(back_to_main ? Run{px, (uint8_t*) pm, n} : Run{pm, (uint8_t*) px, n});
        }
    }

    /// The fixed-size state of the layers here: the GDN states and the indexers' tails, spare keys and block positions.
    void state_runs(bool back_to_main, std::vector<Run>& runs) const {
        const uint64_t row = (uint64_t) s.idx_dim * 4;
        const uint64_t gdn = (uint64_t) gdn_count * core::gdn_state_floats(*g) * 4;
        uint8_t* gm = (uint8_t*) (ss->gdn_state + (uint64_t) gdn_first * core::gdn_state_floats(*g));
        auto add = [&](const void* x2, const void* xm, uint64_t n) {
            runs.push_back(back_to_main ? Run{(const uint8_t*) x2, (uint8_t*) xm, n} : Run{(const uint8_t*) xm, (uint8_t*) x2, n});
        };
        add(ss2.gdn_state, gm, gdn);
        for (size_t i = 0; i < qsa2.size(); ++i) {
            add(qsa2[i].idx_tail, main_qsa(i).idx_tail, (uint64_t) (s.idx_block - 1) * row);
            add(qsa2[i].idx_dead, main_qsa(i).idx_dead, row);
            add(qsa2[i].idx_block_pos, main_qsa(i).idx_block_pos, 4);
        }
    }

    /// The draft layer's cells of one chunk: its copy for cells [p0, p0 + T).
    uint64_t draft_bytes(int64_t chunk, int64_t end) const {
        const int64_t pages = (end + s.page_size - 1) / s.page_size, held = (chunk + s.page_size - 1) / s.page_size + 1;
        return core::qsa_state_prompt_bytes(*g, end, std::max<int64_t>(pages - held, 0), false,
                                            core::qsa_draft_kv_format());
    }

    // ---- the second stage's thread
    bool copy_in(std::string& e) {
        const auto t0 = Clock::now();
        // the weights from the main GPU's, through the staging buffer beside the first stage's first chunk (a pinned
        // host copy of them was no faster: 0.5% of a 32K-token prompt)
        std::vector<Run> runs;
        for (const WSrc& w : wsrc) runs.push_back({(const uint8_t*) w.dev, w_base + w.off, w.bytes});
        state_runs(false, runs);
        qsa_runs(0, pos0, false, runs);
        if (!stage_copy(runs, true, e)) return false;
        cudaStreamWaitEvent(s2, ev_a, 0);   // the stage's work after its state and weights
        cudaEvent_t w_done = nullptr;
        cudaEventCreateWithFlags(&w_done, cudaEventDisableTiming);
        cudaEventRecord(w_done, cp2);
        cudaStreamWaitEvent(s2, w_done, 0);
        cudaEventDestroy(w_done);
        if (cudaStreamSynchronize(cp2) != cudaSuccess) {
            e = std::string("prompt split: copy in: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        self->ms_copy_in += core::ms_since(t0);
        return true;
    }

    bool run_chunk(const Job& j, std::string& e) {
        const int64_t D = g->hc * g->n_embd, n = (j.T + SUB - 1) / SUB;
        // the residual's pieces, each queued once it has landed in host memory (a copy queued behind a wait would hold
        // up the first stage's copies on this GPU's engine)
        float* R = p2.residual();
        for (int64_t k = 0; k < n; ++k) {
            const int64_t P = std::min(SUB, j.T - k * SUB);
            if (cudaEventSynchronize(j.landed[(size_t) k]) != cudaSuccess) {
                e = std::string("prompt split: the residual: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            cudaMemcpyAsync(R + k * SUB * D, host_R + k * SUB * D, (size_t) (P * D * 4), cudaMemcpyHostToDevice, cp2);
            cudaEventRecord(ev_in[(size_t) k], cp2);
        }
        cudaEventRecord(ev_free, cp2);
        {
            std::lock_guard<std::mutex> lk(mu);
            ++taken;
        }
        cv.notify_all();
        // the last chunk's cells have gone out (its copies read the copies this chunk writes next to them)
        cudaStreamWaitEvent(s2, ev_a, 0);
        if (mtp != nullptr &&
            core::qsa_state_prompt_init(*g, j.p0 + j.T, j.p0 / s.page_size, false, core::qsa_draft_kv_format(), draft_base,
                                        draft2, s2) == 0) {
            e = "prompt split: the draft layer's cells";
            return false;
        }
        dkv2.kv = &draft2;
        dkv2.first_cell = first_cell;
        p2.set_in_events(ev_in.data(), n);
        if (!p2.run(j.tokens, j.T, j.p0, e)) return false;
        // the chunk's cells to the main GPU's session
        std::vector<Run> runs;
        qsa_runs(j.p0, j.p0 + j.T, true, runs);
        if (mtp != nullptr) {
            const core::QsaState& dm = mtp->kv_state();
            const auto a2 = core::qsa_kv_arrays(draft2, *g), am = core::qsa_kv_arrays(dm, *g);
            const int64_t c0 = std::max(first_cell, j.p0), fp = j.p0 / s.page_size;
            for (size_t k = 0; k < a2.size() && c0 < j.p0 + j.T; ++k)
                cell_runs(c0, j.p0 + j.T, s.page_size, s.n_head_kv, a2[k].row_bytes, a2[k].data, fp, am[k].data, 0, runs);
        }
        if (!stage_copy(runs, false, e)) return false;
        // a prompt-cache checkpoint at its end: its layers' part, once the main thread has decided
        if (j.run_end && ckpts_on) {
            const int64_t pos = j.p0 + j.T;
            void* host = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || ran_pos >= pos; });
                for (size_t i = 0; i < ckpts.size(); ++i)
                    if (ckpts[i].first == pos) {
                        host = ckpts[i].second;
                        ckpts.erase(ckpts.begin() + (ptrdiff_t) i);
                        break;
                    }
            }
            if (host != nullptr && !core::session_ckpt_save_layers(ss2, *g, pos, host, s2, at, g->n_layers, at)) {
                e = "prompt split: the checkpoint's second part";
                return false;
            }
        }
        return true;
    }

    void worker() {
        cudaSetDevice(dev);
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || !jobs.empty(); });
                if (jobs.empty()) return;
                j = std::move(jobs.front());
                jobs.pop_front();
                busy = true;
            }
            std::string e;
            const bool ok = failed || (j.chunk ? run_chunk(j, e) : copy_in(e));
            {
                std::lock_guard<std::mutex> lk(mu);
                if (j.chunk && taken < pushed && failed) ++taken;   // a failed stage never read it
                if (!ok && !failed) {
                    failed = true;
                    err = e;
                    if (j.chunk && taken < pushed) ++taken;
                }
                busy = false;
            }
            cv.notify_all();
        }
    }
};

PromptSplit::PromptSplit() : impl_(new Impl) { impl_->self = this; }

PromptSplit::~PromptSplit() {
    Impl& m = *impl_;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        m.quit = true;
    }
    m.cv.notify_all();
    if (m.th.joinable()) m.th.join();
    if (m.dev >= 0) {
        int cur = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(m.dev);
        for (cudaStream_t x : {m.s2, m.cp2}) if (x) { cudaStreamSynchronize(x); cudaStreamDestroy(x); }
        for (cudaEvent_t e : m.ev_in) if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : {m.ev_free, m.ev_a}) if (e) cudaEventDestroy(e);
        cudaSetDevice(m.main);
        if (m.back) { cudaStreamSynchronize(m.back); cudaStreamDestroy(m.back); }
        if (m.ev_m) cudaEventDestroy(m.ev_m);
        cudaSetDevice(cur);
    }
    if (m.host_R) cudaFreeHost(m.host_R);
    if (m.stage) cudaFreeHost(m.stage);
}

bool PromptSplit::init(Prefill& first, const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                       core::MtpDrafter* mtp, int64_t at, int device, int main_device, core::ExpertSource* src,
                       const core::ExpertCache* cache, const int32_t* res, int64_t max_chunk, std::string& err) {
    Impl& m = *impl_;
    if (at < 2 || at >= g.n_layers) {
        err = "--prompt-split: the second GPU's first layer must be 2.." + std::to_string(g.n_layers - 1);
        return false;
    }
    // the second stage gathers the draft layer's token rows on its own GPU: from VRAM it could not
    if (mtp != nullptr && (core::native_embed() == nullptr || !core::native_embed()->mapped())) {
        err = "--prompt-split: the draft layer's prompt K/V need the token embedding in mapped host memory";
        return false;
    }
    m.p1 = &first;
    m.wt = &wt;
    m.g = &g;
    m.ss = &ss;
    m.mtp = mtp;
    m.at = at;
    m.dev = device;
    m.main = main_device;
    m.max_chunk = max_chunk;
    m.s = core::qsa_shapes(g);
    // ---- the tensors of the layers from `at` on, and the draft layer's prompt-K/V weights
    for (const auto& [name, ref] : wt.all()) {
        if (name.rfind("blk.", 0) != 0) continue;
        const int64_t l = std::atoll(name.c_str() + 4);
        if (l < at || l >= g.n_layers) continue;
        Tensor t;
        t.name = name;
        t.ref = ref;
        t.bytes[0] = ref.data != nullptr ? ref.bytes : 0;
        if (ref.native_data != nullptr) {
            int be = 0, bb = 0;
            if (!lamina::block_geometry((uint32_t) ref.native_type, be, bb) || be <= 0 || (ref.ne0 * ref.ne1) % be != 0) {
                err = "--prompt-split: " + name + " has a native type without a block geometry";
                return false;
            }
            t.bytes[1] = (uint64_t) (ref.ne0 * ref.ne1 / be) * (uint64_t) bb + kNativeTail;   // its tail copied too
        }
        if (ref.hc_codes != nullptr) t.bytes[2] = (uint64_t) (ref.ne0 * ref.ne1);
        if (ref.hc_scales != nullptr) t.bytes[3] = (uint64_t) (ref.ne0 * ref.ne1 / 32) * 2;
        for (int k = 0; k < 4; ++k) {
            t.off[k] = m.wbytes;
            m.wbytes += up(t.bytes[k]);
        }
        m.tensors.push_back(std::move(t));
    }
    const void* dkv_src[10] = {};
    if (mtp != nullptr) {
        const core::MtpPromptKv k = mtp->prompt_kv();
        const void* f[10] = {k.norm_emb, k.norm_hidden, k.fc_emb, k.fc_hidden, k.hc_norm,
                             k.hc_down,  k.hc_up,       k.k_proj, k.v_proj,    k.k_norm};
        for (int i = 0; i < 10; ++i) {
            dkv_src[i] = f[i];
            m.dkv_bytes[i] = f[i] != nullptr ? mtp->tensor_bytes(f[i]) : 0;
            if (m.dkv_bytes[i] == 0) {
                err = "--prompt-split: a draft layer weight has no size";
                return false;
            }
            m.dkv_off[i] = m.wbytes;
            m.wbytes += up(m.dkv_bytes[i]);
        }
    }
    weight_bytes = m.wbytes;
    for (const Tensor& t : m.tensors) {
        const void* d[4] = {t.ref.data, t.ref.native_data, t.ref.hc_codes, t.ref.hc_scales};
        for (int k = 0; k < 4; ++k)
            if (t.bytes[k] > 0) m.wsrc.push_back({d[k], t.off[k], t.bytes[k]});
    }
    for (int i = 0; i < 10 && mtp != nullptr; ++i) m.wsrc.push_back({dkv_src[i], m.dkv_off[i], m.dkv_bytes[i]});
    // ---- the layers' state indices
    for (int64_t l = 0, qi = 0, gi = 0; l < g.n_layers; ++l) {
        if (core::is_qsa_layer(g, l)) {
            if (l >= at) m.qsa_main.push_back(qi);
            ++qi;
        } else {
            if (l < at) m.gdn_first = gi + 1;
            else ++m.gdn_count;
            ++gi;
        }
    }
    // ---- the second GPU's side
    const int64_t D = g.hc * g.n_embd;
    cudaSetDevice(device);
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    m.ev_in.assign((size_t) ((max_chunk + SUB - 1) / SUB), nullptr);
    bool ok = cudaStreamCreateWithPriority(&m.s2, cudaStreamNonBlocking, least) == cudaSuccess &&
              cudaStreamCreateWithFlags(&m.cp2, cudaStreamNonBlocking) == cudaSuccess &&
              cudaEventCreateWithFlags(&m.ev_free, cudaEventDisableTiming) == cudaSuccess &&
              cudaEventCreateWithFlags(&m.ev_a, cudaEventDisableTiming) == cudaSuccess;
    for (cudaEvent_t& e : m.ev_in) ok = ok && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    // a chunk's cells of every K/V state here and the pooled keys of their blocks, through one staging buffer
    {
        uint64_t per_cell = (uint64_t) m.qsa_main.size() *
                            (core::qsa_kv_cell_bytes(g, core::qsa_kv_format()) + (uint64_t) m.s.idx_dim * 4 / (uint64_t) m.s.idx_block);
        if (mtp != nullptr) per_cell += core::qsa_kv_cell_bytes(g, core::qsa_draft_kv_format());
        m.stage_bytes = std::max<uint64_t>(64ull << 20, up((uint64_t) (max_chunk + 4 * m.s.page_size) * per_cell + (4ull << 20)));
    }
    ok = ok && cudaHostAlloc((void**) &m.host_R, (size_t) max_chunk * (size_t) D * 4, cudaHostAllocPortable) == cudaSuccess &&
         cudaHostAlloc((void**) &m.stage, (size_t) m.stage_bytes, cudaHostAllocPortable) == cudaSuccess;
    if (!ok) {
        cudaSetDevice(main_device);
        err = "--prompt-split: the second GPU's streams, events or pinned buffers";
        return false;
    }

    // its session state: the QSA states sized as the main session's until `begin` makes the copies
    m.qsa2.assign(m.qsa_main.size(), core::QsaState{});
    for (core::QsaState& q : m.qsa2) q.max_cells = ss.qsa_states[0].max_cells;
    m.ss2.qsa_states = m.qsa2.data();
    m.ss2.k = ss.k;
    m.ss2.max_cells = ss.max_cells;
    m.p2.set_layers(at, -1, nullptr);
    m.p2.set_fast_link(true);
    // its GPU's queue of this stage's work short: the first stage's shares queued after it wait for all of it (1, 2 and
    // 4 sections measured alike)
    m.p2.set_pacing(2);
    m.p2.draft = mtp;
    m.p2.draft_kv = mtp != nullptr ? &m.dkv2 : nullptr;
    if (!m.p2.init(m.wt2, g, m.ss2, src, cache, res, max_chunk, m.s2, nullptr, err)) {
        cudaSetDevice(main_device);
        err = "--prompt-split: " + err;
        return false;
    }
    cudaSetDevice(main_device);
    if (cudaStreamCreateWithFlags(&m.back, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&m.ev_m, cudaEventDisableTiming) != cudaSuccess) {
        err = "--prompt-split: the main GPU's copy stream";
        return false;
    }
    m.th = std::thread([&m] { m.worker(); });
    return true;
}

int64_t PromptSplit::at() const { return impl_->at; }
void PromptSplit::set_order(std::vector<int32_t> order) { impl_->p2.set_order(std::move(order)); }
bool PromptSplit::active() const { return impl_->on; }
const PrefillStats& PromptSplit::stats() const { return impl_->p2.stats(); }
void PromptSplit::set_profile(bool on) { impl_->p2.profile = on; }

uint64_t PromptSplit::bytes_for(int64_t chunk, int64_t cells) const {
    const Impl& m = *impl_;
    uint64_t n = up(m.p2.bytes_for(chunk)) + up(m.wbytes);
    n += (uint64_t) m.qsa2.size() * up(core::qsa_state_prompt_bytes(*m.g, cells, 0, true, core::qsa_kv_format()));
    n += up((uint64_t) m.gdn_count * core::gdn_state_floats(*m.g) * 4);
    if (m.mtp != nullptr) n += up(m.draft_bytes(chunk, cells));
    return n;
}

bool PromptSplit::begin(void* region, uint64_t bytes, int64_t chunk, int64_t pos0, int64_t cells, bool checkpoints,
                        std::string& err) {
    Impl& m = *impl_;
    if (m.on) { err = "prompt split: begin twice"; return false; }
    // the copies after the second stage's buffers, whose copies ahead take what is left: bytes_for sized that for the
    // residency before the loan, which made more of its experts stream
    const uint64_t fixed = bytes_for(chunk, cells) - up(m.p2.bytes_for(chunk));
    if (bytes < fixed + (64ull << 20)) { err = "prompt split: the second GPU's region is too small"; return false; }
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(m.dev);
    uint8_t* p = (uint8_t*) region;
    const uint64_t pb = (bytes - fixed) & ~255ull;
    if (!m.p2.bind(p, pb, chunk, err)) {
        cudaSetDevice(cur);
        return false;
    }
    p += pb;
    m.w_base = p;
    p += up(m.wbytes);
    // the weights' rows on the second GPU
    for (const Tensor& t : m.tensors) {
        core::WeightRef r = t.ref;
        r.data = t.bytes[0] ? m.w_base + t.off[0] : nullptr;
        r.native_data = t.bytes[1] ? m.w_base + t.off[1] : nullptr;
        r.hc_codes = t.bytes[2] ? (const int8_t*) (m.w_base + t.off[2]) : nullptr;
        r.hc_scales = t.bytes[3] ? (const uint16_t*) (m.w_base + t.off[3]) : nullptr;
        r.native_q8_1 = nullptr;
        m.wt2.put(t.name, r);
    }
    if (m.mtp != nullptr) {
        const uint8_t* w = m.w_base;
        m.dkv2.norm_emb = (const float*) (w + m.dkv_off[0]);
        m.dkv2.norm_hidden = (const float*) (w + m.dkv_off[1]);
        m.dkv2.fc_emb = w + m.dkv_off[2];
        m.dkv2.fc_hidden = w + m.dkv_off[3];
        m.dkv2.hc_norm = (const float*) (w + m.dkv_off[4]);
        m.dkv2.hc_down = (const uint16_t*) (w + m.dkv_off[5]);
        m.dkv2.hc_up = (const uint16_t*) (w + m.dkv_off[6]);
        m.dkv2.k_proj = w + m.dkv_off[7];
        m.dkv2.v_proj = w + m.dkv_off[8];
        m.dkv2.k_norm = (const float*) (w + m.dkv_off[9]);
        m.first_cell = m.mtp->prompt_kv().first_cell;
    }
    // its session state's copies, zeroed (the prefix's cells and the state at pos0 come in on its thread)
    for (core::QsaState& q : m.qsa2) {
        const uint64_t n = core::qsa_state_prompt_init(*m.g, cells, 0, true, core::qsa_kv_format(), p, q, m.cp2);
        if (n == 0) { cudaSetDevice(cur); err = "prompt split: the K/V copies"; return false; }
        p += up(n);
    }
    m.ss2.gdn_state = (float*) p;
    p += up((uint64_t) m.gdn_count * core::gdn_state_floats(*m.g) * 4);
    m.draft_base = p;
    cudaSetDevice(cur);
    m.pos0 = pos0;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        m.pushed = m.taken = 0;
        m.ran_pos = -1;
        m.ckpts.clear();
        m.failed = false;
        m.err.clear();
        m.ckpts_on = checkpoints;
        m.on = true;
        m.jobs.push_back(Job{});   // the state at pos0 and the weights, first
    }
    m.cv.notify_all();
    m.p1->set_layers(0, m.at, this);
    return true;
}

void PromptSplit::checkpoint(int64_t pos, void* host) {
    Impl& m = *impl_;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        m.ckpts.emplace_back(pos, host);
    }
    m.cv.notify_all();
}

void PromptSplit::ran(int64_t pos) {
    Impl& m = *impl_;
    {
        std::lock_guard<std::mutex> lk(m.mu);
        m.ran_pos = std::max(m.ran_pos, pos);
    }
    m.cv.notify_all();
}

float* PromptSplit::out_begin(std::string& err) {
    Impl& m = *impl_;
    const auto t0 = Clock::now();
    bool any = false;
    {
        std::unique_lock<std::mutex> lk(m.mu);
        m.cv.wait(lk, [&] { return m.failed || m.taken >= m.pushed; });
        if (m.failed) {
            err = "prompt split: " + m.err;
            return nullptr;
        }
        any = m.pushed > 0;
    }
    // the last chunk's copies out of it are done (waited for here: the copies into it would hold up this GPU's
    // copies queued after them)
    if (any && cudaEventSynchronize(m.ev_free) != cudaSuccess) {
        err = std::string("prompt split: ") + cudaGetErrorString(cudaGetLastError());
        return nullptr;
    }
    ms_first_waited += core::ms_since(t0);
    return m.host_R;
}

bool PromptSplit::out_end(const int64_t* tokens, int64_t T, int64_t pos0, const cudaEvent_t* landed, int64_t piece,
                          bool run_end, std::string& err) {
    Impl& m = *impl_;
    if (piece != SUB || T > m.max_chunk) { err = "prompt split: a chunk out of range"; return false; }
    Job j;
    j.chunk = true;
    j.tokens = tokens;
    j.T = T;
    j.p0 = pos0;
    j.run_end = run_end;
    j.landed.assign(landed, landed + (T + SUB - 1) / SUB);
    {
        std::lock_guard<std::mutex> lk(m.mu);
        if (m.failed) {
            err = "prompt split: " + m.err;
            return false;
        }
        m.jobs.push_back(std::move(j));
        ++m.pushed;
    }
    m.cv.notify_all();
    return true;
}

bool PromptSplit::finish(std::string& err) {
    Impl& m = *impl_;
    if (!m.on) return true;
    const auto t0 = Clock::now();
    {
        std::unique_lock<std::mutex> lk(m.mu);
        m.ran_pos = INT64_MAX;
        m.cv.notify_all();
        m.cv.wait(lk, [&] { return m.jobs.empty() && !m.busy; });
    }
    m.on = false;
    m.p1->set_layers(0, -1, nullptr);
    bool ok = !m.failed;
    if (!ok) err = "prompt split: " + m.err;
    // the GDN states and the indexers' fixed-size state go back (the cells went after each chunk)
    if (ok) {
        std::vector<Run> runs;
        m.state_runs(true, runs);
        ok = m.stage_copy(runs, false, err);
    }
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(m.main);
    if (cudaStreamSynchronize(m.back) != cudaSuccess && ok) {
        err = std::string("prompt split: ") + cudaGetErrorString(cudaGetLastError());
        ok = false;
    }
    cudaSetDevice(m.dev);
    if (cudaStreamSynchronize(m.cp2) != cudaSuccess && ok) {
        err = std::string("prompt split: ") + cudaGetErrorString(cudaGetLastError());
        ok = false;
    }
    cudaSetDevice(cur);
    m.p1->add_routed(m.p2.routed());
    m.p2.clear_routed();
    ms_finish += core::ms_since(t0);
    return ok;
}

}  // namespace lamina::prefill
