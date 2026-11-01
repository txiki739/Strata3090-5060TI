// include/lamina/core/conversation_cache.hpp - the parked conversations' ownership and bookkeeping, CPU-only.
//
// The shape (upstream Strata's, see its include/strata/core/conversation_cache.hpp): every conversation's
// checkpoints are its OWN - no shared buffer pool a chatty conversation can drain from a parked one.  A
// conversation's chain caps at `cap` checkpoints (retention: conv_cache.hpp); a parked conversation leaves only
// whole, by the byte budget or the slot count, least recent use first, and never while it lends a shared prefix.
//
// The checkpoint host buffers and the K/V copies are opaque here: `PromptCache` allocates and frees them, this
// header only moves them and counts their bytes.  That keeps the policy testable without a device.
#pragma once

#include "lamina/core/conv_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace lamina::core {

/// Identity of an image embedded in a GENI request.  Token IDs alone only describe the image-pad run; the
/// embedding bytes and grid describe what the model actually saw.  `start` is the first image-pad token position.
struct ImgKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& other) const { return start == other.start && hash == other.hash; }
};

/// One checkpoint of a conversation's chain: the folded state at `pos`, in a buffer owned by this chain.
struct Ckpt {
    int64_t pos = 0;
    void* host = nullptr;      ///< the session_ckpt buffer (opaque here; `PromptCache` owns the allocator)
    uint64_t stamp = 0;        ///< last use: advances on creation and when a request mounts through it
    bool tail = false;         ///< an extra cut near a prompt's end (a fork / msg_end): the first to go
};

/// One conversation's state: the tokens, the checkpoint chain (positions ascending), and - when parked or retained
/// - a host copy of the K/V cells.
struct Seq {
    std::vector<int64_t> tokens;
    std::vector<ImgKey> imgs;
    std::vector<Ckpt> ckpts;
    std::unique_ptr<uint8_t[]> kv;   ///< cells [0, kv_cells) of every K/V state (parked conversations only)
    int64_t kv_cells = 0;
    uint64_t kv_bytes = 0;
    uint64_t stamp = 0;
    /// begin(): the checkpoint a request resumes from, kept through the chain's and the budget's evictions while the
    /// live sequence is parked beside it (a conversation lending its prefix is not evicted either); -1: none.
    int64_t pin = -1;

    /// Chain order: positions ascending, the root (the deepest point every prompt so far shared) first.
    void sort_ckpts() {
        std::sort(ckpts.begin(), ckpts.end(), [](const Ckpt& a, const Ckpt& b) { return a.pos < b.pos; });
    }
    /// The index of the checkpoint at `pos`, or nullptr.
    const Ckpt* find(int64_t pos) const {
        for (const Ckpt& c : ckpts)
            if (c.pos == pos) return &c;
        return nullptr;
    }
    Ckpt* find(int64_t pos) {
        for (Ckpt& c : ckpts)
            if (c.pos == pos) return &c;
        return nullptr;
    }
};

/// Cells of `seq` a prompt can continue from: the whole sequence when it is a prefix of the prompt (live only: the
/// state is there), else its best checkpoint within the common prefix (its position; -1: none).  Image identity is
/// required besides the tokens: a changed image clamps the match to before that image's first pad token.
inline int64_t reusable(const Seq& seq, const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs, bool live) {
    const int64_t m = std::min<int64_t>((int64_t) seq.tokens.size(), (int64_t) ids.size() - 1);
    int64_t lcp = 0;
    while (lcp < m && seq.tokens[(size_t) lcp] == ids[(size_t) lcp]) ++lcp;

    // The token prefix is necessary but not sufficient for multimodal input: every image that falls inside the
    // reusable prefix must have the same embedding/grid identity at the same position.  If an image was changed,
    // removed, added, or reordered, fail closed before its first pad token.
    auto matching_image = [](const std::vector<ImgKey>& all, const ImgKey& wanted) {
        for (const ImgKey& image : all)
            if (image.start == wanted.start) return image == wanted;
        return false;
    };
    for (const ImgKey& image : seq.imgs)
        if (image.start < lcp && !matching_image(imgs, image)) lcp = image.start;
    for (const ImgKey& image : imgs)
        if (image.start < lcp && !matching_image(seq.imgs, image)) lcp = image.start;

    if (live && lcp > 0 && lcp == (int64_t) seq.tokens.size()) return lcp;
    int64_t best = -1;
    for (const Ckpt& c : seq.ckpts)
        if (c.pos <= lcp && c.pos > best) best = c.pos;
    return best;
}

/// The parked conversations: at most `slots` of them, at most `budget` bytes between them (their K/V copies and
/// their checkpoint chains), and only conversations of at least `min_tokens` tokens are parked at all.
class ConversationCache {
public:
    ConversationCache(uint64_t budget, size_t slots, int64_t min_tokens, uint64_t ckpt_bytes)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens), ckpt_bytes_(ckpt_bytes) {}

    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }
    size_t evictions() const { return evictions_; }
    /// Bytes held: every parked conversation's K/V copy plus its chain's buffers.
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const Seq& s : entries_) n += s.kv_bytes + s.ckpts.size() * ckpt_bytes_;
        return n;
    }

    /// Makes room for a conversation of `incoming` bytes: the least recently used unpinned conversation leaves
    /// whole until the incoming one fits and a slot is free.  Evicted conversations move to `evicted` (their
    /// buffers are the caller's to free); false when even an empty cache would not fit the incoming one, or only
    /// pinned conversations could pay for it - then nothing is evicted and the conversation simply is not parked.
    bool make_room(uint64_t incoming, std::vector<Seq>& evicted) {
        if (slots_ == 0 || incoming > budget_) return false;   // the incoming one alone does not fit
        // Dry run first: an eviction cannot be undone, so only commit when the incoming one is known to fit.
        std::vector<bool> gone(entries_.size(), false);
        std::vector<size_t> victims;
        uint64_t held = bytes();
        size_t left = entries_.size();
        while (left >= slots_ || held + incoming > budget_) {
            size_t pick = SIZE_MAX;   // the least recently used unpinned conversation
            for (size_t i = 0; i < entries_.size(); ++i)
                if (!gone[i] && entries_[i].pin < 0 &&
                    (pick == SIZE_MAX || entries_[i].stamp < entries_[pick].stamp))
                    pick = i;
            if (pick == SIZE_MAX) return false;   // only lenders remain: park nothing
            gone[pick] = true;
            held -= entries_[pick].kv_bytes + entries_[pick].ckpts.size() * ckpt_bytes_;
            --left;
            victims.push_back(pick);
        }
        std::sort(victims.begin(), victims.end(), std::greater<size_t>());   // erase from the back: indices hold
        for (size_t i : victims) {
            ++evictions_;
            evicted.push_back(std::move(entries_[i]));
            entries_.erase(entries_.begin() + (ptrdiff_t) i);
        }
        return true;
    }

    /// Admits `s` (the caller made room first).  Its stamp starts fresh: it is the most recent use.
    void park(Seq&& s) {
        entries_.push_back(std::move(s));
        entries_.back().stamp = ++clock_;
    }

    /// Moves out the entry at `at` (the caller continues it live).
    Seq take(size_t at) {
        Seq s = std::move(entries_[at]);
        entries_.erase(entries_.begin() + (ptrdiff_t) at);
        return s;
    }

    /// Moves every entry out (a clear: the caller frees their buffers).
    std::vector<Seq> drain() { return std::exchange(entries_, {}); }

    std::vector<Seq>& entries() { return entries_; }
    const std::vector<Seq>& entries() const { return entries_; }

    /// The shared use clock: stamps advance on park, on mount, and on checkpoint creation.
    uint64_t tick() { return ++clock_; }
    void set_clock(uint64_t c) { clock_ = c; }
    uint64_t clock() const { return clock_; }

private:
    uint64_t budget_;
    size_t slots_;
    int64_t min_tokens_;
    uint64_t ckpt_bytes_;
    uint64_t clock_ = 0;
    size_t evictions_ = 0;
    std::vector<Seq> entries_;
};

}  // namespace lamina::core
