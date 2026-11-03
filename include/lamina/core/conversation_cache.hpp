// include/lamina/core/conversation_cache.hpp - the prompt cache's conversations and their checkpoints, CPU-only.
//
// Every conversation owns its checkpoints: the live one and each parked one (a conversation another request replaced,
// kept in host memory with its K/V cells) hold chains of their own, so no conversation's new checkpoints can take
// another's.  A chain over its cap gives up the point chain_victim picks; a parked conversation leaves only whole, the
// least recently used first, when the park's bytes (its K/V copies and its chains) or its slots run out, and never
// while it lends its start to the live one.
//
// The checkpoint buffers and the K/V copies are opaque here: `PromptCache` allocates and frees them, this header only
// moves them and counts their bytes, so the policy is testable without a device.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace lamina::core {

/// Identity of an image embedded in a GENI request.  Token IDs alone only describe the image-pad run; the embedding
/// bytes and grid describe what the model actually saw.  `start` is the first image-pad token position.
struct ImgKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& other) const { return start == other.start && hash == other.hash; }
};

/// One checkpoint of a conversation's chain: the session's fixed-size state at `pos`, in a buffer the chain owns.
struct Ckpt {
    int64_t pos = 0;
    void* host = nullptr;   ///< the session_ckpt buffer (opaque here: `PromptCache` owns the allocator)
};

/// One conversation: its tokens, its checkpoints (positions ascending) and, when parked, a copy of its K/V cells.
struct Seq {
    std::vector<int64_t> tokens;
    std::vector<ImgKey> imgs;
    std::vector<Ckpt> ckpts;
    std::unique_ptr<uint8_t[]> kv;   ///< cells [0, kv_cells) of every K/V state (parked conversations only)
    int64_t kv_cells = 0;
    uint64_t kv_bytes = 0;
    uint64_t stamp = 0;   ///< the park's clock at its last use
    /// begin(): the checkpoint another conversation borrows from this parked one, which no eviction takes while the
    /// live one is parked beside it; -1: none.
    int64_t pin = -1;

    const Ckpt* find(int64_t pos) const {
        for (const Ckpt& c : ckpts)
            if (c.pos == pos) return &c;
        return nullptr;
    }
};

/// The point of a chain over its cap that leaves (`pos` ascending, n >= 2).  Of those between its first (the chain's
/// root, the deepest point later prompts share) and its last, the one whose removal leaves the smallest gap for its
/// distance from the last (upstream's --prompt-cache-spread rule), so the kept points thin out geometrically back from
/// the newest: a prompt that leaves the chain d cells before its end reads about d to 2d of them again.  With two
/// points, the first: a chain of one keeps its newest.
inline size_t chain_victim(const int64_t* pos, size_t n) {
    size_t v = 0;
    double best = 0;
    for (size_t i = 1; i + 1 < n; ++i) {
        const double gap = double(pos[i + 1] - pos[i - 1]) / double(pos[n - 1] - pos[i - 1]);
        if (v == 0 || gap < best) {
            v = i;
            best = gap;
        }
    }
    return v;
}

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

/// The parked conversations: their K/V copies and chains within `budget` bytes, at most `slots` of them (0: no limit),
/// and only conversations of at least `min_tokens` cells.
class ConversationCache {
public:
    ConversationCache(uint64_t budget, size_t slots, int64_t min_tokens, uint64_t ckpt_bytes)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens), ckpt_bytes_(ckpt_bytes) {}

    /// Whether a conversation of `cells` cells is parked at all.
    bool wants(int64_t cells) const { return budget_ > 0 && cells >= min_tokens_; }
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    uint64_t budget() const { return budget_; }
    size_t slots() const { return slots_; }
    /// A parked conversation's bytes: its K/V copy and its chain.
    uint64_t bytes(const Seq& s) const { return s.kv_bytes + s.ckpts.size() * ckpt_bytes_; }
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const Seq& s : entries_) n += bytes(s);
        return n;
    }

    /// Makes room for a conversation of `incoming` bytes: the least recently used conversation that lends nothing
    /// leaves whole until it fits and a slot is free.  The evicted ones move to `evicted` (their buffers are the
    /// caller's to free).  false, with nothing evicted: it would not fit even so (the budget, or only lenders left).
    bool make_room(uint64_t incoming, std::vector<Seq>& evicted) {
        if (incoming > budget_) return false;
        std::vector<size_t> victims;   // chosen first: an eviction is only committed once the incoming one fits
        uint64_t held = bytes();
        size_t left = entries_.size();
        while ((slots_ > 0 && left >= slots_) || held + incoming > budget_) {
            size_t pick = entries_.size();
            for (size_t i = 0; i < entries_.size(); ++i)
                if (entries_[i].pin < 0 && std::find(victims.begin(), victims.end(), i) == victims.end() &&
                    (pick == entries_.size() || entries_[i].stamp < entries_[pick].stamp))
                    pick = i;
            if (pick == entries_.size()) return false;
            victims.push_back(pick);
            held -= bytes(entries_[pick]);
            --left;
        }
        std::sort(victims.begin(), victims.end(), std::greater<size_t>());   // erased from the back
        for (size_t i : victims) {
            evicted.push_back(std::move(entries_[i]));
            entries_.erase(entries_.begin() + (ptrdiff_t) i);
            ++evictions_;
        }
        return true;
    }

    /// Admits `s` (after make_room) as the most recently used.
    void park(Seq&& s) {
        entries_.push_back(std::move(s));
        entries_.back().stamp = tick();
    }
    /// Moves the entry at `at` out (the conversation continues live).
    Seq take(size_t at) {
        Seq s = std::move(entries_[at]);
        entries_.erase(entries_.begin() + (ptrdiff_t) at);
        return s;
    }
    /// Moves every entry out (the caller frees their buffers).
    std::vector<Seq> drain() { return std::exchange(entries_, {}); }
    std::vector<Seq>& entries() { return entries_; }
    const std::vector<Seq>& entries() const { return entries_; }
    /// The park's use clock: a parked conversation's stamp advances when it is parked and when it lends its start.
    uint64_t tick() { return ++clock_; }

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
