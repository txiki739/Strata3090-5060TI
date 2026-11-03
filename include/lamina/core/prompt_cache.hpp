// include/lamina/core/prompt_cache.hpp - the --serve prompt cache.
//
// A request continues from the longest prefix of its prompt that the engine still holds, instead of processing the
// whole conversation again.  Two things hold prefixes:
//
//   * the LIVE conversation: the tokens whose cells the session state holds right now, and its CHECKPOINTS
//     (session_ckpt_*): the fixed-size state at one position.  The K/V cells are indexed by position, so restoring a
//     checkpoint at `pos` and overwriting the cells from `pos` on continues the sequence from there;
//   * PARKED conversations: a conversation the live one replaced, kept in host memory with its K/V cells and its own
//     checkpoints, so a client that alternates between conversations (a chat and its title request, an agent and its
//     subagents) does not start over on every switch.  A prompt that continues a parked conversation from its
//     newest checkpoint takes it out of the park; one that matches it only up to an earlier checkpoint (another
//     conversation with the same start: sibling subagents, sessions with one system prompt) borrows the cells up to
//     there, and the parked conversation stays whole.
//
// Each conversation owns its checkpoints, at most `chain_cap` of them, which thin out geometrically back from its
// newest (chain_victim); a parked conversation leaves only whole (ConversationCache, conversation_cache.hpp).  One
// allocator serves every chain, a free list between uses: the first chain_cap + 1 buffers pinned where the driver
// allows (the live chain and its end), the rest pageable.
#pragma once

#include "lamina/core/conversation_cache.hpp"
#include "lamina/core/session.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace lamina::core {

class PromptCache {
public:
    /// `kv`: every state whose cells a parked conversation keeps (the QSA layers and the draft layer).  `chain_cap`:
    /// checkpoints a conversation keeps (0: the cache is off); `park_budget`, `park_slots`, `min_tokens`: the park's
    /// (ConversationCache).
    PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream, int chain_cap,
                uint64_t park_budget, size_t park_slots, int64_t min_tokens);
    ~PromptCache();
    PromptCache(const PromptCache&) = delete;
    PromptCache& operator=(const PromptCache&) = delete;

    bool enabled() const { return chain_cap_ > 0; }
    /// The elastic K/V (--kv-grow): called before any copy of `cells` K/V cells to or from the device (a parked
    /// conversation's restore, the live one's park), so the pools can grow to hold them first.  false: no room.
    std::function<bool(int64_t cells)> ensure_kv;
    /// Item bf, while a split prompt path runs (prefill/split.hpp): a checkpoint saves the layers below `split_from`
    /// itself and hands its buffer to `split_save`, whose stage writes the rest once that chunk is through it (the
    /// prompt's end waits for that).  0: not split.
    int64_t split_from = 0;
    std::function<void(int64_t pos, void* host)> split_save;
    /// Cells the live sequence holds (a later park reads all of them).
    int64_t live_cells() const { return (int64_t) live_.tokens.size(); }
    uint64_t ckpt_bytes() const { return ckpt_bytes_; }

    /// Sets the session up for a prompt of `ids`: continues from the longest prefix (at most ids.size() - 1 cells,
    /// the last token always goes through a verify window) whose token and image prefix matches the live sequence, a
    /// checkpoint or a parked conversation, or zeroes the session.  Image identity is checked before the matching
    /// position is returned; a changed image clamps reuse to before that image.  Text-only requests pass an empty
    /// image-key vector.
    int64_t begin(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs);
    /// The cells [pos, pos + n) now hold `tokens`; pos <= live length, and anything after pos is gone.
    void set(int64_t pos, const int64_t* tokens, int64_t n);
    void set(int64_t pos, const int32_t* tokens, int64_t n);
    /// Checkpoints the session, which is at `pos` == the live length.
    void checkpoint(int64_t pos);
    /// Forgets the live sequence, every checkpoint and every parked conversation (after non-finite logits: whatever
    /// the state holds may be poisoned).
    void clear();

    /// The cache's state: the live conversation, the park's contents, its budgets and its evictions (the CACHE line,
    /// which the Monitor shows).
    struct State {
        int64_t live_cells = 0;
        size_t checkpoints = 0;
        size_t parked = 0;
        uint64_t parked_bytes = 0, parked_budget = 0;
        size_t slots = 0, evictions = 0;
    };
    State state() const;
    /// One line for the log.
    std::string summary() const;

private:
    /// One device region of the K/V cells, stored back to back in a parked conversation's buffer.
    struct Part { void* dev; uint64_t bytes; };

    /// A checkpoint buffer (the header's allocator) and its return to the free list.
    void* take_buffer();
    void give_back(void* h);
    void release(Seq& seq, int64_t keep_upto);   // frees the checkpoints after `keep_upto`
    /// Thins `seq`'s chain to the cap (chain_victim).
    void enforce_cap(Seq& seq);
    /// The session's state at `pos` (== the live length) in a buffer of its own; nullptr: none.
    void* save(int64_t pos);
    /// Parks a copy of the live sequence: its K/V cells, its checkpoints after `keep_upto` (the ones before stay with
    /// the live sequence, which continues from there) and one at its end.
    void stash_live(int64_t keep_upto);
    std::vector<Part> kv_parts(int64_t cells) const;
    /// The first `cells` K/V cells between the device and `host`, a parked copy of `held` >= cells cells
    /// (kv_parts(held) back to back: each region's first cells lead its part).
    bool kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host);
    /// The checkpoint at `pos` and, of a parked conversation, its K/V cells: all of them, or with `prefix` only those
    /// before `pos` (a borrowed prefix).
    bool restore(const Seq& seq, int64_t pos, bool prefix = false);

    const ModelGeometry& g_;
    SessionState& ss_;
    std::vector<QsaState*> kv_;
    void* stream_;
    int chain_cap_;
    uint64_t ckpt_bytes_;
    Seq live_;
    ConversationCache parked_;
    std::vector<void*> free_;
    std::vector<std::pair<void*, bool>> owned_;   // every buffer, and whether it is pinned
    int pinned_ = 0;
};

}  // namespace lamina::core
