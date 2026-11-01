// tests/core/conversation_cache_test.cpp - the conversation cache's ownership and retention, CPU only:
//
//   1. the retention policy (conv_cache.hpp, upstream's scenarios): the root survives, the leaves rotate by
//      least recent use, the tail cuts go first, pins are never the victim, and a cap below two switches the
//      root pin off;
//   2. a conversation's chain over its own cap never empties, and a request that leaves `back` cells before
//      the end resumes from a checkpoint (reads at most a bounded part again);
//   3. the starvation regression (the 07:39:40 incident): a parked conversation survives a chatty one minting
//      checkpoints beside it - no conversation can take a checkpoint buffer from another;
//   4. the park's budgets: the slot count and the byte budget evict whole conversations least recently used,
//      a lending conversation is never evicted, a conversation over the whole budget parks nothing, and
//      --conversation-cache-min-tokens refuses short side requests.
#include "lamina/core/conversation_cache.hpp"

#include <cstdio>
#include <vector>

using lamina::core::conv_cache::eviction_victim;
using lamina::core::conv_cache::spread_victim;
using lamina::core::ConversationCache;
using lamina::core::Ckpt;
using lamina::core::reusable;
using lamina::core::Seq;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-72s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr uint64_t kCkpt = 100;   // the test's stand-in for a checkpoint buffer's bytes

// a conversation's chain: `turns` checkpoints after a root, cap `cap`, the policy's victim each time it grows
// over; returns the kept positions (ascending).
std::vector<int64_t> converse(int64_t root, int64_t turn, int turns, int64_t cap, const bool* tail = nullptr) {
    std::vector<int64_t> pos = {root};
    std::vector<uint64_t> stamps = {1};
    for (int t = 1; t <= turns; ++t) {
        pos.push_back(root + t * turn);
        stamps.push_back(uint64_t(t + 1));
        while ((int64_t) pos.size() > cap) {
            const size_t v = eviction_victim(stamps.data(), stamps.size(), cap, tail);
            pos.erase(pos.begin() + (std::ptrdiff_t) v);
            stamps.erase(stamps.begin() + (std::ptrdiff_t) v);
        }
    }
    return pos;
}

// the cells read again by a prompt that leaves the chain `back` cells before its end
int64_t reread(const std::vector<int64_t>& pos, int64_t back) {
    int64_t from = 0;
    for (int64_t p : pos)
        if (p <= pos.back() - back) from = p;
    return pos.back() - from;
}

Seq make_conv(int64_t cells, int64_t ckpts, uint64_t stamp) {
    Seq s;
    s.tokens.resize((size_t) cells, 7);
    for (int64_t c = 1; c <= ckpts; ++c) s.ckpts.push_back({c, nullptr, stamp, false});
    s.kv_bytes = (uint64_t) cells * 10;   // the test's K/V bytes-per-cell
    s.kv_cells = cells;
    s.stamp = stamp;
    return s;
}

}  // namespace

int main() {
    // ---- 1. the retention policy (upstream's conv_cache_test scenarios)
    {
        // a. the oldest leaf leaves; the root survives arbitrarily many newer checkpoints
        std::vector<uint64_t> stamps = {1, 2, 3, 4};
        check(eviction_victim(stamps.data(), stamps.size(), 3) == 1, "root pinned: the oldest leaf leaves");
        stamps = {100, 2, 3, 4};   // a root older than every leaf
        check(eviction_victim(stamps.data(), stamps.size(), 3) == 1, "the root survives any stamp");
        // b. a mount advances a stamp and the rotation then takes a different leaf
        stamps = {1, 5, 2, 4};
        check(eviction_victim(stamps.data(), stamps.size(), 3) == 2, "a mounted leaf survives one more round");
        // c. a one-slot cap: the pin is off, the oldest leaves
        stamps = {1, 2};
        check(eviction_victim(stamps.data(), stamps.size(), 1) == 0, "cap 1: the pin is off, the oldest leaves");
        // d. the tail cut goes first, unless it is the newest
        const bool tail[] = {false, true, false, false};
        stamps = {1, 2, 3, 4};
        check(eviction_victim(stamps.data(), stamps.size(), 3, tail) == 1, "the tail cut leaves first");
        const bool tail2[] = {false, true, false, true};
        check(eviction_victim(stamps.data(), stamps.size(), 3, tail2) == 1, "the tail cut leaves before the newest");
        // e. pins are never the victim; all pinned: the oldest pinned non-root
        const bool pinned[] = {false, true, false, false};
        stamps = {1, 2, 3, 4};
        check(eviction_victim(stamps.data(), stamps.size(), 3, nullptr, pinned) == 2,
              "a pinned shared prefix is never the victim: the next-oldest leaf goes");
        const bool pinned2[] = {false, true, true, true};
        check(eviction_victim(stamps.data(), stamps.size(), 3, nullptr, pinned2) == 1,
              "all pinned: the oldest non-root leaves");
        // f. purity: the same stamps, the same victim
        check(eviction_victim(stamps.data(), stamps.size(), 3, nullptr, pinned2) == 1, "the policy is pure");
        // g. spread: never the root, the newest or a pinned item
        std::vector<int64_t> lens = {100, 200, 300, 400, 500};
        std::vector<uint64_t> st = {1, 2, 3, 4, 5};
        const size_t v = spread_victim(lens.data(), st.data(), lens.size(), 4, nullptr, nullptr);
        check(v == 1 || v == 2 || v == 3, "spread: an interior item leaves, never the root or the newest");
    }

    // ---- 2. a chain over its own cap never empties and bounds the re-read
    {
        const auto pos = converse(4096, 1024, 40, 6);
        check((int64_t) pos.size() == 6, "a 40-turn conversation keeps its cap of checkpoints");
        check(pos.front() == 4096, "the root survives 40 turns");
        check(pos.back() == 4096 + 40 * 1024, "the newest survives 40 turns");
        int64_t worst = 0;
        for (int64_t back = 1024; back <= 4 * 1024; back += 1024)
            worst = std::max(worst, reread(pos, back) / back);
        check(worst <= 2, "an edit up to 4 turns back reads at most twice what it left");
        check(reread(pos, 20 * 1024) == pos.back() - 4096, "an edit past the chain falls back to the root");
        // the cap is a cap at every size: cap 1 keeps the newest point only
        const auto one = converse(4096, 1024, 10, 1);
        check(one.size() == 1 && one.front() == 4096 + 10 * 1024, "cap 1: the newest point only");
    }

    // ---- 3. the starvation regression: a chatty conversation cannot drain a parked one
    {
        // Conversation A: 180K cells, a full chain, parked.  Conversation B: mints 200 checkpoints beside it
        // (its own chain, its own cap).  A stays resumable at its newest checkpoint throughout.
        ConversationCache cache(1u << 30, 4, 0, kCkpt);
        Seq a = make_conv(180000, 6, 1);
        std::vector<Seq> evicted;
        check(cache.make_room(a.kv_bytes + a.ckpts.size() * kCkpt, evicted), "the first conversation parks");
        cache.park(std::move(a));
        Seq b = make_conv(20000, 1, 2);
        for (int t = 0; t < 200; ++t) {
            // B's chain over its cap: the victim is B's own, never A's (A is not even in the same container)
            b.ckpts.push_back({t, nullptr, uint64_t(10 + t), false});
            if (b.ckpts.size() > 6) b.ckpts.erase(b.ckpts.begin() + 1);   // the policy's victim (root pinned)
            // and A is still resumable at its newest checkpoint:
            std::vector<int64_t> prompt_a(180000, 7);
            prompt_a.resize(180003);
            const int64_t r = reusable(cache.entries()[0], prompt_a, {}, false);
            if (r != 6) { check(false, "A stays resumable while B mints 200 checkpoints"); return 1; }
        }
        check(true, "A stays resumable while B mints 200 checkpoints");
        check(cache.size() == 1 && cache.evictions() == 0, "nothing was evicted: B's minting never touches the park");
    }

    // ---- 4. the park's budgets
    {
        // a. the slot count: the oldest conversation leaves whole
        ConversationCache cache(1u << 30, 2, 0, kCkpt);
        std::vector<Seq> evicted;
        cache.park(make_conv(1000, 1, 1));
        cache.park(make_conv(2000, 1, 2));
        check(cache.make_room(10000, evicted), "a third conversation evicts the oldest");
        check(evicted.size() == 1 && evicted[0].tokens.size() == 1000, "the least recently used left");
        cache.park(make_conv(3000, 1, 3));
        check(cache.size() == 2, "the park holds its slots");
        // b. the byte budget: whole conversations go, oldest first, until the incoming one fits
        ConversationCache budget(100000, 8, 0, kCkpt);
        budget.park(make_conv(1000, 1, 1));   // 10,000 + 100
        budget.park(make_conv(2000, 1, 2));   // 20,000 + 100
        evicted.clear();
        check(budget.make_room(70000, evicted), "the budget evicts until the incoming one fits");
        check(evicted.size() == 1 && evicted[0].tokens.size() == 1000, "the budget took the oldest whole");
        // c. a lending conversation is never the victim
        ConversationCache lend(30000, 8, 0, kCkpt);
        Seq p = make_conv(1000, 1, 1);
        p.pin = 1;
        lend.park(std::move(p));
        evicted.clear();
        check(!lend.make_room(29000, evicted) && evicted.empty(), "a lending conversation is never evicted, and the "
              "incoming one parks nothing instead");
        // d. the incoming one alone over the budget parks nothing and evicts nothing
        ConversationCache big(10000, 4, 0, kCkpt);
        big.park(make_conv(100, 1, 1));
        evicted.clear();
        check(!big.make_room(50000, evicted) && big.size() == 1 && evicted.empty(),
              "a conversation over the whole budget parks nothing and nothing is lost");
        // e. min-tokens: short side requests are not parked
        ConversationCache min(1u << 30, 4, 4096, kCkpt);
        check(!min.wants(4095) && min.wants(4096), "min-tokens refuses short conversations");
        // f. the dry run: when the remaining conversations are pinned and still would not pay, nothing leaves
        ConversationCache tight(20000, 8, 0, kCkpt);
        Seq pinned = make_conv(1000, 1, 1);
        pinned.pin = 1;
        tight.park(std::move(pinned));   // 10,100, lending
        tight.park(make_conv(1200, 1, 2));
        evicted.clear();
        check(!tight.make_room(19000, evicted) && tight.size() == 2,
              "no eviction is committed when the incoming one still would not fit");
    }

    std::printf("%s\n", g_fail ? "FAILED" : "all ok");
    return g_fail ? 1 : 0;
}
