// tests/core/conversation_cache_test.cpp - the prompt cache's conversations, CPU only (conversation_cache.hpp):
//
//   1. ownership: a parked conversation keeps its whole chain while another conversation's chain grows and is
//      thinned beside it, and stays resumable at its end (the pool shared by every conversation let a chatty one
//      take a parked one's checkpoints until it was dropped);
//   2. the park's limits: the slots and the byte budget (K/V copies and chains) evict whole conversations, the least
//      recently used first; no slot limit with 0; a lending conversation is never evicted; nothing is evicted for a
//      conversation that would not fit anyway; short conversations and a budget of 0 park nothing;
//   3. reusable(): the live sequence whole, a parked one from its best checkpoint, a changed image clamping the match.
#include "lamina/core/conversation_cache.hpp"

#include <cstdio>
#include <vector>

using lamina::core::chain_victim;
using lamina::core::ConversationCache;
using lamina::core::ImgKey;
using lamina::core::reusable;
using lamina::core::Seq;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

constexpr uint64_t kCkpt = 100;   // a checkpoint's bytes in these tests; a cell's K/V: 10

// a conversation of `cells` tokens (all 7) with checkpoints at `pos`, its K/V copy held
Seq conv(int64_t cells, std::vector<int64_t> pos) {
    Seq s;
    s.tokens.assign((size_t) cells, 7);
    for (int64_t p : pos) s.ckpts.push_back({p, nullptr});
    s.kv_cells = cells;
    s.kv_bytes = (uint64_t) cells * 10;
    return s;
}

std::vector<int64_t> prompt(int64_t cells) { return std::vector<int64_t>((size_t) cells, 7); }

}  // namespace

int main() {
    std::printf("conversation_cache_test\n");
    {   // 1. ownership
        ConversationCache park(1u << 30, 0, 1024, kCkpt);
        std::vector<Seq> evicted;
        Seq a = conv(180000, {16384, 32768, 49152, 179000, 180000});
        check(park.make_room(park.bytes(a), evicted), "a 180K-cell conversation parks");
        park.park(std::move(a));
        std::vector<int64_t> b = {4000};   // another conversation's chain: 200 turns of two checkpoints, 16 kept
        for (int t = 1; t <= 200; ++t)
            for (int64_t p : {4000 + t * 500 - 4, 4000 + t * 500}) {
                b.push_back(p);
                if (b.size() > 16) b.erase(b.begin() + (std::ptrdiff_t) chain_victim(b.data(), b.size()));
            }
        check(b.size() == 16 && b.front() == 4000 && b.back() == 104000,
              "the other chain keeps 16 checkpoints, its root and its newest among them");
        check(park.size() == 1 && park.entries()[0].ckpts.size() == 5 && park.evictions() == 0,
              "the parked conversation keeps all 5 of its checkpoints");
        check(reusable(park.entries()[0], prompt(180500), {}, false) == 180000, "and continues from its end");
    }
    {   // 2. the park's limits
        std::vector<Seq> evicted;
        ConversationCache slots(1u << 30, 2, 0, kCkpt);
        slots.park(conv(1000, {1000}));
        slots.park(conv(2000, {2000}));
        check(slots.make_room(10000, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 1000,
              "two slots: a third conversation evicts the least recently used whole");
        ConversationCache open(1u << 30, 0, 0, kCkpt);
        for (int i = 1; i <= 50; ++i) open.park(conv(1000 * i, {1000 * i}));
        evicted.clear();
        check(open.make_room(10000, evicted) && evicted.empty(), "slots 0: no limit but the budget");
        ConversationCache budget(40000, 0, 0, kCkpt);
        budget.park(conv(1000, {500, 1000}));    // 10,000 + 200
        budget.park(conv(2000, {2000}));         // 20,000 + 100
        budget.entries()[0].stamp = budget.tick();   // the first one used since (it lent its start)
        evicted.clear();
        check(budget.make_room(15000, evicted) && evicted.size() == 1 && evicted[0].kv_cells == 2000,
              "the budget counts the chains: the least recently used leaves, not the oldest parked");
        check(budget.bytes() == 10200 && budget.evictions() == 1, "the park's bytes and evictions follow");
        ConversationCache lend(30000, 0, 0, kCkpt);
        lend.park(conv(1000, {1000}));
        lend.entries()[0].pin = 1000;
        evicted.clear();
        check(!lend.make_room(25000, evicted) && evicted.empty() && lend.size() == 1,
              "a lending conversation is never evicted: the incoming one parks nothing");
        ConversationCache tight(25000, 0, 0, kCkpt);
        tight.park(conv(1000, {1000}));
        tight.entries()[0].pin = 1000;
        tight.park(conv(1200, {1200}));
        evicted.clear();
        check(!tight.make_room(20000, evicted) && evicted.empty() && tight.size() == 2,
              "nothing is evicted when the incoming one would not fit even so");
        check(!tight.make_room(30000, evicted) && tight.size() == 2, "nor for one over the whole budget");
        check(!ConversationCache(1u << 30, 0, 1024, kCkpt).wants(1023) &&
                  ConversationCache(1u << 30, 0, 1024, kCkpt).wants(1024) &&
                  !ConversationCache(0, 0, 0, kCkpt).wants(100000),
              "short conversations, and any with a budget of 0, are not parked");
    }
    {   // 3. reusable()
        const Seq live = conv(5000, {1000, 4000});
        check(reusable(live, prompt(6000), {}, true) == 5000, "the live sequence continues whole");
        check(reusable(live, prompt(6000), {}, false) == 4000, "a parked one from its newest checkpoint");
        std::vector<int64_t> edited = prompt(6000);
        edited[2500] = 8;
        check(reusable(live, edited, {}, true) == 1000, "an edit falls back to the checkpoint before it");
        Seq pics = conv(5000, {1000, 4000});
        pics.imgs = {{2000, 11}};
        check(reusable(pics, prompt(6000), {{2000, 12}}, false) == 1000, "a changed image clamps the match before it");
        check(reusable(pics, prompt(6000), {{2000, 11}}, false) == 4000, "the same image does not");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
