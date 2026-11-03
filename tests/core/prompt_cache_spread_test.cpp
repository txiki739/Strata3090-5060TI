// tests/core/prompt_cache_spread_test.cpp - the prompt cache's checkpoint retention (conv_cache.hpp) against
// least recent use, on conversations checkpointed as the engine does it (CPU only).
#include "lamina/core/conv_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using lamina::core::conv_cache::eviction_victim;
using lamina::core::conv_cache::spread_victim;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

// A conversation's checkpoints, at most `cap`: its first prompt's (every 16,384 cells and its end), then `turns`
// turns of `turn` cells, each adding one 4 cells before its prompt's end (the last <|im_start|>) and one at its
// end.  Over the cap the retention policy picks the victim: spread (--prompt-cache-spread) or the root-pinned
// least recent use default.
std::vector<int64_t> converse(bool spread, int cap, int64_t first, int64_t turn, int turns) {
    std::vector<int64_t> pos;
    std::vector<uint64_t> stamps;
    uint64_t clock = 0;
    auto add = [&](int64_t p) {
        pos.push_back(p);
        stamps.push_back(++clock);
        while ((int) pos.size() > cap) {
            const size_t v = spread
                                 ? spread_victim(pos.data(), stamps.data(), pos.size(), cap, nullptr, nullptr)
                                 : eviction_victim(stamps.data(), stamps.size(), cap);
            pos.erase(pos.begin() + (std::ptrdiff_t) v);
            stamps.erase(stamps.begin() + (std::ptrdiff_t) v);
        }
    };
    for (int64_t c = 16384; c < first; c += 16384) add(c);
    add(first);
    for (int t = 1; t <= turns; ++t) {
        add(first + t * turn - 4);
        add(first + t * turn);
    }
    return pos;
}

// the cells read again by a prompt that leaves the conversation `back` cells before its end
int64_t reread(const std::vector<int64_t>& pos, int64_t back) {
    int64_t from = 0;
    for (int64_t p : pos)
        if (p <= pos.back() - back) from = p;
    return pos.back() - from;
}

// the largest ratio of cells read again to `back` over edits [lo, hi) cells before the end, every `step`
double worst(const std::vector<int64_t>& pos, int64_t lo, int64_t hi, int64_t step) {
    double w = 0;
    for (int64_t b = lo; b < hi; b += step) w = std::max(w, (double) reread(pos, b) / (double) b);
    return w;
}

}  // namespace

int main() {
    std::printf("prompt_cache_spread_test\n");
    {   // an agent session: a 3,000-cell first prompt, then 200 turns of 1,000 cells, 16 checkpoints
        const std::vector<int64_t> lru = converse(false, 16, 3000, 1000, 200);
        const std::vector<int64_t> spread = converse(true, 16, 3000, 1000, 200);
        check(lru.size() == 16 && spread.size() == 16, "16 checkpoints kept either way");
        check(lru.front() == 3000, "least recent use: the root survives 200 turns");
        check(reread(lru, 5000) <= 6000, "least recent use: an edit 5K back is inside the recent turns");
        check(reread(lru, 30000) == lru.back() - 3000, "least recent use: an edit past them falls back to the root");
        check(spread.front() == 3000, "spread: the first prompt's end (the root) stays");
        check(reread(spread, 500) <= 1000, "spread: the turn before is still there");
        check(worst(spread, 1000, spread.back() - 3000, 250) < 2.0,
              "spread: an edit d cells back anywhere reads less than 2d again");
    }
    {   // a 100K first prompt read from cell 0 (its 16K chunks), then 150 turns of 500 cells
        const std::vector<int64_t> spread = converse(true, 16, 100000, 500, 150);
        check(spread.front() == 16384, "spread: the first chunk's checkpoint stays");
        check(worst(spread, 1000, spread.back() - 16384, 250) < 2.0,
              "spread: after a long first prompt, an edit d cells back reads less than 2d again");
    }
    {
        const int64_t pos[] = {100, 200, 300, 400, 1000};
        const uint64_t st[] = {1, 2, 3, 4, 5};
        check(spread_victim(pos, st, 5, 4, nullptr, nullptr) == 1,
              "the point with the closest neighbours for its distance leaves");
        const bool pinned[] = {false, true, false, false, false};
        check(spread_victim(pos, st, 5, 4, nullptr, pinned) == 2, "never a pinned point");
        check(spread_victim(pos, st, 2, 4, nullptr, nullptr) == 1, "two points: the non-root (the policy's fallback)");
        const int64_t three[] = {100, 200, 300};
        const uint64_t st3[] = {1, 2, 3};
        const bool pinned3[] = {false, true, false};
        check(spread_victim(three, st3, 3, 2, nullptr, pinned3) == 2, "the only middle point pinned: the oldest leaf");
        check(spread_victim(three, st3, 3, 2, nullptr, nullptr) == 1, "three points: the middle one, never the root");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
