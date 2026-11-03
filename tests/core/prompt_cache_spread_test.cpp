// tests/core/prompt_cache_spread_test.cpp - a conversation's checkpoint retention (chain_victim,
// conversation_cache.hpp) against least recent use, on conversations checkpointed as the engine does it (CPU only).
#include "lamina/core/conversation_cache.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

using lamina::core::chain_victim;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_fail;
}

// A conversation's checkpoints, at most `cap`: its first prompt's (every 16,384 cells and its end), then `turns` turns
// of `turn` cells, each adding one 4 cells before its prompt's end (the last <|im_start|>) and one at its end.  Over
// the cap, chain_victim's point leaves, or with least recent use the oldest.
std::vector<int64_t> converse(bool spread, int cap, int64_t first, int64_t turn, int turns) {
    std::vector<int64_t> pos;
    auto add = [&](int64_t p) {
        pos.push_back(p);
        if ((int) pos.size() > cap) {
            const size_t v = spread ? chain_victim(pos.data(), pos.size()) : 0;
            pos.erase(pos.begin() + (std::ptrdiff_t) v);
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
        check(reread(lru, 10000) == lru.back(), "least recent use: an edit 10K back reads the whole conversation again");
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
        check(chain_victim(pos, 5) == 1, "the point with the closest neighbours for its distance leaves");
        check(chain_victim(pos, 2) == 0, "two points: the first (a chain of one keeps its newest)");
        const int64_t three[] = {100, 200, 300};
        check(chain_victim(three, 3) == 1, "three points: the middle one, never the root or the last");
    }
    std::printf(g_fail ? "FAIL\n" : "PASS\n");
    return g_fail ? 1 : 0;
}
