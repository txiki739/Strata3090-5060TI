// src/plan/plan_main.cpp - the `lamina-plan` CLI: print the memory plan, or say why it does not close.
//
// The engine must adapt to the GPU it is running on, so the plan is computed from measured free VRAM and the
// requested context rather than configured by hand.  This tool is that computation on its own, which is what
// makes it testable without a GPU.
#include "lamina/plan/plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    uint64_t ctx = 20480;
    uint64_t free_vram = 12ull * 1000 * 1000 * 1000; // RTX 5070 nominal; overridden by --vram
    lamina::plan::Costs costs;
    bool state_given = false;
    uint64_t pool_given = 0;   // --pool: the pool itself (0: from --vram)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](uint64_t& out) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            out = std::strtoull(argv[++i], nullptr, 10);
        };
        if (a == "--max-context")
            next(ctx);
        else if (a == "--vram")
            next(free_vram);
        else if (a == "--dense")
            next(costs.dense_bytes);
        else if (a == "--embd")
            next(costs.embd_bytes);
        else if (a == "--workspace")
            next(costs.workspace_bytes);
        else if (a == "--state") {
            next(costs.state_bytes);
            state_given = true;
        } else if (a == "--pool") {
            next(pool_given);
        } else if (a == "--help" || a == "-h") {
            std::printf("usage: lamina-plan [--max-context N] [--vram BYTES] [--dense B] [--embd B]\n"
                        "                   [--workspace B] [--state B] [--pool B]\n"
                        "  --state defaults to the geometry's own GDN recurrence + conv history, NOT to 0:\n"
                        "  the flag used to default to zero and nothing passed it, so the planner silently\n"
                        "  planned without 117.7 MB (85 expert slots) of non-evictable state.\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    // DEFAULT THE STATE FROM THE GEOMETRY.  A `--state` that defaults to 0 and is never passed is not a
    // default, it is an omission - and this one was worth 85 cache slots of overcommit.
    if (!state_given) costs.state_bytes = lamina::plan::state_bytes(lamina::plan::Geometry{});
    // The pool is what the KV cache and the expert cache share.  Derived from the card's free VRAM less the
    // resident dense weights and embedding (make_plan takes the other fixed costs out of it); `--pool` gives it
    // directly, to reproduce the standard-tier numbers exactly (lamina::plan::vram_pool_bytes()).  A card with no
    // room left plans nothing: 0 does not close.
    const uint64_t pool = pool_given ? pool_given
                          : free_vram > costs.dense_bytes + costs.embd_bytes ? free_vram - costs.dense_bytes - costs.embd_bytes
                                                                              : 0;
    try {
        const lamina::plan::Plan p = lamina::plan::make_plan(ctx, lamina::plan::Geometry{}, costs, pool);
        std::printf("%s", lamina::plan::to_string(p).c_str());
        return 0;
    } catch (const lamina::plan::DoesNotClose& e) {
        std::fprintf(stderr, "lamina-plan: %s\n", e.what());
        return 1;
    }
}
