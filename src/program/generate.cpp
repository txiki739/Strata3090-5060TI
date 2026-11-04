// src/program/generate.cpp - the engine (`lamina`): a model's prompt through the batched prompt path and its
// decoding in verify windows of the last token and the draft layer's drafts, the experts in the GPUs' VRAM tiers and
// the CPU pool.  Generate mode reads pretokenized ids (--tokens) and prints the tokens it decodes; --serve keeps
// the model loaded and takes requests on stdin (serve/server.py speaks to it).

#include "lamina/artifact/gguf_split.hpp"
#if defined(LAMINA_HAVE_QWEN35)
#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/qwen35/engine.hpp"
#endif
#include "lamina/program/engine.hpp"
#include "lamina/program/serve_io.hpp"
#include "lamina/core/adaptive_tier.hpp"
#include "lamina/core/expert_cache.hpp"
#include "lamina/core/gpu_select.hpp"
#include "lamina/core/second_gpu.hpp"
#include "lamina/core/split_head.hpp"
#include "lamina/core/expert_source.hpp"
#include "lamina/core/layer.hpp"
#include "lamina/core/layout.hpp"
#include "lamina/core/session.hpp"
#include "lamina/core/weights.hpp"
#include "lamina/kernels/cpu/expert.hpp"
#include "lamina/kernels/cpu/pool.hpp"
#include "lamina/kernels/cpu/expert_layout.hpp"
#include "lamina/kernels/ngram.hpp"
#include "lamina/kernels/rope_scaling.hpp"
#include "lamina/kernels/s2_expert_grouped.hpp"
#include "lamina/kernels/sampler.hpp"
#include "lamina/kernels/shared_expert.hpp"
#include "lamina/kernels/iq_kernels.hpp"
#include "lamina/kernels/native_mmvq.hpp"
#include "lamina/kernels/mrope.hpp"
#include "lamina/kernels/qsa.hpp"
#include "lamina/kernels/qsa_select.hpp"
#include "lamina/kernels/verify_kernels.hpp"
#include "lamina/core/native_head.hpp"
#include "lamina/core/verify.hpp"
#include "lamina/core/mtp.hpp"
#include "lamina/core/prompt_cache.hpp"
#include "lamina/prefill/experts.hpp"
#include "lamina/prefill/prefill.hpp"
#include "lamina/prefill/split.hpp"
#include "lamina/kernels/f16_bits.hpp"
#include "lamina/spec/suffix_drafter.hpp"
#include "lamina/core/native_dense.hpp"
#include "lamina/core/pack_image.hpp"
#include "lamina/kernels/cpu/hadamard.hpp"
#include "lamina/platform/console_quit.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <array>

using lamina::program::serve_print;   // the --serve protocol's replies (serve_io.hpp)

namespace {

/// The chunk size a batched prompt of `len` tokens reads in: as few chunks of at most `cap` tokens as it takes, of one
/// size (a multiple of 256, the last one shorter), not whole `cap` chunks and the rest.  Every chunk streams each
/// expert its tokens use, so a short last one costs nearly a whole one's copies, and under --prompt-split it leaves the
/// pipeline's last stage alone (UD-Q4_K_XL at 20K tokens: 4.33 -> 4.01 s).
int64_t even_chunk(int64_t len, int64_t cap) {
    if (len <= cap) return len;
    const int64_t k = (len + cap - 1) / cap;
    return std::min(cap, ((len + k - 1) / k + 255) / 256 * 256);
}
// A prompt's chunk: of one size where a second GPU takes the experts the first lacks (a short last chunk streams
// nearly every expert of every layer, and under --prompt-split runs its second stage alone after the first's); whole
// chunks on one GPU, whose copy-ahead area grows with a chunk's tokens (UD-Q4_K_XL's 64K prompt on the 3090 alone
// 45.6 s in whole chunks, 52.4 in chunks of one size)
int64_t prompt_chunk(int64_t len, int64_t cap, bool second_gpu) {
    return second_gpu ? even_chunk(len, cap) : std::min(len, cap);
}

using Clock = std::chrono::steady_clock;

struct Options {
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    std::string ple_gguf;              // the model shard holding the PLE table (default: found among --native's)
    /// rope scaling past the trained 262K (upstream main's knobs, through the CLI: the model files carry no
    /// rope.scaling keys).  A process constant, resolved before session_init and any graph capture.
    std::string rope_scaling;         ///< --rope-scaling none|linear|yarn
    double rope_scale = 0;            ///< --rope-scale F (0 = absent; >= 1 when given)
    double rope_freq_base = 0;        ///< --rope-freq-base N (0 = 1e7)
    double rope_freq_scale = 0;       ///< --rope-freq-scale F (0 = 1/factor)
    double yarn_orig_ctx = 0;         ///< --yarn-orig-ctx N (0 = 262144)
    double yarn_ext_factor = -1.0;    ///< --yarn-ext-factor F (< 0 = auto: 1 for yarn, else 0)
    double yarn_attn_factor = 1.0;    ///< --yarn-attn-factor F
    double yarn_beta_fast = 32.0;     ///< --yarn-beta-fast F
    double yarn_beta_slow = 1.0;      ///< --yarn-beta-slow F
    bool disable_hit = false;          // every routed expert on the CPU (exact's reference)
    std::string expert_math = "fast";  // the GPU's native expert arithmetic: fast (llama.cpp q8_1) or exact (CPU ggml)
    /// the model's GGUF (its first shard: a split model's others are found by name): the quantized tensors,
    /// the head, the token embedding and the PLE table come from it, the float tensors from the pack computed
    /// from it at load
    std::string native_preset;
    std::vector<std::string> native_dense_gguf;   ///< its shards, and the PLE table's (set from it)
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default) or int8 (half the VRAM)
    /// P0.S8's routing trace, and a prerequisite the Phase 3 plan names explicitly.  One record per layer per
    /// position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is what a hit-rate curve for a
    /// candidate VRAM expert cache is computed from, and it needs no new kernels - the doorbell already
    /// publishes exactly this much to pinned memory.
    std::string dump_routing;
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    /// the VRAM tier's slots (--expert-cache: 0 = the profile's count, -1 = auto, sized from the free VRAM)
    int expert_cache = 0;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.
    std::string expert_profile;
    /// #477 (--serve): where to save what the run learned, as a profile `--expert-profile` reads (the routing counted
    /// since the start blended with the profile it started from: blend_learned_profile); at QUIT and, once
    /// `expert_profile_save_min` minutes have passed since the last save, between requests (0: at QUIT only).
    /// Empty (the default): nothing is counted or written.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    bool stats = false;
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// Plan v0.3 P5: the prompt path borrows the last expert-cache slots of each GPU for its buffers and refills
    /// them after the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session.
    bool no_prefill_borrow = false;
    /// With --second-gpu, the prompt path's experts the main GPU's cache does not hold run on the second GPU
    /// (default); `--no-prompt-offload` streams them to the main GPU instead.
    bool no_prompt_offload = false;
    /// verify windows of up to this many tokens: the last emitted one and spec - 1 drafts
    int spec = 0;
    /// Benchmarks: the run emits this continuation (token ids) instead of the window's argmax, and a draft is
    /// accepted when it matches it, so runs with different speculation settings process the same text.
    std::string spec_follow;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// the CUDA device (of the visible ones) the engine runs on: --main-gpu names it as find_gpu (gpu_select.hpp) reads
    /// it, an ordinal, a UUID, a PCI bus id or a name
    std::string main_gpu_name = "0";
    int main_gpu = 0;
    /// GPUs as further expert tiers, in order (--second-gpu A[,B,C], each named as --main-gpu is, or "auto": every
    /// other visible GPU, the fastest link first): `second_gpu` is the first, which also takes the prompt path's
    /// share, the head's rows and --prompt-split's second stage (-1: none).  Per tier, its VRAM for experts (0 = all
    /// but its reserve), its reserve and its prefetch budget: a list's last value stands for the tiers past it
    /// (tier_value).
    std::vector<std::string> tier_names;
    std::vector<int> tier_gpus;
    int second_gpu = -1;
    std::vector<double> second_gpu_gib{0.0};
    /// default (empty): 2048 MiB on a GPU the desktop or another graphics program uses, 512 on a headless one
    /// (gpu_select.hpp's probe)
    std::vector<double> second_gpu_reserve_mib;
    double second_gpu_min_mb = 4.0;   ///< a layer's misses from which the tiers take their shares (smaller: the CPU is quicker)
    /// per layer, the likeliest experts no GPU holds copied ahead in the RAM's idle time (0 = off).  Default (empty):
    /// 12 MiB to a GPU whose link the probe measured at 20 GB/s or more (the 5070 Ti's x16, 46-55 GB/s), none to a
    /// slower one: 4 MiB into the 5060 Ti's x4 (~7 GB/s) made UD-Q4_K_XL's code and after-16K text 6% slower
    std::vector<double> second_gpu_prefetch_mb;
    /// Keep the GPU tiers resident and executing, but do not adapt their residency while the main tier adapts.
    /// The default remains coupled adaptation of every tier.
    bool no_second_gpu_adapt = false;
    /// Fixed epoch-boundary adaptive copies and total tie ordering; opt-in, preserving async adaptation by default.
    bool deterministic_adapt = false;
    /// The profile's hottest experts, up to this many GiB, live only in the main GPU's VRAM (its lowest slots, never
    /// evicted, lent to the prompt path or given to the K/V) and the RAM arena leaves them out; `second_gpu_pin_gib`
    /// the next ones in the second GPU's.  A model whose experts outgrow the RAM then fits (a 119.5 GiB Q8_0 on a
    /// 125.7 GiB machine).
    double vram_pin_gib = 0.0;
    double second_gpu_pin_gib = 0.0;
    /// The arena's blobs placed by the RAM's measured read rate once the GPUs' experts are known
    /// (ArenaExpertSource::place): for RAM with slower regions, off by default.
    bool arena_place = false;
    bool no_file_cache = false;
    /// The prompt path's share of the experts the main GPU holds that the second GPU computes (Prefill::set_shift).
    double prompt_shift = 0.0;
    /// The prompt path in two stages (prefill/split.hpp): the second GPU runs each chunk's layers from this one on
    /// while the main GPU runs the next chunk (0: off).  Prompts of more than one chunk.
    double prompt_split = 0.0;
    /// The elastic K/V (--kv-grow, LAMINA_KV_GROW=1): the K/V takes VRAM as the context grows, the expert cache the rest.
    bool kv_grow = false;
    bool no_kv_grow = false;   ///< --no-kv-grow: off whatever LAMINA_KV_GROW says
    bool hc_q8 = false;   ///< the hyper-connection matrices only in 8 bits (core::HcQ8)
    double head_split = 0.45;                ///< the second GPU's share of the head's rows (0: all on the main GPU)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    /// Prompt lookup: when the text's last N+ tokens occurred before and the tokens after that occurrence are more
    /// drafts than --spec gives the draft layer, the next window verifies up to kVerifyMaxT - 1 of them instead;
    /// 0 = off.
    int spec_lookup = 0;
    /// The verify windows' capacity: --spec, or the kernels' maximum with prompt lookup.
    int max_window() const { return spec_lookup > 0 ? std::max(spec, lamina::kernels::kVerifyMaxT) : spec; }
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: exact, ~7% slower)
    bool window_profile = false;   ///< GPU timestamps between the verify window's stages, printed by --stats
    std::string window_hashes;     ///< per verify window: a hash of its final residual rows and its argmaxes
    std::string window_logits;     ///< per emitted position: the 64 likeliest tokens and their log-probabilities
    std::string dump_draft_data;   ///< --serve: each request's positions for the draft layer's training (DraftDump)
    bool prefill_profile = false;  ///< GPU time per section of the batched prompt path
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    /// --serve prompt cache: checkpoints of the sequence state each conversation keeps in host memory (0 = every
    /// request starts from an empty sequence).  A request continues from the longest prefix the session, one of
    /// the live conversation's checkpoints or a parked conversation holds.
    int prompt_cache = 16;
    /// --conversation-cache-mib: host memory for parked conversations (their K/V cells and their checkpoints), in
    /// MiB (0 = none parked).  A parked conversation leaves only whole, the least recently used first.
    int64_t conversation_cache_mib = 16384;
    /// --conversation-cache-slots: at most this many parked conversations (0 = as many as the memory holds).
    int conversation_cache_slots = 0;
    /// --conversation-cache-min-tokens: park only conversations of at least this many tokens.
    int64_t conversation_cache_min_tokens = 1024;
    /// Checkpoint interval inside long prompts, in whole prompt chunks (counted from the request's first new cell).
    int64_t prompt_cache_every = 16384;
    /// Prompt parts of up to this many tokens go through verify windows instead of the batched prompt path: a new
    /// part that short, and a last prompt chunk that short.  A batched call has a fixed cost (it lends and refills
    /// expert-cache slots and reads every expert its tokens use): ~0.55 s with a second GPU, ~1.6 s with IQ3_XXS on
    /// the 3090 alone; verify windows of 8 tokens (--spec-lookup) cost 5-8 ms per token.  They break even at ~73
    /// (UD-Q4_K_XL) and ~113 (IQ3_XXS) tokens with a second GPU, ~400 without (bench/feed_test.py).
    int64_t feed_max = 128;
    /// --serve (upstream #458): the role token of a short turn the server puts right before the new assistant turn
    /// (the "system" of a trailing reasoning-effort turn, with the config's "effort_position": "end").  When the turn
    /// before the last <|im_start|> opens with it, the turn checkpoint goes in front of that turn, so it holds the
    /// conversation and not the effort.  -1: off.
    int64_t tail_role_token = -1;
    int adapt_swaps = 96;
    float adapt_decay = 0.7f;   ///< --adapt-decay: the routing counts kept after each update (upstream's option)
};

/// A per-tier option's value for tier `t` (--second-gpu's order): its list's t-th, the last one for the tiers past it.
double tier_value(const std::vector<double>& v, size_t t) { return v.empty() ? 0.0 : v[std::min(t, v.size() - 1)]; }

/// "A,B,C" as numbers; false on anything else.
bool number_list(const char* s, std::vector<double>& out) {
    out.clear();
    for (;;) {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s) return false;
        out.push_back(v);
        if (*end == '\0') return true;
        if (*end != ',') return false;
        s = end + 1;
    }
}

void usage() {
    std::fprintf(stderr,
                 "lamina --native MODEL.gguf --spec T --prefill N --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --tokens LIST       the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --ple-gguf PATH      the PLE table: the model shard holding per_layer_token_embd (with --native,\n"
                 "                       the --native model's shard that holds it by default)\n"
                 "  --ple-io direct|mmap|ram  n-gram table reads. direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm; ram:\n"
                 "                       read into RAM once the experts are loaded (28.8 GB), direct reads until then\n"
                 "  --kv fp16|int8|k8v4|q4_0  KV storage: int8 codes + fp16 scale per 64 values (half FP16's VRAM);\n"
                 "                       k8v4: int8 K, 4-bit V of rotated rows (23%% less than int8); q4_0: 4-bit K\n"
                 "                       and V (45%% less; native packs only for both); default fp16. Qwen3.6-35B-A3B\n"
                 "                       and Qwen3.8-27B: fp16 or q8_0 (int8 codes + fp16 scale per 32, about half)\n"
                 "  --rope-scaling T      rotate for a context past the trained one: none, linear (position\n"
                 "                        interpolation) or yarn, as llama.cpp does. Fixed at startup: K in the\n"
                 "                        cache is post-RoPE.\n"
                 "  --rope-scale F        the extension factor (>= 1), e.g. 2 for 524288 over a 262144 training run\n"
                 "  --rope-freq-base N    the rotation base (0 = 1e7)   --rope-freq-scale F  the raw shrink (0 = 1/F)\n"
                 "  --yarn-orig-ctx N     trained context (0 = 262144)  --yarn-ext-factor F  < 0 = auto (1 for yarn)\n"
                 "  --yarn-attn-factor F  --yarn-beta-fast F  --yarn-beta-slow F  the remaining YaRN knobs\n"
                 "  --native MODEL.gguf  the model's GGUF (its first shard; a split model's others are found by\n"
                 "                       name): the quantized tensors, the head, the embedding and the PLE table\n"
                 "                       come from it, and the pack the engine computes from it at load (required)\n"
                 "  --expert-math M      native packs: how the GPU computes the experts it holds. fast (default):\n"
                 "                       llama.cpp's CUDA q8_1 arithmetic. exact (K-quant files): the CPU pool's ggml\n"
                 "                       arithmetic (Q8_K/Q8_0 activations, ggml's x86 summation order, one expf on\n"
                 "                       both sides), so a GPU hit is bit-identical to the CPU's and the output does\n"
                 "                       not depend on which experts are resident (a prompt's experts stay on the\n"
                 "                       main GPU); slower, for regression tests\n"
                 "  --disable-hit        every routed expert on the CPU (the cache stays allocated): exact's reference\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --stop-eos           stop at an end-of-turn token (<|endoftext|>, <|im_end|>; --serve always does)\n"
                 "  --eos-ids LIST       the end-of-turn tokens instead of 248044,248046 (implies --stop-eos)\n"
                 "  --max-context N      KV/state capacity (default 4096; at most the QSA selection's capacity, set\n"
                 "                       by the GPU's shared memory: 1,466,368 cells on 99 KiB, 630,784 on 48 KiB)\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position\n"
                 "  --serve              stay resident and take requests on stdin, as serve/server.py sends them\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from lamina-vision)\n"
                 "  --prompt-cache N     --serve: keep up to N checkpoints of the sequence state per conversation in\n"
                 "                       host memory and continue each request from the longest cached prefix\n"
                 "                       (default 16; 0 = every request starts from an empty sequence)\n"
                 "  --prompt-cache-every N  checkpoint interval inside long prompts, rounded down to whole --prefill\n"
                 "                       chunks (default 16384)\n"
                 "  --conversation-cache-mib MIB  --serve: host memory for conversations a request replaced (their\n"
                 "                       K/V cells and checkpoints), so switching back to one continues where it\n"
                 "                       was; the least recently used leaves whole first (default 16384; 0 = none)\n"
                 "  --conversation-cache-slots N  --serve: at most N such conversations (default 0: no limit)\n"
                 "  --conversation-cache-min-tokens N  --serve: keep only conversations of at least N tokens\n"
                 "                       (default 1024)\n"
                 "  --feed-max N         --serve: new prompt parts and last prompt chunks of up to N tokens run\n"
                 "                       through verify windows instead of the batched prompt path (default 128)\n"
                 "  --tail-role-token ID --serve: a turn of this role right before the last turn stays out of the\n"
                 "                       turn checkpoint (a trailing reasoning-effort turn; default -1 = off)\n"
                 "  --spec T             verify windows of up to T tokens: the last one and T-1 MTP drafts (2..8)\n"
                 "  --spec-min-p P       a draft enters the window only while it and the drafts before it have at\n"
                 "                       least probability P under the draft layer (default 0: always T-1 drafts)\n"
                 "  --spec-lookup N      when the text's last N+ tokens occurred before, the next window verifies\n"
                 "                       up to 7 of the tokens that followed there instead, if they are more than\n"
                 "                       --spec allows the draft layer (windows of up to 8 tokens; 0 = off)\n"
                 "  --spec-split         a window's tokens in two groups, the GPU's work on one beside the CPU's on\n"
                 "                       the other (exact, ~7%% slower: a measurement arm)\n"
                 "  --mtp DIR            the MTP draft layer's runtime files (tools/mtp_rt.py). Qwen3.6-35B-A3B and\n"
                 "                       Qwen3.8-27B: --mtp N, at most N drafts of the file's own MTP block a step\n"
                 "                       (--spec-min-p as above), --draft-vocab N its head over the first N ids\n"
                 "  --mtp-window N       the draft layer attends to the last N cells (default 32768; 0 = every cell)\n"
                 "  --main-gpu GPU       the GPU to run on (default 0): a CUDA device number of the visible ones (in\n"
                 "                       nvidia-smi's order unless CUDA_DEVICE_ORDER says otherwise), a UUID or its\n"
                 "                       start (GPU-f26e9ba6), a PCI bus id (10:00.0) or a name or a part of one\n"
                 "                       (\"RTX 3090\") that only one GPU has: a number moves when a card is added\n"
                 "  --second-gpu G[,G,G] GPUs, named as --main-gpu, as further expert tiers, up to three, in order:\n"
                 "                       each holds the profile's next experts, then the conversation's, and\n"
                 "                       computes them beside the CPU pool (native packs, verify windows); the first\n"
                 "                       also takes the prompt path's share, the head's rows and --prompt-split's\n"
                 "                       stage. One no visible GPU answers to is left out. auto: every other GPU,\n"
                 "                       the fastest link first (a short probe at startup: host-to-device rate x\n"
                 "                       multiprocessors) (--tier-gpus: the same; -1: none). Qwen3.6-35B-A3B and\n"
                 "                       Qwen3.8-27B: the GPUs a layer split may add, in order (auto: CUDA's), the\n"
                 "                       fewest that hold the model (Qwen3.6 where none do: the main GPU alone, the\n"
                 "                       CPU computing the experts it lacks); --layer-split N,N,.. sets the counts\n"
                 "  --second-gpu-gib G[,..]  each tier's VRAM for experts (default: all but its reserve; a list's\n"
                 "                       last value stands for the tiers past it, as in the two below)\n"
                 "  --second-gpu-reserve-mib R[,..]  the VRAM each tier leaves free (default 2048 on a GPU the\n"
                 "                       desktop or another graphics program uses, 512 on a headless one)\n"
                 "  --second-gpu-min-mb M  a layer's missed experts from which the tiers take their shares (default 4;\n"
                 "                       below it the CPU is quicker than their ~100 us round trip)\n"
                 "  --second-gpu-prefetch-mb M[,..]  per layer, copy its likeliest experts that no GPU holds (the next\n"
                 "                       layer's router on this layer's input), up to M MiB, to a tier while the RAM is\n"
                 "                       idle, the first tier's budget filled first (default 12 to a GPU whose link\n"
                 "                       the probe measured at 20 GB/s or more, ~0.28 ms at the 5070 Ti's ~48 GB/s;\n"
                 "                       none to a slower one; 0 = off)\n"
                 "  --no-second-gpu-adapt  with --second-gpu: keep the tiers\' resident expert assignment static while\n"
                 "                       the main GPU\'s adaptive tier continues to replace experts\n"
                 "  --deterministic-adapt  fixed-epoch adaptive copies and stable equal-gain ordering; default async\n"
                 "                       adaptation remains unchanged (with --second-gpu, applies to every tier)\n"
                 "  --adapt-every N      every N verify rounds the adaptive tiers move the conversation's most-routed\n"
                 "                       experts they lack in, in place of their least-routed ones (default 4;\n"
                 "                       0 = the start's placement stays)\n"
                 "  --adapt-swaps N      an update's moves at most (default 96)\n"
                 "  --adapt-decay F      the share of the routing counts the adaptive tiers keep after each update\n"
                 "                       (default 0.7; between 0 and 1)\n"
                 "  --vram-pin-gib G     the profile's hottest experts, up to G GiB, live only in the main GPU's\n"
                 "                       VRAM: its lowest slots keep them for good and the RAM arena leaves them out\n"
                 "                       (experts that outgrow the RAM then fit)\n"
                 "  --second-gpu-pin-gib G  the next ones, up to G GiB, only in the second GPU's VRAM\n"
                 "  --prompt-shift F     with --second-gpu: in prompt chunks of 8K tokens and more the second GPU also\n"
                 "                       computes the likeliest F (0..1) of each layer's experts the main GPU holds, copied\n"
                 "                       ahead with the ones it streams (when the main GPU's share is the longer; default 0)\n"
                 "  --prompt-split L     with --second-gpu: a prompt of more than one chunk runs in two stages, the\n"
                 "                       second GPU running each chunk's layers L.. whole (their weights and state\n"
                 "                       copied into its lent cache slots) while the main GPU runs the next chunk\n"
                 "                       through the first ones (0: off, the default)\n"
                 "  --arena-place        at the start, time each expert's place in the RAM arena and move the experts\n"
                 "                       no GPU holds out of the slow places, into the GPU-resident ones' (RAM with\n"
                 "                       slower regions, as unpaired DIMMs make; ~2-4 s).  Off by default\n"
                 "  --no-file-cache      the experts read from the model's GGUF past the OS file cache, which then\n"
                 "                       keeps no second copy of them on RAM the next starts' arenas would have had\n"
                 "                       (where the arena and the cached file fill the faster RAM); each start reads\n"
                 "                       them from the drive.  Off by default\n"
                 "  --kv-grow            with --serve: the K/V takes VRAM only for the cells the requests\n"
                 "                       reach and the expert cache holds the rest, giving slots up as a conversation\n"
                 "                       grows and taking them back after (LAMINA_KV_GROW=1/0 too).  Default: the\n"
                 "                       whole --max-context allocated at start\n"
                 "  --no-kv-grow         the whole --max-context allocated at start, whatever LAMINA_KV_GROW says\n"
                 "  --hc-q8              with --native: the hyper-connection projections only in 8 bits (int8, an\n"
                 "                       f16 scale a 32: the GGUF's Q8_0 as stored, BF16 ones rounded so); the verify\n"
                 "                       windows read them, the prompt path takes them in BF16 a layer at a time\n"
                 "  --head-split F       with --second-gpu: its share of the output head's rows (default 0.45;\n"
                 "                       each GPU streams its part after the last layer; 0 = all on the main GPU)\n"
                 "  --no-prompt-offload  with --second-gpu: the prompt path streams the experts the main GPU's cache\n"
                 "                       lacks to the main GPU instead of computing them on the second\n"
                 "  --spec-follow PATH   benchmarks: emit this continuation (token ids) instead of the argmax and\n"
                 "                       accept the drafts that match it, so speculation settings compare on the\n"
                 "                       same text (not --serve)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native; auto: 16384)\n"
                 "  --no-prefill-borrow  the prompt path's buffers keep VRAM of their own for the whole session\n"
                 "                       instead of borrowing the expert caches' last slots while a prompt runs\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --window-hashes PATH per verify window, a line with its round, first position and size, a 64-bit hash\n"
                 "                       of its final residual rows and its argmaxes: two builds that do the same\n"
                 "                       arithmetic write the same file (fixed text through --spec-follow,\n"
                 "                       --adapt-every 0; not --serve)\n"
                 "  --window-logits PATH per token the verify windows emit, a line with its index among the generated\n"
                 "                       tokens, the token and the 64 likeliest as id:log-probability, from the\n"
                 "                       head's logits (comparisons with llama.cpp, bench/parity.py; not --serve)\n"
                 "  --dump-draft-data DIR  with --serve: each request's positions for the draft layer's training: the\n"
                 "                       final residual rows, and the verify windows' 64 likeliest next tokens and\n"
                 "                       drafts (scratch/bg; the prompt read in batches whatever --feed-max)\n"
                 "  --window-profile     with --stats: the GPU time of each stage of the verify window (timestamps\n"
                 "                       inside the graph; each costs ~2 us)\n"
                 "  --prefill-profile    the GPU time of each section of the batched prompt path (not --serve)\n"
                 "  --expert-profile P   fill the VRAM tiers from this profile (data/expert-profile.bin, or what\n"
                 "                       --expert-profile-save wrote) instead of admitting experts on first use\n"
                 "  --expert-profile-save P  --serve: save what the run learned (the routing counted since\n"
                 "                       the start, blended with the profile it started from) as a profile at P,\n"
                 "                       at QUIT (or Ctrl+C) and between requests once --expert-profile-save-every\n"
                 "                       MIN minutes have passed since the last save (default 10; 0 = at QUIT\n"
                 "                       only); start from it with --expert-profile P\n"
                 "  --expert-cache N|auto  the main GPU's VRAM tier, in expert slots (auto: sized from the free\n"
                 "                       VRAM, leaving --vram-reserve-mib; 0: the profile's count)\n"
                 "  --vram-reserve-mib M the main GPU's VRAM --expert-cache auto leaves free (default 700)\n"
                 "  --pcie-frac F        in verify windows, the share of each layer's missed experts that the main\n"
                 "                       GPU reads from RAM over its link instead of the CPU computing them\n"
                 "                       (default 0.55; the GPU tiers take theirs first)\n"
                 "  --pcie-mode M        how that share reaches the main GPU: dma (default: the copy engine stages\n"
                 "                       it), kernel (a copy kernel stages it inside the window's graph) or direct\n"
                 "                       (the expert kernels read the mapped RAM)\n"
                 "  --pool-workers N     the CPU expert pool's workers (default 0: every physical core but the\n"
                 "                       host loop's)\n");
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

/// A serve request that failed on a CUDA fault (an illegal address) leaves at once (#224): the fault poisons the
/// context for the whole process, and the destructors run on it can hang until the server's watchdog.  An
/// out-of-memory (a window's graph that did not fit) poisons nothing: it is taken off the thread and the engine ends
/// the ordinary way, so the request's ERR reaches the client and the next request starts the engine again (#37).
/// cudaGetLastError() clears a sticky fault from the thread too, and one of work queued before the out-of-memory can
/// land unseen: cudaMemGetInfo asks the context, without waiting for its queued work, and a fault still leaves at once.
void exit_on_cuda_fault() {
    const cudaError_t e = cudaPeekAtLastError();
    if (e == cudaSuccess) return;
    if (e == cudaErrorMemoryAllocation) {
        (void) cudaGetLastError();
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) return;
    }
    std::fflush(stderr);
    std::_Exit(1);
}

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    lamina::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    int64_t records = 0;   // --dump-routing's
    /// THE ROUTING TRACE, which is P0.S8 and a stated prerequisite of Phase 3.  `drive_pool` is called once
    /// per layer from the main loop - the workers live inside `expert_pool_dispatch` - so a single FILE* here
    /// needs no locking.  `d.layers` is the CURRENT layer on entry (the adapter increments it as it walks the
    /// blob), which is why the layer index comes from there rather than from a counter of our own.
    std::FILE* routing = nullptr;
    /// The first GPU's adaptive tier: its copies cross that GPU's link, and while they ran they slowed the window's
    /// own transfers there (the router's rows and doorbell to the host, the CPU's rows to the combine) by 20-60 us a
    /// layer.  So they go out in pieces while the link has nothing else to carry: once a layer's CPU rows are in, as
    /// much as lands before the next layer's router writes to the host (that layer's running average of the time from
    /// the previous layer's CPU rows to its ring, by window size, less the time since and `kLinkReserveUs`).  A
    /// window's moves are staged before it (`drive_window`).  Without a second GPU the pieces cannot carry what the
    /// tier moves (the CPU takes every miss; pieces during the pool too slowed its RAM reads): `burst`, a window's
    /// start sends all the queued moves, and an update waits for the previous one's.
    lamina::core::AdaptiveTier* tier1 = nullptr;
    /// Where the host and the pool's workers run: checked before each window, once a second (CorePlacement::tick).
    lamina::kernels::cpu::CorePlacement* placement = nullptr;
    bool burst = false;
    std::vector<double> ring_us;   // by window size and layer
    Clock::time_point rows_at{};   // this window's last CPU rows (none yet: the epoch)
    int win_t = 0;                 // this window's tokens
    int64_t n_layers = 0;
    /// The GPU tiers' adaptive tiers, paced: their copies go between windows, where the RAM is idle (the head, the
    /// commit, the drafts); through the layers they slowed the CPU pool's RAM reads, and on the 5070 Ti's one copy engine
    /// they held up the prefetch copies.  An update is ranked at a window's tail, and each tier's copies go out as far
    /// as they end before the next window's first pool (`gap_ms`, a running average of that gap, at that tier's
    /// measured copy rate), the rest at the last tail before the next update.
    std::vector<lamina::core::AdaptiveTier*> paced;
    double gap_ms = 0;
    Clock::time_point tail_at{};                   // the last window's CPU rows done
    std::vector<Clock::time_point> copies_end;     // per paced tier: the gap's queued copies end
    bool gap_open = false;                       // no pool since the tail
    bool adapt_next = false;                     // an update is ranked at this window's tail
    bool flush_next = false;                     // the next window's tail ranks one: this gap takes the rest
    std::string tier_err;
    /// The adaptive tiers' update (ranking, the second tier's first copies) on a thread of its own, started at the
    /// window's tail and joined before the next window.
    std::function<bool()> adapt;
    std::thread adapt_thr;
    bool adapt_ok = true;
    void start_adapt() {
        adapt_ok = true;
        adapt_thr = std::thread([this] { adapt_ok = adapt(); });
    }
    bool join_adapt() {
        if (adapt_thr.joinable()) adapt_thr.join();
        return adapt_ok;
    }
    ~Drive() { join_adapt(); }
};

/// The second GPU's prefetch from a layer's predicted routing (Verifier::set_predict).
void drive_predict(void* user, int64_t layer, const int32_t* ids, const float* w, int64_t n_tok, int64_t k) {
    Drive* t = (Drive*) user;
    lamina::core::expert_prefetch_multi(t->d, layer, ids, w, n_tok, k);
}

// The first tier's piece ends this long before the next ring: the router (20-30 us), the doorbell, and the copy's
// start ~20 us after the call under WDDM (50 and 30 us were slower; pieces between windows too, slower again).
constexpr double kLinkReserveUs = 80;

/// The first tier's copy rate in bytes per ms (a first guess until measured).
double tier1_rate(const Drive* t) {
    const double r = t->tier1->copy_rate();
    return r > 0 ? r : 5e6;
}

/// Before a window of `T` tokens: the tiers' landed moves admitted, and the first tier's next moves staged for what
/// this window's layers take (their residents leave before the window's residency snapshot).
void drive_window(Drive* t, int T) {
    t->win_t = T;
    t->rows_at = Clock::time_point{};
    if (t->placement != nullptr && t->d.pool != nullptr) {
        const std::string moved = t->placement->tick(*t->d.pool);
        if (!moved.empty()) std::fprintf(stderr, "lamina: %s\n", moved.c_str());
    }
    for (lamina::core::AdaptiveTier* p : t->paced) p->apply_pending(false);
    if (t->tier1 == nullptr) return;
    t->tier1->apply_pending(false);
    if (t->burst) {   // all of them, at once
        t->tier1->stage(UINT64_MAX);
        if (!t->tier1->pump_bytes(UINT64_MAX, t->tier_err)) {
            t->d.failed = true;
            t->d.fail = t->tier_err.c_str();
        }
        return;
    }
    double us = 0;
    for (int64_t l = 1; l < t->n_layers; ++l)
        us += std::max(0.0, t->ring_us[(size_t) (T * t->n_layers + l)] - kLinkReserveUs);
    t->tier1->stage((uint64_t) (1.25 * tier1_rate(t) * us / 1e3));
}

/// A layer's CPU rows are in (Verifier::set_rows_in): the first tier's piece that lands before the next router.
void drive_rows_in(void* user, int64_t layer) {
    Drive* t = (Drive*) user;
    if (t->tier1 == nullptr || t->burst || t->d.failed || layer + 1 >= t->n_layers) return;
    const double us = t->ring_us[(size_t) (t->win_t * t->n_layers + layer + 1)] - kLinkReserveUs -
                      std::chrono::duration<double, std::micro>(Clock::now() - t->rows_at).count();
    if (us > 0 && !t->tier1->pump_bytes((uint64_t) (tier1_rate(t) * us / 1e3), t->tier_err)) {
        t->d.failed = true;
        t->d.fail = t->tier_err.c_str();
    }
}

/// The GPU tiers' queued moves that end before the next window's first pool, or all of them; each tier on its own GPU,
/// link and copy engine, at its own rate.
bool pump_gap(Drive* t, std::string& err, bool all = false) {
    for (size_t i = 0; i < t->paced.size(); ++i) {
        lamina::core::AdaptiveTier* p = t->paced[i];
        Clock::time_point& end = t->copies_end[i];
        const Clock::time_point start = std::max(Clock::now(), end);
        // 1.3 of the gap: the moves may reach into the next window's first pool, as many as the gap's own length is
        // worth (at 1.0 the GDN commit's state-only replay, 0.37 ms shorter, cost UD-Q4_K_XL edit text 4.5% in moves
        // left out)
        constexpr double kGapScale = 1.3;
        const double left = kGapScale * t->gap_ms - std::chrono::duration<double, std::milli>(start - t->tail_at).count();
        const double rate = p->copy_rate();
        const bool paced = !all && rate > 0 && t->gap_ms > 0;
        if (paced && left <= 0) continue;
        uint64_t sent = 0;
        if (!p->pump(paced ? (uint64_t) (left * rate) : UINT64_MAX, sent, err)) return false;
        if (rate > 0)
            end = start + std::chrono::duration_cast<Clock::duration>(
                              std::chrono::duration<double, std::milli>((double) sent / rate));
    }
    return true;
}

/// Whether the GPU tiers' work so far succeeded (Verifier::set_watch).
bool drive_watch(void* user, std::string& err) {
    Drive* t = (Drive*) user;
    for (int i = 0; i < t->d.n_tiers; ++i)
        if (!t->d.tiers[i].gpu->healthy(err)) return false;
    return true;
}

/// A window's last CPU rows are in (Verifier::set_tail).
void drive_tail(void* user) {
    Drive* t = (Drive*) user;
    t->tail_at = Clock::now();
    for (Clock::time_point& e : t->copies_end) e = t->tail_at;
    t->gap_open = true;
    if (t->d.failed) return;
    if (t->adapt_next) {
        t->start_adapt();
    } else if (!pump_gap(t, t->tier_err, t->flush_next)) {
        t->d.failed = true;
        t->d.fail = t->tier_err.c_str();
    }
}

/// Plan v0.3 P6: the pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_tok, int64_t k,
                      float* out, int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    const size_t at = (size_t) (t->win_t * t->n_layers + std::min(layer, t->n_layers - 1));
    if (t->rows_at != Clock::time_point{}) {   // the last CPU rows to this ring
        const double us = std::chrono::duration<double, std::micro>(a - t->rows_at).count();
        t->ring_us[at] = t->ring_us[at] > 0 ? 0.8 * t->ring_us[at] + 0.2 * us : us;
    }
    if (t->gap_open) {   // the first pool after a window: the RAM's idle time between them
        t->gap_open = false;
        const double gap = std::chrono::duration<double, std::milli>(a - t->tail_at).count();
        if (gap < 20.0) t->gap_ms = t->gap_ms > 0 ? 0.8 * t->gap_ms + 0.2 * gap : gap;
    }
    lamina::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->rows_at = Clock::now();
    t->cpu_ms += std::chrono::duration<double, std::milli>(t->rows_at - a).count();
    ++t->calls;
    // the routing trace (--dump-routing): per row of the window, drafts too, int32 layer, int32 k, k int32 ids and k
    // float weights
    if (t->routing != nullptr)
        for (int64_t i = 0; i < n_tok; ++i) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + i * k, sizeof(int32_t), (size_t) k, t->routing);
            std::fwrite(weights + i * k, sizeof(float), (size_t) k, t->routing);
            ++t->records;
        }
}

/// LAMINA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized).
/// LAMINA_TRACE=2 also waits for the device at each mark and names the first step after which it reports an error:
/// a sticky failure that surfaces in a later, unrelated call is pinned to the step that ran the faulting work.
void mem_mark(const char* where) {
    static const char* const env = std::getenv("LAMINA_TRACE");
    if (env == nullptr) return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "lamina trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
    if (env[0] == '2') {
        const cudaError_t sync_e = cudaDeviceSynchronize();
        const cudaError_t last_e = cudaGetLastError();
        std::fprintf(stderr, "lamina trace: device %s after %s (sync: %s, last error: %s)\n",
                     sync_e == cudaSuccess && last_e == cudaSuccess ? "ok" : "ERROR", where, cudaGetErrorString(sync_e),
                     cudaGetErrorString(last_e));
    }
}

// The CUDA stack reservation: a context keeps cudaLimitStackSize bytes (1 KiB by default) for every thread its SMs
// can hold, 122 MiB on the 3090 and 104 MiB on the 5070 Ti.  The engine's kernels need at most kStackBytes (cuobjdump
// -res-usage of the build: llama.cpp's MMQ tiles, 104 bytes on sm_120 and 88 on sm_86, past a tuning-only instance;
// from a limit of 0, requests of both models raised it to 64 on either GPU, cuBLAS's GEMMs included), so each GPU gets
// that before the caches size themselves from the free VRAM.  Not 0: a kernel with a larger frame than the limit makes
// the driver grow the reservation at its first launch, mid-request (a resize can wait for the device's queued work).
constexpr size_t kStackBytes = 104;
static size_t g_stack_set[64] = {};   // the limit each GPU took (the driver rounds it up to 16 bytes)
static bool set_stack_limit(int dev, int main_dev) {
    cudaSetDevice(dev);
    const bool ok = cudaDeviceSetLimit(cudaLimitStackSize, kStackBytes) == cudaSuccess &&
                    (dev < 0 || dev >= 64 || cudaDeviceGetLimit(&g_stack_set[dev], cudaLimitStackSize) == cudaSuccess);
    cudaSetDevice(main_dev);
    return ok;
}
// Says once a GPU's limit was grown by a kernel's frame: raise kStackBytes past that kernel's (cuobjdump -res-usage).
static void check_stack_limit(int dev, int main_dev) {
    static bool warned[64] = {};
    cudaSetDevice(dev);
    size_t now = 0;
    if (dev >= 0 && dev < 64 && cudaDeviceGetLimit(&now, cudaLimitStackSize) == cudaSuccess && now > g_stack_set[dev] &&
        !warned[dev]) {
        warned[dev] = true;
        std::fprintf(stderr, "lamina: warning: a kernel grew GPU %d's CUDA stack limit from %zu to %zu "
                             "bytes a thread (the driver resizes it at that kernel's first launch)\n", dev,
                     g_stack_set[dev], now);
    }
    cudaSetDevice(main_dev);
}

// Uploads the expert residency table. Kernels on non-blocking streams read it, which do not wait for the legacy
// stream this copy runs on, and a copy from pageable memory can return before its DMA has landed: so it waits for it.
static cudaError_t res_put(int32_t* dst, const std::vector<int32_t>& res) {
    const cudaError_t e = cudaMemcpy(dst, res.data(), res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    return e != cudaSuccess ? e : cudaStreamSynchronize(cudaStreamLegacy);
}

/// Expert-cache slots lent to the batched prompt path for its buffers: the cache's last slots, as many as hold the
/// buffers, while at least 128 stay.  `lend` takes their experts out of the residency table (the prompt path
/// streams them instead), `give_back` copies them in again from the arena once the prompt is done.
struct SlotLoan {
    lamina::core::ExpertCache* cache = nullptr;
    std::vector<int32_t>* res = nullptr;
    int32_t* d_res = nullptr;   // the table's device copy, or null
    int device = 0;             // the cache's GPU; `main_device` is current again after `give_back`
    int main_device = 0;
    int32_t first = -1;         // the first lent slot, -1 when the cache cannot spare the bytes
    int64_t keep = 0;           // the slots [0, keep) are never lent (experts only in VRAM, --vram-pin-gib)
    std::vector<std::pair<int32_t, int32_t>> lent;   // (residency index, slot) while lent

    uint64_t offset(int64_t slot) const {
        return cache->slot_offsets() ? cache->slot_offsets()[slot]
                                     : (uint64_t) slot * (uint64_t) (cache->bytes() / cache->slots());
    }
    /// The last slots that hold `need` bytes; false when the cache cannot spare them.
    bool plan(uint64_t need) {
        int64_t k = 0;
        while (k < cache->slots() && (uint64_t) cache->bytes() - offset(cache->slots() - k) < need) ++k;
        first = k + 128 <= cache->slots() && cache->slots() - k >= keep ? (int32_t) (cache->slots() - k) : -1;
        return first >= 0;
    }
    void* base() const { return first >= 0 ? (void*) cache->device_slot(first) : nullptr; }
    uint64_t bytes() const { return first >= 0 ? (uint64_t) cache->bytes() - offset(first) : 0; }
    int64_t slots() const { return first >= 0 ? cache->slots() - first : 0; }
    void lend() {
        if (first < 0) return;
        std::vector<int32_t>& r = *res;
        for (size_t i = 0; i < r.size(); ++i)
            if (r[i] >= first) {
                lent.emplace_back((int32_t) i, r[i]);
                r[i] = lamina::core::kNotResident;
            }
        if (d_res != nullptr) res_put(d_res, r);
    }
    /// After the borrower's last use: its GPU work is waited for first.
    bool give_back(lamina::core::ExpertSource& src, int64_t n_expert, std::string& err) {
        if (lent.empty()) return true;
        cudaSetDevice(device);
        cudaDeviceSynchronize();
        bool ok = true;
        const auto& lay = lamina::kernels::cpu::expert_layout();
        for (const auto& [i, slot] : lent) {
            const int64_t l = i / n_expert;
            const uint8_t* b = src.blob(l, i % n_expert);
            if (b == nullptr || !cache->fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(l))) {
                if (b == nullptr) err = "no arena blob for a lent slot";
                ok = false;
                break;
            }
            (*res)[(size_t) i] = slot;
        }
        cudaSetDevice(main_device);
        if (ok && d_res != nullptr) res_put(d_res, *res);
        lent.clear();
        return ok;
    }
};

/// The prompt's counts into the tiers' usage, each layer's scaled to kPromptUsage: about a conversation's settled
/// counts a layer (~30 routed entries a round kept at 0.7 an update), so the answer's own rounds soon outweigh
/// them, while the prompt's most-routed experts pass the tiers' threshold (2) for their first updates.  (400 a
/// layer: the first answer on new content as fast on average, IQ3_S's slower.)
constexpr float kPromptUsage = 120.0f;
template <class T>
static void add_prompt_usage(std::vector<T>& usage, const std::vector<uint32_t>& routed, int64_t n_expert) {
    const float total = kPromptUsage;
    if (usage.size() != routed.size()) return;
    for (size_t l0 = 0; l0 < routed.size(); l0 += (size_t) n_expert) {
        double sum = 0;
        for (size_t e = 0; e < (size_t) n_expert; ++e) sum += routed[l0 + e];
        if (sum > 0)
            for (size_t e = 0; e < (size_t) n_expert; ++e) usage[l0 + e] += (T) (total * routed[l0 + e] / sum);
    }
}

/// What a loan's slots take back after a prompt, (layer, expert, slot) each: per layer the lent slots' own experts,
/// except that with `routed` (the prompt's counts) a missing expert routed clearly more often (twice as often and
/// twice more) takes the place of the least-routed of them.  `taken` (n_layers x n_expert): the experts another GPU
/// holds or takes back; this loan's are added to it.
static std::vector<std::array<int32_t, 3>> refill_plan(const SlotLoan& loan, const std::vector<uint32_t>* routed,
                                                       std::vector<uint8_t>& taken, int64_t n_expert,
                                                       lamina::core::ExpertSource& src) {
    std::vector<std::array<int32_t, 3>> out;
    std::vector<std::pair<int32_t, int32_t>> lent = loan.lent;   // (residency index, slot), by layer
    std::sort(lent.begin(), lent.end());
    std::vector<std::pair<uint32_t, int32_t>> vict, cand;
    for (size_t a = 0; a < lent.size();) {
        const int64_t l = lent[a].first / n_expert;
        size_t b = a;
        while (b < lent.size() && lent[b].first / n_expert == l) ++b;
        const uint32_t* r = routed ? routed->data() + l * n_expert : nullptr;
        vict.clear();
        cand.clear();
        for (size_t i = a; i < b; ++i) {
            const int32_t e = (int32_t) (lent[i].first % n_expert);
            vict.emplace_back(r ? r[e] : 0u, e);
        }
        if (r != nullptr) {
            for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
                const size_t i = (size_t) (l * n_expert + e);
                bool own = false;
                for (size_t k = a; k < b && !own; ++k) own = lent[k].first == (int32_t) i;
                if (!own && !taken[i] && (*loan.res)[i] < 0 && r[e] > 0 && src.in_ram(l, e))
                    cand.emplace_back(r[e], e);
            }
            std::sort(vict.begin(), vict.end());
            std::sort(cand.begin(), cand.end(), std::greater<>());   // the most routed first
            for (size_t k = 0; k < vict.size() && k < cand.size() && cand[k].first >= 2 * vict[k].first + 2; ++k)
                vict[k] = cand[k];
        }
        for (size_t k = 0; k < vict.size(); ++k) {
            out.push_back({(int32_t) l, vict[k].second, lent[a + k].second});
            taken[(size_t) (l * n_expert + vict[k].second)] = 1;
        }
        a = b;
    }
    return out;
}

/// --dump-draft-data DIR: a request's positions for the draft layer's training (scratch/bg), in three files, req<k>.*:
/// .r16 every position's final residual row (hc * n_embd FP16, the prompt's from the batched path), in order; .meta a
/// record a position (position, token, the next token or -1 for the prompt's, and from a verify window the 64
/// likeliest next tokens: ids, then log-probabilities); .rounds a record a window (its first position, T, the drafts
/// accepted, the next window's first three drafts and their probabilities under the draft layer).
struct DraftDump {
    static constexpr int TOP = 64;
    std::string dir;
    int k = 0;
    std::FILE *r16 = nullptr, *meta = nullptr, *rounds = nullptr;
    std::vector<float> rows, logits;
    std::vector<uint16_t> h16;
    std::vector<int32_t> idx;
    int64_t positions = 0, overflow = 0;
    bool on() const { return r16 != nullptr; }
    bool open(std::string& err) {
        close();
        const std::string base = dir + "/req" + std::to_string(k++);
        r16 = std::fopen((base + ".r16").c_str(), "wb");
        meta = std::fopen((base + ".meta").c_str(), "wb");
        rounds = std::fopen((base + ".rounds").c_str(), "wb");
        if (!r16 || !meta || !rounds) { err = "--dump-draft-data: cannot write " + base + ".*"; close(); return false; }
        return true;
    }
    void close() {
        for (std::FILE** f : {&r16, &meta, &rounds}) if (*f) { std::fclose(*f); *f = nullptr; }
    }
    /// `n` rows of `D` floats from the device.
    void put_rows(const float* dev, int64_t n, int64_t D) {
        rows.resize((size_t) (n * D));
        h16.resize(rows.size());
        cudaMemcpy(rows.data(), dev, rows.size() * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (std::fabs(rows[i]) > 65000.0f) ++overflow;
            h16[i] = lamina::kernels::f16_from_f32(rows[i]);
        }
        std::fwrite(h16.data(), 2, h16.size(), r16);
        positions += n;
    }
    /// A position's record; `row` (n_vocab logits) or null for none.
    void put_meta(int32_t pos, int32_t tok, int32_t next, const float* row, int64_t n_vocab) {
        int32_t head[4] = {pos, tok, next, row ? TOP : 0};
        int32_t ids[TOP] = {};
        float lp[TOP] = {};
        if (row != nullptr) {
            idx.resize((size_t) n_vocab);
            for (int64_t i = 0; i < n_vocab; ++i) idx[(size_t) i] = (int32_t) i;
            std::partial_sort(idx.begin(), idx.begin() + TOP, idx.end(),
                              [&](int32_t a, int32_t b) { return row[a] > row[b]; });
            double mx = row[idx[0]], s = 0;
            for (int64_t i = 0; i < n_vocab; ++i) s += std::exp((double) row[i] - mx);
            const double lse = mx + std::log(s);
            for (int j = 0; j < TOP; ++j) {
                ids[j] = idx[(size_t) j];
                lp[j] = (float) ((double) row[idx[(size_t) j]] - lse);
            }
        }
        std::fwrite(head, 4, 4, meta);
        std::fwrite(ids, 4, TOP, meta);
        std::fwrite(lp, 4, TOP, meta);
    }
    void put_round(int32_t p, int32_t T, int32_t a, const int32_t* d, const float* dp) {
        const int32_t h[3] = {p, T, a};
        std::fwrite(h, 4, 3, rounds);
        std::fwrite(d, 4, 3, rounds);
        std::fwrite(dp, 4, 3, rounds);
    }
};

/// The batched prompt path's device buffers on each GPU: expert-cache slots lent for every prompt, as many as its
/// chunk needs, else allocated once for the longest chunk.
struct PromptBuffers {
    lamina::prefill::Prefill* prefill = nullptr;
    lamina::prefill::ExpertRunner* offload = nullptr;   // the second GPU's share, or null
    lamina::prefill::PromptSplit* split = nullptr;      // --prompt-split, or null
    const lamina::core::ModelGeometry* g = nullptr;
    const lamina::core::SessionState* ss = nullptr;
    SlotLoan loan, loan2;   // each GPU's cache (a null cache lends nothing)
    bool lend = false, lend2 = false;
    void* split_region = nullptr;   // take's region of the second GPU's loan for the split (null: none this prompt)
    uint64_t split_bytes = 0;

    /// After the prompt path and its second GPU's runner have been initialized.
    bool init(int64_t max_chunk, bool borrow, std::string& err) {
        lend = borrow && loan.cache != nullptr && loan.res != nullptr && !loan.res->empty() &&
               loan.plan(lamina::prefill::Prefill::bytes_needed(*g, *ss, max_chunk, offload != nullptr));
        lend2 = offload != nullptr && borrow && loan2.cache != nullptr && loan2.res != nullptr && !loan2.res->empty() &&
                loan2.plan(lamina::prefill::ExpertRunner::bytes_needed(max_chunk, g->n_expert, true, true));
        return (lend || prefill->bind(nullptr, 0, max_chunk, err)) &&
               (offload == nullptr || lend2 || offload->bind(nullptr, 0, max_chunk, err));
    }
    /// Before a prompt whose chunks hold up to `chunk` tokens; `split_need`: the split's bytes on the second GPU too
    /// (PromptSplit::bytes_for), which `split_region` then holds when its cache could lend them.
    bool take(int64_t chunk, std::string& err, uint64_t split_need = 0) {
        // sized before either loan, from the residency both GPUs' prefetch areas are sized for
        const uint64_t need = lend ? prefill->bytes_for(chunk) : 0;
        const uint64_t need2 = lend2 ? offload->bytes_for(chunk, loan.res ? loan.res->data() : nullptr) : 0;
        split_region = nullptr;
        split_bytes = 0;
        if (lend) {
            loan.plan(need);
            loan.lend();
        }
        // no loan (none, or not this prompt's bytes): the runner's own buffers, carved for this chunk as a loan is,
        // so neither the last loan's slots, refilled since, nor another chunk size stays bound
        if (!prefill->bind(loan.base(), loan.bytes(), chunk, err)) return false;
        if (offload == nullptr) return true;
        uint64_t r2 = 0;
        bool with_split = false;
        if (lend2) {
            r2 = (need2 + 255) & ~255ull;
            with_split = split_need > 0 && loan2.plan(r2 + split_need);
            if (!with_split) loan2.plan(need2);
            loan2.lend();
        }
        if (!offload->bind(loan2.base(), with_split ? r2 : loan2.bytes(), chunk, err)) return false;
        if (with_split) {
            split_region = (uint8_t*) loan2.base() + r2;
            split_bytes = loan2.bytes() - r2;
        }
        return true;
    }
    /// After it: the lent slots refilled.
    bool give_back(lamina::core::ExpertSource& src, std::string& err) {
        return loan.give_back(src, g->n_expert, err) && loan2.give_back(src, g->n_expert, err);
    }
    /// After it, through the tiers of the loans' GPUs (null or off: that loan's blocking give_back): what the lent
    /// slots take back (refill_plan, `routed` the prompt's counts) goes out as their moves at once, each admitted as
    /// it lands while the first windows run (the blocking refill held a prompt up 0.12 s after 2K tokens, 0.31 s
    /// after 16K: the 3090's share over its x4 link).
    bool give_back_tiers(lamina::core::AdaptiveTier* t1, lamina::core::AdaptiveTier* t2,
                         const std::vector<uint32_t>* routed, lamina::core::ExpertSource& src, std::string& err) {
        std::vector<uint8_t> taken(loan.res ? loan.res->size() : loan2.res ? loan2.res->size() : 0, 0);
        for (const SlotLoan* l : {&loan, &loan2})
            if (l->res != nullptr)
                for (size_t i = 0; i < taken.size(); ++i) taken[i] |= (*l->res)[i] >= 0 ? 1 : 0;
        const std::pair<SlotLoan*, lamina::core::AdaptiveTier*> loans[2] = {{&loan, t1}, {&loan2, t2}};
        for (const auto& [l, t] : loans) {
            if (l->lent.empty()) continue;
            if (t == nullptr || !t->on()) {
                if (!l->give_back(src, g->n_expert, err)) return false;
                continue;
            }
            cudaSetDevice(l->device);
            cudaDeviceSynchronize();   // the prompt path's last use of the slots
            cudaSetDevice(l->main_device);
            const auto moves = refill_plan(*l, routed, taken, g->n_expert, src);
            l->lent.clear();
            if (!t->refill(moves, l == &loan ? (64ull << 20) : (256ull << 20), err)) return false;
        }
        return true;
    }
};

#if defined(LAMINA_HAVE_QWEN35)
constexpr int kNotQwen35 = -1;
/// The family switch: kNotQwen35 for a file this engine runs, else the exit code of the engine path of Qwen3.6-35B-A3B
/// and Qwen3.8-27B (src/qwen35), which qwen35moe and qwen35 files (--native's header) run on.  That path runs on
/// --main-gpu (named as for this engine), made current before its first CUDA call, and splits the layers over it and
/// --second-gpu's GPUs where one does not hold them; it gets the command line without either, the GPUs as --stage-gpus.
int qwen35_switch(int argc, char** argv) {
    std::string model, main_gpu = "0", second;
    lamina::ModelFamily family = lamina::ModelFamily::Unknown;
    std::vector<char*> rest{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--main-gpu") == 0 && i + 1 < argc) {
            main_gpu = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--second-gpu") == 0 && i + 1 < argc) {
            second = argv[++i];
            continue;
        }
        if ((std::strcmp(argv[i], "--native") == 0 || std::strcmp(argv[i], "--model") == 0) && i + 1 < argc)
            model = argv[i + 1];
        rest.push_back(argv[i]);
    }
    if (model.empty()) return kNotQwen35;
    try {
        const lamina::GgufFile meta(lamina::gguf_split_paths(model).front());
        family = lamina::model_family(meta);
        if (family != lamina::ModelFamily::Qwen35Moe && family != lamina::ModelFamily::Qwen35) return kNotQwen35;
    } catch (const std::exception&) {
        return kNotQwen35;   // not readable here: the engine below says why
    }
    const std::vector<lamina::core::GpuInfo> gpus = lamina::core::list_gpus();
    std::string why;
    const int dev = lamina::core::find_gpu(gpus, main_gpu, why);
    if (dev < 0 || cudaSetDevice(dev) != cudaSuccess) {
        std::fprintf(stderr, "lamina: --main-gpu %s: %s\n", main_gpu.c_str(), dev < 0 ? why.c_str() : "cannot use it");
        return 2;
    }
    // --second-gpu: the GPUs a layer split may add, in order (names or numbers, comma-separated, a name no visible GPU
    // answers to left out; auto: every other GPU in CUDA's order; -1: none); the engine takes the fewest that hold the
    // model
    std::vector<int> stages{dev};
    if (second == "auto") {
        for (const lamina::core::GpuInfo& gi : gpus)
            if (gi.ordinal != dev) stages.push_back(gi.ordinal);
    } else if (!second.empty() && second != "-1") {
        for (size_t at = 0; at <= second.size();) {
            const size_t comma = std::min(second.find(',', at), second.size());
            const std::string name = second.substr(at, comma - at);
            at = comma + 1;
            const int d = lamina::core::find_gpu(gpus, name, why);
            const bool twice = d >= 0 && std::find(stages.begin(), stages.end(), d) != stages.end();
            if (d < 0 || twice) {
                std::fprintf(stderr, "lamina: --second-gpu %s left out: %s\n", name.c_str(),
                             d < 0 ? why.c_str() : "the split has that GPU already");
                continue;
            }
            stages.push_back(d);
        }
    }
    std::string list, more, flag = "--stage-gpus";
    for (const int d : stages) {
        list += (list.empty() ? "" : ",") + std::to_string(d);
        if (d != dev) more += (more.empty() ? "" : ", ") + std::string("GPU ") + std::to_string(d) + ", " + gpus[(size_t) d].name;
    }
    if (stages.size() > 1) {
        rest.push_back(flag.data());
        rest.push_back(list.data());
    }
    std::fprintf(stderr, "lamina: %s (%s) on GPU %d, %s%s%s\n", lamina::model_family_title(family),
                 lamina::model_family_arch(family), dev, gpus[(size_t) dev].name.c_str(),
                 more.empty() ? "" : "; a layer split may add ", more.c_str());
    return lamina::qwen35::run((int) rest.size(), rest.data());
}
#endif

}  // namespace

int lamina::program::run(int argc, char** argv) {
    // device ordinals in nvidia-smi's order, not CUDA's fastest-first guess (before the first CUDA call)
    if (std::getenv("CUDA_DEVICE_ORDER") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
#else
        setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
#endif
    }
#if defined(LAMINA_HAVE_QWEN35)
    if (const int rc = qwen35_switch(argc, argv); rc != kNotQwen35) return rc;
#endif
    // The engine may run in the server's process (lamina-server), which starts it again after an unload: what it holds
    // outside the objects below (device buffers, the main stream, the image position tables) goes back at every
    // return, after those objects.
    struct AtExit {
        std::vector<std::function<void()>> undo;
        ~AtExit() {
            for (auto f = undo.rbegin(); f != undo.rend(); ++f) (*f)();
        }
    } at_exit;
    Options o;
    bool have_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        {   // switches outside the chain below: MSVC refuses one this long (C1061)
            bool* on = a == "--hc-q8" ? &o.hc_q8 : a == "--arena-place" ? &o.arena_place
                     : a == "--no-file-cache" ? &o.no_file_cache : nullptr;
            if (on != nullptr) {
                *on = true;
                continue;
            }
        }
        {   // the knobs, the texts and the lists below outside the chain as well
            std::string* text = a == "--expert-profile-save" ? &o.expert_profile_save
                              : a == "--dump-draft-data" ? &o.dump_draft_data
                              : a == "--rope-scaling" ? &o.rope_scaling : nullptr;
            if (text != nullptr) {
                *text = next(a.c_str());
                continue;
            }
            double* knob = a == "--prompt-shift" ? &o.prompt_shift : a == "--prompt-split" ? &o.prompt_split
                         : a == "--expert-profile-save-every" ? &o.expert_profile_save_min
                         : a == "--rope-scale" ? &o.rope_scale : a == "--rope-freq-base" ? &o.rope_freq_base
                         : a == "--rope-freq-scale" ? &o.rope_freq_scale : a == "--yarn-orig-ctx" ? &o.yarn_orig_ctx
                         : a == "--yarn-ext-factor" ? &o.yarn_ext_factor : a == "--yarn-attn-factor" ? &o.yarn_attn_factor
                         : a == "--yarn-beta-fast" ? &o.yarn_beta_fast : a == "--yarn-beta-slow" ? &o.yarn_beta_slow
                         : nullptr;
            if (knob != nullptr) {
                *knob = std::atof(next(a.c_str()));
                continue;
            }
            if (a == "--second-gpu" || a == "--tier-gpus") {   // names, as --main-gpu's (a negative number: none)
                o.tier_names.clear();
                std::stringstream names(next(a.c_str()));
                for (std::string s; std::getline(names, s, ',');) {
                    s.erase(0, s.find_first_not_of(" \t"));
                    s.erase(s.find_last_not_of(" \t") + 1);
                    if (!s.empty() && s[0] != '-') o.tier_names.push_back(s);
                }
                continue;
            }
            std::vector<double>* list = a == "--second-gpu-gib" ? &o.second_gpu_gib
                                      : a == "--second-gpu-reserve-mib" ? &o.second_gpu_reserve_mib
                                      : a == "--second-gpu-prefetch-mb" ? &o.second_gpu_prefetch_mb : nullptr;
            if (list != nullptr) {
                if (!number_list(next(a.c_str()), *list)) {
                    std::fprintf(stderr, "lamina: %s takes a number or a comma-separated list\n", a.c_str());
                    return 2;
                }
                continue;
            }
        }
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--tokens") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--disable-hit") o.disable_hit = true;
        else if (a == "--expert-math") o.expert_math = next("--expert-math");
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib"));
        else if (a == "--prefill") {   // auto: 16K-token chunks, halved until their buffers fit the lendable slots
            const std::string v = next("--prefill");
            o.prefill_chunk = v == "auto" ? 16384 : std::atoll(v.c_str());
        }
        else if (a == "--no-prefill-borrow") o.no_prefill_borrow = true;
        else if (a == "--no-prompt-offload") o.no_prompt_offload = true;
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--spec-follow") o.spec_follow = next("--spec-follow");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--main-gpu") o.main_gpu_name = next("--main-gpu");
        else if (a == "--second-gpu-min-mb") o.second_gpu_min_mb = std::atof(next("--second-gpu-min-mb"));
        else if (a == "--no-second-gpu-adapt") o.no_second_gpu_adapt = true;
        else if (a == "--deterministic-adapt") o.deterministic_adapt = true;
        else if (a == "--vram-pin-gib") o.vram_pin_gib = std::atof(next("--vram-pin-gib"));
        else if (a == "--second-gpu-pin-gib") o.second_gpu_pin_gib = std::atof(next("--second-gpu-pin-gib"));
        else if (a == "--kv-grow") { o.kv_grow = true; o.no_kv_grow = false; }
        else if (a == "--no-kv-grow") { o.kv_grow = false; o.no_kv_grow = true; }
        else if (a == "--head-split") o.head_split = std::atof(next("--head-split"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next("--adapt-every"));
        else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
        else if (a == "--spec-lookup") o.spec_lookup = std::max(0, std::atoi(next("--spec-lookup")));
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--window-profile") o.window_profile = true;
        else if (a == "--prefill-profile") o.prefill_profile = true;
        else if (a == "--window-hashes") o.window_hashes = next("--window-hashes");
        else if (a == "--window-logits") o.window_logits = next("--window-logits");
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max<int64_t>(1, std::atoll(next("--prompt-cache-every")));
        else if (a == "--conversation-cache-mib") o.conversation_cache_mib = std::max<int64_t>(0, std::atoll(next("--conversation-cache-mib")));
        else if (a == "--conversation-cache-slots") o.conversation_cache_slots = std::max(0, std::atoi(next("--conversation-cache-slots")));
        else if (a == "--conversation-cache-min-tokens") o.conversation_cache_min_tokens = std::max<int64_t>(0, std::atoll(next("--conversation-cache-min-tokens")));
        else if (a == "--feed-max") o.feed_max = std::max<int64_t>(0, std::atoll(next("--feed-max")));
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--adapt-decay") {
            o.adapt_decay = (float) std::atof(next("--adapt-decay"));
            // at 1 or more the usage counts grow without end and the swap gains turn NaN (upstream PR #591)
            if (!(o.adapt_decay > 0.0f && o.adapt_decay < 1.0f)) {
                std::fprintf(stderr, "--adapt-decay must be between 0 and 1 (exclusive)\n");
                return 2;
            }
        }
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--stats") o.stats = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        std::fprintf(stderr, "lamina: --tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)\n");
        usage();
        return 2;
    }

    if (o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "ram") {
        std::fprintf(stderr, "lamina: --ple-io must be direct, mmap or ram\n");
        return 2;
    }
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "k8v4" && o.kv != "q4_0") {
        std::fprintf(stderr, "lamina: --kv must be fp16, int8, k8v4 or q4_0\n");
        return 2;
    }
    if (o.expert_math != "fast" && o.expert_math != "exact") {
        std::fprintf(stderr, "lamina: --expert-math must be fast or exact\n");
        return 2;
    }
    // exact: a prompt's experts on the main GPU, whatever it holds (llama.cpp's MMQ kernels round them differently on
    // the second GPU, so a residency would show in the output)
    if (o.expert_math == "exact") o.no_prompt_offload = true;
    // every routed expert on the CPU: no tier may bring one back to a GPU
    if (o.disable_hit && (!o.tier_names.empty() || o.adapt_every > 0)) {
        std::fprintf(stderr, "lamina: --disable-hit: no second GPU, no adaptive tier\n");
        o.tier_names.clear();
        o.adapt_every = 0;
    }
    // before anything builds an expert layout or runs an expert: the GPU's kernels and the CPU pool's SwiGLU
    lamina::kernels::set_expert_math(o.expert_math == "exact" ? lamina::kernels::ExpertMath::Exact
                                                              : lamina::kernels::ExpertMath::Fast);
    lamina::kernels::cpu::set_native_exact_swiglu(o.expert_math == "exact");
    const int q5k_mins = lamina::kernels::cpu::ggml_q5_K_mins_rounding();
    lamina::kernels::set_exact_q5_K_fma(q5k_mins == 1);
    // every CUDA call from here on uses this device; the adaptive tier sets it on its own thread too
    const std::vector<lamina::core::GpuInfo> gpus = lamina::core::list_gpus();
    std::string gpu_why;
    o.main_gpu = lamina::core::find_gpu(gpus, o.main_gpu_name, gpu_why);
    if (o.main_gpu < 0 || cudaSetDevice(o.main_gpu) != cudaSuccess) {
        std::fprintf(stderr, "lamina: --main-gpu %s: %s\n", o.main_gpu_name.c_str(),
                     o.main_gpu < 0 ? gpu_why.c_str() : "cannot use it");
        return 2;
    }
    {
        const lamina::core::GpuInfo& gi = gpus[(size_t) o.main_gpu];
        std::fprintf(stderr, "lamina: main GPU %d, %s (%s, %s)\n", o.main_gpu, gi.name.c_str(),
                     gi.bus_id.c_str(), gi.uuid.c_str());
    }
    // the GPU tiers: the ones named, a name no visible GPU answers to (a card taken out, a failed driver) left out, or
    // with "auto" every other GPU, ordered by the probe below
    const bool tier_auto = o.tier_names.size() == 1 && o.tier_names[0] == "auto";
    for (const lamina::core::GpuInfo& gi : gpus)
        if (tier_auto && gi.ordinal != o.main_gpu) o.tier_gpus.push_back(gi.ordinal);
    if (tier_auto && o.tier_gpus.empty()) std::fprintf(stderr, "lamina: --second-gpu auto: no other GPU\n");
    for (size_t i = 0; i < o.tier_names.size() && !tier_auto; ++i) {
        const int d = lamina::core::find_gpu(gpus, o.tier_names[i], gpu_why);
        const bool twice = std::find(o.tier_gpus.begin(), o.tier_gpus.end(), d) != o.tier_gpus.end();
        const bool full = (int) o.tier_gpus.size() == lamina::core::kMaxGpuTiers;
        if (d < 0 || d == o.main_gpu || twice || full) {
            const std::string why = d < 0 ? gpu_why
                                    : d == o.main_gpu ? "it is the main GPU"
                                    : twice ? "it is named twice"
                                    : "the tiers are at most " + std::to_string(lamina::core::kMaxGpuTiers);
            std::fprintf(stderr, "lamina: --second-gpu %s left out: %s\n", o.tier_names[i].c_str(),
                         why.c_str());
            continue;
        }
        o.tier_gpus.push_back(d);
    }
    o.second_gpu = o.tier_gpus.empty() ? -1 : o.tier_gpus[0];
    if (!set_stack_limit(o.main_gpu, o.main_gpu)) {   // the second GPU's before its tier sizes itself
        std::fprintf(stderr, "lamina: cannot set the CUDA stack limit\n");
        return 2;
    }
    // the GPU tiers' contexts, each made on a thread of its own beside the loads: a first CUDA call took ~2 s here (the
    // head split's or the tier's, whichever came first).  Joined before either, and on any return.
    struct JoinOnExit {
        std::vector<std::thread> t;
        void join() {
            for (std::thread& th : t)
                if (th.joinable()) th.join();
        }
        ~JoinOnExit() { join(); }
    } second_ctx;
    for (const int dev : o.tier_gpus)
        second_ctx.t.emplace_back([dev] {
            if (cudaSetDevice(dev) == cudaSuccess) (void) cudaFree(nullptr);
            (void) cudaGetLastError();   // a failure shows where the GPU is first used
        });
    lamina::core::qsa_set_kv_format(o.kv == "int8"   ? lamina::core::KvFormat::Int8
                                    : o.kv == "k8v4" ? lamina::core::KvFormat::K8V4
                                    : o.kv == "q4_0" ? lamina::core::KvFormat::Q4
                                                     : lamina::core::KvFormat::F16);
    std::vector<std::string> native_shards;   // the --native model's GGUF shards
    if (o.native_preset.empty()) {
        std::fprintf(stderr, "lamina: --native MODEL.gguf is required\n");
        return 2;
    }
    {
        try {
            native_shards = lamina::gguf_split_paths(o.native_preset);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "lamina: --native: %s\n", e.what());
            return 2;
        }
        if (o.ple_gguf.empty()) o.ple_gguf = lamina::kernels::ple_table_shard(native_shards);
        if (o.ple_gguf.empty()) {
            std::fprintf(stderr, "lamina: no shard of the --native model holds per_layer_token_embd; "
                                 "name the file with --ple-gguf\n");
            return 2;
        }
        o.native_dense_gguf = native_shards;
        if (std::find(native_shards.begin(), native_shards.end(), o.ple_gguf) == native_shards.end())
            o.native_dense_gguf.push_back(o.ple_gguf);
    }
    // P7 audit: positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        std::fprintf(stderr, "lamina: --max-context must be below 2^31\n");
        return 2;
    }
    if (const int64_t cells = lamina::kernels::qsa_select_max_cells(lamina::kernels::qsa_real_shapes());
        o.max_context > cells) {
        std::fprintf(stderr, "lamina: --max-context %lld is past the QSA selection's %lld cells on this GPU\n",
                     (long long) o.max_context, (long long) cells);
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        std::fprintf(stderr, "lamina: positive --max-new and --max-context must fit the prompt and generation\n");
        return 2;
    }
    if (o.expert_cache < -1 || o.pool_workers < 0) {
        std::fprintf(stderr, "lamina: invalid --expert-cache or --pool-workers\n");
        return 2;
    }

    std::string err;
    // The pack, computed from --native's GGUF here (core::PackImage: the index and the expert table, dense.bin's bytes
    // made as the loader reads them; nothing on disk).
    lamina::core::PackImage pack_image;
    {
        const Clock::time_point t0 = Clock::now();
        if (!lamina::core::PackImage::build(o.native_preset, pack_image, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        // the dequantized ones are rounded to the engine's form (UD-Q4_K_XL's Q8_0 projections; GSQ-RCO files have none)
        size_t converted = 0, dequantized = 0;
        for (const lamina::core::PackImage::Piece& p : pack_image.pieces()) {
            converted += p.conv != lamina::core::PackImage::Conv::Copy;
            dequantized += p.dequantized();
        }
        std::fprintf(stderr, "lamina: %s in %.0f ms: %zu float tensors (dense.bin %.0f MiB; %zu converted to the "
                             "engine's form, %zu of them dequantized), %zu served from the GGUF\n",
                     pack_image.where().c_str(), std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                     pack_image.pieces().size(), (double) pack_image.dense_bytes() / 1048576.0, converted, dequantized,
                     pack_image.served());
    }
    const std::string pack_name = pack_image.where();
    // Plan v0.3 P6: where the experts live.  The pack keeps every quantized tensor in its GGUF form (the dense
    // projections, head and embedding come from the model file), and its experts run in verify windows only (--spec).
    {
        const lamina::core::ModelGeometry g0;
        lamina::kernels::cpu::hadamard_set(pack_image.hadamard());
        if (pack_image.hadamard().any())
            std::fprintf(stderr, "lamina: Hadamard-folded experts, their inputs rotated per %d values\n", pack_image.hadamard().block);
        if (!lamina::kernels::cpu::expert_layout_load_text(pack_image.native_experts_text(), g0.n_layers, g0.n_expert,
                                                           err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
    }
    for (int64_t l = 0; l < lamina::kernels::cpu::expert_layout().n_layers; ++l) {
        const auto& f = lamina::kernels::cpu::expert_layout().fmt[(size_t) l];
        if (!lamina::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
            std::fprintf(stderr, "lamina: layer %lld's experts (GGML types %d / %d) have no GPU kernels\n",
                         (long long) l, f.gu_type, f.d_type);
            return 1;
        }
        // exact's kernels cover the K-quant files' formats: refuse others here rather than at the first hit
        if (lamina::kernels::expert_math() == lamina::kernels::ExpertMath::Exact &&
            (!lamina::kernels::native_expert_exact_supported(f.gu_type, f.d_type, f.gu_act, f.d_act) ||
             ((f.gu_type == 13 || f.d_type == 13) && q5k_mins < 0))) {
            std::fprintf(stderr, "lamina: --expert-math exact has no bit-exact GPU kernels for layer "
                                 "%lld's experts (GGML types %d / %d, activations %d / %d; Q5_K needs this build's "
                                 "ggml rounding known); use --expert-math fast\n",
                         (long long) l, f.gu_type, f.d_type, f.gu_act, f.d_act);
            return 2;
        }
    }
    // the PCIe share of the missed experts (the paper, finding on PCIe)
    if (o.pcie_frac < 0.0) o.pcie_frac = 0.55;
    if (!lamina::kernels::cpu::cpu_avx512_ok())
        std::fprintf(stderr, "lamina: this CPU has no AVX-512: the expert kernels run on AVX2\n");
    lamina::core::NativeEmbed native_embed;
    {
        if (o.spec < 2 || (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            std::fprintf(stderr, "lamina: the engine needs --spec T (T >= 2) and --prefill CHUNK\n");
            return 2;
        }
        const lamina::core::ModelGeometry g0;
        if (!native_embed.load(native_shards, g0.n_embd, 248320, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        lamina::core::set_native_embed(&native_embed);
        std::fprintf(stderr, "lamina: %s: the experts' largest blob %.2f MB, token embedding type %d in mapped "
                             "host memory (%.0f MiB)\n",
                     pack_name.c_str(), (double) lamina::kernels::cpu::expert_layout().max_blob / 1e6,
                     native_embed.type(), (double) native_embed.bytes() / 1048576.0);
    }
    // the tensors served in their GGUF form are not loaded from the pack
    std::set<std::string> skip;
    {
        if (!lamina::core::NativeDense::served_names(o.native_dense_gguf, true, skip, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        skip.insert("output.weight");
        lamina::prefill::set_hc_q8(o.hc_q8);
        if (o.hc_q8 && !lamina::core::HcQ8::names(native_shards, skip, err)) {   // their 8-bit planes instead
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        skip.insert("token_embd.weight");
    }
    uint64_t pool_bytes = 0;
    const std::set<std::string>* skipped = skip.empty() ? nullptr : &skip;
    if (!lamina::core::WeightTable::pool_bytes(pack_image, pool_bytes, err, skipped)) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    void* arena = nullptr;
    if (cudaMalloc(&arena, pool_bytes) != cudaSuccess) {
        std::fprintf(stderr, "lamina: cudaMalloc(%llu) for the weight arena failed\n",
                     (unsigned long long) pool_bytes);
        return 1;
    }
    at_exit.undo.push_back([arena] { cudaFree(arena); });
    lamina::core::WeightTable wt;
    if (!wt.load(pack_image, arena, pool_bytes, err, skipped)) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "lamina: %llu MiB of weights loaded from %s (%zu canonical tensors skipped: "
                         "served natively)\n",
                 (unsigned long long) (pool_bytes >> 20), pack_name.c_str(), skip.size());

    lamina::core::NativeDense native_dense;
    {
        if (!native_dense.load(o.native_dense_gguf, wt, err, true)) {
            std::fprintf(stderr, "lamina: native dense projections: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "lamina: %zu native projection matrices, %.2f MiB of weights\n",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0));
    }

    lamina::core::HcQ8 hc_q8;
    if (o.hc_q8) {
        if (!hc_q8.load(native_shards, wt, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "lamina: %zu hyper-connection matrices in 8 bits (%zu as the GGUF stores them), "
                             "%.2f MiB\n", hc_q8.matrices(), hc_q8.stored_q8_0(), (double) hc_q8.bytes() / 1048576.0);
    }

    // the native GR write and shared expert (the parity tests switch them off to compare)
    lamina::kernels::gr_set_native_mmvf(true);
    lamina::kernels::shared_expert_set_native_bf16(true);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (lamina/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    // --prompt-split's second stage runs rope kernels on the second GPU: a copy of the table there (setup_prompt_path)
    int32_t* d_mrope2 = nullptr;
    int mrope2_dev = -1;
    auto mrope_upload = [&]() -> bool {   // the host table to each GPU's copy
        const size_t bytes = mrope_host.size() * sizeof(int32_t);
        bool ok = cudaMemcpy(d_mrope, mrope_host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok && d_mrope2 != nullptr) {
            int cur = 0;
            cudaGetDevice(&cur);
            cudaSetDevice(mrope2_dev);
            ok = cudaMemcpy(d_mrope2, mrope_host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
            cudaSetDevice(cur);
        }
        return ok;
    };
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (cudaMalloc(&d_mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "lamina: cannot allocate the image position table\n");
            return 1;
        }
        at_exit.undo.push_back([d_mrope, dev = o.main_gpu] {
            cudaSetDevice(dev);
            lamina::kernels::mrope_table_set(nullptr);
            cudaFree(d_mrope);
        });
        if (cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            std::fprintf(stderr, "lamina: cannot allocate the image position table\n");
            return 1;
        }
        lamina::kernels::mrope_table_set(d_mrope);
    }
    const lamina::core::ModelGeometry g;
    const int64_t K = 10;
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        std::fprintf(stderr, "lamina: --max-context %lld cannot hold %zu prompt + %lld new tokens\n",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    // ---- the elastic K/V (--kv-grow): set before the session and the drafter are sized.  With --serve (the CLI path
    // does not grow), a profiled cache that can give slots up, and every expert in RAM for the CPU to compute the ones
    // it gives up.  The ranges are the main GPU's alone: nothing here reads another card's memory directly (the
    // second GPU's tier gets its rows through the host), so --second-gpu keeps it.
    {
        const char* ev = std::getenv("LAMINA_KV_GROW");
        const bool asked = !o.no_kv_grow && (ev != nullptr && ev[0] != '\0' ? ev[0] != '0' : o.kv_grow);
        const bool on = asked && o.serve && !o.expert_profile.empty() && o.expert_cache != 0 && !o.disable_hit &&
                        lamina::core::vmm_available();
        if (asked && !on)
            std::fprintf(stderr, "lamina: --kv-grow is off here: it needs --serve, --expert-profile, an expert "
                                 "cache with GPU hits (not --disable-hit) and CUDA virtual memory\n");
        const char* iv = std::getenv("LAMINA_KV_GROW_INIT");
        lamina::core::qsa_set_kv_elastic(on, iv != nullptr && std::atoll(iv) > 0 ? std::atoll(iv) : 16384);
        lamina::core::ExpertCache::set_vmm(on);
    }

    void* sbuf = nullptr;
    if (cudaMalloc(&sbuf, lamina::core::session_bytes(g, o.max_context)) != cudaSuccess) {
        std::fprintf(stderr, "lamina: session state allocation failed\n");
        return 1;
    }
    at_exit.undo.push_back([sbuf] { cudaFree(sbuf); });
    at_exit.undo.push_back([] { lamina::core::qsa_kv_elastic_release(); });   // the session's and the drafter's
    // ---- the rope scaling, validated as a whole, then the process constant: session_init builds the rope table from
    // it and the captured graphs bake the rotation kernels' arguments
    {
        using RST = lamina::kernels::RopeScalingType;
        lamina::kernels::RopeScaling rope_cfg;
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            std::fprintf(stderr, "lamina: --rope-scaling must be none, linear or yarn (got '%s')\n",
                         o.rope_scaling.c_str());
            return 2;
        }
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "lamina: a rope scaling knob is not a finite number (%g)\n", v);
                return 2;
            }
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            std::fprintf(stderr, "lamina: --rope-scale %g must be >= 1 (it extends the context)\n",
                         o.rope_scale);
            return 2;
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            std::fprintf(stderr, "lamina: --rope-freq-base must be a base above 1 (0 = 1e7)\n");
            return 2;
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            std::fprintf(stderr, "lamina: invalid rope scaling knob (see usage)\n");
            return 2;
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (const char* why = lamina::kernels::rope_scaling_invalid(rope_cfg)) {
            std::fprintf(stderr, "lamina: invalid rope scaling configuration: %s (factor %g, base %g, "
                                 "original context %g)\n", why, rope_cfg.factor, rope_cfg.freq_base, rope_cfg.orig_ctx);
            return 2;
        }
        if (rope_cfg.type != RST::None) {
            const char* tn = rope_cfg.type == RST::YaRN ? "yarn" : "linear";
            std::fprintf(stderr, "lamina: rope scaling %s, factor %.6g (freq_scale %.6g, mscale %.6f), "
                                 "--max-context %lld against a trained context of %.0f\n", tn, rope_cfg.factor,
                         rope_cfg.freq_scale(), rope_cfg.mscale(), (long long) o.max_context, rope_cfg.orig_ctx);
        } else if (o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0) {
            std::fprintf(stderr, "lamina: note: --rope-scaling none, so the scaling knobs have no effect\n");
        }
        lamina::kernels::rope_scaling_set(rope_cfg);
    }
    lamina::core::SessionState ss;
    struct SessionFree {   // what session_init takes outside sbuf, after everything below that uses it
        lamina::core::SessionState& s;
        const lamina::core::ModelGeometry& g;
        ~SessionFree() { lamina::core::session_free(s, g); }
    } ss_free{ss, g};
    // the engine's own non-blocking stream: on WDDM the legacy default stream costs ~8x a launch
    cudaStream_t main_stream = nullptr;
    if (cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "lamina: cannot create the main stream\n");
        return 1;
    }
    at_exit.undo.push_back([main_stream] { cudaStreamDestroy(main_stream); });
    void* const main_cs = (void*) main_stream;
    if (lamina::core::session_init(g, o.max_context, K, sbuf, ss) == 0) {
        std::fprintf(stderr, "lamina: session_init failed\n");
        return 1;
    }
    // a run starts from a zeroed state (#167): a native pack's prompt goes through the batched path and verify
    // windows, which zero nothing, and the cudaMalloc'd state holds whatever the allocator last held
    lamina::core::session_zero(ss, g, nullptr, main_cs);
    if (cudaStreamSynchronize(main_stream) != cudaSuccess) {
        std::fprintf(stderr, "lamina: zeroing the session state failed\n");
        return 1;
    }



    // ================================ THE PLE ================================
    //
    // **ITS ABSENCE IS WHY GATE C1 FAILED** (LEDGER L123): layer 1 carries six `blk.1.ple_*` tensors, the whole
    // module was built and parity-tested, and nothing called it.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // history is the session's.
    lamina::kernels::PleTable ple_table;
    {
        lamina::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "mmap" ? lamina::kernels::PleIo::Mmap
                 : o.ple_io == "ram"  ? lamina::kernels::PleIo::Ram
                                      : lamina::kernels::PleIo::Direct;
        if (!ple_table.open(o.ple_gguf, err, pio)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        const lamina::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const lamina::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const lamina::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const lamina::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const lamina::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const lamina::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            std::fprintf(stderr, "lamina: the pack has no blk.1.ple_* tensors, so the PLE cannot be "
                                 "wired - and running without it is a DIFFERENT MODEL (LEDGER L123)\n");
            return 1;
        }
        if (!wk->quantized()) {
            // plan v0.3 P6: the IQ model files' BF16 key (the pack's extra.bin, raw BF16)
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else {
            // Q2_0 in the Q2_0 file, Q8_0 in UD-Q4_K_XL
            if (!wk->native_data || !lamina::kernels::native_mmvq_supported(wk->native_type) || !wk->native_q8_1) {
                std::fprintf(stderr, "lamina: native PLE key is absent or incompatible\n");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = lamina::kernels::ple_artifact_consts();
        ss.ple.table = &ple_table;
        ss.ple.hist = ss.ple_hist;
        if (!ss.ple.ready()) {
            std::fprintf(stderr, "lamina: the PLE run is not ready after construction\n");
            return 1;
        }
        std::fprintf(stderr, "lamina: PLE on, table %llu rows of %s\n",
                     (unsigned long long) ple_table.rows(), o.ple_gguf.c_str());
    }

    // ---- the CPU expert pool
    //
    // R2.1: the experts are read from the model's GGUF into a resident arena (ArenaExpertSource): memory the engine
    // owns, which the OS has no cheaper reason to evict.  A mapping of the experts' file ran the pool 1.8x slower warm
    // and 3.7x cold: mapped file pages are the first the OS reclaims.
    // ---- --vram-pin-gib / --second-gpu-pin-gib: the profile's hottest experts live only in VRAM.  The main GPU's
    // go into its lowest slots, the second's into the second GPU's lowest, where nothing evicts them, lends them to
    // the prompt path or gives them to the K/V; a GPU computes them whenever they are routed, and the arena leaves
    // them out (each GiB pinned is a GiB of RAM).  The main GPU's are the profile's first, the second GPU's the ones
    // right after the main GPU's cache: each card holds the experts it holds without pins (the main one, the faster,
    // the hottest).  So with pins the arena loads once the main GPU's cache is sized, just before its fill.
    std::vector<std::pair<int32_t, int32_t>> pin1, pin2;
    std::vector<uint8_t> vram_only;   // per (layer, expert): 1 only in the main GPU's VRAM, 2 only in the second's
    const bool pinning = o.vram_pin_gib > 0 || o.second_gpu_pin_gib > 0;
    if (pinning) {
        const char* why = o.expert_profile.empty() ? "it needs --expert-profile (its first experts are the ones pinned)"
                        : o.disable_hit ? "--disable-hit computes every expert on the CPU, which has no copy of them"
                        : o.second_gpu_pin_gib > 0 && o.second_gpu < 0 ? "--second-gpu-pin-gib needs --second-gpu"
                        : o.second_gpu_pin_gib > 0 && o.no_prompt_offload
                            ? "--second-gpu-pin-gib needs the prompt path's share on the second GPU (no --no-prompt-offload)"
                        : nullptr;
        if (why != nullptr) {
            std::fprintf(stderr, "lamina: --vram-pin-gib: %s\n", why);
            return 2;
        }
    }
    lamina::core::ArenaExpertSource arena_src;
    lamina::core::ExpertSource* srcp = nullptr;
    auto load_experts = [&]() -> bool {
        arena_src.set_gguf(native_shards);
        if (!vram_only.empty()) {
            // --prompt-split: the second stage streams the experts of its layers that its GPU lacks from this arena,
            // and the main GPU's pinned ones of those layers live nowhere else, so they stay here too (the first
            // stage's experts the main GPU lacks are the second GPU's, pinned there or streamed: as without the
            // split).  Before, a prompt of two chunks or more stopped at "the expert source has no blob".
            std::vector<uint8_t> left_out = vram_only;
            if (o.prompt_split > 0)
                for (int64_t l = (int64_t) o.prompt_split; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e)
                        if (uint8_t& v = left_out[(size_t) (l * g.n_expert + e)]; v == 1) v = 0;
            arena_src.set_vram_only(std::move(left_out));
        }
        arena_src.set_uncached(o.no_file_cache);
        if (!arena_src.open(g.n_layers, g.n_expert, /*threads=*/6, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return false;
        }
        std::fprintf(stderr, "lamina: expert arena: %s\n", arena_src.note().c_str());
        std::fprintf(stderr, "lamina: loaded %.2f GiB at %.2f GiB/s\n",
                     (double) (lamina::kernels::cpu::expert_layout().total - arena_src.vram_only_bytes()) /
                         (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        srcp = &arena_src;
        return true;
    };
    if (!pinning && !load_experts()) return 1;
    // The experts only in VRAM go from the model's files straight into their slots: (slot, layer, expert) each,
    // read by a few threads at once, each through a pinned block of its own.
    auto is_vram_only = [&](int64_t layer, int64_t expert) {
        return !vram_only.empty() && vram_only[(size_t) layer * (size_t) g.n_expert + (size_t) expert] != 0;
    };
    auto fill_from_model = [&](lamina::core::ExpertCache& cache, int device,
                               const std::vector<std::array<int32_t, 3>>& list, std::string& e) -> bool {
        if (list.empty()) return true;
        const auto& lay = lamina::kernels::cpu::expert_layout();
        std::atomic<size_t> next{0};
        std::atomic<bool> bad{false};
        std::mutex mu;
        auto worker = [&]() {
            std::string we;
            uint8_t* buf = nullptr;
            if (cudaSetDevice(device) != cudaSuccess || cudaHostAlloc((void**) &buf, lay.max_blob, cudaHostAllocDefault) != cudaSuccess) {
                (void) cudaGetLastError();
                we = "no pinned block to read the VRAM-only experts through";
            }
            for (size_t k; we.empty() && !bad && (k = next++) < list.size();) {
                const auto [slot, l, ex] = list[k];
                if (!arena_src.read_file_blob(l, ex, buf, we)) break;
                if (cudaMemcpy(cache.device_slot(slot), buf, (size_t) lay.blob_bytes(l), cudaMemcpyHostToDevice) != cudaSuccess)
                    we = std::string("copying a VRAM-only expert: ") + cudaGetErrorString(cudaGetLastError());
            }
            if (buf != nullptr) cudaFreeHost(buf);
            if (!we.empty()) {
                std::lock_guard<std::mutex> lk(mu);
                if (!bad.exchange(true)) e = we;
            }
        };
        std::vector<std::thread> ts;
        for (int t = 0; t < 8; ++t) ts.emplace_back(worker);
        for (auto& t : ts) t.join();
        return !bad;
    };
    // a slot holding an expert only in VRAM read back and compared with the model's bytes
    auto verify_from_model = [&](lamina::core::ExpertCache& cache, int32_t slot, int64_t layer, int64_t expert,
                                 std::string& e) -> bool {
        std::vector<uint8_t> want(lamina::kernels::cpu::expert_layout().max_blob);
        const int64_t bytes = (int64_t) lamina::kernels::cpu::expert_layout().blob_bytes(layer);
        return arena_src.read_file_blob(layer, expert, want.data(), e) && cache.verify_slot(slot, want.data(), e, bytes);
    };
    // Plan v0.3 P6: the MTP draft layer, loaded before the VRAM expert tier is sized from what is left.
    lamina::core::MtpDrafter mtp;
    if (!o.mtp.empty()) {
        if (o.spec < 2) {
            std::fprintf(stderr, "lamina: --mtp is ignored without --spec T (T >= 2)\n");
            o.mtp.clear();
        }
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        if (!o.mtp.empty() && !mtp.load(o.mtp, g, ss, o.max_window(), err, o.mtp_window)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        mtp.set_max_drafts(o.spec - 1);
    }
    // The GPU tiers' links measured, and whether the desktop uses each, while no load reads the RAM: "auto" puts the
    // fastest first (host-to-device rate x multiprocessors) and keeps the first kMaxGpuTiers; a reserve or a prefetch
    // budget not given comes from them.
    if (!o.tier_gpus.empty()) {
        second_ctx.join();
        const auto p0 = std::chrono::steady_clock::now();
        const std::vector<lamina::core::GpuProbe> probe = lamina::core::probe_gpus(o.tier_gpus, o.main_gpu);
        std::fprintf(stderr, "lamina: the GPU tiers' links probed in %.2f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count());
        std::vector<size_t> order;
        for (size_t i = 0; i < probe.size(); ++i) order.push_back(i);
        auto score = [&](size_t i) { return probe[i].h2d_gbps * gpus[(size_t) o.tier_gpus[i]].sms; };
        if (tier_auto)
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return score(a) > score(b); });
        std::vector<int> devs;
        std::vector<double> reserve, prefetch;
        for (const size_t i : order) {
            const int d = o.tier_gpus[i];
            if ((int) devs.size() == lamina::core::kMaxGpuTiers) {   // "auto" past the third: its context goes
                std::fprintf(stderr, "lamina: GPU %d, %s, left out: the tiers are at most %d\n", d,
                             gpus[(size_t) d].name.c_str(), lamina::core::kMaxGpuTiers);
                cudaSetDevice(d);
                cudaDeviceReset();
                continue;
            }
            devs.push_back(d);
            reserve.push_back(probe[i].display ? 2048.0 : 512.0);
            prefetch.push_back(probe[i].h2d_gbps >= 20.0 ? 12.0 : 0.0);
        }
        cudaSetDevice(o.main_gpu);
        o.tier_gpus = devs;
        o.second_gpu = devs[0];
        if (o.second_gpu_reserve_mib.empty()) o.second_gpu_reserve_mib = reserve;
        if (o.second_gpu_prefetch_mb.empty()) o.second_gpu_prefetch_mb = prefetch;
        for (size_t t = 0; t < devs.size(); ++t) {
            const lamina::core::GpuInfo& gi = gpus[(size_t) devs[t]];
            const lamina::core::GpuProbe& pr = probe[order[t]];
            std::fprintf(stderr, "lamina: GPU tier %zu: GPU %d, %s (%s, %s): %.1f GB/s from the host, %d "
                                 "multiprocessors, %s; reserve %.0f MiB, prefetch %.0f MiB\n", t + 1, gi.ordinal,
                         gi.name.c_str(), gi.bus_id.c_str(), gi.uuid.c_str(), pr.h2d_gbps, gi.sms,
                         !pr.display_known ? "its use unknown" : pr.display ? "a desktop" : "headless",
                         tier_value(o.second_gpu_reserve_mib, t), tier_value(o.second_gpu_prefetch_mb, t));
        }
    }
    // --ple-io ram: the table goes into RAM once the SSD has delivered the experts and the draft layer (with pinned
    // experts the arena loads later, before the cache's fill: the table follows it there)
    auto start_ple_ram = [&]() {
        if (ple_table.is_open() && !ple_table.start_ram_load(err))
            std::fprintf(stderr, "lamina: %s; its rows stay on the SSD\n", err.c_str());
    };
    if (!pinning) start_ple_ram();
    // where the host loop and the pool's workers run, moved off the processors that take interrupts as generation
    // goes (drive_window); with a second GPU, whose work raises most of them, the first core starts free
    lamina::kernels::cpu::CorePlacement placement(/*spare_first=*/o.second_gpu >= 0);
    lamina::kernels::cpu::ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/true,
                                          placement.workers());
    const lamina::kernels::cpu::HostPin host_pin(placement.host());   // the core the workers leave it
    std::fprintf(stderr, "lamina: session is up; locating the head\n");
    const lamina::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { std::fprintf(stderr, "lamina: output.weight is missing\n"); return 1; }
    const int64_t n_vocab = wo->ne1;
    lamina::core::NativeHead native_head;
    std::unique_ptr<lamina::core::SplitHead> split_head;
    {
        const std::vector<std::string>& head_shards = native_shards;
        if (!native_head.load(head_shards, g.n_embd, n_vocab, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "lamina: native head, type %d, %llu bytes\n", native_head.type(),
                     (unsigned long long) native_head.weight_bytes());
        // the head's last rows on the second GPU (split_head.hpp): each GPU streams its part after the last layer.
        // The drafter gathers its draft head's rows first; without one it needs the whole head here.
        if (o.second_gpu >= 0 && o.head_split > 0.0 && o.head_split < 1.0 && o.spec >= 2) {
            second_ctx.join();
            const int64_t split = (int64_t) ((1.0 - o.head_split) * (double) n_vocab) / 256 * 256;
            auto sh = std::make_unique<lamina::core::SplitHead>();
            std::string e2;
            const bool ok = (o.mtp.empty() || mtp.bind_head(&native_head, e2)) &&
                            (o.mtp.empty() || mtp.has_draft_head()) &&
                            sh->init(o.second_gpu, o.main_gpu, head_shards, g.n_embd, n_vocab, split, o.max_window(),
                                     e2) &&
                            native_head.keep_rows(split, e2);
            if (ok) {
                std::fprintf(stderr, "lamina: the head's rows %lld-%lld on the second GPU (%.0f MiB), the "
                                     "first %lld here\n", (long long) split, (long long) n_vocab,
                             (double) sh->bytes() / 1048576.0, (long long) split);
                split_head = std::move(sh);
            } else {
                std::fprintf(stderr, "lamina: the head stays on the main GPU: %s\n",
                             e2.empty() ? "the draft layer has no draft head of its own" : e2.c_str());
            }
        }
    }
    // ---- R4's slot storage.  Allocated AFTER the weights, the session and the head, so `cudaMemGetInfo` inside
    // `open` sees the memory this process actually has left rather than the card's idle figure - and refuses with
    // both numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    mem_mark("the weights, the session, the drafter and the head");
    lamina::core::ExpertCache xcache;
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!lamina::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        // The profile knows how many slots it was built for.  `--expert-cache 0` means "take the profile's";
        // an explicit smaller number is allowed and simply truncates the ranked list, which is the right
        // behaviour for asking "what would 2,000 slots give" without rebuilding the file.
        if (o.expert_cache == 0) o.expert_cache = (int) pslots;
        std::fprintf(stderr, "lamina: profile %s: %zu ranked pairs, built for %lld slots\n",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    const bool auto_cache = o.expert_cache < 0;
    if (o.expert_cache < 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // Plan v0.3 P5: the batched prompt path's chunk buffers are allocated later, so they are reserved here -
        // under WDDM an over-subscribed allocation does not fail, it pages to system memory and crawls.
        // (with borrowing - the default with a profile - the prompt path lends cache slots instead)
        const bool borrow = !o.no_prefill_borrow && !o.expert_profile.empty();
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !borrow)
            ? (int64_t) (lamina::prefill::Prefill::bytes_needed(g, ss, o.prefill_chunk,
                                                                o.second_gpu >= 0 && !o.no_prompt_offload) >> 20)
            : 0;
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + prefill_mib) << 20;
        int64_t slots = ((int64_t) free_b - reserve) / (int64_t) lamina::kernels::cpu::expert_layout().max_blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        std::fprintf(stderr, "lamina: expert cache auto: %.2f GiB free, %d MiB reserved -> %d slots\n",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, o.expert_cache);
        if (o.expert_cache == 0)   // the verify window cannot run without it (#174)
            std::fprintf(stderr, "lamina: no VRAM is left for the expert cache: lower --max-context or close "
                                 "other programs that use the GPU\n");
    }
    // plan v0.3 P6: a native pack's blobs differ per layer, so with a profile its slots are sized per pair: the
    // same VRAM holds ~30% more IQ3_XXS experts than slots of the largest blob would
    std::vector<int64_t> sized_slots;
    if (o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const auto& lay = lamina::kernels::cpu::expert_layout();
        const uint64_t budget = (uint64_t) o.expert_cache * lay.max_blob;   // what the uniform sizing granted
        uint64_t used = 0;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(pr.first));
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        o.expert_cache = (int) sized_slots.size();
    }
    if (o.expert_cache > 0) {
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: under WDDM an
        // allocation is not resident until it is touched, and the free figure read before it can be ~1 GB too
        // high.  A cache sized from it filled the card to 0 MiB, the driver then paged, and a request that needed a
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        int zero_reads = 0;
        for (int attempt = 0;; ++attempt) {
            const bool ok = sized_slots.empty()
                ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) lamina::kernels::cpu::expert_layout().max_blob, err)
                : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            if (!ok) {
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            if (!auto_cache || attempt >= 6) break;
            cudaMemset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            cudaDeviceSynchronize();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t want = (int64_t) o.vram_reserve_mib << 20;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least": the first two such reads give back 1 GiB
            // each (under WDDM the free figure read before the allocation runs ~0.7 GiB high), later ones a quarter
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20))
                give = std::max<int64_t>(give, ++zero_reads <= 2 ? 1ll << 30 : xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            std::fprintf(stderr, "lamina: only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache\n", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); break; }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() &&
                       used + lamina::core::ExpertCache::slot_stride(sized_slots[keep]) <= keep_bytes)
                    used += lamina::core::ExpertCache::slot_stride(sized_slots[keep++]);
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) lamina::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); break; }
        }
    }
    // the elastic K/V's VMM arena is the main GPU's cache alone: the second GPU's tier opens its own cache on its own
    // card (vmm.cpp's chunks belong to the device current at its first call, the main one)
    lamina::core::ExpertCache::set_vmm(false);
    if (o.expert_cache > 0) {
        std::fprintf(stderr, "lamina: expert cache %lld slots, %.2f GiB of VRAM; policy is\n",
                     (long long) xcache.slots(), xcache.gib());
        mem_mark("opening the expert cache");
        // fast: the GPU computes its experts from q8_1 activations, the CPU pool from ggml's, so a run with the cache
        // can part from a cache-off run at a near tie (rounding; the next-token distributions agree with llama.cpp's).
        // exact: the CPU's activation formats and operations, so it cannot (--disable-hit is the cache-off run).
        std::fprintf(stderr, "lamina: GPU expert math %s%s\n", o.expert_math.c_str(),
                     o.expert_math != "exact" ? " (llama.cpp q8_1; agrees with the CPU pool to rounding)"
                     : q5k_mins == 1          ? " (bit-identical to the CPU pool; ggml's Q5_K mins summed by FMA)"
                                              : " (bit-identical to the CPU pool; ggml's Q5_K mins: a product, then a sum)");
        if (profile.empty()) {
            std::fprintf(stderr, "                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            std::fprintf(stderr, "                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- R4.2e: fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    if (pinning) {
        const auto& lay = lamina::kernels::cpu::expert_layout();
        const size_t n_main = (size_t) std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        auto take = [&](size_t from, size_t to, double gib, std::vector<std::pair<int32_t, int32_t>>& out) {
            const uint64_t budget = (uint64_t) (gib * 1073741824.0);
            uint64_t used = 0;
            for (size_t i = from; i < to; ++i) {
                const uint64_t b = (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(profile[i].first));
                if (used + b > budget) break;
                used += b;
                out.push_back(profile[i]);
            }
            return used;
        };
        const uint64_t b1 = take(0, n_main, o.vram_pin_gib, pin1);
        const uint64_t b2 = take(n_main, profile.size(), o.second_gpu_pin_gib, pin2);
        vram_only.assign((size_t) (g.n_layers * g.n_expert), 0);
        for (const auto& pr : pin1) vram_only[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
        for (const auto& pr : pin2) vram_only[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 2;
        std::fprintf(stderr, "lamina: %zu experts (%.2f GiB) live only in the main GPU's VRAM (its cache's "
                             "hottest, of %zu), %zu (%.2f GiB) only in the second GPU's; the RAM holds the others\n",
                     pin1.size(), (double) b1 / 1073741824.0, n_main, pin2.size(), (double) b2 / 1073741824.0);
        if (!load_experts()) return 1;
        start_ple_ram();
    }
    int64_t prefilled = 0;
    if (!profile.empty() && srcp != nullptr) {
        const int64_t want = std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        std::vector<std::array<int32_t, 3>> from_model;   // (slot, layer, expert): only in VRAM, read from the model
        for (int64_t i = 0; i < want; ++i) {
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == lamina::core::kNotResident) break;
            if (is_vram_only(profile[(size_t) i].first, profile[(size_t) i].second)) {
                from_model.push_back({slot, profile[(size_t) i].first, profile[(size_t) i].second});
                ++prefilled;
                continue;
            }
            const uint8_t* b = srcp->blob(profile[(size_t) i].first, profile[(size_t) i].second);
            if (b == nullptr || !xcache.fill_slot_blocking(slot, b, err,
                    (int64_t) lamina::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first))) {
                std::fprintf(stderr, "lamina: the profile fill failed at pair %lld: %s\n",
                             (long long) i, err.c_str());
                return 1;
            }
            ++prefilled;
        }
        if (!fill_from_model(xcache, o.main_gpu, from_model, err)) {
            std::fprintf(stderr, "lamina: the VRAM-only experts: %s\n", err.c_str());
            return 1;
        }
        cudaSetDevice(o.main_gpu);
        // the pinned experts are the cache's lowest slots, in order (the rest of the engine keeps away from them)
        for (size_t k = 0; k < pin1.size(); ++k)
            if (xcache.slot_of(pin1[k].first, pin1[k].second) != (int32_t) k) {
                std::fprintf(stderr, "lamina: the main GPU's cache holds %lld slots, fewer than the %zu experts "
                                     "--vram-pin-gib pins there: lower it\n", (long long) xcache.slots(), pin1.size());
                return 1;
            }
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one 1.38 MB D2H at startup.
        const int32_t slot0 = xcache.slot_of(profile[0].first, profile[0].second);
        if (prefilled > 0 &&
            !(is_vram_only(profile[0].first, profile[0].second)
                  ? verify_from_model(xcache, slot0, profile[0].first, profile[0].second, err)
                  : xcache.verify_slot(slot0, srcp->blob(profile[0].first, profile[0].second), err,
                                       (int64_t) lamina::kernels::cpu::expert_layout().blob_bytes(profile[0].first)))) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        // and the last pinned one, the far end of the parallel reads
        if (pin1.size() > 1 && !verify_from_model(xcache, (int32_t) pin1.size() - 1, pin1.back().first,
                                                  pin1.back().second, err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the profile fill");
        std::fprintf(stderr, "lamina: pre-filled %lld of %lld slots from the profile; slot 0 verified\n",
                     (long long) prefilled, (long long) want);
    }

    Drive drive;
    drive.d.pool = &pool;
    drive.placement = &placement;
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    // ---- the routing trace (--dump-routing): every verify window row's routed experts (drive_pool_multi)
    std::FILE* routing = nullptr;
    if (!o.dump_routing.empty()) {
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            std::fprintf(stderr, "lamina: cannot write %s\n", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    std::string on;
    for (int i = 0; i < pool.workers() && i < (int) placement.workers().size(); ++i)
        on += (i > 0 ? "," : "") + std::to_string(placement.workers()[(size_t) i]);
    std::fprintf(stderr, "lamina: %d pool workers on logical processors %s, the host thread on %d%s\n",
                 pool.workers(), on.c_str(), placement.host(), pool.host_works() ? " (draining too)" : "");

    mem_mark("the expert cache");
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];   // the first verify window's token: the prompt's last, once the batched path read the rest

    // with a profile-filled cache the residency is static: the verify windows decide the hits on the device from
    // this table
    std::vector<int32_t> host_res;
    int32_t* d_res = nullptr;
    lamina::core::TokenHits thits;
    const bool graph_hits = o.expert_cache > 0 && !profile.empty();
    if (pinning && !graph_hits) {   // the residency table is their only path (checked with the options above)
        std::fprintf(stderr, "lamina: --vram-pin-gib: the experts only in VRAM need the GPU hit path\n");
        return 2;
    }
    if (graph_hits) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), lamina::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                // --disable-hit: the cache stays filled, the table holds none of it, so every expert goes to the CPU
                const int32_t slot = o.disable_hit ? lamina::core::kNotResident : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != lamina::core::kNotResident) ++resident;
            }
        if (cudaMalloc((void**) &d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "lamina: the device residency table could not be staged\n");
            return 1;
        }
        at_exit.undo.push_back([d_res] { cudaFree(d_res); });
        if (res_put(d_res, host_res) != cudaSuccess) {
            std::fprintf(stderr, "lamina: the device residency table could not be staged\n");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        thits.cache_base = (const uint8_t*) xcache.device_slot(0);
        thits.blob = (int64_t) lamina::kernels::cpu::expert_layout().max_blob;
        drive.d.host_res = host_res.data();
        std::fprintf(stderr, "lamina: the residency table: %lld resident experts, decided on the device%s\n",
                     (long long) resident, o.disable_hit ? " (--disable-hit)" : "");
    }
    // plan v0.3 P6: the VRAM tier follows the conversation in both decode loops (--serve and the speculative loop):
    // moves every --adapt-every rounds on the adapt thread, admitted once their copies have landed
    lamina::core::AdaptiveTier tier;
    const bool adaptive = !host_res.empty() && o.adapt_every > 0 && o.adapt_swaps > 0;
    if (adaptive && !tier.init(xcache, *srcp, host_res, g.n_layers, g.n_expert, o.adapt_swaps, err,
                               o.main_gpu, o.main_gpu, o.deterministic_adapt)) {
        std::fprintf(stderr, "lamina: %s\n", err.c_str());
        return 1;
    }
    tier.set_frozen((int64_t) pin1.size());
    // ---- GPUs as further expert tiers (--second-gpu A[,B,C]): each holds the profile's pairs no GPU before it holds,
    // then empty slots for every layer, which its own adaptive tier fills with the conversation's experts the tiers
    // before it have not.  A GPU without free VRAM (another process holds it) stays out; without any the engine runs on
    // the main GPU alone.
    struct TierSlot {
        lamina::core::SecondGpu gpu;
        std::vector<int32_t> res;   // its residency (n_layers x n_expert, slot or kNotResident)
        lamina::core::AdaptiveTier adapt;
    };
    std::array<TierSlot, lamina::core::kMaxGpuTiers> tiers;
    int n_tiers = 0;
    // the first (the "second GPU"): the prompt path's share, the head's rows, --prompt-split's second stage
    lamina::core::SecondGpu& gpu2 = tiers[0].gpu;
    std::vector<int32_t>& host_res2 = tiers[0].res;
    lamina::core::AdaptiveTier& tier2 = tiers[0].adapt;
    static const char* const kTierName[lamina::core::kMaxGpuTiers] = {"second", "third", "fourth"};
    second_ctx.join();
    if (!o.tier_gpus.empty() && (host_res.empty() || o.spec < 2)) {
        std::fprintf(stderr, "lamina: --second-gpu needs --expert-profile and --spec\n");
        return 2;
    }
    {
        const auto& lay = lamina::kernels::cpu::expert_layout();
        uint64_t min_blob = lay.max_blob;   // prefetch slots for the most of the smallest experts a budget holds
        for (int64_t l = 0; l < g.n_layers; ++l) min_blob = std::min<uint64_t>(min_blob, lay.blob_bytes(l));
        const int main_dev = o.main_gpu;
        for (size_t ti = 0; ti < o.tier_gpus.size(); ++ti) {
            const int dev = o.tier_gpus[ti];
            const int t = n_tiers;        // its place among the tiers that work
            const bool first = t == 0;    // the --second-gpu-pin-gib pins and the prompt path's own buffers are the first's
            TierSlot& ts = tiers[(size_t) t];
            lamina::core::SecondGpu& gp = ts.gpu;
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, dev);
            const uint64_t pf_bytes = (uint64_t) (std::max(0.0, tier_value(o.second_gpu_prefetch_mb, ti)) * 1048576.0);
            const double gib = tier_value(o.second_gpu_gib, ti);
            const int reserve_mib = (int) tier_value(o.second_gpu_reserve_mib, ti);
            // its context made here (or by the head's rows): one made at startup took the RAM the pinned arena gets,
            // and the CPU pool read IQ3_S's experts 5-10% slower
            if (!set_stack_limit(dev, main_dev)) {
                std::fprintf(stderr, "lamina: cannot set GPU %d's CUDA stack limit\n", dev);
                return 2;
            }
            bool ok = gp.init(dev, main_dev, g.n_embd, g.n_ff, o.max_window(), ss.k, err) &&
                      gp.init_prefetch((int) std::min<uint64_t>(pf_bytes / min_blob, 64), lay.max_blob, err);
            std::vector<int64_t> sizes;
            std::vector<std::pair<int32_t, int32_t>> pre;
            std::vector<int32_t> empty;
            if (ok) {
                cudaSetDevice(dev);
                size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                // the prompt path's buffers there borrow cache slots, else they are reserved here
                const bool offload_own = first && o.prefill_chunk > 0 && !o.no_prompt_offload && o.no_prefill_borrow;
                const uint64_t reserve = ((uint64_t) reserve_mib << 20) +
                    (offload_own ? lamina::prefill::ExpertRunner::bytes_needed(o.prefill_chunk, g.n_expert, true, true) : 0);
                const uint64_t budget = gib > 0 ? (uint64_t) (gib * 1073741824.0)
                                                : (free_b > reserve ? (uint64_t) free_b - reserve : 0);
                uint64_t used = 0;
                for (size_t i = (size_t) prefilled; i < profile.size(); ++i) {
                    const auto& pr = profile[i];
                    const size_t at = (size_t) (pr.first * g.n_expert + pr.second);
                    bool held = host_res[at] >= 0;   // the main GPU or a tier before it holds it
                    for (int j = 0; j < t && !held; ++j) held = tiers[(size_t) j].res[at] >= 0;
                    if (held) continue;
                    const uint64_t b = (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(pr.first));
                    if (used + b > budget) break;
                    used += b;
                    sizes.push_back((int64_t) lay.blob_bytes(pr.first));
                    pre.push_back(pr);
                }
                {   // the rest in empty slots (without the adaptive tier only the prompt path's loans use them)
                    uint64_t round = 0;
                    for (int64_t l = 0; l < g.n_layers; ++l)
                        round += (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(l));
                    for (uint64_t q = (budget - used) / round; q > 0; --q)
                        for (int64_t l = 0; l < g.n_layers; ++l) {
                            sizes.push_back((int64_t) lay.blob_bytes(l));
                            empty.push_back((int32_t) l);
                        }
                }
                if (sizes.empty()) err = "no VRAM for experts (" + std::to_string(free_b >> 20) + " MiB free)";
                ok = !sizes.empty() && gp.cache().open_sized(sizes, g.n_layers, g.n_expert, err);
                // as on the main GPU, the reserve must still be free once the slots are written (WDDM makes memory
                // resident when it is first touched): short of it, the cache reopens without its last (empty) slots
                for (int attempt = 0; ok && gib <= 0 && attempt < 6; ++attempt) {
                    cudaMemset(gp.cache().device_slot(0), 0, (size_t) gp.cache().bytes());
                    cudaDeviceSynchronize();
                    cudaMemGetInfo(&free_b, &total_b);
                    if ((uint64_t) free_b + (64ull << 20) >= reserve) break;
                    std::fprintf(stderr, "lamina: %s GPU: only %lld MiB free once its slots are written "
                                         "(reserve %d MiB); shrinking its cache\n", kTierName[t],
                                 (long long) (free_b >> 20), reserve_mib);
                    gp.cache().close();
                    bool pins_short = false;
                    for (uint64_t cut = 0; !sizes.empty() && cut < reserve - free_b + (64ull << 20);) {
                        if (empty.empty() && pre.size() <= (first ? pin2.size() : 0)) { pins_short = true; break; }
                        cut += (uint64_t) lamina::core::ExpertCache::slot_stride(sizes.back());
                        sizes.pop_back();
                        if (!empty.empty()) empty.pop_back();
                        else pre.pop_back();
                    }
                    if (pins_short && first && !pin2.empty()) {   // pins stay
                        ok = false;
                        err = "too little VRAM for the experts --second-gpu-pin-gib pins there once its slots are "
                              "written: lower it";
                        break;
                    }
                    if (sizes.empty()) err = "no VRAM for experts once the slots are written";
                    ok = !sizes.empty() && gp.cache().open_sized(sizes, g.n_layers, g.n_expert, err);
                }
                std::vector<std::array<int32_t, 3>> from_model;   // (slot, layer, expert): only in this GPU's VRAM
                for (size_t i = 0; ok && i < pre.size(); ++i) {
                    const int32_t slot = gp.cache().admit(pre[i].first, pre[i].second);
                    if (slot != lamina::core::kNotResident && is_vram_only(pre[i].first, pre[i].second)) {
                        from_model.push_back({slot, pre[i].first, pre[i].second});
                        continue;
                    }
                    const uint8_t* b = srcp->blob(pre[i].first, pre[i].second);
                    ok = slot != lamina::core::kNotResident && b != nullptr &&
                         gp.cache().fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(pre[i].first));
                }
                ok = ok && fill_from_model(gp.cache(), dev, from_model, err);
                cudaSetDevice(dev);
                for (size_t k = 0; ok && first && k < pin2.size(); ++k)
                    if (gp.cache().slot_of(pin2[k].first, pin2[k].second) != (int32_t) k) {
                        ok = false;
                        err = "its cache holds " + std::to_string(gp.cache().slots()) + " slots, fewer than the " +
                              std::to_string(pin2.size()) + " experts --second-gpu-pin-gib pins there: lower it";
                    }
                if (ok && first && !pin2.empty())
                    ok = verify_from_model(gp.cache(), 0, pin2[0].first, pin2[0].second, err) &&
                         verify_from_model(gp.cache(), (int32_t) pin2.size() - 1, pin2.back().first, pin2.back().second,
                                           err);
                cudaSetDevice(main_dev);
            }
            if (ok) {
                ts.res.assign((size_t) (g.n_layers * g.n_expert), lamina::core::kNotResident);
                for (int64_t l = 0; l < g.n_layers; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e) ts.res[(size_t) (l * g.n_expert + e)] = gp.cache().slot_of(l, e);
            }
            if (ok && adaptive) {
                ok = ts.adapt.init(gp.cache(), *srcp, ts.res, g.n_layers, g.n_expert, o.adapt_swaps, err, dev, main_dev,
                                   o.deterministic_adapt);
                if (ok) {
                    if (first) ts.adapt.set_frozen((int64_t) pin2.size());
                    // lowest slots first: the last ones, which the prompt path borrows, stay empty longest
                    for (size_t i = empty.size(); i-- > 0;) ts.adapt.add_free(empty[i], (int32_t) (pre.size() + i));
                    // behind the main GPU's tier and the tiers before it
                    ts.adapt.add_upper(&tier);
                    tier.add_lower(&ts.adapt);
                    for (int j = 0; j < t; ++j) {
                        ts.adapt.add_upper(&tiers[(size_t) j].adapt);
                        tiers[(size_t) j].adapt.add_lower(&ts.adapt);
                    }
                    ts.adapt.set_paced(gp.done_event());   // between windows; a window's last share may still read its slots
                    drive.paced.push_back(&ts.adapt);
                    drive.copies_end.emplace_back();
                }
            }
            if (ok) {
                drive.d.tiers[t].gpu = &gp;
                drive.d.tiers[t].res = ts.res.data();
                drive.d.tiers[t].prefetch_bytes = pf_bytes;
                drive.d.n_tiers = ++n_tiers;
                std::fprintf(stderr, "lamina: %s GPU %d (%s): %zu experts from the profile and %zu empty "
                                     "slots, %.2f GiB\n", kTierName[t], dev, prop.name, pre.size(), empty.size(),
                             gp.cache().gib());
            } else {
                // another process may hold its VRAM (a model, a desktop app): the engine runs without it
                std::fprintf(stderr, "lamina: %s GPU %d unused: %s\n", kTierName[t], dev, err.c_str());
                if (first && !pin2.empty()) {   // its pinned experts are in no RAM: nothing else can compute them
                    std::fprintf(stderr, "lamina: the experts --second-gpu-pin-gib pins have no other home\n");
                    return 1;
                }
                // a fresh slot for the next GPU (the adaptive tier before the cache it points to)
                ts.adapt.~AdaptiveTier();
                new (&ts.adapt) lamina::core::AdaptiveTier();
                ts.res.clear();
                gp.~SecondGpu();
                new (&gp) lamina::core::SecondGpu();
            }
        }
        drive.d.tier_min_bytes = (uint64_t) (o.second_gpu_min_mb * 1048576.0);
        // the tiers that work, in order: the first takes the prompt path's share and the split
        o.tier_gpus.clear();
        for (int t = 0; t < n_tiers; ++t) o.tier_gpus.push_back(tiers[(size_t) t].gpu.device());
        o.second_gpu = n_tiers > 0 ? o.tier_gpus[0] : -1;
    }
    // The largest loan the main GPU's prompt path can take: its buffers for the longest chunk with a whole layer
    // streamed (the prefetch area grows with the experts the cache does not hold).  The elastic K/V stays below it.
    auto worst_loan = [&](bool off) {
        const auto& lay = lamina::kernels::cpu::expert_layout();
        uint64_t stride = 0;
        for (int64_t l = 0; l < lay.n_layers; ++l)
            stride = std::max<uint64_t>(stride, (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(l)));
        // (with pinned experts and the second GPU's share of the prompt path, no layer streams here: the loan is its
        // buffers alone, Prefill::bytes_for)
        const uint64_t worst_pre = !pin1.empty() && off ? 0 : stride * (uint64_t) g.n_expert;
        return lamina::prefill::Prefill::bytes_needed(g, ss, o.prefill_chunk, off) + worst_pre + worst_pre / 10;
    };
    // --arena-place: the RAM's slow places go to the experts that stay on a GPU: below the slots each GPU's prompt
    // path borrows at every batched prompt (its cache's last ones) and, with the elastic K/V, below the slots the whole
    // context's K/V takes under the main GPU's loan (after the PLE table's load, whose reads would slow the timing)
    int64_t stay1 = 0, stay2 = 0;   // the slots below which the GPUs' experts stay there
    std::vector<int32_t> profile_rank;
    // per (layer, expert): whether it stays on a GPU, from the residency as it is now
    auto arena_on_gpu = [&] {
        std::vector<uint8_t> on((size_t) (g.n_layers * g.n_expert), 0);
        for (size_t i = 0; i < on.size(); ++i)
            on[i] = (!host_res.empty() && host_res[i] >= 0 && host_res[i] < stay1) ||
                    (!host_res2.empty() && host_res2[i] >= 0 && host_res2[i] < stay2);
        return on;
    };
    if (o.arena_place && srcp == &arena_src) {
        if (ple_table.is_open() && ple_table.mode() == lamina::kernels::PleIo::Ram && !ple_table.ram_ready())
            ple_table.wait_ram_load();
        const bool off = o.second_gpu >= 0 && !o.no_prompt_offload;
        auto below_loan = [&](lamina::core::ExpertCache& c, size_t pinned, uint64_t need) -> int64_t {
            SlotLoan l;
            l.cache = &c;
            l.keep = (int64_t) pinned;
            return l.plan(need) ? (int64_t) l.first : c.slots();
        };
        // each GPU's loan for the longest chunk (Prefill::bytes_for, ExpertRunner::bytes_for)
        using lamina::prefill::ExpertRunner;
        const int64_t chunk = o.prefill_chunk;
        const int32_t* res1 = host_res.empty() ? nullptr : host_res.data();
        const int32_t* res2 = host_res2.empty() ? nullptr : host_res2.data();
        const uint64_t need1 = lamina::prefill::Prefill::bytes_needed(g, ss, chunk, off) +
                               (off ? 0 : std::min(ExpertRunner::prefetch_need(res1, nullptr, g.n_expert, true),
                                                   ExpertRunner::area_cap(chunk, false)));
        const uint64_t need2 = ExpertRunner::bytes_needed(chunk, g.n_expert, true, true) +
                               std::min(ExpertRunner::prefetch_need(res2, res1, g.n_expert, true,
                                                                    chunk >= ExpertRunner::kShiftMinTokens ? o.prompt_shift : 0.0),
                                        ExpertRunner::area_cap(chunk, true));
        const bool borrow = !o.no_prefill_borrow && d_res != nullptr;
        stay1 = host_res.empty() ? 0 : !borrow ? xcache.slots() : below_loan(xcache, pin1.size(), need1);
        if (lamina::core::qsa_kv_elastic() && xcache.vmm_range() != nullptr && stay1 > 0) {
            if (borrow) stay1 = std::min(stay1, below_loan(xcache, 0, worst_loan(off)));
            const uint64_t G = lamina::core::vmm_granularity();
            const uint64_t kv = (uint64_t) lamina::core::qsa_kv_elastic_need(o.max_context + 64) * G + G;
            const uint64_t top = xcache.slot_offset(stay1);
            while (stay1 > (int64_t) pin1.size() && top - xcache.slot_offset(stay1) < kv) --stay1;
        }
        stay2 = host_res2.empty() ? 0 : !(off && borrow) ? gpu2.cache().slots()
                                      : below_loan(gpu2.cache(), pin2.size(), need2);
        profile_rank.assign((size_t) (g.n_layers * g.n_expert), -1);
        for (size_t k = 0; k < profile.size(); ++k) {
            int32_t& r = profile_rank[(size_t) profile[k].first * (size_t) g.n_expert + (size_t) profile[k].second];
            if (r < 0) r = (int32_t) k;
        }
        std::fprintf(stderr, "lamina: --arena-place: the experts in the main GPU's slots below %lld and the "
                             "second's below %lld stay there\n", (long long) stay1, (long long) stay2);
        std::string line;
        const bool ok = arena_src.place(profile_rank, arena_on_gpu(), line);
        std::fprintf(stderr, "lamina: %s\n", line.c_str());
        if (!ok) return 1;
    }
    // the GPU tiers rank after the main GPU's on the same counts, each after the ones before it; the last decays them
    std::string adapt_err;
    auto adapt = [&]() -> bool {
        const bool tiers_adapt = tier2.on() && !o.no_second_gpu_adapt;
        if (!tier.adapt(drive.d.usage, adapt_err, !tiers_adapt)) return false;
        if (!tiers_adapt) return true;
        for (int t = 0; t < n_tiers; ++t)
            if (!tiers[(size_t) t].adapt.adapt(drive.d.usage, adapt_err, t + 1 == n_tiers)) return false;
        return pump_gap(&drive, adapt_err);
    };
    drive.adapt = adapt;
    if (tier.on()) drive.tier1 = &tier;   // its copies in pieces, or at once without a GPU tier (Drive::tier1)
    drive.burst = !tier2.on();
    tier.set_wait(drive.burst);
    tier.set_decay(o.adapt_decay);
    for (TierSlot& ts : tiers) ts.adapt.set_decay(o.adapt_decay);
    drive.n_layers = g.n_layers;
    drive.ring_us.assign((size_t) ((lamina::kernels::kVerifyMaxT + 1) * g.n_layers), 0.0);

    // ---- the batched prompt path, and the second GPU's share of its experts (not --no-prompt-offload)
    auto setup_prompt_path = [&](lamina::prefill::Prefill& prefill, lamina::prefill::ExpertRunner& offload,
                                 PromptBuffers& pbuf, lamina::prefill::PromptSplit& split, std::string& e) -> bool {
        const bool off = o.second_gpu >= 0 && !o.no_prompt_offload;
        // a cache too small to lend the buffers would make them allocate on top of a card the cache filled to its
        // reserve, which then pages: the chunk halves until they fit in the lendable slots (it only reads slower)
        if (!o.no_prefill_borrow && d_res != nullptr && o.expert_cache > 0) {
            SlotLoan probe;
            probe.cache = &xcache;
            probe.keep = (int64_t) pin1.size();   // the pinned slots are never lent
            int64_t chunk = o.prefill_chunk;
            while (chunk > 256 && !probe.plan(lamina::prefill::Prefill::bytes_needed(g, ss, chunk, off))) chunk /= 2;
            if (chunk != o.prefill_chunk && probe.plan(lamina::prefill::Prefill::bytes_needed(g, ss, chunk, off))) {
                std::fprintf(stderr, "lamina: prompt chunk %lld -> %lld tokens so its buffers fit in the expert "
                                     "cache\n", (long long) o.prefill_chunk, (long long) chunk);
                o.prefill_chunk = chunk;
            }
        }
        if (off && !offload.init(o.second_gpu, o.main_gpu, nullptr, srcp, &gpu2.cache(),
                                 host_res2.empty() ? nullptr : host_res2.data(), g.n_expert, o.prefill_chunk, true, e))
            return false;
        if (!prefill.init(wt, g, ss, srcp, o.expert_cache > 0 ? &xcache : nullptr,
                          host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, main_cs, off ? &offload : nullptr, e))
            return false;
        if (!profile.empty()) {   // the experts a layer streams are copied ahead in the profile's order
            const size_t ne = (size_t) g.n_expert;
            std::vector<int32_t> order((size_t) g.n_layers * ne);
            std::vector<size_t> at((size_t) g.n_layers, 0);
            std::vector<uint8_t> seen(order.size(), 0);
            for (const auto& [l, x] : profile)
                if (!seen[(size_t) l * ne + (size_t) x]) {
                    seen[(size_t) l * ne + (size_t) x] = 1;
                    order[(size_t) l * ne + at[(size_t) l]++] = x;
                }
            for (size_t l = 0; l < (size_t) g.n_layers; ++l)
                for (size_t x = 0; x < ne; ++x)
                    if (!seen[l * ne + x]) order[l * ne + at[l]++] = (int32_t) x;
            split.set_order(order);
            prefill.set_order(std::move(order));
        }
        if (off) {   // --prompt-shift (without a profile the order is by id)
            prefill.set_shift(o.prompt_shift);
            offload.set_shift(o.prompt_shift);
        }
        pbuf.prefill = &prefill;
        pbuf.offload = off ? &offload : nullptr;
        // --prompt-split: the second GPU's stage (without its pinned buffers, no split)
        if (off && o.prompt_split > 0) {
            const auto ts = Clock::now();
            std::string se;
            // the vision path: the second stage's rope kernels read the image positions from their GPU's copy of the
            // table, set before any graph there captures one
            if (d_mrope != nullptr && d_mrope2 == nullptr) {
                const size_t bytes = mrope_host.size() * sizeof(int32_t);
                cudaSetDevice(o.second_gpu);
                int32_t* t = nullptr;
                const bool got = cudaMalloc(&t, bytes) == cudaSuccess;
                if (got && cudaMemcpy(t, mrope_host.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
                    lamina::kernels::mrope_table_set(t);
                    d_mrope2 = t;
                    mrope2_dev = o.second_gpu;
                    at_exit.undo.push_back([t, dev = o.second_gpu] {
                        cudaSetDevice(dev);
                        lamina::kernels::mrope_table_set(nullptr);
                        cudaFree(t);
                    });
                } else {
                    if (got) cudaFree(t);
                    (void) cudaGetLastError();
                    se = "no VRAM for the image position table on the second GPU";
                }
                cudaSetDevice(o.main_gpu);
            }
            if (se.empty() && split.init(prefill, wt, g, ss, o.mtp.empty() ? nullptr : &mtp, (int64_t) o.prompt_split,
                                         o.second_gpu, o.main_gpu, srcp, &gpu2.cache(),
                                         host_res2.empty() ? nullptr : host_res2.data(), o.prefill_chunk, se)) {
                pbuf.split = &split;
                std::fprintf(stderr, "lamina: --prompt-split: the second GPU runs layers %lld-%lld of a prompt's "
                                     "chunks, their %.0f MiB of weights copied from this GPU at each prompt (%.0f ms)\n",
                             (long long) split.at(), (long long) g.n_layers - 1, (double) split.weight_bytes / 1048576.0,
                             std::chrono::duration<double, std::milli>(Clock::now() - ts).count());
            } else {
                std::fprintf(stderr, "lamina: --prompt-split off: %s\n", se.c_str());
            }
        }
        pbuf.g = &g;
        pbuf.ss = &ss;
        if (d_res != nullptr) {
            pbuf.loan.cache = &xcache;
            pbuf.loan.res = &host_res;
            pbuf.loan.d_res = d_res;
            pbuf.loan.device = pbuf.loan.main_device = o.main_gpu;
            pbuf.loan.keep = (int64_t) pin1.size();
        }
        if (off) {
            pbuf.loan2.cache = &gpu2.cache();
            pbuf.loan2.res = &host_res2;
            pbuf.loan2.device = o.second_gpu;
            pbuf.loan2.main_device = o.main_gpu;
            pbuf.loan2.keep = (int64_t) pin2.size();
        }
        if (!pbuf.init(o.prefill_chunk, !o.no_prefill_borrow, e)) return false;
        if (pin1.empty() && pin2.empty()) return true;
        if (!pin2.empty() && !off) {   // the main GPU's share would stream them from RAM
            e = "--second-gpu-pin-gib needs the prompt path's share on the second GPU";
            return false;
        }
        // With experts pinned in the lowest slots, the largest loan each GPU's prompt path can take must fit above
        // them (one that did not would allocate its buffers on a card the cache fills).  Its prefetch area holds a
        // layer's experts that neither GPU holds: at most those the pins leave.
        const auto& lay = lamina::kernels::cpu::expert_layout();
        uint64_t stride = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            stride = std::max<uint64_t>(stride, (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(l)));
        std::vector<int64_t> held1((size_t) g.n_layers, 0), held12((size_t) g.n_layers, 0);
        for (const auto& pr : pin1) { ++held1[(size_t) pr.first]; ++held12[(size_t) pr.first]; }
        for (const auto& pr : pin2) ++held12[(size_t) pr.first];
        int64_t miss1 = 0, miss12 = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            miss1 = std::max<int64_t>(miss1, g.n_expert - held1[(size_t) l]);
            miss12 = std::max<int64_t>(miss12, g.n_expert - held12[(size_t) l]);
        }
        auto pre_area = [&](int64_t miss, bool remote) {
            const uint64_t b = (uint64_t) miss * stride;
            return std::min(b + b / 10, lamina::prefill::ExpertRunner::area_cap(o.prefill_chunk, remote));
        };
        auto check = [&](bool lent, lamina::core::ExpertCache& cache, int64_t keep, uint64_t worst, const char* gpu,
                         const char* flag) {
            SlotLoan probe;
            probe.cache = &cache;
            probe.keep = keep;
            if (lent && probe.plan(worst)) return true;
            const double left = (double) (cache.bytes() - (int64_t) probe.offset(keep)) / 1073741824.0;
            char buf[400];
            std::snprintf(buf, sizeof buf, "the prompt path needs %.2f GiB of the %s cache above its pinned experts, "
                          "which leave %.2f GiB: lower %s by %.2f", (double) worst / 1073741824.0, gpu, left, flag,
                          (double) worst / 1073741824.0 - left + 0.05);
            e = buf;
            return false;
        };
        if (!o.no_prefill_borrow && !pin1.empty() &&
            !check(pbuf.lend, xcache, (int64_t) pin1.size(),
                   lamina::prefill::Prefill::bytes_needed(g, ss, o.prefill_chunk, off) +
                       (off ? 0 : pre_area(miss1, false)),
                   "main GPU's", "--vram-pin-gib"))
            return false;
        if (!o.no_prefill_borrow && !pin2.empty() &&
            !check(pbuf.lend2, gpu2.cache(), (int64_t) pin2.size(),
                   lamina::prefill::ExpertRunner::bytes_needed(o.prefill_chunk, g.n_expert, true, true) +
                       pre_area(miss12, true), "second GPU's", "--second-gpu-pin-gib"))
            return false;
        return true;
    };


    // the verifier's options and the tiers' hooks into it (prefetch, the watch, the updates at the tail), set after
    // init by the serve loop and the one-shot run alike
    auto hook_verifier = [&](lamina::core::Verifier& ver) {
        ver.set_split(o.spec_split);
        ver.set_profile(o.window_profile);
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : o.pcie_mode == "kernel" ? 2
                          : 0);   // auto: DMA
        bool prefetching = false;
        for (int t = 0; t < n_tiers; ++t)
            prefetching = prefetching || (tiers[(size_t) t].gpu.prefetch_slots() > 0 && drive.d.tiers[t].prefetch_bytes > 0);
        if (prefetching) ver.set_predict(&drive_predict, &drive, o.adapt_every > 0 && o.adapt_swaps > 0);
        if (n_tiers > 0) {
            ver.set_tiers(n_tiers);
            ver.set_watch(&drive_watch, &drive);
        }
        if (drive.tier1 != nullptr || !drive.paced.empty()) {   // the tiers' updates start at the tail
            ver.set_tail(&drive_tail, &drive);
            ver.set_rows_in(&drive_rows_in, &drive, tier.guarded());
        }
    };

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length>
    //
    // (`ERR <message>` instead when a request cannot run; `QUIT` ends the process).  Every request starts from an
    // empty sequence (`session_zero`): the prompt goes through the batched prompt path and its last token through
    // the first verify window - the path all three model files share.  Decoding is greedy.
    if (o.serve) {
        if (o.spec < 2 || o.mtp.empty() || o.prefill_chunk <= 0 || thits.d_res == nullptr || host_res.empty()) {
            std::fprintf(stderr, "lamina serve: needs --spec T, --mtp DIR, --prefill CHUNK, --expert-profile P and "
                                 "--expert-cache\n");
            return 2;
        }
        lamina::prefill::Prefill prefill;
        lamina::prefill::ExpertRunner offload;
        lamina::prefill::PromptSplit psplit;   // (its second stage refers to the first: destroyed before it)
        PromptBuffers pbuf;
        DraftDump ddump;   // --dump-draft-data: every prompt position from the batched path, its final rows there
        ddump.dir = o.dump_draft_data;
        if (!ddump.dir.empty()) {
            o.feed_max = 0;
            o.prompt_split = 0;
        }
        if (!setup_prompt_path(prefill, offload, pbuf, psplit, err)) {
            std::fprintf(stderr, "lamina serve: %s\n", err.c_str());
            return 1;
        }
        if (pbuf.lend || pbuf.lend2)
            std::fprintf(stderr, "lamina serve: the prompt path borrows up to %lld cache slots (%.2f GiB), %lld (%.2f GiB) "
                                 "on the second GPU\n", (long long) pbuf.loan.slots(), (double) pbuf.loan.bytes() / 1073741824.0,
                         (long long) pbuf.loan2.slots(), (double) pbuf.loan2.bytes() / 1073741824.0);
        else
            std::fprintf(stderr, "lamina serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n");
        mem_mark("the head and the prompt path");
        // ---- THE ELASTIC K/V (--kv-grow; vmm.hpp; after Niko1221/Strata 0.1.40, 2fbe321 + 4da8a57).  Allocated for
        // the whole --max-context up front, the K/V takes VRAM the expert cache could hold more experts in (at 200K,
        // int8: 7,222 -> 6,142 slots on a 3090, -14% decode).  Now the K/V pools hold physical memory only for the
        // cells the requests reach.  When a request needs more, the slots just below the prompt path's loan give up
        // their experts (the CPU computes those from RAM, like any miss) and their whole chunks are mapped into the
        // K/V; a later short request hands them back and the slots refill with the profile's hottest experts.
        // Neither side's addresses move, so every captured graph stays valid.  The K/V's precision is unchanged.
        struct KvGrow {
            bool on = false;
            int64_t top = 0, lo = 0;     // slots [lo, top) hold no expert; their whole chunks may be with the K/V
            int64_t floor = 128;         // the cache keeps at least these slots
            int64_t step = 8192;         // the K/V grows in whole steps of cells
            int64_t cells = 0;           // what every K/V pool holds now
            std::vector<lamina::core::VmmChunk> spare;   // out of the cache, not (yet) in the K/V
            int64_t grows = 0, trims = 0, fresh = 0, evicted = 0, refilled = 0;
        } kvg;
        int kv_short = 0;   // requests in a row far shorter than the K/V holds (kvg_trim after two)
        {
            // The slots given up must stay below every loan the prompt path can take.  Its buffers grow with the
            // experts the cache does NOT hold (the prefetch area holds a layer's streamed experts, Prefill::bytes_for),
            // and every slot given up adds one: so the bound is the loan for the longest chunk with a whole layer
            // streamed, not the loan at start (that one let a later, larger loan reach the given-up slots).
            int64_t top = xcache.slots();
            if (pbuf.lend) {
                SlotLoan probe;
                probe.cache = &xcache;
                top = probe.plan(worst_loan(pbuf.offload != nullptr))
                          ? std::min<int64_t>(probe.first, pbuf.loan.first >= 0 ? pbuf.loan.first : top)
                          : 0;   // no room for the worst loan: nothing can be given up
            }
            if (const char* v = std::getenv("LAMINA_KV_GROW_STEP"); v != nullptr && std::atoll(v) > 0) kvg.step = std::atoll(v);
            // tests: the cache keeps this many slots (at or above the bound: the K/V grows into new VRAM only)
            if (const char* v = std::getenv("LAMINA_KV_GROW_FLOOR"); v != nullptr && std::atoll(v) > 0)
                kvg.floor = std::min<int64_t>(std::atoll(v), top);
            kvg.floor = std::max<int64_t>(kvg.floor, (int64_t) pin1.size());   // the pinned slots stay with their experts
            kvg.on = lamina::core::qsa_kv_elastic() && xcache.vmm_range() != nullptr && d_res != nullptr &&
                     !host_res.empty() && srcp != nullptr && top >= kvg.floor;
            if (kvg.on) {
                // The whole context's K/V must fit: in the cache's slots between its floor (the pinned ones, if any)
                // and the loan's, then in new VRAM once the cache is at its floor, as far as the VRAM free past
                // --vram-reserve-mib goes (`--expert-cache auto` leaves the reserve alone; a smaller cache leaves
                // more).  A context past that would stop the engine at its first request that long: "ERR the K/V
                // cannot grow to this prompt" (917,504 cells on a 3090: 13.40 GiB needed, ~12.2 GiB of slots).
                const uint64_t G = lamina::core::vmm_granularity();
                const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.floor) + G - 1) / G);
                const int64_t c1 = (int64_t) (xcache.slot_offset(top) / G);
                size_t free_b = 0, total_b = 0;
                cudaMemGetInfo(&free_b, &total_b);
                const int64_t spare = std::max<int64_t>((int64_t) free_b - ((int64_t) o.vram_reserve_mib << 20), 0);
                const int64_t room = std::max<int64_t>(c1 - c0, 0) + spare / (int64_t) G;
                auto need = [](int64_t cells) { return lamina::core::qsa_kv_elastic_need(cells + 64); };
                if (need(o.max_context) > room) {
                    int64_t fit = 0;   // the longest context, in whole K/V steps, whose K/V fits
                    for (int64_t lo = 1, hi = o.max_context / kvg.step; lo <= hi;) {
                        const int64_t mid = lo + (hi - lo) / 2;
                        if (need(mid * kvg.step) <= room) { fit = mid * kvg.step; lo = mid + 1; } else hi = mid - 1;
                    }
                    const double gib = (double) G / 1073741824.0;
                    char pins[64] = "";
                    if (!pin1.empty())
                        std::snprintf(pins, sizeof pins, ", or --vram-pin-gib %.2f lower",
                                      (double) (need(o.max_context) - room) * gib + 0.05);
                    std::fprintf(stderr, "lamina serve: the elastic K/V needs %.2f GiB for the whole context and the main "
                                         "GPU has %.2f for it (its cache's slots%s, the VRAM free past "
                                         "--vram-reserve-mib): --max-context %lld at most here%s\n",
                                 (double) need(o.max_context) * gib, (double) room * gib,
                                 pin1.empty() ? "" : " past the pinned ones", (long long) fit, pins);
                    return 1;
                }
            }
            if (kvg.on) {
                kvg.top = kvg.lo = top;
                kvg.cells = lamina::core::qsa_kv_elastic_cells();
                std::fprintf(stderr, "lamina serve: elastic K/V: %lld of %lld cells in VRAM (%.2f of %.2f GiB); the expert "
                                     "cache (%lld slots) gives the K/V room below slot %lld as the context grows\n",
                             (long long) kvg.cells, (long long) o.max_context,
                             (double) lamina::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                             (double) lamina::core::qsa_kv_elastic_full_bytes() / 1073741824.0,
                             (long long) xcache.slots(), (long long) top);
            } else if (lamina::core::qsa_kv_elastic()) {
                // nothing can grow it later: the whole context now, in new VRAM
                if (!lamina::core::qsa_kv_elastic_grow(o.max_context + 64, [] { return (lamina::core::VmmChunk) 0; })) {
                    std::fprintf(stderr, "lamina serve: elastic K/V: the cache cannot give slots up here and the whole "
                                         "K/V does not fit; start without --kv-grow\n");
                    return 1;
                }
                std::fprintf(stderr, "lamina serve: elastic K/V: the cache cannot give slots up here; the whole K/V is "
                                     "mapped (%lld cells)\n", (long long) lamina::core::qsa_kv_elastic_cells());
            }
        }
        // nothing running on the device, the adaptive tiers' copies landed (a swap in flight could still be writing a
        // slot given up here), and no adaptation round in progress
        auto kv_quiesce = [&] {
            drive.join_adapt();
            cudaDeviceSynchronize();
            tier.apply_pending(true);
            for (TierSlot& ts : tiers) ts.adapt.apply_pending(true);
            cudaDeviceSynchronize();
        };
        // Room for `cells` cells (rounded up to a step); false: no VRAM is left.
        auto kvg_ensure = [&](int64_t cells) -> bool {
            if (!kvg.on || cells <= kvg.cells) return true;
            const int64_t target = std::min<int64_t>(o.max_context + 64, (cells + kvg.step - 1) / kvg.step * kvg.step);
            const int64_t need = lamina::core::qsa_kv_elastic_need(target);
            if (need == 0) { kvg.cells = lamina::core::qsa_kv_elastic_cells(); return true; }
            kv_quiesce();
            lamina::core::VmmRange& r = *xcache.vmm_range();
            const uint64_t G = lamina::core::vmm_granularity();
            std::vector<int32_t> owner;   // slot -> residency index
            // The slots given up are the ones just below the prompt path's loan, and the loan is most of the cache:
            // they hold experts of middling heat.  So an expert there that is hotter than the coldest one in the loan
            // region moves into that one's slot (a device copy), and the coldest is given up instead - the cache then
            // holds what a smaller cache would.  Heat: the adaptive tier's routing counts, then the profile's rank.
            std::vector<int32_t> rank;
            std::vector<std::pair<float, int32_t>> cold;   // the loan region's residents, hottest first
            auto heat = [&](int32_t i) -> float {
                const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[(size_t) i];
                return u * 1048576.0f - (float) rank[(size_t) i];
            };
            const auto& lay = lamina::kernels::cpu::expert_layout();
            int64_t gave = 0, moved = 0;
            auto freeable = [&]() -> int64_t {   // mapped chunks wholly inside [lo, top): a chunk a live slot shares stays
                const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
                const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
                int64_t k = 0;
                for (int64_t c = c0; c < c1; ++c) k += r.mapped(c) ? 1 : 0;
                return k;
            };
            while ((int64_t) kvg.spare.size() + freeable() < need && kvg.lo > kvg.floor) {
                if (owner.empty()) {
                    owner.assign((size_t) xcache.slots(), -1);
                    for (size_t i = 0; i < host_res.size(); ++i)
                        if (host_res[i] >= 0 && host_res[i] < xcache.slots()) owner[(size_t) host_res[i]] = (int32_t) i;
                    rank.assign(host_res.size(), (int32_t) profile.size());
                    for (size_t k = 0; k < profile.size(); ++k)
                        rank[(size_t) profile[k].first * (size_t) g.n_expert + (size_t) profile[k].second] = (int32_t) k;
                    for (size_t i = 0; i < host_res.size(); ++i)
                        if (host_res[i] >= kvg.top) cold.emplace_back(heat((int32_t) i), (int32_t) i);
                    std::sort(cold.begin(), cold.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                }
                --kvg.lo;
                ++gave;
                if (const int32_t oi = owner[(size_t) kvg.lo]; oi >= 0) {
                    const int64_t layer = oi / g.n_expert;
                    int32_t vi = -1;
                    if (!cold.empty() && cold.back().first < heat(oi)) {
                        vi = cold.back().second;
                        const int32_t vs = host_res[(size_t) vi];
                        if (xcache.slot_offset(vs + 1) - xcache.slot_offset(vs) <
                                (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(layer)) ||
                            cudaMemcpy(xcache.device_slot(vs), xcache.device_slot((int32_t) kvg.lo),
                                       (size_t) lay.blob_bytes(layer), cudaMemcpyDeviceToDevice) != cudaSuccess)
                            vi = -1;
                        else {
                            cold.pop_back();
                            host_res[(size_t) oi] = vs;
                            host_res[(size_t) vi] = lamina::core::kNotResident;
                            ++moved;
                        }
                    }
                    if (vi < 0) host_res[(size_t) oi] = lamina::core::kNotResident;
                    ++kvg.evicted;
                }
            }
            // a device-to-device cudaMemcpy does not wait for the copy: the moves still read the slots given up, and
            // a chunk unmapped under a copy faults it
            if (gave > 0 && cudaDeviceSynchronize() != cudaSuccess) {
                std::fprintf(stderr, "lamina serve: moving experts for the K/V failed: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            tier.drop_free_slots(kvg.lo, kvg.top);   // no adaptive move may target a slot given up
            {
                const int64_t c0 = (int64_t) ((xcache.slot_offset(kvg.lo) + G - 1) / G);
                const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
                for (int64_t c = c0; c < c1; ++c)
                    if (r.mapped(c))
                        if (const lamina::core::VmmChunk h = r.unmap(c)) kvg.spare.push_back(h);
            }
            if (gave > 0 && res_put(d_res, host_res) != cudaSuccess) return false;
            int64_t fresh = 0;
            const bool ok = lamina::core::qsa_kv_elastic_grow(target, [&]() -> lamina::core::VmmChunk {
                if (kvg.spare.empty()) { ++fresh; return 0; }   // the cache is at its floor: new memory
                const lamina::core::VmmChunk h = kvg.spare.back();
                kvg.spare.pop_back();
                return h;
            });
            kvg.fresh += fresh;
            kvg.cells = lamina::core::qsa_kv_elastic_cells();
            ++kvg.grows;
            std::fprintf(stderr, "lamina serve: K/V grown to %lld cells (%.2f GiB); the expert cache gave %lld slots for "
                                 "it, %lld hotter experts moved to colder ones' slots (%lld of %lld slots hold experts)%s\n",
                         (long long) kvg.cells, (double) lamina::core::qsa_kv_elastic_mapped_bytes() / 1073741824.0,
                         (long long) gave, (long long) moved, (long long) (kvg.lo + (xcache.slots() - kvg.top)),
                         (long long) xcache.slots(), fresh > 0 ? " - and new VRAM, the cache being at its floor" : "");
            if (!ok)
                std::fprintf(stderr, "lamina serve: the K/V could not grow to %lld cells (%s)\n", (long long) target,
                             cudaGetErrorString(cudaGetLastError()));
            return ok;
        };
        // A request that needs far fewer cells than the K/V holds gives the rest back: the slots refill with the
        // profile's hottest experts the GPU does not hold.  False: a CUDA failure left the tier half-changed.
        auto kvg_trim = [&](int64_t cells) -> bool {
            if (!kvg.on) return true;
            const int64_t target = std::max<int64_t>(kvg.step, (cells + kvg.step - 1) / kvg.step * kvg.step);
            if (kvg.cells < target + 2 * kvg.step || kvg.lo >= kvg.top) return true;
            kv_quiesce();
            lamina::core::qsa_kv_elastic_shrink(target, [&](lamina::core::VmmChunk h) { kvg.spare.push_back(h); });
            kvg.cells = lamina::core::qsa_kv_elastic_cells();
            lamina::core::VmmRange& r = *xcache.vmm_range();
            const uint64_t G = lamina::core::vmm_granularity();
            const int64_t c1 = (int64_t) (xcache.slot_offset(kvg.top) / G);
            {   // one run from the lowest unmapped chunk, as long as the chunks handed back last (one access call)
                const int64_t c0 = (int64_t) (xcache.slot_offset(kvg.lo) / G);
                int64_t cend = c0, need = (int64_t) kvg.spare.size();
                while (cend < c1 && need > 0) need -= r.mapped(cend++) ? 0 : 1;
                if (!r.map_range(c0, cend, [&]() -> lamina::core::VmmChunk {
                        if (kvg.spare.empty()) return 0;
                        const lamina::core::VmmChunk h = kvg.spare.back();
                        kvg.spare.pop_back();
                        return h;
                    }))
                    std::fprintf(stderr, "lamina serve: the expert cache could not take its VRAM back from the K/V\n");
            }
            const int64_t lo0 = kvg.lo;
            while (kvg.lo < kvg.top) {   // a slot whose chunks are all mapped again holds an expert again
                const int64_t a = (int64_t) (xcache.slot_offset(kvg.lo) / G);
                const int64_t b = (int64_t) ((xcache.slot_offset(kvg.lo + 1) + G - 1) / G);
                bool all = true;
                for (int64_t c = a; c < b && all; ++c) all = r.mapped(c);
                if (!all) break;
                ++kvg.lo;
            }
            // the refills on one stream and one sync (a blocking copy each is a driver round trip per expert)
            const auto& lay = lamina::kernels::cpu::expert_layout();
            std::vector<std::pair<size_t, int32_t>> filled;   // (residency index, slot), resident once landed
            size_t pi = 0;
            for (int64_t s = lo0; s < kvg.lo; ++s) {
                const uint64_t room = xcache.slot_offset(s + 1) - xcache.slot_offset(s);
                for (; pi < profile.size(); ++pi) {
                    const int32_t l = profile[pi].first, e = profile[pi].second;
                    const size_t i = (size_t) l * (size_t) g.n_expert + (size_t) e;
                    if (host_res[i] >= 0 ||
                        (uint64_t) lamina::core::ExpertCache::slot_stride((int64_t) lay.blob_bytes(l)) > room)
                        continue;
                    const uint8_t* b = srcp->blob(l, e);
                    if (b == nullptr) continue;
                    cudaMemcpyAsync(xcache.device_slot((int32_t) s), b, (size_t) lay.blob_bytes(l),
                                    cudaMemcpyHostToDevice, nullptr);
                    filled.emplace_back(i, (int32_t) s);
                    ++pi;
                    break;
                }
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                std::fprintf(stderr, "lamina serve: refilling the expert cache failed: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return false;
            }
            for (const auto& [i, s] : filled) host_res[i] = s;
            kvg.refilled += (int64_t) filled.size();
            if (res_put(d_res, host_res) != cudaSuccess) return false;
            for (const lamina::core::VmmChunk h : kvg.spare) lamina::core::vmm_chunk_free(h);   // new ones, if any
            kvg.spare.clear();
            ++kvg.trims;
            std::fprintf(stderr, "lamina serve: K/V trimmed to %lld cells; %lld slots back to the expert cache, refilled "
                                 "from the profile\n", (long long) kvg.cells, (long long) (kvg.lo - lo0));
            return true;
        };
        lamina::core::Verifier ver;
        lamina::core::VerifyHits vh;
        vh.res = host_res.data();
        vh.cache_base = thits.cache_base;
        vh.slot_off = xcache.slot_offsets();
        vh.slots = xcache.slots();
        vh.blob = thits.blob;
        ver.set_split_head(split_head.get());
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.max_window(), err) ||
            !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "lamina serve: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the verifier and the drafter's binding");
        hook_verifier(ver);
        prefill.draft = &mtp;
        drive.d.plan = ver.plan_sink();
        drive.d.host_res = ver.residency();   // the snapshot the windows decide their GPU hits by
        drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
        ver.set_pcie_share(drive.d.pcie_num > 0);
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        // ---- #477 --expert-profile-save: the routing counted since the start (the verify windows' entries, and each
        // prompt's as the tiers take it: kPromptUsage a layer), saved as a profile.  Off: nothing is counted or written.
        std::vector<double> prompt_heat;
        if (!o.expert_profile_save.empty()) {
            if (drive.d.usage.empty())
                std::fprintf(stderr, "lamina serve: --expert-profile-save needs the adaptive tiers (--adapt-every and "
                                     "--adapt-swaps above 0): nothing will be saved\n");
            else {
                drive.d.routed.assign(drive.d.usage.size(), 0);
                prompt_heat.assign(drive.d.usage.size(), 0.0);
            }
        }
        Clock::time_point profile_saved_at = Clock::now();
        bool profile_dirty = false;   // a request ran since the last save
        // the counted routing blended with the profile the engine started from (blend_learned_profile); through <P>.tmp
        auto save_profile = [&](const char* why) {
            if (prompt_heat.empty()) return;
            const auto t0 = Clock::now();
            std::vector<double> heat(prompt_heat.size());
            for (size_t i = 0; i < heat.size(); ++i) heat[i] = (double) drive.d.routed[i] + prompt_heat[i];
            const auto ranked = lamina::core::blend_learned_profile(g.n_layers, g.n_expert, heat, profile);
            std::string e;
            if (lamina::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
                std::fprintf(stderr, "lamina serve: expert profile saved to %s (%s, %.0f ms)\n",
                             o.expert_profile_save.c_str(), why,
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            else
                std::fprintf(stderr, "lamina serve: the expert profile was not saved: %s\n", e.c_str());
            profile_saved_at = Clock::now();
            profile_dirty = false;
        };
        // ---- the prompt cache (lamina/core/prompt_cache.hpp)
        std::vector<lamina::core::QsaState*> kv_states;
        for (int64_t i = 0; i < g.n_qsa_layers(); ++i) kv_states.push_back(&ss.qsa_states[i]);
        kv_states.push_back(&mtp.kv_state());
        lamina::core::PromptCache pcache(g, ss, kv_states, main_cs, o.prompt_cache,
                                         (uint64_t) o.conversation_cache_mib << 20, (size_t) o.conversation_cache_slots,
                                         o.conversation_cache_min_tokens);
        // the elastic K/V: a parked conversation's restore (or the live one's park) copies all its cells
        if (kvg.on) pcache.ensure_kv = [&](int64_t cells) { return kvg_ensure(cells + 256); };
        if (pcache.enabled())
            std::fprintf(stderr, "lamina serve: prompt cache on: up to %d checkpoints of %.1f MiB per conversation, "
                                 "%lld MiB for the conversations requests replace%s\n", o.prompt_cache,
                         (double) pcache.ckpt_bytes() / 1048576.0, (long long) o.conversation_cache_mib,
                         o.conversation_cache_slots > 0
                             ? (", at most " + std::to_string(o.conversation_cache_slots) + " of them").c_str() : "");
        // the CACHE line (the server's Conversation cache card): at READY, then after each request before its DONE
        auto print_cache = [&] {
            if (!pcache.enabled()) return;
            const auto cs = pcache.state();
            serve_print("CACHE live=%lld checkpoints=%zu parked=%zu parked_bytes=%llu parked_budget=%llu slots=%zu "
                        "evictions=%zu\n", (long long) cs.live_cells, cs.checkpoints, cs.parked,
                        (unsigned long long) cs.parked_bytes, (unsigned long long) cs.parked_budget, cs.slots,
                        cs.evictions);
        };
        // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
        // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
        std::atomic<bool> stop_req{false};
        std::mutex in_mu;
        std::condition_variable in_cv;
        std::deque<std::string> in_lines;
        bool in_eof = false;
        const lamina::program::ServeIo& io = lamina::program::serve_io();
        std::thread in_reader([&] {
            std::string l;
            while (io.read_line(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop_req.store(true); continue; }
                std::lock_guard<std::mutex> lk(in_mu);
                in_lines.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_eof = true;
            in_cv.notify_one();
        });
        lamina::program::ReaderEnd reader_end{in_reader, io};
        // Ctrl+C in the console ends the engine as QUIT does, before any request queued: the running one stops at its
        // next chunk or window, then the exit work runs (the learned profile).  In-process the host handles it.
        std::optional<lamina::platform::ConsoleQuit> console_quit;
        if (!io.in_process)
            console_quit.emplace([&] {
                stop_req.store(true);
                std::lock_guard<std::mutex> lk(in_mu);
                in_lines.push_front("QUIT");
                in_cv.notify_one();
            });
        // --arena-place: after a request, the arena's swaps for the residency the tiers left, a slice at a time while
        // no request waits (the GPUs' copies from the arena are done first: nothing reads it meanwhile)
        const char* replace_env = std::getenv("LAMINA_ARENA_REPLACE");   // =0: the A/B arm without it
        const bool replace = o.arena_place && srcp == &arena_src && !profile_rank.empty() &&
                             !(replace_env != nullptr && std::atoi(replace_env) == 0);
        size_t replace_left = 0, replace_planned = 0;
        Clock::time_point replace_t0{};
        auto replan = [&] {
            cudaSetDevice(o.main_gpu);
            cudaDeviceSynchronize();
            for (const int d : o.tier_gpus) {
                cudaSetDevice(d);
                cudaDeviceSynchronize();
            }
            cudaSetDevice(o.main_gpu);
            // the CPU's experts the most routed lately first (the tiers' counts), the profile's order between equals
            const size_t n = profile_rank.size();
            std::vector<int32_t> order(n), rank(n);
            for (size_t i = 0; i < n; ++i) order[i] = (int32_t) i;
            const std::vector<float>& u = drive.d.usage;
            std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
                const float ua = u.empty() ? 0.0f : u[(size_t) a], ub = u.empty() ? 0.0f : u[(size_t) b];
                return ua != ub ? ua > ub : profile_rank[(size_t) a] < profile_rank[(size_t) b];
            });
            for (size_t k = 0; k < n; ++k) rank[(size_t) order[k]] = (int32_t) k;
            replace_planned = replace_left = arena_src.replan(rank, arena_on_gpu());
            replace_t0 = Clock::now();
        };
        auto next_line = [&](std::string& out) -> bool {
            std::unique_lock<std::mutex> lk(in_mu);
            while (replace_left > 0 && in_lines.empty() && !in_eof) {
                lk.unlock();
                replace_left = arena_src.replace_some(16);
                if (replace_left == 0)
                    std::fprintf(stderr, "lamina serve: --arena-place: %zu swaps between requests in %.0f ms\n",
                                 replace_planned, std::chrono::duration<double, std::milli>(Clock::now() - replace_t0).count());
                lk.lock();
            }
            if (replace_left > 0)
                std::fprintf(stderr, "lamina serve: --arena-place: %zu of %zu swaps made before the request\n",
                             replace_planned - replace_left, replace_planned);
            // the learned profile, while no request waits, once its interval has passed since the last save
            const bool periodic = !prompt_heat.empty() && o.expert_profile_save_min > 0;
            const auto every = std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(o.expert_profile_save_min * 60.0));
            while (in_lines.empty() && !in_eof) {
                if (!periodic || !profile_dirty) {
                    in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
                    break;
                }
                if (Clock::now() >= profile_saved_at + every) {
                    lk.unlock();
                    save_profile("between requests");
                    lk.lock();
                } else {
                    in_cv.wait_until(lk, profile_saved_at + every, [&] { return !in_lines.empty() || in_eof; });
                }
            }
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        // PP <done> <total> <ms> <tok/s>: the prompt's progress, one line per chunk (the server's progress and
        // keep-alives); `pp_at` is where the running segment starts, `pp_total` 0 outside a prompt
        int64_t pp_from = 0, pp_at = 0, pp_total = 0;
        Clock::time_point pp_t0 = Clock::now();
        auto pp_line = [&](int64_t done) {
            if (pp_total <= 0) return;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            serve_print("PP %lld %lld %.0f %.1f\n", (long long) (done - pp_from), (long long) pp_total, ms,
                        ms > 0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
        };
        prefill.should_stop = [&] {
            pp_line(pp_at + prefill.processed());
            return stop_req.load();
        };
        // LAMINA_TRACE=1: one stderr line per step of a request (the log shows where a request stops)
        const bool trace = std::getenv("LAMINA_TRACE") != nullptr;
        const Clock::time_point trace_t0 = Clock::now();
        auto tr = [&](const char* what, long long a = -1, long long b = -1) {
            if (!trace) return;
            std::fprintf(stderr, "lamina trace: %s %lld %lld (%.1f ms)\n", what, a, b,
                         std::chrono::duration<double, std::milli>(Clock::now() - trace_t0).count());
            std::fflush(stderr);
        };
        {
            // what is left once everything is allocated: under WDDM a GPU filled to the brim does not fail, it pages -
            // and a page-in while the verify graph spins on a host flag stalls the request for good
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            std::string others;
            for (int t = 0; t < n_tiers; ++t) {
                size_t free_t = 0;
                cudaSetDevice(o.tier_gpus[(size_t) t]);
                cudaMemGetInfo(&free_t, &total_b);
                others += ", " + std::to_string(free_t >> 20) + " MiB on the " + kTierName[t] + " GPU";
            }
            cudaSetDevice(o.main_gpu);
            std::fprintf(stderr, "lamina serve: %lld MiB of VRAM free with everything loaded%s%s\n",
                         (long long) (free_b >> 20), others.c_str(),
                         free_b < ((size_t) 128 << 20)
                             ? " - LOW: requests may stall; lower --max-context or raise --vram-reserve-mib" : "");
        }
        // the penalty-history buffer: one row per window row (`penalty_rows`), allocated once for the widest window
        constexpr int kPenaltyWindowCap = 4096;
        constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) lamina::kernels::kVerifyMaxT;
        int32_t* d_hist = nullptr;
        std::vector<int32_t> hist_stage(kHistSlots, -1);
        if (cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "lamina serve: the penalty-history allocation failed\n");
            return 1;
        }
        at_exit.undo.push_back([d_hist] { cudaFree(d_hist); });
        {
            // what the server's Monitor tab shows; the experts in VRAM are every GPU's, as its VRAM figures
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            int64_t vram_slots = xcache.slots(), vram_bytes = xcache.bytes();
            for (int t = 0; t < n_tiers; ++t) {
                vram_slots += tiers[(size_t) t].gpu.cache().slots();
                vram_bytes += tiers[(size_t) t].gpu.cache().bytes();
            }
            serve_print("INFO context=%lld kv=%s kv_resident=0 expert_slots=%lld expert_cache_mib=%lld spec=%d "
                        "mtp_max=%d lookup=%d vram_free_mib=%lld arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                        "spec_min_p=%.2f second_gpu=%d tail_role_token=%lld engine=" LAMINA_VERSION " fork=lamina\n",
                        (long long) o.max_context, o.kv.c_str(), (long long) vram_slots,
                        (long long) (vram_bytes >> 20), o.spec, o.spec, o.spec_lookup, (long long) (free_b >> 20),
                        (long long) (lamina::kernels::cpu::expert_layout().total >> 20), pool.workers(), o.pcie_frac,
                        o.spec_min_p, o.second_gpu, (long long) o.tail_role_token);
        }
        print_cache();
        // --ple-io ram: READY only once the table is in RAM.  While it loads, its 28.8 GB of reads compete with the
        // requests and the rows still come from the SSD: a first chat right after READY decoded at ~21 tok/s instead
        // of ~95 and a 32K prompt read 13% slower.  LAMINA_PLE_RAM_NOWAIT=1: before.
        if (ple_table.is_open() && ple_table.mode() == lamina::kernels::PleIo::Ram && !ple_table.ram_ready() &&
            !std::getenv("LAMINA_PLE_RAM_NOWAIT")) {
            const auto w0 = std::chrono::steady_clock::now();
            ple_table.wait_ram_load();
            std::fprintf(stderr, "lamina serve: waited %.1f s for the PLE table before READY (%s)\n",
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(),
                         ple_table.ram_ready() ? "in RAM" : "its rows stay on the SSD");
        }
        serve_print("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
        std::string line;
        int64_t rounds = 0;
        const int S = o.spec, W = o.max_window();   // the draft layer's windows; with prompt lookup's
        // prompt lookup: the text so far, kept across requests (a conversation's next request shares its prefix)
        lamina::spec::SuffixDrafter lookup(o.spec_lookup, 32, o.spec_lookup > 0 ? (size_t) o.max_context + 16 : 16);
        std::vector<int32_t> ids32, ldrafts((size_t) W, 0);
        constexpr int64_t kImStart = 248045;   // <|im_start|>: a new turn begins here
        // The previous request's prompt: where a new one leaves it, it gets a checkpoint (see `fork` below).
        std::vector<int64_t> prev_prompt;
        // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
        // or more lamina-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
        // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
        // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
        constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
        bool mrope_identity = true;
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        std::vector<lamina::core::ImgKey> req_imgs;
        bool poisoned = false;   // the last request ended on non-finite logits: the next one reads its prompt anew
        // The expert tiers' counters over a request, as a TIERS line before its DONE (whose positional fields are
        // upstream's protocol): the moves of every GPU's adaptive tier, the GPU tiers' work and prefetch (the first
        // tier's fields are named second_*, the next ones' third_* and fourth_*; second_cpu_layers counts the layers
        // the CPU took all the tiers' shares of).
        struct TierCounters {
            int64_t main_swaps, main_fills, main_dropped, cpu_layers;
            struct Gpu {
                int64_t entries, experts, layers, swaps, fills, dropped, prefetched, prefetch_used;
            } gpu[lamina::core::kMaxGpuTiers];
        };
        const auto tier_counters = [&]() {
            TierCounters c{tier.swaps, tier.fills, tier.dropped, drive.d.tier_skipped, {}};
            for (int t = 0; t < n_tiers; ++t) {
                const lamina::core::SecondGpu& gp = tiers[(size_t) t].gpu;
                const lamina::core::AdaptiveTier& ad = tiers[(size_t) t].adapt;
                c.gpu[t] = {gp.entries_done, gp.experts, gp.layers, ad.swaps, ad.fills, ad.dropped,
                            gp.prefetch_copied, gp.prefetch_used};
            }
            return c;
        };
        TierCounters req_tiers{};
        while (next_line(line)) {
            if (line == "QUIT") break;
            stop_req.store(false);   // a STOP that arrived between requests is stale
            err.clear();             // and so is a stopped request's "cancelled"
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) {
                serve_print("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>\n");
                continue;
            }
            char* endp = nullptr;
            const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
            // optional keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F, penalty_last_n=N,
            // penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N, spec_min_p=F (this request's draft floor),
            // score=ID:ID:... (an `LP` line with those ids' log-probabilities after the prompt, before the first `T`);
            // unknown keys are skipped.  Absent keys: greedy, no penalties.
            float req_temperature = 0.0f, req_top_p = 1.0f, req_min_p = 0.0f;
            int req_top_k = 20;   // the sampled path keeps 1..64 candidates
            unsigned long long req_seed = 0;
            float req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
            int req_penalty_last_n = 0;
            double req_spec_min_p = o.spec_min_p;
            std::vector<int32_t> req_score;   // score=ID:ID:...: see the first window
            // img_keys=<hex>,<hex>,...: each picture's key from the server (its normalized bytes' hash), in prompt
            // order; the conversation cache then matches a picture by key and grid, not by its embedding floats
            std::vector<uint64_t> req_img_keys;
            if (endp != nullptr) {
                for (;;) {
                    while (*endp == ' ') ++endp;
                    const char* start = endp;
                    while (*endp != '\0' && *endp != ' ') ++endp;
                    if (endp == start) break;
                    const std::string kv(start, (size_t) (endp - start));
                    const size_t eq = kv.find('=');
                    if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
                    const std::string key = kv.substr(0, eq);
                    const char* v = kv.c_str() + eq + 1;
                    const float fv = std::strtof(v, nullptr);
                    if (key == "temperature") req_temperature = fv;
                    else if (key == "top_p") req_top_p = fv;
                    else if (key == "top_k") req_top_k = std::atoi(v);
                    else if (key == "min_p") req_min_p = fv;
                    else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(v);
                    else if (key == "penalty_repeat") req_penalty_repeat = fv;
                    else if (key == "penalty_freq") req_penalty_freq = fv;
                    else if (key == "penalty_present") req_penalty_present = fv;
                    else if (key == "seed") req_seed = std::strtoull(v, nullptr, 10);
                    else if (key == "spec_min_p") req_spec_min_p = std::clamp((double) fv, 0.0, 1.0);
                    else if (key == "img_keys") {
                        req_img_keys.clear();
                        for (const char* q = v; *q != '\0';) {
                            char* qe = nullptr;
                            const unsigned long long hk = std::strtoull(q, &qe, 16);
                            if (qe == q || (*qe != ',' && *qe != '\0')) { req_img_keys.clear(); break; }
                            req_img_keys.push_back((uint64_t) hk);
                            q = *qe == ',' ? qe + 1 : qe;
                        }
                    }
                    else if (key == "score") {
                        for (const char* q = v; *q != '\0';) {
                            char* qe = nullptr;
                            const long id = std::strtol(q, &qe, 10);
                            if (qe == q) break;
                            req_score.push_back((int32_t) id);
                            q = (*qe == ':') ? qe + 1 : qe;
                        }
                    }
                }
            }
            std::string emb_path;
            if (geni && endp != nullptr) {
                while (*endp == ' ') ++endp;
                char* gap = std::strchr(endp, ' ');
                if (gap != nullptr) { emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
            }
            std::vector<int64_t> ids;
            std::string pe;
            if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !parse_i64_list(endp, ids, pe)) {
                serve_print("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
                continue;
            }
            const int64_t n = (int64_t) ids.size();
            req_imgs.clear();
            if (geni && !o.vision) { serve_print("ERR this engine was started without --vision\n"); continue; }
            if (geni || !mrope_identity) {
                // positions for every cell this request can reach; the identity again for a text request
                std::string ve;
                row_ptr.assign((size_t) n, nullptr);
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                    mrope_host[(size_t) c * 3] = (int32_t) t;
                    mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                    mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
                };
                if (!geni) {
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                } else {
                    std::vector<lamina::program::VisionImage> imgs;
                    img_rows.clear();
                    ve = lamina::program::read_vision_records(emb_path, g.n_embd, img_rows, imgs);
                    int64_t p = 0, i = 0;
                    size_t k = 0;
                    while (ve.empty() && i < n) {
                        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                        if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                        const lamina::program::VisionImage& im = imgs[k++];
                        req_imgs.push_back({i, lamina::program::vision_image_key(
                                                   im, req_img_keys.size() == imgs.size() ? &req_img_keys[k - 1] : nullptr,
                                                   img_rows.data(), g.n_embd)});
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                                ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                                     " rows but fewer <|image_pad|> tokens";
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                            const int64_t y = j / im.nx, x = j % im.nx;
                            put(i + j, p, p + y, p + x);
                            row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                        }
                        i += im.n;
                        p += std::max(im.nx, im.ny);
                    }
                    if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
                    if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
                    for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
                }
                tr("positions built", (long long) img_rows.size());
                cudaDeviceSynchronize();
                tr("device idle");
                if (ve.empty() && !mrope_upload()) ve = "the image position upload failed";
                if (!ve.empty()) {
                    // leave the table as the identity so the next text request is untouched
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                    mrope_upload();
                    mrope_identity = true;
                    serve_print("ERR %s\n", ve.c_str());
                    continue;
                }
                mrope_identity = !geni;
            }
            prefill.embd_rows = geni ? row_ptr.data() : nullptr;
            if (n + max_new + 8 > o.max_context) {
                serve_print("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                            (long long) max_new, (long long) o.max_context);
                continue;
            }
            bool bad = false;
            for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
            if (bad) { serve_print("ERR a token id is outside the vocabulary\n"); continue; }
            // the elastic K/V: room for this prompt before the prompt cache restores or the prompt path writes a cell
            // (the decode grows it further).  A K/V far longer than the request is given back only after two such
            // requests in a row: an agent's short side requests (a title, a summary) between the turns of a long
            // conversation would otherwise shrink it and grow it again every turn.
            if (kvg.on) {
                if (n + 256 > kvg.cells) {
                    kv_short = 0;
                    if (!kvg_ensure(n + 256)) {   // the tier may be half-changed: stop, as for a CUDA failure
                        serve_print("ERR the K/V cannot grow to this prompt: no VRAM is left\n");
                        return 1;
                    }
                } else if (kvg.cells >= std::max<int64_t>(kvg.step, (n + 256 + kvg.step - 1) / kvg.step * kvg.step) +
                                           2 * kvg.step) {
                    // never below the live sequence: the prompt cache may stash all of its cells next
                    if (++kv_short >= 2 && !kvg_trim(std::max<int64_t>(n, pcache.live_cells()) + 256)) {
                        serve_print("ERR the K/V could not give its VRAM back to the expert cache\n");
                        return 1;
                    }
                } else {
                    kv_short = 0;
                }
            }
            lamina::kernels::SamplerParams req_sp;
            req_sp.greedy = req_temperature <= 0.0f;
            req_sp.temperature = req_temperature;
            req_sp.top_p = req_top_p;
            req_sp.top_k = req_top_k;
            req_sp.seed = req_seed ? req_seed
                                   : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
            req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
            req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
            req_sp.penalty_repeat = req_penalty_repeat;
            req_sp.penalty_freq = req_penalty_freq;
            req_sp.penalty_present = req_penalty_present;
            ver.set_sampling(req_sp);
            const int hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
            ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            req_tiers = tier_counters();
            const Clock::time_point r0 = Clock::now();
            auto report = [&](const std::string& e) {   // a failed request: the engine exits after it
                std::fprintf(stderr, "lamina serve: %s\n", e.c_str());
                serve_print("ERR %s\n", e.c_str());
                exit_on_cuda_fault();
            };
            // ---- the prompt cache: continue from the longest prefix it holds (or start from an empty sequence).
            // GENI requests are cacheable too, but only after the parser has supplied embedding/grid identities.
            const bool track = pcache.enabled();
            if (poisoned) {
                pcache.clear();
                poisoned = false;
            }
            const int64_t reuse = pcache.begin(ids, req_imgs);
            tr("request", n, reuse);
            if (!ddump.dir.empty()) {
                if (!ddump.open(err)) { report(err); return 1; }
                const int64_t HCN = g.hc * g.n_embd;
                prefill.on_final = [&ddump, &ids, HCN](const float* R, int64_t T, int64_t p0, void* s) {
                    cudaStreamSynchronize((cudaStream_t) s);
                    ddump.put_rows(R, T, HCN);
                    for (int64_t t = 0; t < T; ++t) ddump.put_meta((int32_t) (p0 + t), (int32_t) ids[(size_t) (p0 + t)], -1, nullptr, 0);
                };
            }
            mtp.set_prompt_len(n);
            std::vector<int32_t> drafts((size_t) W, 0), window((size_t) W), outv((size_t) W);
            std::vector<float> dprob((size_t) W, 0.0f);
            // ---- the rest of the prompt, [reuse, n - 1).  A turn usually shares everything before its prompt's
            // last <|im_start|> with the next one (chat templates re-render the answer, e.g. without its
            // reasoning), so that position gets a checkpoint, and so do the prompt's end (a regenerated answer)
            // and the end of every `step` cells of a long prompt.  Short parts go through verify windows.
            int64_t q = 0;
            if (track)
                for (int64_t i = n - 2; i > reuse; --i)
                    if (ids[(size_t) i] == kImStart) { q = i; break; }
            if (q > 0 && o.tail_role_token >= 0)   // a trailing effort turn stays out of the checkpoint
                for (int64_t i = q - 1; i > reuse; --i)
                    if (ids[(size_t) i] == kImStart) {
                        if (ids[(size_t) i + 1] == o.tail_role_token) q = i;
                        break;
                    }
            // An agent sends one long prefix again and again with a different tail (a classifier's, a coder's, a
            // critic's question after the same context; a transcript that grows inside one user message): none of
            // the cuts above falls where those prompts part.  Two more checkpoints for them: where this prompt leaves
            // the previous one (the next request sharing that prefix resumes there), and, when the last message grows
            // (the previous prompt held its start already), a few cells before its <|im_end|> (the next request parts
            // there; the margin covers a seam that tokenizes differently once text follows it).  An agent's turn
            // adds a message instead, and takes neither.
            constexpr int64_t kImEnd = 248046;
            int64_t fork = 0, msg_end = 0;
            if (track) {
                const int64_t lim = std::min<int64_t>((int64_t) prev_prompt.size(), n - 1);
                int64_t d = 0;
                while (d < lim && prev_prompt[(size_t) d] == ids[(size_t) d]) ++d;
                if (d > reuse + 256 && d < n - 1 && d != q) fork = d;
                for (int64_t i = (q > 0 ? q : n - 1) - 1; i > reuse; --i)
                    if (ids[(size_t) i] == kImEnd) {
                        int64_t start = i;   // the message's <|im_start|>; then its role and a newline
                        while (start > 0 && ids[(size_t) start] != kImStart) --start;
                        if (d > start + 3 && i - 4 > reuse + 256 && i - 4 != fork) msg_end = i - 4;
                        break;
                    }
            }
            if (!geni) prev_prompt.assign(ids.begin(), ids.end());
            const int64_t cut1 = std::min(fork > 0 ? fork : INT64_MAX, msg_end > 0 ? msg_end : INT64_MAX);
            const int64_t cut2 = std::max(fork, msg_end) > cut1 ? std::max(fork, msg_end) : 0;
            const int64_t cuts[2] = {cut1 == INT64_MAX ? 0 : cut1, cut2};   // ascending, 0 = none
            int64_t batched_end = reuse;   // [reuse, batched_end) through the batched prompt path
            if (geni) {   // image rows reach the batched path only; a short text tail after the last <|im_start|> need
                          // not, and ends there as a text prompt does, with the turn's checkpoint
                batched_end = n - 1;
                if (q > reuse && n - 1 - q <= o.feed_max &&
                    std::find(ids.begin() + q, ids.begin() + (n - 1), kImagePad) == ids.begin() + (n - 1))
                    batched_end = q;
            } else if (n - 1 - reuse > o.feed_max) {
                batched_end = (q > reuse && n - 1 - q <= o.feed_max) ? q : n - 1;
            }
            // Every batched chunk streams each expert its tokens use, however few tokens it holds, so a short last
            // chunk goes through verify windows too.  Chunks and checkpoint segments start at `reuse`.
            if (!geni && batched_end > reuse) {
                const int64_t last = reuse + (batched_end - 1 - reuse) / o.prefill_chunk * o.prefill_chunk;
                if (last > reuse && batched_end - last <= o.feed_max) batched_end = last;
                for (int64_t c : cuts)   // a short piece after a cut goes through verify windows, as a short last chunk
                    if (c > reuse && c < batched_end && batched_end - c <= o.feed_max) { batched_end = c; break; }
            }
            if (batched_end > reuse) {
                tr("prompt tiers", batched_end - reuse);
                tier.apply_pending(true);   // no copy may still land in a lent slot
                for (TierSlot& ts : tiers) ts.adapt.apply_pending(true);
                tr("prompt tiers landed");
                // --prompt-split: a prompt of more than one chunk, when the second GPU can lend the split's bytes.
                // The checkpoints every --prompt-cache-every cells fall at the chunks' ends.
                const int64_t chunk = prompt_chunk(batched_end - reuse, o.prefill_chunk, pbuf.offload != nullptr);
                const int64_t step = std::max<int64_t>(1, o.prompt_cache_every / o.prefill_chunk) * chunk;
                const bool split_try = pbuf.split != nullptr && batched_end - reuse > o.prefill_chunk;
                if (!pbuf.take(chunk, err, split_try ? psplit.bytes_for(chunk, batched_end) : 0)) { report(err); return 1; }
                tr("prompt slots lent", chunk);
                const bool split_on = pbuf.split_region != nullptr;
                if (split_on) {
                    if (!psplit.begin(pbuf.split_region, pbuf.split_bytes, chunk, reuse, batched_end, track, err)) {
                        report(err);
                        return 1;
                    }
                    pcache.split_from = psplit.at();
                    pcache.split_save = [&psplit](int64_t pos, void* host) { psplit.checkpoint(pos, host); };
                }
                prefill.clear_routed();
                if (kvg.on && pbuf.lend && pbuf.loan.first >= 0 && pbuf.loan.first < kvg.top && kvg.lo < kvg.top) {
                    report("the prompt path's loan reached the slots the elastic K/V took (a bug: --no-kv-grow)");
                    return 1;
                }
                tr("prompt start", reuse, batched_end);
                pp_from = reuse;
                pp_total = n - 1 - reuse;
                pp_t0 = r0;
                for (int64_t a0 = reuse; a0 < batched_end;) {
                    int64_t b0 = track ? std::min(batched_end, a0 + step) : batched_end;
                    for (int64_t c : cuts)
                        if (c > a0 && c < b0) { b0 = c; break; }
                    pp_at = a0;
                    if (!prefill.run(ids.data() + a0, b0 - a0, a0, err)) {
                        if (!stop_req.load()) { report(err); return 1; }
                        // STOP: the session holds the chunks read before it
                        if (track && prefill.processed() > 0) pcache.set(a0, ids.data() + a0, prefill.processed());
                        break;
                    }
                    if (track) {
                        pcache.set(a0, ids.data() + a0, b0 - a0);
                        pcache.checkpoint(b0);
                    }
                    if (split_on) psplit.ran(b0);
                    a0 = b0;
                }
                if (split_on) {   // the second stage's last chunk, and its layers' state back in the session
                    pcache.split_from = 0;
                    pcache.split_save = nullptr;
                    if (!psplit.finish(err)) { report(err); return 1; }
                    tr("prompt split done");
                }
                // the tiers' moves (deterministic modes: refilled before the windows, as they were)
                const bool through_tiers = o.adapt_every > 0 && !o.deterministic_adapt;
                if (through_tiers ? !pbuf.give_back_tiers(&tier, &tier2, &prefill.routed(), *srcp, err)
                                  : !pbuf.give_back(*srcp, err)) {
                    report("refilling a lent slot failed: " + err);
                    return 1;
                }
                add_prompt_usage(drive.d.usage, prefill.routed(), g.n_expert);
                add_prompt_usage(prompt_heat, prefill.routed(), g.n_expert);   // (empty: no learned profile)
                tr("prompt done (slots refilled)");
            }
            // A window whose logits held a NaN or an infinity (upstream #879): nothing of it is used, the request
            // ends with the finish "nonfinite", and the next one starts from an empty sequence.
            int64_t poison_at = -1;
            // prompt tokens through verify windows as wide as the verifier's: each window is committed whole, and the
            // draft layer catches up on the same cells (its drafts are not used, so its chain stops at the first)
            auto feed = [&](int64_t from, int64_t to) -> bool {
                mtp.set_max_drafts(1);
                for (int64_t j = from; j < to;) {
                    const int T = (int) std::min<int64_t>(W, to - j);
                    for (int i = 0; i < T; ++i) window[(size_t) i] = (int32_t) ids[(size_t) (j + i)];
                    drive.d.layers = 0;
                    drive.d.experts = 0;
                    drive.d.failed = false;
                    drive.adapt_next = drive.flush_next = false;
                    drive_window(&drive, T);
                    if (!ver.run(T, window.data(), j, &drive_pool_multi, &drive, outv.data(), err) || drive.d.failed) {
                        if (drive.d.failed && drive.d.fail) err = drive.d.fail;
                        return false;
                    }
                    if (const int r = ver.nonfinite(T); r >= 0) {
                        poison_at = j + r;
                        return true;
                    }
                    if (!ver.commit(T, err, false)) return false;   // beside the draft
                    for (int i = 0; i < T; ++i) outv[(size_t) i] = (int32_t) ids[(size_t) (j + i + 1)];
                    if (!mtp.draft(T, outv.data(), j, T - 1, drafts.data(), err, dprob.data(), (float) o.spec_min_p) ||
                        !ver.wait_commit(err))
                        return false;
                    if (track) pcache.set(j, ids.data() + j, T);
                    j += T;
                }
                return true;
            };
            int64_t merge_from = n - 1;   // the first decode window's first position: [merge_from, n) are prompt tokens
            if (!stop_req.load()) {
                struct NoHeadSampling {   // every token is committed and the picks are discarded
                    lamina::core::Verifier& v;
                    explicit NoHeadSampling(lamina::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                    ~NoHeadSampling() { v.set_head_sampling(true); }
                } no_head_sampling(ver);
                if (pp_total == 0) {
                    pp_from = reuse;
                    pp_total = n - 1 - reuse;
                    pp_t0 = r0;
                }
                auto feed_to = [&](int64_t from, int64_t to) -> bool {   // with the `cuts` checkpoints on the way
                    for (int64_t c : cuts)
                        if (track && c > from && c < to) {
                            if (!feed(from, c)) return false;
                            if (poison_at >= 0) return true;
                            pcache.checkpoint(c);
                            from = c;
                        }
                    return feed(from, to);
                };
                const int64_t split = q > batched_end ? q : batched_end;
                // the last feed window takes the prompt's last token too when it has room for it: it runs as the
                // first decode window, its prompt tokens committed as they are and the pick at the last one emitted
                // (a window fewer: the first token 16-20 ms sooner on IQ3_S's follow-up turns); a scored request keeps
                // the last token alone (its row is the first window's first), as does one that emits nothing
                if (n - 1 > split && max_new > 0 && req_score.empty() && !ddump.on()) {
                    int64_t s = split;   // the last piece feed_to runs: from the last cut before the end
                    for (int64_t c : cuts)
                        if (track && c > s && c < n - 1) s = c;
                    const int64_t k = s + (n - 2 - s) / W * W;
                    if (n - 1 - k < W) merge_from = k;
                }
                if (!feed_to(batched_end, split)) { report(err); return 1; }
                if (split > batched_end) pp_line(split);
                if (track && split == q && poison_at < 0) pcache.checkpoint(q);
                if (poison_at < 0 && !feed_to(split, merge_from)) { report(err); return 1; }
                if (merge_from > split) pp_line(merge_from);
                if (track && poison_at < 0) pcache.checkpoint(merge_from);
            }
            pp_total = 0;
            const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            // the verify windows: the first holds the prompt's tokens from merge_from on (its last alone when the last
            // feed window was full, or none ran)
            int64_t p = merge_from;
            int32_t x = (int32_t) ids[(size_t) merge_from];
            bool first_window = true;
            int64_t produced_n = 0;
            const char* finish = poison_at >= 0 ? "nonfinite" : "length";
            const Clock::time_point d0 = Clock::now();
            int64_t draft_offered = 0, draft_accepted = 0;
            // the expert-cache hits of the decode: rows either GPU computes from its VRAM cache, of all routed rows
            const int64_t hits0 = drive.d.cache_hits + drive.d.tier_entries, look0 = hits0 + drive.d.cache_refused;
            std::vector<int32_t> consumed;   // what the state has read, for the penalties
            if (hist_n > 0)
                for (int64_t i = 0; i < merge_from; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
            int lk = 0;   // the next window's drafts from the lookup (0: the draft layer's)
            if (o.spec_lookup > 0) {
                ids32.resize((size_t) n);
                for (int64_t i = 0; i < n; ++i) ids32[(size_t) i] = (int32_t) ids[(size_t) i];
                lookup.assign(ids32.data(), ids32.size());
            }
            mtp.on_draft = [&](int j, int32_t tok, float prob) {   // the drafts the next window will verify
                if (lk == 0 && prob >= (float) req_spec_min_p) ver.ple_ahead(j + 1, tok);
            };
            ver.set_logits_wanted(!req_score.empty() || ddump.on());   // score=: a split head's first row whole
            std::vector<float> dd_logits(ddump.on() ? (size_t) W * (size_t) n_vocab : 0);
            while (poison_at < 0 && produced_n < max_new) {
                if (stop_req.load()) { finish = "cancel"; break; }   // the client went away
                int T = S;
                if (req_spec_min_p > 0.0) {
                    T = 1;
                    while (T < S && dprob[(size_t) T - 1] >= (float) req_spec_min_p) ++T;
                }
                if (lk > 0) T = 1 + lk;
                if (first_window) T = (int) (n - merge_from);   // the prompt's last tokens (1: its last alone)
                const std::vector<int32_t>& dr = lk > 0 ? ldrafts : drafts;
                // Never verify past what this request emits: the committed cells then hold exactly the tokens the
                // client saw, so its next turn continues from them.  So the window stops at the output limit and
                // at a drafted end-of-turn token.
                if (!first_window) {
                    T = (int) std::min<int64_t>(T, max_new - produced_n);
                    for (int i = 1; i < T; ++i)
                        if (std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) dr[(size_t) i - 1]) !=
                            o.eos_ids.end()) {
                            T = i + 1;
                            break;
                        }
                }
                if (p + T > o.max_context) break;
                // the elastic K/V: the window's cells and the drafter's beyond them
                if (kvg.on && p + T + 64 > kvg.cells && !kvg_ensure(p + T + 64)) {
                    drive.join_adapt();
                    report("the K/V cannot grow: no VRAM is left");
                    return 1;
                }
                window[0] = x;
                for (int i = 1; i < T; ++i)
                    window[(size_t) i] = first_window ? (int32_t) ids[(size_t) (merge_from + i)] : dr[(size_t) i - 1];
                if (hist_n > 0) {
                    // row t counts what plain decode would have: the state's tokens, then window[0..t]
                    lamina::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T, hist_n,
                                                  hist_stage.data());
                    cudaMemcpy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                               cudaMemcpyHostToDevice);
                }
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                drive_window(&drive, T);
                tr("window", p, T);
                drive.adapt_next = !drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0;
                drive.flush_next = !drive.d.usage.empty() && ((rounds + 2) % o.adapt_every) == 0;
                if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err) || drive.d.failed) {
                    drive.join_adapt();
                    report(drive.d.failed && drive.d.fail ? drive.d.fail : err);
                    return 1;
                }
                if (const int r = ver.nonfinite(T); r >= 0) {
                    drive.join_adapt();
                    poison_at = p + r;
                    finish = "nonfinite";
                    break;
                }
                // score=: the first window's row is the distribution right after the prompt; the candidates'
                // log-probabilities go out as one `LP id:logprob ...` line before the first `T` (an `LP` alone when
                // the row cannot be read: the request fails, the engine goes on)
                if (first_window && !req_score.empty()) {
                    std::vector<float> lp(req_score.size());
                    serve_print("LP");
                    if (ver.candidate_logprobs(req_score.data(), (int) req_score.size(), lp.data(), err))
                        for (size_t i = 0; i < req_score.size(); ++i)
                            serve_print(" %d:%.6g", (int) req_score[i], lp[i]);
                    else
                        std::fprintf(stderr, "lamina serve: score=: %s\n", err.c_str());
                    serve_print("\n");
                    ver.set_logits_wanted(false);
                }
                const bool fw = first_window;
                int a = fw ? T - 1 : 0;   // the first window's prompt tokens are all committed
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (!ver.commit(a + 1, err, false)) {   // beside the draft
                    drive.join_adapt();
                    report(err);
                    return 1;
                }
                if (track) pcache.set(p, window.data(), a + 1);   // the committed cells p .. p + a
                if (hist_n > 0) consumed.insert(consumed.end(), window.begin(), window.begin() + a + 1);
                if (!fw) {
                    draft_offered += T - 1;
                    draft_accepted += a;
                }
                ver.ple_ahead(0, outv[(size_t) a]);
                first_window = false;
                bool eos = false;
                for (int i = fw ? a : 0; i <= a && produced_n < max_new && !eos; ++i) {
                    serve_print("T %d\n", (int) outv[(size_t) i]);
                    if (o.spec_lookup > 0) lookup.append(outv[(size_t) i]);
                    ++produced_n;
                    eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                ++rounds;
                // the next window's drafts: the lookup's when they are more than the draft layer's chain allows (the
                // chain stops at its first draft then)
                lk = 0;
                if (o.spec_lookup > 0 && !eos && produced_n < max_new) {
                    lk = lookup.propose(W - 1, ldrafts.data());
                    if (lk < S) lk = 0;
                    for (int j = 0; j < lk; ++j) ver.ple_ahead(j + 1, ldrafts[(size_t) j]);
                }
                mtp.set_max_drafts(lk > 0 ? 1 : S - 1);
                if (fw)   // the draft layer catches up on the prompt tokens: each position's next token is the prompt's
                    for (int i = 0; i < T - 1; ++i) outv[(size_t) i] = (int32_t) ids[(size_t) (merge_from + i + 1)];
                const bool drafted = (eos || produced_n >= max_new ||
                                      mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) req_spec_min_p)) &&
                                     ver.wait_commit(err);
                if (!drive.join_adapt()) {
                    report("an adaptive refill failed");
                    return 1;
                }
                if (!drafted) {
                    report(err);
                    return 1;
                }
                if (ddump.on()) {   // the window's committed positions: rows, logits, and the drafts made after them
                    if (!ver.copy_logits(a + 1, dd_logits.data(), err)) { report(err); return 1; }
                    ddump.put_rows(ver.final_R_all(), a + 1, g.hc * g.n_embd);
                    for (int i = 0; i <= a; ++i)
                        ddump.put_meta((int32_t) (p + i), window[(size_t) i], outv[(size_t) i],
                                       dd_logits.data() + (size_t) i * (size_t) n_vocab, n_vocab);
                    const bool none = eos || produced_n >= max_new;
                    const int32_t d3[3] = {none ? -1 : drafts[0], none || W < 3 ? -1 : drafts[1], none || W < 4 ? -1 : drafts[2]};
                    const float p3[3] = {none ? 0.0f : dprob[0], none || W < 3 ? 0.0f : dprob[1], none || W < 4 ? 0.0f : dprob[2]};
                    ddump.put_round((int32_t) p, T, a, d3, p3);
                }
                if (eos) { finish = "stop"; break; }
                x = outv[(size_t) a];
                p += a + 1;
            }
            mtp.on_draft = nullptr;
            ver.set_logits_wanted(false);
            if (ddump.on()) {
                prefill.on_final = nullptr;
                ddump.close();
                if (ddump.overflow > 0)
                    std::fprintf(stderr, "lamina serve: --dump-draft-data: %lld values past FP16's range so far\n",
                                 (long long) ddump.overflow);
            }
            if (poison_at >= 0) {
                std::fprintf(stderr, "lamina serve: non-finite logits at position %lld: the request ends, and the next "
                                     "one reads its prompt from the start\n", (long long) poison_at);
                poisoned = true;
            }
            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
            const int64_t hits1 = drive.d.cache_hits + drive.d.tier_entries, look1 = hits1 + drive.d.cache_refused;
            const TierCounters t1 = tier_counters(), &t0 = req_tiers;
            std::string per_tier;
            for (int t = 0; t < std::max(n_tiers, 1); ++t) {   // the first tier's fields even without one
                const TierCounters::Gpu &a = t0.gpu[t], &b = t1.gpu[t];
                const std::string nm = kTierName[t];
                auto field = [&](const char* key, int64_t v) {
                    per_tier += " " + nm + "_" + key + "=" + std::to_string(v);
                };
                field("entries", b.entries - a.entries);
                field("experts", b.experts - a.experts);
                field("layers", b.layers - a.layers);
                if (t == 0) per_tier += " second_cpu_layers=" + std::to_string(t1.cpu_layers - t0.cpu_layers);
                field("swaps", b.swaps - a.swaps);
                field("fills", b.fills - a.fills);
                field("dropped", b.dropped - a.dropped);
                field("empty_slots", tiers[(size_t) t].adapt.free_slots());
                field("prefetched", b.prefetched - a.prefetched);
                field("prefetch_used", b.prefetch_used - a.prefetch_used);
            }
            serve_print("TIERS main_swaps=%lld main_fills=%lld main_dropped=%lld%s deterministic_adapt=%d\n",
                        (long long) (t1.main_swaps - t0.main_swaps), (long long) (t1.main_fills - t0.main_fills),
                        (long long) (t1.main_dropped - t0.main_dropped), per_tier.c_str(), o.deterministic_adapt ? 1 : 0);
            print_cache();
            // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
            // <expert-cache hits> <lookups>
            serve_print("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld\n", (long long) produced_n, (long long) n,
                        prompt_ms, decode_ms, finish, (long long) draft_accepted, (long long) draft_offered,
                        (long long) reuse, (long long) (hits1 - hits0), (long long) (look1 - look0));
            const int64_t fresh = n - reuse;
            std::fprintf(stderr, "lamina serve: %lld prompt tokens (%lld cached) in %.0f ms (%.1f tok/s), %lld generated "
                                 "in %.0f ms (%.1f tok/s)\n", (long long) n, (long long) reuse, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * fresh / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                         decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0);
            if (pcache.enabled()) std::fprintf(stderr, "lamina serve: prompt cache: %s\n", pcache.summary().c_str());
            check_stack_limit(o.main_gpu, o.main_gpu);
            for (const int d : o.tier_gpus) check_stack_limit(d, o.main_gpu);
            if (replace) replan();
            profile_dirty = true;
        }
        if (profile_dirty) save_profile("exit");   // QUIT, Ctrl+C, or the server closed stdin
        return 0;
    }

    // ---- plan v0.3 P5: the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop
    // then starts at the last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    int64_t spec_pos = 0;   // plan v0.3 P6: where the speculative loop starts (0 = not used)
    lamina::prefill::Prefill prefill;
    lamina::prefill::ExpertRunner offload;
    lamina::prefill::PromptSplit psplit;   // (its second stage refers to the first: destroyed before it)
    PromptBuffers pbuf;
    double prefill_batched_ms = 0;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        const int64_t n_batched = n_prompt - 1;
        const bool path_ok = setup_prompt_path(prefill, offload, pbuf, psplit, err);
        const int64_t chunk = path_ok ? prompt_chunk(n_batched, o.prefill_chunk, pbuf.offload != nullptr) : 0;
        if (!path_ok ||
            !pbuf.take(chunk, err, pbuf.split != nullptr && n_batched > o.prefill_chunk ? psplit.bytes_for(chunk, n_batched) : 0)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        if (pbuf.lend || pbuf.lend2)
            std::fprintf(stderr, "lamina: prompt path borrows %lld cache slots (%.2f GiB), %lld (%.2f GiB) on the "
                                 "second GPU\n", (long long) pbuf.loan.slots(), (double) pbuf.loan.bytes() / 1073741824.0,
                         (long long) pbuf.loan2.slots(), (double) pbuf.loan2.bytes() / 1073741824.0);
        else
            std::fprintf(stderr, "lamina: prompt path allocates its own buffers (no cache slots to borrow)\n");
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            prefill.draft = &mtp;   // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
        }
        prefill.profile = o.prefill_profile;
        const bool split_on = pbuf.split_region != nullptr;
        psplit.set_profile(o.prefill_profile);
        const Clock::time_point tp0 = Clock::now();
        // LAMINA_PREFILL_CUT=n (tests): the prompt in two runs split at position n, as a follow-up turn continues a
        // cached prefix (the split's second stage copies that prefix's state in)
        const char* cut_env = std::getenv("LAMINA_PREFILL_CUT");
        const int64_t cut = cut_env != nullptr ? std::min<int64_t>(std::max<int64_t>(std::atoll(cut_env), 0), n_batched) : 0;
        for (int64_t a = 0; a < n_batched;) {
            const int64_t b = a < cut ? cut : n_batched;
            if ((split_on && !psplit.begin(pbuf.split_region, pbuf.split_bytes, chunk, a, n_batched, false, err)) ||
                !prefill.run(o.tokens.data() + a, b - a, a, err) || (split_on && !psplit.finish(err))) {
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            a = b;
        }
        // refill the lent slots from the arena and give them back to the decode tiers
        const size_t n_lent[2] = {pbuf.loan.lent.size(), pbuf.loan2.lent.size()};
        const Clock::time_point tr = Clock::now();
        const bool through_tiers = o.adapt_every > 0 && !o.deterministic_adapt;
        if (through_tiers ? !pbuf.give_back_tiers(&tier, &tier2, &prefill.routed(), *srcp, err)
                          : !pbuf.give_back(*srcp, err)) {
            std::fprintf(stderr, "lamina: refilling a lent slot failed: %s\n", err.c_str());
            return 1;
        }
        if (n_lent[0] + n_lent[1] > 0)
            std::fprintf(stderr, "lamina: lent slots %s (%zu, %zu on the second GPU) in %.1f ms\n",
                         through_tiers ? "given back to the tiers" : "refilled", n_lent[0], n_lent[1],
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const lamina::prefill::PrefillStats& ps = prefill.stats();
        std::fprintf(stderr, "lamina: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld ahead), host %.1f ms, resident %lld; PLE %.1f ms\n",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_prefetched, ps.ms_experts_host, (long long) ps.experts_resident, ps.ms_ple);
        if (pbuf.offload != nullptr)
            std::fprintf(stderr, "lamina: prefill on the second GPU: experts streamed %lld (%lld ahead), "
                                 "resident %lld; %.1f ms queuing, %.1f ms GPU\n", (long long) offload.experts_streamed,
                         (long long) offload.experts_prefetched, (long long) offload.experts_resident, offload.ms_host,
                         offload.ms_gpu);
        if (split_on) {
            const lamina::prefill::PrefillStats& s2 = psplit.stats();
            std::fprintf(stderr, "lamina: prefill split at layer %lld: the second stage %.1f ms (its experts "
                                 "streamed %lld (%lld ahead), resident %lld, host %.1f ms), the first waited %.1f ms to "
                                 "hand a chunk on, the end %.1f ms, the state's copy in %.1f ms\n",
                         (long long) psplit.at(), s2.ms_total, (long long) s2.experts_streamed,
                         (long long) s2.experts_prefetched, (long long) s2.experts_resident, s2.ms_experts_host,
                         psplit.ms_first_waited, psplit.ms_finish, psplit.ms_copy_in);
        }
        if (o.prefill_profile) {
            for (int st = 0; st < (split_on ? 2 : 1); ++st) {
                const lamina::prefill::PrefillStats& px = st == 0 ? ps : psplit.stats();
                std::fprintf(stderr, "lamina: prefill GPU ms by section%s:", st == 0 ? "" : " (the second stage)");
                for (int i = 0; i < lamina::prefill::kPsCount; ++i)
                    std::fprintf(stderr, "%s %s %.0f", i ? ";" : "", lamina::prefill::prefill_section_name(i),
                                 px.ms_section[i]);
                std::fprintf(stderr, "\n");
            }
        }
    }

    spec_pos = pos_start;   // the last prompt token is the first verify window

    // ================================ plan v0.3 P6: SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t
    // is exactly what greedy decode would emit there, so the first draft that differs ends the round and the
    // round emits (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (spec_pos > 0 && (int64_t) produced.size() < o.max_new && !ended) {
        auto read_ids = [](const std::string& path, std::vector<int64_t>& ids) {
            std::ifstream in(path);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string e;
            return (bool) in && parse_i64_list(text.c_str(), ids, e);
        };
        std::vector<int64_t> follow;
        if (!o.spec_follow.empty() && !read_ids(o.spec_follow, follow)) {
            std::fprintf(stderr, "lamina: cannot read --spec-follow %s\n", o.spec_follow.c_str());
            return 2;
        }
        // --spec-follow: the run ends with the continuation
        const int64_t max_new = follow.empty() ? o.max_new : std::min<int64_t>(o.max_new, (int64_t) follow.size());
        int64_t follow_differ = 0, follow_emitted = 0;
        if (thits.d_res == nullptr) {
            std::fprintf(stderr, "lamina: --spec needs the device residency table (--expert-profile, "
                                 "--expert-cache and the token graph)\n");
            return 2;
        }
        mem_mark("the head and the prompt path");
        lamina::core::Verifier ver;
        lamina::core::VerifyHits vh;
        vh.res = host_res.data();
        vh.cache_base = thits.cache_base;
        vh.slot_off = xcache.slot_offsets();
        vh.slots = xcache.slots();
        vh.blob = thits.blob;
        ver.set_split_head(split_head.get());
        ver.set_logits_wanted(!o.window_logits.empty());
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.max_window(), err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the verifier and the drafter's binding");
        ver.set_host_rows(!o.window_hashes.empty());
        hook_verifier(ver);
        drive.d.plan = ver.plan_sink();
        drive.d.host_res = ver.residency();   // the snapshot the windows decide their GPU hits by
        drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
        if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
        if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
        ver.set_pcie_share(drive.d.pcie_num > 0);
        const int64_t pcie0 = drive.d.pcie_experts;
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        if (o.prefill_chunk > 0 && n_prompt > 1) add_prompt_usage(drive.d.usage, prefill.routed(), g.n_expert);
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        const int W = o.max_window();
        std::vector<int32_t> drafts((size_t) W, 0);
        std::vector<float> dprob((size_t) W, 1.0f);
        std::vector<int64_t> window_hist((size_t) W + 1, 0);
        // the first window is the last prompt token alone (it produces the first generated token and the MTP's
        // first cell)
        bool first_window = true;
        if (use_mtp && !first_window &&
            !mtp.draft_first(ss.R, x, p - 1, drafts.data(), err, dprob.data(), (float) o.spec_min_p)) {
            std::fprintf(stderr, "lamina: %s\n", err.c_str());
            return 1;
        }
        std::vector<int32_t> window((size_t) W), outv((size_t) W);
        std::vector<uint8_t> fdiff((size_t) W, 0);
        std::FILE* hashes = o.window_hashes.empty() ? nullptr : std::fopen(o.window_hashes.c_str(), "w");
        if (!o.window_hashes.empty() && hashes == nullptr) {
            std::fprintf(stderr, "lamina: cannot write %s\n", o.window_hashes.c_str());
            return 1;
        }
        std::FILE* wlog = o.window_logits.empty() ? nullptr : std::fopen(o.window_logits.c_str(), "w");
        if (!o.window_logits.empty() && wlog == nullptr) {
            std::fprintf(stderr, "lamina: cannot write %s\n", o.window_logits.c_str());
            return 1;
        }
        std::vector<float> wl(wlog != nullptr ? (size_t) W * (size_t) n_vocab : 0);
        std::vector<int32_t> wl_ids(wlog != nullptr ? (size_t) n_vocab : 0);
        auto write_logits = [&](int64_t k, int32_t emitted, const float* l) {   // generated token k's line
            constexpr int K = 64;
            double m = l[0], s = 0;
            for (int64_t i = 1; i < n_vocab; ++i) m = std::max(m, (double) l[i]);
            for (int64_t i = 0; i < n_vocab; ++i) s += std::exp((double) l[i] - m);
            const double lse = m + std::log(s);
            for (int64_t i = 0; i < n_vocab; ++i) wl_ids[(size_t) i] = (int32_t) i;
            std::partial_sort(wl_ids.begin(), wl_ids.begin() + K, wl_ids.end(),
                              [&](int32_t a, int32_t b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
            std::fprintf(wlog, "%lld %d", (long long) k, (int) emitted);
            for (int j = 0; j < K; ++j)
                std::fprintf(wlog, " %d:%.6f", (int) wl_ids[(size_t) j], (double) l[wl_ids[(size_t) j]] - lse);
            std::fprintf(wlog, "\n");
        };
        std::vector<int64_t> accepted_hist((size_t) W, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0;
        // prompt lookup: the text so far, the prompt and then each emitted token
        lamina::spec::SuffixDrafter lookup(o.spec_lookup, 32,
                                           o.spec_lookup > 0 ? (size_t) (n_prompt + max_new) + 16 : 16);
        if (o.spec_lookup > 0) {
            for (int64_t t : o.tokens) lookup.append((int32_t) t);
            for (int64_t t : produced) lookup.append((int32_t) t);
        }
        std::vector<int32_t> ldrafts((size_t) W, 0);
        int lk = 0;   // the next window's drafts from the lookup (0: the draft layer's)
        int64_t lookup_rounds = 0, lookup_drafts = 0, lookup_ok = 0;
        const double pool_ms0 = drive.cpu_ms;
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        const uint32_t sleeps0 = pool.sleeps();
        if (o.stats) drive.d.routed.assign((size_t) (g.n_layers * g.n_expert), 0);
        if (use_mtp)
            mtp.on_draft = [&](int j, int32_t tok, float prob) {   // the drafts the next window will verify
                if (lk == 0 && prob >= (float) o.spec_min_p) ver.ple_ahead(j + 1, tok);
            };
        while ((int64_t) produced.size() < max_new) {
            const Clock::time_point t0 = Clock::now();
            int T = o.spec;
            if (use_mtp && o.spec_min_p > 0.0) {
                T = 1;
                while (T < o.spec && dprob[(size_t) T - 1] >= (float) o.spec_min_p) ++T;
            }
            if (lk > 0) T = 1 + lk;
            if (first_window) T = 1;
            T = (int) std::min<int64_t>(T, o.max_context - p);   // near the context's end, the rows that fit
            ++window_hist[(size_t) T];
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                window[(size_t) i] = lk > 0 ? ldrafts[(size_t) i - 1] : use_mtp ? drafts[(size_t) i - 1] : 0;
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            drive_window(&drive, T);
            drive.adapt_next = !drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0;
            drive.flush_next = !drive.d.usage.empty() && ((rounds + 2) % o.adapt_every) == 0;
            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                drive.join_adapt();
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            if (drive.d.failed) {
                drive.join_adapt();
                std::fprintf(stderr, "lamina: the expert pool failed at layer %lld expert %lld: %s\n",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            if (wlog != nullptr && !ver.copy_logits(T, wl.data(), err)) {
                drive.join_adapt();
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            if (hashes != nullptr) {   // FNV-1a over the window's final residual rows
                const size_t n_floats = (size_t) T * (size_t) (g.hc * g.n_embd);
                uint64_t h = 1469598103934665603ull;
                const auto* bytes = (const uint8_t*) ver.final_R_host();
                for (size_t i = 0; i < n_floats * sizeof(float); ++i) h = (h ^ bytes[i]) * 1099511628211ull;
                std::fprintf(hashes, "%lld %lld %d %016llx", (long long) rounds, (long long) p, T, (unsigned long long) h);
                for (int i = 0; i < T; ++i) std::fprintf(hashes, " %d", (int) outv[(size_t) i]);
                std::fprintf(hashes, "\n");
            }
            if (!follow.empty()) {   // the continuation stands in for the argmax
                const size_t k0 = produced.size();
                for (int i = 0; i < T && k0 + (size_t) i < follow.size(); ++i) {
                    fdiff[(size_t) i] = outv[(size_t) i] != (int32_t) follow[k0 + (size_t) i];
                    outv[(size_t) i] = (int32_t) follow[k0 + (size_t) i];
                }
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (lk > 0) {
                ++lookup_rounds;
                lookup_drafts += lk;
                lookup_ok += a;
            }
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            }
            if (!ver.commit(a + 1, err, false)) {   // beside the draft
                drive.join_adapt();
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            ver.ple_ahead(0, outv[(size_t) a]);
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            bool eos = false;
            for (int i = 0; i <= a && (int64_t) produced.size() < max_new && !eos; ++i) {
                if (wlog != nullptr)
                    write_logits((int64_t) produced.size(), outv[(size_t) i], wl.data() + (size_t) i * (size_t) n_vocab);
                produced.push_back(outv[(size_t) i]);
                if (o.spec_lookup > 0) lookup.append(outv[(size_t) i]);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                if (!follow.empty()) { follow_differ += fdiff[(size_t) i]; ++follow_emitted; }
            }
            if (eos) {
                drive.join_adapt();
                if (!ver.wait_commit(err)) {
                    std::fprintf(stderr, "lamina: %s\n", err.c_str());
                    return 1;
                }
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            // the next window's drafts: the lookup's when they are more than the draft layer's chain allows (the
            // chain stops at its first draft then)
            lk = 0;
            if (o.spec_lookup > 0 && (int64_t) produced.size() < max_new) {
                lk = lookup.propose(W - 1, ldrafts.data());
                if (lk < o.spec) lk = 0;
                for (int j = 0; j < lk; ++j) ver.ple_ahead(j + 1, ldrafts[(size_t) j]);
            }
            mtp.set_max_drafts(lk > 0 ? 1 : o.spec - 1);
            const bool drafted = (!use_mtp || (int64_t) produced.size() >= max_new ||
                                  mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) o.spec_min_p)) &&
                                 ver.wait_commit(err);
            if (!drive.join_adapt()) {
                std::fprintf(stderr, "lamina: %s\n", adapt_err.c_str());
                return 1;
            }
            if (!drafted) {
                std::fprintf(stderr, "lamina: %s\n", err.c_str());
                return 1;
            }
            x = outv[(size_t) a];
            p += a + 1;
            total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (rounds % 64 == 0)
                std::fprintf(stderr, "lamina: position %lld, %lld tokens, %lld rounds\n", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        if (hashes != nullptr) std::fclose(hashes);
        if (wlog != nullptr) std::fclose(wlog);
        serve_print("%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                    "speculation", (long long) rounds, o.spec, (long long) drafts_ok, (long long) drafts_total,
                    drafts_total > 0 ? (double) drafts_ok / (double) drafts_total : 0.0,
                    rounds > 0 ? (double) (drafts_ok + rounds) / (double) rounds : 0.0);
        if (!follow.empty())
            serve_print("%-24s %lld of %lld emitted tokens differ from the argmax\n", "follow",
                        (long long) follow_differ, (long long) follow_emitted);
        if (o.spec_min_p > 0.0) {
            serve_print("%-24s", "window sizes");
            for (size_t i = 1; i < window_hist.size(); ++i) serve_print(" T%zu:%lld", i, (long long) window_hist[i]);
            serve_print("  (min draft probability %.2f)\n", o.spec_min_p);
        }
        serve_print("%-24s", "accepted per round");
        for (size_t i = 0; i < accepted_hist.size(); ++i) serve_print(" %zu:%lld", i, (long long) accepted_hist[i]);
        serve_print("\n");
        if (o.spec_lookup > 0)
            serve_print("%-24s %lld windows of its drafts, %lld of %lld accepted\n", "prompt lookup",
                        (long long) lookup_rounds, (long long) lookup_ok, (long long) lookup_drafts);
        if (rounds > 0)
            serve_print("%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "
                        "distinct / %.2f routed per layer\n",
                        "verify window", ver.ms_wait / rounds, ver.ms_pool / rounds, ver.ms_host / rounds,
                        ver.ms_commit / rounds,
                        (double) (drive.d.multi_misses - misses0) / (double) (rounds * g.n_layers),
                        (double) (drive.d.multi_entries - entries0) / (double) (rounds * g.n_layers));
        if (rounds > 0)
            serve_print("%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "
                        "CPU pool call %.3f ms/round; %u worker sleeps; %lld thread moves\n", "pool multi",
                        pool.ms_multi_gu / rounds, pool.ms_multi_q / rounds, pool.ms_multi_down / rounds,
                        (double) pool.multi_bytes / 1e6 / std::max(1e-9, pool.ms_multi_gu + pool.ms_multi_down),
                        (drive.cpu_ms - pool_ms0) / rounds, pool.sleeps() - sleeps0, (long long) placement.moves);
        if (rounds > 0)
            serve_print("%-24s plan %.3f  activation quantize %.3f  jobs %.3f  run %.3f ms/round\n", "dispatch",
                        drive.d.ms_plan / rounds, drive.d.ms_actq / rounds, drive.d.ms_jobs / rounds,
                        drive.d.ms_run / rounds);
        ver.print_profile();
        if (rounds > 0 && !drive.d.usage.empty())
            serve_print("%-24s %lld experts swapped in and %lld into empty slots (every %d rounds, %.3f ms/round), "
                        "%lld moves dropped; %.1f MB/round copied\n", "adaptive tier", (long long) tier.swaps,
                        (long long) tier.fills, o.adapt_every, tier.ms / rounds, (long long) tier.dropped,
                        (double) tier.sent_bytes / 1e6 / (double) rounds);
        for (int t = 0; rounds > 0 && t < n_tiers; ++t) {   // the layers left to the CPU and the predictions: all tiers'
            const lamina::core::SecondGpu& gp = tiers[(size_t) t].gpu;
            const lamina::core::AdaptiveTier& ad = tiers[(size_t) t].adapt;
            serve_print("%-24s %.2f routed entries and %.2f experts per round in %.2f layers (%.2f left to the CPU), "
                        "its rows after the CPU's in %.2f layers, %.3f ms/round; %lld experts swapped in and %lld into "
                        "empty slots (%lld still empty, %lld moves dropped; its copies between windows, %.2f ms, at "
                        "%.1f GB/s); %.2f experts per round prefetched, %.2f of them used (%.3f ms/round of host "
                        "time, %lld predictions waited for)\n", (std::string(kTierName[t]) + " GPU").c_str(),
                        (double) gp.entries_done / (double) rounds, (double) gp.experts / (double) rounds,
                        (double) gp.layers / (double) rounds, (double) drive.d.tier_skipped / (double) rounds,
                        (double) ver.late2[t] / (double) rounds, ver.ms_late2[t] / (double) rounds,
                        (long long) ad.swaps, (long long) ad.fills,
                        (long long) ad.free_slots(), (long long) ad.dropped, drive.gap_ms,
                        ad.copy_rate() / 1e6, (double) gp.prefetch_copied / (double) rounds,
                        (double) gp.prefetch_used / (double) rounds, ver.ms_predict / (double) rounds,
                        (long long) ver.predict_late);
        }
        if (rounds > 0 && !drive.d.routed.empty()) {
            // the share of the routed entries the N most-routed experts of this run take
            std::vector<uint32_t> c = drive.d.routed;
            std::sort(c.begin(), c.end(), std::greater<uint32_t>());
            double total = 0;
            for (uint32_t v : c) total += v;
            serve_print("%-24s", "routing concentration");
            double acc = 0;
            size_t at = 0;
            for (size_t n : {1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000, 20000}) {
                for (; at < n && at < c.size(); ++at) acc += c[at];
                serve_print(" %zu:%.3f", n, total > 0 ? acc / total : 0.0);
            }
            serve_print("  (top-N experts' share of %.0f routed entries)\n", total);
        }
        if (rounds > 0 && drive.d.pcie_num > 0)
            serve_print("%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\n",
                        "pcie experts", (double) (drive.d.pcie_experts - pcie0) / (double) (rounds * g.n_layers),
                        drive.d.pcie_num);
        (void) pool_ms0;
        if (use_mtp && rounds > 0)
            serve_print("%-24s %.3f ms/round drafting (%lld rounds), %.0f MiB of VRAM\n", "mtp",
                        mtp.ms_draft / (double) mtp.rounds, (long long) mtp.rounds, (double) mtp.vram_bytes() / 1048576.0);
        drive.d.host_res = host_res.data();   // the verifier's snapshot goes with it
    }

    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        serve_print("%-24s %s (%lld records of layer, k, ids, weights)\n", "routing dumped",
                    o.dump_routing.c_str(), (long long) drive.records);
    }

    const int64_t decoded = (int64_t) produced.size();
    serve_print("prompt  :");
    for (int64_t t : o.tokens) serve_print(" %lld", (long long) t);
    serve_print("\noutput  :");
    for (int64_t t : produced) serve_print(" %lld", (long long) t);
    serve_print("\n");
    const double decode_ms = decoded > 0 ? total_ms / (double) decoded : 0.0;
    serve_print("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", (long long) decoded, total_ms,
                decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (n_prompt > 1)
        serve_print("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill",
                    (long long) (n_prompt - 1), prefill_ms,
                    prefill_ms > 0 ? 1000.0 * (double) (n_prompt - 1) / prefill_ms : 0.0, ttft_ms);

    if (o.stats) {
        serve_print("%-24s %.3f ms/token (wall clock)\n", "  per token", decode_ms);
        if (const std::string io = ple_table.io_report(); !io.empty()) serve_print("  %s\n", io.c_str());
        // **THE DENOMINATOR IS THE POSITIONS THE POOL ACTUALLY RAN ON, NOT THE DECODED TOKENS (A6).**
        // `drive_pool` is called once per layer per position and PREFILL runs the loop too, so accumulating
        // `cpu_ms` over prefill and then dividing by `decoded` inflates this figure.  `drive.calls / n_layers`
        // is the number of positions - the same correction the ring counters below already received, which is
        // why they print "of 192" rather than "240 of 192".
        const double pool_positions = g.n_layers > 0 ? (double) drive.calls / (double) g.n_layers : 0.0;
        serve_print("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n",
                    "  the CPU expert pool", pool_positions > 0.0 ? drive.cpu_ms / pool_positions : 0.0,
                    (long long) g.n_layers, pool_positions, (long long) drive.calls);
        // **AND WHERE INSIDE `run()` IT WENT.**  They were one number, which cannot tell a pool that is slow at
        // the WORK from one that is slow at the SYNCHRONISATION - opposite fixes: the drain, or the barrier that
        // closes the phase.
        if (pool_positions > 0.0) {
            double dr = 0, cl = 0;
            pool.phase_ms(dr, cl);
            const double per = pool_positions;
            serve_print("%-24s   drain %.3f  close %.3f  ms/token\n", "  pool phases", dr / per, cl / per);
        }
        serve_print("%-24s %lld blobs read\n", "  expert blobs", (long long) srcp->reads());
        // ---- **R4's DISPATCH MEASUREMENT: h, ON THE ENGINE'S OWN ROUTING.**  No offline trace, no corpus
        // question, no k-fold - these are the ids the router actually produced on this run.  Reported as
        // hits/lookups so it can be read directly as the h the cache would deliver, and alongside `refused`
        // so a full cache is visible rather than silently capping the rate.
        if (o.expert_cache > 0) {
            const int64_t look = drive.d.cache_hits + drive.d.cache_refused;
            serve_print("%-24s %lld of %lld = %.4f\n", "  expert-cache hits", (long long) drive.d.cache_hits,
                        (long long) look, look > 0 ? (double) drive.d.cache_hits / (double) look : 0.0);
        }
    }

    check_stack_limit(o.main_gpu, o.main_gpu);
    for (const int d : o.tier_gpus) check_stack_limit(d, o.main_gpu);
    return 0;
}
