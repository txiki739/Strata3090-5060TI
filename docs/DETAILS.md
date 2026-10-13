# Strata - the details

The technical side of Strata: every measured number, the API, images, all settings and how the engine works.
New here? Start with the [README](../README.md); installing step by step is in [INSTALL.md](INSTALL.md), the models in
[MODELS.md](MODELS.md), common problems in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

> **On this page:** [Speed](#speed-measured) · [Other GPUs](#other-gpus-estimated) · [Which model?](#which-model) ·
> [Requirements](#before-you-start) · [Windows](#windows) · [Linux](#linux) · [API](#using-it) ·
> [MCP tools](#tools-from-mcp-servers) · [MCP server](#manage-strata-from-your-ai-assistant-mcp-server) ·
> [Images](#images-vision) ·
> [Troubleshooting](#troubleshooting) · [How it works](#how-it-works)

---

## Speed (measured)

RTX 5070 **12 GB**, Ryzen 5 7600 (6 cores), 64 GB DDR5-5200, Windows, engine 0.1.26 with the settings setup writes
(`--prefill auto`, 8-bit KV above 4K, KV streaming from 64K). One code-agent prompt per length, 256 generated tokens,
MTP speculative decoding on. "262K" is the model's full context window (a 259,943-token prompt). The IQ2_XS row was
measured with Swift 1.5's IQ2_XS, which runs at the original's speed.

**Engine 0.1.36 (#136), the same PC:** Q2_0's prompt experts run on fused int8 tensor-core kernels (RTX 30 and newer):
4K 1,294 -> 1,570, 32K 2,170 -> 2,653, 128K 2,123 -> 2,468 tokens/s (+16-22%), as close to an FP16 reference as the
previous kernels (closer at 32K: teacher-forced KL 0.009 vs 0.012). The decode path's block selection and greedy
argmax run on thread-block clusters (RTX 50, sm_90+; other cards keep the previous kernels; the same tokens): Q2_0 output at 4K 89 -> 93.5, at 128K
64.5 -> 76.4 tokens/s. `STRATA_PF_FUSED=0` keeps the previous prompt kernels (byte-identical answers to 0.1.35);
`STRATA_PF_FUSED=1` also runs the native IQ packs' fused kernels (opt-in: IQ2_XS prompts +12% at 4K, +3% at 32K, the
IQ3 packs about even; on HIP they run on gfx11 and on gfx1200 / gfx1201, see AMD_HIP.md); `STRATA_QSA_CLUSTER=0` / `STRATA_ARGMAX_MULTI=0` turn the decode kernels off. The tables
below are 0.1.26's.

### Prompt processing (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 536 | 1,299 | 2,171 | 2,126 | 2,107 | 1,304† |
| **IQ2_XS** | 534 | 1,256 | 2,092 | 1,754 | 1,752 | 1,181*† |
| **IQ3_XXS** | 482 | 1,007 | 1,745 | 1,609 | 1,602 | - |
| **IQ3_S** | 427 | 913 | 1,624 | 1,640 | 1,443 | - |
| **Coder** | 656 | 1,583 | 2,177 | 2,236 | 2,208 | 1,034** |
| **IQ3_S** (AMD RX 7900 XTX, gfx1100) | 760 | 1,275 | 1,641 | 1,594 | 1,494 | - |

Engine 0.1.26; `bench/results/2026-09-29-speed-0126`. The AMD RX 7900 XTX row: engine 0.1.38 on a ROCm
10.2 nightly with a tuned gfx1100 hipBLASLt-100500 table (upstream as [#755](https://github.com/Niko1221/Strata/pull/755)),
the median of 3 clean cells per tier
(Ryzen 9 7900X, 96 GB; 1K-128K one-shot runs, 256 generated tokens,
greedy). At 32K-128K that is 8-28% faster than 0.1.22. † not measured
again: 0.1.22. \* measured with images on (the image encoder's VRAM reserve leaves fewer experts cached). \*\* not
measured again: 0.1.14.

### Output (tokens/s)

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Q2_0** | 87.3 | 93.0 | 81.8 | 76.2 | 73.7 | 60.3† |
| **IQ2_XS** | 79.6 | 78.6 | 76.3 | 63.7 | 62.7 | 52.8† |
| **IQ3_XXS** | 61.9 | 61.6 | 58.5 | 57.2 | 49.0 | - |
| **IQ3_S** | 52.4 | 53.3 | 48.3 | 46.3 | 45.5 | - |
| **Coder** | 58.9 | 55.1 | 54.9 | 53.2 | 43.0 | 42.8† |
| **IQ3_S** (AMD RX 7900 XTX, gfx1100) | 65.1 | 60.4 | 64.4 | 61.8 | 59.2 | - |

Engine 0.1.26, the same runs. The AMD RX 7900 XTX row: engine 0.1.38 on a ROCm 10.2 nightly (tuned
gfx1100 hipBLASLt-100500 table, upstream as [#755](https://github.com/Niko1221/Strata/pull/755)), median of 3 clean cells per tier (same box; decode is flat in
context - the GDN linear attention is O(1) per token; 1K/4K prefill ran slightly under the 0.1.31
packaged-ROCm numbers, 32K-128K ahead). † not measured again: 0.1.14.

Output speed depends on the text as well: speculative decoding runs faster when more of the drafted tokens are
accepted, so a different answer to the same prompt moves it by several percent. Run back to back on the 4K prompt,
0.1.14 writes 88.5 tokens/s and 0.1.12 85.7. The numbers before 0.1.13 (prompts about half as fast):
[`bench/results/2026-09-24-final`](../bench/results/2026-09-24-final/matrix.md); these:
[`bench/results/2026-09-28-speed-0114`](../bench/results/2026-09-28-speed-0114/README.md).

IQ3_XXS and IQ3_S at 262K are not measured: with their 43 / 50 GB of experts, a 260K-token context brings a 64 GB PC
to its memory limit by setup's estimate (the experts + the context's KV cache + 24 GB), so setup recommends up to 128K
with them on 64 GB. A longer context you choose (`--context 262144`, or a pick in its list) is kept, with a note: users
ran IQ3_S at 256K on 64 GB with RAM to spare (#406). In the low-RAM mode the KV cache stays in VRAM and the context
does not count against RAM. IQ3_S (engine 0.1.4 or newer) is published for the original model and for Swift 1.5
(the same shard layout as Swift's other sizes).

**KV streaming (engine 0.1.5):** at 64K and more, setup keeps the context's KV cache in RAM and only the part the
attention reads in VRAM (`--kv-resident 32768`), so more experts fit on the GPU. Q2_0 at 262K: 50.9 -> 62.6 tokens/s
(1,589 -> 3,872 experts in VRAM); at 128K about +6%. The attention reads exactly the same values (only where the KV lives changes); it
costs ~13.7 KB of RAM per context token (1.7 GB at 128K). Existing installs: run `START-HERE.bat --setup` once to turn
it on. Setup turns it on when the RAM has room for it; `--kv-streaming on|off` overrides that (on past the RAM test with a
note; never under WSL, which cannot stream).

**4-bit KV caches (`--kv q4_0` and `--kv k8v4`, optional):** upstream's formats (PR #21 and #120): each head's keys
and values are rotated by a Walsh-Hadamard transform, which spreads their outlier channels over all 256 values, then
stored as 4-bit codes in blocks of 32 with an FP16 scale each. `q4_0` stores the keys and the values that way (576
bytes a token and QSA layer against 8-bit's 1,056: 45% less), `k8v4` the values only, with 8-bit keys (816 bytes: 23%
less). This fork runs them in its tensor-core attention like 8-bit (native packs only; the MTP draft layer keeps
8-bit), and the VRAM they free goes to the expert cache. On an RTX 3090 at 200K context (with an RTX 5070 Ti as the
second tier):

| | 8-bit | k8v4 | q4_0 |
| --- | ---: | ---: | ---: |
| KV cache of the 12 QSA layers | 2.36 GiB | 1.82 GiB | 1.29 GiB |
| Experts in the 3090's cache: IQ3_S / UD-Q4_K_XL | 7,546 / 4,539 | 7,835 / 4,722 | 8,116 / 4,908 |
| Generation, ms per verify pass on fixed text | | 0.1-1.8% less (one workload 1.5% more) | 0.4-1.9% less |
| The prompt's attention (2,048 queries of 2,051 cells) | 4.7 ms | 3.5 ms | 3.3 ms |

The cost is precision. Along llama.cpp's texts, the next-token distributions are 1.3-3.5x as far from llama.cpp's as
8-bit's (mean KL; IQ3_S thinking 2.1e-3 against 4.7e-3 and 7.4e-3, after a 16K-token prompt 1.0e-3 against 2.5e-3 and
3.2e-3; UD-Q4_K_XL 6.7e-3 against 9.0e-3 and 1.2e-2, after 16K 3.5e-3 against 5.4e-3 and 6.4e-3): k8v4 moves them
about as far as one change of rounding does (llama.cpp's CPU backend against its CUDA backend), q4_0 two to three times
that. Upstream measured q4_0's perplexity on long documents 8-12% higher and k8v4's needle results the same as
8-bit's ([`bench/results/2026-09-27-kv-q4`](../bench/results/2026-09-27-kv-q4/README.md)). 8-bit stays the default;
the 4-bit formats also fit a longer context into the same VRAM (q4_0 at 262K takes less than 8-bit at 200K).

**Reproducible greedy output (0.1.30, opt-in, `STRATA_IQ_MT_MIN=1`):** with the IQ models, the CPU computes an
expert for one token with ggml's dot product and for several tokens with Strata's multi-token kernels, which round
slightly differently. How many tokens share an expert depends on the drafts in a verify window, so the same prompt
at temperature 0 can end in a different (equally good) answer when the drafting, the cache state or a resumed
conversation differ (issue #152). `STRATA_IQ_MT_MIN=1` (in the config's `env`) uses the multi-token kernels for
every group: the answer then no longer depends on the drafting. Measured on a Ryzen 7600 (AVX-512): IQ3_S decode
-1..-3%, the other models the same; the default stays the fastest rule. On an Intel CPU of Alder Lake or later
without AVX-512, where the AVX-2 kernel gathers the IQ3_S grid, `STRATA_IQ3S_MT1=1` (opt-in) gives IQ3_S the multi-token
kernel for one token, which is the faster one there; it changes a lone token's rounding, so it is off by default. Through the server, two more things carry
over from one request to the next (#410): the adaptive tier moves experts between RAM and VRAM (the GPU and the CPU
round an expert differently), and the prompt cache resumes a repeated prompt and reads only its tail through the
decode path. For byte-identical repeats add `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0` to the engine's args
as well (#410): the PCIe share of the missed experts (computed on the GPU instead of the CPU) still made the first
answer after a start differ from the next ones. Measured here (IQ3_XXS, a 3.6K-token prompt, 4 repeats): with all
three switches 1 answer of 4, without `--pcie-frac 0` 2 of 4 (the first one differs), with the defaults 2 of 4.
`--pcie-frac 0` costs decode speed (the missed experts all run on the CPU), so keep it for A/B runs.

**Coupled drafts with Gumbel-max picks (opt-in, `STRATA_SPEC_COUPLED=1` and `STRATA_SPEC_GUMBEL=1`):** for a request
that samples (temperature above 0), `STRATA_SPEC_COUPLED=1` lets the draft layer sample its guesses with the target's own
chain and random draw instead of taking its most likely token. `STRATA_SPEC_GUMBEL=1` changes how both of them pick from
that chain: the token with the largest p / E, where E is exponential noise keyed by the seed, the position and the token
id (the Gumbel-max trick; it is still an exact sample of the same distribution). With the default pick, one random number
walked over the candidates sorted by probability, a draft and a target whose candidate lists differ in one token tend to
land on different tokens; with noise keyed by the token, a token gets the same noise on both sides, so they agree on the
tokens they share. Greedy requests are unchanged, and without the variables nothing changes. Measured on a Ryzen AI
Max+ 395 (Radeon 8060S, Linux, ROCm 7.14.1, the iGPU alone), UD-Q4_K_XL, `--spec 4 --mtp` (the base model's draft
layer), temperature 1.0 / top_p 0.95 / top_k 20, a 1.3K-token prompt and 512 output tokens, 12-13 requests per arm:
drafts accepted 52.8% -> 59.9%, tokens per verify window 2.65 -> 2.88, output 41.1 -> 44.9 tokens/s (+9%); a window
costs the same (draft 8.8 -> 9.1 ms of 64). `sampler_parity` checks the pick against a host reference on every sampled
path, and that its frequencies match the softmax. On an RTX 3060 (IQ3_XXS, `--spec 4 --mtp`, temperature 1.0 / top_p 0.95 / top_k 20, 12 interleaved pairs of 200-token story and code requests) `STRATA_SPEC_COUPLED=1` raised the accepted drafts from 62.1% to 66.9% and adding `STRATA_SPEC_GUMBEL=1` to 68.0%, but decode speed did not move beyond the run-to-run spread (median 43.3 tok/s with coupled alone, 42.8 with Gumbel as well): try it on your own card before relying on it.

**The draft layer's tokens (`--draft-vocab`):** the MTP draft layer can only propose tokens from a subset of the
vocabulary (`mtp/rt/draft_vocab.bin`). This fork keeps the English/code subset (40,525 ids, `data/draft_vocab.bin`,
the same file as upstream's `--draft-vocab en`). Upstream's default since 0.1.27 adds every Chinese, Japanese and
Korean token (106,299 ids): answers in those languages 15-38% faster, English answers 1-2% slower, and its head takes
~110 MiB more VRAM. Setup's `--draft-vocab cyrillic` (58,963 ids: the English/code subset and the Cyrillic
script) and `--draft-vocab fr` (46,211 ids: and the tokens of 99% of a French corpus) are upstream's too.
`data/draft_vocab_es.bin` adds the tokens of Spanish text (47,196 ids; the configs in `examples/r7-5700x` use it,
in `mtp/rt-es`): Spanish drafts 68% -> 75% accepted. `tools/draft_vocab.py` builds and inspects subsets.

**Low-RAM mode (engine 0.1.26, chosen by setup):** normally all of a model's experts are copied into RAM (23-50 GB,
pinned) and the GPU holds a copy of the most-used ones. On a PC whose RAM cannot hold them beside the system (the
experts plus ~10 GB), setup instead maps them from one file in the model's folder (`--mmap-experts`, the pack's
`experts.bin`, +23-50 GB of disk). The OS file cache holds what the GPU does not, and it can give that memory back.
On the Coder the engine's committed memory drops from 36 to ~13 GB, with the same answers. With a big GPU (an RTX
5090 holds all of the Coder's experts, most of Q2_0's) it runs at nearly the usual speed. With a small one, most
experts come from the SSD and it is much slower (setup says so). `START-HERE.bat --setup --low-ram on|off` overrides
the choice.

Time to first token is prompt length / prompt speed: with Q2_0 about 4 s at 4K, 25 s at 32K, under 2 minutes at 128K
and 4.5 minutes at 262K (engine 0.1.13 made long prompts about twice as fast, below).

**Faster prompts (engine 0.1.13):** the prompt is read in chunks of up to 8,192 tokens instead of 2,048 (`--prefill
auto`: the largest chunk whose buffers fit in the expert-cache slots it borrows, and a request borrows only what its
prompt needs); the experts are multiplied by llama.cpp's quantized MMQ kernels instead of being expanded to FP16
first; the next layer's experts stream over PCIe while the current layer's attention runs; the PLE block runs for the
whole chunk at once; unpinned experts are copied by helper threads. Measured on the RTX 5070 12 GB, 64 GB RAM,
32K-token prompt: Q2_0 572 -> 1,290 tokens/s, IQ3_S 383 -> 1,208. Through the server (Q2_0, 128K context): 999 tokens
353 -> 438 tokens/s, 6,927 tokens 529 -> 1,077, 28,584 tokens 584 -> 1,249. Output speed is unchanged. Needles 5/5
(1K-262K). Details and the quality check:
[`bench/results/2026-09-28-prefill-speed`](../bench/results/2026-09-28-prefill-speed/README.md). Existing installs
switch to `--prefill auto` the next time START-HERE / setup.sh starts them. The raw numbers:
[`bench/results/`](../bench/results/). The [paper](paper/Strata-Paper.pdf) explains every number.

## Other GPUs (estimated)

Not measured - estimated from the runs above (same CPU and 64 GB RAM): the GPU part scaled by memory bandwidth, the CPU
part by how many more experts the card's VRAM holds. Treat as **±20%**. Numbers are *prompt / output* tokens/s.
The prompt figures predate engine 0.1.13, which about doubled prompt speed on the measured card; how much of that a
card gains depends on its PCIe link (the experts stream over it), so they are still the older estimates.

| GPU | Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5060 Ti 16GB | Q2_0 | ~341 / ~80 | ~472 / ~87 | ~501 / ~81 | ~492 / ~72 | ~476 / ~62 | ~435 / ~53 |
|  | IQ2_XS | ~291 / ~80 | ~406 / ~77 | ~434 / ~63 | ~426 / ~62 | ~413 / ~51 | ~383 / ~47 |
|  | IQ3_XXS | ~249 / ~66 | ~359 / ~65 | ~381 / ~56 | ~374 / ~54 | ~363 / ~45 | - |
| RTX 3090 24GB | Q2_0 | ~355 / ~128 | ~491 / ~140 | ~521 / ~130 | ~512 / ~115 | ~495 / ~100 | ~453 / ~85 |
|  | IQ2_XS | ~303 / ~131 | ~422 / ~128 | ~451 / ~103 | ~444 / ~102 | ~430 / ~85 | ~398 / ~78 |
|  | IQ3_XXS | ~260 / ~106 | ~374 / ~103 | ~396 / ~89 | ~390 / ~85 | ~378 / ~71 | - |

More VRAM matters more than a faster GPU: every extra GB holds ~700 more experts, and every expert on the GPU is one the
CPU does not have to compute. A 3090's 24 GB takes most of the CPU work away. (Since 0.1.14 the expert profile ranks
all 24,576 experts; before, the cache stopped at 8,000, about 10-14 GB. `tools/make_profile.py` builds a profile from
your own prompts: run the engine once with `--dump-routing trace.bin`, see the tool's help. The shipped profile already
ranks every pair, so a trace only changes the order with `--reorder` (0.1.39, #587): your trace's pairs first, then the
base's, then the rest.)

## Which model?

All three are [ISTA-DASLab's GSQ-RCO quantizations](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

| Model | Download | RAM it uses | Speed | Quality |
| --- | ---: | ---: | --- | --- |
| **Q2_0** | 66 GB | ~34 GB experts + ~6 GB | fastest | good |
| **IQ2_XS** | 68 GB | ~36 GB experts + ~6 GB | close to Q2_0 | a bit better |
| **IQ3_XXS** | 76 GB | ~43 GB experts + ~6 GB | slower (more CPU work) | best |

With 64 GB of RAM all three fit (close the browser for IQ3_XXS, and keep its context at 128K or less). With 48 GB only Q2_0 / IQ2_XS may fit. With 32 GB: the Coder (below).

### Or: the Coder (half the experts, for code)

**[Qwen3.8-Flash-Next GSQ-RCO Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)** is
ISTA-DASLab's expert-pruned release: 256 of each layer's 512 experts are kept (still 10 active per token), chosen with
RCO on code, agentic and vision calibration data; its authors report 91.3% of the full model's SWE-bench Verified and
98.7% of LiveCodeBench v6. One size, named IQ1_M for its 1.89 bits per *original* parameter; the kept experts are
stored like IQ3_S (IQ2_S-IQ4_XS gate/up, IQ4_NL/Q2_0 down). Shard 1 is 29.6 GB (experts: 23 GB of RAM), so it runs
on **32 GB of RAM**, and at 262K on 64 GB. Its shard 2 and its vision encoder are the original's files: with the
original installed, setup downloads only shard 1. Strata ships its expert profile (`data/expert-profile-coder.bin`,
the shipped ranking mapped onto the kept experts through the release's `rco-allocation.txt`: 72% of the expert
reads hit the GPU on a 12 GB card). Images work; the experimental speed projection loads and runs on it (it was made
for the full model).

```
START-HERE.bat --setup --family coder
```

### Or: Swift 1.5 (a fine-tune that thinks shorter)

The setup's first question also offers **[Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)**,
UkisAI's fine-tune of Qwen3.8-Flash-Next, trained to reach the answer with much less thinking (its authors: 63% fewer
thinking tokens, 1.8x sooner answers, under 1% accuracy loss). Same architecture, the original's sizes except Q2_0,
its own
vision encoder; Strata runs it at the same speed (4K, IQ2_XS: 465 prompt / 78.7 output tokens/s, vs 467 / 78.3 for
the original). Its authors recommend **IQ2_XS** (their Q2_0 is marked experimental). Its license is the Swift Open
License 1.0 - read it on the model page.

Our small check (8 reasoning questions, default thinking, IQ2_XS): both models got **8/8**; Swift used **1,234**
output tokens in 28 s, the original **2,682** in 46 s - most of the difference from one question the original
thought about for 1,524 tokens. Not a benchmark, but consistent with the claim.

```
START-HERE.bat --setup --family swift --model IQ2_XS
```

### unsloth's Unsloth Dynamic 3.0 files

The engine also runs [unsloth's quantizations](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), which
START-HERE does not set up. All experts sit in RAM, pinned when they fit the driver's limit:

| Model | Download | Experts in RAM | n-gram table |
| --- | ---: | ---: | --- |
| UD-IQ1_S / UD-IQ1_M | 73-75 GB | 40-42 GB | IQ4_NL, 28.8 GB |
| UD-Q2_K_XL / UD-IQ3_XXS | 79-82 GB | 46-49 GB | IQ4_NL |
| UD-Q3_K_XL / UD-IQ4_XS | 90-94 GB | 56-60 GB | IQ4_NL |
| UD-Q4_K_XL | 111 GB | 77 GB | IQ4_NL |
| UD-Q5_K_XL / UD-Q6_K_XL | 158-169 GB | 98-109 GB | Q8_0, 54.4 GB |
| Q8_0 | 188 GB | 128 GB | Q8_0 |

Pack a model once (seconds: the tokenizer and the float tensors; the experts are read from the GGUF at start):

    python tools/iq_pack.py --gguf <dir>/<model>-00001-of-0000N.gguf --out packs/<name>

and start it with `--pack packs/<name> --native <dir>/<model>-00001-of-0000N.gguf` plus the usual flags (`--spec`,
`--mtp`, `--prefill`, ...; this fork's config: [examples/ud-q4_k_xl.json](../examples/ud-q4_k_xl.json)). The
other shards, and the one holding the n-gram table, are found by name.
UD-Q4_K_XL's speeds on an RTX 3090 with 160 GB of RAM, with and without a second GPU: [COMPARISON.md](COMPARISON.md).

### A second GPU

With `--second-gpu N`, CUDA device N (the engine runs on `--main-gpu`, default 0) holds more experts and computes its share of
each layer while the CPU computes the rest, copies each next layer's likeliest missing experts to itself while the RAM
is idle, computes the prompt's experts the first GPU lacks, and holds the last rows of the output head (`--head-split`,
0.45 of them); it may be the card that drives the display. Build for both GPUs' architectures
(`-DCMAKE_CUDA_ARCHITECTURES="86;120"`). When the card has no free VRAM, the engine runs on the main GPU alone. An RTX
3090 with and without an RTX 5070 Ti as the second GPU, against upstream Strata and llama.cpp:
[COMPARISON.md](COMPARISON.md).

## Before you start

You need **only a graphics driver**: NVIDIA 580 or newer (update it with the NVIDIA App or from
[nvidia.com/drivers](https://www.nvidia.com/drivers)), or for AMD the one in [INSTALL.md](INSTALL.md#what-you-need).
Everything else is installed for you the first time.

| | |
| --- | --- |
| GPU | NVIDIA **RTX 20, 30, 40 or 50 series**, **12 GB VRAM or more** (8 GB runs, slowly). Measured on an RTX 5070 and an RTX 3090; RTX 20 (Turing, since 0.1.27) was tested by a contributor on an RTX 2070. Or AMD **Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700, RX 6800 / 6900 series**: [AMD_HIP.md](AMD_HIP.md). |
| RAM | **64 GB** recommended (see the table above). |
| CPU | x86-64 with AVX2 (any Intel/AMD desktop CPU from the last ~8 years). AVX-512 (Ryzen 7000/9000) is a bit faster. Older CPUs without AVX2 are experimental and slow: [Older CPUs](INSTALL.md#older-cpus-experimental). |
| Disk | ~70-80 GB free for the model, ~6 GB for the MTP layer (+1 GB with images). **Q2_0 on an AVX-512 CPU** also writes a one-time ~40 GB pack for the fast CPU kernel (34 GB of experts in its layout, plus the dense weights). An NVMe SSD is strongly recommended. |
| OS | Windows 10/11, or Linux (Ubuntu 22.04/24.04 get everything installed automatically). |

What the first start installs: in this folder `.venv/`, `engine/` and `third_party/`; the model files (`models/`,
`packs/`, `mtp/`, 70-120 GB) in **`Strata-data` next to this folder**, so a new copy of Strata (an update unzipped
elsewhere) finds them and sets itself up the same way. The place is remembered per user (`%APPDATA%\Strata\settings.json`,
`~/.config/strata/settings.json`); `--data-dir` chooses another. Installs from before 0.1.16 are moved there by the next
start (a rename on the same drive; files on another drive are used where they are).
Python 3.12 if you have none (for your user account, no admin), a private Python environment, NVIDIA's CUDA libraries
(from pip, ~0.4 GB), the ready-made Strata engine for RTX 20/30/40/50, the model and the MTP draft layer. If no
ready-made engine fits your PC, it offers to install the build tools (Visual Studio Build Tools + CUDA Toolkit on
Windows, `build-essential` + CUDA on Ubuntu) and compiles the engine for your GPU (asks first; 20-40 minutes once).

---

## Windows

### Double-click `START-HERE.bat`

**The first time** it asks four questions and does the rest:

1. **Which model?** Qwen3.8-Flash-Next (the original) or Swift 1.5 (the fine-tune that thinks shorter).
2. **Which size?** Q2_0, IQ2_XS or IQ3_XXS (it recommends one for your RAM).
3. **How much context?** 8K to 256K tokens (it recommends one for your VRAM).
4. **Images?** yes / no (see [Images](#images-vision)).

Then it downloads and prepares everything (the model is 66-76 GB, so the first start takes a while; an interrupted
download continues where it stopped) and **starts the model**: your browser opens `http://127.0.0.1:8080`, the Strata
app. It has three tabs:
- **Chat:** streaming answers, the model's thinking (folded away once it answers), code with a copy button, pictures when
  images are on, and sampling and thinking-level settings. Chats stay in your browser.
- **Monitor:** what the model is doing (reading the prompt, with progress, or writing, at how many tokens/s); GPU load,
  VRAM, temperature, power and PCIe traffic; CPU, RAM and disk; the context in use; the last requests.
- **About:** the model and engine settings, and the addresses to connect other apps.

`http://127.0.0.1:8080/?q=your question` opens it with a new chat already asking. The API is at
`http://127.0.0.1:8080/v1` for your apps.

**Every time after that**, `START-HERE.bat` just starts the model (30-90 s to load 34-43 GB into RAM). Nothing is
downloaded again. Closing the window stops the model.

```
START-HERE.bat --setup                          install another model, or change context / images
SETUP.bat                                       the same (double-click it)
START-HERE.bat --model IQ2_XS --context 32768 --vision yes --yes     no questions
START-HERE.bat --gguf-dir D:\models\IQ2_XS       use GGUF files you already have
START-HERE.bat --data-dir E:\Strata-data         keep the model files somewhere else
START-HERE.bat --port 8081                      another port
START-HERE.bat --gpu 1                          another GPU (numbered as nvidia-smi; setup picks the one with the most VRAM)
START-HERE.bat --calibrate                      tune the engine for this PC (about 15-30 minutes, longer on a slow card), then start
```

With more than one model installed, it asks which one to start. `run-<model>.bat` starts a model directly.

**Tuning for your PC (`--calibrate`, engine 0.1.19).** Four engine settings depend on the PC more than on the model:
- the share of the experts missing from VRAM that are copied to the GPU instead of computed by the CPU
  (`--pcie-frac`: a fast PCIe link and a slower CPU want more, a laptop's narrower link less);
- how sure the draft layer must be to add another guess to a check (`--spec-min-p`);
- how many CPU threads compute experts (`--pool-workers`: on CPUs with efficiency cores, fewer can be faster);
- how busily the VRAM expert cache follows the conversation (`--adapt-every`, `--adapt-swaps`, `--adapt-decay`): a PC
  whose CPU reads the missed experts slowly (DDR3 or DDR4 in few channels) gains from swapping more. On a Xeon
  E5-2673 v3 with DDR3 and an RTX 4060 Ti on PCIe 3.0 x8, 160 swaps every window measured +7.9% over the default
  (Q2_0, with the swaps taking effect a window later, #764); see #906.

The defaults were measured on a Ryzen 5 7600 with an RTX 5070. Setup offers to measure them on your PC after an
install; `START-HERE.bat --calibrate` (Linux: `./setup.sh --calibrate`) does it any time. It measures the output
speed with each setting and keeps one only when it is more than 3% faster. The result is remembered per PC and model
(in the settings file next to the data folder's record), so updates keep it.

Measuring the worker count needs a fresh engine, so the model is loaded more than once: the PC is busy, and can
stop responding for a minute or two, once per restart. It then starts the model, like a plain `START-HERE.bat`.

**Manual CPU task granularity.** `--pool-tasks N` sets the target total number of row tasks in each batched
CPU expert Gate/Up and Down phase, not the number of threads or tasks per expert. The default `0` keeps
three tasks per participating thread (including the host when enabled). Values `1..4096` are capped by
each phase's row count; native batches larger than the pool's capacity apply the target to each sub-batch.
For example, `--pool-workers 13 --pool-tasks 192` uses 14 participating threads with the default host worker.
More tasks can reduce imbalance between cores, but also add scheduling overhead: compare against `0` with
the same worker count and workload. This does not change kernels, phase barriers, or PCIe placement, and
does not affect the legacy single-token/oracle fallback. Setup's `--calibrate` does not tune it yet.
For the server, add `"--pool-tasks", "192"` to the existing `args` list in its configuration, then restart it.

### Running it at startup (Task Scheduler)

To have the model up at logon, people start the serve from **Task Scheduler** (or a service). Beware: Windows
throttles such contexts, and the model's ~40 GB expert load then crawls at **~0.05 GiB/s (13-14 minutes)**
instead of **~1.4-1.5 GiB/s (~35 seconds)** - a 24x slower start. Measured on an RTX 5070 Ti + Ryzen 7 9800X3D
+ NVMe, same binary, same args, same cache state:

| How the serve starts | Expert load |
| --- | ---: |
| Double-click / terminal / SSH | 1.42-1.52 GiB/s (~35 s) |
| Task Scheduler with its defaults | 0.05 GiB/s (821-841 s) |
| Task Scheduler with the two settings below | 1.42 GiB/s (35 s) |

In the task's properties set both of these (the defaults are the opposite):

- **Priority level: Normal** (Options tab; the default is Below normal), and
- **Run with highest privileges** (General tab; without it the task runs with a limited user token - which
  also strips `SeLockMemoryPrivilege`, the privilege Windows large pages need).

(Both were changed at once, so the isolated effect of each is not measured.) If the model still starts
slowly, the engine prints a hint under its `loaded ... GiB at ...` line naming this cause.

**Large pages need a new logon (#1412).** Granting "Lock pages in memory" (`secpol.msc`, User Rights Assignment) to the account that runs the engine takes effect at the next logon: Windows puts the privilege in the access token when the session starts, so log off and on (or reboot) after granting it. Until then the startup log still says the large pages were refused.

### Chat in the terminal (optional)

```
.venv\Scripts\python chat.py
```

---

## Linux

```bash
./setup.sh
```

The same questions, the same automatic install (it uses `sudo apt` for Python and, only if it has to compile,
for the build tools), and the same start: `http://127.0.0.1:8080`. Later runs of `./setup.sh` (or `./run-<model>.sh`)
start the model directly. Options as on Windows (`./setup.sh --setup`, `--model Q2_0 --yes`, `--gguf-dir /data/Q2_0`).
Terminal chat: `.venv/bin/python chat.py`.

- **Updating:** `git pull`, then `./setup.sh`: it compiles the engine again when its source changed (a minute or
  two for the changed files). If that compile fails, it says so and starts the engine you had.
- **Other distributions** (Arch, Fedora, ...): install the C++ compiler and the CUDA Toolkit 13 with your package
  manager first (Arch: `sudo pacman -S base-devel cuda`); setup finds `nvcc` on PATH, in `/usr/local/cuda*` and in
  `/opt/cuda*`, and does the rest.
- **WSL** works (Ubuntu 24.04 tested), with one limit: the NVIDIA driver pins only about 1 GB of RAM there, so KV
  streaming (`--kv-resident`) is off and the KV cache stays in VRAM, and the experts are copied to the GPU from
  unpinned RAM (slower prompts than native Linux).

---

## Sharing the GPU with other programs (optional)

By default the model stays loaded until you close Strata. On a PC that also games, renders or runs another model
server, three server options (all off by default; also as keys in `strata-<model>.json`) give the VRAM back:

| Option | Config key | What it does |
| --- | --- | --- |
| `--idle-unload 600` | `"idle_unload_s": 600` | unload the model after 600 s without requests; the next request loads it again |
| `--min-free-vram-mib 11000` | `"min_free_vram_mib": 11000` | load an unloaded model only when that much VRAM is free (it waits up to 15 s for memory being given back), else answer **503** "the GPU is in use by another program" instead of starting into what a game left (with several GPUs it checks the first one: the config's first `"gpu"`, or the engine's `--main-gpu`) |
| `--before-load "cmd"` | `"before_load": "cmd"` or `["cmd", "arg"]` | a command run before the model is loaded again, e.g. one that unloads another server's model |
| | `"before_load_timeout_s": 120` | how long that command may run (default 120 s). When it exits non-zero or times out (a peer that is not running makes `curl` exit 7), the free VRAM decides: with `min_free_vram_mib` set and not enough free, the model is **not** loaded and the request gets **503** with `Retry-After`; with room (or no limit set) it loads, with a warning in the log. `"before_load_strict": true` always refuses on a failure; `"before_load_ignore_failure": true` always loads |

`POST /unload` unloads it now (`409` while a request is running; `POST /unload?wait=1` waits for the running turn first, up to 300 s, or `?wait=<seconds>`, so a second server can really free the card) and `POST /load` loads it ahead of a request (both
with `Content-Type: application/json`, e.g. `curl -X POST -H "Content-Type: application/json" localhost:8080/unload`);
`/health` says `"loaded"`, `/v1/models` lists it as `unloaded` (like llama.cpp's router), `/props` sets
`is_sleeping` and the Monitor shows the state. Unloading ends the engine process - and the image encoder, when images
are on; it is started again first, as at a start - so their VRAM and RAM go straight back. The model files stay in
the OS file cache, so loading again takes seconds while that RAM is not needed elsewhere. Measured on an RTX 5060 Ti
16 GB with Q2_0 in the low-RAM mode: unloading takes ~0.3 s, and a request to an unloaded model answered after
4.6 s (text) or 14.7 s (a picture, image encoder on the CPU).

**Giving part of the VRAM back while it keeps serving (#533, opt-in, one NVIDIA GPU; every request and answer:
[VRAM_ELASTIC.md](VRAM_ELASTIC.md)).** With `"vram_elastic": true`
in the config (the engine flag `--vram-elastic`), the expert cache is allocated in 512 MiB segments
(`"vram_segment_mib"`), and `POST /v1/vram` with `{"reserve_mib": 8000}` shrinks it between requests until that much
VRAM is free for another program; `{"reserve_mib": null}` grows it back towards its full size, keeping the reserve
the engine started with (`--vram-reserve-mib`), and `{"reserve_mib": 0}` takes all of it back. A request that is
running finishes first. The experts of the segments given back are computed on the CPU, like any expert outside the
cache, so answers keep coming, slower; growing back puts the same experts in the same slots. Nothing resizes on its
own. Not with a layer split, the helper caches, `--peer-device` or the resident low-RAM mode. Measured on an RTX 5070
12 GB with Q2_0 (4.8 GiB cache): `{"reserve_mib": 6000}` took 78 ms and freed 4.3 GiB (the cache keeps 0.5 GiB for
the prompt path), decode 44 -> 33 tok/s; growing back took 92 ms and the answers were token for token the ones before
the shrink. Without the flag nothing changes (the same answers as without it).

**Keep what the expert cache learned across restarts (opt-in, engine 0.1.36, #477):** a start fills the GPU's expert
cache from the shipped profile, and the adaptive tier (`--adapt-every`) then moves in the experts your requests use.
With `"expert_profile_save": "expert-profile-learned.bin"` in `strata-<model>.json` the engine saves that as a
profile - the experts in VRAM first, then the routing it counted since the start, then the shipped order - on a
clean exit and every 10 minutes between requests (`"expert_profile_save_every": 5` for another interval, `0` for
exit only), written to a temporary file and renamed, so a crash never leaves half a file. The next start begins from
it instead of the config's `--expert-profile` when it is a profile of the same model (else from the config's, as
before). A relative path is in the Strata folder; one file per model, and a profile per project works the same way
(point the key at another file). The file is a fingerprint of what you used the model for: it stays on your PC.
Without the key nothing is counted or written. Setup rewrites the config when run again: add the key again then.

**The expert cache size is a budget, and on Windows an over-sized one pages instead of failing.** `--expert-cache N`
(`"expert_cache": N`) is `N` times the largest expert blob, and the engine compares it only with the free VRAM it
reads **before** the slots are written. `--expert-cache auto` (the default) sizes from that same figure but **checks
again once the slots are actually written** and shrinks if they do not fit; that second check is auto-only (the loop
breaks on `!auto_cache`). Under WDDM an allocation is not resident until it is touched, and the free figure read
before it can be about a gigabyte too high, and the driver's default sysmem fallback puts an over-committed
allocation in system memory instead of failing - so the start looks normal, the banner still reports the slot count,
and only the speed collapses. The startup line is the tell: `... MiB of VRAM free with everything loaded`, and below
about 256 MiB it adds `LOW: requests may stall; add --vram-reserve-mib N` with the value that would have left room.

Measured on an RTX 4080 SUPER 32 GB, IQ3_S, engine 0.1.39, `--max-context 1048576` (yarn factor 4), fresh
43,969 / 45,670 / 47,956-token prompts with 200 generated each, one engine start per row (the rest of the
configuration is the one in `bench/results/2026-10-04-community-rtx-4080s-iq3s`):

| expert cache | slots | VRAM free at start | decode tok/s |
| --- | ---: | --- | --- |
| `--expert-cache 8900` (11,631 slots, 22.07 GiB) | 11,631 | 0 MiB (`LOW`, suggests `--vram-reserve-mib 1212`) | 13.7 / 14.7 |
| `auto` | 11,178 | 217 MiB (`LOW`) | 102.1 / 122.3 |
| `auto` with `--vram-reserve-mib 1500` | 10,766 | 1,075 MiB | 88.9 / 96.5 / 101.4 |
| `--expert-cache 7000` | 9,148 | 4,130 MiB | 94.1 / 97.3 |
| the same 11,631-slot cache at `--max-context 524288` | 11,631 | 299 MiB | 113.2 / 107.9 |

453 slots (0.86 GiB) separate 13.7 from 102 tok/s, and the slower run had the **higher** cache hit rate (91.4 / 94.7%
against 90.4 / 93.9%), so this is not misses: being at 0 MiB free is what costs 7x. Two ways to stay out of it: leave
the cache on `auto`, or keep an explicit size and raise `--vram-reserve-mib`, which the engine deducts **before** it
sizes the cache (700 -> 1000 -> 1500 took the free figure from 217 to 575 to 1,075 MiB and cost about 400 slots).

**Checking it on Windows.** `\GPU Process Memory(<pid>)\Shared Usage` and `Dedicated Usage` (Performance Monitor, or
`Get-Counter`) look like the right instrument, but on this machine they did **not** separate a 13.7 tok/s run from a
102 tok/s one: `Shared` read about 55.5 GB at 262/524K and 62.4 GB at 1M in every run, because it counts the
deliberately pinned memory (the expert arena, 50.3 GB here, plus the pinned K/V, 6.19 -> 12.38 GiB at 1M). The free
line in the log is the reliable tell, and on NVIDIA hardware `CUDA - Sysmem Fallback Policy` set to
`Prefer No Sysmem Fallback` for the engine's executable is the other way to make an over-sized cache fail instead of
paging (not measured here). Reported with the measurements in
[issue #781](https://github.com/Niko1221/Strata/issues/781).

The same shape appears on a second, smaller card. On a 16 GB RTX 5080 at 786,432, UD-IQ4_XS, INT8 KV, with
`Prefer No Sysmem Fallback` set and an explicit `--expert-cache 4000`, taking `--vram-reserve-mib` down from 770 to 0
grew the cache from 2,782 to 3,115 slots (768 MiB) while the card's used memory moved 129 MiB, the hit rate stayed
flat (59.9% to 63.0%) and decode fell from 40.9 to 16.6 tok/s. Every step there reached READY with 0 MiB free, so
with the fallback off it still starts and degrades instead of failing - the slack left for the rest of the card is
worth more than the extra slots, which is the same ordering measured above (enkynakamura, #781; one 256-token sample
per step, so the upper rows are noisy).

**The free figure moves after startup.** The `... MiB of VRAM free with everything loaded` line is read before the
verify windows are captured (each captured window, and each `--batch` slot layout, takes some tens of MiB: 538 MiB at
that line was 395 MiB after the last capture in #2065, an RTX 4090 D with `--batch 2` at 262,144 context). Below about
a gigabyte on Windows the driver can page GPU memory to system RAM instead of failing: decode fell from 91 to 3 tok/s
at 70K tokens and then a window never finished, the watchdog stopping the engine after 60 s. From 0.1.43.1 each request
re-reads the free VRAM, says `WARNING: only N MiB of VRAM free at the start of this request` once it is under 1 GiB
(512 MiB elsewhere) with the `--vram-reserve-mib` that would leave room, and the stall report prints the figure. The
engine only says it; the fix is a larger `--vram-reserve-mib`, a smaller `--max-context` or `--batch`, or
`Prefer No Sysmem Fallback` for the engine.

**A stall named "request" can be the OS paging the engine's own memory.** Before a request that switches
conversation, the engine parks the outgoing one in RAM: copies of its running checkpoints and K/V, which then sit in
ordinary memory until the conversation comes back. On a 64 GB PC with `--resident-budget-gib 40`, an 8 GiB conversation
cache and a 256K context, the working set was 8-30 GB of 56 GB committed and the engine took about 190 page faults
a second (under 1 MB/s if each is a pagefile read) in #2049: one silent step of more than 60 s, five times, on 0.1.41
and 0.1.43 alike (thread dumps: the main thread inside the copy of the checkpoints, at the same instruction each time). The log's `verify window (last window, not the
current stage) ... host at layer step 48; the GPU rang 48` line in that report is the previous request's last window,
not the stalled one. From 0.1.43.1 the park is named in the stage (`request: parking the previous conversation`), its
copies report progress to the watchdog, and Windows counts page faults at 8 a unit instead of 64 when it decides
whether the file tier is busy. The cure for the slowness itself is memory: a smaller `--resident-budget-gib`, a smaller
`--conversation-cache-mib`, or the pagefile on an SSD.

---

## Using it

The server listens on `http://127.0.0.1:8080` (change with `--port` in setup, or edit the run script).

| API | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#the-responses-api-and-codex-cli)) | `POST /v1/responses` |
| A choice among single-token options in one forward pass, nothing generated ([below](#one-pass-choices)) | `POST /v1/choose` |
| Model list / health | `GET /v1/models`, `GET /models`, `GET /health` |
| Model properties | `GET /props` (also accepts `?model=<loaded-model-id>`) |
| What the model is doing right now | `GET /status`, `GET /slots` (busy or idle, with `n_prompt_tokens`; one entry per slot with `--batch`) |
| Save / restore the conversation to a file (session files, below) | `POST /slots/0?action=save\|restore` |
| Everything the Monitor tab shows (engine, live state, last requests, hardware) | `GET /metrics` |
| The same for Prometheus, with vLLM's metric names (asked with `Accept: text/plain` or `?format=prometheus`) | `GET /metrics` |
| Request totals and what is running in Prometheus' text format, under llama-server's metric names | `GET /metrics/prometheus` |
| The MCP servers, their state and tools ([below](#tools-from-mcp-servers)) | `GET /mcp` |

`GET /metrics` answers a Prometheus scrape (`Accept: text/plain` or `application/openmetrics-text`) in the text
format with vLLM's names - `vllm:num_requests_running` / `_waiting`, `vllm:kv_cache_usage_perc`, the token and
request counters, `vllm:prefix_cache_queries_total` / `_hits_total` (prompt tokens read / reused),
`vllm:spec_decode_num_draft_tokens_total` / `_accepted_tokens_total` (the MTP drafts), and the histograms
`vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds` and `vllm:e2e_request_latency_seconds` - so the
dashboards and alerts written for a vLLM server read this one. With an API key, the scraper sends it as a bearer
token. Any other request keeps the JSON.

The JSON's own facts that vLLM has no name for come in the same scrape under `strata:`, named after their JSON key,
so a Monitor-tab panel and a Grafana panel read the same value: `strata:live_state{state="..."}`, `strata:live_tok_s`,
`strata:live_prefill_tok_s_mean`, `strata:live_prompt_read`, `strata:engine_max_context`,
`strata:totals_prompt_seconds_total` / `strata:totals_decode_seconds_total`, `strata:last_hit_rate` and
`strata:last_decode_tok_s` (the last request's), and the hardware - `strata:gpu_util`, `strata:gpu_mem_used_bytes`,
`strata:gpu_mem_total_bytes`, `strata:gpu_temp_celsius`, `strata:gpu_power_watts` (one sample per card, label
`gpu`), `strata:cpu`, `strata:ram_used_bytes`, `strata:ram_total_bytes`. A value the server does not have (no GPU
telemetry, an older engine) has no sample rather than a zero.

Ready to use, in `docs/monitoring/`: `prometheus.yml` (a scrape config, the API key as a bearer token),
`servicemonitor.yaml` (the same for the Prometheus Operator) and `grafana-strata.json`, a Grafana dashboard to
import (it asks for the Prometheus data source): requests running and waiting, tokens per second, time to first
token, time between tokens and request duration (p50 / p95), the prompt cache and MTP draft rates, and the engine's
own panels (state, batch slots, rates, expert cache hit rate, GPU and RAM). Its first row uses only vLLM's names, so
it also reads a vLLM server.

`/models` and `/v1/models` list only the loaded model, with its context limit and input modalities. `/props` exposes the original chat template, context limit, configured generation defaults (shared settings take precedence), model path and engine version when available. Context means the full engine context, not the resident KV window. `n_predict: -1` means no fixed output cap. Unconfigured sampling fields are omitted. `/slots` uses llama.cpp's names: `n_ctx` and `n_prompt_tokens` (the running request's prompt size, kept after it ends: what a front-end's context meter divides by `n_ctx`). `total_slots` in `/props` is the number of `--batch` slots (1 without). `autoload` has no effect; an unknown `model` returns 404. These metadata endpoints and `/slots` require the API key when one is configured. They do not load, unload or restart models.

An answer's `timings` (llama.cpp's names) and its `/metrics` record also carry `expert_tiers`, the expert tiers over
that request: the adaptive tier's moves on the main GPU (`main_swaps`, `main_fills`, `main_dropped`: moves an update
dropped before they were copied) and, with a second GPU, its routed entries, experts and layers, the layers the CPU took
its share of, its tier's moves, its empty slots, the experts prefetched to it and of those used, and
`deterministic_adapt`. The engine sends them as a `TIERS key=value ...` line before each `DONE`.

`chat_template_caps` uses llama.cpp's field names. Strata checks them when loading the active template by rendering
small requests; a rejected or omitted feature is `false`, and a template that fails a probe in any way only turns that
feature off (the server still starts). `supports_tools` checks tool definitions and XML call
instructions; `supports_tool_calls` checks that calls in the history use Strata's XML format and keep tool results.
`supports_parallel_tool_calls` checks multiple calls in one assistant turn, and `supports_system_role` checks a
leading system message. `supports_preserve_reasoning` means older assistant turns keep their `reasoning_content`
in the rendered prompt. It does not describe whether a new reply returns reasoning. These are compatibility hints
for clients, not a guarantee that the model will follow a request. Zed's llama.cpp provider reads this object to
offer tools.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "strata", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

- **Thinking levels: none, low, medium, high.** The model thinks before it answers (streamed as
  `reasoning_content`, Anthropic: `thinking` blocks). Choose how much per request - in the chat page (the "Thinking"
  menu), in `chat.py` (`/think low`), or over the API:

  | API | how |
  | --- | --- |
  | OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
  | Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

  Without a setting the model uses its own default, **high**. `none` answers at once (fastest); `low` keeps the thinking
  short. The levels are instructions the model was trained with, not a hard token limit: on easy questions all three
  think briefly, on hard ones `high` thinks longest and is most accurate.
  To start with another level for every request that names none, put `"reasoning_effort": "low"` (or `none`, `medium`,
  `high`) in `strata-<model>.json`, outside `"args"` (#1641): the engine has no such option, and `--reasoning-effort` in
  `"args"` is moved to this key with a note at startup. The web app's shared Chat settings ("Use for other apps too") and a
  request's own level win over it.
- **Keep Windows awake while it works (opt-in, #1727).** `"prevent_sleep": true` in `strata-<model>.json` keeps the PC
  from going to sleep while at least one request is running or waiting (Windows' `SetThreadExecutionState` with
  `ES_SYSTEM_REQUIRED`; the screen may still turn off), and gives the normal sleep rules back when the server is idle
  and when it exits. Default off. On Linux and macOS it does nothing (one line at startup says so).
- **A hard thinking budget (opt-in).** `"reasoning_budget_tokens": N` in a request (OpenAI or Anthropic) caps the
  thinking at N tokens: when it gets there the server ends it with a short wrap-up line and `</think>`, and the model
  answers from there (the engine continues from what it already holds, so nothing is read again). The wrap-up is
  part of the thinking the client sees and counts as output tokens. `"reasoning_budget_tokens": N` in
  `strata-<model>.json` sets it for every request; a request's own value wins, and `0` means no budget. Off by default;
  Anthropic's `"thinking": {"budget_tokens": N}` still only chooses the level, as above.
- **A reply that ends inside its thinking (#1053, opt-in).** Some turns write a sentence of reasoning and then the
  end-of-turn token with no `</think>`: the content is empty and an agent stops. `"reasoning_close_retry": true` in
  `strata-<model>.json` closes the thinking once (as the thinking budget does) and continues, once per request, only for
  a reply that ended that way with no answer and no tool call. Off by default.
- **A reply stuck on one token is ended (0.1.39, #606).** When a reply repeats the same token 256 times in a row, the
  server ends it there with `finish_reason` `"length"` and says so in its window: a model in a loop, or a broken
  state that answers one token forever (#606 saw 36,689 tokens of `!`). `"repeat_stop_tokens": N` in
  `strata-<model>.json` sets the run length; `0` turns it off (for a request that really wants one token many times).
- **Non-finite logits end the request and drop the cached state (#879, on by default).** Rarely, a window's logits
  become NaN and the reply degenerates into token 0 (`!`) repeated (#606); the poison sits in the conversation's K/V
  and recurrent state, so with the prompt cache every later turn of that conversation stays poisoned. The q8_1 clamps
  of 0.1.39 and later remove the overflows found so far; this guard catches whatever cause is left. Every verify window
  of `--serve` (the decode windows, the `--pipeline-windows` decode windows of a layer split, the prompt's short
  tail, batch slots) checks its head logits on the GPU beside the greedy pick (one small kernel into mapped flags, read
  after the sync the window already does). A flagged window emits none of its tokens, the engine prints one
  `non-finite logits (#879)` line with the position, ends the request with `DONE ... nonfinite` (a slot:
  `BDONE <slot> <n> nonfinite`), and the next request first drops the live session, every checkpoint, every parked
  conversation and every idle slot (a slot still running finishes its reply and is not kept after it) and reads its
  prompt from token 0. The server retries a request that had sent nothing yet once, transparently; a reply that had
  started ends as `"length"` with a line in the server window; a second fire on the retry ends the request with an
  error (503 before an answer, an error event in a stream). With replicas, the server forgets what that replica held.
  `repeat_stop_tokens` stays as the backstop. The kernel writes only its flags and the host only reads them, so an
  answer without a NaN is the one the engine gives with `STRATA_NAN_GUARD=0`, which records no kernel: on an RTX
  3080 with the RVN IQ3_S config at fixed expert sets (a chat replay of 8 turns on a 16K-token system prompt), the
  guard on and off gave byte-identical answers in 3 pairs (one of them with `STRATA_ROUTE_TAIL_SKIP=0`, one with
  `STRATA_SH_STREAM=0`); one more guard-off run differed from all the others from its first answer on, while a second
  guard-off run matched the guard-on runs, so that was a difference between runs, not the guard. Its cost there: 4.8 us of GPU time a window on average (7.8 us at most), 0.009% of the decode
  time (Nsight Systems, 424 windows). The one-shot command line (`strata` without `--serve`) and the Intel (SYCL) port
  in `sycl/` do not check.
  Debug switch: `STRATA_NAN_INJECT=<n>[,<n>...]` reports the n-th window the guard checks in the engine process
  (counted from 1 over prompt-tail, decode and batch windows) as non-finite, to exercise the whole path without a real
  fire. With `=1,60` on that 3080: a new chat's first window was flagged and the server sent the request again from
  token 0 (the client saw one normal answer); the 60th, mid-reply in the next turn, ended that reply as `"length"`
  after 58 tokens; the turn after it reused 0 cached tokens and answered normally.
- **Repeated reasoning (opt-in, #728).** The single-token guard above does not see a model that repeats whole
  passages. `"reasoning_loop_recovery"` in `strata-<model>.json` is `false` (the default), `"stop"` or `"recover"`
  (`true` means `"recover"`). Every 512 output tokens, at a complete character and parser boundary, the reasoning is
  measured over its last 2,000 words and punctuation marks (counting passages over the last 30,000 words), and its
  last 2,048 characters are checked for a period of at most 64 characters (#1753: a loop written as one long word,
  such as a 24k-digit string cycling an 11-digit pattern, is a single word to the count and never repeats one
  token); a periodic tail counts as fully repeated. If at
  least 25% belong to 12-word passages seen three times, `"stop"` ends the reply there as `"length"` and says so in
  the server window. `"recover"` stops and drains that generation, then goes on once from all its generated token
  ids with the template's low-effort sentence in place of the xhigh one in the first system message (a splice of
  token ids: the rest of the prompt is not decoded or re-encoded). The task stays the same; no answer or `</think>`
  is inserted, and both passes share the original output limit. On recovery only, temperature is raised to at least
  1.0 and presence penalty to at least 1.5; a client's top-p, top-k and seed are never changed. `/metrics` records
  `reasoning_recoveries` and the coverage. This is a policy, not a numerical engine fix or a guarantee of an answer.
  `"recover"` needs the exact xhigh sentence in the first system message and skips requests with images; re-reading
  the changed prefix costs prompt time. Either mode can mistake repeated useful code or checks for a loop, so keep
  the recorded answer quality alongside the completion rate when testing it.
- **Changing the effort without re-reading the prompt (opt-in, 0.1.39, #458).** The effort's instruction is the
  first thing in the prompt, so a request that only changes the effort (an agent's "think harder" switch, `none` for
  a quick tool step) reads the whole conversation again. `"effort_position": "end"` in `strata-<model>.json` renders
  every request's prompt start as the default effort's and puts a `low` / `medium` effort in a short system turn
  right before the answer (no thinking: the empty thinking block, as always); the engine (0.1.39+, which the server
  checks) keeps its checkpoint in front of that turn, so the next request reuses the conversation whatever its
  effort. The default (`"start"`) prompt is unchanged. The model sees a level that is not the default in another
  place than it was trained with; how well it follows it there is not measured yet.
- **Anthropic requests that don't ask for thinking (opt-in, 0.1.32, #278).** By default a `/v1/messages` request
  with no `"thinking"`, effort or budget thinks as the model's template does. `"anthropic_thinking": "on_request"` in
  `strata-<model>.json` renders such a request without thinking - Anthropic's own rule, and what Claude Code's short
  helper calls (a session title in a few dozen tokens) need; its real turns ask for thinking when it is on there.
- **Streaming.** With `"stream": true` everything arrives as it is made: the thinking, the answer, and tool calls
  (the tool's name first, then its arguments piece by piece, like OpenAI and Anthropic do). While the model reads a
  long prompt the stream sends keep-alives, so agents do not time out; the server window prints progress every
  15 s, and `GET /status` says what it is doing (`reading the prompt`, `answering`, tokens so far). Closing the
  connection or pressing stop in your app really stops the model, so the next request starts at once.
- **Tool calls in other forms (opt-in).** `"tool_call_recovery": true` in `strata-<model>.json` reads a call of a
  tool the request declared also when the model writes it as `<parameter=NAME>` instead of `<function=NAME>`, as JSON
  (`{"name": ..., "arguments": ...}`) inside `<tool_call>`, as a `<function=NAME>` block at the start of a line
  without `<tool_call>` (outside code), or as a second call inside the same `<tool_call>` (without the switch the
  second call's parameters merge into the first). Anything else in those forms stays the text it is. Measured on
  1,462 agent turns (Qwen3.8 under Claude Code, from signalnine/q27's drift corpus): 98.3% read as intended with it,
  92.9% without. Off by default, so what a client gets back is unchanged unless you turn it on.
- **Requests without tools.** Tool calls are taken out of the text only when the request declares tools. Without
  them the answer is the model's text as written, `<tool_call>` tags included, for clients with their own text tool
  protocol (OpenAI's API never returns `tool_calls` for a request without tools either).
- **Prefill progress in the stream (opt-in).** `"return_progress": true` puts that progress on the stream instead of
  sending only the keep-alive, as one extra field on a chunk with an empty delta: `prompt_progress` with `total`,
  `cache`, `processed` and `time_ms`. Those are llama.cpp's four fields and mean the same there (`time_ms` is the time
  since the prompt started reading, and the work still to do is `(total-cache) - (processed-cache)`), so a client that
  draws a prefill bar for llama.cpp draws one here too. It is off unless the request asks, as it is in llama.cpp.
  The engine says one line per `--prefill` chunk, so that chunk is the step: measured on two RTX 3090s with
  `--prefill auto` (8192 tokens), a 42,131 token prompt read in 15 s sent six of them. A prompt shorter than one chunk
  sends nothing, and that is on purpose: its only line arrives once the prompt is read, because the last tokens go
  through the verify windows rather than the batched path, so it stops up to `--short-read` tokens short of the end.
- **Chat apps.** Any app with an "OpenAI-compatible" provider works: base URL `http://127.0.0.1:8080/v1`, any API key.
  Zed's llama.cpp provider reads `/props` and offers tools from `chat_template_caps`. Its generic OpenAI-compatible
  provider does not: add `"capabilities": {"tools": true, "chat_completions": true}` on that model in Zed's settings.
- **OpenCode** (#543). A starting point for `opencode.jsonc` (in your project, or `~/.config/opencode/`); the field
  names are OpenCode's, so check its config docs if your version differs:

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "strata": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Strata (local)",
        "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "none" },  // or your api_key
        "models": {
          "strata": {
            "name": "Qwen3.8-Flash-Next (Strata)",
            // context: what you chose in setup; output: what one reply may use (prompt + output must fit)
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },                  // sent as reasoning_effort
            "variants": {                                              // switch between them in OpenCode
              "low": { "reasoningEffort": "low" },
              "medium": { "reasoningEffort": "medium" },
              "none": { "reasoningEffort": "none" }
            }
          }
        }
      }
    },
    "model": "strata/strata"
  }
  ```

  Set `limit.context` to the context you chose in setup: OpenCode compacts the conversation before it gets there.
  Keep `limit.output` well under it: a request whose prompt plus `max_tokens` runs past the context is refused (see
  **Context** below), or add `"fit_max_tokens": true` to `strata-<model>.json`. For a hard cap on the thinking, add
  `"reasoning_budget_tokens": N` to `strata-<model>.json` (see above).
- **Claude Code** (Strata 0.1.17 or newer): set `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` and
  `ANTHROPIC_MODEL` to a Claude model name it knows (it refuses names it doesn't; Strata ignores the name), plus any
  `ANTHROPIC_AUTH_TOKEN` (or your `api_key`, if you set one).
- **Codex CLI** (0.1.39): see [the Responses API](#the-responses-api-and-codex-cli) below.
- **Context.** Chosen in setup (8K-262K). Requests longer than that are refused, never silently cut. A request whose
  `max_tokens` would run past the context is refused too (400); agents that always ask for their full output cap
  can instead get it shortened to the room left: add `"fit_max_tokens": true` to `strata-<model>.json` (or pass
  `--fit-max-tokens` to `serve/server.py`). A prompt that leaves no room at all is still refused.
- **Prompt cache.** A request continues from the longest start of its prompt the engine still holds, so the next turn
  of a chat or an agent processes only what is new (a 16K-token conversation: ~0.3 s instead of ~25 s before the first
  token). It keeps checkpoints of the sequence state in RAM (at every turn's start and end, and every 16K tokens of a
  long prompt: regenerated and edited answers resume from them; for agents also where a prompt leaves the previous
  request's, and before the end of a last message that grows between requests, so one long context asked different
  questions, or a transcript growing inside one message, is read once: a 6K-token prompt's next request in ~0.5-1 s
  instead of ~10 s on an RTX 3060) and moves a conversation that another request replaces (a chat app's title
  request, a subagent) to RAM with its key/value cache, so switching back continues where it was.
  Responses report the reused tokens (`usage.prompt_tokens_details.cached_tokens`; Anthropic:
  `cache_read_input_tokens`). Engine options: `--prompt-cache N` checkpoints (default 16, ~110 MB each; 0 = off),
  `--cache-ram MIB` for replaced conversations (default 16384), `--cache-every N` (default 16384). A changed system
  prompt or reasoning effort changes the prompt from its first tokens, so nothing is reused.
- **Model aliases** (0.1.32). `"aliases": ["qwen", "local-model"]` in `strata-<model>.json` lists the model under
  those names too in `/v1/models` (each with its own `id`, and in the model's `aliases`), like llama-server's
  `--alias`; a request naming one is answered under that name. Any other name is still served, as before.
- **Model settings in the web page (0.1.39, #564).** The About tab's Model settings card shows and changes a few of
  the keys above in `strata-<model>.json`: the `sampling` defaults (temperature, top_p, top_k, min_p),
  `reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases`, `idle_unload_s`,
  `lazy_load`, `engine_silence_s`, `api_monitor`, `open_browser` and `--vram-reserve-mib`. An empty field removes the
  key (its default). Every other key of the file stays as it is, the earlier file is kept as
  `strata-<model>.json.bak`, and the model uses the change from its next start. Only Strata's own page can save
  (JSON, the API key when one is set, as for the Chat settings); the network, key, MCP and program keys are not
  editable there.
- **From other devices on your network.** The server listens on your PC only (`127.0.0.1`) unless you say otherwise:
  run setup with `START-HERE.bat --setup --host 0.0.0.0 --api-key some-long-secret` (or add `"host": "0.0.0.0"` and
  `"api_key": "..."` to `strata-<model>.json`). The server window then prints this PC's addresses
  (`from other devices: http://192.168.x.x:8080/`); open that on the other device, or use `.../v1` as an API base URL.
  On Windows the firewall blocks it until you allow it: accept its prompt for Python (private networks), or run
  `New-NetFirewallRule -DisplayName "Strata 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private`
  in an admin PowerShell, and make sure the network is set to Private.
- **From the internet.** Put a tunnel in front of it, for example [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):
  `cloudflared tunnel --url http://127.0.0.1:8080`. **Set a key first**, or anyone with the link can use your PC:
  add `"api_key": "some-long-secret"` to `strata-<model>.json` (or set the `STRATA_API_KEY` environment variable);
  clients then send it as their API key. Several keys (one per client, so one can be withdrawn alone): separate them
  with commas, `--api-key key1,key2` or `"api_key": "key1,key2"` as llama.cpp does, or give a list,
  `"api_key": ["key1", "key2"]` (the way to use a key that contains a comma); a request passes with any one of them.
  Streamed answers carry `X-Accel-Buffering: no`, so nginx-style proxies pass
  each token on at once. The web app's settings and MCP tools only answer Strata's own page: when you open it through
  a proxy or tunnel whose address differs, add that address, e.g. `"trusted_origins": ["https://strata.example.com"]`.
  With the key set, any `Host` name reaches the server (see Host names below).
- **From web apps in a browser (CORS).** Off by default. `"cors_origins": ["https://chat.example.com"]` lets pages of
  those origins call `/v1/*` from the browser (Open WebUI's direct connections, browser extensions); `["*"]` lets any
  page do it - only sensible with an API key. It never opens `/settings`, `/unload` or the MCP tools.
- **Host names (DNS rebinding).** A web page of another site can point its own name at `127.0.0.1` and then reach
  this server as if it were its own, so without an API key the server answers only requests whose `Host` is a name
  it knows (with a key the check is off: such a page cannot send the key, and tunnels and proxies that pass their
  own name on keep working):
  `localhost` (and `*.localhost`), any IP address (`127.0.0.1`, `[::1]`, `192.168.x.x`, ...), the address it
  listens on and, when it listens beyond this PC (`0.0.0.0` or a LAN address), this PC's name (`mypc`, `mypc.local`)
  and `host.docker.internal`; any port. Others get **403** naming the setting, and the server window prints one line
  for each. Reach it under another name (a reverse proxy that keeps the name, a tunnel, a DNS name on your network,
  another container's name for it)? Add the name: `"allowed_hosts": ["strata.example.com"]` in
  `strata-<model>.json` or `STRATA_ALLOWED_HOSTS=strata.example.com` (comma-separated); `".example.com"` allows that
  name and every name below it, and `["*"]` turns the check off (so does setting `api_key`). The hosts of
  `trusted_origins` count as allowed. Requests without a `Host` header (HTTP/1.0 clients) pass.
- **Web pages without an API key.** Without `api_key`, a `POST` to `/v1/*` that carries an `Origin` header (a
  browser page sent it) is answered only for Strata's own page, pages on `localhost` or an allowed host name (any
  port), the origins in `trusted_origins` or `cors_origins`, and browser extensions and desktop apps
  (`chrome-extension://`, `moz-extension://`, `app://`: no web site can send those), and only with a JSON body; any
  other page, and `Origin: null`, gets **403**. Clients that send no `Origin` (curl, the OpenAI and Anthropic SDKs,
  other servers) are not affected. With
  an API key, the key decides. `POST /unload` and `POST /load` take `Content-Type: application/json` from Strata's
  own page (or no `Origin`), like `/settings`. `POST /slots/0?action=save|restore` keeps the Host and API-key checks
  and also takes only JSON from no `Origin`, Strata's own page or a trusted origin - also when an API key is set.

**Current limits (v1):** one request at a time; requests with images are not cached; images only when set up with them
(below); no video. **Temperature / top_p / top_k / min_p / seed** are honored per request (OpenAI and Anthropic fields),
and so are stop strings (OpenAI `stop`, a string or up to 4; Anthropic `stop_sequences`): the answer ends before the
first one, which is not sent, and the engine stops there; with the default adaptive expert tiers a sampled result is not
reproducible run to run - for seed-reproducible output add `--adapt-every 0` to the engine arguments. The run config's
optional `sampling` block sets the defaults for requests that leave the fields out (`"sampling": {"temperature": 1.0,
"top_p": 0.95, "top_k": 20}`); a request's own fields always win, and with no block at all a request without sampling
keys decodes greedy. The penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty`, with `penalty_last_n`
capping how many recent tokens they count over, default 64 when any penalty is set; llama.cpp's names `repeat_penalty`
and `repeat_last_n` work too) ride the same path; they count the tokens the request has consumed, so a repetition
penalty suppresses what the model itself just said, not the prompt alone, and they apply to every token the speculative
decoding checks at once, exactly as if it decoded one token at a time. `top_k` keeps at most 64 candidates: `0` ("off")
or anything above 64 uses all 64.

**Batch MTP on a layer split (`--batch-mtp`).** Our measurement on 2x R9700 with all experts in VRAM: 80.2 -> 62.7 tok/s (-21.9%, 0/5 pairs faster) against the pipelined default, so it pays only when the experts do not all fit in VRAM (docs/BATCHING.md).

**Token biases (opt-in).** The chat API supports request-scoped `logit_bias` objects and token/bias pair lists on supporting CUDA/HIP engines without continuous batching. A value of -100 (or `false` in a pair list) excludes a token. Unsupported engines return an error. See [LOGIT_BIAS.md](LOGIT_BIAS.md) for formats, limits and validation.

**Sampled drafting (opt-in, 0.1.40.2).** With temperature above 0 the engine keeps a draft only when it equals the token the model sampled for that position, and the draft layer proposes its best guess. Two opt-in switches change how the draft layer drafts a sampled request; greedy requests (temperature 0) are never touched and stay byte-identical. `STRATA_SPEC_COUPLED=1` (or `--coupled-draft`) drafts by sampling with the request's own settings and the random number the checking row will use; the text for a seed does not change. `STRATA_SPEC_PROB=1` is speculative rejection sampling: the draft layer samples its guess from its own distribution q, the check accepts it with probability min(1, p/q) (p is the model's distribution after penalties, top_k, top_p, min_p and temperature) and otherwise samples the replacement from the leftover distribution, so every token is distributed exactly as without drafts (the text for a seed then depends on the drafts, so it is reproducible only for the same engine, flags and prompt). Guesses without a distribution (prompt lookup, suffix drafts) keep the exact-match rule, which is already exact. Neither is faster in a way that holds up: on an RTX 3060 (IQ3_XXS, 200-token story and code answers, 10 interleaved pairs) the default's median output speed was 42.4 / 43.2 / 43.3 tok/s at temperature 0.3 / 0.7 / 1.0, coupled drafting 40.6 / 42.4 / 43.3 and rejection sampling 41.0 / 42.6 / 42.6, with run-to-run differences of up to 5% between repeats; drafts kept per round fell at 0.3 and rose by about 5 points at 1.0, which the cost of the longer sampled windows ate. `STRATA_SPEC_DEPTH=1` prints the acceptance by draft depth for each request; `STRATA_SPEC_MIN_TEMP=0.9` limits sampled drafting to hotter requests. Proof that the rejection rule keeps the distribution: `spec_prob_test` (chi-square, 2 million trials per case) and `spec_verify_parity` (GPU against the host reference).

---

### The Responses API and Codex CLI

`POST /v1/responses` (0.1.39, #451) speaks OpenAI's newer Responses API, which Codex CLI uses (it no longer
speaks Chat Completions). It runs on the same path as `/v1/chat/completions`, so the thinking levels, the thinking
budget, the conversation cache and the same API key, Host and Origin checks apply.

It is **stateless**: nothing is stored, so the client sends the whole conversation in `input` every time (Codex does,
with `store: false`). `previous_response_id`, `conversation`, `background` and the retrieve/delete/cancel endpoints
are refused with an error that says so.

| Request | What Strata does |
| --- | --- |
| `input` as a string, or as items | `message` items (`user`, `assistant`, `system`, `developer`; text and images), `reasoning`, `function_call`, `function_call_output`, `custom_tool_call(_output)` |
| `instructions` | The system message (with leading `developer` messages; later ones become user messages, as on the chat path) |
| `tools` | `function` tools, `namespace` tools (the model sees `namespace.name`; calls come back with `namespace` and `name`), `custom` tools (one free-form `input` string). Hosted tools (`web_search`, `file_search`, ...) are left out: the model cannot run them |
| `tool_choice` | `"none"` hides the tools; anything else lets the model choose (it cannot be forced) |
| `reasoning.effort` | `none`/`minimal`, `low`, `medium`, `high`/`xhigh`; without it the model's default (high) |
| `max_output_tokens` | The output cap (thinking included). Running out ends the response `incomplete` (`max_output_tokens`) |
| `text.format` | `json_schema` and `json_object` use the [JSON response formats](#json-response-formats) (checked, not constrained) |
| `thread_source` `thread_title` in `client_metadata["x-codex-turn-metadata"]` | Codex's thread-title request (opt-in: `"codex_thread_titles": true` in `strata-<model>.json`, off by default), sent beside each user turn from another session with `tools: []`: answered without the engine as `{"title": "<the user's line, cut to 36 characters>"}`, so the conversation keeps the prompt cache. The title is the user's line, not one the model wrote |
| `temperature`, `top_p`, `reasoning_budget_tokens`, ... | As on the chat path |

The model's thinking comes back as a `reasoning` output item with `reasoning_text` content (streamed as
`response.reasoning_text.delta`). This model writes no separate summaries, so `summary` is empty. With
`"include": ["reasoning.encrypted_content"]` the item also carries `encrypted_content`: an opaque string (base64, not
encrypted; the client already holds the text). Send the reasoning items back with the rest of the conversation, as
Codex does: their thinking goes back into the prompt, so it matches what the model wrote and the conversation cache
is reused.

With `"stream": true` the events are the official ones, in order: `response.created`, `response.in_progress`, then
for each output item `response.output_item.added`, its deltas (`response.reasoning_text.delta`,
`response.output_text.delta`, `response.function_call_arguments.delta`), its `...done` events and
`response.output_item.done`, and at the end `response.completed`, `response.incomplete` or `response.failed`.
While a long prompt is read, a `response.in_progress` event goes out every 15 s, which Codex counts as activity.
Errors before the answer starts have the Responses form (`{"error": {"message", "type", "param", "code"}}`); after
it started they arrive as `response.failed`.

**Codex CLI.** In `~/.codex/config.toml` (Windows: `%USERPROFILE%\.codex\config.toml`):

```toml
model = "strata"                        # any name; Strata answers with its model
model_provider = "strata"
model_context_window = 32768            # the context you chose in setup: Codex compacts before it gets there
show_raw_agent_reasoning = true         # show the model's thinking (it writes no summaries)
# model_reasoning_effort = "medium"     # none, low, medium or high; default: the model's (high)

[model_providers.strata]
name = "Strata (local)"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # a first, long prompt can take minutes to read
# env_key = "STRATA_API_KEY"            # only if the server has an api_key: the variable holding it
```

Then run `codex` (or `codex exec "..."`) as usual. Codex warns `Model metadata for ... not found` for a local model
name; that is expected. Measured with Codex CLI 0.160.0 and Q2_0 on an RTX 5070: its first prompt (instructions and
tool descriptions) was 9,443 tokens, read in 10 s; in a tool loop, each later turn reused about 96% of the prompt from
the cache and read only the new part in 1-2 s. On Windows, Codex's sandbox rejected every shell command in that test
until it was started with `-c 'windows.sandbox="unelevated"'` (a Codex setting, not Strata's).

**Codex's compaction (opt-in: `"codex_compaction_cache": true` in `strata-<model>.json`).** When the context fills up (or
on `/compact`), Codex sends the conversation once more with a request to summarize it, and with `tools: []`. The template
writes the tools at the top of the prompt, so that prompt would share only its first few tokens with the conversation the
engine holds and be read again from the start, at its longest. With the option on, Strata keeps which Codex conversation
sent the last prompt other than a compaction (`session_id` and `thread_id` from `client_metadata["x-codex-turn-metadata"]`)
and the tools that prompt was rendered with; a request Codex marks `"request_kind": "compaction"`, from that same
conversation and without tools of its own, is rendered with them. Only the prompt: for the output parser and in the
response the request's tools stay what Codex sent, none, so a tool call the model writes comes back as its text, as for
any request without tools. It is one entry, replaced by each such prompt; a compaction of another
conversation, a request without that metadata (Codex before 0.140) or a restart renders the request as sent, and so does
one that would not fit the context with the kept tools. Off by default: the prompt of a compaction differs from the
one the client sent. The author measured, with Codex CLI 0.160.0 and an engine that only counts the shared prompt start,
that a compaction after an 85,000-token conversation reused 84,895 of its 84,997 tokens and read 102; without it, 41 of
80,683.

## Tools from MCP servers

The chat page can give the model tools from [MCP](https://modelcontextprotocol.io) servers, as LM Studio and Claude
Desktop do: reading your files, fetching web pages, searching, anything an MCP server offers. List the servers in
`strata-<model>.json` under `"mcp_servers"` - the same shape as Claude Desktop's `mcpServers` block, which you can
also paste as it is (key `"mcpServers"`):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me\\Documents\\notes"]},
  "search": {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8}
```

Or keep them in their own file and start the server with `--mcp-config path\to\claude_desktop_config.json` (a file
with an `mcpServers` block; add it to the `serve/server.py` line of your run script). Restart Strata after a change.

- **A program** (`command`, `args`, optional `env` and `cwd`) is started by Strata and spoken to over its
  stdin/stdout; `npx`, `uvx`, `python` and friends are found on `PATH` as usual (Node.js is needed for `npx`
  servers). **An address** (`url`, optional `headers`) uses MCP's Streamable HTTP transport (the older SSE-only
  transport is not supported). `"disabled": true` leaves an entry out.
- The servers start with Strata, in the background; the server window says what each one offers
  (`MCP server 'files': 14 tools (...)`), or why it did not start - its tools are then left out and the chat works
  without them. The Monitor tab lists them, and the Sampling drawer has **Use tools from MCP servers** (on by
  default). A server that stops later is started again at its next call.
- In the chat each call shows as a small block (tool, arguments, result); the model reads the result and goes on,
  up to `max_rounds` calls in a row per answer. A tool that fails or takes longer than `timeout_s` (default 60 s)
  gives the model an `error: ...` result instead of ending the chat. Results longer than `max_result_chars`
  (default 20,000 characters) are cut, with a note, before the model reads them. Stop stops a running tool too.
- Only the chat page uses them. API clients (omp, Claude Code, OpenAI and Anthropic SDKs) see the API exactly as
  before and keep their own tools; a request to `/v1/chat/completions` opts in with `"strata_mcp": true` (it then
  gets `strata_mcp` tool events in the stream).

**Security.** MCP tools run on your PC with your user's rights, and **the model decides when to call them** - also
because of what it reads (a web page or a file can contain instructions). Give a filesystem server only the folders
it needs, prefer read-only tools, and don't add servers you don't trust. The tools can only be used from the chat
page itself (a request with another site's Origin or without a JSON content type is refused); if Strata is reachable
from other devices, set an API key.

---

## Manage Strata from your AI assistant (MCP server)

`tools/strata_mcp.py` is an MCP server for Claude Code, Claude Desktop, Cursor, VS Code, Codex and other assistants.
Once it is added, you can ask your assistant "install Strata for this PC", "start Strata" or "is Strata running?".
In Claude Code, add it with:

```bash
claude mcp add strata -- python C:\Users\you\Strata\tools\strata_mcp.py
```

It has eight tools: status (the running model, what is installed, the hardware, a recommended size), the model
list, install, start, stop, logs, a speed test, and connection settings for other apps.

Install runs `setup.py` with `--yes` in the background. Before it downloads anything, it shows the plan and waits
for your OK. Start and stop work like the run scripts and the server's own unload. The MCP server only ends
processes it started itself. It uses only Python's standard library, so it works before `.venv` exists.

The config snippets for every client, the tool arguments and the safety rules are in
[docs/MCP_SERVER.md](MCP_SERVER.md). This is the opposite direction from
[Tools from MCP servers](#tools-from-mcp-servers) above, where the Strata model calls *your* MCP tools.

---

## Images (vision)

The model has a vision encoder: [`mmproj-Qwen3.8-Flash-Next-BF16.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(0.9 GB, a 27-layer ViT plus the projector into the language model). It is **optional**: say yes when the setup asks
"Images?", or run it again with `--vision gpu` (or `--vision cpu`). The setup downloads the encoder, builds a small
helper (`strata-vision`, from llama.cpp's `mtmd` library) and adds it to your start script. Nothing else changes.

| Encoder on | Time per picture | Cost |
| --- | --- | --- |
| **GPU** (recommended) | **0.1-0.5 s** (up to 1,024 image tokens) | ~1.4 GB of VRAM is kept free for it, so the expert cache is smaller: text output is a few % slower (table below) |
| CPU | about 3 s at 300 image tokens on 8 cores (6-13 s on 4 threads); more tokens take longer, in proportion (#767, #625) | nothing on the GPU |

A picture becomes up to 1,024 tokens of the context (a 640x480 photo: 300). The same picture sent again, as chat apps
do on every turn, is encoded only once.

**More image tokens (0.1.39, #625):** `--vision-tokens N` at setup (`START-HERE.bat --setup --vision cpu
--vision-tokens 768`) sets the most tokens a picture becomes - `"max_tokens"` in the `"vision"` section of
`strata-<model>.json`, which you can also edit by hand. More tokens keep more detail (small text, charts, screenshots)
and take longer to encode, on the CPU most of all; a setup run again keeps the value.

**A minimum number of image tokens (0.1.40, #767):** `"min_tokens": N` in the `"vision"` section of
`strata-<model>.json` (edit it by hand; a setup run again keeps it) is passed to the encoder as `--min-tokens N`
(mtmd's `image_min_tokens`): a small picture is scaled up to at least N tokens. llama.cpp's mtmd prints that Qwen-VL
models want at least 1,024 for grounding tasks (pointing, counting small items); the CPU default stays at 300 at most
and no minimum, because a larger minimum changes the image answers and costs encode time (about 3 s at 300 tokens on
8 cores, in proportion to the tokens).

**A Q8_0 encoder (#625):** `"mmproj"` in the `"vision"` section can point to another mmproj file of this model, for
example a Q8_0 one (llama.cpp's `convert_hf_to_gguf.py --mmproj --outtype q8_0` makes one): the encoder's library
reads quantized weights, the file is smaller (590 MiB against 865 MiB for the BF16 one, from `llama-quantize
mmproj-Qwen3.8-Flash-Next-BF16.gguf mmproj-Qwen3.8-Flash-Next-Q8_0.gguf Q8_0`), and on the CPU it uses less RAM
(about 280 MiB less in the encoder) and can encode faster than BF16 at the default 300 tokens. Setup downloads the
BF16 file, and a setup run again keeps a file of your own that still exists. **Recommended for `--vision cpu`.**
Accuracy against BF16, per image token: the cosine of the embeddings is 0.997-0.999 on average (#625's report, with
a community Q8_0 file, at 300, 768 and 1,024 tokens), and 0.9988 and 0.9987 on two pictures with the file made by
the command above (300 tokens; the worst single token 0.94-0.96). Encode time above 768 tokens is the same as BF16's
within about 3%. Numbers on more pictures (charts, small text) are welcome in #625.

**Flash attention on the CPU (#660):** ggml's fast CPU kernel for flash attention needs the encoder's head size (72)
to be a multiple of the vector width: 8 floats with AVX2, which the release builds use, but 16 with AVX-512, which a
build from source gets on a CPU that has it (Zen 4 and newer, some Intel). There ggml falls back to a kernel that is
several times slower and accumulates in FP16, so `strata-vision` turns flash attention off on the CPU when its ggml has
AVX-512 and leaves it at `auto` (on) otherwise; `--flash-attn on|off|auto` overrides that. A 1024x1024 picture
(1,024 image tokens) on a Ryzen 7 7700X, 8 threads, an F16 mmproj, one encode per fresh process:

| Build | Flash attention | Encode | Peak RSS | Output vs FP32 attention |
| --- | --- | --- | --- | --- |
| AVX2 (as released) | on (default) | 8.3 / 9.9 s | 1,152 MiB | 1.2% |
| AVX2 | off | 15.2 / 14.0 s | 2,194 MiB | 1.1% |
| AVX-512 (from source) | on | 43.5 s | 1,152 MiB | 26% |
| AVX-512 (from source) | off (default) | 13.3 s | 2,195 MiB | - |

The last column is the relative L2 distance of the embeddings from the AVX-512 run without flash attention; two builds
differ by about 1% anyway.

**A spare GPU for the encoder (0.1.33, #408):** with a card the engine doesn't use, add `"cuda_device": 2` (numbered
like `nvidia-smi`) to the `"vision"` section of `strata-<model>.json`: the encoder then runs on that card alone. Lower
`--vram-reserve-mib` in `"args"` to 700 as well, so the engine's cards keep that VRAM for the expert cache. The
encoder's card needs code in the ready-made encoder (RTX 20/30/40/50).

### Sending a picture

**Terminal chat:** type `/image <path to a picture>`, press Enter, then type your question.

```
you> /image C:\Users\me\Pictures\receipt.jpg
(picture attached: receipt.jpg - now type your question)
you> What is the total on this receipt?
```

**OpenAI API** (an `image_url` part: a `data:` URL, an `http(s)://` URL or a local file path):

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
img = base64.b64encode(open("photo.jpg", "rb").read()).decode()
r = client.chat.completions.create(model="strata", messages=[{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
    {"type": "text", "text": "What is in this picture?"}]}])
print(r.choices[0].message.content)
```

**Anthropic API:** an `image` block with a `base64` (or `url`) source, as usual.

JPEG, PNG, BMP, GIF, WebP, TIFF and AVIF work (the last ones are converted to PNG first; agents such as omp send
WebP). Chat apps with image upload work the same way.

### Speed with images on (4K context, measured)

| Model | Prompt tok/s, images off | Prompt tok/s, images on | Output tok/s, images off | Output tok/s, images on |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 541 | 531 | 90.7 | 86.9 |
| **IQ2_XS** | 468 | 458 | 77.1 | 74.0 |
| **IQ3_XXS** | 411 | 401 | 64.7 | 59.5 |

"Off" is the published setup; "on" runs with the encoder loaded on the GPU and its VRAM kept free, so ~1,000 fewer
experts fit in VRAM and the CPU computes a few more per token: 2-8% slower. For text, turning images on changes
nothing else (the output is bit-identical when the VRAM is the same).

A question about a picture (a 640x480 newspaper page = 300 image tokens, a 328-token prompt; the whole request, encoder
on the GPU, measured through the API):

| Model | Answer ("MEN WALK ON MOON") | Picture encoded | Prompt | Output |
| --- | ---: | ---: | ---: | ---: |
| **Q2_0** | 1.9 s | ~0.1 s | 207 tok/s | 65-73 tok/s |
| **IQ2_XS** | 2.3 s | ~0.1 s | 173 tok/s | 50-60 tok/s |
| **IQ3_XXS** | 2.7 s | ~0.1 s | 144 tok/s | 41-50 tok/s |

(Short prompts run below the 4K prompt speed: a 2,048-token chunk is where the prompt path is efficient. Short answers
run below the long-output speed: the first rounds have no draft yet.)

**How it works inside:** the encoder turns the picture into rows of the same width as the model's word embeddings;
Strata puts them where the prompt has `<|image_pad|>` tokens and gives each one its 2-D position (row and column in the
picture; the model uses interleaved M-RoPE). Answers match llama.cpp's multimodal implementation token for token on our
test images.

---

**Stager waits (0.1.40.2, `STRATA_STAGER_SLEEP`):** the prompt-staging threads sleep while they wait on Linux and spin on Windows, because spinning is a little faster there (a 5070 read an 8K Q2_0 prompt 1.2% faster). `STRATA_STAGER_SLEEP=1` makes Windows sleep too, which is the choice when sharing the machine matters more than speed: on a Ryzen 9 7940HS laptop with an RTX 4070 (IQ3_S, 64K context) sleeping waits read 5,914 and 22,305 token prompts 5-6% slower while the whole-machine CPU use fell from 77-90% to 21-25% (issue #1101, thanks to midhatn). `STRATA_STAGER_SLEEP=0` forces spinning on Linux. The output is the same either way.

**Stager threads for a GGUF read in place (0.1.41, #1353):** a native pack read from its GGUF shards in place (UD-Q4_K_XL, a Q8_0 pack) copies each expert with 32 threads and a 128-deep ring when the bytes are page faults on an SSD. If the shards' pages are already in the page cache (a big-RAM box, a warm start) those copies are memory copies and the same profile ran 4-7x slower than the 4-thread default (a 124K-token Q8 prompt: 57 -> 350-410 tok/s with the default). The engine now checks a sample of the experts with `mincore` at the first prompt (Linux) and takes the RAM profile when 90% or more of their pages are resident, saying so in its log; a cold or partly cached GGUF keeps the SSD profile. `STRATA_STAGER_SSD=1` / `0` forces either profile; `STRATA_STAGER_THREADS` still wins. The bytes copied are the same.

## Short prompts: the CPU shares the experts (on by default on one NVIDIA GPU, `STRATA_PREFILL_CPU_SHARE`)

A prompt chunk of a few thousand tokens (an agent's tool result, a test's output, a short follow-up) streams every routed
expert that is not in VRAM over PCIe, while the CPU pool that decodes sits idle and RAM already holds those experts.
`STRATA_PREFILL_CPU_SHARE=auto` hands the pool the experts few of the chunk's tokens route to, measures per layer how
long each side takes and gives the CPU the share at which both end together (`STRATA_PREFILL_CPU_SHARE=0.4` fixes a
share). `auto` also checks that sharing pays: it times layers with the share and without it (the first ones alternate until three comparisons are in, then one in 29 runs the other way) and shares only while the layers that share are the faster ones. Where sharing is slower (every share was, on an RX 7900 GRE with a Ryzen 7 5700X3D under ROCm, #1282), the experts stay on the GPU apart from those measuring layers. It is off unless you set it, and then the output is byte-identical to the build without it.

**0.1.41: on by default** on CUDA builds with one GPU and no `--batch` slots, for prompt chunks below 1,024 tokens: unset behaves as `auto` with `STRATA_PREFILL_CPU_SHARE_MAX=1024`, and the engine prints one line at start saying so. Measured (interleaved whole-engine runs, 8 rounds, medians): 512 tokens -28..-34% and 1,000 tokens -20..-26% prompt time on an RTX 3060, a Tesla P100 and an RTX 5070, 8/8 pairs each; mean KL against the share off 0.004 (max 0.025). The answers can differ slightly from 0.1.40.3. **`STRATA_PREFILL_CPU_SHARE=0` turns it off and gives 0.1.40.3's exact bytes.** The layer split, batch slots and AMD (HIP) builds keep it off unless you set the variable; `STRATA_PREFILL_CPU_SHARE_MAX=3072` (the extension to larger chunks) also stays opt-in: it lost 2% at 2,048 tokens on the 5070.

With it on, chunks below 3,072 tokens are staged after their routing (only those can hand the CPU a share) instead of
streaming every expert from 1,024 tokens on; `STRATA_PREFILL_CPU_SHARE_MAX=1024` keeps the old limit. The CPU takes
experts it reads from RAM as they are: the arena, the page-locked copy of the resident RAM mode, or the mapped
`experts.bin`'s pages in the file cache (`--mmap-experts`), not only page-locked ones. Serve without `--batch`; on a
layer split every stage has the pool and one stage at a time takes it for a chunk.

**0.1.42: not armed by default when memory is short.** A 0.1.41 report (RTX 5090, RAM already tight) had chats failing and the engine restarting until `STRATA_PREFILL_CPU_SHARE=0` was set. The default is now skipped, with one line in the log saying why, when the RAM this process can get (the cgroup limit counts on Linux) is below the share's own buffers plus 3 GiB, when on Windows the commit limit (RAM + page file) has less than 4 GiB left beyond them, or when the experts it would read are file pages (the mapped `experts.bin` outside the GPU cache and the RAM copy) that the free RAM plus 3 GiB cannot keep. An explicit `STRATA_PREFILL_CPU_SHARE=auto` (or a number) is kept and only warns. The share's two page-locked buffers (about 50 MB) are now taken once, at their largest, and a failed allocation ends the share for that run instead of failing the request.

When on, the CPU's rows are computed in the CPU's own activation format, so the output changes in the last bits (first
token KL against off: mean 0.006, max 0.026 nats over 22 prompts; about half of the 32-token greedy answers on 500 and
1,000-token prompts are identical, the rest part at a near tie after about 23 tokens). Chunks of 3,072 tokens and more
read the same either way. `auto` picks the share from wall-clock timings, so the same request can be split
differently - and answer differently - from one run to the next (a resumed conversation's re-read differed in 31 of
1,536 teacher-forced argmax tokens, #1684); a fixed share repeats, `0` gives the exact bytes of a run without it.

Prompt time, medians of 10 interleaved pairs (off / auto, ms, `--expert-cache 1500`):

| machine | 512 tokens | 1,000 tokens | 2K / 4K / 16K |
|---|---|---|---|
| RTX 5070, Ryzen 5 7600, Q2_0 | 1,376 / 1,019 (-26%) | 1,788 / 1,392 (-22%) | unchanged |
| RTX 3060, Core Ultra 7 265, IQ3_XXS | 1,620 / 1,159 (-28%) | 2,196 / 1,770 (-19%) | unchanged |
| Tesla P100, Xeon E5-2690 v4, IQ3_XXS | 4,805 / 3,126 (-35%) | 6,821 / 5,085 (-25%) | unchanged |

Up to 3,072 tokens and with mapped experts, fresh prompts read after an 8K prewarm, medians of 5 interleaved rounds
(off / auto, ms; Ryzen 9 5900XT, 96 GB DDR4-3200, Windows 11, `--mmap-experts`, q4_0 KV; before this the share took no
expert here: none was page-locked):

| machine | 512 tokens | 1,000 tokens | 2,000 tokens | 3,000 tokens |
|---|---|---|---|---|
| RTX 5070 Ti (PCIe 3.0 x8), Swift 1.5 IQ3_XXS (huihui-ai's build) | 3,322 / 1,706 (-49%) | 3,990 / 2,357 (-41%) | 5,413 / 3,354 (-38%) | 5,525 / 4,309 (-22%) |
| RTX 3060 + RTX 5070 Ti (layers 0-11 / 12-47, both PCIe 3.0 x8), IQ3_S | 3,452 / 2,037 (-41%) | 4,506 / 2,680 (-41%) | 6,376 / 3,869 (-39%) | 6,733 / 5,138 (-24%) |

A coding agent's recorded conversation on the two cards (a 100K-token start, then 8 turns of 1-5K tokens of code, 128
tokens written a turn, the same tokens read by both): 137.4 s off, 130.6 s auto (reading 121.1 -> 114.3 s).

## Missed experts in a verify window: the tail is skipped (on by default on CUDA, `STRATA_ROUTE_TAIL_SKIP`)

When a card cannot hold every routed expert, most of a decode token's waiting time is spent on the experts that are not in VRAM:
their copy over PCIe or their rows on the CPU pool. In a verify window of several tokens, an expert that **every** token of the window
routes at the bottom of its top 10 (rank 7 or lower, the smallest weights) adds little to any of them. **0.1.42: on by default on CUDA builds**:
such a missed expert is neither copied nor computed, its contribution is zero, and the other experts keep the weight the router gave them
(no renormalisation). An expert that any token ranks above 7, and every expert already in VRAM, is served as before.

**`STRATA_ROUTE_TAIL_SKIP=0` turns it off** and restores the answers of a build without it byte for byte (checked: `mg_norepeat` 10 of 10 and `exact_pp` 4096 and 20000
tokens identical to 0.1.42 without the change, on the RTX 3060, the RTX A4000 alone and as a 2-GPU split, and the Tesla P100). Another rank
(`STRATA_ROUTE_TAIL_SKIP=8`, `=9`; 1-9 are accepted) is taken as given. The start-up line
`STRATA_ROUTE_TAIL_SKIP=7 is ON (default)` names the opt-out.
Since 0.1.43 the same default holds on HIP for gfx1200 / gfx1201 (RDNA4), under the same rules and with the same `STRATA_ROUTE_TAIL_SKIP=0` opt-out; other AMD parts are unchanged.

Where it applies: serve on a CUDA build with one GPU or a layer split (every stage skips by itself). Not applied, and no line printed: when every expert is
in VRAM (nothing is missed), with `--batch` / `parallel` slots (a window then mixes requests, and one request's skip would depend on another's
tokens: kept off until measured), with a peer-expert tier or a helper GPU, on HIP and SYCL builds, and (0.1.42.1, #1884) when 90% or more
of all the experts are in the GPU cache: the few misses are cheap there and the skip costs more draft acceptance than it saves (a Tesla V100 32 GB
with Coder IQ1_M, 94.8% of the experts cached, 99.9% decode hits: draft acceptance 70.1% to 68.2%, decode 1-2% slower than 0.1.41). A start-up line
says so; `STRATA_ROUTE_TAIL_SKIP=7` turns it on anyway, and `STRATA_TAIL_SKIP_MAX_CACHED=<percent>` moves the line (101 = never). On an AMD layer split the line is 60% (2x R9700 at 16 GB-equivalent: Q2_0 with 73% cached lost 6.1% story decode with the skip, IQ3_S with 45% cached gained 7-10%). The request log gets
`route tail skip: N missed experts skipped` (cumulative) while it is active.

**Answers differ slightly from 0.1.41** (`STRATA_ROUTE_TAIL_SKIP=0` brings them back). Teacher-forced KL of the decode path (the default's own greedy text and the default sampled at T=0.8, 28 chat prompts of code,
reasoning, chat, six languages and long-document summaries, 220 tokens each, about 6,000 scored tokens per set, through the real verify windows with the MTP drafts;
`phaseA-tests/cuda-route-kl.md`):

| | RTX 3060 12 GB, IQ3_XXS | RTX A4000 16 GB, Q2_0 |
|---|---|---|
| KL(0.1.41 behaviour \|\| tail skip), mean, greedy / sampled text | 0.0058 / 0.0056 | 0.0032 / 0.0030 |
| p99, max (greedy) | 0.061, 0.29 | 0.035, 0.32 |
| top-1 token agreement | 97.7% / 97.5% | 98.2% / 98.2% |
| perplexity of the forced text, skip / no skip | x1.011 / x1.007 | x1.005 / x1.003 |
| KL of the first token after the prompt | 0.024 / 0.033 | 0.015 / 0.016 |
| the same engine run twice (the floor; adaptive tier and CPU/GPU split depend on timing) | 0.0006-0.0007 | 0.0007 |
| `STRATA_PREFILL_CPU_SHARE` on / off (already the default) | 0.0006-0.0007 (first token 0.002-0.0035) | 0.0005-0.0007 (first token 0.002-0.004) |
| Q2_0 against IQ3_XXS, the same text | - | 0.1006 (p99 1.08, top-1 90.6%) |

The effect is not even: code and reasoning prompts have a KL of 0.002-0.004, chat and non-English prompts 0.004-0.009.
Task check, greedy, thinking off (HumanEval every second problem, 82; 30 generated reasoning tasks), default / tail skip:
RTX 3060 76 and 78 of 82 / 78 of 82, reasoning 27 and 27 / 29; RTX A4000 76 and 76 / 76, reasoning 27 and 30 / 27. The runs do not separate the settings
(one run each; a change of 3 reasoning tasks is noise).

Decode speed, `STRATA_ROUTE_TAIL_SKIP` default against `=0`, 5 interleaved rounds on fresh engines, a story, a code answer and a 6K-token document, 200 tokens each,
median over the rounds (ratio of tokens/s, default over `=0`; rounds in which the default was faster):

| machine | story | code | 6K document | all three |
|---|---|---|---|---|
| RTX 3060 12 GB, IQ3_XXS, resident experts | 1.104 (5/5) | 1.197 (5/5) | 1.241 (5/5) | **1.156 (5/5)** |
| RTX A4000 16 GB alone, Q2_0 | 1.151 (5/5) | 1.135 (5/5) | 1.101 (5/5) | **1.129 (5/5)** |
| Tesla P100 16 GB, IQ3_XXS, `--pcie-frac 0.29` | 1.149 (5/5) | 1.182 (5/5) | 1.118 (5/5) | **1.146 (5/5)** |
| two RTX A4000, layer split (hit rate 99.7%) | 0.987 (0/5) | 1.004 (5/5) | 1.002 (3/5) | 0.998 (1/5) |

The gain follows how much of a token is spent waiting for missed experts (hit rate 75-90% on the single cards in the profile, `phaseA-tests/cuda-profile.md`); a 2-GPU
split that holds nearly every expert (hit rate 99.7%) gains nothing. It does not touch prompt reading. Switches that stay opt-in and are not part of this:
`STRATA_ROUTE_PRIOR` (adds speed, doubles the KL), `STRATA_ROUTE_RESIDENT` and `STRATA_ROUTE_RESIDENT_MTP` (the latter is not faster than the former).

## The PCIe share of the missed experts is set from the CPU's speed (CUDA, single GPU, no `--pcie-frac`)

Of the experts missing from VRAM, a share goes over PCIe to the GPU and the rest is computed by the CPU pool; `--pcie-frac` sets the share.
Up to 0.1.41 the default came from the PCIe probe only (0.55 from 20 GB/s up), which ignores how fast the CPU is: with the route tail skip on, the
best share was 0.15-0.20 on an RTX 3060 (default 0.55), 0.20 on a Tesla P100 (default 0.29), and the RTX A4000 is flat from 0.15 to 0.40.

Since 0.1.42, on CUDA with one GPU and no `--pcie-frac`, the share starts at the old value and is refined in the first decode windows from what the engine
already times: the wall time of the CPU pool per missed expert. With `R` = expert bytes / that time (GB/s) and `L` = the probe's link speed, the share is
`L / (L + 1.85 R)`, rounded to 0.05 and kept between 0.10 and 0.60; it is changed at most three times (after 48, 96 and 96 more windows, about 10 s of
decoding), and only by a full step. Each change prints one `PCIe share:` line. A request that sets its own `pcie_frac` is left alone.
**`--pcie-frac N` always wins; `STRATA_PCIE_FRAC_DEFAULT=old` keeps the 0.1.41 rule.** Not used with a layer split, `--batch`, a peer GPU, when every
expert is in VRAM, or on SYCL or on HIP other than gfx1200 / gfx1201 (RDNA4, where the measured share is the default since 0.1.43; `STRATA_PCIE_FRAC_DEFAULT=old` opts out). The canonical Q2_0 pack uses it on CUDA since 0.1.43, on Linux by default and on Windows with `STRATA_PCIE_AUTO_CANON=1` (an RTX 5070 there: code +2.5%, a short answer -3.3%). The constant 1.85 is a fit to three machines, so it lands within a step of each machine's best, not on it. On HIP, `STRATA_PCIE_AUTO_SPLIT=1` (opt-in, 0.1.43) also refines a two-stage layer split: the pool speed is read once and each stage takes the share its own probed link gives (2x R9700 at 16 GB-eq, tail skip on: Q2_0 story +4.5%, code +9.9%, 4K +8.8%; IQ3_S story -2.1%, code +8.0%, 4K +23.3%; with three or more stages the pipelined loop does not run it).

Measured (interleaved pairs of whole engine runs, new default against `STRATA_PCIE_FRAC_DEFAULT=old`, medians, route tail skip on):

| card | the pool takes | share (old -> new) | decode, new / old | pairs faster |
|---|---|---|---|---|
| RTX 3060 12 GB, IQ3_XXS | 32-37 us per expert | 0.55 -> 0.20 | **1.300** | 5/5 |
| Tesla P100 16 GB, IQ3_XXS | 84-97 us | 0.29 -> 0.20-0.25 | 1.011 | 4/5 |
| RTX A4000 16 GB, Q2_0 | 97-115 us | 0.30 -> 0.30 (unchanged) | 0.999 (same setting) | - |

Explicit-share runs are byte-identical to 0.1.41 (`mg_norepeat` 10 of 10, `exact_pp` 4096 and 20000). With the new default the answers differ a little
from the old default, as any change of the share does (CPU and GPU round differently): teacher-forced KL against the old default 0.0015 on the 3060
(run-to-run floor 0.0013) and 0.0011 on the A4000 (floor 0.0012; at that time the share was 0.25).
The sweeps behind the constant: `phaseA-tests/notes-142-pciefrac.md`.

## More opt-in switches measured for 0.1.41 (all off unless you set them)

None of these changes the default answers; each was measured against the default with interleaved off/on pairs of whole
engine runs (a fresh engine per side, warmed, 200-token greedy answers on three fixed prompts, medians; "pairs faster" counts
the pairs in which the switched arm won). They are here so you can try them on your own card.

- **`STRATA_GDN_CHUNKED=1`: the DeltaNet recurrence of the prompt in chunks of 32 tokens** (sergiywith, #1372; WY form, FP32,
  other bits than the default: first-token KL against the default was 0.002-0.009 nats on average, the same top token in
  every prompt tried). The kernel needs 128 SMs in one wave, so by itself it runs only on a card with 128 or more
  (RTX 5090: 1.8x on the recurrence). On smaller cards `STRATA_GDN_CHUNKED=2` forces it for a measurement, and it is
  slower there: 0.73x on an RTX 3060 (28 SMs) and 0.87x on an RTX 5070 (48 SMs) at 2K-32K tokens (`gdn_rec_parity --bench`),
  and the prompt time did not improve: RTX 5070 2K 1.01x, 8K 1.00x of the default's time; RTX 3060 2K, 8K, 16K 1.01x, no pair faster.
- **`STRATA_FS_SLOTS=N`: the Foresight swap space** (q8atnight, #1348): N VRAM slots per layer, refilled on a copy stream
  with the experts that just missed and with the ones the next layer's router predicts (`STRATA_FS_AHEAD=0` for the
  misses only), so that an expert the cache lacks can be computed on the GPU instead of by the CPU. The model's own router
  decides everything; the slots only hold copies of the same bytes (`STRATA_FS_VERIFY=1` reads every landed slot back and
  compares it: 0 bad in 2,300+ reads on an RTX 5070). The slots are taken from the VRAM the expert cache would have
  (raise `--vram-reserve-mib` by about N x 80 MiB), and in every box measured the smaller cache cost more than the slots
  returned: decode 4 slots, median tok/s B/A (pairs faster): RTX 3060 0.95 (0/6) with misses only, 0.91 (0/6) with the
  look-ahead; Tesla P100 0.97 (2/6) and 1.00 (4/6); 16 slots on the P100 0.96 (0/6); RTX 5070 0.98 (1/6). The reporter
  measured the same on 2x RTX 3090 (the misses per layer are below one there, so the CPU round trip stays). Left off.
- **`STRATA_STAGE_PIN`: pinned stage buffers** (Zhong Uncle, #1237; **opt-in, `STRATA_STAGE_PIN=1`**. It was on by default in
  development, but the 0.1.41 release gate found IQ3_S decode corrupted after a 4096-token prompt with it on, likely a stage
  buffer recycled before its asynchronous copy finished; it stays off until that is fixed). Where the experts are served from the GGUF in
  place (`--mmap-experts`, or too little RAM for the arena) the buffers the cache fill copies from are page-locked, so the
  copy no longer goes through the driver's bounce buffer (about 0.4-0.6 GiB of pinned RAM, one buffer falls back to
  pageable when the driver refuses). Decode, median tok/s on, off, 6 pairs of whole runs: Tesla P100 with the RAM capped at
  24 GB (cgroup) 13.9 -> 15.0 (+8%, 6/6 pairs faster); RTX 5070 `--mmap-experts` +1.2% (5/6); RTX 3060 `--mmap-experts`
  -0.1% (5/6). Boxes that hold all experts in RAM never use the stage buffers.
- **`STRATA_ADAPT_LAG=2`: the adaptive tier's copies are waited for one window later** (#764). Decode, 6 pairs: Tesla P100
  (PCIe 3.0 x16) +3.5% (6/6 pairs faster), RTX 5070 +0.2% (4/6), RTX 3060 -1.3% (0/6), so it stays opt-in.
- **`STRATA_ADAPT_OVERLAP` (`=1` turns it on, on every backend; the startup log names the state and, when it cannot run, why): the adaptive tier's copies land while the next
  window runs, with the same tokens.** The window takes the swaps at the same moment as with #463's wait (so the same experts run
  on the GPU and the CPU in every window), but the copies go in layer order and the verify window holds a layer's plan back
  until that layer's copies have landed, instead of the whole window waiting for the round (about 10 ms every fourth window on
  an A4000, PCIe 3.0 x16). One GPU, the whole arena in RAM, no `--adapt-async`, `--pipeline-windows`, `STRATA_ADAPT_LAG`,
  `STRATA_ADAPT_NOWAIT` or `STRATA_DMA_BATCH`. On by default on CUDA on Linux since 0.1.43, opt-in on Windows and HIP;
  opt-in on every backend since #2010, after an RTX 5090 Laptop lost 21% decode with it. What it is worth per box: an RTX A4000 (6
  interleaved pairs each, decode median tok/s: Q2_0 16 GB 62.5 -> 66.2 (+5.9%), Q2_0 as a 12 GB card 51.3 -> 54.2 (+5.7%),
  IQ3_XXS 16 GB 48.6 -> 52.6 (+8.3%), IQ3_XXS as 12 GB 39.3 -> 41.5 (+5.5%), 6/6 pairs faster every time) and a Tesla P100
  (+4.4%) gain, but an RTX 5070 on Windows loses 2.6% on Q2_0 (IQ3_XXS no faster) and an RTX 5090 Laptop (WSL2, Core Ultra 9
  275HX, native IQ3_XXS pack, `--expert-cache-per-layer`) loses 21%: 84.6 vs 106.6 tok/s medians of 5 generations of 256
  tokens from one build, the slowest of the fast run above the fastest of the slow one (#2010).
- **`STRATA_ADAPT_CALLBACK_WARMUP` (Linux, on by default since 0.1.43.1; =0 turns it off): start the GPU host-callback thread before the CPU
  pool pins the host thread.** A callback thread created later can inherit the host's one-CPU affinity and compete
  with the verifier spinning for its signal (#1975). The warm-up runs and synchronizes one no-op host callback before
  constructing the pool; swap selection, per-layer callbacks and residency updates stay unchanged. On an RTX 5090 /
  Ryzen 9 9950X, a paired 4K three-prompt test averaged 142.82 -> 207.75 tok/s. Other GPUs and drivers remain untested;
  see the [reproducer, settings and limitations](../bench/results/2026-10-11-rtx-5090-callback-affinity/README.md).
- **`STRATA_ADAPT_SIGNAL=host`: the overlap's per-layer "landed" signal as a host function** (the 0.1.43 way). From 0.1.43.1 the default is a
  4-byte stream-ordered copy into pinned memory, which needs no driver callback thread: on a 20-core desktop (RTX 3060) the host
  functions still cost ~7 ms per window with the thread started early, because the pool's workers occupy every other CPU.

## Thread placement: `--aux-cpus` (Linux, off by default)

The CPU pool pins its workers (one logical CPU per physical core) and the host thread pins itself to the pool's reserved
core. A thread the engine starts after that inherits the host's one CPU: the adaptive tier's job thread, the prefill
helpers, the router look-ahead. CUDA driver threads created before pinning can run on any allowed CPU, the workers'
included; a lazily created host-callback thread can instead inherit the pinned host's one CPU (see the warm-up above).
`--aux-cpus` puts every thread that is neither a pool worker nor the host thread on spare CPUs. It moves threads only and
never changes a result.

- **Which CPUs.** `--aux-cpus auto` (or `STRATA_AUX_CPUS=auto`): the SMT siblings of the host's core first, then the CPUs of
  physical cores that have no pool worker. A CPU that shares a core with a worker is never taken; when no CPU is spare
  the feature does nothing and the start-up line says so. A list (`--aux-cpus 24,26` or `24-27`) is used as given, minus
  the host's CPU and the workers' CPUs, and only CPUs the process may run on. `--aux-cpus off` is the default.
- **How.** Threads the engine starts pin themselves first thing: the adaptive tier's job thread and its per-round
  threads, the prefill copy threads and helpers, the router look-ahead, the PLE reader, the file readers, the stdin reader
  and the watchdog. The threads someone else started (the CUDA driver's) are moved by a sweep over `/proc/self/task`
  once the pool exists and again at the start of every request. The pool's workers and the host thread are never moved,
  and neither is a thread somebody pinned to one CPU. An explicit `STRATA_ADAPT_JOB_CPU` still wins for the adaptive tier's
  job thread. `STRATA_AUX_STAGER=0` leaves the prefill copy threads where they were started: with a one-CPU spare set up
  to 32 of them (`STRATA_STAGER_THREADS`) would share that CPU, which can slow the read of a long prompt.
- **Per request.** `"strata_tune": {"aux_cpus": 1}` (`serve/server.py` forwards `1`, `0`, `true` and `false`) moves the
  threads to the set; `0` puts every moved thread back to the placement it had. A request without the key goes back to the
  start-up setting. With `--batch` the most recently admitted request's value holds, as for the other `strata_tune` keys.
  An engine started without `--aux-cpus` plans the set anyway, so `1` takes `auto`.
- **What it prints.** At start: `strata aux cpus: threads that are neither pool workers nor the host go to CPUs 24 (on;
  N threads placed so far)`, or `(off at start; a request's aux_cpus=1 turns it on)`. A request that changes the state
  prints `strata serve: aux cpus ON/off for this request`.
- **Not covered.** Other processes (the Python launcher, a container runtime) and the engine's host thread.
- **Measured.** One machine: 2x RTX 3080 20 GB at 220 W, Xeon E5-2696 v4 (24 logical CPUs in the container, 16 pool
  workers each on its own CPU), UD-Q4_K_XL, layer split 23, `--batch-mtp`, two slots, adaptive tier on. `auto` put the
  host thread on CPU 2 and the helper threads on CPU 24. Same binary in every arm, only the flag differs. Two pairs of
  whole engine restarts in the order `auto`, off, `auto`, off (12 solo decodes and 10 two-stream rounds per restart,
  medians): solo decode 79.7 and 80.4 tok/s with `auto` against 72.1 and 72.4 off (off / `auto` = 0.904 and 0.900), two
  streams in aggregate 90.8 and 90.7 against 80.8 and 82.8 (0.890 and 0.912). The same configuration restarted twice
  moved solo by 1.009 and 1.005 and two streams by 0.999 and 1.024. Live on one engine, switching per request with
  `strata_tune` and rotating the order (20 solo requests per arm, 16 two-stream rounds): off / on = 0.888 solo and
  0.917 two streams, against 0.985 and 0.990 for an A/A control of the same setting. The host thread's involuntary
  context switches over a bench of about ten minutes were 43.8k and 42.8k with the flag off and 3.9k and 2.8k with
  `auto`. Prompt reads (24K, 25K, 51K, 104K) do not depend on the flag: off is +0.7% to +1.5% against `auto`, inside the
  restart noise. The mean and the p99 of the doorbell-to-flag time are not worse without the flag; what appears is rare
  stalls above 100 us, in 7 and 8 of 12 requests off against 0 or 1 with `auto`. That preemption of the host causes the
  loss is an inference; the counters do not prove it. The warm-up requests of each run were discarded.

---

## Experimental speed projection (EXPERIMENTAL, off by default)

**This is an experiment, not a finished feature.** It ships with Strata but stays off unless you turn it on.

A 480 KB control vector for Qwen3.8-Flash-Next (`data/experimental-speed-projection/`, see its README). After each
of layers 4-44 the engine removes one direction from every hyper-connection stream of the residual: `h -= (h . v) v`,
one unit vector `v` per layer, exactly as llama.cpp does with the package's `--cvec-mode project` patches.

**What it changes.** The vector's own package describes it as a **refusal-direction projection**: with it the model
declines far fewer requests (it reports 1 of 50 vs 50 of 50 on its test set), and removing refusals removes a safety
behaviour - you are responsible for what the model writes with it on. It also shifts ordinary answers a little
(measured below). It is not an optimization in the engine: on the same text it costs 0.2-0.4% per token. What a
chat's tokens/s does with it on depends on the text the model writes (length, repetition, how well the drafts land),
so measure it on your own prompts; the Monitor marks every request ESP or stock.

**Turning it on (at setup).** `START-HERE.bat --setup` asks "Turn on the experimental speed projection?" (default:
no), or pass `--experimental-speed-projection on` (`off`, or a path to another vector GGUF). Only for the original
Qwen3.8-Flash-Next, not Swift 1.5. It writes these engine flags (llama.cpp's) into `strata-<model>.json`:

```
--control-vector-scaled <Strata>\data\experimental-speed-projection\Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

The engine log then says `control vector mode = project, dir = per-layer, layers 4..44 (41 steered)`, and the web
app's About tab lists it. (`--cvec-mode add` is llama.cpp's stock additive mode, for additive vectors.)

**Per request.** A loaded vector is on for every request unless it says otherwise: the web app's Sampling drawer has
a switch, and the API takes `"experimental_speed_projection": false` in the request body (OpenAI and Anthropic; a
config default goes in `"sampling": {"experimental_speed_projection": false}`). Switching drops the conversation
cache once, since the model state was computed the other way. Switched off, the output is token-for-token the stock
model's.

**Measured here** (Q2_0, fixed experts, 2,557 teacher-forced tokens of code, a document and a chat): the top-1 token
changes at 10% of positions, mean KL from the stock model 0.063 nats (max 4.1), perplexity +15% on code, +2.3% on
the document, +0.4% on the chat. Details: `bench/results/2026-09-27-esp/`.

---

## Elastic K/V, experts only in VRAM and other switches (off by default)

Measured on a Ryzen 7 9700X (DDR5-5600) with an RTX 3090 and an RTX 5070 Ti as the second expert tier, this fork's
configs, fixed text (`--spec-follow`, 2 runs each in both orders; time per verify round against the default, code /
thinking / after a 16K prompt / an edit quoting its prompt):

- **`--kv-grow` (`--serve`): the K/V takes VRAM as the context grows.** Upstream 0.1.40's elastic K/V: the K/V pools
  and the main GPU's expert cache are CUDA virtual-memory ranges; the K/V maps 16,384 cells at start and 8,192 more
  whenever a request needs them, taking the chunks of the cache slots just below the prompt path's loan (a hotter
  expert there moves into the loan region's coldest slot first); after two far shorter requests in a row it gives
  them back and the slots refill from the profile. No address moves, so no graph is captured again, and
  `--max-context` cells always fit (this fork's configs turn it on). At 200,192 cells the RTX 3090's cache holds 8,787 slots instead of 7,835 with IQ3_S
  (k8v4) and 5,330 instead of 4,602 with UD-Q4_K_XL (INT8) while a conversation is short. The configs at
  `--max-context 24576`, which gives the cache the same room, took UD-Q4_K_XL -4.8 / -2.4 / -2.5 / -5.6% time a round
  and IQ3_S -1.5 / -0.4 / -0.2 / -1.8%. Under `--expert-math exact` (where residency cannot change the output) a
  multi-turn session with forced small steps, a trim, a regrow and stash round trips gave the same tokens as without
  it. Off with `--disable-hit` and `--mmap-experts`; a growth or trim that fails stops the engine (the server starts
  it again).
- **`--vram-pin-gib G`, `--second-gpu-pin-gib G`: experts only in VRAM.** The profile's hottest experts, up to G GiB,
  stay in the main GPU's lowest cache slots for good (the second option: the next ones in the second GPU's), and the
  RAM arena leaves them out, so a model whose experts outgrow the RAM or the pinning limit can fit. The adaptive tiers
  never evict them, so less of the cache follows the text: with 4 GiB on the RTX 3090, IQ3_S -0.8 / -2.3 / -2.9 / -2.2%
  (the tier copied 4-7x fewer MB over the 3090's link), UD-Q4_K_XL -0.6 / -2.5 / -4.7 / +13.2% (on the edit text its
  hits fell from 80% to 61%); 8 GiB the same. Needs `--expert-profile`, a native pack read from its GGUF and the
  token graph's GPU hits.
- **`STRATA_POOL_FUSED=1`: a native layer's gate/up and down rows in one CPU pool phase** (an expert's down rows start
  once its own hidden row is quantized, no barrier between the halves; the same bytes). IQ3_S -0.8 / -0.5 / -0.3 / -0.7%,
  UD-Q4_K_XL +2.2 / -0.2 / +0.6 / 0.0%.
- **`--adapt-decay F`: the share of the routing counts the adaptive tiers keep after each update** (default 0.7;
  between 0 and 1). 0.85 and 0.92 were within -1.3 / +2.2% of 0.7 on both models.

## GPU expert math (`--expert-math`)

The experts the GPUs hold in VRAM and the ones the CPU computes in RAM are the same weights, but the two can round
differently. `--expert-math` chooses how the GPUs compute theirs (native packs; the CPU always runs ggml's kernels):

- **`fast` (default)**: llama.cpp's CUDA arithmetic (q8_1 activations, lane-parallel sums). A GPU expert agrees with
  the CPU computing the same expert to rounding, not bit for bit, so greedy output can change with which experts happen
  to be in VRAM.
- **`exact`**: the GPUs compute an expert bit for bit as the CPU pool does: each layer's ggml activation formats (Q8_K,
  Q8_0, Q8_1), the order in which ggml's x86 dot products sum, the one place where that order is the compiler's (Q5_K's
  sum of the mins: one FMA from GCC or Clang, a product then a sum from MSVC, probed at start), and one `expf` in the
  SwiGLU on both sides (`expf_contract`, the algorithm of glibc's: the C runtimes' differ, MSVC's at 193,922 of the
  2^32 floats). A prompt's experts stay on the main GPU (llama.cpp's MMQ kernels round them differently on the second).
  The output then does not depend on which experts are resident: it is the output of a run with every expert on the CPU
  (`--disable-hit`, which also turns off the second GPU and the adaptive tiers). For regression tests and A/B
  comparisons. Its kernels cover the K-quant files' formats (Q4_K / Q5_K gate/up, Q5_1 / Q8_0 down: UD-Q4_K_XL); the
  engine refuses `exact` for others at start. The MTP draft layer's experts keep the fast kernels (drafts are verified).

Measured with UD-Q4_K_XL on an RTX 3090 + RTX 5070 Ti, Ryzen 7 9700X (AVX-512; ggml's x86 dots for these formats have
no AVX-512 branches), fixed text (`--spec-follow`, `--window-hashes`): two runs with the config's adaptive tiers
on both GPUs and a run with every expert on the CPU wrote identical window hashes, 633 windows of thinking text and 108
after a 16K-token prompt; `fast` parts from the CPU run at the first window. Generation about 84 against 108 tok/s on
the thinking text, 88 against 122 after the 16K prompt. `native_expert_parity` compares every gate, up and down dot of
the exact kernels with ggml's `vec_dot`, the hidden rows' bytes and the outputs; `swiglu_expf_parity` the SwiGLU over
every float. The PR's measurements on an i5-14600K (AVX2): `bench/results/2026-10-04-expert-math/`.

---

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `the NVIDIA driver is too old` | Update the driver (NVIDIA App or nvidia.com/drivers), restart, run `START-HERE.bat` again. |
| Python or the build tools could not be installed | Install what it names (links are printed), then run it again. Everything already done is kept. |
| `port 8080 is already in use` | Strata is already running (look for its window), or another program uses the port: `START-HERE.bat --port 8081`. |
| `cudaHostRegister ... out of memory` in the log | Normal on Windows: the engine pins the experts in per-layer slices instead. Only a problem if the load then fails. |
| `ExpertCache: cudaMalloc(...) failed: out of memory` although VRAM is free | Windows' page file is off or tiny: every allocation on the graphics card is also charged to Windows' commit (RAM + page file). Set a fixed page file size, the same initial and maximum size (e.g. 65536 MB: System > About > Advanced system settings > Performance > Advanced > Virtual memory > Custom size), and restart: a page file Windows grows on demand ("System managed", or an initial size below the maximum) may not grow in time while the card's memory is charged (#60: "System managed" and 4096-32768 MB still failed, a fixed 64 GB worked). Since 0.1.19 the engine retries with a smaller cache instead of stopping, and setup warns about a page file under 4 GB (issue #60). |
| The first start takes minutes | It is reading 34-55 GB into RAM; the second start is faster while the files are in the OS cache. |
| The PC freezes for a few minutes at the start | Normal, most of all the first time (the server window says when it happens): the engine loads the experts into RAM, pins part of it for the GPU and sizes the expert cache. Wait; don't close the window. Still frozen after 10 minutes: restart the PC, close other programs, try again, or pick a smaller size. |
| `the engine stopped unexpectedly (exit code ...)` | The engine process ended mid-answer - usually out of RAM (Linux ends the biggest program: `sudo dmesg \| grep -i -E 'killed process\|out of memory'`). The next request starts it again by itself. If it repeats: close other programs or pick a smaller size. The server also warns at start when the model's experts leave less than ~6 GB of RAM for everything else. |
| Slow output, disk light busy | Not enough free RAM: close other programs, or choose Q2_0 / IQ2_XS. |
| `prompt ... exceeds the context` | The request is longer than the context you chose: run setup again with a bigger `--context`. |
| Slower than the tables | The monitor plugged into the GPU and other GPU programs take VRAM from the expert cache; RAM running below its rated speed (enable EXPO/XMP in the BIOS) slows the CPU half. |
| `this server was started without the vision encoder` | The model was set up for text only: run setup again with `--vision gpu`. |
| Setup puts the model on an old drive after you moved or reinstalled Strata (#1070) | Setup remembers the data folder you used last in its settings file (`%APPDATA%\Strata\settings.json` on Windows, `~/.config/strata/settings.json` on Linux; the `data_dir` entry) and uses it again. Choose another place with `--data-dir DIR` (model files, packs, MTP layer) or `--models-dir DIR` (the GGUF files only), or delete the `data_dir` line; `--gguf-dir DIR` uses GGUF files you already have without copying. To move a model, copy its folder and run setup with the new `--data-dir`. |
| A picture is refused or `cannot read the image` | The file is not a picture Pillow can open (JPEG, PNG, WebP, GIF, BMP, TIFF, AVIF work). |
| Pictures are slow (3-30 s) | The encoder runs on the CPU: run setup again with `--vision gpu` (needs ~1.4 GB of VRAM). |
| A request never finishes: "reading the prompt", GPU "100%" at low power | The GPU ran out of VRAM (engines before 0.1.9 could end with ~30 MiB free at large contexts). Run `START-HERE.bat` once to get engine 0.1.9 or newer; the log then says `... MiB of VRAM free with everything loaded` (a few hundred) and names the `--vram-reserve-mib` to add if it is low. |
| Output much slower than the tables above, only with a large `--context`, and the start says a small `... MiB of VRAM free with everything loaded` | The expert cache does not fit the card at that context. With `--expert-cache auto` the engine shrinks it to fit; with a fixed number it does not check again after the slots are written, and on Windows the extra is put in system memory instead of failing, so only the speed shows it. Use `auto`, a smaller number, or raise `--vram-reserve-mib` (it is deducted before the cache is sized). See "The expert cache size is a budget" under Sharing the GPU. |
| Generation stops mid-answer, GPU "100%", one CPU core busy | Fixed in engine 0.1.12 (issue #29, a race in the CPU expert pool on big-VRAM cards). Since then a request that stops moving ends with an error instead of hanging (after 2 minutes; 1 minute from 0.1.13): the log says `no progress for ... s ... (issue #29)` with where it stopped, and the next request starts the engine again. If you see that line, please open an issue with it. Engine 0.1.13 adds a stall report under it (what every expert-pool thread and the GPU handshake were doing, memory and page faults) and, on Windows, a `strata-stall-<pid>.dmp` file with every thread's stack: attach both. (`STRATA_WATCHDOG_S` sets the time in seconds; 0 turns it off.) Engine 0.1.14 fixes the stall those reports found (issue #31: with the IQ packs the host could wait forever inside the NVIDIA driver while copying experts in a verify window; the experts are now copied by a GPU kernel, `--pcie-mode dma` restores the old way). |
| `no progress for 60 s ... reading the prompt` on Linux, and the stall report says `threads waiting on the disk (state D): 16 ...` | The engine waits for the drive, not a deadlock: the n-gram table is read at random (`--ple-io direct`), which a rotational disk cannot keep up with (#605). The engine warns at start when the table is on one; `--ple-io ram` (Linux, needs RAM for the table) or the model on an SSD fixes it. Setup adds `--ple-io ram` itself on a rotational disk when the RAM holds the table (0.1.39). |
| `the engine said nothing for ... s during the request` or `... did not finish the request after it was stopped (STOP)` | Issue #481: the engine and the server lost step (the engine waits for its next command, the server for the request's end; GPU at 0 %, nothing in the log). The server ends the engine after 300 s without a line from it during a request (while a prompt is read: each chunk may take three times the previous one's time, the first one up to its tokens at 50 tok/s more), the request ends with an error and the next request starts the engine again. `"engine_silence_s": 600` in `strata-<model>.json` sets the time (0 = wait forever, as before). If you see it, please add the end of the engine log to #481. |
| `the engine said nothing for ... s and used no CPU or disk in that time (frozen ...)` | Issue #1317: a quicker version of the check above for an engine that is not slow but stopped (a deadlock inside a CUDA call, a driver stall): after 90 s without a line, if the engine process has also used no CPU time and moved no disk bytes in that time, the server ends it and the next request starts it again. An engine that is silent but still working is never ended by this check, the server prints one note (`... but is still working; it is not ended`) and leaves it to `engine_silence_s`. `STRATA_ENGINE_STALL_S=180` sets the time, `0` turns the check off (it needs `psutil`, which setup installs). |
| `out of memory: cudaFuncSetAttribute` in the log (IQ3_XXS, long prompt) | Fixed in engine 0.1.15: CUDA loaded a kernel's code when it was first needed, and mid-prompt there was no VRAM left for it. Run `START-HERE.bat` (Windows) or `./setup.sh` (Linux) once to update. |
| Anything else | The engine log is `strata-<model>.log` in this folder. |

---

## How it works

<p align="center"><img src="paper/tiers.svg" width="760" alt="memory tiers"></p>

- **GPU (VRAM):** attention and DeltaNet mixers, the gated-residual weights, routers, shared experts, output head, the MTP
  draft layer, the KV cache (from 64K: only its most-read part, the rest streams from RAM), and an **expert cache** that fills the rest of VRAM with the most-used experts (it adapts to
  the conversation while you chat).
- **RAM:** all 24,576 experts, pinned. The CPU computes the experts that are not on the GPU **in place**, at the same time
  as the GPU works on the cached ones (AVX-512 / AVX2 kernels, ggml's for the i-quants).
- **SSD:** the 28.8 GB n-gram table, a few rows per token, read unbuffered past the OS cache (`--ple-io direct`, the
  default, made for SSDs; on a rotational disk `--ple-io ram` keeps the table in RAM, #605). A start-up probe warns when direct reads measure slow (under 15K rows/s) but never switches by itself (since 0.1.43.1; `STRATA_PLE_AUTO_MMAP=1` brings back the 0.1.42 switch to `--ple-io mmap`): the bar is calibrated for NVMe, a SATA SSD tops out at 10-14K rows/s with direct's 16 reads in flight, and on one (Linux, 77.8K-token prompt) mmap read 2.1K rows/s against direct's 14.4K (#2040), while on a Windows box with a slow mapped file mmap stalled prompts (#1984). The engine reads the table
  in the format the GGUF has it: IQ4_NL (the default table), Q4_0, Q5_0, Q5_1, Q8_0, FP8 (E4M3 with a scale) or BF16.
  Measured on 4,000 random rows against the checkpoint's own BF16 table, the mean per-row error is Q8_0 0.53%, FP8 2.64%,
  Q5_1 3.78%, Q5_0 4.25%, IQ4_NL 7.60%, Q4_1 7.80%, Q4_0 8.55% (a Q8_0 table is 54 GB, an IQ4_NL one 28.8 GB). On the Q2_0 model, 700 teacher-forced tokens, the mean KL against the BF16 table is 0.0117 (Q5_0), 0.0121 (Q5_1), 0.0125
  (Q8_0), 0.0128 (FP8), 0.0140 (Q4_0) and 0.0146 (IQ4_NL), with the perplexity within 1% of BF16's either way: any change to the
  table moves the 2-bit model by about 0.012, so the formats are hard to tell apart. Setup does not offer another table;
  it only changes which GGUF the engine is given.
- **Speculation:** the model's own MTP layer drafts up to 3 tokens; one pass over all 48 layers checks them. 2.4-3.2
  tokens per pass on average. Where the answer repeats earlier text for 16+ tokens (a file returned with an edit), the
  pass checks up to 7 tokens that followed it there instead.
- **Prompts** are processed in 2,048-token chunks with the experts streamed to the GPU over PCIe.

The full story, with measurements, bottlenecks and what comes next: **[docs/paper/Strata-Paper.pdf](paper/Strata-Paper.pdf)**.

---

## Credits and licenses

Strata itself: [MIT](../LICENSE). The model files are not part of it; their licenses apply to them (below).

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; quantizations:
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).
  The Coder: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (Apache-2.0 per its card); its support in Strata came from @pjgmobile's PR #54.
  Swift 1.5: [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  by UkisAI. Their licenses apply to the weights.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the i-quant formats, the GPU dot products and
  dequantizers transcribed in `src/kernels/cuda/iq_kernels.cu`, the CPU backend linked for the i-quant experts, the
  `mtmd` library behind the image encoder (`tools/vision/`), and `gguf-py` used by the tools. See
  `third_party/ggml/LICENSE`.
- Ideas from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen); references in the paper.
- The web app's font: [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1, see
  `serve/web/fonts/OFL.txt`). Its Monitor tab started from @code-martin's dashboard idea (PR #22).
- The experimental speed projection's vector (`data/experimental-speed-projection/`): Qwen Community License 1.0,
  made from the model's activations (see its README).

### Start the API without loading the model

`serve/server.py --engine strata --config strata-<model>.json --lazy` (or `"lazy_load": true` in that
config) starts the lightweight HTTP API without starting the native engine or the vision encoder. The first
generation request, or `POST /load`, starts the vision encoder first when configured, then loads the engine through
the existing reload path, including `before_load` and `min_free_vram_mib`. This also works with image requests:
the load finishes before Strata encodes the image. `POST /unload` stops both processes. Eager startup remains the
default.

`POST /v1/load` and `/v1/unload` are JSON control aliases for integrations, accepting `{}` or
`{"model":"<configured model>"}` and returning model status. They require the configured API key,
`application/json`, and no foreign browser Origin. They return **409** while a request is active or queued,
and **404** for an unknown model. Existing `/load` and `/unload` behavior is preserved. `/api/health` aliases
`/health`; `/v1/status` exposes `loaded` and `auto_load`. The unloaded model remains discoverable.

Unloading and shutdown close the native engine's stdin after sending `QUIT`, allowing Windows' detached
stdin reader to see EOF. Cleanup waits for process exit before releasing handles; if forced shutdown still
times out, the server keeps ownership and reports an error rather than claiming the model was unloaded.

### One-pass choices

`POST /v1/choose` answers a multiple-choice question without generating: the chat (and its `tools`, if any) is
rendered as `/v1/chat/completions` renders it with thinking off, so it shares a conversation's cached prompt;
`answer_prefix` (optional) follows the assistant header, and the engine reads the probability of each option's token
there from the head's logits, before any sampling setting. Options must be single tokens (letters are; after a
prefix ending in a word, `" A"` is one token and `"A"` another). It suits classifiers and routers in agent loops: a
decision costs one prompt read, and the probabilities double as a confidence to act on or to fall back from.

```bash
curl http://127.0.0.1:8080/v1/choose -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Is Paris the capital of France? A) yes B) no. Answer with the letter."}],
  "options": ["A", "B"] }'
# {"probs": {"A": 0.99988, "B": 0.00012}, "logprobs": {"A": -0.0009, "B": -9.005}, "mass": 0.9992, "choice": "A",
#  "confidence": 0.99988, "prompt_tokens": 31, "reused": 0, "seconds": 0.71}
```

`probs` are renormalized over the options; `mass` is the probability the options got out of everything the model could
have said there (low: it wanted to answer something else, and the choice means little). Through the engine's own
protocol the same is a `score=ID:ID:...` key on `GEN`, which prints `LP id:logprob ...` before the first `T` (with the
output head split across two GPUs, that window joins the whole rows).

### JSON response formats

`POST /v1/chat/completions` accepts `response_format: {"type":"json_object"}` or
`{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}}}`.
The schema must accept only JSON objects at its root: `"type":"object"`, or an `anyOf`/`oneOf` whose branches are all
object schemas (an `allOf` with an object member, or a local `$ref` to one, also counts), as apps written for
llama.cpp's `json_schema` send. A root that also allows an array, string, number, boolean or null is refused. Local
`#` references work; remote references are refused.
`json_schema` is checked with the Python package `jsonschema` when it is installed (`python -m pip install
"jsonschema>=4.23,<5"`; setup does not add it); without it the answer is only checked to be one JSON object, and the
server says so once.

This is **schema prompting followed by server validation**, not grammar-constrained decoding. One generation
is made per request, with no hidden retry. Successful responses contain a validated JSON object. Malformed JSON,
duplicate keys, non-finite numbers, schema violations and incomplete generations return **502** with
`error.code: structured_output_failed`; invalid request schemas return **400**. JSON formats combined with
tools/MCP are refused explicitly. Without `response_format`, ordinary text and tool behavior stays the same.

Structured SSE buffers the answer while sending keep-alive comments. It emits content only after validation,
then usage/timings and `[DONE]`; failures emit an SSE error and `[DONE]` without invalid content deltas.
`/v1/status.structured_output` advertises the formats, validation method and buffered streaming behavior.

### API request monitor

Off by default, since it keeps prompts and answers in memory: turn it on with `"api_monitor": true` in
`strata-<model>.json` (or `serve/server.py --api-monitor`); otherwise nothing is recorded and the two endpoints below
answer 404.
Open `/api-monitor` to inspect API traffic without opening a chat. It shows the model state, safe
load/unload controls, active/queued requests, original request bodies, output, separate reasoning and
non-stream response bodies. Total wall-clock includes FIFO waits and automatic loading; load, queue,
first-token, prompt/output tokens and engine decode timing are shown separately.

`GET /api/requests` returns compact summaries; `GET /api/requests?id=<id>` returns one retained request.
Both use the existing API-key check. The monitor retains the newest **100 requests in memory** until restart,
with **262,144 characters per input/output/reasoning/response field** and visible truncation flags. The actual API
responses are unaffected. Headers are not recorded, and the monitor key is kept in this tab's session storage.
Treat request history as sensitive input/output when exposing Strata on a network: set an API key as above.
The page uses relative URLs and works through the existing host binding or a reverse proxy.

### Prompt buffers: `bo` shares `emb` (#1454)

The prompt path's half-output buffer `bo` reuses the embedding buffer `emb`, which is dead after the first hyper-connection broadcast: T x 2560 floats less VRAM per chunk (320 MiB at 32768 rows). The planner still counts those bytes by default, so the auto chunk and the cache slots the prompt path borrows are exactly those of 0.1.40.3 and the output bits are unchanged. `STRATA_EMB_REUSE_ACCOUNT=1` lets the planner use the saved bytes: where VRAM limits the chunk it grows (RTX 3060, IQ3_XXS: 6400 to 6656 tokens, 1652 to 1670 borrowed slots, prompt about +3.9%). A different chunk changes the prompt path's rounding, so prompt residuals are not byte-identical to the default; greedy output matched in our runs. Opt-in until it has a KL measurement.

**Fused int8 prompt experts on gfx12 (RDNA4, R9700): on by default for IQ3_S and IQ3_XXS packs.** `STRATA_PF_FUSED=0` turns them off (back to the MMQ prompt kernels, the previous bits). R9700, 1 card: prompt 4K +15.7%, 16K +13..+17%, 32K +19..+20%, a 96K prompt +16% (answers identical text); first-token KL against the FP16 prompt path over 96 prompts is equal to MMQ within noise (IQ3_XXS 0.00613 against 0.00683, IQ3_S 0.00416 against 0.00372). Other quants and other GPUs are unchanged; `STRATA_PF_FUSED=1` still opts in every covered format.

**gfx12 (Linux): a bare `--prefill auto` may take 16384-token chunks.** Explicit `--prefill N`, `auto:N` and `STRATA_PREFILL_AUTO_MAX` are untouched; `STRATA_GFX12_CHUNK=0` or `STRATA_GFX12_DEFAULTS=0` keeps 8192.
