# Measurements: Ryzen 7 5700X + RTX 3090 + RTX 5060 Ti (Linux)

Speeds of this branch (`custom-plus`) with unsloth's UD-IQ4_XS and UD-Q4_K_XL, on one GPU and on two. The configs
are in [examples/r7-5700x](../examples/r7-5700x/). Spanish version: [MEASUREMENTS.es.md](MEASUREMENTS.es.md).

## The machine

| | |
|---|---|
| CPU | AMD Ryzen 7 5700X: 8 cores / 16 threads, Zen 3, AVX2 (no AVX-512) |
| RAM | 128 GB DDR4-3200 CL22 (4 x 32 GB), ~36 GB/s read (STREAM) |
| Main GPU | NVIDIA RTX 3090 24 GB, PCIe 4.0 x8, power limit 250 W |
| Second GPU (expert tier) | NVIDIA RTX 5060 Ti 16 GB, PCIe 4.0 x8, power limit 150 W |
| Software | Linux (kernel 7.2), NVIDIA driver 615.71, CUDA 13.4, engine built with `CMAKE_CUDA_ARCHITECTURES=86;120` |

At the cards' stock power limits (350 W and 180 W) the speeds were the same (UD-Q4_K_XL: 63.5 -> 63.8 tok/s on the
3090, 80.9 -> 80.3 with both cards): the engine is bound by the CPU, the RAM and PCIe, not by the GPUs' power.

## The models

Qwen3.8-Flash-Next (48 layers x 512 routed experts), unsloth's GGUF files. The experts stay in RAM and the engine
keeps the most used ones in VRAM (an adaptive cache); the CPU computes the rest.

| | UD-IQ4_XS | UD-Q4_K_XL |
|---|---|---|
| GGUF files | 3 | 4 |
| Expert weights | 55.4 GiB | 71.7 GiB |
| Expert formats (gate/up · down) | IQ3_S (47 layers), IQ4_XS (1) · IQ4_NL (43), Q8_0 (5) | Q4_K (47), Q5_K (1) · Q5_1 (43), Q8_0 (5) |
| RAM taken by the engine | ~58 GiB | ~81 GB |
| CPU expert kernels | compute-bound: scale with cores | bandwidth-bound: DDR4 saturates at ~4 cores |

Both: 200,192 tokens of context, int8 KV cache, the MTP draft layer with a Spanish draft vocabulary
(`data/draft_vocab_es.bin`) plus prompt lookup, greedy verification (the speculation never changes the output).
These figures were measured with the 94,962-id Spanish subset shipped before; the 47,196-id one shipped now
measured the same on UD-IQ4_XS (74.8% vs 75.1% of the Spanish drafts accepted, 90.4 vs 89.5 tok/s) with half the
draft head.

## Settings

| | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPUs | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPUs |
|---|---:|---:|---:|---:|
| `--pool-workers` | 7 | 7 | 4 | 4 |
| `--adapt-swaps` | 192 | 192 | 96 | 96 |
| `--adapt-decay` | 0.92 | 0.7 | 0.7 | 0.7 |
| `STRATA_IQ256_GATHER` | 1 | 1 | - | - |
| `--pcie-frac` (kernel) | 0 | 0 | 0.3 | 0 |
| Second GPU | - | `--second-gpu 0`, 512 MiB reserve, 6 MB prefetch, `--head-split 0.45` | - | same as IQ4_XS |
| `--kv-grow` | on | on | on | on |
| `STRATA_POOL_FUSED` | 1 | 1 | 1 | 1 |

## How it was measured

- The engine driven through its own `--serve` protocol, prompt cache off (every prompt read in full), greedy,
  speculation on (`--spec 4 --spec-lookup 16`).
- Six fixed prompts, mostly in Spanish: `es_chat` (71-token question, 400 tokens out), `es_think` (133 tokens, 700
  out, reasoning on), `es_doc` (an 18,076-token document, then 400 out), `code` (62 tokens, 600 out), `edit` (a
  3,340-token script returned edited, ~2,600 out), `proto5k` (a 5,296-token prompt, 256 out). The long texts are
  private notes and are not published.
- Decode speed = the geometric mean of the six; each config is the mean of two runs (UD-IQ4_XS 1 GPU 93.0 / 92.8,
  2 GPUs 116.8 / 115.7; UD-Q4_K_XL 71.0 / 71.4 and 85.4 / 84.3).
- Agent sessions: 18 turns through Niko1221/Strata's server, the conversation growing to ~31K tokens (pasted source
  files, docs, topic changes), 600 tokens per answer.

## Results

Decode speed (tokens/s):

| Prompt | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPUs | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPUs |
|---|---:|---:|---:|---:|
| es_chat | 89.9 | 116.7 | 70.2 | 88.8 |
| es_think | 97.5 | 122.9 | 83.7 | 86.4 |
| es_doc (after 18K) | 79.5 | 101.8 | 59.3 | 77.9 |
| code | 99.1 | 129.4 | 74.8 | 89.9 |
| edit | 132.8 | 183.2 | 97.0 | 129.2 |
| proto5k (after 5K) | 70.0 | 71.4 | 51.7 | 53.8 |
| **Geometric mean** | **92.9** | **116.3** | **71.2** | **84.9** |
| (before `--kv-grow` and the fused pool) | 83.6 | 109.4 | 64.2 | 80.5 |

Prompt reading (prefill, tokens/s):

| Prompt | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPUs | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPUs |
|---|---:|---:|---:|---:|
| 18,076 tokens | 1,803 | 1,836 | 1,730 | 1,758 |
| 5,296 tokens | 1,180 | 1,550 | 899 | 1,113 |
| 3,340 tokens | 851 | 1,086 | 660 | 794 |

- The second GPU adds **+25%** to UD-IQ4_XS and **+19%** to UD-Q4_K_XL (it holds ~14.5 GB more experts and computes
  its share of each layer while the CPU computes its own). It barely changes the 18K prefill, which is bound by the
  3090's x8 link.
- Expert VRAM hit rate: 83.6% / 90.6% (UD-IQ4_XS, 1 / 2 GPUs), 80.7% / 85.1% (UD-Q4_K_XL). Draft acceptance ~89%.
- Agent sessions (one GPU, one run each): UD-IQ4_XS 86.2 tok/s, UD-Q4_K_XL 64.5 tok/s (before: 78.1 and 59.9).
- UD-IQ4_XS is faster than UD-Q4_K_XL here (+30% on one GPU, +37% on two): fewer bytes per expert, so more of them fit
  in VRAM, and its compute-bound CPU kernels use all 7 workers. UD-Q4_K_XL is the larger quant of the two.

## Long prompts

Prompts of 32K to 200K distinct tokens (prose, then this repo's docs and source code), 128 tokens of answer, one run each.

| Prompt | Read in (prefill tok/s) UD-IQ4_XS 1 / 2 GPUs | UD-Q4_K_XL 1 / 2 GPUs | Decode after: UD-IQ4_XS 1 / 2 GPUs | UD-Q4_K_XL 1 / 2 GPUs |
|---|---:|---:|---:|---:|
| 32,022 tokens | 13.4 s (2,390) / 13.3 s (2,400) | 13.6 s (2,363) / 13.8 s (2,320) | 73.5 / 94.7 | 58.6 / 83.0 |
| 64,022 tokens | 26.4 s (2,427) / 25.6 s (2,504) | 26.6 s (2,405) / 26.5 s (2,418) | 93.6 / 99.9 | 77.0 / 90.0 |
| 120,019 tokens | 50.4 s (2,383) / 48.1 s (2,494) | 51.0 s (2,355) / 50.0 s (2,399) | 66.2 / 98.1 | 48.4 / 69.8 |
| 200,019 tokens | 88.5 s (2,261) / 83.6 s (2,392) | 90.7 s (2,205) / 87.9 s (2,276) | 60.0 / 97.3 | 41.6 / 74.0 |

The K/V grows with the prompt: on the 3090 alone UD-IQ4_XS reaches the whole 200,192 cells by giving up 452 of its
7,160 expert slots (6,121 still hold experts, as without `--kv-grow`).

## With 64 GB of RAM

The same PC with the engine limited to 60 GiB (what a 64 GB PC leaves free) by a systemd cgroup (`MemoryMax=60G`;
the engine's memory and the file cache of the files it reads were all charged to it). UD-Q4_K_XL's 71.7 GiB of experts
do not fit: it runs from a pack with `experts.bin` (`iq_pack.py --experts-bin`) through `--mmap-experts`, reading from
the NVMe (Crucial P3 Plus), and `--kv-grow` is off there. Before each run the model files were dropped from the page
cache. Runs: UD-IQ4_XS 2 and 2, UD-Q4_K_XL 1 and 1.

| Decode, tokens/s | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPUs | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPUs |
|---|---:|---:|---:|---:|
| Spanish chat | 89.8 | 112.4 | 25.2 | 37.3 |
| Reasoning | 97.3 | 120.2 | 30.5 | 39.7 |
| After an 18K document | 79.0 | 104.4 | 19.9 | 32.9 |
| Code | 99.2 | 127.7 | 28.3 | 43.1 |
| Edit | 132.8 | 183.3 | 44.2 | 71.3 |
| After a 5K prompt | 69.6 | 72.8 | 15.7 | 26.1 |
| **Geometric mean** | **92.7** | **115.7** | **25.9** | **39.7** |
| **Geometric mean** (128 GB) | 92.9 | 116.3 | 71.2 | 84.9 |

| Prefill, tokens/s | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPUs | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPUs |
|---|---:|---:|---:|---:|
| 18,076 tokens | 1,813 | 1,833 | 438 | 491 |
| 5,296 tokens | 1,180 | 1,550 | 162 | 213 |
| 3,340 tokens | 851 | 1,084 | 147 | 157 |

Long prompts with 64 GB (one run each): time to read the prompt · decode right after it. UD-Q4_K_XL's are from the
engine before the elastic K/V and the fused pool (not run again: the 200K prompt took 43.8 min).

| Prompt | UD-IQ4_XS 1 GPU | UD-IQ4_XS 2 GPUs | UD-Q4_K_XL 1 GPU | UD-Q4_K_XL 2 GPUs |
|---|---:|---:|---:|---:|
| 32,022 tokens | 13.4 s · 75.1 | 13.4 s · 101.4 | 42.7 s · 17.8 | 44.1 s · 33.1 |
| 64,022 tokens | 26.3 s · 95.0 | 25.6 s · 97.7 | 2.7 min · 22.9 | 1.7 min · 50.0 |
| 120,019 tokens | 50.3 s · 64.9 | 48.1 s · 105.6 | 4.6 min · 2.7 | 72.1 s · 50.2 |
| 200,019 tokens | 88.3 s · 59.4 | 83.3 s · 98.2 | 43.8 min · 6.0 | 2.0 min · 8.2 |

Agent sessions with 64 GB (one run each, `--prompt-cache 4`): UD-IQ4_XS 85.6 tok/s (128 GB: 86.2), UD-Q4_K_XL 30.7
(128 GB: 64.5). With the engine's default 16 checkpoints (~112 MiB of RAM each) UD-IQ4_XS's session went over the
60 GiB limit at the 12th turn.

## What the changes on top of eddoursul's `custom` gave

UD-IQ4_XS, one GPU, six-prompt geometric mean:

| Step | tok/s |
|---|---:|
| eddoursul `custom` + #2 + its follow-up + this branch's fixes (five runs) | 77.3 (76.8-77.8) |
| + the IQ4_XS AVX-2 multi-token kernel (Niko1221/Strata) | 78.0 (+1.3%) |
| + the AVX2 gather of the IQ3_S grid (`STRATA_IQ256_GATHER=1`) | 80.9 (+4.9%) |
| + `--adapt-decay 0.92` | 83.3 (+7.7%) |
| + the later additions from upstream's PRs (#863, #851, #606/#838) | 83.6 (+8.2%) |
| + the elastic K/V (`--kv-grow`, upstream 0.1.40) | 90.9 |
| + the fused CPU pool (`STRATA_POOL_FUSED=1`) | 92.9 |

The last two were measured against each other and the step before in one session, two runs each (82.5 -> 90.9 ->
92.9: +10.1%, +12.5%); the earlier steps the days before.

In the agent sessions the same steps gave 75.0 -> 76.0 -> 77.8 (+3.7%). The gather is bit-exact (the parity tool's
output is identical with and without it); the draft acceptance and the prefill did not change.

`--adapt-decay` on UD-IQ4_XS (one GPU, with the gather; two runs each, before the elastic K/V and the fused pool): 0.70 76.8 · 0.85 82.1 · 0.92 83.6 · 0.95 81.0 · 0.97 81.3 · 0.99 70.8
(1.0 is refused since PR #591). With the second GPU 0.92 cancels the gather's gain and
0.95 is -2%, and on UD-Q4_K_XL it gives nothing in the agent sessions, so those configs keep 0.7.

### Later additions (from upstream's open pull requests)

- **q8_1 blocks kept finite** (upstream #606, PR #838, at this fork's five quantizers): a massive activation could turn
  a block's fp16 scale or sum into inf, then NaN, and the model would answer one token forever. The same bits for
  every normal block, no speed cost.
- **The AVX2 gather reorganized** (PR #863 by Hardin22): UD-IQ4_XS on the 3090 83.1 -> 83.6 tok/s (five runs), with both cards
  108.0 -> 109.4 (four runs). Its one-token IQ3_S path stays off on AMD, where ggml's dot is still slightly faster for
  one token (0.322 vs 0.334 ms per expert on the 5700X).
- **An AVX2 Q8_K activation quantizer** (PR #851 by Hardin22): byte-identical to ggml's; no measurable change here.
- **`--adapt-decay` outside (0, 1) refused** (PR #591). UD-Q4_K_XL is unchanged by all four.

### From upstream 0.1.40 and other forks

- **The elastic K/V** (upstream 2fbe321 + 4da8a57, `--kv-grow`): 82.5 -> 90.9 on UD-IQ4_XS on the 3090. Checked:
  with a fixed cache of 3,000 slots and no adaptation, the K/V grown from 4,096 cells in steps of 2,048 into free VRAM
  gives the six prompts' tokens bit for bit; so do nine requests of two interleaved conversations through the server
  with the prompt cache stashing and restoring them. Off with `--mmap-experts` (a crash after the first growth there,
  as upstream asks for every expert in RAM).
- **The fused CPU pool** (`STRATA_POOL_FUSED=1`, after Hardin22/Strata-DualGPU 11b1f25): +2.3% (2 GPUs) to +3.7%
  (UD-Q4_K_XL, 1 GPU), the same arithmetic.
- **The residency table's uploads wait for their copy** (upstream #1001): no output change.
- Tried without a gain: upstream PR #930's scalar IQ3_S decode (0.403 -> 0.278 ms an expert at one token on one
  thread, but 83.6 -> 83.3 in the engine: its 8 threads are bound by the RAM), `--no-second-gpu-adapt` (108.4 ->
  108.5), and the 4-bit K/V caches (not kept: they lose some accuracy).
- The context reserved, before `--kv-grow` (UD-IQ4_XS, 3090, two runs each): 200K 83.3 · 128K 86.7 · 64K 89.9 ·
  32K 95.3 tok/s.

Other settings tried on this machine without a gain: more pool workers than 7 (UD-IQ4_XS) or 4 (UD-Q4_K_XL),
`--spec` 3/5/6, `--spec-min-p` 0.3/0.7, `--spec-lookup` 8/24, `--adapt-every 2`, `--adapt-swaps 192` on
UD-Q4_K_XL, `--pcie-frac` above 0.3, the GPUs' stock power limits.

On this SMT CPU the Linux fix in this branch (one pool worker per physical core) took UD-Q4_K_XL on the 3090 from
50.7 tok/s (15 workers on 16 logical processors) to 58.1 (7 workers on 7 cores), before the later tuning.
