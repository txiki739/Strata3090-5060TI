# Lamina

Lamina runs [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a 125-billion-parameter
mixture-of-experts model, on one PC with NVIDIA GPUs and plenty of RAM, and
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on one GPU or a few. It serves them through the OpenAI and
Anthropic APIs and its own web app, for chat, coding agents and pictures.

The experts live in RAM and run where it is fastest: the GPUs keep the ones the model uses most and the CPU computes
the rest beside them. Each step decodes several tokens at once: a draft layer proposes them and one pass of the model
checks them all, so the text is the model's own. How it works: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

Lamina is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors. Since 2026-10-29 it
follows its own course: NVIDIA GPUs only, extra GPUs as further expert tiers, and a draft layer trained on the engine's
own output. Its speeds against llama.cpp and Strata on one PC are in [docs/COMPARISON.md](docs/COMPARISON.md).

## What you need

- **An NVIDIA GPU** of compute capability 8.0 or newer (RTX 30, 40 or 50 series, or the workstation cards of those
  generations), with a driver for CUDA 13. The model runs on one GPU that nothing else uses: no display, no other
  program. Up to three more GPUs become further expert tiers, and these may drive the display. The engine was developed
  on an RTX 3090 (24 GB), with an RTX 5070 Ti and an RTX 5060 Ti (16 GB each) as tiers. A smaller card holds fewer
  experts and runs slower. Qwen3.6-35B-A3B and Qwen3.8-27B take more GPUs differently: their layers are split over the
  fewest that hold the whole model. Qwen3.8-27B, a dense model, needs ~36 GB of VRAM at its full context (262,144
  tokens, K and V in 8 bits): two GPUs, such as a 24 GB and a 16 GB one.
- **RAM** for the experts, which are pinned while the model runs:

  | Model file | RAM taken | Download |
  | --- | --- | --- |
  | Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S | ~53 GB | ~86 GB |
  | Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL | ~81 GB | ~114 GB |
  | Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL | ~22 GB on one GPU, ~2 GB split over GPUs that hold it | ~24 GB |
  | Qwen3.8-27B, unsloth's UD-Q6_K_M | ~2 GB | ~24 GB |

  The engine also runs unsloth's other Flash-Next quants, all but UD-IQ1_S and BF16: UD-IQ1_M, UD-Q2_0, UD-IQ2_XS,
  UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0, whose experts take ~35 GB (UD-IQ1_M)
  to ~128 GB (Q8_0) of RAM, and [Agention's Gyro-L](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF).
  Setup downloads none of them: [docs/SETUP.md](docs/SETUP.md) says how to run one.

- **An SSD**: Flash-Next reads a 28.8 GB table of per-layer embeddings from the model file at every token.
- **A 64-bit x86 CPU with AVX2** (AVX-512 is used when it is there). The CPU computes the experts no GPU holds, so its
  cores and the RAM's bandwidth set much of the speed.
- **Windows 10 / 11 or Linux.**
- **To build:** the CUDA toolkit 13.0 or newer, Visual Studio 2022's C++ tools (Windows) or GCC (Linux), Python 3.10 or
  newer, and Git.

## Install

```
git clone https://github.com/eddoursul/Lamina
cd Lamina
setup.bat --model iq3_s          (Windows)
./setup.sh --model iq3_s         (Linux)
```

Setup builds the engine and the image encoder for this PC's GPUs, downloads the model's files at pinned revisions and
checks them, prepares them, and writes a config and a launcher. `--model ud-q4_k_xl`, `--model qwen36-ud-q4_k_xl` and
`--model qwen38-27b-ud-q6_k_m` set up the others; `--check` says what the PC has and what a model needs, without
changing anything. Setup asks nothing and installs no system software: it names what is missing. Details:
[docs/SETUP.md](docs/SETUP.md).

## Run

```
run-iq3_s.bat                    (Windows)
./run-iq3_s.sh                   (Linux)
```

The model loads in a minute or so. Then:

- **http://127.0.0.1:8081** is the web app: chat with pictures, the monitor, settings.
- **http://127.0.0.1:8081/v1** is the API, for any OpenAI- or Anthropic-compatible client, coding agents such as
  Claude Code, Codex CLI, OpenCode and pi included. Chat completions, Messages, Responses and the models list are
  there, with tool calls, streaming and pictures.

How to connect clients, the config's keys, images, image generation, MCP tools and serving on a network:
[docs/USING.md](docs/USING.md).

## A fixed chat template

The server renders prompts with the chat template in the model's file. Lamina also carries
[froggeric's fixed template for Qwen 3.5, 3.6 and 3.8](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
(v22.5) in `data/chat-template-froggeric-v22.5.jinja`. To enable it, add this key to the model's config,
`lamina-<model>.json`, and start the server again:

```json
"chat_template_file": "data/chat-template-froggeric-v22.5.jinja"
```

A relative path starts at the config's `cwd`. The server's log then says
`[lamina] chat template: ...data/chat-template-froggeric-v22.5.jinja`. Remove the key to go back to the model's own
template. A conversation's history renders as it does under the model's template but for its system turn and the warnings
below, and a cached conversation is read again once. It has been tested on Qwen3.8-Flash-Next, on both model files.

What it changes:

- **The default effort.** A request that names no reasoning effort runs at `medium` instead of `xhigh`, with no effort
  sentence in the prompt, and thinks less. `high` still means `xhigh`. A config's `"reasoning_effort": "high"` keeps
  `xhigh` as the default.
- **The tool instructions.** The system turn says more about how to call tools. Under a coding agent's prompt the model
  invents fewer instructions when a conversation opens with a greeting, though not on every prompt
  (`"short_opening_tokens"` in [docs/USING.md](docs/USING.md#behaviour) covers that case on its own).
- **Tool errors.** A tool result the template takes for an error (from its first 120 characters: `"error":`,
  `command not found`, a non-zero exit code, ...) gets a `⚠️ SYSTEM WARNING` line telling the model to retry differently.
  A tool that prints such text without failing gets the line too.
- **Tags in a message.** `<|think_off|>`, `<|think_on|>`, `<|think_low|>`, `<|think_medium|>` and `<|think_xhigh|>` in a
  message set the thinking for that turn and the later ones, and are removed from the text.

## Documentation

- [docs/SETUP.md](docs/SETUP.md): setup, the configs, choosing the GPUs, Qwen3.6 and Qwen3.8-27B, Docker.
- [docs/USING.md](docs/USING.md): the server, its API and web app, clients, pictures, image generation, MCP tools.
- [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md): the engine.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): when something goes wrong.
- [docs/COMPARISON.md](docs/COMPARISON.md): speeds, memory and agreement with llama.cpp, against llama.cpp and
  Strata.
- [examples/](examples/README.md): the configs of the PC Lamina is developed on.
- [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md): reporting problems, pull requests, what the
  server exposes.

## License and credits

Lamina is under the MIT license ([LICENSE](LICENSE)), as Strata is. Some parts carry their own: `third_party/ggml`
(MIT, llama.cpp's), the web app's font, [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1,
`src/server/web/fonts/OFL.txt`), froggeric's chat template (Apache-2.0, unmodified,
`data/chat-template-froggeric-v22.5.LICENSE`), and what lamina-server takes from llama.cpp's checkout (its Jinja engine,
cpp-httplib, nlohmann/json, stb_image: MIT). It builds on:

- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors: the engine Lamina grew from, with ideas
  Strata took from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen).
- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp) (MIT): the CPU and CUDA kernels it shares, the prompt
  path's matrix products, the image encoder, the GGUF tools. Setup downloads it at a pinned commit.
- The models: [Qwen](https://huggingface.co/Qwen)'s, under their licenses, in the quantizations of
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) and [unsloth](https://huggingface.co/unsloth).
- The draft layer, [eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP):
  Qwen's MTP layer, its dense part trained on this engine's output, under the Qwen license.
