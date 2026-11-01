# How Lamina works

Qwen3.8-Flash-Next has 125 billion parameters, but each token passes through a small part of them: most of the model
is experts, and each token uses 10 of a layer's 512. Lamina keeps all of them in RAM, puts the ones the model uses most
on the GPUs, computes the rest on the CPU beside the GPUs, and checks several tokens in each pass of the model. This
page describes the parts; the options named here are in [SETUP.md](SETUP.md#engine-options), the measured speeds in
[COMPARISON.md](COMPARISON.md). The figures are from the development PC (RTX 3090 as the main GPU, RTX 5070 Ti and RTX
5060 Ti as tiers, Ryzen 7 9700X, DDR5-5600).

## The model

- **48 layers**: 36 Gated DeltaNet layers, whose memory of the text is a fixed-size state, and 12 attention layers
  (every fourth), whose indexer scores the context and picks 2,048 of its tokens for each query to attend to.
- **The residual stream** is four streams of 2,560 values (hyper-connections), which each layer reads and writes
  through small learned mixes.
- **Experts**: each layer has 512 routed experts and a shared one. A router picks 10 routed experts for each token; the
  48 layers hold 24,576 of them, about 47 GiB in IQ3_S and 72 GiB in UD-Q4_K_XL.
- **Per-layer embeddings**: an extra input to the layers, from a table of 320 million rows (28.8 GB). Each token reads
  16 of its rows, picked by hashes of the token and the two before it.
- **An MTP layer**, trained with the model to predict the token after the next: Lamina's draft layer.

## Where the parts live

- **The main GPU** runs everything each token passes through: the DeltaNet and attention layers, the hyper-connection
  mixes, the routers, the shared experts, the output head, the draft layer, the K/V cache of the attention layers and
  the DeltaNet states. The rest of its VRAM holds experts: on a 24 GB GPU about 9,200 of IQ3_S's, 5,600 of UD-Q4_K_XL's.
- **Further GPUs** (`--second-gpu`) hold the next experts in line, about 6,300 and 7,500 of IQ3_S's on the two 16 GB
  tiers, and compute them in each pass beside the CPU.
- **RAM** holds every expert, in one pinned block per layer that the GPUs can copy from. The CPU computes the experts
  no GPU holds, where they lie: a pool of worker threads, one per physical core but the one that drives the GPUs, with
  AVX2 or AVX-512 kernels (ggml's and the engine's own).
- **The SSD** holds the per-layer embeddings. Each token's 16 rows are read past the OS file cache, as soon as the token
  is known, and arrive while the GPU runs the embedding and the first layer; rows read before stay in a small cache.

On the development PC, with all three GPUs, 87-98% of the routed experts the tokens use are on a GPU (UD-Q4_K_XL
on an edit 87%, IQ3_S on thinking text 98%); the CPU computes the rest, from under one to six experts a layer.

## Which experts the GPUs hold

Routing is uneven: some experts serve far more tokens than others, and a conversation favours its own. The engine
starts from a profile that ranks all 24,576 experts by how often the model routes to them (`data/expert-profile.bin`,
or what this PC's runs learned: [SETUP.md](SETUP.md#the-config)); the main GPU takes the top of the list and each
tier the next.

While the model runs, the engine counts which experts the conversation routes to. Every round (`--adapt-every 1` in
the configs) each GPU's adaptive tier ranks the counts, older ones decayed, and replaces its least-used experts with
more-used ones it lacks, up to 96 an update. The copies go out while the GPU's link carries nothing else: between
layers for the main GPU, between passes for the tiers. A tier counts an expert that a later tier holds at half its use,
since moving it would only shift work from one GPU to another.

The first tier also gets the next layer's likely experts ahead of time: on a side branch of each layer, the next
layer's router runs on an estimate of its input, and while the RAM is idle the likeliest experts no GPU holds are
copied to that tier, up to 12 MiB a layer (to a tier whose link moves 20 GB/s or more).

## Generating: verify windows

Each step of generation is one pass of the whole model over a **verify window**: the last token and the drafts after
it, up to 4 tokens in the configs (`--spec 4`), up to 8 with prompt lookup. Each window size is a CUDA graph, captured
once. For each layer:

1. The main GPU runs the layer's mixer and its router, and writes the router's choices and the layer's input, in
   8-bit form, to host memory that the CPU and the GPUs all reach.
2. The host splits the routed experts: those the main GPU holds, each tier's, and the rest for the CPU pool.
3. All of them compute at once: the main GPU its experts, each tier its share (its rows are copied over to the main
   GPU when they are done), the CPU pool the rest, writing its rows to host memory.
4. The main GPU waits for the others' rows, adds them to its own and goes on with the next layer.

The main GPU waits for the host and the other devices by spinning on flags in host memory: its graph never stops, so
the hand-offs cost microseconds. It is also why nothing else may use that GPU: a display or another program makes the
driver share it in time slices, and a spin that loses its slice stalls the pass.

After the last layer, the output head gives each token of the window its next-token distribution. The window's tokens
are accepted from the first while each draft is the token the model picks there (greedy: the likeliest; sampled: the
one drawn from the model's distribution), and the model's own token follows the last accepted draft. So a pass yields
one token or several, and every token is the one the model itself picks at its position. A pass over several tokens
takes far less time than as many passes over one: the weights are read once for all of them.

## Drafts

- **The draft layer** is the model's MTP layer, run on the last token's final state and its own output: it proposes up
  to 3 tokens, each kept while its probability under the draft layer stays at 0.5 or more (`--spec-min-p 0.5`). Its
  experts are 2-bit, it attends to the last 32,768 tokens, and its head covers a fixed subset of 40,525 tokens of the
  vocabulary (it never drafts the others), scored through a 2-bit copy and the best ones again exactly. Its dense part
  was trained further on this engine's own output, so its drafts are right more often
  ([eddoursul/Lamina-Qwen3.8-Flash-Next-MTP](https://huggingface.co/eddoursul/Lamina-Qwen3.8-Flash-Next-MTP)).
- **Prompt lookup**: where the last 16 tokens of the text occurred before (`--spec-lookup 16`), up to 7 of the tokens
  that followed them there take the drafts' place. An answer that quotes its prompt, such as a file returned with an
  edit, then passes about 7.5 tokens a step.

On the development PC a greedy pass yields 2.0-2.4 tokens on thinking text, 3.8-4.1 on code and 7.5 on an edit;
sampled requests keep fewer drafts (thinking: 1.9-2.1).

## Reading a prompt

A prompt goes through the model in chunks of 16,384 tokens (`--prefill`), each chunk through all 48 layers before the
next. The dense layers run as matrix products on the main GPU, the experts through llama.cpp's quantized matrix
kernels: those the main GPU holds where they are; with a tier, the first tier computes the experts the main GPU lacks,
the chunk's input sent to it and the sums back in pieces of 2,048 tokens, so neither GPU waits for the other's whole
share. Without a tier, the main GPU streams them from RAM, ahead of use. In a chunk each expert serves about 320
tokens on average, so every expert is worth bringing to a GPU; the CPU computes none in a prompt. The buffers a chunk
needs borrow the expert caches' last slots, refilled after the prompt.

A short new part of a prompt, up to 46 tokens in the configs (`--feed-max`), runs through verify windows instead,
where the batched path's fixed cost would be the larger.

`--prompt-split L` runs a prompt of several chunks in two stages: the first tier runs each chunk's layers from L on
while the main GPU runs the next chunk's first layers. On the development PC it reads prompts of 32K-128K tokens
1.1-1.6x as fast, but its layer and its gain depend on the GPUs and their links, so it is off unless a config sets it.

## The context

The 12 attention layers keep keys and values for every token: in 8 bits (`--kv int8`, 3.1 GiB at 262,144 tokens), 8-bit
keys and 4-bit values (`k8v4`, 2.4 GiB) or 4 bits (`q4_0`, 1.7 GiB). The 4-bit formats rotate each head's values
(Walsh-Hadamard) before rounding them, and the attention kernels read every format as stored. With `--kv-grow` the
K/V takes VRAM only for the tokens the requests reach: the expert cache holds the rest and gives slots back as a
conversation grows. The DeltaNet layers' states have one size at any length.

The configs allow 262,144 tokens, the context the model was trained for; YaRN (`--rope-scaling yarn`) extends it.

## The prompt cache

A request continues from the longest start of its prompt that the engine still holds:

- **The live conversation**: a follow-up turn reads only its new tokens.
- **Checkpoints** of the sequence's fixed-size state (the DeltaNet states, the indexer's and the embeddings' recent
  history: 113 MiB each, up to 6 per conversation by default, in pinned RAM), taken before the last turn of a prompt,
  at its end, and along long prompts. Each conversation owns its chain: over the cap its own oldest leaf leaves
  (the root, the point every prompt shares, stays). The K/V is indexed by position, so going back to a checkpoint
  copies only the state: a client that rewrites the end of its history (an agent's last message, a regenerated
  answer) is read again from the checkpoint before it.
- **Conversations parked in RAM**: one that another request replaces moves to RAM with its K/V and its own
  checkpoints (`--conversation-cache-mib`, `--conversation-cache-slots`), and a request that continues it later
  takes it back. One that shares only its start with it (subagents of one agent, sessions with one system prompt)
  borrows that start. A parked conversation leaves only whole - least recently used first, never while it lends
  its start - so a chatty conversation can never drain a parked one's memory out from under it.

The API reports how many prompt tokens were reused (`cached_tokens`, `cache_read_input_tokens`).

## Pictures

The image encoder, `lamina-vision`, is llama.cpp's multimodal library with the model's projector, run on the CPU in a
process of its own: the GPUs keep their VRAM for experts. The server sends it each picture and passes the embeddings
it returns to the engine, which places them in the prompt with their positions in the picture's grid.

## The server

`lamina-server` serves the OpenAI, Anthropic and Responses APIs and the web app, which is built into it. It renders
the model's chat template (from its GGUF, or the config's `chat_template_file`, with llama.cpp's Jinja engine) and tokenizes with the model's tokenizer (from
the GGUF too), and runs the engine on a thread of its own process: the engine takes prompts as token ids and writes
tokens as each pass accepts them, through in-process channels that carry the line protocol `lamina --serve` speaks on
stdin and stdout. The server turns the tokens back into text, reasoning and tool calls, streamed. Flash-Next runs one
request at a time, the others wait; when the engine stops, the server starts it again. With replicas (opt-in) each
engine is a `lamina --serve` child on its own GPUs.

## Exactness

The experts are computed with llama.cpp's arithmetic: ggml's CPU and CUDA kernels, and kernels of the engine's own
that sum in another order. So Lamina's next-token distributions match llama.cpp's about as closely as llama.cpp's CPU
and CUDA backends match each other ([COMPARISON.md](COMPARISON.md#agreement-with-llamacpp)). Where an expert is
computed changes the rounding a little, and the adaptive tiers move experts as their copies land, so two runs of one
greedy request can part at a near-tie. Runs repeat bitwise with `--adapt-every 0`, or with `--deterministic-adapt`
(each update settled before the next, a little slower); `--expert-math exact` (K-quant files) also makes the output
independent of where each expert is computed.

## Qwen3.6-35B-A3B and Qwen3.8-27B

Qwen3.6-35B-A3B and Qwen3.8-27B, a dense model of the same family, run on a path of their own (`src/qwen35`), chosen
from the model file's header. It reads the GGUF as it is and drafts with the file's own MTP layer. With `--second-gpu`
it splits the layers over the fewest GPUs that hold the whole model, each running its layers and handing the
activations to the next; where none do, or on one GPU, it keeps every expert that fits in VRAM and computes the rest on
the CPU. With the config's `parallel` it runs several requests at once on one GPU
([SETUP.md](SETUP.md#qwen36-35b-a3b-and-qwen38-27b)).
