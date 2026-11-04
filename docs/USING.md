# Using Lamina

The server, `lamina-server`, runs the engine in its own process and serves the OpenAI, Anthropic and Responses APIs and
a web app. The launchers setup writes start it with the model's config (`run-iq3_s.bat`, or from the clone's folder
`build\lamina-server.exe --config lamina-iq3_s.json`). It listens on the config's `port` (setup's defaults: 8081 for
IQ3_S, 8082 for UD-Q4_K_XL, 8083 for Qwen3.6-35B-A3B, 8084 for Qwen3.8-27B; `--port` overrides it), on this PC only
unless told otherwise ([below](#from-other-devices)). Ctrl+C in its window stops the engine, which saves its learned
expert profile, and the server.

## The web app

`http://127.0.0.1:8081/`: **Chat** (with pictures, the thinking level, MCP tools, chats kept in the browser),
**Monitor** (what the engine is doing, the last requests, the GPUs, CPU and RAM) and **About** (the addresses to give
clients, the settings). Its **Model settings** card changes a few keys of the config ([below](#the-config)) from the next start on;
only the page itself can save them.

## The API

| | Endpoint |
| --- | --- |
| OpenAI Chat Completions (stream and non-stream, tools) | `POST /v1/chat/completions` |
| Anthropic Messages (stream and non-stream, tools) | `POST /v1/messages`, `POST /v1/messages/count_tokens` |
| OpenAI Responses (stream and non-stream, tools; stateless, [below](#codex-cli-and-the-responses-api)) | `POST /v1/responses` |
| A choice among single-token options in one pass, nothing generated ([below](#one-pass-choices)) | `POST /v1/choose` |
| The model list, health | `GET /v1/models`, `GET /models`, `GET /health` |
| The model's properties (llama-server's fields) | `GET /props` |
| What the model is doing now | `GET /status`, `GET /slots` |
| Everything the Monitor shows; Prometheus' format with vLLM's names on `Accept: text/plain` | `GET /metrics` |
| Request totals in Prometheus' format under llama-server's names | `GET /metrics/prometheus` |
| The MCP servers, their state and tools | `GET /mcp` |
| The prompt a chat request is read as; a text's token ids and back (llama-server's fields) | `POST /apply-template`, `POST /tokenize`, `POST /detokenize` |
| Image generation, when set up ([below](#image-generation)) | `POST /v1/images/generations` |
| Unload the model now, or load it ahead of a request | `POST /unload`, `POST /load` |

```bash
curl http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}], "max_tokens": 512}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="none")
r = client.chat.completions.create(model="local", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

Any model name is served; `/v1/models` lists the config's `model_name` and its `aliases`. An answer's `timings` use
llama.cpp's names, with the drafts offered and accepted. Prometheus' scrape of `/metrics` carries vLLM's metric names
(`vllm:num_requests_running`, the token and request counters, the draft counters, time to first token and between
tokens), so dashboards written for vLLM read this server; the Monitor's other facts come under `lamina:` and their
JSON key (`lamina:live_tok_s`, `lamina:gpu_util`, ...). Ready to use in [monitoring/](monitoring/): a Prometheus scrape
config (`prometheus.yml`, the API key sent as a bearer token), a Prometheus Operator `ServiceMonitor` and a Grafana
dashboard of both name sets (`grafana-lamina.json`).

## Clients

- **Apps with an OpenAI-compatible provider**: base URL `http://127.0.0.1:8081/v1`, any API key (or the config's
  `api_key`). Zed's llama.cpp provider reads `/props` and offers tools from its `chat_template_caps`.
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8081`, `ANTHROPIC_MODEL` set to a Claude model name it knows
  (the server ignores the name), and any `ANTHROPIC_AUTH_TOKEN` (or the `api_key`).
- **OpenCode**: a provider in `opencode.jsonc`, with the context setup chose:

  ```jsonc
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "lamina": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Lamina",
        "options": { "baseURL": "http://127.0.0.1:8081/v1", "apiKey": "none" },
        "models": {
          "flash-next": {
            "name": "Qwen3.8-Flash-Next",
            "limit": { "context": 262144, "output": 32768 },
            "options": { "reasoningEffort": "high" },
            "variants": { "low": { "reasoningEffort": "low" }, "none": { "reasoningEffort": "none" } }
          }
        }
      }
    },
    "model": "lamina/flash-next"
  }
  ```

  OpenCode compacts the conversation before `limit.context`; keep `limit.output` well under it.
- **Codex CLI**: [below](#codex-cli-and-the-responses-api).
- **A terminal chat**: `.venv\Scripts\python.exe chat.py --port 8081` (`/image <path>` attaches a picture, `/think`
  sets the thinking level, `/reset` starts over).

## Thinking

The model thinks before it answers (streamed as `reasoning_content`; Anthropic: `thinking` blocks). Choose how much
per request:

| API | |
| --- | --- |
| OpenAI | `"reasoning_effort": "none" \| "low" \| "medium" \| "high"` (also `"reasoning": {"effort": ...}`, or `"chat_template_kwargs": {"enable_thinking": false}`) |
| Anthropic | `"output_config": {"effort": "low" \| "medium" \| "high"}`, `"thinking": {"type": "disabled"}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` (under 2K = low, under 8K = medium, more = high) |

Without a setting the model uses its own default, high; `"reasoning_effort"` in the config sets another for requests
that name none. The levels are instructions the model was trained with, not a token limit:

- **A thinking budget**: `"reasoning_budget_tokens": N` in a request or the config caps the thinking at N tokens; the
  server then ends it with a short wrap-up and `</think>`, and the model answers from there (nothing is read again).
- **The answer keeps its room**: with thinking on, no budget and a `max_tokens` of 1,024 or more, the thinking is
  wrapped up when a quarter of the cap (at least 512 tokens) is left; without `max_tokens`, at three quarters of the
  context. `"answer_room": false` in the config turns it off.
- **`"effort_position": "end"`** in the config puts a non-default effort in a short system turn before the answer
  instead of the top of the prompt, so a request that only changes the effort reuses the cached conversation.
- **`"anthropic_thinking": "on_request"`**: an Anthropic request that asks for no thinking gets none (Anthropic's rule;
  Claude Code's short helper calls want it). The default thinks as the model's template does.

## Sampling

A request's `temperature`, `top_p`, `top_k`, `min_p`, `seed`, stop strings (`stop`; Anthropic `stop_sequences`) and
penalties (`presence_penalty`, `frequency_penalty`, `repetition_penalty` with `penalty_last_n`; llama.cpp's
`repeat_penalty` and `repeat_last_n` too) are honored. The config's `sampling` block sets the defaults for requests that
leave them out; what it leaves out comes from the model card's settings for the request's thinking state (thinking:
temperature 1.0, top_p 0.95, top_k 20; not thinking: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5).
The example configs sample at temperature 0.7, top_p 0.95, top_k 20, min_p 0, presence_penalty 0. `"temperature": 0`
decodes greedy. `top_k` keeps at most 64 candidates.

Every token, drafted or not, is the model's own pick: with sampling, a draft is kept only when it equals the token the
model sampled for its position, so the drafts change the speed, never the distribution. The expert tiers adapt to the
conversation in the background, so a sampled answer is reproducible run to run only with `--adapt-every 0` in the
engine's arguments.

## Behaviour

- **Streaming**: with `"stream": true` the thinking, the answer and tool calls (name first, then the arguments piece by
  piece) arrive as they are made; a long prompt sends keep-alives so agents do not time out, and `"return_progress":
  true` sends llama.cpp's `prompt_progress` (total, cache, processed, time_ms) per prompt chunk instead. Closing the
  connection stops the model, so the next request starts at once.
- **Tool calls** are taken out of the text when the request declares tools; without them the answer is the model's
  text as written. `"tool_call_recovery": true` in the config also reads calls the model writes in a few other forms
  (`<parameter=NAME>`, JSON inside `<tool_call>`, a bare `<function=NAME>` block).
- **Context**: a request longer than the context is refused (400), never cut. One whose `max_tokens` would run past the
  context is answered with `max_tokens` shortened to the room left (agents ask for 32,000 on every request);
  `"fit_max_tokens": false` in the config (or `--no-fit-max-tokens`) gives llama.cpp's 400 instead.
- **The prompt cache**: a request continues from the longest start of its prompt the engine holds, so the next turn of a
  chat or an agent reads only what is new. Each conversation keeps its own checkpoints of the sequence state in RAM
  (at each turn and in long prompts; `--prompt-cache N`, default 16 per conversation), and a conversation another
  request replaces (a chat app's title request, a subagent) is parked in RAM with its K/V and its checkpoints
  (`--conversation-cache-mib MIB`, default 16384; `--conversation-cache-slots N` caps their number, by default only
  the memory does) so switching back continues where it was. A parked conversation leaves only whole, the least
  recently used first; `--conversation-cache-min-tokens N` (default 1024) keeps shorter ones out. The Monitor's
  Conversation cache card shows them. The engine keeps the model's own token ids of each reply for the next turn.
  Responses report the reused tokens
  (`usage.prompt_tokens_details.cached_tokens`; Anthropic: `cache_read_input_tokens`).
- **Guards**: a reply that repeats one token 256 times ends as `"length"` (`"repeat_stop_tokens"` sets the run, 0 turns
  it off); a verify window whose logits are not finite emits nothing, the engine drops its cached state and the server
  retries the request once if nothing was sent yet. `"reasoning_close_retry": true` closes a thinking that ended with
  no answer and continues once; `"reasoning_loop_recovery": "stop"` or `"recover"` ends or restarts a reply whose
  reasoning repeats whole passages. A request the engine stays silent on for `"engine_silence_s"` (default 300) ends.
  The model writes `<|im_end|>` and `<|endoftext|>` as the one token that ends a turn, also when it means to quote them:
  drawn inside a code span or fence, after an opening quote, in a tool call's open parameter or in the middle of its
  thinking, the token becomes the marker's text and the reply goes on, up to 64 times a reply (`"literal_markers": false`
  ends the turn there).
- **A greeting**: a tool-using agent's prompt and an opening message with no task ("hello") can make the model invent
  instructions in its thinking, and act on them, at the top reasoning effort (on llama.cpp too).
  `"short_opening_tokens": 4` reads a conversation's first message of up to that many tokens as at `medium`, which has no
  effort sentence; the next turn's prompt starts differently, so it is read again.
- **JSON output**: `response_format` with `json_object` or `json_schema` (an object at the schema's root) asks the model
  for JSON and validates the answer: one generation, no hidden retry, a 502 (`structured_output_failed`) when it does
  not hold. The schema is checked with the `jsonschema` package when it is installed; without it, only that the answer
  is one JSON object. Prompting and validation, not constrained decoding.
- **Model aliases**: `"aliases": ["qwen", "local-model"]` lists the model under those names too.
- **`"prevent_sleep": true`** keeps Windows from sleeping while a request runs or waits.

## One-pass choices

`POST /v1/choose` answers a multiple-choice question without generating: the chat is rendered as for
`/v1/chat/completions` with thinking off (so it shares a conversation's cached prompt), `answer_prefix` (optional)
follows the assistant header, and each option's probability is read from the head's logits there. Options must be
single tokens (letters are; after a prefix ending in a word, `" A"` is one token and `"A"` another):

```bash
curl http://127.0.0.1:8081/v1/choose -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": "Is Paris the capital of France? A) yes B) no. Answer with the letter."}],
  "options": ["A", "B"]}'
# {"probs": {"A": 0.99988, "B": 0.00012}, "logprobs": {"A": -0.0009, "B": -9.005}, "mass": 0.9992, "choice": "A",
#  "confidence": 0.99988, "prompt_tokens": 31, "reused": 0, "seconds": 0.71}
```

`probs` are renormalized over the options; `mass` is what they got of everything the model could have said there.
Not on the path of Qwen3.6-35B-A3B and Qwen3.8-27B yet.

## Codex CLI and the Responses API

`POST /v1/responses` takes OpenAI's Responses requests, on the same path as chat completions: `input` as text or items
(messages, reasoning, function and custom tool calls and their outputs), `instructions`, `tools` (function, namespace
and custom tools; hosted tools such as `web_search` are left out), `reasoning.effort`, `max_output_tokens`,
`text.format` (the JSON formats above), and streaming with Responses' events. It is stateless: nothing is stored, so
the client sends the whole conversation every time (Codex does), and `previous_response_id`, `conversation`,
`background` and the retrieve, delete and cancel endpoints are refused. The thinking comes back as a `reasoning` item;
sending it back with the conversation keeps the prompt cache.

**Codex CLI**, in `~/.codex/config.toml` (Windows: `%USERPROFILE%\.codex\config.toml`):

```toml
model = "qwen3.8-flash-next"
model_provider = "lamina"

[model_providers.lamina]
name = "Lamina"
base_url = "http://127.0.0.1:8081/v1"
wire_api = "responses"
# env_key = "LAMINA_API_KEY"    # only with an api_key: the variable holding it
```

Codex warns that it lacks the model's metadata; that is only the warning. `"codex_compaction_cache": true` in the
config renders Codex's compaction requests with the conversation's tools, so they reuse the cached conversation;
`"codex_thread_titles": true` answers Codex's thread-title requests without the engine.

## Pictures

With pictures set up (setup's default), OpenAI `image_url` parts and Anthropic `image` blocks (base64 data, http(s)
URLs or local file paths) go through the image encoder, `lamina-vision`, and reach the model as up to 1,024 tokens each
(the `vision` section's `max_tokens`; a 640x480 photo is ~300). JPEG, PNG, BMP and GIF are read as they are; WebP,
TIFF, AVIF and others are converted to PNG first. The encoder runs on the CPU (the `vision` section's `threads`), so
the GPUs keep their VRAM for experts: on the development PC's 8-core Ryzen 7 9700X a picture of ~1,000 tokens takes
~9 s, once per picture (a picture sent again, as chat apps do every turn, is not encoded again). Set up with
`--vision gpu`, it runs on a GPU instead (the section's `gpu` and `cuda_device`): on a PC with three cards, an RTX 5060
Ti as the last expert tier gave ~1.3 GB of its experts to it and read a picture of ~1,000 tokens in ~1.9 s instead of
~8.3 s on the CPU. `--offline` refuses URLs. Qwen3.6-35B-A3B and Qwen3.8-27B read pictures with their repositories' own projectors.

## Image generation

Optional: [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) in
[unsloth's GGUF quants](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF), run by
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s `sd-cli`. The model is under the Qwen
Research License, not Apache 2.0: read [its license](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/LICENSE)
before using the images commercially.

```
.venv\Scripts\python.exe tools\image\install.py --config lamina-iq3_s.json
```

compiles `sd-cli` at a pinned commit for this PC's GPUs (`--backend cpu` for the CPU), downloads three files into the
data folder's `models/qwen-image-2.1-turbo/`, checked by SHA-256 (the denoiser by the GPU's VRAM: Q4_K_M up to 8 GB,
Q6_K up to 12 GB, Q8_0 above, 4.2-7.6 GB; the text encoder 5.1 GB; the VAE 0.7 GB; `--quant` picks), and writes an
`images` section into the config (`--print` shows it instead).

`POST /v1/images/generations` takes OpenAI's request: `prompt`, `size` (`WIDTHxHEIGHT`, each side a multiple of 32 from
256 to 2048; default 1024x1024), `n` (1-4), `response_format` (`b64_json`, or `url`: the server keeps the newest 64),
plus `seed` (the answer says which it used) and `steps` (8, Turbo's schedule). A job takes the server's turn on the GPU:
a chat request waits for the image and the other way round. The section's `gpu_policy` keeps the engine loaded beside
the job (`keep`), stops it for the job (`unload`: the next request starts it again, or the job's end with
`"reload": "eager"`, and reads its conversation again), or picks by the free VRAM (`auto`, the default).

## MCP tools

MCP servers named in the config's `mcp_servers` (Claude Desktop's format; or a file with `--mcp-config`) give the web
app's chat their tools: the server starts them (or connects to them over Streamable HTTP), lists them at `GET /mcp`, and
runs the calls the model makes. Only requests that ask for them get them (`"lamina_mcp": true`, which the web app
sends); other clients see the plain API. `--offline` starts none.

## Sharing the GPU

On a PC that also games or runs another model server, these keys (also server options) give the VRAM back:

| Config key | Option | |
| --- | --- | --- |
| `"idle_unload_s": 600` | `--idle-unload 600` | unload the model after 600 s without requests; the next request loads it again |
| `"min_free_vram_mib": 11000` | `--min-free-vram-mib 11000` | load an unloaded model only when that much VRAM is free (waiting up to 15 s), else answer 503 |
| `"before_load": "cmd"` | `--before-load "cmd"` | a command run before the model loads again (one that unloads another server's model); `before_load_timeout_s`, `before_load_strict`, `before_load_ignore_failure` say what a failure does |
| `"lazy_load": true` | `--lazy` | start the API unloaded; the first request loads the model |

`POST /unload` unloads it now (`?wait=1` waits for a running request first) and `POST /load` loads it, both with
`Content-Type: application/json`. Unloading ends the engine and the image encoder, so their VRAM and RAM go back (the
server keeps each GPU's CUDA context: about 200 MiB on the main GPU, less on the others); the model files stay in the
OS file cache, so loading again is quick while that RAM is not needed elsewhere.

**Behind [llama-swap](https://github.com/mostlygeek/llama-swap)**, which starts the server a request names and stops
the others: run the server itself with llama-swap's port, so that stopping it stops the engine too
(`cmd: sh -c 'cd /path/to/Lamina && exec build/lamina-server --config lamina-iq3_s.json --port ${PORT}'`),
with `checkEndpoint: /health` (the server listens once the model has loaded) and a `healthCheckTimeout` of a few
minutes. llama-swap passes the client's `Host` through: without an API key, add the name clients reach it by to
`allowed_hosts` ([below](#from-other-devices)), or those requests get 403.

## Replicas

`"replicas": N` with a `"gpu"` list cuts the GPUs into N groups and runs one engine on each, the requests spread over
them (or `"replicas": [{"gpus": [0, 1]}, {"gpus": [2, 3]}]`). Each engine pins its own experts in RAM.

## From other devices

The server listens on this PC only (`127.0.0.1`). `"host": "0.0.0.0"` in the config (or `--host`) opens it to the
network, with an API key: `"api_key": "a-long-secret"` (or the `LAMINA_API_KEY` variable; several keys
comma-separated or as a list), which clients then send as their API key. The server window prints this PC's addresses;
on Windows the firewall blocks them until lamina-server is allowed on private networks. Through a tunnel or a proxy, set the
key first, and add the address the page is reached at to `"trusted_origins"` for the web app's settings and MCP tools.
A request without the right key is refused before its body is read. The server has no deadline for a request's
headers, though: a client that sends them a byte at a time holds one of its threads, so on a network you do not trust,
put a reverse proxy in front of it.

- **Browsers (CORS)**: off by default; `"cors_origins": ["https://chat.example.com"]` lets those pages call `/v1/*`.
- **Host names**: without an API key the server answers only `Host` names it knows (localhost, IP addresses, this PC's
  name when it listens beyond it), against DNS rebinding; `"allowed_hosts"` (or `LAMINA_ALLOWED_HOSTS`) adds others.
- **Web pages without a key**: a browser's `POST` to `/v1/*` from a page other than the server's own, localhost, an
  allowed host or a trusted origin gets 403; clients that send no `Origin` (curl, the SDKs) are not affected.

## The config

The server reads `lamina-<model>.json` at its start. Setup's keys: [SETUP.md](SETUP.md#the-config). The server's:

| Key | |
| --- | --- |
| `port`, `host`, `api_key` | where it listens and the key it asks for |
| `cors_origins`, `trusted_origins`, `allowed_hosts` | browsers and host names ([above](#from-other-devices)) |
| `sampling`, `reasoning_effort`, `reasoning_budget_tokens`, `answer_room` | the defaults for requests ([above](#thinking)) |
| `fit_max_tokens`, `anthropic_thinking`, `effort_position`, `aliases` | [above](#behaviour) |
| `tool_call_recovery`, `repeat_stop_tokens`, `reasoning_close_retry`, `literal_markers`, `short_opening_tokens`, `reasoning_loop_recovery`, `engine_silence_s` | [above](#behaviour) |
| `chat_template_file` | a chat template file in place of the model file's ([the README](../README.md#a-fixed-chat-template)) |
| `codex_compaction_cache`, `codex_thread_titles`, `env_to_user` | Codex's compaction and titles; opencode's `<env>` block moved to the first user message, so the system turn stays the same between sessions |
| `idle_unload_s`, `min_free_vram_mib`, `before_load`, `lazy_load` | [sharing the GPU](#sharing-the-gpu) |
| `mcp_servers`, `images`, `replicas`, `parallel` | MCP tools, image generation, replicas; `parallel`: the batch slots of Qwen3.6 and Qwen3.8-27B (`--batch`, one GPU) |
| `api_monitor`, `open_browser`, `prevent_sleep` | the request monitor at `/api-monitor` (the last 100 prompts and answers in memory); a browser at the start; Windows awake while busy |

The web app's Model settings card changes `sampling`, `reasoning_budget_tokens`, `fit_max_tokens`,
`anthropic_thinking`, `effort_position`, `aliases`, `idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`,
`open_browser` and the engine's `--vram-reserve-mib`, and keeps the earlier file as `<config>.bak`.
