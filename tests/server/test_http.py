"""lamina-server's HTTP behavior on the scripted (mock) engine, against the recorded answers in http_golden.json.

The requests cover the OpenAI, Anthropic and Responses APIs (streamed and not, tools, structured output, thinking
budgets, stop strings, the guards), the errors, the security checks (Host, Origin, API keys) and the endpoints the web
app and the Monitor read. Ids, timestamps and timings are normalized. The answers were recorded from lamina-server once
it answered as the Python server it was ported from did, request by request.

    set LAMINA_TEST_GGUF=<a Qwen3.8-Flash-Next GGUF (its first shard)>   the tokenizer and chat template
    python -m unittest discover -s tests/server            (from the clone's folder; skipped without the variable)
    python tests/server/test_http.py --record               writes http_golden.json from the server as it is now
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GOLDEN = HERE / "http_golden.json"


def server_exe() -> Path:
    exe = os.environ.get("LAMINA_SERVER")
    if exe:
        return Path(exe)
    return ROOT / "build" / ("lamina-server.exe" if os.name == "nt" else "lamina-server")


SCRIPTS = [
    "Let me think about it.</think>\n\nHello there, this is the answer.",
    "Checking the weather.</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "</function>\n</tool_call>",
    "</think>\n\n{\"name\": \"Ada\", \"age\": 36}",
    "Short thought.</think>\n\nOne. Two. STOP here. Three.",
    "Plain answer without thinking markers.",
    "Thinking.</think>\n\nThe file says: <think> is a tag and </tool_call> too.",
    "Thinking.</think>\n\n" + " a" * 300,
    "Two calls.</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n"
    "</tool_call>\n<tool_call>\n<function=get_time>\n<parameter=zone>\nUTC\n</parameter>\n</function>\n</tool_call>",
]

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}},
         {"type": "function", "function": {"name": "get_time", "description": "Time in a zone",
                                           "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}}}}]
A_TOOLS = [{"name": "get_weather", "description": "Weather of a city",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]
R_TOOLS = [{"type": "function", "name": "get_weather", "description": "Weather of a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]


def msgs(text="Hi"):
    return [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": text}]


def chat(**kw):
    return {"model": "m", "messages": msgs(), **kw}


CASES = []   # (name, method, path, body or None, headers)


def case(name, method, path, body=None, headers=None):
    CASES.append((name, method, path, body, headers or {}))


# every request consumes one script (by position), so each case runs once per script index used
for i in range(len(SCRIPTS)):
    case(f"chat-{i}", "POST", "/v1/chat/completions", chat(tools=TOOLS))
    case(f"chat-stream-{i}", "POST", "/v1/chat/completions", chat(tools=TOOLS, stream=True))
    case(f"messages-{i}", "POST", "/v1/messages", {"model": "m", "max_tokens": 200, "system": "Be brief.",
                                                     "messages": [{"role": "user", "content": "Hi"}], "tools": A_TOOLS})
    case(f"messages-stream-{i}", "POST", "/v1/messages", {"model": "m", "max_tokens": 200, "stream": True,
                                                            "messages": [{"role": "user", "content": "Hi"}], "tools": A_TOOLS})
    case(f"responses-{i}", "POST", "/v1/responses", {"model": "m", "input": "Hi", "tools": R_TOOLS})
    case(f"responses-stream-{i}", "POST", "/v1/responses", {"model": "m", "input": "Hi", "stream": True, "tools": R_TOOLS})
case("chat-stop", "POST", "/v1/chat/completions", chat(stop=["STOP"]))
case("chat-json", "POST", "/v1/chat/completions", chat(response_format={"type": "json_object"}))
case("chat-max-tokens", "POST", "/v1/chat/completions", chat(max_tokens=3))
case("chat-no-think", "POST", "/v1/chat/completions", chat(chat_template_kwargs={"enable_thinking": False}))
case("chat-effort", "POST", "/v1/chat/completions", chat(reasoning_effort="low"))
case("chat-forced", "POST", "/v1/chat/completions", chat(tools=TOOLS, tool_choice={"type": "function",
                                                                                      "function": {"name": "get_weather"}}))
case("chat-required", "POST", "/v1/chat/completions", chat(tools=TOOLS, tool_choice="required"))
case("chat-history", "POST", "/v1/chat/completions", {"model": "m", "messages": msgs() + [
    {"role": "assistant", "content": "Hello", "reasoning_content": "thinking"},
    {"role": "user", "content": "And now?"}]})
case("chat-tool-result", "POST", "/v1/chat/completions", {"model": "m", "tools": TOOLS, "messages": msgs() + [
    {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "Sunny"}]})
case("count-tokens", "POST", "/v1/messages/count_tokens", {"model": "m", "messages": [{"role": "user", "content": "Hi"}]})
case("chat-budget", "POST", "/v1/chat/completions", chat(reasoning_budget_tokens=3))
case("chat-budget-stream", "POST", "/v1/chat/completions", chat(reasoning_budget_tokens=2, stream=True))
case("chat-answer-room", "POST", "/v1/chat/completions", chat(max_tokens=1024))
case("chat-small-max", "POST", "/v1/chat/completions", chat(max_tokens=12))
case("messages-budget", "POST", "/v1/messages", {"model": "m", "max_tokens": 300, "thinking": {"type": "enabled", "budget_tokens": 4},
                                                  "messages": [{"role": "user", "content": "Hi"}]})
case("messages-stop", "POST", "/v1/messages", {"model": "m", "max_tokens": 300, "stop_sequences": ["Two"],
                                                "messages": [{"role": "user", "content": "Hi"}]})
case("messages-no-think", "POST", "/v1/messages", {"model": "m", "max_tokens": 300, "thinking": {"type": "disabled"},
                                                    "messages": [{"role": "user", "content": "Hi"}]})
case("chat-stop-stream", "POST", "/v1/chat/completions", chat(stop="Two", stream=True))
case("chat-seed-temp", "POST", "/v1/chat/completions", chat(temperature=0.5, top_p=0.9, top_k=10, seed=7, presence_penalty=0.5))
case("chat-image-no-vision", "POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is this?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]})
case("chat-lamina-prefix", "POST", "/v1/chat/completions", {**chat(), "lamina_prefix": {"messages": 1}})
case("chat-lamina-prefix-bad", "POST", "/v1/chat/completions", {**chat(), "lamina_prefix": {"messages": 0}})
case("responses-json", "POST", "/v1/responses", {"model": "m", "input": "Give JSON", "text": {"format": {"type": "json_object"}}})
case("chat-json-schema", "POST", "/v1/chat/completions", chat(response_format={"type": "json_schema", "json_schema": {
    "name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                            "required": ["name", "age"]}}}))
# errors (no script used)
case("bad-json", "POST", "/v1/chat/completions", "{not json")
case("not-object", "POST", "/v1/chat/completions", "[1, 2]")
case("too-deep", "POST", "/v1/chat/completions", "[" * 1001 + "]" * 1001)   # past Python's recursion limit
case("deep", "POST", "/v1/chat/completions", '{"model": "m", "messages": ' + "[" * 998 + "]" * 998 + "}")
case("bad-messages", "POST", "/v1/chat/completions", {"model": "m", "messages": 5})
case("bad-temperature", "POST", "/v1/chat/completions", chat(temperature="hot"))
case("bad-stop", "POST", "/v1/chat/completions", chat(stop=5))
case("bad-effort", "POST", "/v1/chat/completions", chat(reasoning_effort="extreme"))
case("bad-budget", "POST", "/v1/chat/completions", chat(reasoning_budget_tokens=-3))
case("bad-tool-choice", "POST", "/v1/chat/completions", chat(tools=TOOLS, tool_choice={"type": "function",
                                                                                          "function": {"name": "nope"}}))
case("bad-logit-bias", "POST", "/v1/chat/completions", chat(logit_bias={"x": 1}))
case("logit-bias", "POST", "/v1/chat/completions", chat(logit_bias={"5": 1}))
case("too-long", "POST", "/v1/chat/completions", chat(max_tokens=10 ** 7, fit_max_tokens=False))
case("responses-retrieve", "POST", "/v1/responses/resp_x", {"model": "m"})
case("images-off", "POST", "/v1/images/generations", {"prompt": "a cat"})
case("unknown-post", "POST", "/v1/unknown", {"x": 1})
case("form-page", "POST", "/v1/chat/completions", chat(), {"Origin": "https://evil.example.com", "Content-Type": "text/plain"})
case("rebinding", "GET", "/health", None, {"Host": "evil.example.com"})
case("health", "GET", "/health")
case("models", "GET", "/v1/models")
case("models2", "GET", "/models")
case("props", "GET", "/props")
case("props-bad", "GET", "/props?model=nope")
case("slots", "GET", "/slots")
case("status", "GET", "/status")
case("settings", "GET", "/settings")
case("mcp", "GET", "/mcp")
case("config", "GET", "/config")
case("unknown-get", "GET", "/nope")
case("settings-post-foreign", "POST", "/settings", {"defaults": {"temperature": 0.5}},
     {"Origin": "https://evil.example.com"})
case("settings-post-bad", "POST", "/settings", {"x": 1})
case("settings-post", "POST", "/settings", {"defaults": {"temperature": 0.5, "top_k": 30}})
case("settings-after", "GET", "/settings")
case("chat-shared", "POST", "/v1/chat/completions", chat())
case("settings-clear", "POST", "/settings", {"defaults": None})
case("unload-form", "POST", "/unload", None, {"Content-Type": "text/plain"})
case("options", "OPTIONS", "/v1/chat/completions")
case("tokenize", "POST", "/tokenize", {"content": "<|im_start|>user\nHi café 日本"})
case("tokenize-pieces", "POST", "/tokenize", {"content": "日本 \U0001d518 <|im_end|>", "with_pieces": True,
                                               "parse_special": False})
case("detokenize", "POST", "/detokenize", {"tokens": [248045, 846, 198, 12675, 39579]})
case("detokenize-bad", "POST", "/detokenize", {"tokens": [1, -1]})
case("apply-template", "POST", "/apply-template", chat(tools=TOOLS, chat_template_kwargs={"enable_thinking": False}))

IDS = re.compile(r"\b(chatcmpl|msg|call|resp|rs|fc|msg_item|toolu)[-_][0-9A-Za-z]{6,}")


REQUEST_ID = re.compile(r"^[0-9a-f]{12}$")
# Prometheus lines whose values are timings or this PC's live readings
LIVE_METRIC = re.compile(r"^((?:vllm:\w+_seconds_(?:bucket|sum|count)|lamina:(?:gpu_\w+|cpu|ram_\w+|disk_\w+))(?:\{[^}]*\})?) .*$",
                         re.MULTILINE)


def norm(v, path=""):
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k in ("created", "created_at", "timings", "started", "uptime_s", "since", "time", "duration_s",
                     "first_engine_token_s", "decode_tok_s", "prompt_ms", "decode_ms", "seconds", "elapsed_s",
                     "instance_id", "last_request_at", "wallclock_s", "queued_at", "admitted_at", "finished_at",
                     "started_at", "completed_at", "queue_s", "load_s", "first_token_s", "tok_s", "tok_s_mean"):
                out[k] = "<t>"
            elif k in ("hardware", "hardware_static", "history"):   # this PC's live readings
                out[k] = "<hw>"
            elif k == "id" and isinstance(x, str) and REQUEST_ID.match(x):   # a request's random id
                out[k] = "<id>"
            else:
                out[k] = norm(x, path + "/" + k)
        return out
    if isinstance(v, list):
        return [norm(x, path) for x in v]
    if isinstance(v, str):
        v = LIVE_METRIC.sub(lambda m: m.group(1) + " <v>", v)
        return IDS.sub(lambda m: m.group(1) + "_<id>", v)
    return v


def parse_body(ctype, raw: bytes):
    text = raw.decode("utf-8", "replace")
    if "text/event-stream" in ctype:
        events = []
        for block in text.replace("\r\n", "\n").split("\n\n"):
            block = block.strip("\n")
            if not block or block.startswith(":"):
                continue                      # keep-alives can differ in number
            ev = {}
            for line in block.split("\n"):
                if line.startswith("event: "):
                    ev["event"] = line[7:]
                elif line.startswith("data: "):
                    d = line[6:]
                    try:
                        ev["data"] = json.loads(d)
                    except ValueError:
                        ev["data"] = d
            events.append(ev)
        return events
    try:
        return json.loads(text)
    except ValueError:
        return text


def call(port, method, path, body, headers):
    data = None
    h = {"Content-Type": "application/json", **headers}
    if body is not None:
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    if "Host" in headers:
        h["Host"] = headers["Host"]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


KEY_CASES = []   # with --api-key k1,k2 --api-monitor


def kcase(name, method, path, body=None, headers=None):
    KEY_CASES.append((name, method, path, body, headers or {}))


kcase("key-missing", "POST", "/v1/chat/completions", chat())
kcase("key-wrong", "POST", "/v1/chat/completions", chat(), {"Authorization": "Bearer nope"})
kcase("key-bearer", "POST", "/v1/chat/completions", chat(), {"Authorization": "Bearer k1"})
kcase("key-x-api-key", "POST", "/v1/messages", {"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "Hi"}]},
      {"x-api-key": "k2"})
kcase("key-models-open", "GET", "/v1/models")
kcase("key-models", "GET", "/v1/models", None, {"Authorization": "Bearer k1"})
kcase("key-health", "GET", "/health")
kcase("key-other-host", "GET", "/health", None, {"Host": "evil.example.com"})
kcase("key-monitor-list", "GET", "/api/requests", None, {"Authorization": "Bearer k1"})
kcase("key-metrics", "GET", "/metrics?requests=all", None, {"Authorization": "Bearer k1"})
kcase("key-status", "GET", "/status", None, {"Authorization": "Bearer k1"})
kcase("key-prom", "GET", "/metrics", None, {"Authorization": "Bearer k1", "Accept": "text/plain"})


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(port, proc, what, log):
    end = time.time() + 120
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited ({proc.returncode}):\n" + Path(log).read_text(errors="replace")[-2000:])
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.3)
    raise RuntimeError(f"{what} did not start")


PHASES = (("base", lambda: CASES, []), ("key", lambda: KEY_CASES, ["--api-key", "k1,k2", "--api-monitor"]))


def server_args(gguf, port, extra):
    scripts = sum((["--script", s] for s in SCRIPTS), [])
    return ["--engine", "mock", "--port", str(port), "--tokenizer", gguf, "--script-cycle", *scripts, *extra]


def answers(gguf, exe=None, log_dir=None):
    """{phase: {case name: [status, normalized body]}} from one server per phase."""
    out = {}
    log_dir = Path(log_dir or os.environ.get("TEMP", "/tmp"))
    for label, cases, extra in PHASES:
        port = free_port()
        log = log_dir / f"lamina_http_test_{label}.log"
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.Popen([str(exe or server_exe()), *server_args(gguf, port, extra)], cwd=ROOT, stdout=f,
                                    stderr=subprocess.STDOUT)
        try:
            wait_ready(port, proc, "lamina-server", log)
            got = {}
            for name, method, path, body, headers in cases():
                status, ctype, raw = call(port, method, path, body, headers)
                got[name] = [status, norm(parse_body(ctype, raw))]
            out[label] = got
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return out


def mock_server(test, *args):
    """A mock server for one test, stopped at its end: its port, and the file holding its output."""
    port = free_port()
    log = Path(os.environ.get("TEMP", "/tmp")) / "lamina_http_test_single.log"
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen([str(server_exe()), "--engine", "mock", "--port", str(port), "--tokenizer",
                                 os.environ["LAMINA_TEST_GGUF"], *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    test.addCleanup(proc.wait, 10)
    test.addCleanup(proc.terminate)
    return port, proc, log


def config_file(**keys):
    path = Path(os.environ.get("TEMP", "/tmp")) / "lamina_http_test_single.json"
    path.write_text(json.dumps(keys), encoding="utf-8")
    return str(path)


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class TemplateFile(unittest.TestCase):
    def test_it_replaces_the_files_template(self):
        tpl = Path(os.environ.get("TEMP", "/tmp")) / "lamina_http_test_template.jinja"
        tpl.write_text("X{{ messages[-1].content }}", encoding="utf-8")
        port, proc, log = mock_server(self, "--config", config_file(chat_template_file=str(tpl)))
        wait_ready(port, proc, "lamina-server", log)
        status, ctype, raw = call(port, "POST", "/apply-template", chat(), {})
        self.assertEqual((status, parse_body(ctype, raw)["prompt"]), (200, "XHi"))

    def test_the_shipped_template(self):
        port, proc, log = mock_server(self, "--script", "Plain answer.", "--config",
                                      config_file(chat_template_file="data/chat-template-froggeric-v22.5.jinja"))
        wait_ready(port, proc, "lamina-server", log)

        def prompt(**body):
            status, ctype, raw = call(port, "POST", "/apply-template", {"model": "m", **body}, {})
            self.assertEqual(status, 200, raw)
            return parse_body(ctype, raw)["prompt"]
        sentence = "Reasoning effort is set to xhigh."
        self.assertIn(sentence, prompt(messages=msgs(), reasoning_effort="high"))
        self.assertNotIn(sentence, prompt(messages=msgs()))
        self.assertTrue(prompt(messages=msgs()).endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(prompt(messages=msgs(), reasoning_effort="none").endswith("<think>\n\n</think>\n\n"))
        failed = msgs() + [{"role": "assistant", "tool_calls": [{"function": {"name": "get_time", "arguments": "{}"}}]},
                           {"role": "tool", "content": "Error: no such zone"}]
        self.assertIn("SYSTEM WARNING", prompt(messages=failed, tools=TOOLS))
        # the template closes the thinking for a tag in the message: the reply is the answer
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", {"model": "m", "messages": msgs("Hi <|think_off|>")}, {})
        message = parse_body(ctype, raw)["choices"][0]["message"]
        self.assertEqual((status, message.get("reasoning_content"), message.get("content")), (200, None, "Plain answer."))
        # and opens it for a tag when the request has it off: the reply is reasoning until </think>
        status, ctype, raw = call(port, "POST", "/v1/chat/completions",
                                  {"model": "m", "messages": msgs("Hi <|think_low|>"), "reasoning_effort": "none"}, {})
        message = parse_body(ctype, raw)["choices"][0]["message"]
        self.assertEqual((status, message.get("reasoning_content"), message.get("content")), (200, "Plain answer.", None))

    def test_a_missing_file_stops_the_start(self):
        _, proc, log = mock_server(self, "--config", config_file(chat_template_file="no/such/template.jinja"))
        self.assertEqual(proc.wait(60), 1)
        self.assertIn("chat_template_file", Path(log).read_text(errors="replace"))


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class ShortOpening(unittest.TestCase):
    """A conversation's opening message of a few tokens is read without the top effort's sentence."""
    TOP = "Reasoning effort is set to xhigh."

    def prompts(self, *extra):
        port, proc, log = mock_server(self, *extra)
        wait_ready(port, proc, "lamina-server", log)

        def prompt(**body):
            status, ctype, raw = call(port, "POST", "/apply-template", {"model": "m", **body}, {})
            self.assertEqual(status, 200, raw)
            return parse_body(ctype, raw)["prompt"]
        return prompt

    def test_on(self):
        prompt = self.prompts("--config", config_file(short_opening_tokens=4))
        self.assertNotIn(self.TOP, prompt(messages=msgs("hello")))
        self.assertNotIn(self.TOP, prompt(messages=msgs("hello"), reasoning_effort="high"))
        self.assertNotIn(self.TOP, prompt(messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]))
        # a longer message, a later turn, another effort
        self.assertIn(self.TOP, prompt(messages=msgs("Add a docblock to the PostCreator class.")))
        later = msgs("hello") + [{"role": "assistant", "content": "Hi."}, {"role": "user", "content": "ok"}]
        self.assertIn(self.TOP, prompt(messages=later))
        self.assertIn("effort is set to low", prompt(messages=msgs("hello"), reasoning_effort="low"))

    def test_off_by_default(self):
        self.assertIn(self.TOP, self.prompts()(messages=msgs("hello")))


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class MarkerNames(unittest.TestCase):
    """A stop token drawn while the reply quotes the marker is the marker's text; elsewhere it ends the turn."""
    # one script per engine pass: a pass ends at the marker, and the server's next pass takes the next script
    SCRIPTS = ["The marker is `<|im_end|>", "` in its text. Also \"<|endoftext|>", "\" here.</think>\n\nThe `<|im_end|>",
               "` token ends a turn.",
               "A thought.</think>\n\nHe said \"hello\"",
               "A last sentence.",
               "Fenced:\n```\nhello<|im_end|>", "\n```\nDone.</think>\n\nA block:\n```\nx\n```"]

    def serve(self, *extra):
        port, proc, log = mock_server(self, *sum((["--script", s] for s in self.SCRIPTS), []), *extra)
        wait_ready(port, proc, "lamina-server", log)

        def ask():
            status, ctype, raw = call(port, "POST", "/v1/chat/completions", chat(), {})
            self.assertEqual(status, 200, raw)
            choice = parse_body(ctype, raw)["choices"][0]
            return choice["finish_reason"], choice["message"].get("reasoning_content"), choice["message"].get("content")
        return ask

    def test_named_in_thinking_and_in_the_answer(self):
        ask = self.serve()
        self.assertEqual(ask(), ("stop", "The marker is `<|im_end|>` in its text. Also \"<|endoftext|>\" here.",
                                 "The `<|im_end|>` token ends a turn."))
        # a closing quote and a sentence's end are real ends
        self.assertEqual(ask(), ("stop", "A thought.", "He said \"hello\""))
        self.assertEqual(ask(), ("stop", "A last sentence.", None))
        # in a fence, while thinking; a fence's closing line in the answer is a real end
        self.assertEqual(ask(), ("stop", "Fenced:\n```\nhello<|im_end|>\n```\nDone.", "A block:\n```\nx\n```"))

    def test_off(self):
        self.assertEqual(self.serve("--config", config_file(literal_markers=False))(), ("stop", "The marker is `", None))


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class MalformedCall(unittest.TestCase):
    """A tool call that does not parse is the server's failure (tool_call_recovery: its text), and the next request
    is answered."""
    CALL = "<tool_call>\n<function=get_time>\n</function>\n<parameter=zone\n</tool_call>"

    def serve(self, *extra):
        answer = "A thought.</think>\n\n" + self.CALL
        port, proc, log = mock_server(self, "--script", answer, "--script", answer, "--script", "</think>\n\nNext.", *extra)
        wait_ready(port, proc, "lamina-server", log)
        return port

    def next_is_answered(self, port):
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", chat(), {})
        self.assertEqual((status, parse_body(ctype, raw)["choices"][0]["message"].get("content")), (200, "Next."))

    def test_strict(self):
        port = self.serve()
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", chat(tools=TOOLS), {})
        self.assertEqual((status, parse_body(ctype, raw)["error"].get("code")), (502, "tool_call_failed"), raw)
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", chat(tools=TOOLS, stream=True), {})
        events = parse_body(ctype, raw)
        errors = [e["data"]["error"] for e in events if isinstance(e.get("data"), dict) and "error" in e["data"]]
        self.assertEqual((status, events[-1].get("data"), len(errors)), (200, "[DONE]", 1), raw)
        self.assertIn("does not parse", errors[0]["message"])
        self.next_is_answered(port)

    def test_recovery(self):
        port = self.serve("--config", config_file(tool_call_recovery=True))
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", chat(tools=TOOLS), {})
        self.assertEqual((status, parse_body(ctype, raw)["choices"][0]["message"].get("content")), (200, self.CALL), raw)
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", chat(tools=TOOLS, stream=True), {})
        self.assertEqual((status, parse_body(ctype, raw)[-1].get("data")), (200, "[DONE]"), raw)
        self.next_is_answered(port)


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class TemplateRefusal(unittest.TestCase):
    """The template's raise_exception is a 400 with its message, and the request gives its admission back."""

    def test_a_400_that_frees_its_admission(self):
        port, proc, log = mock_server(self, "--max-queue", "0", "--script", "</think>\n\nFine.")
        wait_ready(port, proc, "lamina-server", log)
        for _ in range(2):
            status, ctype, raw = call(port, "POST", "/v1/chat/completions", {"model": "m", "messages": []}, {})
            self.assertEqual((status, parse_body(ctype, raw)["error"]["message"]), (400, "No messages provided."), raw)
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", chat(), {})
        self.assertEqual((status, parse_body(ctype, raw)["choices"][0]["message"]["content"]), (200, "Fine."), raw)


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class HugeLimits(unittest.TestCase):
    """A limit past int64 is Python's big int: as many as fit (max_tokens shortened to the room), never a 500."""

    def test_max_tokens(self):
        port, proc, log = mock_server(self, "--script", "</think>\n\nFine.")
        wait_ready(port, proc, "lamina-server", log)
        for value in ('"99999999999999999999"', '"-99999999999999999999"', "1e300", "18446744073709551615"):
            body = '{"model": "m", "max_tokens": %s, "messages": [{"role": "user", "content": "Hi"}]}' % value
            status, ctype, raw = call(port, "POST", "/v1/chat/completions", body, {})
            self.assertEqual((status, parse_body(ctype, raw)["choices"][0]["message"]["content"]), (200, "Fine."), value)


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class SchemaBudget(unittest.TestCase):
    """Unions whose two branches name the same next $def double the work per level (Python's too): refused quickly,
    the schema's with a 400, an answer's check with a 502."""

    @staticmethod
    def chain(key, levels, last):
        defs = {f"d{k}": {key: [{"$ref": f"#/$defs/d{k + 1}"}] * 2} for k in range(levels)}
        defs[f"d{levels}"] = last
        return defs

    def ask(self, port, schema):
        body = {"model": "m", "messages": msgs(), "response_format": {"type": "json_schema",
                                                                      "json_schema": {"name": "x", "schema": schema}}}
        t0 = time.time()
        status, ctype, raw = call(port, "POST", "/v1/chat/completions", body, {})
        self.assertLess(time.time() - t0, 20, "took too long")
        return status, parse_body(ctype, raw)["error"]["message"]

    def test_refused(self):
        port, proc, log = mock_server(self, "--script", "</think>\n\n{}")
        wait_ready(port, proc, "lamina-server", log)
        status, message = self.ask(port, {"$ref": "#/$defs/d0", "$defs": self.chain("anyOf", 40, {"type": "object"})})
        self.assertEqual(status, 400, message)
        self.assertIn("too complex", message)
        status, message = self.ask(port, {"type": "object", "$ref": "#/$defs/d0", "$defs": self.chain("allOf", 30, {})})
        self.assertEqual(status, 502, message)
        self.assertIn("steps to check", message)


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class MissingCallEnd(unittest.TestCase):
    """A whole call whose </tool_call> never comes (the turn ends, or max_tokens) keeps its arguments: Responses'
    custom tools take theirs from the finished call."""

    def test_custom_tool_input(self):
        text = "<tool_call>\n<function=apply_patch>\n<parameter=input>\n*** Begin Patch\n*** End Patch\n</parameter>\n</function>"
        port, proc, log = mock_server(self, "--script", "</think>\n\n" + text)
        wait_ready(port, proc, "lamina-server", log)
        body = {"model": "m", "input": "Patch it.", "tools": [{"type": "custom", "name": "apply_patch",
                                                               "description": "Apply a patch to the files."}]}
        status, ctype, raw = call(port, "POST", "/v1/responses", body, {})
        self.assertEqual(status, 200, raw)
        items = [i for i in parse_body(ctype, raw)["output"] if i.get("type") == "custom_tool_call"]
        self.assertEqual([(i["name"], i["input"]) for i in items], [("apply_patch", "*** Begin Patch\n*** End Patch")])


@unittest.skipUnless(os.environ.get("LAMINA_TEST_GGUF"), "LAMINA_TEST_GGUF names no model file")
class HttpGolden(unittest.TestCase):
    def test_answers(self):
        want = json.loads(GOLDEN.read_text(encoding="utf-8"))
        got = answers(os.environ["LAMINA_TEST_GGUF"])
        for label in want:
            for name, expected in want[label].items():
                with self.subTest(phase=label, case=name):
                    self.assertEqual(got[label].get(name), expected)


if __name__ == "__main__":
    if "--record" in sys.argv:
        gguf = os.environ.get("LAMINA_TEST_GGUF") or sys.exit("set LAMINA_TEST_GGUF to a model's GGUF")
        GOLDEN.write_text(json.dumps(answers(gguf), indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {GOLDEN}")
    else:
        unittest.main()
