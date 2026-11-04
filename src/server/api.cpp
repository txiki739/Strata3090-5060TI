// src/server/api.cpp - a request's events in the APIs' shapes (api.hpp).
#include "api.hpp"

#include "log.hpp"
#include "mcp.hpp"
#include "structured.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

namespace lamina {
namespace {

// the engine's last PP line stops short of the prompt: up to --short-read tokens (64 by default) are read through
// the verify windows, so a PP that close to the end says the prompt is read (a client would see a full bar at once)
constexpr int64_t kPpDoneTail = 64;

bool is_json_text(const std::string& text) {
    json v;
    return py_loads(text, v);
}

}  // namespace

json prompt_progress(Service& svc) {
    // llama.cpp's prompt_progress, so a client written for llama.cpp reads both servers: `processed` counts from token
    // 0, and the client takes the reused prefix out with `cache`
    Engine& e = *svc.engine;
    const Engine::Progress pr = e.progress();
    if (e.batch() > 0 || !pr.read) return nullptr;   // --batch: the engine's progress is one global line
    const auto [read, total] = *pr.read;
    if (total <= 0 || total - read <= kPpDoneTail) return nullptr;
    return {{"total", total}, {"cache", pr.reused}, {"processed", read}, {"time_ms", (int64_t) pr.ms}};
}

void openai_chunks(Service& svc, const json& req, const std::vector<int32_t>& ids, bool thinking, const json& tools,
                   int64_t max_new, Cancel& cancel, const Runner* run, const std::optional<std::string>& force,
                   const ChunkSink& out) {
    const std::string cid = "chatcmpl-" + random_hex(24);
    const int64_t created = (int64_t) now_s();
    const std::string model = svc.model_for(req);
    // llama.cpp's flag and its default: a client that wants prompt progress asks for it
    const bool want_progress = truthy(jget(req, "return_progress"));
    auto chunk = [&](json delta, const json& finish = nullptr) {
        return json{{"id", cid}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model},
                    {"choices", json::array({{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", finish}}})}};
    };
    {
        const json first = chunk({{"role", "assistant"}, {"content", ""}});
        if (!out(&first)) return;
    }
    int64_t calls = 0;
    std::map<std::string, int64_t> streamed;   // a call sent piece by piece -> its index
    std::set<std::string> finished;            // ... and the ones whose final tool_call came (#211)
    const Sink sink = [&](const std::string& kind, const json& x, const Event* ev) -> bool {
        if (kind == "ping") {
            const json pp = want_progress ? prompt_progress(svc) : json(nullptr);
            if (pp.is_null()) return out(nullptr);
            json c = chunk(json::object());
            c["prompt_progress"] = pp;
            return out(&c);
        }
        if (kind == "mcp") {
            json c = chunk(json::object());
            c["lamina_mcp"] = x;
            return out(&c);
        }
        if (kind == "event") {
            if (ev->kind == "reasoning" && !ev->text.empty()) {
                const json c = chunk({{"reasoning_content", ev->text}});
                return out(&c);
            }
            if (ev->kind == "content" && !ev->text.empty()) {
                const json c = chunk({{"content", ev->text}});
                return out(&c);
            }
            if (ev->kind == "tool_start") {
                streamed[ev->call->id] = calls++;
                const json c = chunk({{"tool_calls", json::array({{{"index", streamed[ev->call->id]}, {"id", ev->call->id},
                                                                   {"type", "function"},
                                                                   {"function", {{"name", ev->call->name}, {"arguments", ""}}}}})}});
                return out(&c);
            }
            if (ev->kind == "tool_args") {
                const json c = chunk({{"tool_calls", json::array({{{"index", streamed[ev->call->id]},
                                                                   {"function", {{"arguments", ev->text}}}}})}});
                return out(&c);
            }
            if (ev->kind == "tool_call" && streamed.count(ev->call->id)) {
                finished.insert(ev->call->id);
                return true;
            }
            if (ev->kind == "tool_call") {
                const json c = chunk({{"tool_calls", json::array({{{"index", calls}, {"id", ev->call->id}, {"type", "function"},
                                                                   {"function", {{"name", ev->call->name},
                                                                                 {"arguments", py_dumps(ev->call->arguments, false)}}}}})}});
                ++calls;
                return out(&c);
            }
            return true;
        }
        // done: a streamed call without its final tool_call is one the output ended inside, not "tool_calls" (#211)
        bool whole = calls > 0;
        for (const auto& [id, i] : streamed) whole = whole && finished.count(id);
        const std::string f = jstr(x, "finish");
        const std::string finish = whole && f == "stop" ? "tool_calls" : f == "cancel" ? "stop" : f;
        json last = chunk(json::object(), finish);
        const int64_t pt = jget(x, "prompt_tokens").is_number() ? x["prompt_tokens"].get<int64_t>() : (int64_t) ids.size();
        const int64_t ct = x["completion_tokens"].get<int64_t>();
        // cached_tokens: the part of the prompt the conversation cache already held (OpenAI's field)
        last["usage"] = {{"prompt_tokens", pt}, {"completion_tokens", ct}, {"total_tokens", pt + ct},
                         {"prompt_tokens_details", {{"cached_tokens", truthy(jget(x, "reused")) ? x["reused"] : json(0)}}}};
        if (truthy(jget(x, "timings"))) last["timings"] = x["timings"];   // llama.cpp's field: the speed its clients show
        return out(&last);
    };
    if (run != nullptr) (*run)(sink);
    else svc.run(ids, thinking, tools, max_new, req, cancel, force, sink);
}

json openai_collect(const std::vector<json>& chunks) {
    std::string content, reasoning;
    std::map<int64_t, json> by_index;
    json mcp = json::array();
    const json* last = nullptr;
    for (const json& c : chunks) {
        if (truthy(jget(c, "lamina_mcp"))) mcp.push_back(c["lamina_mcp"]);
        const json& d = c["choices"][0]["delta"];
        if (jget(d, "content").is_string()) content += d["content"].get<std::string>();
        if (jget(d, "reasoning_content").is_string()) reasoning += d["reasoning_content"].get<std::string>();
        if (jget(d, "tool_calls").is_array())
            for (const json& tc : d["tool_calls"]) {   // streamed calls arrive in pieces: merged by index
                const int64_t i = jget(tc, "index").is_number() ? tc["index"].get<int64_t>() : (int64_t) by_index.size();
                if (!by_index.count(i))
                    by_index[i] = {{"id", nullptr}, {"type", "function"}, {"function", {{"name", ""}, {"arguments", ""}}}};
                json& cur = by_index[i];
                if (truthy(jget(tc, "id"))) cur["id"] = tc["id"];
                const json& fn = jget(tc, "function");
                if (jget(fn, "name").is_string())
                    cur["function"]["name"] = cur["function"]["name"].get<std::string>() + fn["name"].get<std::string>();
                if (jget(fn, "arguments").is_string())
                    cur["function"]["arguments"] = cur["function"]["arguments"].get<std::string>() + fn["arguments"].get<std::string>();
            }
        last = &c;
    }
    json calls = json::array();
    const bool tool_finish = jget((*last)["choices"][0], "finish_reason") == "tool_calls";
    for (auto& [i, c] : by_index)   // a call the output ended inside (#211) is left out
        if (tool_finish || is_json_text(c["function"]["arguments"].get<std::string>())) calls.push_back(c);
    json msg = {{"role", "assistant"}, {"content", content.empty() ? json(nullptr) : json(content)}};
    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
    if (!calls.empty()) msg["tool_calls"] = calls;
    if (!mcp.empty()) msg["lamina_mcp"] = mcp;
    json result = {{"id", (*last)["id"]}, {"object", "chat.completion"}, {"created", (*last)["created"]},
                   {"model", (*last)["model"]},
                   {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", (*last)["choices"][0]["finish_reason"]}}})},
                   {"usage", jget(*last, "usage")}};
    if (truthy(jget(*last, "timings"))) result["timings"] = (*last)["timings"];
    return result;
}

ChunkSink StructuredBuffer::sink() {
    heartbeat_ = mono();
    return [this](const json* c) -> bool {
        const bool progress = c != nullptr && c->contains("prompt_progress");
        if (c != nullptr && !progress) buffered_.push_back(*c);   // a progress chunk has nothing to validate
        if (c == nullptr || progress || mono() - heartbeat_ >= 1) {
            heartbeat_ = mono();
            return out_(progress ? c : nullptr);
        }
        return true;
    };
}

bool StructuredBuffer::finish() {
    const json result = openai_collect(buffered_);
    const json& choice = result["choices"][0];
    const json& content = choice["message"]["content"];
    const std::string valid = validated_json(content.is_string() ? content.get<std::string>() : "", validator_,
                                             jstr(choice, "finish_reason"));
    if (!out_(&buffered_.front())) return false;
    json delta = {{"content", valid}};
    if (truthy(jget(choice["message"], "reasoning_content"))) delta["reasoning_content"] = choice["message"]["reasoning_content"];
    json c = buffered_.front();
    c["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
    if (!out_(&c)) return false;
    return out_(&buffered_.back());
}

void anthropic_events(Service& svc, const json& req, const std::vector<int32_t>& ids, bool thinking, const json& tools,
                      int64_t max_new, Cancel& cancel, const std::optional<std::string>& force, const EventSink& out) {
    const std::string mid = "msg_" + random_hex(24);
    auto send = [&](const char* name, const json& e) {
        const std::string n = name;
        return out(&n, &e);
    };
    if (!send("message_start", {{"type", "message_start"},
                                {"message", {{"id", mid}, {"type", "message"}, {"role", "assistant"}, {"model", svc.model_for(req)},
                                             {"content", json::array()}, {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                                             {"usage", {{"input_tokens", ids.size()}, {"output_tokens", 0}}}}}}))
        return;
    int64_t index = -1;
    std::string open_kind;
    bool used_tool = false;
    std::set<std::string> streamed, finished;   // calls sent piece by piece; those whose final tool_call came (#211)
    auto close = [&] { return send("content_block_stop", {{"type", "content_block_stop"}, {"index", index}}); };
    const Sink sink = [&](const std::string& kind, const json& x, const Event* ev) -> bool {
        if (kind == "ping") return out(nullptr, nullptr);
        if (kind == "event") {
            if (ev->kind == "tool_args")
                return send("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
                                                    {"delta", {{"type", "input_json_delta"}, {"partial_json", ev->text}}}});
            if (ev->kind == "tool_call" && streamed.count(ev->call->id)) {
                finished.insert(ev->call->id);
                return true;
            }
            const std::string want = ev->kind == "reasoning" ? "thinking" : ev->kind == "content" ? "text" : "tool_use";
            if (ev->kind != "tool_call" && ev->kind != "tool_start" && ev->text.empty()) return true;
            if (ev->kind == "tool_start") {
                streamed.insert(ev->call->id);
                used_tool = true;
            }
            if (open_kind != want || want == "tool_use") {
                if (!open_kind.empty() && !close()) return false;
                ++index;
                open_kind = want;
                json block;
                if (want == "thinking") block = {{"type", "thinking"}, {"thinking", ""}, {"signature", ""}};
                else if (want == "text") block = {{"type", "text"}, {"text", ""}};
                else block = {{"type", "tool_use"}, {"id", ev->call ? ev->call->id : ""}, {"name", ev->call ? ev->call->name : ""},
                              {"input", json::object()}};
                if (!send("content_block_start", {{"type", "content_block_start"}, {"index", index}, {"content_block", block}}))
                    return false;
            }
            if (want == "thinking")
                return send("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
                                                    {"delta", {{"type", "thinking_delta"}, {"thinking", ev->text}}}});
            if (want == "text")
                return send("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
                                                    {"delta", {{"type", "text_delta"}, {"text", ev->text}}}});
            if (ev->kind == "tool_start") return true;   // its input follows as tool_args pieces
            used_tool = true;
            return send("content_block_delta", {{"type", "content_block_delta"}, {"index", index},
                                                {"delta", {{"type", "input_json_delta"},
                                                           {"partial_json", py_dumps(ev->call->arguments, false)}}}});
        }
        if (kind != "done") return true;
        if (!open_kind.empty() && !close()) return false;
        bool all_finished = true;
        for (const auto& id : streamed) all_finished = all_finished && finished.count(id);
        const std::string f = jstr(x, "finish");
        std::string stop;
        if (!jget(x, "stop_sequence").is_null()) stop = "stop_sequence";
        else if (used_tool && all_finished && f == "stop") stop = "tool_use";
        else if (f == "stop" || f == "cancel") stop = "end_turn";
        else if (f == "length") stop = "max_tokens";
        else throw std::out_of_range(f);
        // Anthropic's counts: input_tokens leaves out what the conversation cache held (cache_read_input_tokens). At
        // least the last prompt token is always read: Claude Code takes input_tokens 0 as "not given" (#1013)
        const int64_t n = (int64_t) ids.size();
        const int64_t reused = std::max<int64_t>(0, std::min<int64_t>(jget(x, "reused").is_number() ? x["reused"].get<int64_t>() : 0, n - 1));
        if (!send("message_delta", {{"type", "message_delta"},
                                    {"delta", {{"stop_reason", stop}, {"stop_sequence", jget(x, "stop_sequence")}}},
                                    {"usage", {{"input_tokens", n - reused}, {"cache_read_input_tokens", reused},
                                               {"output_tokens", x["completion_tokens"]}}}}))
            return false;
        return send("message_stop", {{"type", "message_stop"}});
    };
    svc.run(ids, thinking, tools, max_new, req, cancel, force, sink);
}

json anthropic_collect(const std::vector<std::pair<std::string, json>>& events) {
    json msg;
    std::vector<json> blocks;
    for (const auto& [name, e] : events) {
        if (name == "message_start") {
            msg = e["message"];
        } else if (name == "content_block_start") {
            blocks.push_back(e["content_block"]);
        } else if (name == "content_block_delta") {
            const json& d = e["delta"];
            json& b = blocks.back();
            const std::string type = jstr(d, "type");
            if (type == "text_delta") b["text"] = b["text"].get<std::string>() + d["text"].get<std::string>();
            else if (type == "thinking_delta") b["thinking"] = b["thinking"].get<std::string>() + d["thinking"].get<std::string>();
            else b["_json"] = jstr(b, "_json") + d["partial_json"].get<std::string>();   // parsed when complete
        } else if (name == "content_block_stop" && !blocks.empty() && blocks.back().contains("_json")) {
            json& b = blocks.back();
            std::string text = b["_json"].get<std::string>();
            b.erase("_json");
            json input;
            if (py_loads(text.empty() ? "{}" : text, input)) b["input"] = input;
            else blocks.pop_back();   // a call the output ended inside (#211): it has no input to give
        } else if (name == "message_delta") {
            msg["stop_reason"] = e["delta"]["stop_reason"];
            msg["stop_sequence"] = jget(e["delta"], "stop_sequence");
            for (auto it = e["usage"].begin(); it != e["usage"].end(); ++it) msg["usage"][it.key()] = it.value();
        }
    }
    msg["content"] = blocks;
    return msg;
}

void run_with_mcp(Service& svc, const std::shared_ptr<McpHub>& hub, json messages, const json& tools, const json& kw,
                  std::vector<int32_t> ids, bool thinking, int64_t max_new, int64_t max_req, const json& sampling,
                  const std::shared_ptr<Cancel>& cancel_owner, const std::set<std::string>& mcp_names, const Sink& sink) {
    Cancel& cancel = *cancel_owner;
    const int64_t max_rounds = hub->settings["max_rounds"].get<int64_t>();
    int64_t total = 0, rounds = 0;
    json done;
    for (;;) {
        std::string text, reasoning;
        std::vector<std::shared_ptr<ToolCall>> calls;
        int own_calls = 0;
        bool gone = false;
        svc.run(ids, thinking, tools, max_new, sampling, cancel, std::nullopt,
                [&](const std::string& kind, const json& x, const Event* ev) -> bool {
                    if (kind == "done") {
                        done = x;
                        return true;
                    }
                    if (kind == "event") {
                        if (ev->call != nullptr && mcp_names.count(ev->call->name)) {
                            if (ev->kind == "tool_start") {   // the model has started writing a call: said at once
                                if (!sink("mcp", {{"event", "start"}, {"id", ev->call->id}, {"name", ev->call->name}}, nullptr))
                                    return !(gone = true);
                            } else if (ev->kind == "tool_call") {
                                calls.push_back(ev->call);
                            }
                            return true;   // its argument pieces are not streamed to the client
                        }
                        if (ev->kind == "tool_call") ++own_calls;
                        else if (ev->kind == "content") text += ev->text;
                        else if (ev->kind == "reasoning") reasoning += ev->text;
                    }
                    if (!sink(kind, x, ev)) return !(gone = true);
                    return true;
                });
        if (gone || done.is_null()) return;
        total += done["completion_tokens"].get<int64_t>();
        bool run_them = !calls.empty() && !own_calls && jstr(done, "finish") == "stop" && !cancel.is_set();
        if (run_them && rounds >= max_rounds) {
            if (!sink("mcp", {{"event", "limit"}, {"max_rounds", max_rounds}}, nullptr)) return;
            run_them = false;
        }
        if (!run_them) {
            for (const auto& c : calls)   // announced, never run: closed in the client's view
                if (!sink("mcp", {{"event", "result"}, {"id", c->id}, {"ok", false}, {"skipped", true}, {"text", "not run"},
                                  {"chars", 0}, {"truncated", false}, {"ms", 0}}, nullptr))
                    return;
            break;
        }
        ++rounds;
        std::vector<std::string> results;
        for (const auto& c : calls) {
            const auto routes = hub->routes();
            const auto it = routes.find(c->name);
            const json server = it != routes.end() && it->second.first != nullptr ? json(it->second.first->name) : json(nullptr);
            const std::string tool = it != routes.end() ? it->second.second : c->name;
            if (!sink("mcp", {{"event", "call"}, {"id", c->id}, {"name", c->name}, {"server", server}, {"tool", tool},
                              {"arguments", c->arguments}, {"round", rounds}}, nullptr))
                return;
            // the call runs on a thread while heartbeats go out: they reach the client as keep-alives, which is how a
            // client that went away (the web app's Stop) is noticed during a slow tool
            struct Box {
                std::mutex mu;
                std::condition_variable cv;
                bool finished = false, cancelled = false;
                json result;
                std::string error;
            };
            auto box = std::make_shared<Box>();
            std::thread([box, hub, name = c->name, args = c->arguments, cancel_owner] {
                json r;
                bool cancelled = false;
                std::string error;
                try {
                    r = hub->call(name, args, cancel_owner.get());
                } catch (const McpCancelled&) {
                    cancelled = true;
                } catch (const std::exception& e) {
                    error = e.what();
                }
                std::lock_guard<std::mutex> lk(box->mu);
                box->result = std::move(r);
                box->cancelled = cancelled;
                box->error = std::move(error);
                box->finished = true;
                box->cv.notify_all();
            }).detach();
            for (;;) {
                std::unique_lock<std::mutex> lk(box->mu);
                if (box->cv.wait_for(lk, std::chrono::seconds(1), [&] { return box->finished; })) break;
                lk.unlock();
                if (!sink("ping", nullptr, nullptr)) {
                    cancel.set();   // the client is gone: the tool is stopped too
                    std::unique_lock<std::mutex> wait(box->mu);
                    box->cv.wait_for(wait, std::chrono::seconds(10), [&] { return box->finished; });
                    return;
                }
            }
            if (box->cancelled || !box->error.empty()) {
                if (!box->error.empty()) throw std::runtime_error(box->error);
                break;
            }
            const json& r = box->result;
            char buf[64];
            std::snprintf(buf, sizeof buf, "%.1f", r["ms"].get<double>() / 1000);
            std::string chars = std::to_string(r["chars"].get<int64_t>());
            for (int i = (int) chars.size() - 3; i > 0; i -= 3) chars.insert((size_t) i, ",");
            say("[lamina] tool " + c->name + ": " + (truthy(r["ok"]) ? "ok" : "error") + ", " + chars + " characters in " +
                buf + " s" + (truthy(r["truncated"]) ? " (truncated for the model)" : ""));
            results.push_back(r["text"].get<std::string>());
            json ev = {{"event", "result"}, {"id", c->id}};
            for (const char* k : {"ok", "text", "chars", "truncated", "ms"}) ev[k] = r[k];
            if (!sink("mcp", ev, nullptr)) return;
        }
        if (cancel.is_set() || results.size() < calls.size()) {
            done["finish"] = "cancel";
            break;
        }
        json call_list = json::array();
        for (const auto& c : calls) call_list.push_back({{"function", {{"name", c->name}, {"arguments", c->arguments}}}});
        json assistant = {{"role", "assistant"}, {"content", std::string(strip(text))}};
        if (!reasoning.empty()) assistant["reasoning_content"] = std::string(strip(reasoning));
        assistant["tool_calls"] = call_list;
        messages.push_back(assistant);
        for (const auto& r : results) messages.push_back({{"role", "tool"}, {"content", r}});
        try {
            Prepared p = svc.prepare(messages, tools, kw, max_req, std::nullopt, nullptr, &cancel);
            ids = std::move(p.ids);
            thinking = p.thinking;
            max_new = p.max_new;
        } catch (const ClientGone&) {   // #1999: the client left while a picture was prepared
            done["finish"] = "cancel";
            break;
        }
    }
    json last = done;
    last["completion_tokens"] = total;
    last["prompt_tokens"] = ids.size();
    sink("done", last, nullptr);
}

void debug_request(const std::string& api, const json& req, const json& messages, const json& tools, int64_t max_new,
                   bool thinking, size_t prompt_tokens) {
    if (std::getenv("LAMINA_DEBUG") == nullptr || !*std::getenv("LAMINA_DEBUG")) return;
    const json last = messages.empty() ? json::object() : messages.back();
    std::string body;
    const json& content = jget(last, "content");
    if (content.is_array()) {
        for (const auto& p : content) body += (body.empty() ? "" : " ") + (p.is_object() ? jstr(p, "text") : std::string());
    } else if (content.is_string()) {
        body = content.get<std::string>();
    }
    std::string preview;
    size_t chars = 0;
    for (size_t i = 0; i < body.size() && chars < 80; ++chars) {
        size_t n = 1;
        utf8_at(body, i, &n);
        preview += body.substr(i, n);
        i += n;
    }
    preview = replace_all(preview, "\n", " ");
    say("[lamina] req " + api + ": msgs=" + std::to_string(messages.size()) +
        " tools=" + std::to_string(tools.is_array() ? tools.size() : 0) + " max_tokens_raw=" + py_repr_value(jget(req, "max_tokens")) +
        "/" + py_repr_value(jget(req, "max_completion_tokens")) + " max_new=" + std::to_string(max_new) +
        " thinking=" + (thinking ? "True" : "False") + " stream=" + (truthy(jget(req, "stream")) ? "True" : "False") +
        " prompt_tokens=" + std::to_string(prompt_tokens) + " last=" + py_repr_value(jget(last, "role")) + ":" +
        py_repr_value(json(preview)));
}

}  // namespace lamina
