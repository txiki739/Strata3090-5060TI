// src/server/api.hpp - a request's events in the APIs' shapes: OpenAI Chat Completions chunks, Anthropic Messages
// events (streamed or collected into one answer), the structured-output buffer, and the MCP tool loop that runs the
// web app's MCP tools between rounds of generation (serve/server.py's openai_chunks ... run_with_mcp).
#pragma once

#include "json.hpp"
#include "service.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace lamina {

class McpHub;
class Validator;

/// a request's events: Service::run's, or the MCP loop's
using Runner = std::function<void(const Sink&)>;
/// an OpenAI chunk to send (nullptr: a heartbeat, the SSE comment); false: the client is gone
using ChunkSink = std::function<bool(const json* chunk)>;
/// an Anthropic event to send (nullptr name: a heartbeat); false: the client is gone
using EventSink = std::function<bool(const std::string* name, const json* event)>;

/// the last PP line as llama.cpp's prompt_progress (null when there is nothing to say)
json prompt_progress(Service& svc);
/// `run`: the events to send instead of Service::run's (the MCP loop), whose "mcp" items become chunks with an empty
/// delta and a lamina_mcp field
void openai_chunks(Service& svc, const json& req, const std::vector<int32_t>& ids, bool thinking, const json& tools,
                   int64_t max_new, Cancel& cancel, const Runner* run, const std::optional<std::string>& force,
                   const ChunkSink& out);
/// the chunks (heartbeats left out) as one chat.completion
json openai_collect(const std::vector<json>& chunks);

/// a structured stream is held until its answer is validated: heartbeats pass (at least one a second), and at the
/// end the first chunk, one chunk with the whole validated content, and the last
class StructuredBuffer {
public:
    StructuredBuffer(const Validator* validator, ChunkSink out) : validator_(validator), out_(std::move(out)) {}
    ChunkSink sink();
    /// after the run: validates (StructuredOutputError) and sends the three chunks; false: the client is gone
    bool finish();

private:
    const Validator* validator_;
    ChunkSink out_;
    std::vector<json> buffered_;
    double heartbeat_ = 0;
};

void anthropic_events(Service& svc, const json& req, const std::vector<int32_t>& ids, bool thinking, const json& tools,
                      int64_t max_new, Cancel& cancel, const std::optional<std::string>& force, const EventSink& out);
/// the events (heartbeats left out) as one message
json anthropic_collect(const std::vector<std::pair<std::string, json>>& events);

/// Service::run with the MCP tools executed here: a call to an MCP tool is run, its call and result join the
/// conversation and the model goes on, up to the hub's max_rounds. "mcp" events say what the tools do; one "done"
/// at the very end counts every round's output tokens. `mcp_names`: this request's MCP tools; a call to any other
/// is the client's and ends the turn as always. `cancel` is shared with each tool's thread, which can outlive the
/// request when a tool does not notice its Stop.
void run_with_mcp(Service& svc, const std::shared_ptr<McpHub>& hub, json messages, const json& tools, const json& kw,
                  std::vector<int32_t> ids, bool thinking, int64_t max_new, int64_t max_req, const json& sampling,
                  const std::shared_ptr<Cancel>& cancel, const std::set<std::string>& mcp_names, const Sink& sink);

/// one line per request while diagnosing blank turns (LAMINA_DEBUG=1)
void debug_request(const std::string& api, const json& req, const json& messages, const json& tools, int64_t max_new,
                   bool thinking, size_t prompt_tokens);

}  // namespace lamina
