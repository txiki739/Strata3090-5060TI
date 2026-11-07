// src/server/frontend.hpp - the request/response layer between API clients and the engine (serve/frontend.py's port).
//
// Text only: no model, no GPU. It normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat
// template's message form, renders the model's own Jinja chat template as Hugging Face does, and parses the streamed
// output into reasoning (<think>...</think>), content and tool calls in the template's XML form
// (<tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>), never emitting a partial tag.
#pragma once

#include "json.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jinja {
struct program;
}

namespace lamina {

/// the template refused the request's messages (raise_exception): a 400 with its message
struct TemplateRequestError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
/// a request field that is not what it must be: a 400 naming it
struct RequestError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
/// a tool call the model wrote that does not parse (tool_call_recovery off): the server's failure, never a 400
struct ToolCallError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

class ChatTemplate {
public:
    explicit ChatTemplate(std::string source);
    ~ChatTemplate();
    /// messages, tools (null: none), add_generation_prompt, the template's other variables (an object)
    std::string render(const json& messages, const json& tools = nullptr, bool add_generation_prompt = true,
                       const json& kwargs = json::object()) const;
    const std::string& source() const { return source_; }
    /// llama.cpp's capability names: rendering hints checked once against this template
    const json& caps() const { return caps_; }

private:
    json detect_caps() const;
    std::string source_;
    std::unique_ptr<jinja::program> prog_;
    json caps_;
};

// ---- requests
extern const std::vector<std::string_view> kTextParts;     ///< "text", "input_text" (and a part without a type)
extern const std::vector<std::string_view> kImageParts;    ///< "image_url", "input_image", "image"
std::string text_of(const json& content);
bool has_image(const json& content);
std::string image_source(const json& part);
json parts_of(const json& content);
/// the image sources of the rendered conversation, in prompt order
std::vector<std::string> images_of(const json& messages);
/// a reasoning effort as a client gives it -> the template's kwargs (RequestError for an unknown one)
json effort_kwargs(const json& value);
json budget_effort(const json& tokens);
json tool_arguments(const json& raw);

/// the literal tags kept as text inside a message: (tag, its private-use mark), in matching order
using LiteralTags = std::vector<std::pair<std::string, std::string>>;
const LiteralTags& default_literal_tags();
/// default_literal_tags plus a mark for each control token's text, longest first
LiteralTags literal_tags(const std::vector<std::string>& controls);
/// (messages, tools) with each literal tag in their text swapped for its mark; `changed` false: the same values back
struct Marked {
    json messages, tools;
    bool changed = false;
};
Marked mark_think_literals(const json& messages, const json& tools, const LiteralTags& tags);
/// the rendered prompt with the marks turned back into their tags' text, and the byte spans of those tags in it
std::pair<std::string, std::vector<std::pair<size_t, size_t>>> unmark_think_literals(const std::string& prompt,
                                                                                      const LiteralTags& tags);

/// the config's "env_to_user": opencode's <env> block moved from the system message to the first user message
extern bool g_env_to_user;
struct Converted {
    json messages = json::array();
    json tools = nullptr;
    json kwargs = json::object();
};
/// the template takes a system message only at the start: later ones (Claude Code's hook context, a late
/// "developer" message) become user messages in place
json late_system_to_user(json messages);
Converted openai_to_messages(const json& req);
std::string pin_billing_stamp(std::string system);
/// `think_unasked`: a request without "thinking", an effort or a budget gets the template's default (it thinks);
/// false renders it without thinking. May set req["reasoning_budget_tokens"] from Anthropic's budget_tokens.
Converted anthropic_to_messages(json& req, bool think_unasked = true);

// ---- the output parser
struct ToolCall {
    std::string name;
    json arguments = json::object();
    std::string id = "call_" + random_hex(24);
};

struct Event {
    std::string kind;   ///< "reasoning" | "content" | "tool_call" | "tool_start" | "tool_args"
    std::string text;
    std::shared_ptr<ToolCall> call;
};

inline constexpr std::string_view kThinkEnd = "</think>";
inline constexpr std::string_view kCallStart = "<tool_call>";
inline constexpr std::string_view kCallEnd = "</tool_call>";
inline constexpr std::string_view kParamEnd = "</parameter>";
inline constexpr std::string_view kParamStart = "<parameter=";
inline constexpr std::string_view kFuncStart = "<function=";
inline constexpr std::string_view kFuncEnd = "</function>";

/// where a parameter value ends (-1: none yet; -2: a candidate whose follower has not arrived, `final` accepts it)
long param_end(std::string_view text, bool final = false);
/// where a tool call's body ends (the </tool_call> after its own </function>); -1: not complete
long call_end(std::string_view text);
/// where a <function=NAME>...</function> at the start of `text` ends; -1: not complete or not that form
long function_end(std::string_view text);
/// tool_choice in either API's shape -> ("auto" | "none" | "required" | "named" | "unknown", name)
std::pair<std::string, std::string> tool_choice_of(const json& tool_choice);
/// the text that opens the call a reply must make, or nullopt (the model decides)
std::optional<std::string> forced_call(const json& tool_choice, const json& tools);
/// {"name": N, "arguments": {...}} of a declared tool -> its call (or none)
std::shared_ptr<ToolCall> json_tool_call(std::string_view body, const json& schemas);
/// <function=NAME><parameter=P>VALUE</parameter>...</function> -> its call; std::invalid_argument when malformed
ToolCall parse_tool_call(std::string_view body, const json& schema = nullptr);

class OutputParser {
public:
    OutputParser(bool thinking = true, const json& tools = nullptr, bool stream_tools = false, bool recover = false);
    std::vector<Event> feed(std::string_view delta);
    /// the end of generation: whatever is held; `reason` is how the turn ended (empty or "stop": a natural stop)
    std::vector<Event> finish(std::string_view reason = "");
    int rescued = 0;   ///< calls delivered from the reasoning
    int refused = 0;   ///< declared calls kept as reasoning text
    const std::string& state() const { return state_; }
    const std::string& buf() const { return buf_; }      ///< text held back (a tag still arriving)
    bool has_pending() const { return !pending_.empty(); }   ///< calls from the reasoning waiting to settle

private:
    bool ok_at(size_t p);
    std::string follower(std::string_view after) const;
    std::pair<long, bool> bare_opener();
    std::string track(std::string text);
    bool opener_ok() const;
    bool in_code() const;
    std::optional<bool> quoted_think_end(size_t i) const;
    std::vector<Event> release(bool deliver);
    void reset_scan();
    long wrapped_call_end();
    std::vector<Event> scan();
    std::vector<Event> close_scan();
    std::vector<Event> finish_call(std::string body);

    std::string state_, buf_;
    bool lead_ = false;
    json schemas_ = json::object();   ///< name -> tool
    bool stream_tools_ = false, recover_ = false, bare_ = false, batch_ = false;
    struct Pending {
        std::string raw;
        std::shared_ptr<ToolCall> call;
    };
    std::vector<Pending> pending_;
    std::string fence_, line_;
    long ticks_ = 0;
    // the streaming view of a call body (stream_tools)
    long call_end_checked_ = 0;
    size_t sp_ = 0;
    std::string ss_;
    std::shared_ptr<ToolCall> scall_;
    bool sfirst_ = true, sval_started_ = false;
    json sdeclared_ = json::object();
    std::string sparam_;   ///< the parameter being streamed
};

}  // namespace lamina
