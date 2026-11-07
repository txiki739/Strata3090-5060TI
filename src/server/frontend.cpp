#include "frontend.hpp"

#include "log.hpp"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <list>
#include <map>
#include <mutex>
#include <set>

namespace lamina {
namespace {

jinja::value to_jinja(const json& j) {
    using namespace jinja;
    switch (j.type()) {
        case json::value_t::null: return mk_val<value_none>();
        case json::value_t::boolean: return mk_val<value_bool>(j.get<bool>());
        case json::value_t::number_integer: return mk_val<value_int>(j.get<int64_t>());
        case json::value_t::number_unsigned: return mk_val<value_int>((int64_t) j.get<uint64_t>());
        case json::value_t::number_float: return mk_val<value_float>(j.get<double>());
        case json::value_t::string: return mk_val<value_string>(j.get<std::string>());
        case json::value_t::array: {
            auto arr = mk_val<value_array>();
            for (const auto& x : j) arr->push_back(to_jinja(x));
            return arr;
        }
        case json::value_t::object: {
            auto obj = mk_val<value_object>();
            for (auto it = j.begin(); it != j.end(); ++it) obj->insert(it.key(), to_jinja(it.value()));
            return obj;
        }
        default: return mk_val<value_none>();
    }
}

bool in(std::string_view v, const std::vector<std::string_view>& set) {
    return std::find(set.begin(), set.end(), v) != set.end();
}

/// a part's "type" as TEXT_PARTS reads it: a missing type is None (which TEXT_PARTS holds)
bool is_text_part(const json& part) {
    const json& t = jget(part, "type");
    return t.is_null() || (t.is_string() && in(t.get_ref<const std::string&>(), {"text", "input_text"}));
}

bool is_image_part(const json& part) {
    const json& t = jget(part, "type");
    return t.is_string() && in(t.get_ref<const std::string&>(), kImageParts);
}

/// dict.update
void merge(json& dst, const json& src) {
    for (auto it = src.begin(); it != src.end(); ++it) dst[it.key()] = it.value();
}

std::string type_of(const json& part) {
    const json& t = jget(part, "type");
    return t.is_string() ? t.get<std::string>() : std::string();
}

std::string utf8_prefix(std::string_view s, size_t chars) {
    size_t i = 0;
    for (size_t k = 0; k < chars && i < s.size(); ++k) {
        size_t n = 1;
        utf8_at(s, i, &n);
        i += n;
    }
    return std::string(s.substr(0, i));
}

}  // namespace

// ---------------------------------------------------------------------------------------------- template
ChatTemplate::ChatTemplate(std::string source) : source_(std::move(source)) {
    jinja::lexer lexer;
    const auto tokens = lexer.tokenize(source_);
    prog_ = std::make_unique<jinja::program>(jinja::parse_from_tokens(tokens));
    caps_ = detect_caps();
}

ChatTemplate::~ChatTemplate() = default;

namespace {
/// the message of the template's raise_exception() in a runtime error's text, or "" for any other error: the runtime
/// (vendor/jinja/value.cpp) throws it as "Jinja Exception: <message>", and each statement wraps it after its place in
/// the source ("... Error: Jinja Exception: <message>")
std::string refusal_of(const std::string& what) {
    static const std::string kMark = "Jinja Exception: ";
    const size_t at = what.rfind(kMark);
    if (at == std::string::npos || (at != 0 && what.compare(at - 2, 2, ": ") != 0)) return "";
    return what.substr(at + kMark.size());
}
}  // namespace

std::string ChatTemplate::render(const json& messages, const json& tools, bool add_generation_prompt,
                                 const json& kwargs) const {
    // completed empty assistant turns are no examples for the next reply; the final message stays (#843;
    // LAMINA_KEEP_EMPTY_TURNS=1 renders them)
    const char* keep = std::getenv("LAMINA_KEEP_EMPTY_TURNS");
    json msgs = json::array();
    for (size_t i = 0; i < messages.size(); ++i) {
        const json& m = messages[i];
        const bool drop = i + 1 < messages.size() && !(keep != nullptr && std::string_view(keep) == "1") &&
                          m.is_object() && jstr(m, "role") == "assistant" &&
                          strip(text_of(jget(m, "content"))).empty() && !has_image(jget(m, "content")) &&
                          !truthy(jget(m, "tool_calls"));
        if (!drop) msgs.push_back(m);
    }
    jinja::context ctx(source_);
    ctx.set_val("messages", to_jinja(msgs));
    ctx.set_val("tools", to_jinja(tools));
    ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(add_generation_prompt));
    if (kwargs.is_object())
        for (auto it = kwargs.begin(); it != kwargs.end(); ++it) ctx.set_val(it.key(), to_jinja(it.value()));
    try {
        jinja::runtime rt(ctx);
        const jinja::value results = rt.execute(*prog_);
        return jinja::runtime::gather_string_parts(results)->as_string().str();
    } catch (const std::exception& e) {
        // the template refused the request (raise_exception, e.g. "No messages provided."): a 400 with its message;
        // any other error is the template's own (the server's 500)
        const std::string refusal = refusal_of(e.what());
        if (refusal.empty()) throw;
        throw TemplateRequestError(refusal);
    }
}

json ChatTemplate::detect_caps() const {
    auto render_probe = [&](const json& messages, const json& tools = nullptr) -> std::string {
        try {
            return render(messages, tools);
        } catch (const std::exception&) {   // a hint only: a template that rejects a feature has it off
            return "";
        }
    };
    const json user = {{"role", "user"}, {"content", "lamina_caps_user"}};
    json tools = json::array();
    for (int i = 0; i < 2; ++i)
        tools.push_back({{"name", "lamina_caps_call_" + std::to_string(i)},
                         {"description", "lamina_caps_description"},
                         {"parameters", {{"type", "object"}, {"properties", {{"arg", {{"type", "string"}}}}}}}});
    const std::string tool_prompt = render_probe(json::array({user}), tools);

    auto calls_supported = [&](int count) {
        json calls = json::array(), tool_calls = json::array();
        for (int i = 0; i < count; ++i) {
            json call = {{"name", "lamina_caps_call_" + std::to_string(i)},
                         {"arguments", {{"arg", "lamina_caps_arg_" + std::to_string(i)}}}};
            calls.push_back(call);
            tool_calls.push_back({{"function", call}});
        }
        json messages = json::array({user, {{"role", "assistant"}, {"content", ""}, {"tool_calls", tool_calls}}});
        std::vector<std::string> replies;
        for (int i = 0; i < count; ++i) {
            replies.push_back("lamina_caps_result_" + std::to_string(i));
            messages.push_back({{"role", "tool"}, {"content", replies.back()}});
        }
        messages.push_back(user);
        const std::string prompt = render_probe(messages, tools);
        // the complete calls to the probe functions, as re.findall(r"<tool_call>\s*(<function=lamina_caps_call_\d+>
        // .*?</function>)\s*</tool_call>", re.S) finds them
        json parsed = json::array();
        size_t at = 0;
        while ((at = prompt.find(kCallStart, at)) != std::string::npos) {
            const size_t body = prompt.size() - lstrip(std::string_view(prompt).substr(at + kCallStart.size())).size();
            const std::string_view head = "<function=lamina_caps_call_";
            size_t d = body + head.size();
            const bool shape = prompt.compare(body, head.size(), head) == 0 && d < prompt.size() &&
                               std::isdigit((unsigned char) prompt[d]);
            bool found = false;
            if (shape) {
                while (d < prompt.size() && std::isdigit((unsigned char) prompt[d])) ++d;
                if (d < prompt.size() && prompt[d] == '>') {
                    for (size_t fe = prompt.find(kFuncEnd, d); fe != std::string::npos; fe = prompt.find(kFuncEnd, fe + 1)) {
                        const size_t end = fe + kFuncEnd.size();
                        const std::string_view after = lstrip(std::string_view(prompt).substr(end));
                        if (starts_with(after, kCallEnd)) {
                            try {
                                const ToolCall c = parse_tool_call(std::string_view(prompt).substr(body, end - body));
                                parsed.push_back({{"name", c.name}, {"arguments", c.arguments}});
                            } catch (const std::invalid_argument&) {
                                return false;
                            }
                            at = prompt.size() - after.size() + kCallEnd.size();
                            found = true;
                            break;
                        }
                    }
                }
            }
            if (!found) at += kCallStart.size();
        }
        if (parsed != calls) return false;
        for (const auto& r : replies)
            if (prompt.find(r) == std::string::npos) return false;
        return true;
    };
    const json history = json::array({user,
                                      {{"role", "assistant"}, {"content", "lamina_caps_answer"},
                                       {"reasoning_content", "lamina_caps_reasoning"}},
                                      user});
    auto has = [&](const std::string& s, std::string_view x) { return s.find(x) != std::string::npos; };
    return {{"supports_tools", has(tool_prompt, "lamina_caps_call_0") && has(tool_prompt, "lamina_caps_description") &&
                                   has(tool_prompt, "<tool_call>") && has(tool_prompt, "<function=")},
            {"supports_tool_calls", calls_supported(1)},
            {"supports_system_role",
             has(render_probe(json::array({{{"role", "system"}, {"content", "lamina_caps_system"}}, user})),
                 "lamina_caps_system")},
            {"supports_parallel_tool_calls", calls_supported(2)},
            {"supports_preserve_reasoning", has(render_probe(history), "lamina_caps_reasoning")}};
}

// ---------------------------------------------------------------------------------------------- requests
const std::vector<std::string_view> kTextParts = {"text", "input_text"};
const std::vector<std::string_view> kImageParts = {"image_url", "input_image", "image"};

std::string text_of(const json& content) {
    if (content.is_null()) return "";
    if (content.is_string()) return content.get<std::string>();
    std::string out;
    if (content.is_array())
        for (const auto& part : content)
            if (part.is_object() && is_text_part(part)) out += jstr(part, "text");
    return out;
}

bool has_image(const json& content) {
    if (!content.is_array()) return false;
    for (const auto& p : content)
        if (p.is_object() && is_image_part(p)) return true;
    return false;
}

std::string image_source(const json& part) {
    if (type_of(part) == "image") {
        const json& src = jget(part, "source");   // a missing or empty source reads as {}
        if (jstr(src, "type") == "base64")
            return "data:" + jstr(src, "media_type", "image/png") + ";base64," + jstr(src, "data");
        if (truthy(jget(src, "url"))) return jstr(src, "url");
        return jstr(src, "path");
    }
    json url = jget(part, "image_url");
    if (url.is_object()) url = jget(url, "url");
    return url.is_string() ? url.get<std::string>() : std::string();
}

json parts_of(const json& content) {
    if (!has_image(content)) return text_of(content);
    json items = json::array();
    for (const auto& part : content) {
        if (!part.is_object()) continue;
        if (is_image_part(part)) {
            items.push_back({{"type", "image"}, {"source", image_source(part)}});
        } else if (is_text_part(part) && part.contains("text")) {
            items.push_back({{"type", "text"}, {"text", jstr(part, "text")}});
        }
    }
    return items;
}

namespace {

/// a request's content parts as the readers take them: a "refusal" part is read as its text; a part of a type this
/// server does not read (a file, audio, a search result ...) is left out by the readers, with no error (real clients
/// send such parts beside text the model can answer from)
json check_parts(const json& content) {
    if (!content.is_array()) return content;
    json out = content;
    for (auto& part : out)
        if (part.is_object() && type_of(part) == "refusal") part = {{"type", "text"}, {"text", jstr(part, "refusal")}};
    return out;
}

}  // namespace

std::vector<std::string> images_of(const json& messages) {
    std::vector<std::string> out;
    for (const auto& m : messages) {
        const json& c = jget(m, "content");
        if (!c.is_array()) continue;
        for (const auto& item : c)
            if (jstr(item, "type") == "image") out.push_back(jstr(item, "source"));
    }
    return out;
}

json effort_kwargs(const json& value) {
    // the template knows low, medium and xhigh (its default; "high" means xhigh), and enable_thinking=false for none
    static const std::map<std::string, std::string> effort = {
        {"none", ""},     {"off", ""},      {"minimal", ""},  {"disabled", ""}, {"false", ""},
        {"low", "low"},   {"medium", "medium"}, {"high", "xhigh"}, {"xhigh", "xhigh"}, {"max", "xhigh"},
        {"maximum", "xhigh"}};
    if (value.is_null() || (value.is_string() && value.get_ref<const std::string&>().empty())) return json::object();
    if (value.is_boolean() && !value.get<bool>()) return {{"enable_thinking", false}};
    // str(value).strip().lower(); the message shows repr(value)
    const std::string text = value.is_string() ? value.get<std::string>() : py_repr_value(value);
    const auto it = effort.find(lower(strip(text)));
    if (it == effort.end())
        throw RequestError("unknown reasoning effort " + py_repr_value(value) + ": use none, low, medium or high");
    if (it->second.empty()) return {{"enable_thinking", false}};
    return {{"reasoning_effort", it->second}};
}

json budget_effort(const json& tokens) {
    // int(tokens): an int, a float cut to one, True as 1, or a string that is a whole integer
    long long n;
    if (is_int(tokens)) {
        n = tokens.get<long long>();
    } else if (tokens.is_number_float()) {
        n = (long long) tokens.get<double>();
    } else if (tokens.is_boolean()) {
        n = tokens.get<bool>() ? 1 : 0;
    } else if (tokens.is_string()) {
        const std::string s(strip(tokens.get<std::string>()));
        try {
            size_t used = 0;
            n = std::stoll(s, &used);
            if (used != s.size()) return json::object();
        } catch (const std::exception&) {
            return json::object();
        }
    } else {
        return json::object();
    }
    return {{"reasoning_effort", n < 2048 ? "low" : n < 8192 ? "medium" : "xhigh"}};
}

// ---- literal tags
namespace {

std::string mark_char(uint32_t cp) {
    std::string s;
    utf8_append(s, cp);
    return s;
}

const std::vector<std::string> kToolLiteralTags = {"</tool_call>", "<tool_call>", "<tool_response>", "</tool_response>"};

bool is_tool_literal(const std::string& tag) {
    return std::find(kToolLiteralTags.begin(), kToolLiteralTags.end(), tag) != kToolLiteralTags.end();
}

std::string mark(std::string text, const LiteralTags& tags) {
    for (const auto& [tag, m] : tags) text = replace_all(std::move(text), tag, m);
    return text;
}

json mark_deep(const json& v, const LiteralTags& tags) {
    if (v.is_string()) return mark(v.get<std::string>(), tags);
    if (v.is_object()) {
        json out = json::object();
        for (auto it = v.begin(); it != v.end(); ++it) out[it.key()] = mark_deep(it.value(), tags);
        return out;
    }
    if (v.is_array()) {
        json out = json::array();
        for (const auto& x : v) out.push_back(mark_deep(x, tags));
        return out;
    }
    return v;
}

bool has_tag(const json& v, const LiteralTags& tags) {
    if (v.is_string()) {
        const std::string& s = v.get_ref<const std::string&>();
        for (const auto& t : tags)
            if (s.find(t.first) != std::string::npos) return true;
        return false;
    }
    if (v.is_object() || v.is_array()) {
        for (const auto& x : v)
            if (has_tag(x, tags)) return true;
    }
    return false;
}

/// a user message whose whole text is <tool_response>...</tool_response> keeps that outer pair as the markers (the
/// template tells such a turn from a question by it); only the text inside is marked. nullopt: not such a message
std::optional<json> keep_tool_result_wrap(const json& content, const LiteralTags& tags) {
    const std::string o = "<tool_response>", c = "</tool_response>";
    auto wrap = [&](const std::string& text, bool first, bool last) {
        const size_t i = first ? text.find(o) + o.size() : 0;
        const size_t j = last ? text.rfind(c) : text.size();
        const std::string head = first ? mark(text.substr(0, i - o.size()), tags) + o : "";
        const std::string tail = last ? c + mark(text.substr(j + c.size()), tags) : "";
        return head + mark(text.substr(i, j - i), tags) + tail;
    };
    if (content.is_string()) {
        const std::string_view body = strip(content.get_ref<const std::string&>());
        if (body.size() >= o.size() + c.size() && starts_with(body, o) && ends_with(body, c))
            return json(wrap(content.get<std::string>(), true, true));
        return std::nullopt;
    }
    if (!content.is_array() || content.empty()) return std::nullopt;
    for (const auto& x : content)
        if (!x.is_object() || !x.contains("text") || !x["text"].is_string()) return std::nullopt;
    std::string joined;
    for (const auto& x : content) joined += x["text"].get<std::string>();
    const std::string_view body = strip(joined);
    if (!(body.size() >= o.size() + c.size() && starts_with(body, o) && ends_with(body, c))) return std::nullopt;
    const std::string first = content.front()["text"].get<std::string>(), last = content.back()["text"].get<std::string>();
    if (!(starts_with(lstrip(first), o) && ends_with(rstrip(last), c))) return std::nullopt;   // split across parts
    json out = json::array();
    if (content.size() == 1) {
        json p = content[0];
        p["text"] = wrap(first, true, true);
        out.push_back(p);
        return out;
    }
    json p0 = content.front();
    p0["text"] = wrap(first, true, false);
    out.push_back(p0);
    for (size_t k = 1; k + 1 < content.size(); ++k) out.push_back(mark_deep(content[k], tags));
    json pn = content.back();
    pn["text"] = wrap(last, false, true);
    out.push_back(pn);
    return out;
}

}  // namespace

const LiteralTags& default_literal_tags() {
    // #537: <think> / </think> inside a message's text is text, not the model's markers; #554: the vision markers;
    // #2026 / 0.1.44: the tool-call and tool-result tags (all type 4: the tokenizer matches them everywhere)
    static const LiteralTags tags = {
        {"<think>", mark_char(0xF0E01)},          {"</think>", mark_char(0xF0E02)},
        {"<|vision_start|>", mark_char(0xF0E03)}, {"<|image_pad|>", mark_char(0xF0E04)},
        {"<|vision_end|>", mark_char(0xF0E05)},   {"<|video_pad|>", mark_char(0xF0E06)},
        {"</tool_call>", mark_char(0xF0E07)},     {"<tool_call>", mark_char(0xF0E08)},
        {"<tool_response>", mark_char(0xF0E09)},  {"</tool_response>", mark_char(0xF0E0A)}};
    return tags;
}

LiteralTags literal_tags(const std::vector<std::string>& controls) {
    // the control tokens' texts too (<|im_start|> written inside a message opened a turn there); longest first, as
    // the tokenizer matches them
    LiteralTags tags = default_literal_tags();
    uint32_t k = 0;
    for (const std::string& c : controls) {
        if (std::any_of(tags.begin(), tags.end(), [&](const auto& t) { return t.first == c; })) continue;
        tags.emplace_back(c, mark_char(0xF0E10 + k++));
    }
    std::stable_sort(tags.begin(), tags.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    return tags;
}

Marked mark_think_literals(const json& messages, const json& tools, const LiteralTags& tags) {
    if (!has_tag(messages, tags) && !has_tag(tools, tags)) return {messages, tools, false};
    LiteralTags own;   // what an assistant message keeps as written: its tool tags stay the tokens it wrote
    for (const auto& t : tags)
        if (!is_tool_literal(t.first)) own.push_back(t);
    json out = json::array();
    for (const auto& m0 : messages) {
        json m = m0;
        const json content = jget(m0, "content");
        const std::string role = jstr(m0, "role");
        const LiteralTags& mine = role == "assistant" ? own : tags;
        if (m.is_object())
            for (auto it = m.begin(); it != m.end(); ++it)
                if (it.key() != "role") it.value() = mark_deep(it.value(), mine);
        if (role == "user")
            if (auto kept = keep_tool_result_wrap(content, tags)) m["content"] = *kept;
        if (role == "assistant" && content.is_string()) {
            const std::string& c = content.get_ref<const std::string&>();
            // a whole <think>...</think> block at its start (clients that send the reasoning inline) stays the markers
            if (starts_with(lstrip(c), "<think>") && c.find("</think>") != std::string::npos) {
                const size_t i = c.find("<think>") + 7, j = c.find("</think>");
                m["content"] = c.substr(0, i) + mark(c.substr(i, j > i ? j - i : 0), own) + "</think>" +
                               mark(c.substr(j + 8), own);
            }
        }
        out.push_back(m);
    }
    return {out, mark_deep(tools, tags), true};
}

std::pair<std::string, std::vector<std::pair<size_t, size_t>>> unmark_think_literals(const std::string& prompt,
                                                                                      const LiteralTags& tags) {
    // the marks are 4-byte private-use characters (U+F0E01 ...), each its tag's
    std::map<std::string, std::string> back;
    for (const auto& [tag, m] : tags) back[m] = tag;
    std::string out;
    std::vector<std::pair<size_t, size_t>> spans;
    out.reserve(prompt.size());
    size_t pos = 0;
    for (size_t i = 0; i + 4 <= prompt.size();) {
        if ((unsigned char) prompt[i] == 0xF3) {
            const auto it = back.find(prompt.substr(i, 4));
            if (it != back.end()) {
                out.append(prompt, pos, i - pos);
                spans.emplace_back(out.size(), out.size() + it->second.size());
                out += it->second;
                i += 4;
                pos = i;
                continue;
            }
        }
        ++i;
    }
    out.append(prompt, pos, std::string::npos);
    return {out, spans};
}

// ---- conversions
bool g_env_to_user = false;

namespace {

json env_to_user(json messages) {
    if (!g_env_to_user || messages.empty() || jstr(messages[0], "role") != "system") return messages;
    const json& sys = jget(messages[0], "content");
    if (!sys.is_string()) return messages;
    const std::string& s = sys.get_ref<const std::string&>();
    // re.search((?:[^\n]*environment you are running in:\n)?<env>\n.*?\n</env>\n?, re.S): the first <env> block
    // that closes, from the start of the line before it when that line ends in the phrase
    size_t start = std::string::npos, end = 0;
    for (size_t o = s.find("<env>\n"); o != std::string::npos; o = s.find("<env>\n", o + 1)) {
        const size_t close = s.find("\n</env>", o + 5);
        if (close == std::string::npos) continue;
        start = o;
        const std::string_view lead = "environment you are running in:\n";
        if (o >= lead.size() && s.compare(o - lead.size(), lead.size(), lead) == 0) {
            const size_t phrase = o - lead.size();
            const size_t nl = phrase == 0 ? std::string::npos : s.rfind('\n', phrase - 1);
            start = nl == std::string::npos ? 0 : nl + 1;
        }
        end = close + 7;
        if (end < s.size() && s[end] == '\n') ++end;
        break;
    }
    if (start == std::string::npos) return messages;
    for (size_t i = 1; i < messages.size(); ++i) {
        if (jstr(messages[i], "role") != "user") continue;
        std::string block = s.substr(start, end - start);
        block = std::string(rstrip_chars(block, "\n"));
        const json& c = jget(messages[i], "content");
        json nc;
        if (c.is_string()) {
            nc = block + "\n\n" + c.get<std::string>();
        } else if (c.is_array()) {
            nc = json::array({{{"type", "text"}, {"text", block + "\n\n"}}});
            for (const auto& x : c) nc.push_back(x);
        } else {
            return messages;
        }
        messages[0]["content"] = s.substr(0, start) + s.substr(end);
        messages[i]["content"] = nc;
        return messages;
    }
    return messages;
}

}  // namespace

json late_system_to_user(json messages) {
    for (size_t i = 1; i < messages.size(); ++i)
        if (jstr(messages[i], "role") == "system") messages[i]["role"] = "user";
    return messages;
}

namespace {

/// #460: a request's "messages" (or a message's "tool_calls") as a list of objects; a JSON string of one is decoded
json object_list(const json& value, const std::string& name) {
    if (value.is_null()) return json::array();
    json v = value;
    if (value.is_string()) {
        if (!py_loads(value.get<std::string>(), v))
            throw RequestError(name + " must be a list of objects (a string was sent that is not JSON)");
    }
    if (!v.is_array()) throw RequestError(name + " must be a list of objects");
    for (const auto& m : v)
        if (!m.is_object()) throw RequestError(name + " must be a list of objects");
    return v;
}

// the same set of tools comes back in the order it was first seen (a client's MCP servers register in any order,
// and the template renders the tools before the system prompt: one swap would move the shared prefix back)
std::mutex g_orders_mu;
std::list<std::pair<std::set<std::string>, std::vector<std::string>>> g_orders;   // most recent first
constexpr size_t kOrdersMax = 64;

json stable_tool_order(const json& tools, const std::vector<std::string>& names) {
    if (tools.size() < 2) return tools;
    const std::set<std::string> key(names.begin(), names.end());
    if (key.size() != names.size()) return tools;
    std::vector<std::string> order;
    {
        std::lock_guard<std::mutex> lk(g_orders_mu);
        auto it = std::find_if(g_orders.begin(), g_orders.end(), [&](const auto& e) { return e.first == key; });
        if (it == g_orders.end()) {
            g_orders.emplace_front(key, names);
            while (g_orders.size() > kOrdersMax) g_orders.pop_back();
            return tools;
        }
        g_orders.splice(g_orders.begin(), g_orders, it);
        order = it->second;
    }
    std::map<std::string, size_t> pos;
    for (size_t k = 0; k < order.size(); ++k) pos[order[k]] = k;
    std::vector<size_t> idx(tools.size());
    for (size_t k = 0; k < idx.size(); ++k) idx[k] = k;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return pos[names[a]] < pos[names[b]]; });
    json out = json::array();
    for (const size_t k : idx) out.push_back(tools[k]);
    return out;
}

/// #592: a request's "tools" as tool objects, each with a name: the OpenAI shape {"type": "function", "function":
/// {...}} (`wrapper` "function"; a bare {"name": ...} too) or the Anthropic shape {"name": ...} (`wrapper` empty)
json tool_list(const json& value, const std::string& wrapper) {
    if (!truthy(value)) return json::array();
    const std::string shape = !wrapper.empty() ? R"({"type": "function", "function": {"name": ..., "parameters": {...}}})"
                                               : R"({"name": ..., "input_schema": {...}})";
    json tools;
    try {
        tools = object_list(value, "tools");
    } catch (const RequestError&) {
        throw RequestError("tools must be a list of tool objects, each " + shape);
    }
    const std::string key = !wrapper.empty() ? "parameters" : "input_schema";
    std::vector<std::string> names;
    for (size_t i = 0; i < tools.size(); ++i) {
        const json& t = tools[i];
        const json& fn = !wrapper.empty() && jstr(t, "type") == wrapper ? (t.contains(wrapper) ? t[wrapper] : t) : t;
        if (!fn.is_object() || !jget(fn, "name").is_string() || jstr(fn, "name").empty())
            throw RequestError("tools[" + std::to_string(i) + "] has no name: each tool must be " + shape);
        const json& schema = jget(fn, key);
        if (!schema.is_null() && !schema.is_object()) {
            static const std::map<json::value_t, std::string> names_of = {
                {json::value_t::string, "str"}, {json::value_t::array, "list"},   {json::value_t::boolean, "bool"},
                {json::value_t::number_integer, "int"}, {json::value_t::number_unsigned, "int"},
                {json::value_t::number_float, "float"}};
            const auto it = names_of.find(schema.type());
            throw RequestError("tools[" + std::to_string(i) + "] (" + jstr(fn, "name") + "): \"" + key +
                               "\" must be an object (the JSON schema of its parameters), not " +
                               (it == names_of.end() ? "NoneType" : it->second));
        }
        names.push_back(jstr(fn, "name"));
    }
    return stable_tool_order(tools, names);
}

}  // namespace

json tool_arguments(const json& raw) {
    // a call's arguments from a client's history -> the mapping the template renders; anything but an object is
    // kept as text under "arguments" (#510)
    if (raw.is_null() || (raw.is_string() && strip(raw.get_ref<const std::string&>()).empty())) return json::object();
    if (raw.is_object()) return raw;
    if (raw.is_string()) {
        json v;
        if (!py_loads(raw.get<std::string>(), v)) return {{"arguments", raw}};
        return v.is_object() ? v : json{{"arguments", raw}};
    }
    return {{"arguments", py_dumps(raw, false)}};
}

Converted openai_to_messages(const json& req) {
    Converted r;
    const json msgs = object_list(jget(req, "messages"), "messages");
    for (size_t i = 0; i < msgs.size(); ++i) {
        const json& m = msgs[i];
        std::string role = jstr(m, "role");
        json role_v = jget(m, "role");
        if (role == "developer") role_v = role = "system";
        // an image is read in a user, tool or assistant message; the system prompt takes text only
        const bool images = role == "user" || role == "tool" || role == "assistant";
        const json content = check_parts(jget(m, "content"));
        json out = {{"role", role_v}, {"content", images ? parts_of(content) : json(text_of(content))}};
        if (truthy(jget(m, "reasoning_content"))) out["reasoning_content"] = m["reasoning_content"];
        if (truthy(jget(m, "tool_calls"))) {
            json calls = json::array();
            for (const auto& c : object_list(m["tool_calls"], "tool_calls")) {
                const json& fn = c.contains("function") ? c["function"] : c;
                if (!fn.is_object())
                    throw RequestError("tool_calls must be a list of objects (each with a \"function\" object)");
                calls.push_back({{"function", {{"name", jget(fn, "name")}, {"arguments", tool_arguments(jget(fn, "arguments"))}}}});
            }
            out["tool_calls"] = calls;
        }
        r.messages.push_back(out);
    }
    json tools = json::array();
    for (const auto& t : tool_list(jget(req, "tools"), "function"))
        tools.push_back(jstr(t, "type") == "function" ? (t.contains("function") ? t["function"] : t) : t);
    r.tools = tools.empty() ? json(nullptr) : tools;
    // OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    const json& reasoning = jget(req, "reasoning");
    const json& given = jget(req, "reasoning_effort");
    const json effort = !given.is_null() && !(given.is_string() && given.get_ref<const std::string&>().empty())
                            ? given
                            : jget(reasoning, "effort");
    merge(r.kwargs, effort_kwargs(effort));
    // the vLLM / llama.cpp convention: {"chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"}}
    const json& ctk = jget(req, "chat_template_kwargs");
    if (ctk.is_object())
        for (auto it = ctk.begin(); it != ctk.end(); ++it) {
            if (it.key() == "enable_thinking" && !truthy(it.value())) {
                r.kwargs = {{"enable_thinking", false}};
            } else if (it.key() == "reasoning_effort" && !r.kwargs.contains("enable_thinking")) {
                merge(r.kwargs, effort_kwargs(it.value()));
            }
        }
    r.messages = env_to_user(late_system_to_user(r.messages));
    return r;
}

std::string pin_billing_stamp(std::string system) {
    // Claude Code's "x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;" changes
    // on every request (cch) and session (the version's 4th part): both are pinned to f's, as llama.cpp does
    // (ggml-org/llama.cpp#21793), only in a header at the start, within its first 160 characters
    const std::string_view head = "x-anthropic-billing-header:";
    if (!starts_with(system, head)) return system;
    std::string s = system;
    const size_t cch = system.find("cch=", head.size());
    if (cch != std::string::npos && cch <= 160) {
        const size_t v = cch + 4, end = system.find(';', cch + 4);
        if (end != std::string::npos && end > v && end - v <= 16)
            for (size_t k = v; k < end; ++k) s[k] = 'f';
    }
    const size_t cv = system.find("cc_version=");
    if (cv != std::string::npos && cv <= 160) {
        const size_t v = cv + 11, end = system.find(';', cv);
        if (end != std::string::npos && v < end && end - v <= 64) {
            const std::vector<std::string> parts = split(std::string_view(system).substr(v, end - v), ".");
            if (parts.size() > 3) {
                const size_t tail = v + parts[0].size() + parts[1].size() + parts[2].size() + 3;
                for (size_t k = tail; k < end; ++k) s[k] = 'f';
            }
        }
    }
    return s;
}

Converted anthropic_to_messages(json& req, bool think_unasked) {
    Converted r;
    const json& system = jget(req, "system");
    if (truthy(system))
        r.messages.push_back({{"role", "system"}, {"content", pin_billing_stamp(text_of(check_parts(system)))}});
    const json msgs = object_list(jget(req, "messages"), "messages");
    for (size_t i = 0; i < msgs.size(); ++i) {
        const json& m = msgs[i];
        const json content = check_parts(jget(m, "content"));
        if (content.is_string()) {
            r.messages.push_back({{"role", jget(m, "role")}, {"content", content}});
            continue;
        }
        const bool any_result = content.is_array() && std::any_of(content.begin(), content.end(), [](const json& b) {
            return b.is_object() && jstr(b, "type") == "tool_result";
        });
        if (jstr(m, "role") == "user" && has_image(content) && !any_result) {
            r.messages.push_back({{"role", "user"}, {"content", parts_of(content)}});
            continue;
        }
        std::string text, reasoning;
        bool any_text = false, any_reasoning = false;
        json calls = json::array(), parts = json::array();
        if (content.is_array())
            for (const auto& block : content) {
                const std::string kind = type_of(block);
                if (kind == "text") {
                    text += jstr(block, "text");
                    any_text = true;
                    parts.push_back(block);
                } else if (in(kind, kImageParts)) {
                    parts.push_back(block);
                } else if (kind == "thinking") {
                    reasoning += jstr(block, "thinking");
                    any_reasoning = true;
                } else if (kind == "tool_use") {
                    calls.push_back({{"function", {{"name", jget(block, "name")}, {"arguments", tool_arguments(jget(block, "input"))}}}});
                } else if (kind == "tool_result") {
                    // a tool's image (Claude Code's Read of a picture) reaches the encoder like a user's
                    r.messages.push_back({{"role", "tool"}, {"content", parts_of(check_parts(jget(block, "content")))}});
                }
            }
        if (any_text || !calls.empty() || any_reasoning || !parts.empty()) {
            json out = {{"role", jget(m, "role")}, {"content", has_image(parts) ? parts_of(parts) : json(text)}};
            if (any_reasoning) out["reasoning_content"] = reasoning;
            if (!calls.empty()) out["tool_calls"] = calls;
            r.messages.push_back(out);
        }
    }
    json tools = json::array();
    for (const auto& t : tool_list(jget(req, "tools"), ""))
        tools.push_back({{"name", t["name"]}, {"description", t.contains("description") ? t["description"] : json("")},
                         {"parameters", t.contains("input_schema") ? t["input_schema"] : json::object()}});
    r.tools = tools.empty() ? json(nullptr) : tools;
    // "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N}; "output_config": {"effort": ...}
    const json thinking = jget(req, "thinking");
    const json& oc = jget(req, "output_config");
    const json effort = oc.is_object() ? jget(oc, "effort") : json(nullptr);
    // budget_tokens is a real budget: the thinking is closed after that many tokens (a request's own
    // reasoning_budget_tokens wins)
    const json bt = thinking.is_object() && jstr(thinking, "type") != "disabled" ? jget(thinking, "budget_tokens") : json(nullptr);
    if (is_int(bt) && bt.get<long long>() > 0 && jget(req, "reasoning_budget_tokens").is_null())
        req["reasoning_budget_tokens"] = bt;
    if (thinking.is_object() && jstr(thinking, "type") == "disabled") {
        r.kwargs["enable_thinking"] = false;
    } else if (truthy(effort)) {
        merge(r.kwargs, effort_kwargs(effort));
    } else if (thinking.is_object() && truthy(jget(thinking, "budget_tokens"))) {
        merge(r.kwargs, budget_effort(thinking["budget_tokens"]));
    } else if (thinking.is_null() && !truthy(jget(req, "reasoning_budget_tokens")) && !think_unasked) {
        // the config's "anthropic_thinking": "on_request": a request that asks for no thinking gets none
        r.kwargs["enable_thinking"] = false;
    }
    r.messages = env_to_user(late_system_to_user(r.messages));
    return r;
}

// ---------------------------------------------------------------------------------------------- output parser
long param_end(std::string_view text, bool final) {
    // the first </parameter> followed (after whitespace) by the next <parameter= or </function>: the same text
    // inside a value (a file that documents the call format, #210) is part of the value
    size_t at = text.find(kParamEnd);
    while (at != std::string_view::npos) {
        const std::string_view after = lstrip(text.substr(at + kParamEnd.size()));
        if (starts_with(after, kParamStart) || starts_with(after, kFuncEnd)) return (long) at;
        if (after.empty() || starts_with(kParamStart, after) || starts_with(kFuncEnd, after)) return final ? (long) at : -2;
        at = text.find(kParamEnd, at + 1);
    }
    return -1;
}

long call_end(std::string_view text) {
    std::string_view s = lstrip(text);
    size_t pos = text.size() - s.size();
    if (!starts_with(s, kFuncStart)) {
        if (starts_with(kFuncStart, s)) return -1;
        const size_t f = text.find(kCallEnd);
        return f == std::string_view::npos ? -1 : (long) f;
    }
    size_t gt = text.find('>', pos);
    if (gt == std::string_view::npos) return -1;
    pos = gt + 1;
    while (true) {
        const std::string_view rest = text.substr(pos);
        s = lstrip(rest);
        pos += rest.size() - s.size();
        if (starts_with(s, kParamStart)) {
            gt = text.find('>', pos);
            if (gt == std::string_view::npos) return -1;
            const long end = param_end(text.substr(gt + 1));
            if (end < 0) return -1;
            pos = gt + 1 + (size_t) end + kParamEnd.size();
        } else if (starts_with(s, kFuncEnd)) {
            const std::string_view r2 = text.substr(pos + kFuncEnd.size());
            const std::string_view s2 = lstrip(r2);
            if (starts_with(s2, kCallEnd)) return (long) (pos + kFuncEnd.size() + r2.size() - s2.size());
            if (starts_with(kCallEnd, s2)) return -1;
            const size_t f = text.find(kCallEnd, pos);
            return f == std::string_view::npos ? -1 : (long) f;
        } else if (s.empty() || starts_with(kParamStart, s) || starts_with(kFuncEnd, s)) {
            return -1;
        } else {
            const size_t f = text.find(kCallEnd, pos);
            return f == std::string_view::npos ? -1 : (long) f;
        }
    }
}

long function_end(std::string_view text) {
    std::string_view s = lstrip(text);
    size_t pos = text.size() - s.size();
    if (!starts_with(s, kFuncStart)) return -1;
    size_t gt = text.find('>', pos);
    if (gt == std::string_view::npos) return -1;
    pos = gt + 1;
    while (true) {
        const std::string_view rest = text.substr(pos);
        s = lstrip(rest);
        pos += rest.size() - s.size();
        if (starts_with(s, kParamStart)) {
            gt = text.find('>', pos);
            if (gt == std::string_view::npos) return -1;
            const long end = param_end(text.substr(gt + 1));
            if (end < 0) return -1;
            pos = gt + 1 + (size_t) end + kParamEnd.size();
        } else if (starts_with(s, kFuncEnd)) {
            return (long) (pos + kFuncEnd.size());
        } else {
            return -1;
        }
    }
}

std::pair<std::string, std::string> tool_choice_of(const json& tc) {
    if (tc.is_null()) return {"auto", ""};
    if (tc.is_string()) {
        const std::string& s = tc.get_ref<const std::string&>();
        return s == "auto" || s == "none" || s == "required" ? std::make_pair(s, std::string()) : std::make_pair(std::string("unknown"), std::string());
    }
    if (tc.is_object()) {
        const std::string kind = jstr(tc, "type");
        if (kind == "auto" || kind == "none") return {kind, ""};
        if (kind == "any" || kind == "required") return {"required", ""};
        if (kind == "function" || kind == "tool") {
            const json& fn = jget(tc, "function");
            std::string name = fn.is_object() ? jstr(fn, "name") : "";
            if (name.empty()) name = jstr(tc, "name");
            if (!name.empty()) return {"named", name};
        }
    }
    return {"unknown", ""};
}

std::optional<std::string> forced_call(const json& tool_choice, const json& tools) {
    auto [kind, name] = tool_choice_of(tool_choice);
    std::set<std::string> names;
    if (tools.is_array())
        for (const auto& t : tools)
            if (t.is_object()) names.insert(jstr(t, "name"));
    if (kind == "auto" || kind == "none") return std::nullopt;
    if (kind == "required" && names.size() == 1) {   // one tool to call: name it, the model cannot invent another
        kind = "named";
        name = *names.begin();
    }
    if (kind == "required" && !names.empty()) return std::string(kCallStart) + "\n<function=";
    if (kind == "named" && names.count(name)) return std::string(kCallStart) + "\n<function=" + name + ">\n";
    say("[lamina] tool_choice " + utf8_prefix(py_dumps(tool_choice), 200) +
        " is not supported here (or names no tool of the request): the model decides, as with \"auto\"");
    return std::nullopt;
}

namespace {

/// _JSON_FENCE.match: ```[ \t]*(?:json)?[ \t]*\r?\n\s* (case-insensitive) -> its end, or npos
size_t json_fence(std::string_view s) {
    if (!starts_with(s, "```")) return std::string_view::npos;
    size_t i = 3;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    if (i + 4 <= s.size() && lower(s.substr(i, 4)) == "json") i += 4;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    if (i < s.size() && s[i] == '\r') ++i;
    if (i >= s.size() || s[i] != '\n') return std::string_view::npos;
    ++i;
    return s.size() - lstrip(s.substr(i)).size();
}

}  // namespace

std::shared_ptr<ToolCall> json_tool_call(std::string_view body0, const json& schemas) {
    std::string_view body = strip(body0);
    const size_t f = json_fence(body);
    if (f != std::string_view::npos) {
        body = rstrip(body.substr(f));
        if (ends_with(body, "```")) body = body.substr(0, body.size() - 3);
    }
    json obj;
    if (!py_loads(body, obj)) return nullptr;
    if (!obj.is_object() || !jget(obj, "name").is_string() || !schemas.contains(jstr(obj, "name"))) return nullptr;
    json args = obj.contains("arguments") ? obj["arguments"] : obj.contains("parameters") ? obj["parameters"] : json::object();
    if (args.is_string()) {
        json v;
        if (!py_loads(args.get<std::string>(), v)) return nullptr;
        args = v;
    }
    if (args.is_null()) args = json::object();
    if (!args.is_object()) return nullptr;
    auto call = std::make_shared<ToolCall>();
    call->name = jstr(obj, "name");
    call->arguments = args;
    return call;
}

/// a parameter value not declared a string: JSON; failing that, for a boolean or null declaration, the Python
/// spellings the model sometimes writes (True/False/None, any case), as vLLM's and llama.cpp's Qwen3-Coder parsers
/// read them; failing that, the text
static json param_value(const std::string& value, const json& decl) {
    json v;
    if (py_loads(value, v)) return v;
    auto declares = [&decl](std::string_view type) {
        if (decl.is_string()) return decl.get_ref<const std::string&>() == type;
        if (decl.is_array())
            for (const auto& t : decl)
                if (t.is_string() && t.get_ref<const std::string&>() == type) return true;
        return false;
    };
    std::string low(strip(value));
    for (char& c : low) c = (char) std::tolower((unsigned char) c);
    if (declares("boolean") && (low == "true" || low == "false")) return json(low == "true");
    if (declares("null") && (low == "none" || low == "null")) return json(nullptr);
    return json(value);
}

ToolCall parse_tool_call(std::string_view body0, const json& schema) {
    const std::string_view body = strip(body0);
    if (!starts_with(body, kFuncStart) || body.find('>') == std::string_view::npos)
        throw std::invalid_argument("malformed tool call: " + utf8_prefix(body, 80));
    const size_t gt = body.find('>');
    ToolCall call;
    call.name = std::string(strip(body.substr(kFuncStart.size(), gt - kFuncStart.size())));
    std::string_view rest = body.substr(gt + 1);
    const json& props = jget(jget(schema, "parameters"), "properties");
    while (rest.find(kParamStart) != std::string_view::npos) {
        rest = rest.substr(rest.find(kParamStart) + kParamStart.size());
        const size_t pg = rest.find('>');
        if (pg == std::string_view::npos) throw std::invalid_argument("substring not found");
        const std::string pname(rest.substr(0, pg));
        rest = rest.substr(pg + 1);
        const long end = param_end(rest, true);
        std::string value(end >= 0 ? rest.substr(0, (size_t) end) : rest);
        rest = end >= 0 ? rest.substr((size_t) end + kParamEnd.size()) : std::string_view();
        if (starts_with(value, "\n")) value.erase(0, 1);
        if (ends_with(value, "\n")) value.pop_back();
        const json& decl = jget(jget(props, pname), "type");
        if (decl.is_string() && decl.get_ref<const std::string&>() == "string") {
            call.arguments[pname] = value;
        } else {
            call.arguments[pname] = param_value(value, decl);
        }
    }
    return call;
}

namespace {
constexpr size_t kRcallMax = 32768;   // the most of a <tool_call> inside the reasoning held while waiting for its end
}

OutputParser::OutputParser(bool thinking, const json& tools, bool stream_tools, bool recover)
    : state_(thinking ? "reasoning" : "content"), stream_tools_(stream_tools), recover_(recover) {
    if (tools.is_array())
        for (const auto& t : tools) schemas_[jstr(t, "name")] = t;
    reset_scan();
}

bool OutputParser::ok_at(size_t p) {
    // buf[p] would open a call: at the start of a line, outside a code fence and inline code
    const std::string fence = fence_, line = line_;
    const long ticks = ticks_;
    track(buf_.substr(0, p));
    const bool ok = opener_ok();
    fence_ = fence;
    line_ = line;
    ticks_ = ticks;
    return ok;
}

std::string OutputParser::follower(std::string_view after) const {
    if (starts_with(after, kFuncStart)) return "call";
    if (starts_with(after, kParamStart)) {
        const size_t gt = after.find('>');
        if (gt == std::string_view::npos) return after.find('\n') == std::string_view::npos ? "wait" : "text";
        return schemas_.contains(std::string(strip(after.substr(kParamStart.size(), gt - kParamStart.size())))) ? "param"
                                                                                                              : "text";
    }
    if (starts_with(after, "{")) return schemas_.empty() ? "text" : "json";
    if (starts_with(after, "```") || (!after.empty() && starts_with("```", after))) {
        // #2037: the JSON call inside a Markdown fence in the explicit wrapper
        const size_t f = json_fence(after);
        if (f != std::string_view::npos) {
            const std::string_view rest = after.substr(f);
            if (starts_with(rest, "{")) return schemas_.empty() ? "text" : "json";
            return rest.empty() ? "wait" : "text";
        }
        return after.find('\n') == std::string_view::npos ? "wait" : "text";
    }
    if (after.empty() || starts_with(kFuncStart, after) || starts_with(kParamStart, after)) return "wait";
    return "text";
}

std::pair<long, bool> OutputParser::bare_opener() {
    size_t p = buf_.find(kFuncStart);
    while (p != std::string::npos) {
        if (ok_at(p)) {
            const size_t gt = buf_.find('>', p);
            const std::string name(strip(std::string_view(buf_).substr(
                p + kFuncStart.size(), (gt == std::string::npos ? buf_.size() : gt) - p - kFuncStart.size())));
            if (gt == std::string::npos && !contains_space(name)) return {(long) p, false};
            if (gt != std::string::npos && schemas_.contains(name)) return {(long) p, true};
        }
        p = buf_.find(kFuncStart, p + 1);
    }
    return {-1, false};
}

std::string OutputParser::track(std::string text) {
    // follow the text that has gone out: the open code fence, the current line, the backticks of the paragraph
    if (text.find('\n') == std::string::npos) {
        line_ += text;
        return text;
    }
    const std::vector<std::string> parts = split(text, "\n");
    for (size_t k = 0; k < parts.size(); ++k) {
        if (k + 1 < parts.size()) {
            const std::string line = line_ + parts[k];
            line_.clear();
            const std::string_view s = lstrip(line);
            if (!fence_.empty()) {
                if (starts_with(s, fence_ + fence_ + fence_)) fence_.clear();
            } else if ((starts_with(s, "```") || starts_with(s, "~~~")) &&
                       s.substr(3).find(std::string(3, s[0])) == std::string_view::npos) {
                fence_ = std::string(1, s[0]);
            } else if (s.empty()) {
                ticks_ = 0;
            } else {
                ticks_ += (long) std::count(line.begin(), line.end(), '`');
            }
        } else {
            line_ += parts[k];
        }
    }
    return text;
}

bool OutputParser::opener_ok() const { return fence_.empty() && strip(line_).empty() && ticks_ % 2 == 0; }

bool OutputParser::in_code() const {
    return !fence_.empty() || (ticks_ + (long) std::count(line_.begin(), line_.end(), '`')) % 2 == 1;
}

std::optional<bool> OutputParser::quoted_think_end(size_t i) const {
    // #537: a </think> right after a quote or backtick and followed by anything but a line break is quoted
    const char before = i ? buf_[i - 1] : (line_.empty() ? '\0' : line_.back());
    if (before != '"' && before != '\'' && before != '`') return false;
    const size_t a = i + kThinkEnd.size();
    if (a >= buf_.size()) return std::nullopt;
    return buf_[a] != '\r' && buf_[a] != '\n';
}

std::vector<Event> OutputParser::release(bool deliver) {
    std::vector<Event> out;
    for (auto& [raw, call] : pending_) {
        if (call != nullptr && deliver) {
            out.push_back({"tool_call", "", call});
            ++rescued;
        } else {
            if (call != nullptr) ++refused;
            if (!raw.empty()) out.push_back({"reasoning", track(raw), nullptr});
        }
    }
    pending_.clear();
    return out;
}

void OutputParser::reset_scan() {
    call_end_checked_ = 0;
    sp_ = 0;
    ss_ = "name";
    scall_ = nullptr;
    sfirst_ = true;
    sval_started_ = false;
    sdeclared_ = json::object();
    sparam_.clear();
}

long OutputParser::wrapped_call_end() {
    // without a closing tag the structural walk cannot finish: search only the new text (with an overlap for a
    // tag split across feeds) until a candidate appears; recovery can rewrite the buffer, so it always walks
    if (!recover_ && call_end_checked_ >= 0) {
        const size_t start = (size_t) std::max<long>(0, call_end_checked_ - (long) kCallEnd.size() + 1);
        if (buf_.find(kCallEnd, start) == std::string::npos) {
            call_end_checked_ = (long) buf_.size();
            return -1;
        }
        call_end_checked_ = -1;
    }
    return call_end(buf_);
}

namespace {

/// the longest suffix of `text` that is a proper prefix of one of `tags`
size_t hold(std::string_view text, std::initializer_list<std::string_view> tags) {
    size_t best = 0;
    const size_t size = text.size();
    for (const std::string_view tag : tags) {
        if (tag.size() <= 1) continue;
        size_t start = text.find(tag[0], size >= tag.size() - 1 ? size - tag.size() + 1 : 0);
        while (start != std::string_view::npos && size - start > best) {
            if (starts_with(tag, text.substr(start))) {
                best = size - start;
                break;
            }
            start = text.find(tag[0], start + 1);
        }
    }
    return best;
}

std::string json_string_body(const std::string& s) {
    const std::string d = py_dumps(json(s));   // ensure_ascii, as json.dumps(value)[1:-1]
    return d.substr(1, d.size() - 2);
}

}  // namespace

std::vector<Event> OutputParser::scan() {
    std::vector<Event> out;
    auto args = [&](const std::string& s) {
        if (!s.empty()) out.push_back({"tool_args", s, scall_});
    };
    while (true) {
        std::string_view rest = std::string_view(buf_).substr(std::min(sp_, buf_.size()));
        if (ss_ == "name") {
            const size_t a = rest.find(kFuncStart);
            const size_t b = a != std::string_view::npos ? rest.find('>', a + 10) : std::string_view::npos;
            if (b == std::string_view::npos) return out;
            scall_ = std::make_shared<ToolCall>();
            scall_->name = std::string(strip(rest.substr(a + 10, b - a - 10)));   // as parse_tool_call reads it
            const json& props = jget(jget(jget(schemas_, scall_->name), "parameters"), "properties");
            sdeclared_ = json::object();
            if (props.is_object())
                for (auto it = props.begin(); it != props.end(); ++it) sdeclared_[it.key()] = jget(it.value(), "type");
            out.push_back({"tool_start", "", scall_});
            args("{");
            sp_ += b + 1;
            ss_ = "between";
        } else if (ss_ == "between") {
            const std::string_view stripped = lstrip(rest);
            sp_ += rest.size() - stripped.size();
            if (starts_with(stripped, kParamStart)) {
                const size_t b = stripped.find('>');
                if (b == std::string_view::npos) return out;
                const std::string pname(stripped.substr(11, b - 11));
                sparam_ = pname;
                args(std::string(sfirst_ ? "" : ",") + py_dumps(json(pname)) + ":");
                sfirst_ = false;
                sp_ += b + 1;
                const json& d = jget(sdeclared_, pname);
                if (d.is_string() && d.get_ref<const std::string&>() == "string") {
                    args("\"");
                    ss_ = "str";
                    sval_started_ = false;
                } else {
                    ss_ = "raw";
                }
            } else if (starts_with(stripped, kFuncEnd)) {
                args("}");
                sp_ += kFuncEnd.size();
                ss_ = "done";
            } else {
                return out;   // a tag still arriving (or trailing text): wait
            }
        } else if (ss_ == "str") {
            if (!sval_started_) {
                if (rest.empty()) return out;
                if (rest[0] == '\n') {   // the value's leading newline is not part of it
                    ++sp_;
                    rest = rest.substr(1);
                }
                sval_started_ = true;
            }
            const long end = param_end(rest);
            if (end >= 0) {
                std::string value(rest.substr(0, (size_t) end));
                if (ends_with(value, "\n")) value.pop_back();
                args(json_string_body(value) + "\"");
                sp_ += (size_t) end + kParamEnd.size();
                ss_ = "between";
                continue;
            }
            // an undecided </parameter> (-2) is held from its start, like a tag still arriving
            size_t safe = end == -2 ? rest.find(kParamEnd) : rest.size() - hold(rest, {kParamEnd});
            if (safe > 0 && safe != std::string_view::npos && rest[safe - 1] == '\n') --safe;   // the newline before it
            if (safe > 0 && safe != std::string_view::npos) {
                args(json_string_body(std::string(rest.substr(0, safe))));
                sp_ += safe;
            }
            return out;
        } else if (ss_ == "raw") {
            const long end = param_end(rest);
            if (end < 0) return out;
            std::string value(rest.substr(0, (size_t) end));
            if (starts_with(value, "\n")) value.erase(0, 1);
            if (ends_with(value, "\n")) value.pop_back();
            args(py_dumps(param_value(value, jget(sdeclared_, sparam_)), false));
            sp_ += (size_t) end + kParamEnd.size();
            ss_ = "between";
        } else {
            return out;
        }
    }
}

std::vector<Event> OutputParser::close_scan() {
    std::vector<Event> out;
    if (scall_ == nullptr || ss_ == "done") return out;
    std::string tail;
    if (ss_ == "str") tail += "\"";
    else if (ss_ == "raw") tail += py_dumps(json(std::string(strip_chars(std::string_view(buf_).substr(std::min(sp_, buf_.size())), "\n"))));
    tail += "}";
    out.push_back({"tool_args", tail, scall_});
    ss_ = "done";
    return out;
}

std::vector<Event> OutputParser::feed(std::string_view delta) {
    buf_ += delta;
    std::vector<Event> out;
    auto append = [&](std::vector<Event> evs) {
        for (auto& e : evs) out.push_back(std::move(e));
    };
    while (true) {
        if (state_ == "rcall") {
            // #804: a <tool_call> inside the reasoning, with tools declared: held whole until it ends; a declared
            // name is then a tool call, anything else stays reasoning text
            const std::string body = buf_.substr(kCallStart.size());
            const long end = call_end(body);
            const size_t think = body.find(kThinkEnd);
            if (end >= 0 && (think == std::string::npos || think >= (size_t) end)) {
                std::shared_ptr<ToolCall> call;
                try {
                    const std::string_view b = strip(std::string_view(body).substr(0, (size_t) end));
                    std::string name(b.size() >= kFuncStart.size() ? b.substr(kFuncStart.size()) : std::string_view());
                    name = std::string(strip(name.substr(0, std::min(name.find('>'), name.size()))));
                    if (schemas_.contains(name))
                        call = std::make_shared<ToolCall>(parse_tool_call(std::string_view(body).substr(0, (size_t) end),
                                                                          schemas_[name]));
                } catch (const std::invalid_argument&) {
                }
                const std::string raw = buf_.substr(0, kCallStart.size() + (size_t) end + kCallEnd.size());
                if (call != nullptr || !pending_.empty()) {
                    pending_.push_back({raw, call});   // settled by what follows it
                } else {
                    out.push_back({"reasoning", track(raw), nullptr});
                }
                buf_ = body.substr(std::min(body.size(), (size_t) end + kCallEnd.size()));
                state_ = "reasoning";
            } else if (think != std::string::npos) {   // the thinking ended inside it: it never was a call
                append(release(false));
                out.push_back({"reasoning", track(buf_.substr(0, kCallStart.size() + think)), nullptr});
                buf_ = body.substr(think);
                state_ = "reasoning";
            } else if (utf8_len(body) > kRcallMax) {   // a tag in the prose that never closes
                append(release(false));
                out.push_back({"reasoning", track(buf_), nullptr});
                buf_.clear();
                state_ = "reasoning";
            } else {
                return out;
            }
        } else if (state_ == "reasoning") {
            if (!pending_.empty()) {
                // calls wait for what follows: more calls or whitespace keep them, </think> makes them acts, any
                // other text means they were quoted
                const std::string_view stripped = lstrip(buf_);
                if (stripped.size() < buf_.size()) {
                    pending_.back().raw += buf_.substr(0, buf_.size() - stripped.size());
                    buf_ = std::string(stripped);
                }
                if (buf_.empty()) return out;
                if (starts_with(buf_, kCallStart)) {
                    state_ = "rcall";
                } else if (starts_with(buf_, kThinkEnd)) {
                    append(release(true));
                } else if (starts_with(kCallStart, buf_) || starts_with(kThinkEnd, buf_)) {
                    return out;
                } else {
                    append(release(false));
                }
                continue;
            }
            const size_t i = buf_.find(kThinkEnd);
            const size_t tool = schemas_.empty() ? std::string::npos : buf_.find(kCallStart);
            if (tool != std::string::npos && (i == std::string::npos || tool < i)) {
                if (tool) out.push_back({"reasoning", track(buf_.substr(0, tool)), nullptr});
                if (opener_ok()) {
                    buf_ = buf_.substr(tool);
                    state_ = "rcall";
                } else {   // mid-sentence, in a fence or in inline code: a quote
                    out.push_back({"reasoning", track(std::string(kCallStart)), nullptr});
                    buf_ = buf_.substr(tool + kCallStart.size());
                }
                continue;
            }
            if (i == std::string::npos) {
                const size_t keep = schemas_.empty() ? hold(buf_, {kThinkEnd}) : hold(buf_, {kThinkEnd, kCallStart});
                if (buf_.size() > keep) {
                    out.push_back({"reasoning", track(buf_.substr(0, buf_.size() - keep)), nullptr});
                    buf_ = buf_.substr(buf_.size() - keep);
                }
                return out;
            }
            const std::optional<bool> quoted = quoted_think_end(i);
            if (!quoted.has_value()) {   // held like a partial tag until the character after it is here
                if (i) {
                    out.push_back({"reasoning", track(buf_.substr(0, i)), nullptr});
                    buf_ = buf_.substr(i);
                }
                return out;
            }
            if (*quoted) {   // #537: a quoted </think> is reasoning text
                out.push_back({"reasoning", track(buf_.substr(0, i + kThinkEnd.size())), nullptr});
                buf_ = buf_.substr(i + kThinkEnd.size());
                continue;
            }
            if (i) out.push_back({"reasoning", track(buf_.substr(0, i)), nullptr});
            buf_ = buf_.substr(i + kThinkEnd.size());
            state_ = "content";
            lead_ = true;
            fence_.clear();   // the answer's own text starts here
            line_.clear();
            ticks_ = 0;
        } else if (state_ == "content") {
            if (lead_) {   // newlines right after </think> or a call
                const std::string_view stripped = lstrip_chars(buf_, "\n");
                if (stripped.empty()) {
                    buf_.clear();
                    return out;
                }
                buf_ = std::string(stripped);
                lead_ = false;
            }
            // a request that declares no tools gets the model's text verbatim, tool-call tags included
            const size_t i = schemas_.empty() ? std::string::npos : buf_.find(kCallStart);
            const auto [b, decided] = recover_ && !schemas_.empty() ? bare_opener() : std::make_pair(-1L, false);
            if (b >= 0 && (i == std::string::npos || (size_t) b < i)) {
                // recover: <function=NAME> alone at the start of a line - the call without its <tool_call>
                size_t j = (size_t) b;
                while (j > 0 && buf_[j - 1] == '\n') --j;
                if (j > 0 && !strip(std::string_view(buf_).substr(0, j)).empty())
                    out.push_back({"content", track(buf_.substr(0, j)), nullptr});
                if (!decided) {
                    buf_ = buf_.substr(j);
                    return out;
                }
                buf_ = buf_.substr((size_t) b);
                state_ = "call";
                bare_ = true;
                continue;
            }
            if (i == std::string::npos) {
                // hold a partial tag AND the newlines before it: if a call follows, they are dropped
                size_t j = buf_.size() - (schemas_.empty() ? 0 : hold(buf_, {kCallStart}));
                if (recover_ && !schemas_.empty()) {   // a line that may be becoming a bare <function=
                    const size_t nl = buf_.rfind('\n');
                    const size_t k = nl == std::string::npos ? 0 : nl + 1;
                    const std::string tail = buf_.substr(k);
                    if (!tail.empty() && starts_with(kFuncStart, tail) && ok_at(k)) j = std::min(j, k);
                }
                while (j > 0 && buf_[j - 1] == '\n') --j;
                if (j > 0) {
                    out.push_back({"content", track(buf_.substr(0, j)), nullptr});
                    buf_ = buf_.substr(j);
                }
                return out;
            }
            // #1058: an opener inside a code fence or inline code is text (a quoted example), never a call
            {
                const std::string fence = fence_, line = line_;
                const long ticks = ticks_;
                track(buf_.substr(0, i));
                const bool code = in_code();
                fence_ = fence;
                line_ = line;
                ticks_ = ticks;
                if (code) {
                    out.push_back({"content", track(buf_.substr(0, i + kCallStart.size())), nullptr});
                    buf_ = buf_.substr(i + kCallStart.size());
                    continue;
                }
            }
            // a call is <tool_call> and then (after whitespace) <function=; the tag with anything else after it is
            // prose that names the format; until its follower has arrived it is held, like a partial tag
            std::string after(lstrip(std::string_view(buf_).substr(i + kCallStart.size())));
            const std::string kind = recover_ ? follower(after) : "";
            if (kind == "param" || kind == "json") {
                if (i && !strip(std::string_view(buf_).substr(0, i)).empty())
                    out.push_back({"content", track(std::string(rstrip_chars(std::string_view(buf_).substr(0, i), "\n"))), nullptr});
                std::string rest = buf_.substr(i + kCallStart.size());
                if (kind == "param") {   // <parameter=NAME> as the opener: the call's <function=NAME>
                    const size_t at = rest.size() - after.size();
                    rest = rest.substr(0, at) + std::string(kFuncStart) + after.substr(kParamStart.size());
                }
                buf_ = rest;
                state_ = kind == "param" ? "call" : "json";
                continue;
            }
            if (kind == "wait") after.clear();
            if (!after.empty() && !starts_with(after, kFuncStart) && !starts_with(kFuncStart, after)) {
                out.push_back({"content", track(buf_.substr(0, i + kCallStart.size())), nullptr});
                buf_ = buf_.substr(i + kCallStart.size());
                continue;
            }
            if (!starts_with(after, kFuncStart)) {
                size_t j = i;
                while (j > 0 && buf_[j - 1] == '\n') --j;
                if (j > 0) {
                    out.push_back({"content", track(buf_.substr(0, j)), nullptr});
                    buf_ = buf_.substr(j);
                }
                return out;
            }
            if (i && !strip(std::string_view(buf_).substr(0, i)).empty())
                out.push_back({"content", track(std::string(rstrip_chars(std::string_view(buf_).substr(0, i), "\n"))), nullptr});
            buf_ = buf_.substr(i + kCallStart.size());
            state_ = "call";
        } else if (state_ == "json") {
            // recover: {"name": ..., "arguments": {...}} inside the wrapper is the call once its </tool_call> has
            // arrived and it names a declared tool; anything else is the text it is
            const size_t i = buf_.find(kCallEnd);
            if (i == std::string::npos) return out;
            auto call = json_tool_call(strip(std::string_view(buf_).substr(0, i)), schemas_);
            if (call == nullptr) {
                out.push_back({"content", track(std::string(kCallStart) + buf_.substr(0, i + kCallEnd.size())), nullptr});
            } else {
                out.push_back({"tool_call", "", call});
            }
            buf_ = buf_.substr(i + kCallEnd.size());
            state_ = "content";
            lead_ = call != nullptr;
        } else {
            long i = -1;
            std::optional<size_t> drop;
            if (bare_) {   // no wrapper: the call ends at its </function>; a stray </tool_call> after it is dropped
                i = function_end(buf_);
                if (i >= 0) {
                    const std::string_view after = lstrip(std::string_view(buf_).substr((size_t) i));
                    if (starts_with(after, kCallEnd)) {
                        drop = (buf_.size() - (size_t) i) - after.size() + kCallEnd.size();
                    } else if (!after.empty() && !starts_with(kCallEnd, after)) {
                        drop = 0;
                    }
                }
            } else {
                i = wrapped_call_end();
                if (i >= 0) drop = kCallEnd.size();
                if (recover_) {   // a batch in one wrapper: the next call follows this one's </function>
                    const long f = function_end(buf_);
                    std::string nxt = f >= 0 ? std::string(lstrip(std::string_view(buf_).substr((size_t) f))) : "";
                    if (starts_with(nxt, kParamStart) && follower(nxt) == "param") {
                        const size_t at = buf_.size() - nxt.size();
                        buf_ = buf_.substr(0, at) + std::string(kFuncStart) + nxt.substr(kParamStart.size());
                        nxt = buf_.substr(at);
                    }
                    if (starts_with(nxt, kFuncStart)) {
                        i = f;
                        drop = 0;
                        batch_ = true;
                    }
                }
            }
            if (stream_tools_) {
                if (i >= 0) {
                    const std::string whole = buf_;
                    buf_ = buf_.substr(0, (size_t) i);   // scan only the body
                    append(scan());
                    append(close_scan());
                    buf_ = whole;
                } else {
                    append(scan());
                }
            }
            if (i < 0 || !drop.has_value()) return out;
            const std::string body = buf_.substr(0, (size_t) i);
            buf_ = buf_.substr(std::min(buf_.size(), (size_t) i + *drop));
            const bool batch = batch_;
            batch_ = false;
            append(finish_call(body));
            if (batch && !out.empty() && out.back().kind == "tool_call") {
                state_ = "call";
                lead_ = false;
            }
        }
    }
}

std::vector<Event> OutputParser::finish(std::string_view reason) {
    std::vector<Event> out;
    auto append = [&](std::vector<Event> evs) {
        for (auto& e : evs) out.push_back(std::move(e));
    };
    if (!pending_.empty()) append(release((reason.empty() || reason == "stop") && state_ == "reasoning" && buf_.empty()));
    if (state_ == "call" && bare_ && function_end(buf_) >= 0) {   // recover: a bare call is whole at its </function>
        const long i = function_end(buf_);
        const std::string body = buf_.substr(0, (size_t) i);
        const std::string rest(strip(std::string_view(buf_).substr((size_t) i)));
        buf_.clear();
        append(finish_call(body));
        if (!rest.empty() && !starts_with(kCallEnd, rest)) out.push_back({"content", track(rest), nullptr});
        return out;
    }
    if (state_ == "call" && stream_tools_ && scall_ != nullptr) {
        append(scan());   // the output ended inside a call that was already announced
        const long i = ss_ == "done" ? function_end(buf_) : -1;
        if (i >= 0) {   // only its </tool_call> is missing: the call parsed (the streamed view holds no arguments)
            const std::string body = buf_.substr(0, (size_t) i);
            const std::string rest(strip(std::string_view(buf_).substr((size_t) i)));
            buf_.clear();
            append(finish_call(body));
            if (!rest.empty() && !starts_with(kCallEnd, rest)) out.push_back({"content", track(rest), nullptr});
            return out;
        }
        buf_.clear();
        reset_scan();
        return out;
    }
    if (state_ == "reasoning" && buf_ == kThinkEnd) {   // the output ends on a held </think>: its end
        buf_.clear();
        state_ = "content";
    }
    if (!buf_.empty()) {
        // an unfinished call inside the reasoning (#804) is reasoning text, never a call
        const std::string kind = state_ == "reasoning" || state_ == "rcall" ? "reasoning" : "content";
        const std::string text = (state_ == "call" || state_ == "json") && !bare_ ? std::string(kCallStart) + buf_ : buf_;
        out.push_back({kind, text, nullptr});
        buf_.clear();
    }
    return out;
}

std::vector<Event> OutputParser::finish_call(std::string body) {
    const std::string_view b = strip(body);
    std::string name(b.size() >= kFuncStart.size() ? b.substr(kFuncStart.size()) : std::string_view());
    name = std::string(strip(name.substr(0, std::min(name.find('>'), name.size()))));
    const bool bare = bare_;
    bare_ = false;
    ToolCall call;
    try {
        call = parse_tool_call(body, jget(schemas_, name));
    } catch (const std::invalid_argument& e) {
        if (!recover_) throw ToolCallError(std::string("the model wrote a tool call that does not parse (") + e.what() + ")");
        reset_scan();
        state_ = "content";
        return {{"content", track(bare ? body : std::string(kCallStart) + body + std::string(kCallEnd)), nullptr}};
    }
    if (scall_ != nullptr) call.id = scall_->id;
    reset_scan();
    state_ = "content";
    lead_ = true;
    return {{"tool_call", "", std::make_shared<ToolCall>(std::move(call))}};
}

}  // namespace lamina
