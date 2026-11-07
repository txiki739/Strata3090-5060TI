// The parameter values of a Qwen tool call (<parameter=P>VALUE</parameter>) against the tool's schema, whole and
// streamed. A value declared boolean or null also takes the Python spellings the model sometimes writes (True, False,
// None, any case), as vLLM's and llama.cpp's Qwen3-Coder parsers read them; a string stays the text; anything else
// is JSON or, failing that, the text. (CPU only.)
#include "frontend.hpp"

#include <cstdio>
#include <string>

using lamina::json;

static int fails = 0;
#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);   \
            ++fails;                                                              \
        }                                                                         \
    } while (0)

static const json kTool = json::parse(R"({"name": "Agent", "parameters": {"type": "object", "properties": {
    "prompt": {"type": "string"}, "run_in_background": {"type": "boolean"}, "count": {"type": "integer"},
    "either": {"type": ["boolean", "null"]}, "nothing": {"type": "null"}, "odd": {"type": "boolean"}}}})");

static const std::string kBody =
    "<function=Agent>\n<parameter=prompt>\nFalse\n</parameter>\n<parameter=run_in_background>\nFalse\n</parameter>\n"
    "<parameter=count>\n3\n</parameter>\n<parameter=either>\nTRUE\n</parameter>\n<parameter=nothing>\nNone\n"
    "</parameter>\n<parameter=odd>\nmaybe\n</parameter>\n</function>";

static void check_args(const json& a) {
    CHECK(a["prompt"] == "False");                 // a string is the text
    CHECK(a["run_in_background"] == false);        // Python's False for a boolean
    CHECK(a["count"] == 3);                        // JSON as before
    CHECK(a["either"] == true);                    // a type list that has boolean, any case
    CHECK(a["nothing"].is_null());                 // Python's None for null
    CHECK(a["odd"] == "maybe");                    // not a boolean spelling: the text, as before
}

int main() {
    check_args(lamina::parse_tool_call(kBody, kTool).arguments);

    // streamed: the tool_args deltas join into the same object, and the final call agrees
    lamina::OutputParser p(false, json::array({kTool}), true);
    const std::string text = "<tool_call>\n" + kBody + "\n</tool_call>";
    std::string args;
    json final_args;
    auto take = [&](const std::vector<lamina::Event>& evs) {
        for (const auto& e : evs) {
            if (e.kind == "tool_args") args += e.text;
            if (e.kind == "tool_call" && e.call) final_args = e.call->arguments;
        }
    };
    for (size_t i = 0; i < text.size(); i += 7) take(p.feed(std::string_view(text).substr(i, 7)));
    take(p.finish("stop"));
    CHECK(!final_args.is_null());
    if (!final_args.is_null()) check_args(final_args);
    json streamed;
    try {
        streamed = json::parse(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL streamed arguments are not JSON: %s\n%s\n", e.what(), args.c_str());
        ++fails;
    }
    if (!streamed.is_null()) check_args(streamed);

    if (fails == 0) std::puts("tool_args_test: OK");
    return fails == 0 ? 0 : 1;
}
