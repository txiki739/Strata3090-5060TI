// src/server/structured.hpp - JSON response formats at the HTTP boundary: prompt once, validate before delivery
// (serve/structured.py's port).
//
// The engine has no grammar decoder: a failed generation is an error, never silently retried or returned as
// successful structured output. json_object answers must be one JSON object; json_schema answers are checked against
// their schema as Python's jsonschema checks them (the dialect its $schema names, Draft 2020-12 by default), with its
// error messages.
#pragma once

#include "json.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lamina {

/// a structured answer that is incomplete, not JSON or not valid under its schema: a 502 to the client
struct StructuredOutputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// how answers are checked: one JSON object (json_object), or the request's JSON Schema (json_schema)
class Validator;

/// response_format (null: none) -> (the messages with the format's directive, its validator; nullptr for none and
/// text). RequestError (a 400) for a malformed format, an invalid schema or one nested too deep; std::runtime_error
/// where Python raises something other than a ValueError (a 500). `with_tools`: /v1/responses with tools (#782), the
/// schema is for the final answer.
/// `request`, `schema_at`: the request's text and the keys that lead to its schema there, which is then read with
/// json.loads' numbers (ints past 64 bits whole); without them, from `response_format`
std::pair<json, std::shared_ptr<const Validator>> prepare_format(const json& response_format, const json& messages,
                                                                 bool with_tools = false, std::string_view request = {},
                                                                 const std::vector<std::string>& schema_at = {});

/// the answer's JSON (compact, non-ASCII as is) when the turn ended with "stop" and the answer is valid;
/// StructuredOutputError otherwise
std::string validated_json(const std::string& text, const Validator* validator, const std::string& finish);

}  // namespace lamina
