// src/server/json.hpp - JSON as the Python server handled it: objects keep their keys' order, json.dumps' formatting,
// Python's truthiness and dict.get, and str methods on UTF-8 text (Unicode whitespace as str.isspace has it).
#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lamina {

using json = nlohmann::ordered_json;

/// json.dumps: separators ", " and ": " (or `compact`: "," and ":"), floats as repr, NaN / Infinity, non-ASCII as
/// \uXXXX with `ensure_ascii` (surrogate pairs above the BMP), invalid UTF-8 as U+FFFD
std::string py_dumps(const json& v, bool ensure_ascii = true, bool compact = false, int indent = -1);
/// json.loads; false (and `err`) on text that is not JSON
bool py_loads(std::string_view text, json& out, std::string* err = nullptr);
/// a float as Python's repr writes it
std::string py_repr(double v);
/// round(x, n): correctly rounded, as Python rounds (0.125 -> 0.12)
double py_round(double x, int n);
/// format(v, "g"): six significant digits ("%g")
std::string py_g(double v);
/// repr() of the decoded value: None / True / False, quoted strings ('x', "it's"), dict and list reprs
std::string py_repr_value(const json& v);
/// repr() of a str: quoted as Python quotes it, \x / \u / \U escapes for what str.isprintable() refuses;
/// an invalid UTF-8 byte reads as U+FFFD (what a str decoded with errors="replace" holds)
std::string py_repr_str(std::string_view s);
void py_repr_str(std::string& out, std::string_view s);   ///< the same, appended to `out`
/// str(): a str as it is, anything else as its repr()
std::string py_str(const json& v);
/// type(v).__name__
std::string py_type_name(const json& v);

/// dict.get: the member, or a null value (also when `o` is not an object)
const json& jget(const json& o, std::string_view key);
/// Python's bool(): None, False, 0, "", [] and {} are false
bool truthy(const json& v);
/// a string member, or `dflt` when it is missing or not a string
std::string jstr(const json& o, std::string_view key, std::string_view dflt = "");
bool is_int(const json& v);       ///< an int, not a bool
bool is_num(const json& v);       ///< an int or a float, not a bool
bool is_str(const json& v, std::string_view s);   ///< v == s: a str, equal to `s`

// ---- str methods on UTF-8 text
bool py_isspace(uint32_t cp);
/// str.isprintable() of one character: not Cc, Cf, Cs, Co, Cn, Zl, Zp or Zs (but the space)
bool py_isprintable(uint32_t cp);
/// assigned in llama.cpp's Unicode 15.1 tables, unassigned in Python 3.12's Unicode 15.0
bool after_unicode_15(uint32_t cp);
std::string_view lstrip(std::string_view s);
std::string_view rstrip(std::string_view s);
std::string_view strip(std::string_view s);
std::string_view lstrip_chars(std::string_view s, std::string_view chars);
std::string_view rstrip_chars(std::string_view s, std::string_view chars);
std::string_view strip_chars(std::string_view s, std::string_view chars);
bool starts_with(std::string_view s, std::string_view p);
bool ends_with(std::string_view s, std::string_view p);
/// s.startswith(p) for any of `ps`
bool starts_with_any(std::string_view s, std::initializer_list<std::string_view> ps);
bool contains_space(std::string_view s);       ///< any(c.isspace() for c in s)
std::string replace_all(std::string s, std::string_view from, std::string_view to);
std::vector<std::string> split(std::string_view s, std::string_view sep);
std::vector<std::string> split_ws(std::string_view s);   ///< str.split(): the fields between ASCII whitespace
/// str.splitlines(): \n, \r, \r\n, \v, \f, \x1c-\x1e, \x85, U+2028 and U+2029 end a line
std::vector<std::string_view> splitlines(std::string_view s);
std::string lower(std::string_view s);         ///< ASCII lower case (str.lower on the inputs that reach it)
/// urllib.parse.unquote_to_bytes: each %XX as its byte, the rest (a % without two hex digits too) as it is
std::string unquote_to_bytes(std::string_view s);
size_t utf8_len(std::string_view s);           ///< len() of the decoded text
/// the code point at s[i] and its length (an invalid byte reads as itself, length 1)
uint32_t utf8_at(std::string_view s, size_t i, size_t* n);
void utf8_append(std::string& out, uint32_t cp);
/// bytes.decode("utf-8", "replace"): each ill-formed part becomes one U+FFFD, cut where CPython's decoder
/// cuts (a truncated sequence is one part; a byte that cannot continue it starts the next)
std::string decode_replace(std::string_view b);
#ifdef _WIN32
std::wstring widen(std::string_view s);    ///< UTF-8 -> the Windows API's UTF-16
std::string narrow(std::wstring_view w);   ///< the Windows API's UTF-16 -> UTF-8
#endif
/// a random hex string of n digits (uuid.uuid4().hex[:n])
std::string random_hex(size_t n);

}  // namespace lamina
