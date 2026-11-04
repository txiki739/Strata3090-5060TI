// src/server/structured.cpp - JSON response formats at the HTTP boundary: prompt once, validate before delivery
// (serve/structured.py's port).
//
// The engine has no grammar decoder: failed generations are errors, never silently retried or returned as successful
// structured output. json_object answers must be one JSON object. json_schema answers are checked as the Python server
// checks them with the jsonschema package (4.23+) and its referencing library: the validator class the schema's
// $schema names (Draft 2020-12 without one), the schema first checked against that draft's metaschema (its `regex`
// formats asserted), local references only, `format` not asserted in answers, and the first error iter_errors yields
// reported with jsonschema's message and path. What that takes from Python is ported here too: json.loads (its errors,
// ints of any size, lone surrogates), json.dumps, repr() of the decoded values, the `re` module (pattern,
// patternProperties), list.sort's comparisons (uniqueItems), the recursion limit, and the parts of urllib.parse that
// references use.
//
// The stack: request threads have 1 MB. What Python's recursion limit stops stops here too, counted the same way
// (Python's frames), and that stays below 400 KB; passes over JSON nest without recursion; a pattern may nest 100
// groups (Python about 490), and patterns are compiled before validation starts. Too deep a schema or pattern is a
// 400, too deep an answer a 502, with Python's "maximum recursion depth exceeded".
//
// Generated tables: kMetaDocs (jsonschema-specifications 2025.9.1's files, compact), kDecimal and sre_upper's kExtended
// (Python 3.12's unicodedata); sre_extra_cases is CPython 3.12's re/_casefix.py.
#include "structured.hpp"

#include "frontend.hpp"         // RequestError
#include "src/unicode-data.h"   // llama.cpp's code point tables (Unicode 15.1)
#include "src/unicode.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lamina {
namespace {

constexpr size_t npos = std::string::npos;

// ============================================================================================ Python's exceptions
/// a Python exception: its class name and str()
struct PyError : std::exception {
    std::string type, msg;
    PyError(std::string t, std::string m) : type(std::move(t)), msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};
[[noreturn]] void raise(const char* type, std::string msg) { throw PyError(type, std::move(msg)); }

/// referencing's Unresolvable and its subclasses
bool unresolvable(const std::string& type) {
    return type == "Unresolvable" || type == "PointerToNowhere" || type == "NoSuchAnchor" || type == "InvalidAnchor";
}

// ============================================================================================ Python values
struct Member;

/// a JSON value as Python's json module holds it: ints of any size, str with lone surrogates, dicts in key order
struct Val {
    enum class K : uint8_t { None, Bool, Int, Big, Float, Str, List, Dict };
    K k = K::None;
    bool b = false;
    int64_t i = 0;
    double f = 0;
    std::string s;                 ///< Str: UTF-8 (a lone surrogate as its 3 bytes); Big: the digits, '-' first if negative
    std::vector<Val> list;
    std::vector<Member> dict;
    std::vector<uint32_t> order;   ///< a big Dict's member positions sorted by key, for get()

    Val() = default;
    Val(const Val&) = default;
    Val(Val&&) noexcept = default;
    Val& operator=(const Val&) = default;
    Val& operator=(Val&&) noexcept = default;
    ~Val();   ///< without recursion: answers nest as deep as json.loads takes them

    static Val boolean(bool v);
    static Val integer(int64_t v);
    static Val real(double v);
    static Val str(std::string v);
    bool number() const { return k == K::Int || k == K::Big || k == K::Float; }   ///< is_type "number": not bool
    const Val* get(std::string_view key) const;
    bool has(std::string_view key) const { return get(key) != nullptr; }
    void index();   ///< after the members are added
};
struct Member {
    std::string key;
    Val val;
};
using K = Val::K;

Val::~Val() {
    if (list.empty() && dict.empty()) return;
    std::vector<Val> pending;   // the descendants, taken out before each is destroyed
    auto take = [&pending](Val& v) {
        for (Val& e : v.list) pending.push_back(std::move(e));
        for (Member& m : v.dict) pending.push_back(std::move(m.val));
        v.list.clear();
        v.dict.clear();
    };
    take(*this);
    while (!pending.empty()) {
        Val v = std::move(pending.back());
        pending.pop_back();
        take(v);
    }
}

Val Val::boolean(bool v) {
    Val x;
    x.k = K::Bool;
    x.b = v;
    return x;
}
Val Val::integer(int64_t v) {
    Val x;
    x.k = K::Int;
    x.i = v;
    return x;
}
Val Val::real(double v) {
    Val x;
    x.k = K::Float;
    x.f = v;
    return x;
}
Val Val::str(std::string v) {
    Val x;
    x.k = K::Str;
    x.s = std::move(v);
    return x;
}

const Val* Val::get(std::string_view key) const {
    if (k != K::Dict) return nullptr;
    if (order.empty()) {
        for (const Member& m : dict)
            if (m.key == key) return &m.val;
        return nullptr;
    }
    const auto it = std::lower_bound(order.begin(), order.end(), key,
                                     [this](uint32_t p, std::string_view x) { return dict[p].key < x; });
    return it != order.end() && dict[*it].key == key ? &dict[*it].val : nullptr;
}

void Val::index() {
    order.clear();
    if (dict.size() <= 8) return;
    order.resize(dict.size());
    for (uint32_t j = 0; j < (uint32_t) order.size(); ++j) order[j] = j;
    std::sort(order.begin(), order.end(), [this](uint32_t a, uint32_t c) { return dict[a].key < dict[c].key; });
}

/// a JSON scalar as Python decoded it (nlohmann's ints are at most 64-bit); an empty list or dict for a container
Val from_scalar(const json& j) {
    switch (j.type()) {
        case json::value_t::boolean: return Val::boolean(j.get<bool>());
        case json::value_t::number_integer: return Val::integer(j.get<int64_t>());
        case json::value_t::number_unsigned: {
            const uint64_t u = j.get<uint64_t>();
            if (u <= (uint64_t) INT64_MAX) return Val::integer((int64_t) u);
            Val x;
            x.k = K::Big;
            x.s = std::to_string(u);
            return x;
        }
        case json::value_t::number_float: return Val::real(j.get<double>());
        case json::value_t::string: return Val::str(j.get<std::string>());
        case json::value_t::array: {
            Val x;
            x.k = K::List;
            return x;
        }
        case json::value_t::object: {
            Val x;
            x.k = K::Dict;
            return x;
        }
        default: return {};
    }
}

/// the request's JSON as Python decoded it (without recursion)
Val from_json(const json& root) {
    if (!root.is_structured()) return from_scalar(root);
    struct Open {
        const json* j;
        json::const_iterator it;
        Val v;
    };
    std::vector<Open> st;
    st.push_back({&root, root.begin(), from_scalar(root)});
    for (;;) {
        Open& o = st.back();
        if (o.it == o.j->end()) {   // complete: into its parent
            if (o.v.k == K::Dict) o.v.index();
            Val done = std::move(o.v);
            st.pop_back();
            if (st.empty()) return done;
            Open& p = st.back();
            if (p.v.k == K::Dict) p.v.dict.push_back({p.it.key(), std::move(done)});
            else p.v.list.push_back(std::move(done));
            ++p.it;
            continue;
        }
        const json& child = o.it.value();
        if (child.is_structured()) {
            st.push_back({&child, child.begin(), from_scalar(child)});
            continue;
        }
        if (o.v.k == K::Dict) o.v.dict.push_back({o.it.key(), from_scalar(child)});
        else o.v.list.push_back(from_scalar(child));
        ++o.it;
    }
}

/// type(x).__name__
const char* type_name(const Val& v) {
    switch (v.k) {
        case K::None: return "NoneType";
        case K::Bool: return "bool";
        case K::Int:
        case K::Big: return "int";
        case K::Float: return "float";
        case K::Str: return "str";
        case K::List: return "list";
        case K::Dict: return "dict";
    }
    return "object";
}

bool truthy(const Val& v) {
    switch (v.k) {
        case K::None: return false;
        case K::Bool: return v.b;
        case K::Int: return v.i != 0;
        case K::Big: return true;
        case K::Float: return v.f != 0;
        case K::Str: return !v.s.empty();
        case K::List: return !v.list.empty();
        case K::Dict: return !v.dict.empty();
    }
    return false;
}

[[noreturn]] void no_attribute(const Val& v, const char* attr) {
    raise("AttributeError", std::string("'") + type_name(v) + "' object has no attribute '" + attr + "'");
}

// ============================================================================================ code points
constexpr uint32_t kReplacement = 0xFFFD;

/// the code point at s[i] and its length (an invalid byte reads as U+FFFD)
uint32_t cp_at(std::string_view s, size_t i, size_t* n) {
    const uint32_t cp = utf8_at(s, i, n);
    return *n == 1 && (unsigned char) s[i] >= 0x80 ? kReplacement : cp;
}

std::u32string code_points(std::string_view s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        out += (char32_t) cp_at(s, i, &n);
        i += n;
    }
    return out;
}

std::string utf8_of(std::u32string_view cps) {
    std::string out;
    for (const char32_t c : cps) utf8_append(out, (uint32_t) c);
    return out;
}

/// text with each invalid UTF-8 byte as U+FFFD (what a str decoded with errors="replace" holds, per byte)
std::string sanitized(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const uint32_t cp = cp_at(s, i, &n);
        if (cp == kReplacement && n == 1) utf8_append(out, kReplacement);
        else out.append(s.data() + i, n);
        i += n;
    }
    return out;
}

bool is_digit(uint32_t c) { return c >= '0' && c <= '9'; }
bool is_ascii_letter(uint32_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/// str.isalpha() of one character
bool py_alpha(uint32_t cp) {
    if (cp < 0x80) return is_ascii_letter(cp);
    return !after_unicode_15(cp) && unicode_cpt_flags_from_cpt(cp).is_letter;
}

/// str.isalnum() or "_": re's \w
bool py_word(uint32_t cp) {
    if (cp < 0x80) return is_digit(cp) || is_ascii_letter(cp) || cp == '_';
    if (after_unicode_15(cp)) return false;
    const unicode_cpt_flags f = unicode_cpt_flags_from_cpt(cp);
    return f.is_letter || f.is_number;
}

/// the Nd category (str.isdecimal(), re's \d), Unicode 15.0
constexpr std::pair<uint32_t, uint32_t> kDecimal[] = {
    {0x30, 0x39}, {0x660, 0x669}, {0x6F0, 0x6F9}, {0x7C0, 0x7C9}, {0x966, 0x96F}, {0x9E6, 0x9EF}, {0xA66, 0xA6F},
    {0xAE6, 0xAEF}, {0xB66, 0xB6F}, {0xBE6, 0xBEF}, {0xC66, 0xC6F}, {0xCE6, 0xCEF}, {0xD66, 0xD6F}, {0xDE6, 0xDEF},
    {0xE50, 0xE59}, {0xED0, 0xED9}, {0xF20, 0xF29}, {0x1040, 0x1049}, {0x1090, 0x1099}, {0x17E0, 0x17E9},
    {0x1810, 0x1819}, {0x1946, 0x194F}, {0x19D0, 0x19D9}, {0x1A80, 0x1A89}, {0x1A90, 0x1A99}, {0x1B50, 0x1B59},
    {0x1BB0, 0x1BB9}, {0x1C40, 0x1C49}, {0x1C50, 0x1C59}, {0xA620, 0xA629}, {0xA8D0, 0xA8D9}, {0xA900, 0xA909},
    {0xA9D0, 0xA9D9}, {0xA9F0, 0xA9F9}, {0xAA50, 0xAA59}, {0xABF0, 0xABF9}, {0xFF10, 0xFF19}, {0x104A0, 0x104A9},
    {0x10D30, 0x10D39}, {0x11066, 0x1106F}, {0x110F0, 0x110F9}, {0x11136, 0x1113F}, {0x111D0, 0x111D9},
    {0x112F0, 0x112F9}, {0x11450, 0x11459}, {0x114D0, 0x114D9}, {0x11650, 0x11659}, {0x116C0, 0x116C9},
    {0x11730, 0x11739}, {0x118E0, 0x118E9}, {0x11950, 0x11959}, {0x11C50, 0x11C59}, {0x11D50, 0x11D59},
    {0x11DA0, 0x11DA9}, {0x11F50, 0x11F59}, {0x16A60, 0x16A69}, {0x16AC0, 0x16AC9}, {0x16B50, 0x16B59},
    {0x1D7CE, 0x1D7FF}, {0x1E140, 0x1E149}, {0x1E2F0, 0x1E2F9}, {0x1E4F0, 0x1E4F9}, {0x1E950, 0x1E959},
    {0x1FBF0, 0x1FBF9}};

/// the digit's value (0-9), or -1 when not a decimal
int py_decimal(uint32_t cp) {
    if (is_digit(cp)) return (int) (cp - '0');
    if (cp < 0x660) return -1;
    const auto it = std::upper_bound(std::begin(kDecimal), std::end(kDecimal), cp,
                                     [](uint32_t c, const std::pair<uint32_t, uint32_t>& r) { return c < r.first; });
    if (it == std::begin(kDecimal)) return -1;
    const auto& r = *(it - 1);
    return cp <= r.second ? (int) ((cp - r.first) % 10) : -1;
}

/// sre's lower case (the simple mapping)
uint32_t sre_lower(uint32_t cp) { return unicode_tolower(cp); }

/// sre's upper case: the simple mapping, or the first character of a multi-character one
uint32_t sre_upper(uint32_t cp) {
    static constexpr std::pair<uint32_t, uint32_t> kExtended[] = {
        {0xDF, 0x53}, {0x149, 0x2BC}, {0x1F0, 0x4A}, {0x390, 0x399}, {0x3B0, 0x3A5}, {0x587, 0x535},
        {0x1E96, 0x48}, {0x1E97, 0x54}, {0x1E98, 0x57}, {0x1E99, 0x59}, {0x1E9A, 0x41}, {0x1F50, 0x3A5},
        {0x1F52, 0x3A5}, {0x1F54, 0x3A5}, {0x1F56, 0x3A5}, {0x1F88, 0x1F08}, {0x1F89, 0x1F09}, {0x1F8A, 0x1F0A},
        {0x1F8B, 0x1F0B}, {0x1F8C, 0x1F0C}, {0x1F8D, 0x1F0D}, {0x1F8E, 0x1F0E}, {0x1F8F, 0x1F0F}, {0x1F98, 0x1F28},
        {0x1F99, 0x1F29}, {0x1F9A, 0x1F2A}, {0x1F9B, 0x1F2B}, {0x1F9C, 0x1F2C}, {0x1F9D, 0x1F2D}, {0x1F9E, 0x1F2E},
        {0x1F9F, 0x1F2F}, {0x1FA8, 0x1F68}, {0x1FA9, 0x1F69}, {0x1FAA, 0x1F6A}, {0x1FAB, 0x1F6B}, {0x1FAC, 0x1F6C},
        {0x1FAD, 0x1F6D}, {0x1FAE, 0x1F6E}, {0x1FAF, 0x1F6F}, {0x1FB2, 0x1FBA}, {0x1FB4, 0x386}, {0x1FB6, 0x391},
        {0x1FB7, 0x391}, {0x1FBC, 0x391}, {0x1FC2, 0x1FCA}, {0x1FC4, 0x389}, {0x1FC6, 0x397}, {0x1FC7, 0x397},
        {0x1FCC, 0x397}, {0x1FD2, 0x399}, {0x1FD3, 0x399}, {0x1FD6, 0x399}, {0x1FD7, 0x399}, {0x1FE2, 0x3A5},
        {0x1FE3, 0x3A5}, {0x1FE4, 0x3A1}, {0x1FE6, 0x3A5}, {0x1FE7, 0x3A5}, {0x1FF2, 0x1FFA}, {0x1FF4, 0x38F},
        {0x1FF6, 0x3A9}, {0x1FF7, 0x3A9}, {0x1FFC, 0x3A9}, {0xFB00, 0x46}, {0xFB01, 0x46}, {0xFB02, 0x46},
        {0xFB03, 0x46}, {0xFB04, 0x46}, {0xFB05, 0x53}, {0xFB06, 0x53}, {0xFB13, 0x544}, {0xFB14, 0x544},
        {0xFB15, 0x544}, {0xFB16, 0x54E}, {0xFB17, 0x544}};
    auto by_first = [](const std::pair<uint32_t, uint32_t>& p, uint32_t c) { return p.first < c; };
    const auto e = std::lower_bound(std::begin(kExtended), std::end(kExtended), cp, by_first);
    if (e != std::end(kExtended) && e->first == cp) return e->second;
    const auto u = std::lower_bound(unicode_map_uppercase.begin(), unicode_map_uppercase.end(), cp, by_first);
    return u != unicode_map_uppercase.end() && u->first == cp ? u->second : cp;
}

bool sre_iscased(uint32_t cp) { return sre_lower(cp) != cp || sre_upper(cp) != cp; }

/// re._casefix._EXTRA_CASES: a lower cased character -> the other lower cased characters with its upper case
const std::vector<uint32_t>* sre_extra_cases(uint32_t lo) {
    static const std::map<uint32_t, std::vector<uint32_t>> kCases = {
        {0x69, {0x131}}, {0x73, {0x17F}}, {0xB5, {0x3BC}}, {0x131, {0x69}}, {0x17F, {0x73}},
        {0x345, {0x3B9, 0x1FBE}}, {0x390, {0x1FD3}}, {0x3B0, {0x1FE3}}, {0x3B2, {0x3D0}}, {0x3B5, {0x3F5}},
        {0x3B8, {0x3D1}}, {0x3B9, {0x345, 0x1FBE}}, {0x3BA, {0x3F0}}, {0x3BC, {0xB5}}, {0x3C0, {0x3D6}},
        {0x3C1, {0x3F1}}, {0x3C2, {0x3C3}}, {0x3C3, {0x3C2}}, {0x3C6, {0x3D5}}, {0x3D0, {0x3B2}}, {0x3D1, {0x3B8}},
        {0x3D5, {0x3C6}}, {0x3D6, {0x3C0}}, {0x3F0, {0x3BA}}, {0x3F1, {0x3C1}}, {0x3F5, {0x3B5}}, {0x432, {0x1C80}},
        {0x434, {0x1C81}}, {0x43E, {0x1C82}}, {0x441, {0x1C83}}, {0x442, {0x1C84, 0x1C85}}, {0x44A, {0x1C86}},
        {0x463, {0x1C87}}, {0x1C80, {0x432}}, {0x1C81, {0x434}}, {0x1C82, {0x43E}}, {0x1C83, {0x441}},
        {0x1C84, {0x442, 0x1C85}}, {0x1C85, {0x442, 0x1C84}}, {0x1C86, {0x44A}}, {0x1C87, {0x463}},
        {0x1C88, {0xA64B}}, {0x1E61, {0x1E9B}}, {0x1E9B, {0x1E61}}, {0x1FBE, {0x345, 0x3B9}}, {0x1FD3, {0x390}},
        {0x1FE3, {0x3B0}}, {0xA64B, {0x1C88}}, {0xFB05, {0xFB06}}, {0xFB06, {0xFB05}}};
    const auto it = kCases.find(lo);
    return it == kCases.end() ? nullptr : &it->second;
}

// ============================================================================================ repr() and str()

void repr_scalar(std::string& out, const Val& v) {
    switch (v.k) {
        case K::None: out += "None"; break;
        case K::Bool: out += v.b ? "True" : "False"; break;
        case K::Int: out += std::to_string(v.i); break;
        case K::Big: out += v.s; break;
        case K::Float: out += py_repr(v.f); break;
        case K::Str: py_repr_str(out, v.s); break;
        default: break;
    }
}

/// repr(), without recursion (answers nest as deep as json.loads takes them)
void repr_into(std::string& out, const Val& root) {
    struct Open {
        const Val* v;
        size_t next;
    };
    std::vector<Open> st;
    auto visit = [&](const Val& v) {
        if (v.k == K::List) {
            out += '[';
            st.push_back({&v, 0});
        } else if (v.k == K::Dict) {
            out += '{';
            st.push_back({&v, 0});
        } else {
            repr_scalar(out, v);
        }
    };
    visit(root);
    while (!st.empty()) {
        const Val& v = *st.back().v;
        const bool list = v.k == K::List;
        const size_t j = st.back().next++;
        if (j == (list ? v.list.size() : v.dict.size())) {
            out += list ? ']' : '}';
            st.pop_back();
            continue;
        }
        if (j > 0) out += ", ";
        if (list) {
            visit(v.list[j]);
        } else {
            py_repr_str(out, v.dict[j].key);
            out += ": ";
            visit(v.dict[j].val);
        }
    }
}

std::string repr(const Val& v) {
    std::string out;
    repr_into(out, v);
    return out;
}

/// str(): a str as is, anything else as its repr()
std::string py_str(const Val& v) { return v.k == K::Str ? v.s : repr(v); }

std::string joined(const std::vector<std::string>& parts, std::string_view sep = ", ") {
    std::string out;
    for (size_t j = 0; j < parts.size(); ++j) {
        if (j) out += sep;
        out += parts[j];
    }
    return out;
}

/// jsonschema._utils.extras_msg: "'a', 'b' were" (as "%s %s")
std::string extras_msg(const std::vector<std::string>& reprs) {
    return joined(reprs) + (reprs.size() == 1 ? " was" : " were");
}

// ============================================================================================ numbers
/// the decimal digits of an integral double's magnitude, exactly
std::string exact_digits(double d) {
    int e = 0;
    const double m = std::frexp(std::fabs(d), &e);
    uint64_t mant = (uint64_t) std::ldexp(m, 53);
    e -= 53;
    while (e < 0 && mant != 0 && mant % 2 == 0) {
        mant /= 2;
        ++e;
    }
    if (e < 0) mant >>= -e;
    std::vector<uint32_t> limbs;   // base 1e9, least significant first
    for (; mant; mant /= 1000000000u) limbs.push_back((uint32_t) (mant % 1000000000u));
    for (int k = 0; k < e; ++k) {
        uint32_t carry = 0;
        for (uint32_t& l : limbs) {
            const uint64_t x = (uint64_t) l * 2 + carry;
            l = (uint32_t) (x % 1000000000u);
            carry = (uint32_t) (x / 1000000000u);
        }
        if (carry) limbs.push_back(carry);
    }
    if (limbs.empty()) return "0";
    std::string out = std::to_string(limbs.back());
    char buf[16];
    for (size_t k = limbs.size() - 1; k-- > 0;) {
        std::snprintf(buf, sizeof buf, "%09u", limbs[k]);
        out += buf;
    }
    return out;
}

int sign_of(int c) { return c < 0 ? -1 : c > 0 ? 1 : 0; }

/// two signed decimal integers (no leading zeros)
int cmp_signed(std::string_view a, std::string_view b) {
    const bool na = !a.empty() && a[0] == '-', nb = !b.empty() && b[0] == '-';
    if (na != nb) return na ? -1 : 1;
    const std::string_view x = a.substr(na ? 1 : 0), y = b.substr(nb ? 1 : 0);
    const int m = x.size() != y.size() ? (x.size() < y.size() ? -1 : 1) : sign_of(x.compare(y));
    return na ? -m : m;
}

int cmp_int_float(int64_t a, double b) {
    if (b >= 9223372036854775808.0) return -1;
    if (b < -9223372036854775808.0) return 1;
    const double t = std::trunc(b);
    const int64_t ti = (int64_t) t;
    if (a != ti) return a < ti ? -1 : 1;
    const double frac = b - t;
    return frac > 0 ? -1 : frac < 0 ? 1 : 0;
}

int cmp_big_float(const std::string& big, double b) {
    if (std::isinf(b)) return b > 0 ? -1 : 1;
    if (std::fabs(b) < 9.2e18) return big[0] == '-' ? -1 : 1;   // a Big is beyond int64
    return cmp_signed(big, (b < 0 ? "-" : "") + exact_digits(b));
}

bool numeric(const Val& v) { return v.number() || v.k == K::Bool; }

/// Python's comparison of two numbers (bools as 0 and 1): exact across int and float
int num_cmp(const Val& a, const Val& b) {
    const bool ai = a.k == K::Int || a.k == K::Bool, bi = b.k == K::Int || b.k == K::Bool;
    const int64_t x = a.k == K::Bool ? (int64_t) a.b : a.i, y = b.k == K::Bool ? (int64_t) b.b : b.i;
    if (ai && bi) return x < y ? -1 : x > y ? 1 : 0;
    if (ai && b.k == K::Float) return cmp_int_float(x, b.f);
    if (a.k == K::Float && bi) return -cmp_int_float(y, a.f);
    if (a.k == K::Float && b.k == K::Float) return a.f < b.f ? -1 : a.f > b.f ? 1 : 0;
    if (a.k == K::Big && b.k == K::Big) return cmp_signed(a.s, b.s);
    if (a.k == K::Big && bi) return a.s[0] == '-' ? -1 : 1;
    if (ai && b.k == K::Big) return b.s[0] == '-' ? 1 : -1;
    if (a.k == K::Big) return cmp_big_float(a.s, b.f);
    return -cmp_big_float(b.s, a.f);
}

/// `a <op> b` as Python orders numbers: -1, 0 or 1; TypeError for what it cannot order
int order(const Val& a, const Val& b, const char* op) {
    if (numeric(a) && numeric(b)) return num_cmp(a, b);
    raise("TypeError", std::string("'") + op + "' not supported between instances of '" + type_name(a) + "' and '" +
                           type_name(b) + "'");
}

/// float(x) of a number
double to_float(const Val& v) {
    if (v.k == K::Float) return v.f;
    if (v.k == K::Bool) return v.b ? 1.0 : 0.0;
    if (v.k == K::Int) return (double) v.i;
    double d = 0;
    const auto r = std::from_chars(v.s.data(), v.s.data() + v.s.size(), d);
    if (r.ec == std::errc::result_out_of_range || std::isinf(d)) raise("OverflowError", "int too large to convert to float");
    return d;
}

/// the magnitude of an int (or bool) in decimal digits
std::string int_digits(const Val& v) {
    if (v.k == K::Big) return v.s[0] == '-' ? v.s.substr(1) : v.s;
    const int64_t x = v.k == K::Bool ? (int64_t) v.b : v.i;
    return x < 0 ? std::to_string((uint64_t) 0 - (uint64_t) x) : std::to_string(x);
}

uint64_t add_mod(uint64_t a, uint64_t b, uint64_t m) { return a >= m - b ? a - (m - b) : a + b; }   // a, b < m

/// a decimal magnitude modulo m (m > 0)
uint64_t mod_digits(std::string_view digits, uint64_t m) {
    uint64_t r = 0;
    for (const char c : digits) {
        uint64_t x = 0;
        for (int k = 0; k < 10; ++k) x = add_mod(x, r, m);   // r * 10
        r = add_mod(x, (uint64_t) (c - '0') % m, m);
    }
    return r;
}

/// a decimal magnitude halved (it is even)
std::string halved(const std::string& digits) {
    std::string out;
    int carry = 0;
    for (const char c : digits) {
        const int x = carry * 10 + (c - '0');
        if (!out.empty() || x / 2) out += (char) ('0' + x / 2);
        carry = x % 2;
    }
    return out.empty() ? "0" : out;
}

/// not bool(instance % dB) for an int (or bool) dB: Python's remainder is zero
bool remainder_zero(const Val& inst, const Val& dB) {
    const std::string d = int_digits(dB);
    if (d == "0") raise("ZeroDivisionError", inst.k == K::Float ? "float modulo" : "integer modulo by zero");
    if (inst.k == K::Float) return std::fmod(inst.f, to_float(dB)) == 0;
    uint64_t m = 0;
    if (std::from_chars(d.data(), d.data() + d.size(), m).ec != std::errc()) return false;   // beyond the request's ints
    return mod_digits(int_digits(inst), m) == 0;
}

/// (Fraction(instance) / Fraction(dB)).denominator == 1, for a float dB
bool fraction_multiple(const Val& inst, double dB) {
    auto split = [](double d, uint64_t& m, int& e) {   // |d| = m * 2^e, m odd
        int x = 0;
        const double fr = std::frexp(std::fabs(d), &x);
        m = (uint64_t) std::ldexp(fr, 53);
        e = x - 53;
        while (m != 0 && m % 2 == 0) {
            m /= 2;
            ++e;
        }
    };
    uint64_t mb = 0;
    int eb = 0;
    split(dB, mb, eb);
    if (inst.k == K::Float) {
        if (inst.f == 0) return true;
        uint64_t mi = 0;
        int ei = 0;
        split(inst.f, mi, ei);
        return mi % mb == 0 && ei >= eb;
    }
    std::string digits = int_digits(inst);
    if (digits == "0") return true;
    if (mod_digits(digits, mb) != 0) return false;   // the odd part of dB divides it
    for (int k = 0; k < eb; ++k) {                   // and 2^eb
        if ((digits.back() - '0') % 2) return false;
        digits = halved(digits);
    }
    return true;
}

// ============================================================================================ equality and order
/// Python's == (without recursion)
bool py_eq(const Val& a0, const Val& b0) {
    std::vector<std::pair<const Val*, const Val*>> todo{{&a0, &b0}};
    while (!todo.empty()) {
        const auto [a, b] = todo.back();
        todo.pop_back();
        if (numeric(*a) && numeric(*b)) {
            if (num_cmp(*a, *b) != 0) return false;
            continue;
        }
        if (a->k != b->k) return false;
        switch (a->k) {
            case K::Str:
                if (a->s != b->s) return false;
                break;
            case K::List:
                if (a->list.size() != b->list.size()) return false;
                for (size_t j = 0; j < a->list.size(); ++j) todo.push_back({&a->list[j], &b->list[j]});
                break;
            case K::Dict:
                if (a->dict.size() != b->dict.size()) return false;
                for (const Member& m : a->dict) {
                    const Val* o = b->get(m.key);
                    if (o == nullptr) return false;
                    todo.push_back({&m.val, o});
                }
                break;
            default: break;   // None
        }
    }
    return true;
}

struct Unordered {};   ///< list.sort's TypeError

/// Python's <, Unordered for the values it cannot order
bool py_lt(const Val* a, const Val* b) {
    for (;;) {
        if (numeric(*a) && numeric(*b)) return num_cmp(*a, *b) < 0;
        if (a->k == K::Str && b->k == K::Str) return a->s < b->s;   // UTF-8 byte order is code point order
        if (a->k != K::List || b->k != K::List) throw Unordered{};
        size_t j = 0;   // the first items that differ decide
        while (j < a->list.size() && j < b->list.size() && py_eq(a->list[j], b->list[j])) ++j;
        if (j >= a->list.size() || j >= b->list.size()) return a->list.size() < b->list.size();
        a = &a->list[j];
        b = &b->list[j];
    }
}

/// jsonschema._utils.equal: bools are not numbers, sequences and mappings compare item by item. `frames`: what is
/// left of Python's recursion limit (equal, _sequence_equal and its generator per level)
bool js_equal(const Val& a, const Val& b, int frames) {
    if (frames <= 0) raise("RecursionError", "maximum recursion depth exceeded");
    if (a.k == K::Str || b.k == K::Str) return a.k == b.k && a.s == b.s;
    if (a.k == K::List && b.k == K::List) {
        if (a.list.size() != b.list.size()) return false;
        for (size_t j = 0; j < a.list.size(); ++j)
            if (!js_equal(a.list[j], b.list[j], frames - 3)) return false;
        return true;
    }
    if (a.k == K::Dict && b.k == K::Dict) {
        if (a.dict.size() != b.dict.size()) return false;
        for (const Member& m : a.dict) {
            const Val* o = b.get(m.key);
            if (o == nullptr || !js_equal(m.val, *o, frames - 3)) return false;
        }
        return true;
    }
    if (a.k == K::Bool || b.k == K::Bool) return a.k == b.k && a.b == b.b;
    if (a.number() && b.number()) return num_cmp(a, b) == 0;
    return a.k == K::None && b.k == K::None;
}

/// sorted(): CPython's list.sort for fewer than 64 items (a run, then binary insertion), else a stable sort after
/// checking that every pair orders
std::vector<const Val*> py_sorted(const std::vector<Val>& items, bool lists) {
    std::vector<const Val*> a;
    a.reserve(items.size());
    for (const Val& v : items) a.push_back(&v);
    const size_t n = a.size();
    if (n < 2) return a;
    auto lt = [](const Val* x, const Val* y) { return py_lt(x, y); };
    if (n < 64) {
        size_t run = 2;
        if (lt(a[1], a[0])) {
            while (run < n && lt(a[run], a[run - 1])) ++run;
            std::reverse(a.begin(), a.begin() + (std::ptrdiff_t) run);
        } else {
            while (run < n && !lt(a[run], a[run - 1])) ++run;
        }
        for (size_t start = run; start < n; ++start) {
            const Val* pivot = a[start];
            size_t l = 0, r = start;
            do {
                const size_t p = l + ((r - l) >> 1);
                if (lt(pivot, a[p])) r = p;
                else l = p + 1;
            } while (l < r);
            for (size_t p = start; p > l; --p) a[p] = a[p - 1];
            a[l] = pivot;
        }
        return a;
    }
    if (lists)
        for (size_t j = 0; j < n; ++j)
            for (size_t m = j + 1; m < n; ++m) (void) lt(a[j], a[m]);
    std::stable_sort(a.begin(), a.end(), lt);
    return a;
}

/// jsonschema._utils.uniq: neighbours after sorting, or every pair when the items cannot be sorted
bool uniq(const std::vector<Val>& items, int frames) {
    const size_t n = items.size();
    int kind = -1;
    bool sortable = true;
    if (n >= 2) {
        for (const Val& v : items) {   // unbool's markers, None and dicts never order; neither do mixed kinds
            const int kv = v.number() ? 0 : v.k == K::Str ? 1 : v.k == K::List ? 2 : 3;
            if (kv == 3 || (kind >= 0 && kv != kind)) {
                sortable = false;
                break;
            }
            kind = kv;
        }
    }
    if (sortable) {
        try {
            const std::vector<const Val*> sorted = py_sorted(items, kind == 2);
            for (size_t j = 1; j < sorted.size(); ++j)
                if (js_equal(*sorted[j - 1], *sorted[j], frames)) return false;
            return true;
        } catch (const Unordered&) {
        }
    }
    for (size_t j = 1; j < n; ++j)
        for (size_t m = 0; m < j; ++m)
            if (js_equal(items[m], items[j], frames)) return false;
    return true;
}

// ============================================================================================ json.dumps
/// a str as json.dumps(ensure_ascii=False) writes it; a lone surrogate, which Python writes as is (no UTF-8), as its
/// \u escape
void dump_str(std::string& out, std::string_view s) {
    out += '"';
    char buf[16];
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const uint32_t cp = cp_at(s, i, &n);
        switch (cp) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (cp < 0x20 || (cp >= 0xD800 && cp <= 0xDFFF)) {
                    std::snprintf(buf, sizeof buf, "\\u%04x", cp);
                    out += buf;
                } else {
                    utf8_append(out, cp);
                }
        }
        i += n;
    }
    out += '"';
}

/// json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
std::string dumps(const Val& root) {
    std::string out;
    struct Open {
        const Val* v;
        size_t next;
    };
    std::vector<Open> st;
    auto visit = [&](const Val& v) {
        switch (v.k) {
            case K::None: out += "null"; break;
            case K::Bool: out += v.b ? "true" : "false"; break;
            case K::Int: out += std::to_string(v.i); break;
            case K::Big: out += v.s; break;
            case K::Float:
                if (!std::isfinite(v.f))
                    raise("ValueError", "Out of range float values are not JSON compliant: " + py_repr(v.f));
                out += py_repr(v.f);
                break;
            case K::Str: dump_str(out, v.s); break;
            case K::List:
                out += '[';
                st.push_back({&v, 0});
                break;
            case K::Dict:
                out += '{';
                st.push_back({&v, 0});
                break;
        }
    };
    visit(root);
    while (!st.empty()) {
        const Val& v = *st.back().v;
        const bool list = v.k == K::List;
        const size_t j = st.back().next++;
        if (j == (list ? v.list.size() : v.dict.size())) {
            out += list ? ']' : '}';
            st.pop_back();
            continue;
        }
        if (j > 0) out += ',';
        if (list) {
            visit(v.list[j]);
        } else {
            dump_str(out, v.dict[j].key);
            out += ':';
            visit(v.dict[j].val);
        }
    }
    return out;
}

// ============================================================================================ json.loads
/// json.loads(text, object_pairs_hook=<duplicate keys are errors>, parse_constant=<NaN and Infinity are errors>):
/// CPython's C scanner, its messages and positions (in characters)
class Decoder {
public:
    explicit Decoder(std::string_view text) : s_(code_points(text)), n_(s_.size()) {}
    Val decode();

private:
    struct Stop {   ///< StopIteration(idx): "Expecting value"
        size_t idx;
    };
    /// Python's C recursion limit, less the frames below json.loads
    static constexpr size_t kMaxDepth = 2995;

    [[noreturn]] void fail(const char* msg, size_t pos) const;
    size_t skip(size_t i) const {
        while (i < n_ && (s_[i] == ' ' || s_[i] == '\t' || s_[i] == '\n' || s_[i] == '\r')) ++i;
        return i;
    }
    bool at(size_t i, const char* word) const {
        const size_t len = std::strlen(word);
        if (i + len > n_) return false;
        for (size_t k = 0; k < len; ++k)
            if (s_[i + k] != (char32_t) word[k]) return false;
        return true;
    }
    std::string scanstring(size_t end, size_t* next) const;
    Val number(size_t start, size_t* next) const;
    Val scan(size_t idx, size_t* next) const;

    std::u32string s_;
    size_t n_;
};

void Decoder::fail(const char* msg, size_t pos) const {
    size_t line = 1, nl = npos;
    for (size_t j = 0; j < pos && j < n_; ++j) {
        if (s_[j] == '\n') {
            ++line;
            nl = j;
        }
    }
    const size_t col = nl == npos ? pos + 1 : pos - nl;
    char buf[96];
    std::snprintf(buf, sizeof buf, ": line %zu column %zu (char %zu)", line, col, pos);
    raise("ValueError", std::string(msg) + buf);
}

int hex_value(char32_t c) {
    if (c >= '0' && c <= '9') return (int) (c - '0');
    if (c >= 'a' && c <= 'f') return (int) (c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return (int) (c - 'A' + 10);
    return -1;
}

std::string Decoder::scanstring(size_t end, size_t* next) const {
    const size_t begin = end - 1;
    std::string out;
    for (;;) {
        size_t j = end;
        char32_t c = 0;
        for (; j < n_; ++j) {
            c = s_[j];
            if (c == '"' || c == '\\') break;
            if (c <= 0x1F) fail("Invalid control character at", j);
        }
        if (j >= n_) fail("Unterminated string starting at", begin);
        for (size_t k = end; k < j; ++k) utf8_append(out, (uint32_t) s_[k]);
        ++j;
        if (c == '"') {
            end = j;
            break;
        }
        if (j == n_) fail("Unterminated string starting at", begin);
        c = s_[j];
        uint32_t ch = 0;
        if (c != 'u') {
            end = j + 1;
            switch (c) {
                case '"': ch = '"'; break;
                case '\\': ch = '\\'; break;
                case '/': ch = '/'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                default: fail("Invalid \\escape", end - 2);
            }
        } else {
            ++j;
            end = j + 4;
            if (end >= n_) fail("Invalid \\uXXXX escape", j - 1);
            for (; j < end; ++j) {
                const int d = hex_value(s_[j]);
                if (d < 0) fail("Invalid \\uXXXX escape", end - 5);
                ch = (ch << 4) | (uint32_t) d;
            }
            if (ch >= 0xD800 && ch <= 0xDBFF && end + 6 < n_ && s_[j] == '\\' && s_[j + 1] == 'u') {   // a pair
                j += 2;
                uint32_t c2 = 0;
                end += 6;
                for (; j < end; ++j) {
                    const int d = hex_value(s_[j]);
                    if (d < 0) fail("Invalid \\uXXXX escape", end - 5);
                    c2 = (c2 << 4) | (uint32_t) d;
                }
                if (c2 >= 0xDC00 && c2 <= 0xDFFF) ch = 0x10000 + ((ch - 0xD800) << 10) + (c2 - 0xDC00);
                else end -= 6;
            }
        }
        utf8_append(out, ch);
    }
    *next = end;
    return out;
}

Val Decoder::number(size_t start, size_t* next) const {
    const size_t last = n_ - 1;
    auto digit = [this](size_t k) { return s_[k] >= '0' && s_[k] <= '9'; };
    size_t idx = start;
    if (s_[idx] == '-') {
        ++idx;
        if (idx > last) throw Stop{start};
    }
    if (s_[idx] >= '1' && s_[idx] <= '9') {
        ++idx;
        while (idx <= last && digit(idx)) ++idx;
    } else if (s_[idx] == '0') {
        ++idx;
    } else {
        throw Stop{start};
    }
    bool is_float = false;
    if (idx < last && s_[idx] == '.' && digit(idx + 1)) {
        is_float = true;
        idx += 2;
        while (idx <= last && digit(idx)) ++idx;
    }
    if (idx < last && (s_[idx] == 'e' || s_[idx] == 'E')) {
        const size_t e_start = idx++;
        if (idx < last && (s_[idx] == '-' || s_[idx] == '+')) ++idx;
        while (idx <= last && digit(idx)) ++idx;
        if (digit(idx - 1)) is_float = true;
        else idx = e_start;
    }
    std::string num;
    for (size_t k = start; k < idx; ++k) num += (char) s_[k];
    *next = idx;
    if (is_float) {
        double d = 0;
        if (std::from_chars(num.data(), num.data() + num.size(), d).ec == std::errc::result_out_of_range)
            d = std::strtod(num.c_str(), nullptr);   // inf, or 0.0 / a subnormal
        return Val::real(d);
    }
    const size_t digits = num.size() - (num[0] == '-' ? 1 : 0);
    if (digits > 4300)
        raise("ValueError", "Exceeds the limit (4300 digits) for integer string conversion: value has " +
                                std::to_string(digits) + " digits; use sys.set_int_max_str_digits() to increase the limit");
    int64_t v = 0;
    if (std::from_chars(num.data(), num.data() + num.size(), v).ec == std::errc()) return Val::integer(v);
    Val big;
    big.k = K::Big;
    big.s = num;
    return big;
}

/// the object_pairs_hook: the first repeated key is an error
void finish_object(Val& v) {
    if (v.dict.size() > 1) {
        std::unordered_set<std::string_view> seen;
        for (const Member& m : v.dict)
            if (!seen.insert(m.key).second) raise("ValueError", "duplicate JSON key: " + m.key);
    }
    v.index();
}

/// scan_once, with containers on a stack of their own
Val Decoder::scan(size_t idx, size_t* next) const {
    struct Open {
        bool obj;
        Val v;
        std::string key;
    };
    std::vector<Open> st;
    enum class S { Value, Key, Done } state = S::Value;
    Val cur;
    for (;;) {
        if (state == S::Key) {
            if (idx >= n_ || s_[idx] != '"') fail("Expecting property name enclosed in double quotes", idx);
            std::string key = scanstring(idx + 1, &idx);
            idx = skip(idx);
            if (idx >= n_ || s_[idx] != ':') fail("Expecting ':' delimiter", idx);
            idx = skip(idx + 1);
            st.back().key = std::move(key);
            state = S::Value;
            continue;
        }
        if (state == S::Value) {
            if (idx >= n_) throw Stop{idx};
            const char32_t c = s_[idx];
            if (c == '"') {
                cur = Val::str(scanstring(idx + 1, &idx));
            } else if (c == '{' || c == '[') {
                if (st.size() >= kMaxDepth)
                    raise("RecursionError", c == '{' ? "maximum recursion depth exceeded while decoding a JSON object "
                                                       "from a unicode string"
                                                     : "maximum recursion depth exceeded while decoding a JSON array "
                                                       "from a unicode string");
                Open o;
                o.obj = c == '{';
                o.v.k = o.obj ? K::Dict : K::List;
                idx = skip(idx + 1);
                if (idx < n_ && s_[idx] == (o.obj ? U'}' : U']')) {
                    ++idx;
                    cur = std::move(o.v);
                } else {
                    const bool obj = o.obj;
                    st.push_back(std::move(o));
                    state = obj ? S::Key : S::Value;
                    continue;
                }
            } else if (c == 'n' && at(idx, "null")) {
                cur = Val();
                idx += 4;
            } else if (c == 't' && at(idx, "true")) {
                cur = Val::boolean(true);
                idx += 4;
            } else if (c == 'f' && at(idx, "false")) {
                cur = Val::boolean(false);
                idx += 5;
            } else if (c == 'N' && at(idx, "NaN")) {
                raise("ValueError", "invalid JSON constant: NaN");
            } else if (c == 'I' && at(idx, "Infinity")) {
                raise("ValueError", "invalid JSON constant: Infinity");
            } else if (c == '-' && at(idx, "-Infinity")) {
                raise("ValueError", "invalid JSON constant: -Infinity");
            } else {
                cur = number(idx, &idx);
            }
            state = S::Done;
        }
        if (st.empty()) {
            *next = idx;
            return cur;
        }
        Open& o = st.back();
        if (o.obj) o.v.dict.push_back({std::move(o.key), std::move(cur)});
        else o.v.list.push_back(std::move(cur));
        idx = skip(idx);
        if (idx < n_ && s_[idx] == (o.obj ? U'}' : U']')) {
            ++idx;
            if (o.obj) finish_object(o.v);
            cur = std::move(o.v);
            st.pop_back();
            continue;
        }
        if (idx >= n_ || s_[idx] != ',') fail("Expecting ',' delimiter", idx);
        idx = skip(idx + 1);
        state = o.obj ? S::Key : S::Value;
    }
}

Val Decoder::decode() {
    if (n_ > 0 && s_[0] == 0xFEFF) fail("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0);
    size_t end = 0;
    Val v;
    try {
        v = scan(skip(0), &end);
    } catch (const Stop& stop) {
        fail("Expecting value", stop.idx);
    }
    end = skip(end);
    if (end != n_) fail("Extra data", end);
    return v;
}

/// the JSON object taken out of a code fence or the prose around it (#762)
std::string extract_json(std::string_view text) {
    std::string_view s = strip(text);
    if (s.empty()) return "";
    if (starts_with(s, "```")) {
        const size_t first = s.find('\n');
        if (first != npos) {
            const size_t end = s.find("```", first + 1);
            if (end != npos) s = strip(s.substr(first + 1, end - first - 1));
        }
    }
    const size_t start = s.find('{');
    if (start == npos) return std::string(s);
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = start; i < s.size(); ++i) {
        const char ch = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (ch == '\\') esc = true;
            else if (ch == '"') in_str = false;
        } else if (ch == '"') {
            in_str = true;
        } else if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) return std::string(s.substr(start, i + 1 - start));
        }
    }
    return std::string(s.substr(start));
}

// ============================================================================================ re
// Python's `re` for str patterns: re/_parser.py's syntax and errors, re/_compiler.py's flags and case folding, and
// sre's matching, here a backtracking machine over code points that makes sre's attempts in sre's order. search() is
// all jsonschema needs: whether the pattern matches anywhere.
namespace re {

enum Flag : int {
    kTemplate = 1,
    kIgnoreCase = 2,
    kLocale = 4,
    kMultiline = 8,
    kDotAll = 16,
    kUnicode = 32,
    kVerbose = 64,
    kDebug = 128,
    kAscii = 256
};
constexpr int kTypeFlags = kAscii | kLocale | kUnicode;
constexpr int kGlobalFlags = kDebug | kTemplate;
constexpr uint32_t kMaxRepeat = 0xFFFFFFFFu;   // _sre.MAXREPEAT
constexpr uint64_t kMaxCode = 0xFFFFFFFFu;
constexpr int64_t kMaxGroups = 1073741823;     // _sre.MAXGROUPS
constexpr int kMaxNesting = 100;               // _parse levels (Python's recursion limit allows about 490): a
                                               // pattern's stack, here, grows by about 2 KB a level
constexpr uint64_t kMaxSteps = 100000000;      // a search's budget (Python's has none)

enum class Op : uint8_t {
    Literal, NotLiteral, Any, In, At, Branch, Subpattern, Atomic, MaxRepeat, MinRepeat, PossRepeat, GroupRef,
    GroupRefExists, Assert, AssertNot, Category, Range, Negate
};
enum AtCode : uint32_t {
    AtBeginning, AtBeginningLine, AtBeginningString, AtBoundary, AtNonBoundary, AtEnd, AtEndLine, AtEndString,
    AtUniBoundary, AtUniNonBoundary
};
enum CatCode : uint32_t { CatDigit, CatNotDigit, CatSpace, CatNotSpace, CatWord, CatNotWord };

struct Seq;
using SeqP = std::shared_ptr<Seq>;
/// one (op, av) item of a parsed pattern
struct Node {
    Op op = Op::Literal;
    uint32_t a = 0, b = 0;    ///< a character, code, range, repeat's bounds, group; Assert: 1 for a lookbehind
    int add = 0, del = 0;     ///< a Subpattern's flags
    std::vector<Node> set;    ///< In
    SeqP sub, no;             ///< the body; GroupRefExists' yes and no
    std::vector<SeqP> alts;   ///< Branch
};
/// a SubPattern
struct Seq {
    std::vector<Node> items;
    mutable bool sized = false;
    mutable uint64_t lo = 0, hi = 0;
};

Node node(Op op, uint32_t a = 0, uint32_t b = 0) {
    Node n;
    n.op = op;
    n.a = a;
    n.b = b;
    return n;
}

struct State {
    int flags = 0;
    std::map<std::u32string, int> groupdict;
    std::vector<std::optional<std::pair<uint64_t, uint64_t>>> widths{std::nullopt};   ///< per group, once closed
    std::optional<int> lookbehind;                                                     ///< lookbehindgroups
    std::vector<std::pair<int64_t, size_t>> grouprefpos;
    int groups() const { return (int) widths.size(); }
    bool checkgroup(int64_t gid) const { return gid < groups() && widths[(size_t) gid].has_value(); }
};

uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a != 0 && b > UINT64_MAX / a ? UINT64_MAX : a * b; }

/// SubPattern.getwidth (MAXWIDTH as UINT64_MAX)
std::pair<uint64_t, uint64_t> width(const Seq& p, const State& st) {
    if (p.sized) return {p.lo, p.hi};
    uint64_t lo = 0, hi = 0;
    for (const Node& n : p.items) {
        switch (n.op) {
            case Op::Branch: {
                uint64_t i = UINT64_MAX, j = 0;
                for (const SeqP& alt : n.alts) {
                    const auto w = width(*alt, st);
                    i = std::min(i, w.first);
                    j = std::max(j, w.second);
                }
                lo = sat_add(lo, i);
                hi = sat_add(hi, j);
                break;
            }
            case Op::Atomic:
            case Op::Subpattern: {
                const auto w = width(*n.sub, st);
                lo = sat_add(lo, w.first);
                hi = sat_add(hi, w.second);
                break;
            }
            case Op::MaxRepeat:
            case Op::MinRepeat:
            case Op::PossRepeat: {
                const auto w = width(*n.sub, st);
                lo = sat_add(lo, sat_mul(w.first, n.a));
                hi = n.b == kMaxRepeat && w.second ? UINT64_MAX : sat_add(hi, sat_mul(w.second, n.b));
                break;
            }
            case Op::Any:
            case Op::In:
            case Op::Literal:
            case Op::NotLiteral:
            case Op::Category:
            case Op::Range:
                lo = sat_add(lo, 1);
                hi = sat_add(hi, 1);
                break;
            case Op::GroupRef:
                if (const auto& w = st.widths[n.a]) {
                    lo = sat_add(lo, w->first);
                    hi = sat_add(hi, w->second);
                }
                break;
            case Op::GroupRefExists: {
                auto w = width(*n.sub, st);
                if (n.no) {
                    const auto o = width(*n.no, st);
                    w.first = std::min(w.first, o.first);
                    w.second = std::max(w.second, o.second);
                } else {
                    w.first = 0;
                }
                lo = sat_add(lo, w.first);
                hi = sat_add(hi, w.second);
                break;
            }
            default: break;
        }
    }
    p.sized = true;
    p.lo = lo;
    p.hi = hi;
    return {lo, hi};
}

/// re.error(msg, pattern, pos)
[[noreturn]] void error(const std::u32string& pattern, std::string msg, std::optional<size_t> pos) {
    if (pos) {
        msg += " at position " + std::to_string(*pos);
        if (pattern.find(U'\n') != std::u32string::npos) {
            size_t line = 1, nl = npos;
            for (size_t k = 0; k < *pos && k < pattern.size(); ++k) {
                if (pattern[k] == '\n') {
                    ++line;
                    nl = k;
                }
            }
            msg += " (line " + std::to_string(line) + ", column " +
                   std::to_string(nl == npos ? *pos + 1 : *pos - nl) + ")";
        }
    }
    raise("re.error", std::move(msg));
}

bool is_oct(char32_t c) { return c >= '0' && c <= '7'; }
bool is_hex(char32_t c) { return hex_value(c) >= 0; }
bool is_digit32(char32_t c) { return c >= '0' && c <= '9'; }

int flag_of(const std::u32string& t) {
    if (t.size() != 1) return 0;
    switch (t[0]) {
        case 'i': return kIgnoreCase;
        case 'L': return kLocale;
        case 'm': return kMultiline;
        case 's': return kDotAll;
        case 'x': return kVerbose;
        case 'a': return kAscii;
        case 't': return kTemplate;
        case 'u': return kUnicode;
        default: return 0;
    }
}

/// str.isidentifier(), with llama.cpp's letters and numbers for XID_Start and XID_Continue
bool py_isidentifier(const std::u32string& s) {
    if (s.empty()) return false;
    for (size_t k = 0; k < s.size(); ++k) {
        const uint32_t c = (uint32_t) s[k];
        const unicode_cpt_flags f = unicode_cpt_flags_from_cpt(c);
        const bool start = c == '_' || (c < 0x80 ? is_ascii_letter(c) : f.is_letter && !after_unicode_15(c));
        const bool cont = start || (c < 0x80 ? is_digit(c) : (f.is_number || f.is_accent_mark) && !after_unicode_15(c));
        if (k == 0 ? !start : !cont) return false;
    }
    return true;
}

/// the ESCAPES table: \a \b \f \n \r \t \v \\ as their characters
int simple_escape(char32_t c) {
    switch (c) {
        case 'a': return 7;
        case 'b': return 8;
        case 'f': return 12;
        case 'n': return 10;
        case 'r': return 13;
        case 't': return 9;
        case 'v': return 11;
        case '\\': return '\\';
        default: return -1;
    }
}

/// \d \D \s \S \w \W: (IN, [(CATEGORY, ...)])
std::optional<Node> category_escape(char32_t c) {
    uint32_t cat = 0;
    switch (c) {
        case 'd': cat = CatDigit; break;
        case 'D': cat = CatNotDigit; break;
        case 's': cat = CatSpace; break;
        case 'S': cat = CatNotSpace; break;
        case 'w': cat = CatWord; break;
        case 'W': cat = CatNotWord; break;
        default: return std::nullopt;
    }
    Node in = node(Op::In);
    in.set.push_back(node(Op::Category, cat));
    return in;
}

uint32_t parse_base(const std::u32string& digits, int base) {
    uint32_t v = 0;
    for (const char32_t c : digits) v = v * (uint32_t) base + (uint32_t) hex_value(c);
    return v;
}

/// re/_parser.py: the tokenizer and the parser
class Parser {
public:
    explicit Parser(std::u32string pattern) : pat_(std::move(pattern)) {}
    SeqP parse();
    State state;
    const std::u32string& pattern() const { return pat_; }

private:
    void advance();
    bool match(char32_t c) {
        if (next_is(c)) {
            advance();
            return true;
        }
        return false;
    }
    std::optional<std::u32string> get() {
        std::optional<std::u32string> t = next_;
        advance();
        return t;
    }
    bool next_is(char32_t c) const { return next_ && next_->size() == 1 && (*next_)[0] == c; }
    bool next_in(bool (*in)(char32_t)) const { return next_ && next_->size() == 1 && in((*next_)[0]); }
    std::u32string getwhile(int n, bool (*in)(char32_t)) {
        std::u32string out;
        for (int k = 0; k < n && next_in(in); ++k) {
            out += *next_;
            advance();
        }
        return out;
    }
    std::u32string getuntil(char32_t terminator, const char* name);
    size_t tell() const { return index_ - (next_ ? next_->size() : 0); }
    void seek(size_t i) {
        index_ = i;
        advance();
    }
    [[noreturn]] void fail(std::string msg, size_t offset = 0) const { error(pat_, std::move(msg), tell() - offset); }
    void checkgroupname(const std::u32string& name, size_t offset) const {
        if (!py_isidentifier(name)) fail("bad character in group name " + py_repr_str(utf8_of(name)), name.size() + offset);
    }
    void checklookbehind(int64_t gid) const {
        if (!state.lookbehind) return;
        if (!state.checkgroup(gid)) fail("cannot refer to an open group");
        if (gid >= *state.lookbehind) fail("cannot refer to group defined in the same lookbehind subpattern");
    }
    Node unicode_escape(std::u32string esc);
    Node escape(std::u32string esc);
    Node class_escape(std::u32string esc);
    SeqP parse_sub(bool verbose, int nested);
    SeqP parse_seq(bool verbose, int nested, bool first);
    std::optional<std::pair<int, int>> parse_flags(std::u32string ch);

    struct Depth {   ///< a level of _parse
        explicit Depth(Parser& p) : parser(p) {
            if (++parser.depth_ > kMaxNesting) raise("RecursionError", "maximum recursion depth exceeded");
        }
        ~Depth() { --parser.depth_; }
        Depth(const Depth&) = delete;
        Depth& operator=(const Depth&) = delete;
        Parser& parser;
    };

    std::u32string pat_;
    size_t index_ = 0;
    std::optional<std::u32string> next_;
    int depth_ = 0;
};

void Parser::advance() {
    size_t i = index_;
    if (i >= pat_.size()) {
        next_.reset();
        return;
    }
    std::u32string ch(1, pat_[i]);
    if (pat_[i] == '\\') {
        ++i;
        if (i >= pat_.size()) error(pat_, "bad escape (end of pattern)", pat_.size() - 1);
        ch += pat_[i];
    }
    index_ = i + 1;
    next_ = std::move(ch);
}

std::u32string Parser::getuntil(char32_t terminator, const char* name) {
    std::u32string out;
    for (;;) {
        const std::optional<std::u32string> c = next_;
        advance();
        if (!c) {
            if (out.empty()) fail(std::string("missing ") + name);
            fail("missing " + utf8_of(std::u32string(1, terminator)) + ", unterminated name", out.size());
        }
        if (c->size() == 1 && (*c)[0] == terminator) {
            if (out.empty()) fail(std::string("missing ") + name, 1);
            break;
        }
        out += *c;
    }
    return out;
}

/// \x \u \U \N, the escapes classes and patterns share
Node Parser::unicode_escape(std::u32string esc) {
    const char32_t c = esc[1];
    if (c == 'x') {
        esc += getwhile(2, is_hex);
        if (esc.size() != 4) fail("incomplete escape " + utf8_of(esc), esc.size());
        return node(Op::Literal, parse_base(esc.substr(2), 16));
    }
    if (c == 'u') {
        esc += getwhile(4, is_hex);
        if (esc.size() != 6) fail("incomplete escape " + utf8_of(esc), esc.size());
        return node(Op::Literal, parse_base(esc.substr(2), 16));
    }
    if (c == 'U') {
        esc += getwhile(8, is_hex);
        if (esc.size() != 10) fail("incomplete escape " + utf8_of(esc), esc.size());
        const uint64_t v = std::strtoull(utf8_of(esc.substr(2)).c_str(), nullptr, 16);
        if (v > 0x10FFFF) fail("bad escape " + utf8_of(esc), esc.size());   // chr() raises ValueError
        return node(Op::Literal, (uint32_t) v);
    }
    // \N{name}: Python looks the name up in the Unicode database, which is not here
    if (!match('{')) fail("missing {");
    const std::u32string name = getuntil('}', "character name");
    fail("undefined character name " + py_repr_str(utf8_of(name)), name.size() + 4);
}

Node Parser::escape(std::u32string esc) {
    const char32_t c = esc[1];
    switch (c) {   // CATEGORIES first: \b is a boundary here
        case 'A': return node(Op::At, AtBeginningString);
        case 'b': return node(Op::At, AtBoundary);
        case 'B': return node(Op::At, AtNonBoundary);
        case 'Z': return node(Op::At, AtEndString);
        default: break;
    }
    if (std::optional<Node> cat = category_escape(c)) return *cat;
    if (const int e = simple_escape(c); e >= 0) return node(Op::Literal, (uint32_t) e);
    if (c == 'x' || c == 'u' || c == 'U' || c == 'N') return unicode_escape(std::move(esc));
    if (c == '0') {
        esc += getwhile(2, is_oct);
        return node(Op::Literal, parse_base(esc.substr(1), 8));
    }
    if (is_digit32(c)) {   // an octal escape or a group reference
        if (next_in(is_digit32)) {
            esc += *get();
            if (is_oct(esc[1]) && is_oct(esc[2]) && next_in(is_oct)) {
                esc += *get();
                const uint32_t v = parse_base(esc.substr(1), 8);
                if (v > 0377) fail("octal escape value " + utf8_of(esc) + " outside of range 0-0o377", esc.size());
                return node(Op::Literal, v);
            }
        }
        const int64_t group = (int64_t) parse_base(esc.substr(1), 10);
        if (group < state.groups()) {
            if (!state.checkgroup(group)) fail("cannot refer to an open group", esc.size());
            checklookbehind(group);
            return node(Op::GroupRef, (uint32_t) group);
        }
        fail("invalid group reference " + std::to_string(group), esc.size() - 1);
    }
    if (is_ascii_letter(c)) fail("bad escape " + utf8_of(esc), esc.size());
    return node(Op::Literal, (uint32_t) c);
}

Node Parser::class_escape(std::u32string esc) {
    const char32_t c = esc[1];
    if (const int e = simple_escape(c); e >= 0) return node(Op::Literal, (uint32_t) e);   // \b is a backspace here
    if (std::optional<Node> cat = category_escape(c)) return *cat;
    if (c == 'x' || c == 'u' || c == 'U' || c == 'N') return unicode_escape(std::move(esc));
    if (is_oct(c)) {
        esc += getwhile(2, is_oct);
        const uint32_t v = parse_base(esc.substr(1), 8);
        if (v > 0377) fail("octal escape value " + utf8_of(esc) + " outside of range 0-0o377", esc.size());
        return node(Op::Literal, v);
    }
    if (is_digit32(c) || is_ascii_letter(c)) fail("bad escape " + utf8_of(esc), esc.size());
    return node(Op::Literal, (uint32_t) c);
}

std::optional<std::pair<int, int>> Parser::parse_flags(std::u32string ch) {
    int add = 0, del = 0;
    auto alpha = [](const std::u32string& t) { return t.size() == 1 && py_alpha((uint32_t) t[0]); };
    if (ch != U"-") {
        for (;;) {
            const int flag = flag_of(ch);
            if (ch == U"L") fail("bad inline flags: cannot use 'L' flag with a str pattern");
            add |= flag;
            if ((flag & kTypeFlags) && (add & kTypeFlags) != flag)
                fail("bad inline flags: flags 'a', 'u' and 'L' are incompatible");
            const std::optional<std::u32string> t = get();
            if (!t) fail("missing -, : or )");
            ch = *t;
            if (ch == U")" || ch == U"-" || ch == U":") break;
            if (!flag_of(ch)) fail(alpha(ch) ? "unknown flag" : "missing -, : or )", ch.size());
        }
    }
    if (ch == U")") {
        state.flags |= add;
        return std::nullopt;
    }
    if (add & kGlobalFlags) fail("bad inline flags: cannot turn on global flag", 1);
    if (ch == U"-") {
        const std::optional<std::u32string> t = get();
        if (!t) fail("missing flag");
        ch = *t;
        if (!flag_of(ch)) fail(alpha(ch) ? "unknown flag" : "missing flag", ch.size());
        for (;;) {
            const int flag = flag_of(ch);
            if (flag & kTypeFlags) fail("bad inline flags: cannot turn off flags 'a', 'u' and 'L'");
            del |= flag;
            const std::optional<std::u32string> u = get();
            if (!u) fail("missing :");
            ch = *u;
            if (ch == U":") break;
            if (!flag_of(ch)) fail(alpha(ch) ? "unknown flag" : "missing :", ch.size());
        }
    }
    if (del & kGlobalFlags) fail("bad inline flags: cannot turn off global flag", 1);
    if (add & del) fail("bad inline flags: flag turned on and off", 1);
    return std::make_pair(add, del);
}

SeqP Parser::parse_sub(bool verbose, int nested) {
    std::vector<SeqP> items;
    for (;;) {
        items.push_back(parse_seq(verbose, nested + 1, nested == 0 && items.empty()));
        if (!match('|')) break;
        if (nested == 0) verbose = (state.flags & kVerbose) != 0;
    }
    if (items.size() == 1) return items[0];
    auto p = std::make_shared<Seq>();
    Node br = node(Op::Branch);
    br.alts = std::move(items);
    p->items.push_back(std::move(br));
    return p;
}

/// repeat bounds as int(digits), which must stay below MAXREPEAT
uint32_t repeat_bound(const std::u32string& digits) {
    uint64_t v = 0;
    for (const char32_t c : digits) {
        v = v * 10 + (uint64_t) (c - '0');
        if (v >= kMaxRepeat) raise("OverflowError", "the repetition number is too large");
    }
    return (uint32_t) v;
}

bool verbose_space(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

SeqP Parser::parse_seq(bool verbose, int nested, bool first) {
    const Depth frame(*this);
    auto seq = std::make_shared<Seq>();
    std::vector<Node>& items = seq->items;
    while (next_ && !next_is('|') && !next_is(')')) {
        const std::u32string th = *next_;
        advance();
        if (verbose) {
            if (th.size() == 1 && verbose_space(th[0])) continue;
            if (th == U"#") {
                for (;;) {
                    const std::optional<std::u32string> t = get();
                    if (!t || *t == U"\n") break;
                }
                continue;
            }
        }
        if (th[0] == '\\') {
            items.push_back(escape(th));
            continue;
        }
        const char32_t c = th[0];
        if (std::u32string_view(U".\\[{()*+?^$|").find(c) == std::u32string_view::npos) {
            items.push_back(node(Op::Literal, (uint32_t) c));
        } else if (c == '[') {
            const size_t here = tell() - 1;
            std::vector<Node> set;
            const bool negate = match('^');
            for (;;) {
                const std::optional<std::u32string> t = get();
                if (!t) fail("unterminated character set", tell() - here);
                const std::u32string th1 = *t;
                if (th1 == U"]" && !set.empty()) break;
                Node code1 = th1[0] == '\\' ? class_escape(th1) : node(Op::Literal, (uint32_t) th1[0]);
                if (match('-')) {
                    const std::optional<std::u32string> t2 = get();
                    if (!t2) fail("unterminated character set", tell() - here);
                    const std::u32string that = *t2;
                    if (that == U"]") {
                        if (code1.op == Op::In) code1 = code1.set[0];
                        set.push_back(code1);
                        set.push_back(node(Op::Literal, '-'));
                        break;
                    }
                    const Node code2 = that[0] == '\\' ? class_escape(that) : node(Op::Literal, (uint32_t) that[0]);
                    const std::string msg = "bad character range " + utf8_of(th1) + "-" + utf8_of(that);
                    if (code1.op != Op::Literal || code2.op != Op::Literal || code2.a < code1.a)
                        fail(msg, th1.size() + 1 + that.size());
                    set.push_back(node(Op::Range, code1.a, code2.a));
                } else {
                    if (code1.op == Op::In) code1 = code1.set[0];
                    set.push_back(code1);
                }
            }
            bool one = !set.empty() && set[0].op == Op::Literal;   // after _uniq: one literal
            for (const Node& n : set) one = one && n.op == Op::Literal && n.a == set[0].a;
            if (one) {
                items.push_back(node(negate ? Op::NotLiteral : Op::Literal, set[0].a));
            } else {
                Node in = node(Op::In);
                if (negate) in.set.push_back(node(Op::Negate));
                in.set.insert(in.set.end(), set.begin(), set.end());
                items.push_back(std::move(in));
            }
        } else if (c == '*' || c == '+' || c == '?' || c == '{') {
            const size_t here = tell();
            uint32_t mn = 0, mx = kMaxRepeat;
            if (c == '?') {
                mx = 1;
            } else if (c == '+') {
                mn = 1;
            } else if (c == '{') {
                if (next_is('}')) {
                    items.push_back(node(Op::Literal, '{'));
                    continue;
                }
                std::u32string lo, hi;
                while (next_in(is_digit32)) lo += *get();
                if (match(',')) {
                    while (next_in(is_digit32)) hi += *get();
                } else {
                    hi = lo;
                }
                if (!match('}')) {
                    items.push_back(node(Op::Literal, '{'));
                    seek(here);
                    continue;
                }
                if (!lo.empty()) mn = repeat_bound(lo);
                if (!hi.empty()) {
                    mx = repeat_bound(hi);
                    if (mx < mn) fail("min repeat greater than max repeat", tell() - here);
                }
            }
            if (items.empty() || items.back().op == Op::At) fail("nothing to repeat", tell() - here + th.size());
            const Op last = items.back().op;
            if (last == Op::MaxRepeat || last == Op::MinRepeat || last == Op::PossRepeat)
                fail("multiple repeat", tell() - here + th.size());
            Node item = std::move(items.back());
            SeqP body;
            if (item.op == Op::Subpattern && item.a == 0 && !item.add && !item.del) {
                body = item.sub;
            } else {
                body = std::make_shared<Seq>();
                body->items.push_back(std::move(item));
            }
            Node rep = node(Op::MaxRepeat, mn, mx);
            rep.sub = body;
            if (match('?')) rep.op = Op::MinRepeat;
            else if (match('+')) rep.op = Op::PossRepeat;
            items.back() = std::move(rep);
        } else if (c == '.') {
            items.push_back(node(Op::Any));
        } else if (c == '(') {
            const size_t start = tell() - 1;
            bool capture = true, atomic = false;
            std::optional<std::u32string> name;
            int add = 0, del = 0;
            if (match('?')) {
                const std::optional<std::u32string> t = get();
                if (!t) fail("unexpected end of pattern");
                std::u32string ch = *t;
                if (ch == U"P") {
                    if (match('<')) {
                        name = getuntil('>', "group name");
                        checkgroupname(*name, 1);
                    } else if (match('=')) {
                        const std::u32string ref = getuntil(')', "group name");
                        checkgroupname(ref, 1);
                        const auto it = state.groupdict.find(ref);
                        if (it == state.groupdict.end()) fail("unknown group name " + py_repr_str(utf8_of(ref)), ref.size() + 1);
                        if (!state.checkgroup(it->second)) fail("cannot refer to an open group", ref.size() + 1);
                        checklookbehind(it->second);
                        items.push_back(node(Op::GroupRef, (uint32_t) it->second));
                        continue;
                    } else {
                        const std::optional<std::u32string> t2 = get();
                        if (!t2) fail("unexpected end of pattern");
                        fail("unknown extension ?P" + utf8_of(*t2), t2->size() + 2);
                    }
                } else if (ch == U":") {
                    capture = false;
                } else if (ch == U"#") {
                    for (;;) {
                        if (!next_) fail("missing ), unterminated comment", tell() - start);
                        if (*get() == U")") break;
                    }
                    continue;
                } else if (ch == U"=" || ch == U"!" || ch == U"<") {
                    bool behind = false;
                    std::optional<int> saved;
                    if (ch == U"<") {
                        const std::optional<std::u32string> t2 = get();
                        if (!t2) fail("unexpected end of pattern");
                        ch = *t2;
                        if (ch != U"=" && ch != U"!") fail("unknown extension ?<" + utf8_of(ch), ch.size() + 2);
                        behind = true;
                        saved = state.lookbehind;
                        if (!saved) state.lookbehind = state.groups();
                    }
                    SeqP p = parse_sub(verbose, nested + 1);
                    if (behind && !saved) state.lookbehind.reset();
                    if (!match(')')) fail("missing ), unterminated subpattern", tell() - start);
                    Node as = node(ch == U"=" ? Op::Assert : Op::AssertNot, behind ? 1 : 0);
                    as.sub = p;
                    items.push_back(std::move(as));
                    continue;
                } else if (ch == U"(") {
                    const std::u32string condname = getuntil(')', "group name");
                    int64_t condgroup = 0;
                    if (!std::all_of(condname.begin(), condname.end(), is_digit32)) {
                        checkgroupname(condname, 1);
                        const auto it = state.groupdict.find(condname);
                        if (it == state.groupdict.end())
                            fail("unknown group name " + py_repr_str(utf8_of(condname)), condname.size() + 1);
                        condgroup = it->second;
                    } else {
                        std::string digits = utf8_of(condname);
                        digits.erase(0, std::min(digits.find_first_not_of('0'), digits.size() - 1));
                        condgroup = digits.size() > 18 ? INT64_MAX : std::stoll(digits);
                        if (!condgroup) fail("bad group number", condname.size() + 1);
                        if (condgroup >= kMaxGroups) fail("invalid group reference " + digits, condname.size() + 1);
                        const bool seen = std::any_of(state.grouprefpos.begin(), state.grouprefpos.end(),
                                                      [&](const auto& g) { return g.first == condgroup; });
                        if (!seen) state.grouprefpos.push_back({condgroup, tell() - condname.size() - 1});
                    }
                    checklookbehind(condgroup);
                    SeqP yes = parse_seq(verbose, nested + 1, false);
                    SeqP no;
                    if (match('|')) {
                        no = parse_seq(verbose, nested + 1, false);
                        if (next_is('|')) fail("conditional backref with more than two branches");
                    }
                    if (!match(')')) fail("missing ), unterminated subpattern", tell() - start);
                    Node ge = node(Op::GroupRefExists, (uint32_t) condgroup);
                    ge.sub = yes;
                    ge.no = no;
                    items.push_back(std::move(ge));
                    continue;
                } else if (ch == U">") {
                    capture = false;
                    atomic = true;
                } else if (flag_of(ch) || ch == U"-") {
                    const std::optional<std::pair<int, int>> flags = parse_flags(ch);
                    if (!flags) {   // global flags
                        if (!first || !items.empty())
                            fail("global flags not at the start of the expression", tell() - start);
                        verbose = (state.flags & kVerbose) != 0;
                        continue;
                    }
                    add = flags->first;
                    del = flags->second;
                    capture = false;
                } else {
                    fail("unknown extension ?" + utf8_of(ch), ch.size() + 1);
                }
            }
            int group = 0;
            if (capture) {
                group = state.groups();
                state.widths.push_back(std::nullopt);
                if (name) {
                    const auto it = state.groupdict.find(*name);
                    if (it != state.groupdict.end())
                        fail("redefinition of group name " + py_repr_str(utf8_of(*name)) + " as group " +
                                 std::to_string(group) + "; was group " + std::to_string(it->second),
                             name->size() + 1);
                    state.groupdict[*name] = group;
                }
            }
            const bool sub_verbose = (verbose || (add & kVerbose)) && !(del & kVerbose);
            SeqP p = parse_sub(sub_verbose, nested + 1);
            if (!match(')')) fail("missing ), unterminated subpattern", tell() - start);
            if (group) state.widths[(size_t) group] = width(*p, state);
            Node sp = node(atomic ? Op::Atomic : Op::Subpattern, (uint32_t) group);
            sp.add = add;
            sp.del = del;
            sp.sub = p;
            items.push_back(std::move(sp));
        } else if (c == '^') {
            items.push_back(node(Op::At, AtBeginning));
        } else if (c == '$') {
            items.push_back(node(Op::At, AtEnd));
        }
    }
    return seq;
}

SeqP Parser::parse() {
    advance();
    SeqP p = parse_sub(false, 0);
    if (state.flags & kLocale) raise("ValueError", "cannot use LOCALE flag with a str pattern");
    if (!(state.flags & kAscii)) state.flags |= kUnicode;
    else if (state.flags & kUnicode) raise("ValueError", "ASCII and UNICODE flags are incompatible");
    if (next_) fail("unbalanced parenthesis");
    for (const auto& [g, pos] : state.grouprefpos)
        if (g >= state.groups()) error(pat_, "invalid group reference " + std::to_string(g), pos);
    return p;
}

// ---- the program
/// a character set: ranges of characters (case folded where the pattern ignores case) and categories
struct CharSet {
    bool negate = false;
    uint8_t fold = 0;   ///< what is tested: 0 the character, 1 its lower case, 2 its ASCII lower case
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    std::vector<std::pair<uint32_t, uint32_t>> uni_ranges;   ///< RANGE_UNI_IGNORE: the character or its upper case
    std::vector<std::pair<uint32_t, bool>> cats;              ///< a category, Unicode or ASCII
    bool match(uint32_t ch) const;
};

uint32_t fold(uint32_t ch, uint8_t mode) {
    if (mode == 1) return sre_lower(ch);
    if (mode == 2) return ch >= 'A' && ch <= 'Z' ? ch + 32 : ch;
    return ch;
}

bool category(uint32_t cat, bool uni, uint32_t ch) {
    switch (cat) {
        case CatDigit: return uni ? py_decimal(ch) >= 0 : is_digit(ch);
        case CatNotDigit: return !category(CatDigit, uni, ch);
        case CatSpace: return uni ? py_isspace(ch) : ch == ' ' || (ch >= 9 && ch <= 13);
        case CatNotSpace: return !category(CatSpace, uni, ch);
        case CatWord: return uni ? py_word(ch) : ch < 0x80 && (is_ascii_letter(ch) || is_digit(ch) || ch == '_');
        default: return !category(CatWord, uni, ch);
    }
}

bool CharSet::match(uint32_t ch) const {
    const uint32_t c = re::fold(ch, fold);
    bool hit = false;
    const auto it = std::upper_bound(ranges.begin(), ranges.end(), c,
                                     [](uint32_t x, const std::pair<uint32_t, uint32_t>& r) { return x < r.first; });
    if (it != ranges.begin() && c <= (it - 1)->second) hit = true;
    for (size_t k = 0; !hit && k < uni_ranges.size(); ++k) {
        const uint32_t up = sre_upper(c);
        hit = (c >= uni_ranges[k].first && c <= uni_ranges[k].second) ||
              (up >= uni_ranges[k].first && up <= uni_ranges[k].second);
    }
    for (size_t k = 0; !hit && k < cats.size(); ++k) hit = category(cats[k].first, cats[k].second, c);
    return hit != negate;
}

void merge(std::vector<std::pair<uint32_t, uint32_t>>& r) {
    std::sort(r.begin(), r.end());
    std::vector<std::pair<uint32_t, uint32_t>> m;
    for (const auto& x : r) {
        if (!m.empty() && x.first <= m.back().second + 1) m.back().second = std::max(m.back().second, x.second);
        else m.push_back(x);
    }
    r = std::move(m);
}

enum class I : uint8_t {
    Char, NotChar, Set, Any, AnyAll, At, Split, Jmp, Mark, Rep1, RepInit, Until, Poss, Atomic, Assert, GroupRef,
    GroupExists, Match
};
struct Inst {
    I op = I::Match;
    uint8_t mode = 0;        ///< Char/NotChar/GroupRef: case folding; Rep1: greedy, lazy, possessive; Until: 1 greedy;
                             ///< Assert: 1 negative
    uint32_t x = 0, y = 0;   ///< a character, set, code, slot, repeat or group; jump targets
    uint32_t min = 0, max = 0;
    uint64_t back = 0;       ///< a lookbehind's width
};
struct Prog {
    std::vector<Inst> code;
    std::vector<CharSet> sets;
    int groups = 0;    ///< capturing groups
    int repeats = 0;   ///< repeats of more than one character
};

int combine(int flags, int add, int del) {
    if (add & kTypeFlags) flags &= ~kTypeFlags;
    return (flags | add) & ~del;
}

/// _compiler._simple: one character at a time
bool simple(const Seq& p) {
    if (p.items.size() != 1) return false;
    const Node& n = p.items[0];
    if (n.op == Op::Subpattern) return n.a == 0 && simple(*n.sub);
    return n.op == Op::Literal || n.op == Op::NotLiteral || n.op == Op::Any || n.op == Op::In;
}

bool ascii_iscased(uint32_t ch) { return ch < 0x80 && is_ascii_letter(ch); }

/// re/_compiler.py: the program of a parsed pattern
class Compiler {
public:
    explicit Compiler(const State& st) : st_(st) {}
    Prog prog;
    void seq(const Seq& p, int flags) {
        for (const Node& n : p.items) item(n, flags);
    }
    uint32_t emit(I op) {
        Inst in;
        in.op = op;
        return emit(in);
    }
    uint32_t emit(const Inst& in) {
        prog.code.push_back(in);
        return (uint32_t) prog.code.size() - 1;
    }
    uint32_t pc() const { return (uint32_t) prog.code.size(); }

private:
    void item(const Node& n, int flags);
    void literal(uint32_t ch, bool negate, int flags);
    uint32_t charset(const std::vector<Node>& set, int flags);
    void unit(const Seq& p, int flags) {
        const Node& n = p.items[0];
        if (n.op == Op::Subpattern) unit(*n.sub, combine(flags, n.add, n.del));
        else item(n, flags);
    }
    const State& st_;
};

void Compiler::literal(uint32_t ch, bool negate, int flags) {
    Inst in;
    in.op = negate ? I::NotChar : I::Char;
    in.x = ch;
    const bool uni = (flags & kUnicode) != 0;
    if ((flags & kIgnoreCase) && (uni ? sre_iscased(ch) : ascii_iscased(ch))) {
        const uint32_t lo = uni ? sre_lower(ch) : fold(ch, 2);
        if (const std::vector<uint32_t>* extra = uni ? sre_extra_cases(lo) : nullptr) {
            CharSet cs;
            cs.negate = negate;
            cs.fold = 1;
            cs.ranges.push_back({lo, lo});
            for (const uint32_t e : *extra) cs.ranges.push_back({e, e});
            merge(cs.ranges);
            in.op = I::Set;
            in.x = (uint32_t) prog.sets.size();
            prog.sets.push_back(std::move(cs));
        } else {
            in.x = lo;
            in.mode = uni ? 1 : 2;
        }
    }
    emit(in);
}

/// _optimize_charset's meaning: with IGNORECASE the set holds the lower cases (and their extra cases), and a set with
/// a cased character tests the lower case of the subject's
uint32_t Compiler::charset(const std::vector<Node>& set, int flags) {
    CharSet cs;
    const bool ic = (flags & kIgnoreCase) != 0, uni = (flags & kUnicode) != 0;
    bool hascased = false;
    auto add_folded = [&](uint32_t c) {
        const uint32_t lo = uni ? sre_lower(c) : fold(c, 2);
        cs.ranges.push_back({lo, lo});
        if (const std::vector<uint32_t>* extra = uni ? sre_extra_cases(lo) : nullptr)
            for (const uint32_t e : *extra) cs.ranges.push_back({e, e});
        return lo;
    };
    for (const Node& n : set) {
        switch (n.op) {
            case Op::Negate: cs.negate = true; break;
            case Op::Literal:
                if (!ic) {
                    cs.ranges.push_back({n.a, n.a});
                } else {
                    const uint32_t lo = add_folded(n.a);
                    if (uni ? sre_iscased(lo) : ascii_iscased(lo)) hascased = true;
                }
                break;
            case Op::Range:
                if (!ic) {
                    cs.ranges.push_back({n.a, n.b});
                    break;
                }
                for (uint32_t c = n.a; c <= std::min(n.b, 0xFFFFu); ++c) add_folded(c);
                if (n.b > 0xFFFF) {   // the non-BMP part (IndexError in Python): a range tested on the lower case
                    (uni ? cs.uni_ranges : cs.ranges).push_back({n.a, n.b});
                    hascased = true;
                } else {
                    for (uint32_t c = n.a; c <= n.b && !hascased; ++c) hascased = uni ? sre_iscased(c) : ascii_iscased(c);
                }
                break;
            case Op::Category: cs.cats.push_back({n.a, uni}); break;
            default: break;
        }
    }
    cs.fold = ic && hascased ? (uni ? 1 : 2) : 0;
    merge(cs.ranges);
    prog.sets.push_back(std::move(cs));
    return (uint32_t) prog.sets.size() - 1;
}

void Compiler::item(const Node& n, int flags) {
    switch (n.op) {
        case Op::Literal:
        case Op::NotLiteral: literal(n.a, n.op == Op::NotLiteral, flags); break;
        case Op::In: {
            Inst in;
            in.op = I::Set;
            in.x = charset(n.set, flags);
            emit(in);
            break;
        }
        case Op::Any: emit(flags & kDotAll ? I::AnyAll : I::Any); break;
        case Op::At: {
            uint32_t code = n.a;
            if (flags & kMultiline) {
                if (code == AtBeginning) code = AtBeginningLine;
                else if (code == AtEnd) code = AtEndLine;
            }
            if (flags & kUnicode) {
                if (code == AtBoundary) code = AtUniBoundary;
                else if (code == AtNonBoundary) code = AtUniNonBoundary;
            }
            Inst in;
            in.op = I::At;
            in.x = code;
            emit(in);
            break;
        }
        case Op::Branch: {
            std::vector<uint32_t> ends;
            for (size_t k = 0; k < n.alts.size(); ++k) {
                if (k + 1 == n.alts.size()) {
                    seq(*n.alts[k], flags);
                    break;
                }
                const uint32_t split = emit(I::Split);
                prog.code[split].x = pc();
                seq(*n.alts[k], flags);
                ends.push_back(emit(I::Jmp));
                prog.code[split].y = pc();
            }
            for (const uint32_t e : ends) prog.code[e].x = pc();
            break;
        }
        case Op::Subpattern: {
            Inst mark;
            mark.op = I::Mark;
            if (n.a) {
                mark.x = (n.a - 1) * 2;
                emit(mark);
            }
            seq(*n.sub, combine(flags, n.add, n.del));
            if (n.a) {
                mark.x = (n.a - 1) * 2 + 1;
                emit(mark);
            }
            break;
        }
        case Op::Atomic: {
            const uint32_t at = emit(I::Atomic);
            prog.code[at].x = pc();
            seq(*n.sub, flags);
            emit(I::Match);
            prog.code[at].y = pc();
            break;
        }
        case Op::MaxRepeat:
        case Op::MinRepeat:
        case Op::PossRepeat: {
            if (flags & kTemplate)
                raise("re.error", std::string("internal: unsupported template operator ") +
                                      (n.op == Op::MaxRepeat ? "MAX_REPEAT"
                                                             : n.op == Op::MinRepeat ? "MIN_REPEAT" : "POSSESSIVE_REPEAT"));
            Inst rep;
            rep.min = n.a;
            rep.max = n.b;
            if (simple(*n.sub)) {
                rep.op = I::Rep1;
                rep.mode = n.op == Op::MaxRepeat ? 0 : n.op == Op::MinRepeat ? 1 : 2;
                const uint32_t at = emit(rep);
                prog.code[at].x = pc();
                unit(*n.sub, flags);
                prog.code[at].y = pc();
            } else if (n.op == Op::PossRepeat) {
                rep.op = I::Poss;
                const uint32_t at = emit(rep);
                prog.code[at].x = pc();
                seq(*n.sub, flags);
                emit(I::Match);
                prog.code[at].y = pc();
            } else {
                Inst init;
                init.op = I::RepInit;
                init.x = (uint32_t) prog.repeats++;
                const uint32_t at = emit(init);
                const uint32_t body = pc();
                seq(*n.sub, flags);
                rep.op = I::Until;
                rep.x = init.x;
                rep.y = body;
                rep.mode = n.op == Op::MaxRepeat ? 1 : 0;
                prog.code[at].y = emit(rep);
            }
            break;
        }
        case Op::GroupRef: {
            Inst g;
            g.op = I::GroupRef;
            g.x = n.a - 1;
            g.mode = !(flags & kIgnoreCase) ? 0 : (flags & kUnicode) ? 1 : 2;
            emit(g);
            break;
        }
        case Op::GroupRefExists: {
            Inst g;
            g.op = I::GroupExists;
            g.x = n.a - 1;
            const uint32_t at = emit(g);
            seq(*n.sub, flags);
            if (n.no) {
                const uint32_t j = emit(I::Jmp);
                prog.code[at].y = pc();
                seq(*n.no, flags);
                prog.code[j].x = pc();
            } else {
                prog.code[at].y = pc();
            }
            break;
        }
        case Op::Assert:
        case Op::AssertNot: {
            Inst as;
            as.op = I::Assert;
            as.mode = n.op == Op::AssertNot ? 1 : 0;
            if (n.a) {
                const auto w = width(*n.sub, st_);
                if (w.first > kMaxCode) raise("re.error", "looks too much behind");
                if (w.first != w.second) raise("re.error", "look-behind requires fixed-width pattern");
                as.back = w.first;
            }
            const uint32_t at = emit(as);
            prog.code[at].x = pc();
            seq(*n.sub, flags);
            emit(I::Match);
            prog.code[at].y = pc();
            break;
        }
        default: break;
    }
}

/// a backtracking machine: choice points and an undo log of the registers (group marks, repeat counts)
class Machine {
public:
    Machine(const Prog& p, const std::u32string& s) : p_(p), s_(s), n_((int64_t) s.size()) {}
    bool search();

private:
    enum : uint8_t { kAlt, kGreedy, kLazy, kUntilLazy };
    struct Choice {
        uint32_t pc;
        uint8_t kind;
        int64_t sp;
        size_t undo;
        int64_t a;       ///< Rep1: the count to try (greedy) or tried (lazy); Until: the count
        uint32_t inst;   ///< the Rep1 or Until
    };
    struct Undo {
        int64_t* slot;
        int64_t old;
    };
    bool run(uint32_t pc, int64_t sp, int64_t& end);
    bool sub(uint32_t pc, int64_t sp, int64_t& end) {   // an atomic part: its first match, or nothing changed
        const size_t base = stack_.size(), u = undo_.size();
        const bool ok = run(pc, sp, end);
        stack_.resize(base);
        if (!ok) undo_to(u);
        return ok;
    }
    bool backtrack(size_t base, uint32_t& pc, int64_t& sp);
    bool unit(const Inst& u, uint32_t ch) const {
        switch (u.op) {
            case I::Char: return fold(ch, u.mode) == u.x;
            case I::NotChar: return fold(ch, u.mode) != u.x;
            case I::Set: return p_.sets[u.x].match(ch);
            case I::Any: return ch != '\n';
            default: return true;
        }
    }
    bool at(uint32_t code, int64_t sp) const;
    void set(int64_t& slot, int64_t v) {
        undo_.push_back({&slot, slot});
        slot = v;
    }
    void undo_to(size_t n) {
        while (undo_.size() > n) {
            *undo_.back().slot = undo_.back().old;
            undo_.pop_back();
        }
    }
    void step() {
        if (++steps_ > kMaxSteps) raise("RuntimeError", "regular expression matching took too long");
    }
    uint32_t ch(int64_t k) const { return (uint32_t) s_[(size_t) k]; }

    const Prog& p_;
    const std::u32string& s_;
    int64_t n_;
    std::vector<int64_t> marks_, count_, last_;
    std::vector<Undo> undo_;
    std::vector<Choice> stack_;
    uint64_t steps_ = 0;
};

bool Machine::at(uint32_t code, int64_t sp) const {
    switch (code) {
        case AtBeginning:
        case AtBeginningString: return sp == 0;
        case AtBeginningLine: return sp == 0 || ch(sp - 1) == '\n';
        case AtEnd: return (n_ - sp == 1 && ch(sp) == '\n') || sp == n_;
        case AtEndLine: return sp == n_ || ch(sp) == '\n';
        case AtEndString: return sp == n_;
        default: {
            if (n_ == 0) return false;
            const bool uni = code == AtUniBoundary || code == AtUniNonBoundary;
            const bool that = sp > 0 && category(CatWord, uni, ch(sp - 1));
            const bool here = sp < n_ && category(CatWord, uni, ch(sp));
            return code == AtBoundary || code == AtUniBoundary ? here != that : here == that;
        }
    }
}

bool Machine::backtrack(size_t base, uint32_t& pc, int64_t& sp) {
    while (stack_.size() > base) {
        step();
        const Choice c = stack_.back();
        stack_.pop_back();
        undo_to(c.undo);
        switch (c.kind) {
            case kAlt:
                pc = c.pc;
                sp = c.sp;
                return true;
            case kGreedy: {   // one character fewer
                if (c.a > (int64_t) p_.code[c.inst].min) stack_.push_back({c.pc, kGreedy, c.sp, c.undo, c.a - 1, c.inst});
                pc = c.pc;
                sp = c.sp + c.a;
                return true;
            }
            case kLazy: {     // one character more
                const Inst& r = p_.code[c.inst];
                if (r.max != kMaxRepeat && c.a >= (int64_t) r.max) continue;
                const int64_t k = c.sp + c.a;
                if (k >= n_ || !unit(p_.code[r.x], ch(k))) continue;
                stack_.push_back({c.pc, kLazy, c.sp, c.undo, c.a + 1, c.inst});
                pc = c.pc;
                sp = k + 1;
                return true;
            }
            default: {        // a lazy repeat's next iteration
                const Inst& u = p_.code[c.inst];
                set(count_[u.x], c.a);
                set(last_[u.x], c.sp);
                pc = u.y;
                sp = c.sp;
                return true;
            }
        }
    }
    return false;
}

bool Machine::run(uint32_t pc, int64_t sp, int64_t& end) {
    const size_t base = stack_.size();
    for (;;) {
        step();
        const Inst& in = p_.code[pc];
        switch (in.op) {
            case I::Char:
            case I::NotChar:
            case I::Set:
            case I::Any:
            case I::AnyAll:
                if (sp < n_ && unit(in, ch(sp))) {
                    ++sp;
                    ++pc;
                    continue;
                }
                break;
            case I::At:
                if (at(in.x, sp)) {
                    ++pc;
                    continue;
                }
                break;
            case I::Split:
                stack_.push_back({in.y, kAlt, sp, undo_.size(), 0, 0});
                pc = in.x;
                continue;
            case I::Jmp: pc = in.x; continue;
            case I::Mark:
                set(marks_[in.x], sp);
                ++pc;
                continue;
            case I::Rep1: {
                const Inst& u = p_.code[in.x];
                const int64_t mn = in.min, mx = in.max == kMaxRepeat ? INT64_MAX : (int64_t) in.max;
                if (mn > n_ - sp) break;
                int64_t c = 0;
                if (in.mode == 1) {
                    while (c < mn && unit(u, ch(sp + c))) ++c;
                    if (c < mn) break;
                    stack_.push_back({in.y, kLazy, sp, undo_.size(), c, pc});
                } else {
                    while (c < mx && sp + c < n_ && unit(u, ch(sp + c))) ++c;
                    if (c < mn) break;
                    if (in.mode == 0 && c > mn) stack_.push_back({in.y, kGreedy, sp, undo_.size(), c - 1, pc});
                }
                sp += c;
                pc = in.y;
                continue;
            }
            case I::RepInit:
                set(count_[in.x], -1);
                set(last_[in.x], -1);
                pc = in.y;
                continue;
            case I::Until: {
                const int64_t count = count_[in.x] + 1;
                if (count < (int64_t) in.min) {
                    set(count_[in.x], count);
                    pc = in.y;
                    continue;
                }
                const bool more = (in.max == kMaxRepeat || count < (int64_t) in.max) && sp != last_[in.x];
                if (in.mode == 1) {   // greedy: another iteration first, then the tail
                    if (more) {
                        stack_.push_back({pc + 1, kAlt, sp, undo_.size(), 0, 0});
                        set(count_[in.x], count);
                        set(last_[in.x], sp);
                        pc = in.y;
                        continue;
                    }
                } else if (more) {
                    stack_.push_back({0, kUntilLazy, sp, undo_.size(), count, pc});
                }
                ++pc;
                continue;
            }
            case I::Poss: {   // each iteration atomic, as many as match
                int64_t count = 0, e = 0;
                bool failed = false;
                while (count < (int64_t) in.min) {
                    if (!sub(in.x, sp, e)) {
                        failed = true;
                        break;
                    }
                    sp = e;
                    ++count;
                }
                if (failed) break;
                int64_t last = -1;
                while ((in.max == kMaxRepeat || count < (int64_t) in.max) && sp != last) {
                    last = sp;
                    if (!sub(in.x, sp, e)) break;
                    sp = e;
                    ++count;
                }
                pc = in.y;
                continue;
            }
            case I::Atomic: {
                int64_t e = 0;
                if (!sub(in.x, sp, e)) break;
                sp = e;
                pc = in.y;
                continue;
            }
            case I::Assert: {
                int64_t e = 0;
                const bool room = sp >= (int64_t) in.back;
                if (in.mode == 0) {
                    if (!room || !sub(in.x, sp - (int64_t) in.back, e)) break;
                } else if (room) {
                    const size_t u = undo_.size();
                    if (sub(in.x, sp - (int64_t) in.back, e)) {
                        undo_to(u);
                        break;
                    }
                }
                pc = in.y;
                continue;
            }
            case I::GroupRef: {
                const int64_t a = marks_[2 * (size_t) in.x], b = marks_[2 * (size_t) in.x + 1];
                if (a < 0 || b < 0 || b < a || sp + (b - a) > n_) break;
                bool same = true;
                for (int64_t k = 0; same && k < b - a; ++k) same = fold(ch(sp + k), in.mode) == fold(ch(a + k), in.mode);
                if (!same) break;
                sp += b - a;
                ++pc;
                continue;
            }
            case I::GroupExists: {
                const int64_t a = marks_[2 * (size_t) in.x], b = marks_[2 * (size_t) in.x + 1];
                pc = a < 0 || b < 0 || b < a ? in.y : pc + 1;
                continue;
            }
            case I::Match: end = sp; return true;
        }
        if (!backtrack(base, pc, sp)) return false;
    }
}

bool Machine::search() {
    const bool anchored = !p_.code.empty() && p_.code[0].op == I::At &&
                          (p_.code[0].x == AtBeginning || p_.code[0].x == AtBeginningString);
    for (int64_t start = 0; start <= n_; ++start) {
        marks_.assign((size_t) p_.groups * 2, -1);
        count_.assign((size_t) p_.repeats, -1);
        last_.assign((size_t) p_.repeats, -1);
        undo_.clear();
        stack_.clear();
        int64_t end = 0;
        if (run(0, start, end)) return true;
        if (anchored) break;
    }
    return false;
}

/// re.compile(pattern): re.error, or ValueError / OverflowError / RecursionError where Python raises those
std::shared_ptr<const Prog> compile(const std::string& pattern) {
    Parser ps(code_points(pattern));
    const SeqP p = ps.parse();
    Compiler cc(ps.state);
    cc.seq(*p, ps.state.flags);
    cc.emit(I::Match);
    cc.prog.groups = ps.state.groups() - 1;
    return std::make_shared<const Prog>(std::move(cc.prog));
}

bool search(const Prog& prog, std::string_view subject) {
    const std::u32string s = code_points(subject);
    Machine m(prog, s);
    return m.search();
}

}  // namespace re

// ============================================================================================ urllib.parse
namespace url {

constexpr std::string_view kUsesRelative[] = {"",     "ftp",  "http",     "gopher", "nntp",  "imap",  "wais",
                                              "file", "https", "shttp",   "mms",    "prospero", "rtsp", "rtsps",
                                              "rtspu", "sftp", "svn",     "svn+ssh", "ws",   "wss"};
constexpr std::string_view kUsesNetloc[] = {"",      "ftp",     "http",  "gopher", "nntp",  "telnet",
                                            "imap",  "wais",    "file",  "mms",    "https", "shttp",
                                            "snews", "prospero", "rtsp", "rtsps",  "rtspu", "rsync",
                                            "svn",   "svn+ssh", "sftp",  "nfs",    "git",   "git+ssh",
                                            "ws",    "wss",     "itms-services"};
constexpr std::string_view kUsesParams[] = {"",      "ftp",  "hdl", "prospero", "http", "imap", "https", "shttp",
                                            "rtsp",  "rtsps", "rtspu", "sip",   "sips", "mms",  "sftp",  "tel"};

template <size_t N>
bool in(std::string_view s, const std::string_view (&list)[N]) {
    return std::find(std::begin(list), std::end(list), s) != std::end(list);
}

struct Split {
    std::string scheme, netloc, path, query, fragment;
};
struct Parse {
    std::string scheme, netloc, path, params, query, fragment;
};

bool scheme_char(char c) {
    return is_ascii_letter((unsigned char) c) || is_digit((unsigned char) c) || c == '+' || c == '-' || c == '.';
}

Split urlsplit(std::string url, std::string scheme = "") {
    size_t k = 0;
    while (k < url.size() && (unsigned char) url[k] <= 0x20) ++k;
    url.erase(0, k);
    for (const char bad : {'\t', '\r', '\n'}) url.erase(std::remove(url.begin(), url.end(), bad), url.end());
    Split out;
    const size_t i = url.find(':');
    if (i != npos && i > 0 && is_ascii_letter((unsigned char) url[0]) &&
        std::all_of(url.begin(), url.begin() + (std::ptrdiff_t) i, scheme_char)) {
        scheme = lower(url.substr(0, i));
        url.erase(0, i + 1);
    }
    out.scheme = scheme;
    if (url.compare(0, 2, "//") == 0) {
        size_t delim = url.size();
        for (const char c : {'/', '?', '#'}) {
            const size_t w = url.find(c, 2);
            if (w != npos) delim = std::min(delim, w);
        }
        out.netloc = url.substr(2, delim - 2);
        url.erase(0, delim);
        if ((out.netloc.find('[') != npos) != (out.netloc.find(']') != npos)) raise("ValueError", "Invalid IPv6 URL");
    }
    if (const size_t h = url.find('#'); h != npos) {
        out.fragment = url.substr(h + 1);
        url.resize(h);
    }
    if (const size_t q = url.find('?'); q != npos) {
        out.query = url.substr(q + 1);
        url.resize(q);
    }
    out.path = url;
    return out;
}

std::string urlunsplit(const Split& p) {
    std::string url = p.path;
    if (!p.netloc.empty()) {
        if (!url.empty() && url[0] != '/') url = "/" + url;
        url = "//" + p.netloc + url;
    } else if (url.compare(0, 2, "//") == 0) {
        url = "//" + url;
    } else if (!p.scheme.empty() && in(p.scheme, kUsesNetloc) && (url.empty() || url[0] == '/')) {
        url = "//" + url;
    }
    if (!p.scheme.empty()) url = p.scheme + ":" + url;
    if (!p.query.empty()) url += "?" + p.query;
    if (!p.fragment.empty()) url += "#" + p.fragment;
    return url;
}

Parse urlparse(const std::string& url, const std::string& scheme = "") {
    Split s = urlsplit(url, scheme);
    Parse out{s.scheme, s.netloc, s.path, "", s.query, s.fragment};
    if (in(s.scheme, kUsesParams) && s.path.find(';') != npos) {
        const size_t i = s.path.find('/') != npos ? s.path.find(';', s.path.rfind('/')) : s.path.find(';');
        if (i != npos) {
            out.params = s.path.substr(i + 1);
            out.path = s.path.substr(0, i);
        }
    }
    return out;
}

std::string urlunparse(const Parse& p) {
    return urlunsplit({p.scheme, p.netloc, p.params.empty() ? p.path : p.path + ";" + p.params, p.query, p.fragment});
}

std::string urljoin(const std::string& base, const std::string& url) {
    if (base.empty()) return url;
    if (url.empty()) return base;
    const Parse b = urlparse(base);
    Parse u = urlparse(url, b.scheme);
    if (u.scheme != b.scheme || !in(u.scheme, kUsesRelative)) return url;
    if (in(u.scheme, kUsesNetloc)) {
        if (!u.netloc.empty()) return urlunparse(u);
        u.netloc = b.netloc;
    }
    if (u.path.empty() && u.params.empty()) {
        u.path = b.path;
        u.params = b.params;
        if (u.query.empty()) u.query = b.query;
        return urlunparse(u);
    }
    std::vector<std::string> base_parts = split(b.path, "/");
    if (!base_parts.back().empty()) base_parts.pop_back();
    std::vector<std::string> segments;
    if (u.path[0] == '/') {
        segments = split(u.path, "/");
    } else {
        segments = base_parts;
        for (std::string& x : split(u.path, "/")) segments.push_back(std::move(x));
        if (segments.size() > 2) {   // no empty segments but the first and the last
            std::vector<std::string> kept{segments.front()};
            for (size_t k = 1; k + 1 < segments.size(); ++k)
                if (!segments[k].empty()) kept.push_back(segments[k]);
            kept.push_back(segments.back());
            segments = std::move(kept);
        }
    }
    std::vector<std::string> resolved;
    for (const std::string& seg : segments) {
        if (seg == "..") {
            if (!resolved.empty()) resolved.pop_back();
        } else if (seg != ".") {
            resolved.push_back(seg);
        }
    }
    if (segments.back() == "." || segments.back() == "..") resolved.push_back("");
    u.path = joined(resolved, "/");
    if (u.path.empty()) u.path = "/";
    return urlunparse(u);
}

std::pair<std::string, std::string> urldefrag(const std::string& url) {
    if (url.find('#') == npos) return {url, ""};
    Parse p = urlparse(url);
    std::string frag = std::move(p.fragment);
    p.fragment.clear();
    return {urlunparse(p), frag};
}

/// urllib.parse.unquote: %XX as bytes, decoded as UTF-8 (an invalid byte as U+FFFD)
std::string unquote(const std::string& s) {
    if (s.find('%') == npos) return s;
    std::string bytes;
    for (size_t k = 0; k < s.size(); ++k) {
        if (s[k] == '%' && k + 2 < s.size() + 0 && hex_value((unsigned char) s[k + 1]) >= 0 &&
            hex_value((unsigned char) s[k + 2]) >= 0) {
            bytes += (char) (hex_value((unsigned char) s[k + 1]) * 16 + hex_value((unsigned char) s[k + 2]));
            k += 2;
        } else {
            bytes += s[k];
        }
    }
    return sanitized(bytes);
}

}  // namespace url

// ============================================================================================ referencing
enum class Draft : uint8_t { D3, D4, D6, D7, D2019, D2020 };

/// the dialect ids referencing knows (Specification lookup by $schema, without a trailing "#")
constexpr std::string_view kSpecIds[] = {
    "http://json-schema.org/draft-03/schema",       "http://json-schema.org/draft-04/schema",
    "http://json-schema.org/draft-06/schema",       "http://json-schema.org/draft-07/schema",
    "https://json-schema.org/draft/2019-09/schema", "https://json-schema.org/draft/2020-12/schema"};

std::string rstrip_hash(std::string s) {
    while (!s.empty() && s.back() == '#') s.pop_back();
    return s;
}

/// Specification.detect with a default: the specification a subschema's $schema names
Draft spec_of(const Val& contents, Draft dflt) {
    if (contents.k != K::Dict) return dflt;
    const Val* v = contents.get("$schema");
    if (v == nullptr || v->k == K::None) return dflt;
    if (v->k != K::Str) no_attribute(*v, "rstrip");
    const std::string id = rstrip_hash(v->s);
    for (int d = 0; d < 6; ++d)
        if (id == kSpecIds[d]) return (Draft) d;
    return dflt;
}

/// `key in contents`
bool contains_key(const Val& contents, const char* key) {
    switch (contents.k) {
        case K::Dict: return contents.has(key);
        case K::Str: return contents.s.find(key) != npos;
        case K::List:
            return std::any_of(contents.list.begin(), contents.list.end(),
                               [&](const Val& e) { return e.k == K::Str && e.s == key; });
        default: raise("TypeError", std::string("argument of type '") + type_name(contents) + "' is not iterable");
    }
}

/// the specification's id_of (nullptr: None)
const Val* id_of(const Val& c, Draft d) {
    if (d >= Draft::D2019) {
        if (c.k == K::Bool) return nullptr;
        if (c.k != K::Dict) no_attribute(c, "get");
        const Val* id = c.get("$id");
        return id != nullptr && id->k != K::None ? id : nullptr;
    }
    const bool dollar = d >= Draft::D6;
    if (dollar && c.k == K::Bool) return nullptr;
    if (contains_key(c, "$ref")) return nullptr;
    if (c.k != K::Dict) no_attribute(c, "get");
    const Val* id = c.get(dollar ? "$id" : "id");
    if (id == nullptr || id->k == K::None) return nullptr;
    if (id->k != K::Str) no_attribute(*id, "startswith");
    return starts_with(id->s, "#") ? nullptr : id;
}

/// Resource.id()
std::optional<std::string> resource_id(const Val& c, Draft d) {
    const Val* id = id_of(c, d);
    if (id == nullptr) return std::nullopt;
    if (id->k != K::Str) no_attribute(*id, "rstrip");
    return rstrip_hash(id->s);
}

/// the specification's anchors_in: (name, dynamic)
std::vector<std::pair<std::string, bool>> anchors_in(const Val& c, Draft d) {
    std::vector<std::pair<std::string, bool>> out;
    if (d >= Draft::D2019) {
        if (c.k == K::Bool) return out;
        if (c.k != K::Dict) no_attribute(c, "get");
        if (const Val* a = c.get("$anchor"); a != nullptr && a->k == K::Str) out.push_back({a->s, false});
        if (d == Draft::D2020)
            if (const Val* a = c.get("$dynamicAnchor"); a != nullptr && a->k == K::Str) out.push_back({a->s, true});
        return out;
    }
    if (d >= Draft::D6 && c.k == K::Bool) return out;
    if (c.k != K::Dict) no_attribute(c, "get");
    const Val* id = c.get(d >= Draft::D6 ? "$id" : "id");
    if (id == nullptr) return out;
    if (id->k != K::Str) no_attribute(*id, "startswith");
    if (starts_with(id->s, "#")) out.push_back({id->s.substr(1), false});
    return out;
}

struct SpecSets {
    std::vector<std::string_view> value, subarray, subvalues;   ///< subresources_of's keywords
    std::vector<std::string_view> in_value, in_child;           ///< maybe_in_subresource's
};

const SpecSets& spec_sets(Draft d) {
    static const SpecSets kSets[6] = {
        {{}, {"extends"}, {"definitions", "patternProperties", "properties"},
         {"additionalItems", "additionalProperties"}, {"extends", "definitions", "patternProperties", "properties"}},
        {{"not"}, {"allOf", "anyOf", "oneOf"}, {"definitions", "patternProperties", "properties"},
         {"additionalItems", "additionalProperties", "not"},
         {"allOf", "anyOf", "oneOf", "definitions", "patternProperties", "properties"}},
        {{"additionalItems", "additionalProperties", "contains", "not", "propertyNames"},
         {"allOf", "anyOf", "oneOf"}, {"definitions", "patternProperties", "properties"},
         {"additionalItems", "additionalProperties", "contains", "not", "propertyNames"},
         {"allOf", "anyOf", "oneOf", "definitions", "patternProperties", "properties"}},
        {{"additionalItems", "additionalProperties", "contains", "else", "if", "not", "propertyNames", "then"},
         {"allOf", "anyOf", "oneOf"}, {"definitions", "patternProperties", "properties"},
         {"additionalItems", "additionalProperties", "contains", "else", "if", "not", "propertyNames", "then"},
         {"allOf", "anyOf", "oneOf", "definitions", "patternProperties", "properties"}},
        {{"additionalItems", "additionalProperties", "contains", "contentSchema", "else", "if", "not",
          "propertyNames", "then", "unevaluatedItems", "unevaluatedProperties"},
         {"allOf", "anyOf", "oneOf"},
         {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"},
         {"additionalItems", "additionalProperties", "contains", "contentSchema", "else", "if", "not",
          "propertyNames", "then", "unevaluatedItems", "unevaluatedProperties"},
         {"allOf", "anyOf", "oneOf", "$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}},
        {{"additionalProperties", "contains", "contentSchema", "else", "if", "items", "not", "propertyNames", "then",
          "unevaluatedItems", "unevaluatedProperties"},
         {"allOf", "anyOf", "oneOf", "prefixItems"},
         {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"},
         {"additionalProperties", "contains", "contentSchema", "else", "if", "items", "not", "propertyNames", "then",
          "unevaluatedItems", "unevaluatedProperties"},
         {"allOf", "anyOf", "oneOf", "prefixItems", "$defs", "definitions", "dependentSchemas", "patternProperties",
          "properties"}}};
    return kSets[(int) d];
}

bool listed(const std::vector<std::string_view>& keys, std::string_view key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

/// the specification's subresources_of
std::vector<const Val*> subresources_of(const Val& c, Draft d) {
    std::vector<const Val*> out;
    if (c.k == K::Bool && d >= Draft::D6) return out;
    if (c.k != K::Dict) {
        if (d <= Draft::D4) contains_key(c, "items");   // `each in contents` raises for what is not iterable
        return out;
    }
    const SpecSets& s = spec_sets(d);
    for (const std::string_view key : s.value)
        if (const Val* v = c.get(key)) out.push_back(v);
    for (const std::string_view key : s.subarray)
        if (const Val* v = c.get(key); v != nullptr && v->k == K::List)
            for (const Val& e : v->list) out.push_back(&e);
    for (const std::string_view key : s.subvalues) {
        if (const Val* v = c.get(key)) {
            if (v->k != K::Dict) no_attribute(*v, "values");
            for (const Member& m : v->dict) out.push_back(&m.val);
        }
    }
    if (d <= Draft::D2019) {   // the crazy items
        if (const Val* items = c.get("items"); items != nullptr && items->k != K::None) {
            if (items->k == K::List)
                for (const Val& e : items->list) out.push_back(&e);
            else out.push_back(items);
        }
    }
    if (d <= Draft::D7) {      // dependencies, when the first is a schema
        if (const Val* deps = c.get("dependencies"); deps != nullptr && deps->k != K::None) {
            if (deps->k != K::Dict) no_attribute(*deps, "values");
            if (!deps->dict.empty() && deps->dict[0].val.k == K::Dict)
                for (const Member& m : deps->dict) out.push_back(&m.val);
        }
    }
    if (d <= Draft::D4)
        for (const char* key : {"additionalItems", "additionalProperties"})
            if (const Val* v = c.get(key); v != nullptr && v->k == K::Dict) out.push_back(v);
    return out;
}

struct Resource {
    const Val* contents = nullptr;
    Draft spec = Draft::D2020;
};
struct Anchor {
    Resource res;
    bool dynamic = false;
};
using Resources = std::map<std::string, Resource>;
using Anchors = std::map<std::pair<std::string, std::string>, Anchor>;

/// Registry.crawl: each resource under its id, the anchors under their resource's
void crawl(std::vector<std::pair<std::string, Resource>>& todo, Resources& resources, Anchors& anchors) {
    while (!todo.empty()) {
        auto [uri, res] = todo.back();
        todo.pop_back();
        if (const std::optional<std::string> id = resource_id(*res.contents, res.spec)) {
            uri = url::urljoin(uri, *id);
            resources[uri] = res;
        }
        for (const auto& [name, dynamic] : anchors_in(*res.contents, res.spec)) anchors[{uri, name}] = Anchor{res, dynamic};
        for (const Val* sub : subresources_of(*res.contents, res.spec))
            todo.push_back({uri, Resource{sub, spec_of(*sub, res.spec)}});
    }
}

// ============================================================================================ jsonschema
struct Run;
struct Ctx;
struct Sink;
using Keyword = bool (*)(Run&, const Ctx&, const Val& value, const Val& inst, const Val& schema, Sink&);

/// a jsonschema validator class: its draft, metaschema and keywords
struct Dialect {
    Draft draft = Draft::D2020;
    std::string id;                                         ///< the metaschema's URI, as validator_for normalizes it
    std::map<std::string, Keyword, std::less<>> keywords;   ///< VALIDATORS
    const Val* meta = nullptr;                              ///< META_SCHEMA
};

/// what jsonschema-specifications holds: the metaschemas and vocabularies, crawled, and the six validator classes
struct Meta {
    std::deque<Val> docs;
    Resources resources;
    Anchors anchors;
    Dialect dialects[6];
};
const Meta& meta();

/// a validator's registry: the metaschemas', and its schema's resources and anchors (crawled when first needed)
class Registry {
public:
    explicit Registry(const Meta& m) : m_(m) {}
    void add_root(const std::string& uri, Resource r) {
        own_[uri] = r;
        todo_.push_back({uri, r});
    }
    /// get_or_retrieve: nullptr for NoSuchResource
    const Resource* resource(const std::string& uri) {
        if (const Resource* r = find(uri)) return r;
        crawl_once();
        return find(uri);
    }
    /// anchor(): NoSuchAnchor or InvalidAnchor when there is none
    const Anchor& anchor(const std::string& uri, const std::string& name) {
        if (const Anchor* a = find_anchor(uri, name)) return *a;
        crawl_once();
        if (const Anchor* a = find_anchor(uri, name)) return *a;
        const Resource* r = find(uri);
        if (r == nullptr) raise("KeyError", py_repr_str(uri));
        if (const std::optional<std::string> canonical = resource_id(*r->contents, r->spec))
            if (const Anchor* a = find_anchor(*canonical, name)) return *a;
        if (name.find('/') != npos)
            raise("InvalidAnchor", "'#" + name + "' is not a valid anchor, neither as a plain name anchor nor as a "
                                   "JSON Pointer. You may have intended to use '#/" + name +
                                   "', as the slash is required *before each segment* of a JSON pointer.");
        raise("NoSuchAnchor", py_repr_str(name) + " does not exist within " + repr(*r->contents));
    }
    /// anchor() for DynamicAnchor.resolve, where NoSuchAnchor is no anchor
    const Anchor* try_anchor(const std::string& uri, const std::string& name) {
        try {
            return &anchor(uri, name);
        } catch (const PyError& e) {
            if (e.type == "NoSuchAnchor") return nullptr;
            throw;
        }
    }

private:
    const Resource* find(const std::string& uri) const {
        if (const auto it = own_.find(uri); it != own_.end()) return &it->second;
        const auto it = m_.resources.find(uri);
        return it == m_.resources.end() ? nullptr : &it->second;
    }
    const Anchor* find_anchor(const std::string& uri, const std::string& name) const {
        const std::pair<std::string, std::string> key{uri, name};
        if (const auto it = own_anchors_.find(key); it != own_anchors_.end()) return &it->second;
        const auto it = m_.anchors.find(key);
        return it == m_.anchors.end() ? nullptr : &it->second;
    }
    void crawl_once() { crawl(todo_, own_, own_anchors_); }

    const Meta& m_;
    Resources own_;
    Anchors own_anchors_;
    std::vector<std::pair<std::string, Resource>> todo_;
};

struct Scope {   ///< one base URI of the dynamic scope (Resolver._previous)
    const std::string* uri;
    const Scope* next;
};
struct Resolver {
    const std::string* base = nullptr;
    const Scope* prev = nullptr;
};
struct Resolved {
    const Val* contents = nullptr;
    Resolver resolver;
};

/// Python's recursion limit (1000 frames), less the server's frames below validated_json
constexpr int kMaxFrames = 981;
/// frames in all for one run (check_schema, an answer's iter_errors): Python's has no limit, and allOf / oneOf chains
/// whose two branches name the same next one double per level; a large schema's check takes a few million
constexpr uint64_t kMaxRunFrames = 5000000;

/// compiled patterns, and the errors of those that do not compile (raised where Python would compile them)
struct Patterns {
    std::unordered_map<std::string, std::shared_ptr<const re::Prog>> progs;
    std::unordered_map<std::string, PyError> errors;

    const re::Prog& get(const std::string& pattern) {
        if (const auto it = progs.find(pattern); it != progs.end()) return *it->second;
        if (const auto it = errors.find(pattern); it != errors.end()) throw it->second;
        try {
            return *progs.emplace(pattern, re::compile(pattern)).first->second;
        } catch (const PyError& e) {
            errors.emplace(pattern, e);
            throw;
        }
    }
    /// every pattern a schema holds (pattern, patternProperties' keys and their "|" join), compiled now, near the
    /// bottom of the stack: deep in validation the stack is the validator's
    void add_from(const Val& root) {
        std::vector<const Val*> todo{&root};
        auto add = [this](const std::string& p) {
            try {
                get(p);
            } catch (const PyError&) {
            }
        };
        while (!todo.empty()) {
            const Val* v = todo.back();
            todo.pop_back();
            if (v->k == K::List)
                for (const Val& e : v->list) todo.push_back(&e);
            if (v->k != K::Dict) continue;
            if (const Val* p = v->get("pattern"); p != nullptr && p->k == K::Str) add(p->s);
            if (const Val* pp = v->get("patternProperties"); pp != nullptr && pp->k == K::Dict && !pp->dict.empty()) {
                std::vector<std::string> keys;
                for (const Member& m : pp->dict) {
                    add(m.key);
                    keys.push_back(m.key);
                }
                add(joined(keys, "|"));
            }
            for (const Member& m : v->dict) todo.push_back(&m.val);
        }
    }
};

/// one iter_errors: the registry, the instance path, the Python frames in use
struct Run {
    Run(Registry& r, Patterns& p) : reg(r), patterns(p) {}
    Registry& reg;
    Patterns& patterns;
    bool formats = false;            ///< check_schema's format checker
    int frames = 0;
    uint64_t frames_all = 0;         ///< the run's frames so far (kMaxRunFrames)
    std::vector<std::string> path;   ///< where in the instance
    std::deque<std::string> uris;
    std::deque<Scope> scopes;
    std::deque<Val> temps;           ///< characters of a str a pointer went into
    const std::string* intern(std::string s) {
        uris.push_back(std::move(s));
        return &uris.back();
    }
    int frames_left() const { return kMaxFrames - frames; }
};

/// a Python frame (a generator of iter_errors, descend, a keyword, ...): RecursionError past the limit
struct PyFrame {
    explicit PyFrame(Run& run) : r(run) {
        if (++r.frames > kMaxFrames) {
            --r.frames;
            raise("RecursionError", "maximum recursion depth exceeded");
        }
        if (++r.frames_all > kMaxRunFrames) {   // a ValueError: the schema's 400, an answer's 502
            --r.frames;
            raise("ValueError", "the schema takes more than " + std::to_string(kMaxRunFrames) + " steps to check");
        }
    }
    ~PyFrame() { --r.frames; }
    PyFrame(const PyFrame&) = delete;
    PyFrame& operator=(const PyFrame&) = delete;
    Run& r;
};

/// Resolver._evolve: a new base, the old one entering the dynamic scope
Resolver evolve_base(Run& r, const Resolver& res, const std::string& base) {
    const Scope* prev = res.prev;
    if (!res.base->empty() && (prev == nullptr || base != *res.base)) {
        r.scopes.push_back({res.base, prev});
        prev = &r.scopes.back();
    }
    return Resolver{base == *res.base ? res.base : r.intern(base), prev};
}

/// Resolver.in_subresource: the subresource's id, if it has one, joined to the base
Resolver in_subresource(Run& r, const Resolver& res, const Val& contents, Draft spec, bool* changed = nullptr) {
    const std::optional<std::string> id = resource_id(contents, spec);
    if (changed != nullptr) *changed = id.has_value();
    if (!id) return res;
    return Resolver{r.intern(url::urljoin(*res.base, *id)), res.prev};
}

struct Segment {
    bool index = false;
    std::string key;
};

/// the specification's maybe_in_subresource: a pointer that went into a subschema enters its resource
Resolver maybe_in_subresource(Run& r, const std::vector<Segment>& segs, const Resolver& res, const Val& sub, Draft spec,
                              bool* changed) {
    *changed = false;
    const SpecSets& s = spec_sets(spec);
    size_t k = 0;
    while (k < segs.size()) {
        const Segment& g = segs[k++];
        if (spec != Draft::D2020 && !g.index && sub.k == K::Dict &&
            (g.key == "items" || (spec != Draft::D2019 && g.key == "dependencies")))
            return in_subresource(r, res, sub, spec, changed);
        if (!g.index && listed(s.in_value, g.key)) continue;
        if (g.index || !listed(s.in_child, g.key) || k >= segs.size()) return res;
        ++k;
    }
    return in_subresource(r, res, sub, spec, changed);
}

/// int(s) as Python parses a str: whitespace, a sign, Unicode digits, single underscores (nullopt: ValueError)
std::optional<int64_t> py_int(const std::string& s) {
    const std::u32string c = code_points(strip(s));
    size_t k = 0;
    bool neg = false;
    if (k < c.size() && (c[k] == '+' || c[k] == '-')) neg = c[k++] == '-';
    int64_t v = 0;
    bool digit_before = false;
    for (; k < c.size(); ++k) {
        if (c[k] == '_') {
            if (!digit_before) return std::nullopt;
            digit_before = false;
            continue;
        }
        const int d = py_decimal((uint32_t) c[k]);
        if (d < 0) return std::nullopt;
        v = v > (INT64_MAX - 9) / 10 ? INT64_MAX : v * 10 + d;
        digit_before = true;
    }
    if (!digit_before) return std::nullopt;
    return neg ? -v : v;
}

[[noreturn]] void nowhere(const std::string& ptr, const Resource& rs) {
    std::string msg = py_repr_str(ptr) + " does not exist within " + repr(*rs.contents);
    if (ptr == "/")
        msg += ". The pointer '/' is a valid JSON Pointer but it points to an empty string property ''. If you "
               "intended to point to the entire resource, you should use '#'.";
    raise("PointerToNowhere", std::move(msg));
}

/// Resource.pointer
Resolved pointer(Run& r, const Resource& rs, const std::string& ptr, Resolver res) {
    const Val* contents = rs.contents;
    std::vector<Segment> segs;
    for (const std::string& raw : split(url::unquote(ptr.substr(1)), "/")) {
        Segment seg;
        if (contents->k == K::List || contents->k == K::Str) {
            const std::optional<int64_t> idx = py_int(raw);
            if (!idx) raise("ValueError", "invalid literal for int() with base 10: " + py_repr_str(raw));
            const bool list = contents->k == K::List;
            const int64_t n = list ? (int64_t) contents->list.size() : (int64_t) utf8_len(contents->s);
            const int64_t k = *idx < 0 ? *idx + n : *idx;
            if (k < 0 || k >= n) nowhere(ptr, rs);
            if (list) {
                contents = &contents->list[(size_t) k];
            } else {
                r.temps.push_back(Val::str(utf8_of(std::u32string(1, code_points(contents->s)[(size_t) k]))));
                contents = &r.temps.back();
            }
            seg.index = true;
        } else {
            seg.key = replace_all(replace_all(raw, "~1", "/"), "~0", "~");
            if (contents->k != K::Dict)
                raise("TypeError", std::string("'") + type_name(*contents) + "' object is not subscriptable");
            const Val* v = contents->get(seg.key);
            if (v == nullptr) nowhere(ptr, rs);
            contents = v;
        }
        segs.push_back(std::move(seg));
        bool changed = false;
        res = maybe_in_subresource(r, segs, res, *contents, rs.spec, &changed);
        if (changed) segs.clear();
    }
    return Resolved{contents, res};
}

/// Resolver.lookup
Resolved lookup(Run& r, const Resolver& res, const std::string& ref) {
    std::string uri, fragment;
    if (starts_with(ref, "#")) {
        uri = *res.base;
        fragment = ref.substr(1);
    } else {
        std::tie(uri, fragment) = url::urldefrag(url::urljoin(*res.base, ref));
    }
    const Resource* rs = r.reg.resource(uri);
    if (rs == nullptr) raise("Unresolvable", ref);
    if (starts_with(fragment, "/")) return pointer(r, *rs, fragment, evolve_base(r, res, uri));
    if (!fragment.empty()) {
        const Anchor& a = r.reg.anchor(uri, fragment);
        const Resolver ev = evolve_base(r, res, uri);
        if (!a.dynamic) return Resolved{a.res.contents, ev};
        Resource last = a.res;   // DynamicAnchor.resolve: the outermost of its name in the dynamic scope
        for (const Scope* sc = ev.prev; sc != nullptr; sc = sc->next)
            if (const Anchor* b = r.reg.try_anchor(*sc->uri, fragment); b != nullptr && b->dynamic) last = b->res;
        return Resolved{last.contents, in_subresource(r, ev, *last.contents, last.spec)};
    }
    return Resolved{rs->contents, evolve_base(r, res, uri)};
}

/// a $ref's value
Resolved lookup(Run& r, const Resolver& res, const Val& ref) {
    if (ref.k != K::Str) no_attribute(ref, "startswith");
    return lookup(r, res, ref.s);
}

/// lookup_recursive_ref: "#", or the outermost of the scope's resources with $recursiveAnchor
Resolved recursive_ref(Run& r, const Resolver& res) {
    auto anchored = [](const Resolved& x) {
        const Val* a = x.contents->k == K::Dict ? x.contents->get("$recursiveAnchor") : nullptr;
        return a != nullptr && truthy(*a);
    };
    Resolved found = lookup(r, res, std::string("#"));
    if (anchored(found)) {
        for (const Scope* sc = res.prev; sc != nullptr; sc = sc->next) {
            Resolved next = lookup(r, res, *sc->uri);
            if (!anchored(next)) break;
            found = next;
        }
    }
    return found;
}

// ---- iteration
/// a validator: its class and resolver
struct Ctx {
    const Dialect* d;
    Resolver res;
};

/// how errors are consumed: next(...) (the first) or list(...) (all of them)
struct Sink {
    bool all = false;
    bool messages = false;   ///< the first error's message and path are wanted
    size_t count = 0;
    std::string message;
    std::vector<std::string> path;
};

/// an error yielded; false when the consumer stops
template <class F>
bool emit(Run& r, Sink& s, F&& message) {
    if (s.count++ == 0 && s.messages) {
        s.message = message();
        s.path = r.path;
    }
    return s.all;
}

struct PathPush {
    PathPush(Run& run, const std::string* item) : r(run), on(item != nullptr) {
        if (on) r.path.push_back(*item);
    }
    ~PathPush() {
        if (on) r.path.pop_back();
    }
    PathPush(const PathPush&) = delete;
    PathPush& operator=(const PathPush&) = delete;
    Run& r;
    bool on;
};

/// _META_SCHEMAS.get($schema, default): the URI normalized as urlsplit(...).geturl() normalizes it
const Dialect* dialect_named(const Val& v, const Dialect* dflt) {
    if (v.k == K::List || v.k == K::Dict) raise("TypeError", std::string("unhashable type: '") + type_name(v) + "'");
    if (v.k != K::Str) {
        if (!truthy(v)) return dflt;
        no_attribute(v, "decode");
    }
    const std::string id = url::urlunsplit(url::urlsplit(v.s));
    for (const Dialect& d : meta().dialects)
        if (d.id == id) return &d;
    return dflt;
}

/// validator_for(schema, default)
const Dialect* dialect_for(const Val& schema, const Dialect* dflt) {
    switch (schema.k) {
        case K::Bool: return dflt;
        case K::Dict: {
            const Val* v = schema.get("$schema");
            return v != nullptr ? dialect_named(*v, dflt) : dflt;
        }
        case K::Str:
            if (schema.s.find("$schema") != npos) raise("TypeError", "string indices must be integers, not 'str'");
            return dflt;
        case K::List:
            for (const Val& e : schema.list)
                if (e.k == K::Str && e.s == "$schema") raise("TypeError", "list indices must be integers or slices, not str");
            return dflt;
        default: raise("TypeError", std::string("argument of type '") + type_name(schema) + "' is not iterable");
    }
}

/// validator.evolve(schema=...): the class its $schema names, the keywords listed as __attrs_post_init__ lists them
Ctx evolve(const Ctx& c, const Val& schema, const Resolver* res = nullptr) {
    const Dialect* d = dialect_for(schema, c.d);
    if (schema.k != K::Dict && schema.k != K::Bool) no_attribute(schema, d->draft <= Draft::D7 ? "get" : "items");
    return Ctx{d, res != nullptr ? *res : c.res};
}

/// the keywords of a schema in order (applicable_validators of `applicable`'s class, VALIDATORS of c's)
bool keywords(Run& r, const Ctx& c, const Dialect& applicable, const Val& inst, const Val& schema, Sink& s) {
    if (applicable.draft <= Draft::D7) {   // ignore_ref_siblings
        if (const Val* ref = schema.get("$ref"); ref != nullptr && ref->k != K::None) {
            const auto it = c.d->keywords.find("$ref");
            if (it == c.d->keywords.end()) return true;
            const PyFrame f(r);
            return it->second(r, c, *ref, inst, schema, s);
        }
    }
    for (const Member& m : schema.dict) {
        const auto it = c.d->keywords.find(m.key);
        if (it == c.d->keywords.end()) continue;
        const PyFrame f(r);
        if (!it->second(r, c, m.val, inst, schema, s)) return false;
    }
    return true;
}

/// iter_errors
bool iter_errors(Run& r, const Ctx& c, const Val& inst, const Val& schema, Sink& s) {
    const PyFrame f(r);
    if (schema.k == K::Bool) return schema.b || emit(r, s, [&] { return "False schema does not allow " + repr(inst); });
    return keywords(r, c, *c.d, inst, schema, s);
}

/// descend: a subschema (its errors under `path`, but a False schema's), in its resource unless `res` is given
bool descend(Run& r, const Ctx& c, const Val& inst, const Val& schema, Sink& s, const std::string* path = nullptr,
             const Resolver* res = nullptr) {
    const PyFrame f(r);
    if (schema.k == K::Bool) return schema.b || emit(r, s, [&] { return "False schema does not allow " + repr(inst); });
    const Resolver sub = res != nullptr ? *res : in_subresource(r, c.res, schema, c.d->draft);
    const Ctx ev = evolve(c, schema, &sub);
    const PathPush p(r, path);
    return keywords(r, ev, *c.d, inst, schema, s);
}

/// iter_errors of an evolved validator: no error
bool valid_with(Run& r, const Ctx& ev, const Val& inst, const Val& schema) {
    const PyFrame f(r);
    Sink s;
    iter_errors(r, ev, inst, schema, s);
    return s.count == 0;
}

/// validator.evolve(schema=schema).is_valid(instance)
bool is_valid(Run& r, const Ctx& c, const Val& inst, const Val& schema) {
    return valid_with(r, evolve(c, schema), inst, schema);
}

/// next(validator.descend(instance, schema), None) is None
bool descend_valid(Run& r, const Ctx& c, const Val& inst, const Val& schema) {
    Sink s;
    descend(r, c, inst, schema, s);
    return s.count == 0;
}

// ---- helpers of the keywords
/// `for x in v`: a list's items, a dict's keys, a str's characters; fn returns false to stop
template <class F>
bool each_item(const Val& v, F&& fn) {
    switch (v.k) {
        case K::List:
            for (const Val& e : v.list)
                if (!fn(e)) return false;
            return true;
        case K::Dict:
            for (const Member& m : v.dict) {
                const Val key = Val::str(m.key);
                if (!fn(key)) return false;
            }
            return true;
        case K::Str:
            for (const char32_t ch : code_points(v.s)) {
                const Val one = Val::str(utf8_of(std::u32string(1, ch)));
                if (!fn(one)) return false;
            }
            return true;
        default: raise("TypeError", std::string("'") + type_name(v) + "' object is not iterable");
    }
}

/// `needle in container`
bool py_in(const Val& needle, const Val& container) {
    switch (container.k) {
        case K::Dict:
            if (needle.k == K::List || needle.k == K::Dict)
                raise("TypeError", std::string("unhashable type: '") + type_name(needle) + "'");
            return needle.k == K::Str && container.has(needle.s);
        case K::List:
            return std::any_of(container.list.begin(), container.list.end(), [&](const Val& e) { return py_eq(e, needle); });
        case K::Str:
            if (needle.k != K::Str)
                raise("TypeError", std::string("'in <string>' requires string as left operand, not ") + type_name(needle));
            return container.s.find(needle.s) != npos;
        default: raise("TypeError", std::string("argument of type '") + type_name(container) + "' is not iterable");
    }
}

/// len(x)
int64_t py_len(const Val& v) {
    switch (v.k) {
        case K::Str: return (int64_t) utf8_len(v.s);
        case K::List: return (int64_t) v.list.size();
        case K::Dict: return (int64_t) v.dict.size();
        default: raise("TypeError", std::string("object of type '") + type_name(v) + "' has no len()");
    }
}

/// the type checker's is_type for the names jsonschema uses itself
bool is_kind(const Val& v, std::string_view type, Draft d) {
    if (type == "array") return v.k == K::List;
    if (type == "object") return v.k == K::Dict;
    if (type == "string") return v.k == K::Str;
    if (type == "boolean") return v.k == K::Bool;
    if (type == "null") return v.k == K::None;
    if (type == "number") return v.number();
    if (type == "integer")
        return v.k == K::Int || v.k == K::Big ||
               (d >= Draft::D6 && v.k == K::Float && std::isfinite(v.f) && v.f == std::trunc(v.f));
    return false;
}

/// pprint.pformat(x, width=72, sort_dicts=False), for what fits on its line; longer values as their repr
std::string pformat(const Val& v) { return repr(v); }

/// validator.is_type(instance, type) for a schema's type name: UnknownType when the checker has no such type
bool type_is(const Ctx& c, const Val& inst, const Val& type, const Val& schema) {
    if (type.k == K::List || type.k == K::Dict) raise("TypeError", std::string("unhashable type: '") + type_name(type) + "'");
    static constexpr std::string_view kTypes[] = {"array", "boolean", "integer", "null", "number", "object", "string"};
    if (type.k == K::Str) {
        if (type.s == "any" && c.d->draft == Draft::D3) return true;
        if (std::find(std::begin(kTypes), std::end(kTypes), type.s) != std::end(kTypes)) return is_kind(inst, type.s, c.d->draft);
    }
    raise("UnknownType", "Unknown type " + repr(type) + " for validator with schema:\n    " + pformat(schema) +
                             "\n\nWhile checking instance:\n    " + pformat(inst));
}

/// re.search(pattern, string)
bool re_search(Run& r, const std::string& pattern, std::string_view subject) {
    return re::search(r.patterns.get(pattern), subject);
}

bool re_search(Run& r, const Val& pattern, std::string_view subject) {
    if (pattern.k != K::Str) raise("TypeError", "first argument must be string or compiled pattern");
    return re_search(r, pattern.s, subject);
}

std::string index_str(size_t k) { return std::to_string(k); }

std::string items_repr(const std::vector<Val>& items, size_t from) {
    std::string out = "[";
    for (size_t k = from; k < items.size(); ++k) {
        if (k > from) out += ", ";
        out += repr(items[k]);
    }
    return out + "]";
}

// ---- the keywords (jsonschema/_keywords.py and _legacy_keywords.py)
#define KEYWORD(name)                                                                                    \
    bool name([[maybe_unused]] Run& r, [[maybe_unused]] const Ctx& c, [[maybe_unused]] const Val& v,     \
              [[maybe_unused]] const Val& inst, [[maybe_unused]] const Val& schema, [[maybe_unused]] Sink& s)

/// _validate_reference: unresolvable references wrapped as jsonschema wraps them
bool validate_reference(Run& r, const Ctx& c, const Val& ref, const Val& inst, Sink& s) {
    Resolved res;
    try {
        res = lookup(r, c.res, ref);
    } catch (const PyError& e) {
        if (unresolvable(e.type)) raise("WrappedReferencingError", e.type + ": " + e.msg);
        throw;
    }
    return descend(r, c, inst, *res.contents, s, nullptr, &res.resolver);
}

KEYWORD(kw_ref) { return validate_reference(r, c, v, inst, s); }

KEYWORD(kw_recursive_ref) {
    const Resolved res = recursive_ref(r, c.res);
    return descend(r, c, inst, *res.contents, s, nullptr, &res.resolver);
}

KEYWORD(kw_additional_items) {
    if (inst.k != K::List) return true;
    const Val* items = schema.get("items");
    if (items == nullptr || items->k == K::Dict) return true;
    const int64_t len = py_len(*items);
    if (v.k == K::Dict) {
        for (int64_t k = len; k < (int64_t) inst.list.size(); ++k) {
            const std::string p = index_str((size_t) k);
            if (!descend(r, c, inst.list[(size_t) k], v, s, &p)) return false;
        }
        return true;
    }
    if (truthy(v) || (int64_t) inst.list.size() <= len) return true;
    return emit(r, s, [&] {
        std::vector<std::string> reprs;
        for (size_t k = (size_t) len; k < inst.list.size(); ++k) reprs.push_back(repr(inst.list[k]));
        return "Additional items are not allowed (" + extras_msg(reprs) + " unexpected)";
    });
}

KEYWORD(kw_additional_properties) {
    if (inst.k != K::Dict) return true;
    const Val* props = schema.get("properties");
    const Val* pp = schema.get("patternProperties");
    std::vector<std::string> pats;
    if (pp != nullptr)
        each_item(*pp, [&](const Val& p) {
            if (p.k != K::Str)
                raise("TypeError", "sequence item " + std::to_string(pats.size()) + ": expected str instance, " +
                                       type_name(p) + " found");
            pats.push_back(p.s);
            return true;
        });
    const std::string patterns = joined(pats, "|");
    std::vector<const Member*> extras;
    for (const Member& m : inst.dict) {
        if (props != nullptr && py_in(Val::str(m.key), *props)) continue;
        if (!patterns.empty() && re_search(r, patterns, m.key)) continue;
        extras.push_back(&m);
    }
    if (v.k == K::Dict) {   // Python takes them in set order (by hash); here in the instance's
        for (const Member* m : extras)
            if (!descend(r, c, m->val, v, s, &m->key)) return false;
        return true;
    }
    if (truthy(v) || extras.empty()) return true;
    return emit(r, s, [&] {
        std::vector<std::string> keys;
        for (const Member* m : extras) keys.push_back(m->key);
        std::sort(keys.begin(), keys.end());
        std::vector<std::string> reprs;
        for (const std::string& k : keys) reprs.push_back(py_repr_str(k));
        if (pp != nullptr) {
            std::sort(pats.begin(), pats.end());
            std::vector<std::string> preprs;
            for (const std::string& p : pats) preprs.push_back(py_repr_str(p));
            return joined(reprs) + (extras.size() == 1 ? " does" : " do") + " not match any of the regexes: " + joined(preprs);
        }
        return "Additional properties are not allowed (" + extras_msg(reprs) + " unexpected)";
    });
}

KEYWORD(kw_items) {
    if (inst.k != K::List) return true;
    const Val* pi = schema.get("prefixItems");
    const int64_t prefix = pi != nullptr ? py_len(*pi) : 0, total = (int64_t) inst.list.size(), extra = total - prefix;
    if (extra <= 0) return true;
    if (v.k == K::Bool && !v.b) {
        return emit(r, s, [&] {
            const std::string rest = extra != 1 ? items_repr(inst.list, (size_t) prefix) : repr(inst.list[(size_t) prefix]);
            return "Expected at most " + std::to_string(prefix) + (prefix != 1 ? " items" : " item") + " but found " +
                   std::to_string(extra) + " extra: " + rest;
        });
    }
    for (int64_t k = prefix; k < total; ++k) {
        const std::string p = index_str((size_t) k);
        if (!descend(r, c, inst.list[(size_t) k], v, s, &p)) return false;
    }
    return true;
}

KEYWORD(kw_const) {
    if (js_equal(inst, v, r.frames_left())) return true;
    return emit(r, s, [&] { return repr(v) + " was expected"; });
}

KEYWORD(kw_contains) {
    if (inst.k != K::List) return true;
    const Val* minc = schema.get("minContains");
    const Val* maxc = schema.get("maxContains");
    const Val min_contains = minc != nullptr ? *minc : Val::integer(1);
    const Val max_contains = maxc != nullptr ? *maxc : Val::integer((int64_t) inst.list.size());
    const Ctx cv = evolve(c, v);
    int64_t matches = 0;
    for (const Val& each : inst.list) {
        if (!valid_with(r, cv, each, v)) continue;
        if (order(Val::integer(++matches), max_contains, ">") > 0)
            return emit(r, s, [&] {
                return "Too many items match the given schema (expected at most " + py_str(max_contains) + ")";
            });
    }
    if (order(Val::integer(matches), min_contains, "<") >= 0) return true;
    if (matches == 0) return emit(r, s, [&] { return repr(inst) + " does not contain items matching the given schema"; });
    return emit(r, s, [&] {
        return "Too few items match the given schema (expected at least " + py_str(min_contains) + " but only " +
               std::to_string(matches) + " matched)";
    });
}

KEYWORD(kw_exclusive_minimum) {
    if (!inst.number() || order(inst, v, "<=") > 0) return true;
    return emit(r, s, [&] { return repr(inst) + " is less than or equal to the minimum of " + repr(v); });
}

KEYWORD(kw_exclusive_maximum) {
    if (!inst.number() || order(inst, v, ">=") < 0) return true;
    return emit(r, s, [&] { return repr(inst) + " is greater than or equal to the maximum of " + repr(v); });
}

KEYWORD(kw_minimum) {
    if (!inst.number() || order(inst, v, "<") >= 0) return true;
    return emit(r, s, [&] { return repr(inst) + " is less than the minimum of " + repr(v); });
}

KEYWORD(kw_maximum) {
    if (!inst.number() || order(inst, v, ">") <= 0) return true;
    return emit(r, s, [&] { return repr(inst) + " is greater than the maximum of " + repr(v); });
}

KEYWORD(kw_multiple_of) {
    if (!inst.number()) return true;
    bool failed = false;
    if (v.k == K::Float) {
        if (v.f == 0) raise("ZeroDivisionError", "float division by zero");
        const double q = to_float(inst) / v.f;
        failed = std::isfinite(q) ? q != std::trunc(q) : !fraction_multiple(inst, v.f);
    } else if (numeric(v)) {
        failed = !remainder_zero(inst, v);
    } else {
        raise("TypeError", std::string("unsupported operand type(s) for %: '") + type_name(inst) + "' and '" +
                               type_name(v) + "'");
    }
    if (!failed) return true;
    return emit(r, s, [&] { return repr(inst) + " is not a multiple of " + py_str(v); });
}

KEYWORD(kw_min_items) {
    if (inst.k != K::List || order(Val::integer((int64_t) inst.list.size()), v, "<") >= 0) return true;
    return emit(r, s, [&] { return repr(inst) + (py_eq(v, Val::integer(1)) ? " should be non-empty" : " is too short"); });
}

KEYWORD(kw_max_items) {
    if (inst.k != K::List || order(Val::integer((int64_t) inst.list.size()), v, ">") <= 0) return true;
    return emit(r, s, [&] { return repr(inst) + (py_eq(v, Val::integer(0)) ? " is expected to be empty" : " is too long"); });
}

KEYWORD(kw_unique_items) {
    if (!truthy(v) || inst.k != K::List || uniq(inst.list, r.frames_left())) return true;
    return emit(r, s, [&] { return repr(inst) + " has non-unique elements"; });
}

KEYWORD(kw_pattern) {
    if (inst.k != K::Str || re_search(r, v, inst.s)) return true;
    return emit(r, s, [&] { return repr(inst) + " does not match " + repr(v); });
}

/// check_schema's format checker: of the formats the metaschemas use, "regex" is the one with a checker (uri and
/// uri-reference need packages jsonschema does not install)
KEYWORD(kw_format) {
    if (!r.formats) return true;
    if (v.k == K::List || v.k == K::Dict) raise("TypeError", std::string("unhashable type: '") + type_name(v) + "'");
    if (v.k != K::Str || v.s != "regex" || inst.k != K::Str) return true;
    try {
        r.patterns.get(inst.s);
        return true;
    } catch (const PyError& e) {
        if (e.type != "re.error") throw;
    }
    return emit(r, s, [&] { return repr(inst) + " is not a " + repr(v); });
}

KEYWORD(kw_min_length) {
    if (inst.k != K::Str || order(Val::integer((int64_t) utf8_len(inst.s)), v, "<") >= 0) return true;
    return emit(r, s, [&] { return repr(inst) + (py_eq(v, Val::integer(1)) ? " should be non-empty" : " is too short"); });
}

KEYWORD(kw_max_length) {
    if (inst.k != K::Str || order(Val::integer((int64_t) utf8_len(inst.s)), v, ">") <= 0) return true;
    return emit(r, s, [&] { return repr(inst) + (py_eq(v, Val::integer(0)) ? " is expected to be empty" : " is too long"); });
}

/// the dependency of a present property: each name in it must be present too
bool required_names(Run& r, const Val& names, const std::string& property, const Val& inst, Sink& s) {
    return each_item(names, [&](const Val& each) {
        if (py_in(each, inst)) return true;
        return emit(r, s, [&] { return repr(each) + " is a dependency of " + py_repr_str(property); });
    });
}

KEYWORD(kw_dependent_required) {
    if (inst.k != K::Dict) return true;
    if (v.k != K::Dict) no_attribute(v, "items");
    for (const Member& m : v.dict)
        if (inst.has(m.key) && !required_names(r, m.val, m.key, inst, s)) return false;
    return true;
}

KEYWORD(kw_dependent_schemas) {
    if (inst.k != K::Dict) return true;
    if (v.k != K::Dict) no_attribute(v, "items");
    for (const Member& m : v.dict)
        if (inst.has(m.key) && !descend(r, c, inst, m.val, s)) return false;
    return true;
}

KEYWORD(kw_enum) {
    bool found = false;
    each_item(v, [&](const Val& e) {
        found = js_equal(e, inst, r.frames_left());
        return !found;
    });
    if (found) return true;
    return emit(r, s, [&] { return repr(inst) + " is not one of " + repr(v); });
}

KEYWORD(kw_type) {
    std::vector<std::string> names;
    bool ok = false;
    auto check = [&](const Val& t) {
        ok = type_is(c, inst, t, schema);
        return !ok;
    };
    if (v.k == K::Str) check(v);
    else each_item(v, check);
    if (ok) return true;
    return emit(r, s, [&] {
        std::vector<std::string> reprs;
        if (v.k == K::Str) reprs.push_back(repr(v));
        else each_item(v, [&](const Val& t) {
            reprs.push_back(repr(t));
            return true;
        });
        return repr(inst) + " is not of type " + joined(reprs);
    });
}

KEYWORD(kw_properties) {
    if (inst.k != K::Dict) return true;
    if (v.k != K::Dict) no_attribute(v, "items");
    for (const Member& m : v.dict)
        if (const Val* x = inst.get(m.key); x != nullptr && !descend(r, c, *x, m.val, s, &m.key)) return false;
    return true;
}

KEYWORD(kw_required) {
    if (inst.k != K::Dict) return true;
    return each_item(v, [&](const Val& p) {
        if (py_in(p, inst)) return true;
        return emit(r, s, [&] { return repr(p) + " is a required property"; });
    });
}

KEYWORD(kw_min_properties) {
    if (inst.k != K::Dict || order(Val::integer((int64_t) inst.dict.size()), v, "<") >= 0) return true;
    return emit(r, s, [&] {
        return repr(inst) + (py_eq(v, Val::integer(1)) ? " should be non-empty" : " does not have enough properties");
    });
}

KEYWORD(kw_max_properties) {
    if (inst.k != K::Dict || order(Val::integer((int64_t) inst.dict.size()), v, ">") <= 0) return true;
    return emit(r, s, [&] {
        return repr(inst) + (py_eq(v, Val::integer(0)) ? " is expected to be empty" : " has too many properties");
    });
}

KEYWORD(kw_all_of) {
    return each_item(v, [&](const Val& sub) { return descend(r, c, inst, sub, s); });
}

KEYWORD(kw_any_of) {
    bool valid = false;
    each_item(v, [&](const Val& sub) {
        Sink all;
        all.all = true;
        descend(r, c, inst, sub, all);
        valid = all.count == 0;
        return !valid;
    });
    if (valid) return true;
    return emit(r, s, [&] { return repr(inst) + " is not valid under any of the given schemas"; });
}

KEYWORD(kw_one_of) {
    std::deque<Val> copies;
    std::vector<const Val*> subs;
    if (v.k == K::List) {
        for (const Val& e : v.list) subs.push_back(&e);
    } else {
        each_item(v, [&](const Val& e) {
            copies.push_back(e);
            subs.push_back(&copies.back());
            return true;
        });
    }
    size_t first = npos;
    for (size_t k = 0; k < subs.size() && first == npos; ++k) {
        Sink all;
        all.all = true;
        descend(r, c, inst, *subs[k], all);
        if (all.count == 0) first = k;
    }
    if (first == npos) return emit(r, s, [&] { return repr(inst) + " is not valid under any of the given schemas"; });
    std::vector<size_t> more;
    for (size_t k = first + 1; k < subs.size(); ++k)
        if (is_valid(r, c, inst, *subs[k])) more.push_back(k);
    if (more.empty()) return true;
    more.push_back(first);
    return emit(r, s, [&] {
        std::vector<std::string> reprs;
        for (const size_t k : more) reprs.push_back(repr(*subs[k]));
        return repr(inst) + " is valid under each of " + joined(reprs);
    });
}

KEYWORD(kw_not) {
    if (!is_valid(r, c, inst, v)) return true;
    return emit(r, s, [&] { return repr(inst) + " should not be valid under " + repr(v); });
}

KEYWORD(kw_if) {
    if (is_valid(r, c, inst, v)) {
        const Val* then = schema.get("then");
        return then == nullptr || descend(r, c, inst, *then, s);
    }
    const Val* other = schema.get("else");
    return other == nullptr || descend(r, c, inst, *other, s);
}

KEYWORD(kw_prefix_items) {
    if (inst.k != K::List) return true;
    if (v.k != K::List) return true;
    for (size_t k = 0; k < inst.list.size() && k < v.list.size(); ++k) {
        const std::string p = index_str(k);
        if (!descend(r, c, inst.list[k], v.list[k], s, &p)) return false;
    }
    return true;
}

KEYWORD(kw_pattern_properties) {
    if (inst.k != K::Dict) return true;
    if (v.k != K::Dict) no_attribute(v, "items");
    for (const Member& pm : v.dict)
        for (const Member& m : inst.dict)
            if (re_search(r, pm.key, m.key) && !descend(r, c, m.val, pm.val, s, &m.key)) return false;
    return true;
}

KEYWORD(kw_property_names) {
    if (inst.k != K::Dict) return true;
    for (const Member& m : inst.dict) {
        const Val key = Val::str(m.key);
        if (!descend(r, c, key, v, s)) return false;
    }
    return true;
}

KEYWORD(kw_dependencies_d3) {
    if (inst.k != K::Dict) return true;
    if (v.k != K::Dict) no_attribute(v, "items");
    for (const Member& m : v.dict) {
        if (!inst.has(m.key)) continue;
        const Val& dep = m.val;
        if (dep.k == K::Dict) {
            if (!descend(r, c, inst, dep, s)) return false;
        } else if (dep.k == K::Str) {
            if (!inst.has(dep.s) && !emit(r, s, [&] { return repr(dep) + " is a dependency of " + py_repr_str(m.key); }))
                return false;
        } else if (!required_names(r, dep, m.key, inst, s)) {
            return false;
        }
    }
    return true;
}

KEYWORD(kw_dependencies_d4) {
    if (inst.k != K::Dict) return true;
    if (v.k != K::Dict) no_attribute(v, "items");
    for (const Member& m : v.dict) {
        if (!inst.has(m.key)) continue;
        if (m.val.k == K::List ? !required_names(r, m.val, m.key, inst, s) : !descend(r, c, inst, m.val, s)) return false;
    }
    return true;
}

KEYWORD(kw_disallow_d3) {
    auto check = [&](const Val& d) {
        Val types;
        types.k = K::List;
        types.list.push_back(d);
        Val sch;
        sch.k = K::Dict;
        sch.dict.push_back({"type", std::move(types)});
        if (!is_valid(r, c, inst, sch)) return true;
        return emit(r, s, [&] { return repr(d) + " is disallowed for " + repr(inst); });
    };
    if (v.k == K::Str) return check(v);
    return each_item(v, check);
}

KEYWORD(kw_extends_d3) {
    if (v.k == K::Dict) return descend(r, c, inst, v, s);
    return each_item(v, [&](const Val& sub) { return descend(r, c, inst, sub, s); });
}

KEYWORD(kw_items_d3) {
    if (inst.k != K::List) return true;
    for (size_t k = 0; k < inst.list.size(); ++k) {
        const Val* sub = v.k == K::Dict ? &v : v.k == K::List && k < v.list.size() ? &v.list[k] : nullptr;
        if (sub == nullptr) break;
        const std::string p = index_str(k);
        if (!descend(r, c, inst.list[k], *sub, s, &p)) return false;
    }
    return true;
}

KEYWORD(kw_items_d6) {
    if (inst.k != K::List) return true;
    for (size_t k = 0; k < inst.list.size(); ++k) {
        const Val* sub = v.k != K::List ? &v : k < v.list.size() ? &v.list[k] : nullptr;
        if (sub == nullptr) break;
        const std::string p = index_str(k);
        if (!descend(r, c, inst.list[k], *sub, s, &p)) return false;
    }
    return true;
}

/// minimum / maximum of drafts 3 and 4, with their boolean exclusiveMinimum / exclusiveMaximum
bool bound_d34(Run& r, const Val& v, const Val& inst, const Val& schema, Sink& s, bool lower) {
    if (!inst.number()) return true;
    const Val* ex = schema.get(lower ? "exclusiveMinimum" : "exclusiveMaximum");
    const bool exclusive = ex != nullptr && truthy(*ex);
    const int cmp = order(inst, v, lower ? (exclusive ? "<=" : "<") : (exclusive ? ">=" : ">"));
    const bool failed = lower ? (exclusive ? cmp <= 0 : cmp < 0) : (exclusive ? cmp >= 0 : cmp > 0);
    if (!failed) return true;
    return emit(r, s, [&] {
        return repr(inst) + " is " + (lower ? "less than" : "greater than") + (exclusive ? " or equal to" : "") + " the " +
               (lower ? "minimum" : "maximum") + " of " + repr(v);
    });
}

KEYWORD(kw_minimum_d34) { return bound_d34(r, v, inst, schema, s, true); }
KEYWORD(kw_maximum_d34) { return bound_d34(r, v, inst, schema, s, false); }

KEYWORD(kw_properties_d3) {
    if (inst.k != K::Dict) return true;
    if (v.k != K::Dict) no_attribute(v, "items");
    for (const Member& m : v.dict) {
        if (const Val* x = inst.get(m.key)) {
            if (!descend(r, c, *x, m.val, s, &m.key)) return false;
            continue;
        }
        if (m.val.k != K::Dict) no_attribute(m.val, "get");
        const Val* req = m.val.get("required");
        if (req != nullptr && truthy(*req)) {
            const PathPush p(r, &m.key);
            if (!emit(r, s, [&] { return py_repr_str(m.key) + " is a required property"; })) return false;
        }
    }
    return true;
}

KEYWORD(kw_type_d3) {
    std::deque<Val> keys;   // a dict's keys or a str's characters, as Python iterates them
    std::vector<const Val*> types;
    if (v.k == K::Str) {
        types.push_back(&v);
    } else if (v.k == K::List) {
        for (const Val& t : v.list) types.push_back(&t);
    } else {
        each_item(v, [&](const Val& t) {
            keys.push_back(t);
            types.push_back(&keys.back());
            return true;
        });
    }
    for (const Val* t : types) {
        if (t->k == K::Dict) {
            Sink all;
            all.all = true;
            descend(r, c, inst, *t, all);
            if (all.count == 0) return true;
        } else if (type_is(c, inst, *t, schema)) {
            return true;
        }
    }
    return emit(r, s, [&] {
        std::vector<std::string> reprs;
        for (const Val* t : types) {
            const Val* name = t->k == K::Dict ? t->get("name") : nullptr;
            reprs.push_back(repr(name != nullptr ? *name : *t));
        }
        return repr(inst) + " is not of type " + joined(reprs);
    });
}

KEYWORD(kw_contains_d6) {
    if (inst.k != K::List) return true;
    for (const Val& e : inst.list)
        if (is_valid(r, c, e, v)) return true;
    return emit(r, s, [&] { return "None of " + repr(inst) + " are valid under the given schema"; });
}

// ---- unevaluatedItems / unevaluatedProperties
/// the reference's resolution with its schema's class (find_evaluated_*'s, which do not wrap Unresolvable)
std::pair<Ctx, const Val*> resolved_ctx(const Ctx& c, const Resolved& res) {
    return {evolve(c, *res.contents, &res.resolver), res.contents};
}

/// find_evaluated_item_indexes_by_schema (2020-12, or 2019-09's in _legacy_keywords)
void evaluated_items(Run& r, const Ctx& c, const Val& inst, const Val& schema, std::vector<char>& out, bool d2019) {
    const PyFrame f(r);
    if (schema.k == K::Bool) return;
    auto all = [&] { std::fill(out.begin(), out.end(), (char) 1); };
    auto mark = [&](int64_t n) {
        for (int64_t k = 0; k < n && k < (int64_t) out.size(); ++k) out[(size_t) k] = 1;
    };
    if (!d2019 && schema.has("items")) return all();
    if (const Val* ref = schema.get("$ref"); ref != nullptr && ref->k != K::None) {
        const auto [ev, contents] = resolved_ctx(c, lookup(r, c.res, *ref));
        evaluated_items(r, ev, inst, *contents, out, d2019);
    }
    if (d2019) {
        if (schema.has("$recursiveRef")) {
            const auto [ev, contents] = resolved_ctx(c, recursive_ref(r, c.res));
            evaluated_items(r, ev, inst, *contents, out, d2019);
        }
        if (const Val* items = schema.get("items")) {
            if (schema.has("additionalItems") || items->k == K::Dict) return all();
            mark(py_len(*items));
        }
    } else {
        if (const Val* dyn = schema.get("$dynamicRef"); dyn != nullptr && dyn->k != K::None) {
            const auto [ev, contents] = resolved_ctx(c, lookup(r, c.res, *dyn));
            evaluated_items(r, ev, inst, *contents, out, d2019);
        }
        if (const Val* pi = schema.get("prefixItems")) mark(py_len(*pi));
    }
    if (const Val* iff = schema.get("if")) {
        if (is_valid(r, c, inst, *iff)) {
            evaluated_items(r, c, inst, *iff, out, d2019);
            if (const Val* then = schema.get("then")) evaluated_items(r, c, inst, *then, out, d2019);
        } else if (const Val* other = schema.get("else")) {
            evaluated_items(r, c, inst, *other, out, d2019);
        }
    }
    for (const char* kw : {"contains", "unevaluatedItems"})
        if (const Val* sub = schema.get(kw))
            for (size_t k = 0; k < inst.list.size(); ++k)
                if (is_valid(r, c, inst.list[k], *sub)) out[k] = 1;
    for (const char* kw : {"allOf", "oneOf", "anyOf"})
        if (const Val* subs = schema.get(kw))
            each_item(*subs, [&](const Val& sub) {
                if (descend_valid(r, c, inst, sub)) evaluated_items(r, c, inst, sub, out, d2019);
                return true;
            });
}

/// find_evaluated_property_keys_by_schema (2020-12, or 2019-09's), as marks on the instance's members
void evaluated_keys(Run& r, const Ctx& c, const Val& inst, const Val& schema, std::vector<char>& out, bool d2019) {
    const PyFrame f(r);
    if (schema.k == K::Bool) return;
    auto mark_key = [&](std::string_view key) {
        for (size_t k = 0; k < inst.dict.size(); ++k)
            if (inst.dict[k].key == key) out[k] = 1;
    };
    if (const Val* ref = schema.get("$ref"); ref != nullptr && ref->k != K::None) {
        const auto [ev, contents] = resolved_ctx(c, lookup(r, c.res, *ref));
        evaluated_keys(r, ev, inst, *contents, out, d2019);
    }
    if (d2019) {
        if (schema.has("$recursiveRef")) {
            const auto [ev, contents] = resolved_ctx(c, recursive_ref(r, c.res));
            evaluated_keys(r, ev, inst, *contents, out, d2019);
        }
        for (const char* kw : {"properties", "additionalProperties", "unevaluatedProperties"}) {
            const Val* value = schema.get(kw);
            if (value == nullptr) continue;
            if (value->k == K::Bool && value->b) std::fill(out.begin(), out.end(), (char) 1);
            else if (value->k == K::Dict)   // the subschema's keywords, as Python's loop takes them
                for (const Member& m : value->dict) mark_key(m.key);
        }
    } else {
        if (const Val* dyn = schema.get("$dynamicRef"); dyn != nullptr && dyn->k != K::None) {
            const auto [ev, contents] = resolved_ctx(c, lookup(r, c.res, *dyn));
            evaluated_keys(r, ev, inst, *contents, out, d2019);
        }
        if (const Val* props = schema.get("properties"); props != nullptr && props->k == K::Dict)
            for (const Member& m : props->dict) mark_key(m.key);
        for (const char* kw : {"additionalProperties", "unevaluatedProperties"}) {
            const Val* sub = schema.get(kw);
            if (sub == nullptr || sub->k == K::None) continue;
            for (size_t k = 0; k < inst.dict.size(); ++k)
                if (descend_valid(r, c, inst.dict[k].val, *sub)) out[k] = 1;
        }
    }
    if (const Val* pp = schema.get("patternProperties"))
        for (size_t k = 0; k < inst.dict.size(); ++k)
            each_item(*pp, [&](const Val& pattern) {
                if (re_search(r, pattern, inst.dict[k].key)) out[k] = 1;
                return true;
            });
    if (const Val* deps = schema.get("dependentSchemas")) {
        if (deps->k != K::Dict) no_attribute(*deps, "items");
        for (const Member& m : deps->dict)
            if (inst.has(m.key)) evaluated_keys(r, c, inst, m.val, out, d2019);
    }
    for (const char* kw : {"allOf", "oneOf", "anyOf"})
        if (const Val* subs = schema.get(kw))
            each_item(*subs, [&](const Val& sub) {
                if (descend_valid(r, c, inst, sub)) evaluated_keys(r, c, inst, sub, out, d2019);
                return true;
            });
    if (const Val* iff = schema.get("if")) {
        if (is_valid(r, c, inst, *iff)) {
            evaluated_keys(r, c, inst, *iff, out, d2019);
            if (const Val* then = schema.get("then")) evaluated_keys(r, c, inst, *then, out, d2019);
        } else if (const Val* other = schema.get("else")) {
            evaluated_keys(r, c, inst, *other, out, d2019);
        }
    }
}

bool unevaluated_items(Run& r, const Ctx& c, const Val& inst, const Val& schema, Sink& s, bool d2019) {
    if (inst.k != K::List) return true;
    std::vector<char> ev(inst.list.size(), 0);
    evaluated_items(r, c, inst, schema, ev, d2019);
    if (std::all_of(ev.begin(), ev.end(), [](char e) { return e != 0; })) return true;
    return emit(r, s, [&] {
        std::vector<std::string> reprs;
        for (size_t k = 0; k < inst.list.size(); ++k)
            if (!ev[k]) reprs.push_back(repr(inst.list[k]));
        return "Unevaluated items are not allowed (" + extras_msg(reprs) + " unexpected)";
    });
}

bool unevaluated_properties(Run& r, const Ctx& c, const Val& v, const Val& inst, const Val& schema, Sink& s, bool d2019) {
    if (inst.k != K::Dict) return true;
    std::vector<char> ev(inst.dict.size(), 0);
    evaluated_keys(r, c, inst, schema, ev, d2019);
    std::vector<std::string> keys;   // a key for each of its errors
    for (size_t k = 0; k < inst.dict.size(); ++k) {
        if (ev[k]) continue;
        Sink all;
        all.all = true;
        descend(r, c, inst.dict[k].val, v, all, &inst.dict[k].key);
        keys.insert(keys.end(), all.count, inst.dict[k].key);
    }
    if (keys.empty()) return true;
    return emit(r, s, [&] {
        const bool none_allowed = v.k == K::Bool && !v.b;
        if (none_allowed) std::sort(keys.begin(), keys.end());
        std::vector<std::string> reprs;
        for (const std::string& k : keys) reprs.push_back(py_repr_str(k));
        return none_allowed ? "Unevaluated properties are not allowed (" + extras_msg(reprs) + " unexpected)"
                            : "Unevaluated properties are not valid under the given schema (" + extras_msg(reprs) +
                                  " unevaluated and invalid)";
    });
}

KEYWORD(kw_unevaluated_items) { return unevaluated_items(r, c, inst, schema, s, false); }
KEYWORD(kw_unevaluated_items_2019) { return unevaluated_items(r, c, inst, schema, s, true); }
KEYWORD(kw_unevaluated_properties) { return unevaluated_properties(r, c, v, inst, schema, s, false); }
KEYWORD(kw_unevaluated_properties_2019) { return unevaluated_properties(r, c, v, inst, schema, s, true); }

#undef KEYWORD

// ---- the validator classes and the metaschemas
// The metaschemas and vocabularies jsonschema checks schemas against (jsonschema-specifications 2025.9.1), compact.
const char* const kMetaDocs[] = {
    R"j({"$schema":"http://json-schema.org/draft-03/schema#","id":"http://json-schema.org/draft-03/schema#","type":"object","properties":{"type":{"type":["string","array"],"items":{"type":["string",{"$ref":"#"}]},"uniqueItems":true,"default":"any"},"properties":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"patternProperties":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"additionalProperties":{"type":[{"$ref":"#"},"boolean"],"default":{}},"items":{"type":[{"$ref":"#"},"array"],"items":{"$ref":"#"},"default":{}},"additionalItems":{"type":[{"$ref":"#"},"boolean"],"default":{}},"required":{"type":"boolean","default":false},"dependencies":{"type":"object","additionalProperties":{"type":["string","array",{"$ref":"#"}],"items":{"type":"string"}},"default":{}},"minimum":{"type":"number"},"maximum":{"type":"number"},"exclusiveMinimum":{"type":"boolean","default":false},"exclusiveMaximum":{"type":"boolean","default":false},"minItems":{"type":"integer","minimum":0,"default":0},"maxItems":{"type":"integer","minimum":0},"uniqueItems":{"type":"boolean","default":false},"pattern":{"type":"string","format":"regex"},"minLength":{"type":"integer","minimum":0,"default":0},"maxLength":{"type":"integer"},"enum":{"type":"array","minItems":1,"uniqueItems":true},"default":{"type":"any"},"title":{"type":"string"},"description":{"type":"string"},"format":{"type":"string"},"divisibleBy":{"type":"number","minimum":0,"exclusiveMinimum":true,"default":1},"disallow":{"type":["string","array"],"items":{"type":["string",{"$ref":"#"}]},"uniqueItems":true},"extends":{"type":[{"$ref":"#"},"array"],"items":{"$ref":"#"},"default":{}},"id":{"type":"string"},"$ref":{"type":"string"},"$schema":{"type":"string","format":"uri"}},"dependencies":{"exclusiveMinimum":"minimum","exclusiveMaximum":"maximum"},"default":{}})j",
    R"j({"id":"http://json-schema.org/draft-04/schema#","$schema":"http://json-schema.org/draft-04/schema#","description":"Core schema meta-schema","definitions":{"schemaArray":{"type":"array","minItems":1,"items":{"$ref":"#"}},"positiveInteger":{"type":"integer","minimum":0},"positiveIntegerDefault0":{"allOf":[{"$ref":"#/definitions/positiveInteger"},{"default":0}]},"simpleTypes":{"enum":["array","boolean","integer","null","number","object","string"]},"stringArray":{"type":"array","items":{"type":"string"},"minItems":1,"uniqueItems":true}},"type":"object","properties":{"id":{"type":"string"},"$schema":{"type":"string"},"title":{"type":"string"},"description":{"type":"string"},"default":{},"multipleOf":{"type":"number","minimum":0,"exclusiveMinimum":true},"maximum":{"type":"number"},"exclusiveMaximum":{"type":"boolean","default":false},"minimum":{"type":"number"},"exclusiveMinimum":{"type":"boolean","default":false},"maxLength":{"$ref":"#/definitions/positiveInteger"},"minLength":{"$ref":"#/definitions/positiveIntegerDefault0"},"pattern":{"type":"string","format":"regex"},"additionalItems":{"anyOf":[{"type":"boolean"},{"$ref":"#"}],"default":{}},"items":{"anyOf":[{"$ref":"#"},{"$ref":"#/definitions/schemaArray"}],"default":{}},"maxItems":{"$ref":"#/definitions/positiveInteger"},"minItems":{"$ref":"#/definitions/positiveIntegerDefault0"},"uniqueItems":{"type":"boolean","default":false},"maxProperties":{"$ref":"#/definitions/positiveInteger"},"minProperties":{"$ref":"#/definitions/positiveIntegerDefault0"},"required":{"$ref":"#/definitions/stringArray"},"additionalProperties":{"anyOf":[{"type":"boolean"},{"$ref":"#"}],"default":{}},"definitions":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"properties":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"patternProperties":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"dependencies":{"type":"object","additionalProperties":{"anyOf":[{"$ref":"#"},{"$ref":"#/definitions/stringArray"}]}},"enum":{"type":"array","minItems":1,"uniqueItems":true},"type":{"anyOf":[{"$ref":"#/definitions/simpleTypes"},{"type":"array","items":{"$ref":"#/definitions/simpleTypes"},"minItems":1,"uniqueItems":true}]},"format":{"type":"string"},"allOf":{"$ref":"#/definitions/schemaArray"},"anyOf":{"$ref":"#/definitions/schemaArray"},"oneOf":{"$ref":"#/definitions/schemaArray"},"not":{"$ref":"#"}},"dependencies":{"exclusiveMaximum":["maximum"],"exclusiveMinimum":["minimum"]},"default":{}})j",
    R"j({"$schema":"http://json-schema.org/draft-06/schema#","$id":"http://json-schema.org/draft-06/schema#","title":"Core schema meta-schema","definitions":{"schemaArray":{"type":"array","minItems":1,"items":{"$ref":"#"}},"nonNegativeInteger":{"type":"integer","minimum":0},"nonNegativeIntegerDefault0":{"allOf":[{"$ref":"#/definitions/nonNegativeInteger"},{"default":0}]},"simpleTypes":{"enum":["array","boolean","integer","null","number","object","string"]},"stringArray":{"type":"array","items":{"type":"string"},"uniqueItems":true,"default":[]}},"type":["object","boolean"],"properties":{"$id":{"type":"string","format":"uri-reference"},"$schema":{"type":"string","format":"uri"},"$ref":{"type":"string","format":"uri-reference"},"title":{"type":"string"},"description":{"type":"string"},"default":{},"examples":{"type":"array","items":{}},"multipleOf":{"type":"number","exclusiveMinimum":0},"maximum":{"type":"number"},"exclusiveMaximum":{"type":"number"},"minimum":{"type":"number"},"exclusiveMinimum":{"type":"number"},"maxLength":{"$ref":"#/definitions/nonNegativeInteger"},"minLength":{"$ref":"#/definitions/nonNegativeIntegerDefault0"},"pattern":{"type":"string","format":"regex"},"additionalItems":{"$ref":"#"},"items":{"anyOf":[{"$ref":"#"},{"$ref":"#/definitions/schemaArray"}],"default":{}},"maxItems":{"$ref":"#/definitions/nonNegativeInteger"},"minItems":{"$ref":"#/definitions/nonNegativeIntegerDefault0"},"uniqueItems":{"type":"boolean","default":false},"contains":{"$ref":"#"},"maxProperties":{"$ref":"#/definitions/nonNegativeInteger"},"minProperties":{"$ref":"#/definitions/nonNegativeIntegerDefault0"},"required":{"$ref":"#/definitions/stringArray"},"additionalProperties":{"$ref":"#"},"definitions":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"properties":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"patternProperties":{"type":"object","additionalProperties":{"$ref":"#"},"propertyNames":{"format":"regex"},"default":{}},"dependencies":{"type":"object","additionalProperties":{"anyOf":[{"$ref":"#"},{"$ref":"#/definitions/stringArray"}]}},"propertyNames":{"$ref":"#"},"const":{},"enum":{"type":"array"},"type":{"anyOf":[{"$ref":"#/definitions/simpleTypes"},{"type":"array","items":{"$ref":"#/definitions/simpleTypes"},"minItems":1,"uniqueItems":true}]},"format":{"type":"string"},"allOf":{"$ref":"#/definitions/schemaArray"},"anyOf":{"$ref":"#/definitions/schemaArray"},"oneOf":{"$ref":"#/definitions/schemaArray"},"not":{"$ref":"#"}},"default":{}})j",
    R"j({"$schema":"http://json-schema.org/draft-07/schema#","$id":"http://json-schema.org/draft-07/schema#","title":"Core schema meta-schema","definitions":{"schemaArray":{"type":"array","minItems":1,"items":{"$ref":"#"}},"nonNegativeInteger":{"type":"integer","minimum":0},"nonNegativeIntegerDefault0":{"allOf":[{"$ref":"#/definitions/nonNegativeInteger"},{"default":0}]},"simpleTypes":{"enum":["array","boolean","integer","null","number","object","string"]},"stringArray":{"type":"array","items":{"type":"string"},"uniqueItems":true,"default":[]}},"type":["object","boolean"],"properties":{"$id":{"type":"string","format":"uri-reference"},"$schema":{"type":"string","format":"uri"},"$ref":{"type":"string","format":"uri-reference"},"$comment":{"type":"string"},"title":{"type":"string"},"description":{"type":"string"},"default":true,"readOnly":{"type":"boolean","default":false},"examples":{"type":"array","items":true},"multipleOf":{"type":"number","exclusiveMinimum":0},"maximum":{"type":"number"},"exclusiveMaximum":{"type":"number"},"minimum":{"type":"number"},"exclusiveMinimum":{"type":"number"},"maxLength":{"$ref":"#/definitions/nonNegativeInteger"},"minLength":{"$ref":"#/definitions/nonNegativeIntegerDefault0"},"pattern":{"type":"string","format":"regex"},"additionalItems":{"$ref":"#"},"items":{"anyOf":[{"$ref":"#"},{"$ref":"#/definitions/schemaArray"}],"default":true},"maxItems":{"$ref":"#/definitions/nonNegativeInteger"},"minItems":{"$ref":"#/definitions/nonNegativeIntegerDefault0"},"uniqueItems":{"type":"boolean","default":false},"contains":{"$ref":"#"},"maxProperties":{"$ref":"#/definitions/nonNegativeInteger"},"minProperties":{"$ref":"#/definitions/nonNegativeIntegerDefault0"},"required":{"$ref":"#/definitions/stringArray"},"additionalProperties":{"$ref":"#"},"definitions":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"properties":{"type":"object","additionalProperties":{"$ref":"#"},"default":{}},"patternProperties":{"type":"object","additionalProperties":{"$ref":"#"},"propertyNames":{"format":"regex"},"default":{}},"dependencies":{"type":"object","additionalProperties":{"anyOf":[{"$ref":"#"},{"$ref":"#/definitions/stringArray"}]}},"propertyNames":{"$ref":"#"},"const":true,"enum":{"type":"array","items":true},"type":{"anyOf":[{"$ref":"#/definitions/simpleTypes"},{"type":"array","items":{"$ref":"#/definitions/simpleTypes"},"minItems":1,"uniqueItems":true}]},"format":{"type":"string"},"contentMediaType":{"type":"string"},"contentEncoding":{"type":"string"},"if":{"$ref":"#"},"then":{"$ref":"#"},"else":{"$ref":"#"},"allOf":{"$ref":"#/definitions/schemaArray"},"anyOf":{"$ref":"#/definitions/schemaArray"},"oneOf":{"$ref":"#/definitions/schemaArray"},"not":{"$ref":"#"}},"default":true})j",
    R"j({"$schema":"https://json-schema.org/draft/2019-09/schema","$id":"https://json-schema.org/draft/2019-09/schema","$vocabulary":{"https://json-schema.org/draft/2019-09/vocab/core":true,"https://json-schema.org/draft/2019-09/vocab/applicator":true,"https://json-schema.org/draft/2019-09/vocab/validation":true,"https://json-schema.org/draft/2019-09/vocab/meta-data":true,"https://json-schema.org/draft/2019-09/vocab/format":false,"https://json-schema.org/draft/2019-09/vocab/content":true},"$recursiveAnchor":true,"title":"Core and Validation specifications meta-schema","allOf":[{"$ref":"meta/core"},{"$ref":"meta/applicator"},{"$ref":"meta/validation"},{"$ref":"meta/meta-data"},{"$ref":"meta/format"},{"$ref":"meta/content"}],"type":["object","boolean"],"properties":{"definitions":{"$comment":"While no longer an official keyword as it is replaced by $defs, this keyword is retained in the meta-schema to prevent incompatible extensions as it remains in common use.","type":"object","additionalProperties":{"$recursiveRef":"#"},"default":{}},"dependencies":{"$comment":"\"dependencies\" is no longer a keyword, but schema authors should avoid redefining it to facilitate a smooth transition to \"dependentSchemas\" and \"dependentRequired\"","type":"object","additionalProperties":{"anyOf":[{"$recursiveRef":"#"},{"$ref":"meta/validation#/$defs/stringArray"}]}}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2019-09/schema","$id":"https://json-schema.org/draft/2019-09/meta/applicator","$vocabulary":{"https://json-schema.org/draft/2019-09/vocab/applicator":true},"$recursiveAnchor":true,"title":"Applicator vocabulary meta-schema","type":["object","boolean"],"properties":{"additionalItems":{"$recursiveRef":"#"},"unevaluatedItems":{"$recursiveRef":"#"},"items":{"anyOf":[{"$recursiveRef":"#"},{"$ref":"#/$defs/schemaArray"}]},"contains":{"$recursiveRef":"#"},"additionalProperties":{"$recursiveRef":"#"},"unevaluatedProperties":{"$recursiveRef":"#"},"properties":{"type":"object","additionalProperties":{"$recursiveRef":"#"},"default":{}},"patternProperties":{"type":"object","additionalProperties":{"$recursiveRef":"#"},"propertyNames":{"format":"regex"},"default":{}},"dependentSchemas":{"type":"object","additionalProperties":{"$recursiveRef":"#"}},"propertyNames":{"$recursiveRef":"#"},"if":{"$recursiveRef":"#"},"then":{"$recursiveRef":"#"},"else":{"$recursiveRef":"#"},"allOf":{"$ref":"#/$defs/schemaArray"},"anyOf":{"$ref":"#/$defs/schemaArray"},"oneOf":{"$ref":"#/$defs/schemaArray"},"not":{"$recursiveRef":"#"}},"$defs":{"schemaArray":{"type":"array","minItems":1,"items":{"$recursiveRef":"#"}}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2019-09/schema","$id":"https://json-schema.org/draft/2019-09/meta/content","$vocabulary":{"https://json-schema.org/draft/2019-09/vocab/content":true},"$recursiveAnchor":true,"title":"Content vocabulary meta-schema","type":["object","boolean"],"properties":{"contentMediaType":{"type":"string"},"contentEncoding":{"type":"string"},"contentSchema":{"$recursiveRef":"#"}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2019-09/schema","$id":"https://json-schema.org/draft/2019-09/meta/core","$vocabulary":{"https://json-schema.org/draft/2019-09/vocab/core":true},"$recursiveAnchor":true,"title":"Core vocabulary meta-schema","type":["object","boolean"],"properties":{"$id":{"type":"string","format":"uri-reference","$comment":"Non-empty fragments not allowed.","pattern":"^[^#]*#?$"},"$schema":{"type":"string","format":"uri"},"$anchor":{"type":"string","pattern":"^[A-Za-z][-A-Za-z0-9.:_]*$"},"$ref":{"type":"string","format":"uri-reference"},"$recursiveRef":{"type":"string","format":"uri-reference"},"$recursiveAnchor":{"type":"boolean","default":false},"$vocabulary":{"type":"object","propertyNames":{"type":"string","format":"uri"},"additionalProperties":{"type":"boolean"}},"$comment":{"type":"string"},"$defs":{"type":"object","additionalProperties":{"$recursiveRef":"#"},"default":{}}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2019-09/schema","$id":"https://json-schema.org/draft/2019-09/meta/format","$vocabulary":{"https://json-schema.org/draft/2019-09/vocab/format":true},"$recursiveAnchor":true,"title":"Format vocabulary meta-schema","type":["object","boolean"],"properties":{"format":{"type":"string"}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2019-09/schema","$id":"https://json-schema.org/draft/2019-09/meta/meta-data","$vocabulary":{"https://json-schema.org/draft/2019-09/vocab/meta-data":true},"$recursiveAnchor":true,"title":"Meta-data vocabulary meta-schema","type":["object","boolean"],"properties":{"title":{"type":"string"},"description":{"type":"string"},"default":true,"deprecated":{"type":"boolean","default":false},"readOnly":{"type":"boolean","default":false},"writeOnly":{"type":"boolean","default":false},"examples":{"type":"array","items":true}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2019-09/schema","$id":"https://json-schema.org/draft/2019-09/meta/validation","$vocabulary":{"https://json-schema.org/draft/2019-09/vocab/validation":true},"$recursiveAnchor":true,"title":"Validation vocabulary meta-schema","type":["object","boolean"],"properties":{"multipleOf":{"type":"number","exclusiveMinimum":0},"maximum":{"type":"number"},"exclusiveMaximum":{"type":"number"},"minimum":{"type":"number"},"exclusiveMinimum":{"type":"number"},"maxLength":{"$ref":"#/$defs/nonNegativeInteger"},"minLength":{"$ref":"#/$defs/nonNegativeIntegerDefault0"},"pattern":{"type":"string","format":"regex"},"maxItems":{"$ref":"#/$defs/nonNegativeInteger"},"minItems":{"$ref":"#/$defs/nonNegativeIntegerDefault0"},"uniqueItems":{"type":"boolean","default":false},"maxContains":{"$ref":"#/$defs/nonNegativeInteger"},"minContains":{"$ref":"#/$defs/nonNegativeInteger","default":1},"maxProperties":{"$ref":"#/$defs/nonNegativeInteger"},"minProperties":{"$ref":"#/$defs/nonNegativeIntegerDefault0"},"required":{"$ref":"#/$defs/stringArray"},"dependentRequired":{"type":"object","additionalProperties":{"$ref":"#/$defs/stringArray"}},"const":true,"enum":{"type":"array","items":true},"type":{"anyOf":[{"$ref":"#/$defs/simpleTypes"},{"type":"array","items":{"$ref":"#/$defs/simpleTypes"},"minItems":1,"uniqueItems":true}]}},"$defs":{"nonNegativeInteger":{"type":"integer","minimum":0},"nonNegativeIntegerDefault0":{"$ref":"#/$defs/nonNegativeInteger","default":0},"simpleTypes":{"enum":["array","boolean","integer","null","number","object","string"]},"stringArray":{"type":"array","items":{"type":"string"},"uniqueItems":true,"default":[]}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/schema","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/core":true,"https://json-schema.org/draft/2020-12/vocab/applicator":true,"https://json-schema.org/draft/2020-12/vocab/unevaluated":true,"https://json-schema.org/draft/2020-12/vocab/validation":true,"https://json-schema.org/draft/2020-12/vocab/meta-data":true,"https://json-schema.org/draft/2020-12/vocab/format-annotation":true,"https://json-schema.org/draft/2020-12/vocab/content":true},"$dynamicAnchor":"meta","title":"Core and Validation specifications meta-schema","allOf":[{"$ref":"meta/core"},{"$ref":"meta/applicator"},{"$ref":"meta/unevaluated"},{"$ref":"meta/validation"},{"$ref":"meta/meta-data"},{"$ref":"meta/format-annotation"},{"$ref":"meta/content"}],"type":["object","boolean"],"$comment":"This meta-schema also defines keywords that have appeared in previous drafts in order to prevent incompatible extensions as they remain in common use.","properties":{"definitions":{"$comment":"\"definitions\" has been replaced by \"$defs\".","type":"object","additionalProperties":{"$dynamicRef":"#meta"},"deprecated":true,"default":{}},"dependencies":{"$comment":"\"dependencies\" has been split and replaced by \"dependentSchemas\" and \"dependentRequired\" in order to serve their differing semantics.","type":"object","additionalProperties":{"anyOf":[{"$dynamicRef":"#meta"},{"$ref":"meta/validation#/$defs/stringArray"}]},"deprecated":true,"default":{}},"$recursiveAnchor":{"$comment":"\"$recursiveAnchor\" has been replaced by \"$dynamicAnchor\".","$ref":"meta/core#/$defs/anchorString","deprecated":true},"$recursiveRef":{"$comment":"\"$recursiveRef\" has been replaced by \"$dynamicRef\".","$ref":"meta/core#/$defs/uriReferenceString","deprecated":true}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/applicator","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/applicator":true},"$dynamicAnchor":"meta","title":"Applicator vocabulary meta-schema","type":["object","boolean"],"properties":{"prefixItems":{"$ref":"#/$defs/schemaArray"},"items":{"$dynamicRef":"#meta"},"contains":{"$dynamicRef":"#meta"},"additionalProperties":{"$dynamicRef":"#meta"},"properties":{"type":"object","additionalProperties":{"$dynamicRef":"#meta"},"default":{}},"patternProperties":{"type":"object","additionalProperties":{"$dynamicRef":"#meta"},"propertyNames":{"format":"regex"},"default":{}},"dependentSchemas":{"type":"object","additionalProperties":{"$dynamicRef":"#meta"},"default":{}},"propertyNames":{"$dynamicRef":"#meta"},"if":{"$dynamicRef":"#meta"},"then":{"$dynamicRef":"#meta"},"else":{"$dynamicRef":"#meta"},"allOf":{"$ref":"#/$defs/schemaArray"},"anyOf":{"$ref":"#/$defs/schemaArray"},"oneOf":{"$ref":"#/$defs/schemaArray"},"not":{"$dynamicRef":"#meta"}},"$defs":{"schemaArray":{"type":"array","minItems":1,"items":{"$dynamicRef":"#meta"}}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/content","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/content":true},"$dynamicAnchor":"meta","title":"Content vocabulary meta-schema","type":["object","boolean"],"properties":{"contentEncoding":{"type":"string"},"contentMediaType":{"type":"string"},"contentSchema":{"$dynamicRef":"#meta"}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/core","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/core":true},"$dynamicAnchor":"meta","title":"Core vocabulary meta-schema","type":["object","boolean"],"properties":{"$id":{"$ref":"#/$defs/uriReferenceString","$comment":"Non-empty fragments not allowed.","pattern":"^[^#]*#?$"},"$schema":{"$ref":"#/$defs/uriString"},"$ref":{"$ref":"#/$defs/uriReferenceString"},"$anchor":{"$ref":"#/$defs/anchorString"},"$dynamicRef":{"$ref":"#/$defs/uriReferenceString"},"$dynamicAnchor":{"$ref":"#/$defs/anchorString"},"$vocabulary":{"type":"object","propertyNames":{"$ref":"#/$defs/uriString"},"additionalProperties":{"type":"boolean"}},"$comment":{"type":"string"},"$defs":{"type":"object","additionalProperties":{"$dynamicRef":"#meta"}}},"$defs":{"anchorString":{"type":"string","pattern":"^[A-Za-z_][-A-Za-z0-9._]*$"},"uriString":{"type":"string","format":"uri"},"uriReferenceString":{"type":"string","format":"uri-reference"}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/format-annotation","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/format-annotation":true},"$dynamicAnchor":"meta","title":"Format vocabulary meta-schema for annotation results","type":["object","boolean"],"properties":{"format":{"type":"string"}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/format-assertion","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/format-assertion":true},"$dynamicAnchor":"meta","title":"Format vocabulary meta-schema for assertion results","type":["object","boolean"],"properties":{"format":{"type":"string"}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/meta-data","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/meta-data":true},"$dynamicAnchor":"meta","title":"Meta-data vocabulary meta-schema","type":["object","boolean"],"properties":{"title":{"type":"string"},"description":{"type":"string"},"default":true,"deprecated":{"type":"boolean","default":false},"readOnly":{"type":"boolean","default":false},"writeOnly":{"type":"boolean","default":false},"examples":{"type":"array","items":true}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/unevaluated","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/unevaluated":true},"$dynamicAnchor":"meta","title":"Unevaluated applicator vocabulary meta-schema","type":["object","boolean"],"properties":{"unevaluatedItems":{"$dynamicRef":"#meta"},"unevaluatedProperties":{"$dynamicRef":"#meta"}}})j",
    R"j({"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://json-schema.org/draft/2020-12/meta/validation","$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/validation":true},"$dynamicAnchor":"meta","title":"Validation vocabulary meta-schema","type":["object","boolean"],"properties":{"type":{"anyOf":[{"$ref":"#/$defs/simpleTypes"},{"type":"array","items":{"$ref":"#/$defs/simpleTypes"},"minItems":1,"uniqueItems":true}]},"const":true,"enum":{"type":"array","items":true},"multipleOf":{"type":"number","exclusiveMinimum":0},"maximum":{"type":"number"},"exclusiveMaximum":{"type":"number"},"minimum":{"type":"number"},"exclusiveMinimum":{"type":"number"},"maxLength":{"$ref":"#/$defs/nonNegativeInteger"},"minLength":{"$ref":"#/$defs/nonNegativeIntegerDefault0"},"pattern":{"type":"string","format":"regex"},"maxItems":{"$ref":"#/$defs/nonNegativeInteger"},"minItems":{"$ref":"#/$defs/nonNegativeIntegerDefault0"},"uniqueItems":{"type":"boolean","default":false},"maxContains":{"$ref":"#/$defs/nonNegativeInteger"},"minContains":{"$ref":"#/$defs/nonNegativeInteger","default":1},"maxProperties":{"$ref":"#/$defs/nonNegativeInteger"},"minProperties":{"$ref":"#/$defs/nonNegativeIntegerDefault0"},"required":{"$ref":"#/$defs/stringArray"},"dependentRequired":{"type":"object","additionalProperties":{"$ref":"#/$defs/stringArray"}}},"$defs":{"nonNegativeInteger":{"type":"integer","minimum":0},"nonNegativeIntegerDefault0":{"$ref":"#/$defs/nonNegativeInteger","default":0},"simpleTypes":{"enum":["array","boolean","integer","null","number","object","string"]},"stringArray":{"type":"array","items":{"type":"string"},"uniqueItems":true,"default":[]}}})j",
};

Meta make_meta() {
    Meta m;
    using Map = std::map<std::string, Keyword, std::less<>>;
    const Map common = {
        {"$ref", kw_ref}, {"additionalProperties", kw_additional_properties}, {"enum", kw_enum},
        {"format", kw_format}, {"maxItems", kw_max_items}, {"maxLength", kw_max_length}, {"minItems", kw_min_items},
        {"minLength", kw_min_length}, {"pattern", kw_pattern}, {"patternProperties", kw_pattern_properties},
        {"uniqueItems", kw_unique_items}};
    auto with = [&](Map extra) {
        Map out = common;
        for (auto& [k, fn] : extra) out[k] = fn;
        return out;
    };
    const Map d4 = with({{"additionalItems", kw_additional_items}, {"allOf", kw_all_of}, {"anyOf", kw_any_of},
                         {"dependencies", kw_dependencies_d4}, {"items", kw_items_d3}, {"maxProperties", kw_max_properties},
                         {"maximum", kw_maximum_d34}, {"minProperties", kw_min_properties}, {"minimum", kw_minimum_d34},
                         {"multipleOf", kw_multiple_of}, {"not", kw_not}, {"oneOf", kw_one_of},
                         {"properties", kw_properties}, {"required", kw_required}, {"type", kw_type}});
    Map d6 = d4;
    for (const auto& [k, fn] : Map{{"const", kw_const}, {"contains", kw_contains_d6},
                                   {"exclusiveMaximum", kw_exclusive_maximum}, {"exclusiveMinimum", kw_exclusive_minimum},
                                   {"items", kw_items_d6}, {"maximum", kw_maximum}, {"minimum", kw_minimum},
                                   {"propertyNames", kw_property_names}})
        d6[k] = fn;
    Map d7 = d6;
    d7["if"] = kw_if;
    Map d2019 = d7;
    d2019.erase("dependencies");
    for (const auto& [k, fn] : Map{{"$recursiveRef", kw_recursive_ref}, {"contains", kw_contains},
                                   {"dependentRequired", kw_dependent_required},
                                   {"dependentSchemas", kw_dependent_schemas},
                                   {"unevaluatedItems", kw_unevaluated_items_2019},
                                   {"unevaluatedProperties", kw_unevaluated_properties_2019}})
        d2019[k] = fn;
    Map d2020 = d2019;
    for (const char* k : {"$recursiveRef", "additionalItems"}) d2020.erase(k);
    for (const auto& [k, fn] : Map{{"$dynamicRef", kw_ref}, {"items", kw_items}, {"prefixItems", kw_prefix_items},
                                   {"unevaluatedItems", kw_unevaluated_items},
                                   {"unevaluatedProperties", kw_unevaluated_properties}})
        d2020[k] = fn;
    const Map d3 = with({{"additionalItems", kw_additional_items}, {"dependencies", kw_dependencies_d3},
                         {"disallow", kw_disallow_d3}, {"divisibleBy", kw_multiple_of}, {"extends", kw_extends_d3},
                         {"items", kw_items_d3}, {"maximum", kw_maximum_d34}, {"minimum", kw_minimum_d34},
                         {"properties", kw_properties_d3}, {"type", kw_type_d3}});
    const Map* tables[6] = {&d3, &d4, &d6, &d7, &d2019, &d2020};
    for (int d = 0; d < 6; ++d) {
        m.dialects[d].draft = (Draft) d;
        m.dialects[d].id = std::string(kSpecIds[d]);
        m.dialects[d].keywords = *tables[d];
    }
    std::vector<std::pair<std::string, Resource>> todo;
    for (const char* doc : kMetaDocs) {
        const Val& v = m.docs.emplace_back(Decoder(doc).decode());
        const Draft spec = spec_of(v, Draft::D2020);
        const std::string id = resource_id(v, spec).value_or("");
        m.resources[id] = Resource{&v, spec};
        todo.push_back({id, Resource{&v, spec}});
    }
    crawl(todo, m.resources, m.anchors);
    for (Dialect& d : m.dialects) d.meta = m.resources.at(d.id).contents;
    return m;
}

const Meta& meta() {
    static const Meta m = make_meta();
    return m;
}

/// Validator.check_schema: the first error of the schema under its draft's metaschema
std::optional<std::string> check_schema(const Val& schema, const Dialect& d, Patterns& patterns) {
    Registry reg(meta());
    Run run(reg, patterns);
    run.formats = true;
    const Dialect* md = dialect_for(*d.meta, &d);
    const Ctx c{md, Resolver{run.intern(resource_id(*d.meta, md->draft).value_or("")), nullptr}};
    Sink s;
    s.messages = true;
    iter_errors(run, c, schema, *d.meta, s);
    if (s.count == 0) return std::nullopt;
    return s.message;
}

// ---- the response format
/// Python's recursion limit, less the server's frames below prepare_format
constexpr int kMaxSchemaFrames = 980;

/// a schema nested too deep for Python's recursion limit: refused (Python fails with RecursionError)
[[noreturn]] void too_deep() { throw RequestError("maximum recursion depth exceeded"); }

/// the local reference's target as _only_objects follows it (no percent-decoding), or nullptr
const json* ref_target(const json& root, std::string_view ref) {
    const json* target = &root;
    const std::vector<std::string> parts = split(ref.substr(1), "/");
    for (size_t k = 1; k < parts.size(); ++k) {
        if (!target->is_object()) return nullptr;
        const auto it = target->find(replace_all(replace_all(parts[k], "~1", "/"), "~0", "~"));
        if (it == target->end()) return nullptr;
        target = &*it;
    }
    return target;
}

/// _only_objects' calls in all: Python's walks a union's branches again wherever a $ref chain reaches it, 2^n for n
/// levels of unions that name the same next one (days at 40), so past this a schema is refused instead
constexpr int64_t kMaxObjectVisits = 1000000;

/// _only_objects: every value the schema accepts is a JSON object. `frames`: Python's in use (a call each, and for a
/// union the generator of all() / any() too); `visits`: the calls so far
bool only_objects(const json& node, const json& root, std::vector<std::string>& refs, int frames, int64_t& visits) {
    if (frames > kMaxSchemaFrames) too_deep();
    if (++visits > kMaxObjectVisits)
        throw RequestError("response_format schema is too complex to check: its unions are walked more than " +
                           std::to_string(kMaxObjectVisits) + " times");
    if (!node.is_object()) return false;
    if (jget(node, "type") == "object") return true;
    if (const json& ref = jget(node, "$ref"); ref.is_string()) {
        const std::string& r = ref.get_ref<const std::string&>();
        if (starts_with(r, "#") && std::find(refs.begin(), refs.end(), r) == refs.end()) {
            const json* target = ref_target(root, r);
            if (target == nullptr) return false;
            refs.push_back(r);
            const bool ok = only_objects(*target, root, refs, frames + 1, visits);
            refs.pop_back();
            return ok;
        }
    }
    for (const char* key : {"anyOf", "oneOf"}) {
        const json& branches = jget(node, key);
        if (!branches.is_array() || branches.empty()) continue;
        bool all = true;
        for (const json& b : branches) {
            if (!only_objects(b, root, refs, frames + 2, visits)) {
                all = false;
                break;
            }
        }
        if (all) return true;
    }
    const json& all = jget(node, "allOf");
    if (!all.is_array()) return false;
    for (const json& b : all)
        if (only_objects(b, root, refs, frames + 2, visits)) return true;
    return false;
}

/// check_refs: local references only, in Python's order (a frame for each value it visits)
void check_refs(const json& root) {
    struct Open {
        const json* node;
        json::const_iterator it;
    };
    std::vector<Open> st;
    auto call = [&st](const json& n) {
        if ((int) st.size() + 1 > kMaxSchemaFrames) too_deep();
        if (n.is_structured()) st.push_back({&n, n.begin()});
    };
    call(root);
    while (!st.empty()) {
        Open& o = st.back();
        if (o.it == o.node->end()) {
            st.pop_back();
            continue;
        }
        const auto it = o.it++;
        if (o.node->is_object() && (it.key() == "$ref" || it.key() == "$dynamicRef") && it.value().is_string() &&
            !starts_with(it.value().get_ref<const std::string&>(), "#"))
            throw RequestError("response_format supports only local schema references (#...)");
        call(it.value());
    }
}

/// json.dumps(allow_nan=False)'s error for the schema's first non-finite float
void finite_floats(const json& root) {
    std::vector<std::pair<const json*, json::const_iterator>> st;
    auto visit = [&st](const json& v) {
        if (v.is_number_float() && !std::isfinite(v.get<double>()))
            throw RequestError("Out of range float values are not JSON compliant: " + py_repr(v.get<double>()));
        if (v.is_structured()) st.push_back({&v, v.begin()});
    };
    visit(root);
    while (!st.empty()) {
        auto& top = st.back();
        if (top.second == top.first->end()) {
            st.pop_back();
            continue;
        }
        const json& v = top.second.value();
        ++top.second;
        visit(v);
    }
}

const char* json_type_name(const json& v) {
    switch (v.type()) {
        case json::value_t::null: return "NoneType";
        case json::value_t::boolean: return "bool";
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: return "int";
        case json::value_t::number_float: return "float";
        case json::value_t::string: return "str";
        case json::value_t::array: return "list";
        default: return "dict";
    }
}

}  // namespace

/// how answers are checked
class Validator {
public:
    bool object_only = true;           ///< json_object: one JSON object
    Val schema;                        ///< json_schema: the schema as Python decoded the request
    const Dialect* dialect = nullptr;  ///< validator_for(schema)
    std::string root;                  ///< the schema's id ("" without one)
    Patterns patterns;                 ///< the schema's patterns, compiled
};

namespace {

/// next(validator.iter_errors(value), None): (its path, its message)
std::optional<std::pair<std::string, std::string>> first_error(const Validator& v, const Val& value) {
    if (v.object_only) {
        if (value.k == K::Dict) return std::nullopt;
        return std::make_pair(std::string("/"), std::string("the answer is not a JSON object"));
    }
    Registry reg(meta());
    reg.add_root(v.root, Resource{&v.schema, v.dialect->draft});
    Patterns patterns = v.patterns;
    Run run(reg, patterns);
    const Ctx c{v.dialect, Resolver{run.intern(v.root), nullptr}};
    Sink s;
    s.messages = true;
    iter_errors(run, c, value, v.schema, s);
    if (s.count == 0) return std::nullopt;
    return std::make_pair("/" + joined(s.path, "/"), s.message);
}

}  // namespace

std::pair<json, std::shared_ptr<const Validator>> prepare_format(const json& response_format, const json& messages,
                                                                 bool with_tools) {
    if (response_format.is_null()) return {messages, nullptr};
    if (!response_format.is_object()) throw RequestError("response_format must be an object");
    const json& kind = jget(response_format, "type");
    if (kind == "text") return {messages, nullptr};
    auto validator = std::make_shared<Validator>();
    static const json object_schema = json{{"type", "object"}};
    const json* chosen = &object_schema;   // the request's own (a copy of deep JSON would recurse)
    if (kind == "json_object") {
    } else if (kind == "json_schema") {
        const json& spec = jget(response_format, "json_schema");
        if (!spec.is_object() || !jget(spec, "schema").is_object())
            throw RequestError("response_format.json_schema needs a schema object");
        const json& name = jget(spec, "name");
        if (!name.is_string() || name.get_ref<const std::string&>().empty())
            throw RequestError("response_format.json_schema needs a name");
        if (spec.contains("strict") && !jget(spec, "strict").is_boolean())
            throw RequestError("response_format.json_schema.strict must be boolean");
        chosen = &jget(spec, "schema");
        std::vector<std::string> refs;
        int64_t visits = 0;
        if (!only_objects(*chosen, *chosen, refs, 1, visits))
            throw RequestError("response_format schema must accept only JSON objects at its root "
                               "(type object, or anyOf/oneOf of object schemas)");
        validator->object_only = false;
    } else {
        throw RequestError("response_format.type must be text, json_object or json_schema");
    }
    const json& schema = *chosen;
    check_refs(schema);
    if (!validator->object_only) {
        try {
            validator->schema = from_json(schema);
            validator->dialect = dialect_for(validator->schema, &meta().dialects[(int) Draft::D2020]);
            validator->patterns.add_from(validator->schema);
            if (const std::optional<std::string> error =
                    check_schema(validator->schema, *validator->dialect, validator->patterns))
                throw RequestError("invalid response_format schema: " + *error);
            validator->root = resource_id(validator->schema, validator->dialect->draft).value_or("");
        } catch (const PyError& e) {   // a ValueError is a 400, and so is a schema too deep; the others are 500s
            if (e.type == "ValueError" || e.type == "RecursionError") throw RequestError(e.msg);
            throw std::runtime_error(e.msg);
        }
    }
    finite_floats(schema);
    std::string directive =
        "OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
        "No Markdown, headings, code fences, commentary, or text outside the JSON. "
        "Use every required field, correct types, and only allowed fields. "
        "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
        py_dumps(schema, false, true);
    if (with_tools)   // /v1/responses (#782): the schema is for the final answer; a turn that calls a tool is not one
        directive += "\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is "
                     "what you write once you are done with the tools.";
    json out = json::array();
    if (messages.is_array())
        for (const auto& m : messages) out.push_back(m);
    if (!out.empty() && jget(out[0], "role") == "system") {
        json& first = out[0];
        const json content = jget(first, "content");
        if (content.is_array() && !content.empty()) {
            json parts = content;
            parts.push_back(json{{"type", "text"}, {"text", directive}});
            first["content"] = std::move(parts);
        } else {
            std::string text;
            if (truthy(content)) {
                if (!content.is_string())
                    throw std::runtime_error(std::string("unsupported operand type(s) for +: '") +
                                             json_type_name(content) + "' and 'str'");
                text = content.get<std::string>();
            }
            first["content"] = text + "\n\n" + directive;
        }
    } else {
        out.insert(out.begin(), json{{"role", "system"}, {"content", directive}});
    }
    return {out, validator};
}

std::string validated_json(const std::string& text, const Validator* validator, const std::string& finish) {
    if (finish != "stop")
        throw StructuredOutputError("structured output was incomplete (finish_reason=" + finish +
                                    "); increase the output budget");
    Val value;
    std::string canonical;
    try {
        value = Decoder(extract_json(sanitized(text))).decode();
        canonical = dumps(value);
    } catch (const PyError& e) {
        throw StructuredOutputError("model did not return valid JSON: " + e.msg);
    }
    std::optional<std::pair<std::string, std::string>> error;
    try {
        if (validator == nullptr) raise("AttributeError", "'NoneType' object has no attribute 'iter_errors'");
        error = first_error(*validator, value);
    } catch (const PyError& e) {
        throw StructuredOutputError("could not validate structured output: " + e.msg);
    }
    if (error) throw StructuredOutputError("model output failed the JSON Schema at " + error->first + ": " + error->second);
    return canonical;
}

}  // namespace lamina
