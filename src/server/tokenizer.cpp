#include "tokenizer.hpp"

#include "json.hpp"

#include "gguf_header.h"
#include "src/unicode.h"   // llama.cpp's: the pre-tokenizer split (unicode_regex_split)

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <queue>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace lamina {
namespace {

// the GPT-2 byte alphabet: bytes 33..126, 161..172 and 174..255 map to themselves, the 68 others to 256 + n
std::array<uint32_t, 256> byte_to_unicode() {
    std::array<uint32_t, 256> map{};
    std::array<bool, 256> direct{};
    for (int b = 0x21; b <= 0x7E; ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) map[b] = direct[b] ? (uint32_t) b : 256 + n++;
    return map;
}

/// a token's text (byte-alphabet characters) -> its bytes; false when a character is outside the alphabet
bool alphabet_to_bytes(const std::string& text, std::string& out) {
    static const std::array<uint32_t, 256> to_cp = byte_to_unicode();
    static const std::unordered_map<uint32_t, uint8_t> to_byte = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        for (int b = 0; b < 256; ++b) m[to_cp[b]] = (uint8_t) b;
        return m;
    }();
    out.clear();
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = (unsigned char) text[i];
        uint32_t cp;
        size_t n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; }
        else { cp = c & 0x07; n = 4; }
        if (i + n > text.size()) return false;
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char) text[i + k] & 0x3F);
        const auto it = to_byte.find(cp);
        if (it == to_byte.end()) return false;
        out += (char) it->second;
        i += n;
    }
    return true;
}

FILE* open_utf8(const std::string& path) {
#if defined(_WIN32)
    const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w((size_t) std::max(n, 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), n);
    return _wfopen(w.c_str(), L"rb");
#else
    return std::fopen(path.c_str(), "rb");
#endif
}

// the pre-tokenizers this tokenizer runs, as llama.cpp's llama-vocab.cpp spells them (its splitter has a custom
// implementation for each)
const char* kQwen35Regex =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s"
    "\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

constexpr size_t kHeapMin = 64;          // longer words merge through a heap (same order as the scan)
constexpr size_t kPieceCacheMax = 200000;

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const std::string& path) {
    FILE* f = open_utf8(path);
    if (f == nullptr) throw std::runtime_error("cannot open " + path);
    std::string herr;
    gguf_context* g = lamina_gguf::read_header(f, herr);
    std::fclose(f);
    if (g == nullptr) throw std::runtime_error(path + " is not a readable GGUF file: " + herr);
    struct Free {
        gguf_context* g;
        ~Free() { gguf_free(g); }
    } free_g{g};

    auto key = [&](const char* k) { return gguf_find_key(g, k); };
    auto str = [&](const char* k) -> std::string {
        const int64_t i = key(k);
        return i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_STRING ? gguf_get_val_str(g, i) : "";
    };
    const int64_t kt = key("tokenizer.ggml.tokens"), km = key("tokenizer.ggml.merges");
    if (kt < 0 || km < 0) throw std::runtime_error(path + " carries no tokenizer (tokenizer.ggml.tokens / merges)");
    if (str("tokenizer.ggml.model") != "gpt2")
        throw std::runtime_error(path + ": expected a byte-level BPE tokenizer (model 'gpt2'), got '" +
                                 str("tokenizer.ggml.model") + "'");
    auto t = std::make_unique<Tokenizer>();
    t->pre = str("tokenizer.ggml.pre");
    if (t->pre.empty()) t->pre = "qwen35";
    if (t->pre != "qwen35" && t->pre != "qwen2" && t->pre != "qwen3")
        throw std::runtime_error(path + ": the pre-tokenizer '" + t->pre + "' is not one this server runs");
    t->regex_ = kQwen35Regex;
    t->chat_template = str("tokenizer.chat_template");
    t->architecture = str("general.architecture");
    for (int64_t i = 0, n = gguf_get_n_kv(g); i < n; ++i) {
        const std::string k = gguf_get_key(g, i);
        if (k.rfind("tokenizer.ggml.", 0) != 0 || k.size() < 9 || k.compare(k.size() - 9, 9, "_token_id") != 0) continue;
        int64_t v = -1;
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT32: v = gguf_get_val_u32(g, i); break;
            case GGUF_TYPE_INT32: v = gguf_get_val_i32(g, i); break;
            case GGUF_TYPE_UINT64: v = (int64_t) gguf_get_val_u64(g, i); break;
            case GGUF_TYPE_INT64: v = gguf_get_val_i64(g, i); break;
            default: continue;
        }
        t->special_ids[k.substr(15, k.size() - 15 - 9)] = v;
    }

    const size_t n_tok = gguf_get_arr_n(g, kt);
    std::vector<std::string> texts(n_tok);
    std::unordered_map<std::string, int32_t> ids;
    ids.reserve(n_tok * 2);
    for (size_t i = 0; i < n_tok; ++i) {
        texts[i] = gguf_get_arr_str(g, kt, i);
        ids.emplace(texts[i], (int32_t) i);
    }
    if (ids.size() != n_tok)
        throw std::runtime_error("vocabulary has duplicate tokens: " + std::to_string(n_tok) + " entries, " +
                                 std::to_string(ids.size()) + " unique");
    std::vector<int32_t> types(n_tok, 1);
    if (const int64_t ky = key("tokenizer.ggml.token_type"); ky >= 0 && gguf_get_arr_n(g, ky) == n_tok &&
                                                             gguf_get_arr_type(g, ky) == GGUF_TYPE_INT32) {
        const int32_t* d = (const int32_t*) gguf_get_arr_data(g, ky);
        types.assign(d, d + n_tok);
    }

    // each token's bytes; a literal (types 3 and 4) is its text as written
    t->bytes_.resize(n_tok);
    std::string raw;
    for (size_t i = 0; i < n_tok; ++i) {
        if (types[i] == 3 || types[i] == 4) {
            t->literal_ids_.emplace(texts[i], (int32_t) i);
            t->literal_texts_.push_back(texts[i]);
            if (types[i] == 3) t->control_.push_back(texts[i]);
            if (!texts[i].empty()) t->lits_.push_back({(int32_t) i, texts[i], types[i] == 4});
            t->max_special_len_ = std::max(t->max_special_len_, texts[i].size());
        }
        // a literal's text is in the byte alphabet as well (ASCII), so its bytes come the same way
        if (!alphabet_to_bytes(texts[i], raw)) {
            if (types[i] == 3 || types[i] == 4) raw = texts[i];
            else raw.clear();   // not decodable: such a token decodes to nothing (none in these vocabularies)
        }
        t->bytes_[i] = raw;
    }
    for (uint32_t k = 0; k < (uint32_t) t->lits_.size(); ++k)
        t->by_first_[(unsigned char) t->lits_[k].text[0]].push_back(k);
    for (auto& v : t->by_first_)
        std::stable_sort(v.begin(), v.end(), [&](uint32_t a, uint32_t b) {
            return t->lits_[a].text.size() > t->lits_[b].text.size();
        });

    const std::array<uint32_t, 256> b2u = byte_to_unicode();
    for (int b = 0; b < 256; ++b) {
        std::string ch;
        utf8_append(ch, b2u[b]);
        const auto it = ids.find(ch);
        if (it == ids.end()) throw std::runtime_error("the vocabulary lacks the byte token of " + std::to_string(b));
        t->byte_token_[b] = it->second;
    }

    const size_t n_merge = gguf_get_arr_n(g, km);
    t->merges_.reserve(n_merge * 2);
    for (size_t i = 0; i < n_merge; ++i) {
        const std::string m = gguf_get_arr_str(g, km, i);
        const size_t sp = m.find(' ');
        if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos)
            throw std::runtime_error("merge " + std::to_string(i) + " is not a pair: " + m);
        const auto a = ids.find(m.substr(0, sp)), b = ids.find(m.substr(sp + 1)),
                   c = ids.find(m.substr(0, sp) + m.substr(sp + 1));
        if (a == ids.end() || b == ids.end())
            throw std::runtime_error("merge " + std::to_string(i) + " names a token outside the vocabulary: " + m);
        if (c == ids.end())
            throw std::runtime_error("merge " + std::to_string(i) + " makes a token outside the vocabulary: " + m);
        const uint64_t k = ((uint64_t) (uint32_t) a->second << 32) | (uint32_t) b->second;
        t->merges_.emplace(k, std::make_pair((int32_t) i, c->second));   // the first (lowest) rank of a pair stays
    }
    return t;
}

int32_t Tokenizer::literal_id(std::string_view text) const {
    const auto it = literal_ids_.find(std::string(text));
    return it == literal_ids_.end() ? -1 : it->second;
}

const std::string& Tokenizer::token_bytes(int32_t id) const {
    if (id < 0 || id >= n_vocab())
        throw std::out_of_range("token id " + std::to_string(id) + " is outside the vocabulary (" +
                                std::to_string(n_vocab()) + ")");
    return bytes_[(size_t) id];
}

void Tokenizer::bpe(const std::string& piece, std::vector<int32_t>& out) const {
    const size_t n = piece.size();
    std::vector<int32_t> parts(n);
    for (size_t i = 0; i < n; ++i) parts[i] = byte_token_[(unsigned char) piece[i]];
    auto merge_of = [&](int32_t a, int32_t b) -> const std::pair<int32_t, int32_t>* {
        const auto it = merges_.find(((uint64_t) (uint32_t) a << 32) | (uint32_t) b);
        return it == merges_.end() ? nullptr : &it->second;
    };
    if (n <= kHeapMin) {
        // the scan: the lowest rank over all adjacent pairs, the leftmost of equal ranks, until none merges
        while (parts.size() > 1) {
            size_t best = SIZE_MAX;
            int32_t best_rank = 0, best_id = 0;
            for (size_t i = 0; i + 1 < parts.size(); ++i) {
                const auto* m = merge_of(parts[i], parts[i + 1]);
                if (m != nullptr && (best == SIZE_MAX || m->first < best_rank)) {
                    best = i;
                    best_rank = m->first;
                    best_id = m->second;
                }
            }
            if (best == SIZE_MAX) break;
            parts[best] = best_id;
            parts.erase(parts.begin() + (ptrdiff_t) best + 1);
        }
        out.insert(out.end(), parts.begin(), parts.end());
        return;
    }
    // the same merges in the same order: a linked list of the symbols and a heap of (rank, position) pairs, a popped
    // pair whose symbols have merged since being stale
    std::vector<int64_t> nxt(n), prv(n);
    for (size_t i = 0; i < n; ++i) {
        nxt[i] = i + 1 < n ? (int64_t) i + 1 : -1;
        prv[i] = (int64_t) i - 1;
    }
    struct Item {
        int32_t rank;
        int64_t pos;
        int32_t left, right;
        bool operator>(const Item& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (size_t i = 0; i + 1 < n; ++i)
        if (const auto* m = merge_of(parts[i], parts[i + 1])) heap.push({m->first, (int64_t) i, parts[i], parts[i + 1]});
    std::vector<uint8_t> gone(n, 0);
    while (!heap.empty()) {
        const Item it = heap.top();
        heap.pop();
        const int64_t i = it.pos, j = nxt[(size_t) i];
        if (gone[(size_t) i] || parts[(size_t) i] != it.left || j < 0 || parts[(size_t) j] != it.right) continue;
        parts[(size_t) i] = merge_of(it.left, it.right)->second;
        gone[(size_t) j] = 1;
        const int64_t k = nxt[(size_t) j];
        nxt[(size_t) i] = k;
        if (k >= 0) {
            prv[(size_t) k] = i;
            if (const auto* m = merge_of(parts[(size_t) i], parts[(size_t) k]))
                heap.push({m->first, i, parts[(size_t) i], parts[(size_t) k]});
        }
        const int64_t p = prv[(size_t) i];
        if (p >= 0)
            if (const auto* m = merge_of(parts[(size_t) p], parts[(size_t) i]))
                heap.push({m->first, p, parts[(size_t) p], parts[(size_t) i]});
    }
    for (size_t i = 0; i < n; ++i)
        if (!gone[i]) out.push_back(parts[i]);
}

void Tokenizer::encode_plain(std::string_view text, std::vector<int32_t>& out) const {
    if (text.empty()) return;
    // a piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
    for (const std::string& piece : unicode_regex_split(std::string(text), {regex_}, false)) {
        {
            std::lock_guard<std::mutex> lk(cache_mu_);
            const auto it = piece_cache_.find(piece);
            if (it != piece_cache_.end()) {
                out.insert(out.end(), it->second.begin(), it->second.end());
                continue;
            }
        }
        std::vector<int32_t> got;
        bpe(piece, got);
        out.insert(out.end(), got.begin(), got.end());
        std::lock_guard<std::mutex> lk(cache_mu_);
        if (piece_cache_.size() < kPieceCacheMax) piece_cache_.emplace(piece, std::move(got));
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special, const std::vector<Span>& plain,
                                       std::vector<Mark>* marks) const {
    std::vector<int32_t> out;
    std::vector<Span> spans(plain);
    std::sort(spans.begin(), spans.end());
    size_t next_span = 0, plain_end = 0, pos = 0, i = 0;
    // left to right, the longest literal at each position first; a match is taken whole or, when it starts inside
    // a plain span, left to the text around it (the scan goes on after it, as a regex finditer does)
    while (i < text.size()) {
        const Literal* hit = nullptr;
        for (const uint32_t k : by_first_[(unsigned char) text[i]]) {
            const Literal& lit = lits_[k];
            if (!parse_special && !lit.always) continue;
            if (text.compare(i, lit.text.size(), lit.text) == 0) {
                hit = &lit;
                break;
            }
        }
        if (hit == nullptr) {
            ++i;
            continue;
        }
        const size_t end = i + hit->text.size();
        while (next_span < spans.size() && spans[next_span].a <= i) plain_end = std::max(plain_end, spans[next_span++].b);
        if (i < plain_end) {
            i = end;
            continue;
        }
        encode_plain(text.substr(pos, i - pos), out);
        out.push_back(hit->id);
        pos = i = end;
        if (marks != nullptr) marks->push_back({pos, out.size()});
    }
    encode_plain(text.substr(pos), out);
    return out;
}

std::string Utf8Decoder::push(std::string_view bytes, bool final) {
    // Python's UTF-8 decoder with errors="replace": each maximal invalid subpart becomes one U+FFFD
    static const char kRepl[] = "\xEF\xBF\xBD";
    std::string in = held_;
    in.append(bytes);
    held_.clear();
    std::string out;
    size_t i = 0;
    const size_t n = in.size();
    while (i < n) {
        const unsigned char c = (unsigned char) in[i];
        if (c < 0x80) {
            out += (char) c;
            ++i;
            continue;
        }
        size_t need;
        unsigned char lo = 0x80, hi = 0xBF;   // the second byte's range
        if (c >= 0xC2 && c <= 0xDF) need = 2;
        else if (c >= 0xE0 && c <= 0xEF) {
            need = 3;
            if (c == 0xE0) lo = 0xA0;
            if (c == 0xED) hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 4;
            if (c == 0xF0) lo = 0x90;
            if (c == 0xF4) hi = 0x8F;
        } else {
            out += kRepl;
            ++i;
            continue;
        }
        size_t k = 1;
        bool bad = false;
        for (; k < need; ++k) {
            if (i + k >= n) break;
            const unsigned char d = (unsigned char) in[i + k];
            if (k == 1 ? (d < lo || d > hi) : (d < 0x80 || d > 0xBF)) {
                bad = true;
                break;
            }
        }
        if (bad) {   // the valid prefix so far is one replacement; the byte that broke it starts afresh
            out += kRepl;
            i += k;
            continue;
        }
        if (k < need) {   // the input ends inside a character
            if (final) {
                out += kRepl;
            } else {
                held_ = in.substr(i);
            }
            break;
        }
        out.append(in, i, need);
        i += need;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string raw;
    for (const int32_t id : ids) raw += token_bytes(id);
    Utf8Decoder d;
    return d.push(raw, true);
}

// ---------------------------------------------------------------------------------------------- PromptEncoder
size_t common_prefix_len(std::string_view a, std::string_view b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i + 8 <= n && std::memcmp(a.data() + i, b.data() + i, 8) == 0) i += 8;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

size_t plain_agree_until(const std::vector<Span>& a, const std::vector<Span>& b, size_t limit) {
    auto clip = [&](const std::vector<Span>& s) {
        std::vector<Span> out;
        for (const Span& x : s)
            if (x.a < limit && std::min(x.b, limit) > x.a) out.push_back({x.a, std::min(x.b, limit)});
        return out;
    };
    const std::vector<Span> ca = clip(a), cb = clip(b);
    for (size_t i = 0; i < std::min(ca.size(), cb.size()); ++i)
        if (!(ca[i] == cb[i])) return ca[i].a != cb[i].a ? std::min(ca[i].a, cb[i].a) : std::min(ca[i].b, cb[i].b);
    if (ca.size() != cb.size()) return (ca.size() > cb.size() ? ca : cb)[std::min(ca.size(), cb.size())].a;
    return limit;
}

bool PromptEncoder::straddles(const std::string& text, size_t c) const {
    // a literal that starts before `c` and ends after it: a cut there would split it
    const size_t from = c + 1 > tok_.max_special_len() ? c + 1 - tok_.max_special_len() : 0;
    for (size_t p = text.find('<', from); p != std::string::npos && p < c; p = text.find('<', p + 1))
        for (const std::string& s : tok_.literals())
            if (s.size() > c - p && text.compare(p, s.size(), s) == 0) return true;
    return false;
}

std::vector<int32_t> PromptEncoder::encode(const std::string& text, std::vector<Span> plain, bool store) {
    std::sort(plain.begin(), plain.end());
    std::vector<std::shared_ptr<const Entry>> entries;
    {
        std::lock_guard<std::mutex> lk(mu_);
        entries = entries_;
    }
    const size_t margin = tok_.max_special_len() - 1;
    std::shared_ptr<const Entry> src;
    int64_t cut_k = -1;
    for (const auto& e : entries) {
        const size_t same = plain_agree_until(e->plain, plain, common_prefix_len(e->text, text));
        // the last boundary c with c <= same - margin
        int64_t k = same >= margin ? (int64_t) (std::upper_bound(e->ends.begin(), e->ends.end(), same - margin) -
                                                e->ends.begin()) - 1
                                   : -1;
        if (e->reply_at != SIZE_MAX && same >= e->reply_at) {   // a reply's token boundaries: no margin
            int64_t r = (int64_t) (std::upper_bound(e->ends.begin(), e->ends.end(), same) - e->ends.begin()) - 1;
            while (r > k && e->ends[(size_t) r] >= e->reply_at && straddles(text, e->ends[(size_t) r])) --r;
            if (r > k && e->ends[(size_t) r] >= e->reply_at) k = r;
        }
        if (k >= 0 && (src == nullptr || e->ends[(size_t) k] > src->ends[(size_t) cut_k])) {
            src = e;
            cut_k = k;
        }
    }
    auto entry = std::make_shared<Entry>();
    entry->text = text;
    entry->plain = plain;
    std::vector<Mark> marks;
    if (src == nullptr) {
        entry->ids = tok_.encode(text, true, plain, &marks);
        for (const Mark& m : marks) {
            entry->ends.push_back(m.end);
            entry->counts.push_back(m.count);
        }
        last_reused_ = 0;
    } else {
        const size_t c = src->ends[(size_t) cut_k], n = src->counts[(size_t) cut_k];
        std::vector<Span> rest;
        for (const Span& s : plain)
            if (s.b > c) rest.push_back({s.a > c ? s.a - c : 0, s.b - c});
        const std::vector<int32_t> tail = tok_.encode(std::string_view(text).substr(c), true, rest, &marks);
        entry->ids.assign(src->ids.begin(), src->ids.begin() + (ptrdiff_t) n);
        entry->ids.insert(entry->ids.end(), tail.begin(), tail.end());
        entry->ends.assign(src->ends.begin(), src->ends.begin() + cut_k + 1);
        entry->counts.assign(src->counts.begin(), src->counts.begin() + cut_k + 1);
        for (const Mark& m : marks) {
            entry->ends.push_back(c + m.end);
            entry->counts.push_back(n + m.count);
        }
        last_reused_ = c;
    }
    std::vector<int32_t> ids = entry->ids;
    if (!store) return ids;
    // every prompt starts with the template's header, so only the entry whose text is mostly reused is replaced
    const Entry* old = src != nullptr && 2 * src->ends[(size_t) cut_k] > src->text.size() ? src.get() : nullptr;
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::shared_ptr<const Entry>> next{entry};
    for (const auto& e : entries_)
        if (e.get() != old && next.size() < keep_) next.push_back(e);
    entries_ = std::move(next);
    return ids;
}

bool PromptEncoder::remember_reply(const std::vector<int32_t>& prompt_ids, const std::vector<int32_t>& gen_ids) {
    if (keep_ == 0 || gen_ids.empty()) return false;
    std::vector<std::shared_ptr<const Entry>> entries;
    {
        std::lock_guard<std::mutex> lk(mu_);
        entries = entries_;
    }
    std::shared_ptr<const Entry> base;
    for (const auto& e : entries)
        if (e->ids == prompt_ids) {
            base = e;
            break;
        }
    if (base == nullptr) return false;
    auto entry = std::make_shared<Entry>();
    entry->plain = base->plain;
    entry->ends = base->ends;
    entry->counts = base->counts;
    entry->reply_at = base->text.size();
    std::string text = base->text;
    Utf8Decoder dec;
    for (size_t j = 0; j < gen_ids.size(); ++j) {
        text += dec.push(tok_.token_bytes(gen_ids[j]));
        if (!dec.pending()) {   // no character split across this boundary
            entry->ends.push_back(text.size());
            entry->counts.push_back(prompt_ids.size() + j + 1);
        }
    }
    entry->text = std::move(text);
    entry->ids = prompt_ids;
    entry->ids.insert(entry->ids.end(), gen_ids.begin(), gen_ids.end());
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::shared_ptr<const Entry>> next{entry};
    for (const auto& e : entries_)
        if (e != base && e->text != entry->text && next.size() < keep_) next.push_back(e);
    entries_ = std::move(next);
    return true;
}

void PromptEncoder::forget(const std::vector<int32_t>& ids) {
    std::lock_guard<std::mutex> lk(mu_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e->ids == ids; }),
                   entries_.end());
}

}  // namespace lamina
