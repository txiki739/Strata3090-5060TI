// tools/vision/gguf_header.h - a GGUF's key/value header in a gguf_context without tensors, whatever its weight tensors are.
//
// ggml's reader goes through the tensor directory too and refuses a tensor type it does not know, as a Gyro quantization's
// trellis types are.  The picture reader and the server's tokenizer need the header alone.  The split keys are dropped.
#pragma once

#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace lamina_gguf {

struct Reader {
    FILE* f = nullptr;
    std::string err;
    bool take(void* dst, size_t n) {
        if (std::fread(dst, 1, n, f) != n) { err = "the GGUF header is cut short"; return false; }
        return true;
    }
    template <class T> bool num(T& v) { return take(&v, sizeof v); }
    bool str(std::string& s, uint64_t cap = (1ull << 26)) {
        uint64_t n = 0;
        if (!num(n)) return false;
        if (n > cap) { err = "a string in the GGUF header is too long"; return false; }
        s.resize((size_t) n);
        return n == 0 || take(&s[0], (size_t) n);
    }
};

inline size_t scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

inline bool dropped_key(const std::string& k) {
    return k == "split.no" || k == "split.count" || k == "split.tensors.count";
}

// The header of the GGUF open at `f`'s start: a new context (gguf_free it), or null and `err` when it holds something this
// reader does not handle.
inline gguf_context* read_header(FILE* f, std::string& err) {
    Reader r;
    r.f = f;
    char magic[4];
    uint32_t version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!r.take(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) { err = r.err.empty() ? "not a GGUF file" : r.err; return nullptr; }
    if (!r.num(version) || !r.num(n_tensors) || !r.num(n_kv)) { err = r.err; return nullptr; }
    if (version < 2 || version > 3) { err = "GGUF version " + std::to_string(version) + " is not handled"; return nullptr; }
    if (n_kv > 100000) { err = "too many keys in the GGUF header"; return nullptr; }
    gguf_context* out = gguf_init_empty();
    if (!out) { err = "out of memory"; return nullptr; }
    struct Freer { gguf_context* g; ~Freer() { if (g) gguf_free(g); } } freer{out};
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        uint32_t type = 0;
        if (!r.str(key, 1 << 16) || !r.num(type)) { err = r.err; return nullptr; }
        const bool keep = !dropped_key(key);
        if (type == GGUF_TYPE_STRING) {
            std::string v;
            if (!r.str(v)) { err = r.err; return nullptr; }
            if (keep) gguf_set_val_str(out, key.c_str(), v.c_str());
        } else if (type == GGUF_TYPE_ARRAY) {
            uint32_t et = 0;
            uint64_t n = 0;
            if (!r.num(et) || !r.num(n)) { err = r.err; return nullptr; }
            if (n > 50'000'000ull) { err = "an array in the GGUF header is too long"; return nullptr; }
            if (et == GGUF_TYPE_STRING) {
                std::vector<std::string> items((size_t) n);
                for (auto& s : items) if (!r.str(s)) { err = r.err; return nullptr; }
                if (keep) {
                    std::vector<const char*> ptrs;
                    ptrs.reserve(items.size());
                    for (const auto& s : items) ptrs.push_back(s.c_str());
                    gguf_set_arr_str(out, key.c_str(), ptrs.data(), ptrs.size());
                }
            } else if (scalar_size(et)) {
                std::vector<uint8_t> raw((size_t) (n * scalar_size(et)));
                if (!raw.empty() && !r.take(raw.data(), raw.size())) { err = r.err; return nullptr; }
                if (keep) gguf_set_arr_data(out, key.c_str(), (gguf_type) et, raw.data(), (size_t) n);
            } else { err = "an array of arrays in the GGUF header is not handled"; return nullptr; }
        } else if (scalar_size(type)) {
            uint64_t raw = 0;
            if (!r.take(&raw, scalar_size(type))) { err = r.err; return nullptr; }   // little endian, as the file
            if (!keep) continue;
            switch (type) {
                case GGUF_TYPE_UINT8: gguf_set_val_u8(out, key.c_str(), (uint8_t) raw); break;
                case GGUF_TYPE_INT8: gguf_set_val_i8(out, key.c_str(), (int8_t) raw); break;
                case GGUF_TYPE_UINT16: gguf_set_val_u16(out, key.c_str(), (uint16_t) raw); break;
                case GGUF_TYPE_INT16: gguf_set_val_i16(out, key.c_str(), (int16_t) raw); break;
                case GGUF_TYPE_UINT32: gguf_set_val_u32(out, key.c_str(), (uint32_t) raw); break;
                case GGUF_TYPE_INT32: gguf_set_val_i32(out, key.c_str(), (int32_t) raw); break;
                case GGUF_TYPE_FLOAT32: { float v; std::memcpy(&v, &raw, 4); gguf_set_val_f32(out, key.c_str(), v); break; }
                case GGUF_TYPE_BOOL: gguf_set_val_bool(out, key.c_str(), raw != 0); break;
                case GGUF_TYPE_UINT64: gguf_set_val_u64(out, key.c_str(), raw); break;
                case GGUF_TYPE_INT64: gguf_set_val_i64(out, key.c_str(), (int64_t) raw); break;
                case GGUF_TYPE_FLOAT64: { double v; std::memcpy(&v, &raw, 8); gguf_set_val_f64(out, key.c_str(), v); break; }
                default: break;
            }
        } else { err = "a value type in the GGUF header is not handled"; return nullptr; }
    }
    freer.g = nullptr;
    return out;
}

}  // namespace lamina_gguf
