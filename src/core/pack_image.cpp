// src/core/pack_image.cpp - the native pack, computed from the model's GGUF at load.  See the header.
//
// In order: the shards and their tensors' sizes, the expert table (native_experts.txt), then the index rows and
// dense.bin's pieces with their conversions.
#include "lamina/core/pack_image.hpp"
#include "lamina/kernels/cpu/expert.hpp"
#include "lamina/kernels/cpu/tq_cpu.hpp"

#include "lamina/artifact/gguf_reader.hpp"
#include "lamina/artifact/gguf_split.hpp"
#include "lamina/kernels/bf16_bits.hpp"
#include "lamina/kernels/f16_bits.hpp"

#if defined(LAMINA_NATIVE_EXPERTS)
#include "ggml.h"
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace lamina::core {
namespace {

constexpr uint64_t kAlign = 64;   // ALIGN: every dense.bin piece and every arena row starts on 64 B
constexpr uint32_t kF32 = 0, kF16 = 1, kBf16 = 30, kNvfp4 = 40;
// elements converted at a time by `read`: a multiple of every block size, so only a range's ends split a block
constexpr uint64_t kChunk = 1ull << 20;

/// Elements and bytes per block (the reader's block_geometry): false for a type it cannot size.
bool block_of(uint32_t type, uint64_t& elems, uint64_t& bytes) {
    int be = 0, bb = 0;
    if (!lamina::block_geometry(type, be, bb)) return false;
    elems = (uint64_t) be;
    bytes = (uint64_t) bb;
    return true;
}

/// A tensor's bytes: false for a type the pack cannot size or an element count that is not whole blocks.
bool expected_bytes(const TensorInfo& t, uint64_t& out) {
    uint64_t be = 0, bb = 0;
    if (!block_of(t.type, be, bb)) return false;
    const uint64_t n = t.elements();
    if (n % be) return false;
    out = n / be * bb;
    return true;
}

/// The GGML type names (tools/gguf_reader.py's GGML_TYPES), for the messages.
std::string type_str(uint32_t t) {
    static const char* const names[] = {
        "F32",     "F16",      "Q4_0",     "Q4_1",       "Q4_2",       "Q4_3",       "Q5_0",   "Q5_1",
        "Q8_0",    "Q8_1",     "Q2_K",     "Q3_K",       "Q4_K",       "Q5_K",       "Q6_K",   "Q8_K",
        "IQ2_XXS", "IQ2_XS",   "IQ3_XXS",  "IQ1_S",      "IQ4_NL",     "IQ3_S",      "IQ2_S",  "IQ4_XS",
        "I8",      "I16",      "I32",      "I64",        "F64",        "IQ1_M",      "BF16",   "Q4_0_4_4",
        "Q4_0_4_8", "Q4_0_8_8", "TQ1_0",   "TQ2_0",      "IQ4_NL_4_4", "IQ4_NL_4_8", "IQ4_NL_8_8", "MXFP4",
        "NVFP4",   "Q1_0",     "Q2_0",
    };
    if (t < sizeof names / sizeof names[0]) return names[t];
    switch (t) {
        case 143: return "PTQ1_0";
        case 144: return "TQ2_T";
        case 145: return "TQK6";
        case 146: return "TQK7";
        case 147: return "TQP5";
        case 148: return "TQP6";
        case 149: return "TQP7";
        default: return "type" + std::to_string(t);
    }
}

bool is_float(uint32_t t) { return t == kF32 || t == kF16 || t == kBf16; }

template <size_t N>
bool one_of(uint32_t t, const uint32_t (&set)[N]) {
    return std::find(set, set + N, t) != set + N;
}

bool starts_with(const std::string& s, const char* p) { return s.compare(0, std::strlen(p), p) == 0; }
bool ends_with(const std::string& s, const char* p) {
    const size_t n = std::strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

/// is_expert: the routed experts, which native_experts.txt describes and the engine reads from the GGUF.
bool is_expert(const std::string& name) { return starts_with(name, "blk.") && ends_with(name, "_exps.weight"); }

/// prism.hadamard.*, what a Hadamard-folded file says of its experts (hadamard.hpp): into `h` (block 0 without the metadata), or
/// false with the reason.  What this engine cannot apply is refused: a folded weight run unrotated computes plausible garbage.
bool read_hadamard(const lamina::GgufFile& meta, int64_t n_layers, lamina::kernels::cpu::HadamardSpec& h, std::string& err) {
    using lamina::MetaType;
    const auto text = [&](const char* key) -> std::string {
        const lamina::MetaValue* v = meta.get(key);
        return v != nullptr && v->type == MetaType::STRING ? v->s : std::string();
    };
    const lamina::MetaValue* ver = meta.get("prism.hadamard.version");
    if (ver == nullptr) return true;
    if (ver->num() != 1.0) {
        err = "prism.hadamard: an unknown version";
        return false;
    }
    const lamina::MetaValue* blk = meta.get("prism.hadamard.block_size");
    const int block = blk != nullptr ? (int) blk->num() : 0;
    if (block != 128) {   // the kernels' registers and warps are cut for it
        err = "prism.hadamard: a block of " + std::to_string(block) + " values; this engine rotates 128";
        return false;
    }
    if (text("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard" ||
        text("prism.hadamard.axis") != "input-last-dimension") {
        err = "prism.hadamard: an unknown transform or axis";
        return false;
    }
    if (meta.get("prism.hadamard.inverse_weight_names") != nullptr) {
        err = "prism.hadamard: a rotated token embedding, which this engine does not read";
        return false;
    }
    const std::string mode = text("prism.hadamard.sign_mode");
    if (mode != "identity" && mode != "explicit") {
        err = "prism.hadamard: an unknown sign mode";
        return false;
    }
    const lamina::MetaValue* names = meta.get("prism.hadamard.weight_names");
    if (names == nullptr || names->type != MetaType::ARRAY || names->items.size() != names->count || names->count == 0) {
        err = "prism.hadamard: no list of folded weights";
        return false;
    }
    h.roles.assign((size_t) n_layers, 0);
    for (const lamina::MetaValue& n : names->items) {
        static const struct { const char* suffix; uint8_t bit; } roles[] = {
            {".ffn_gate_exps.weight", 1}, {".ffn_up_exps.weight", 2}, {".ffn_down_exps.weight", 4}};
        uint8_t bit = 0;
        for (const auto& r : roles)
            if (ends_with(n.s, r.suffix)) bit = r.bit;
        const int64_t l = starts_with(n.s, "blk.") ? std::atoll(n.s.c_str() + 4) : -1;
        if (bit == 0 || l < 0 || l >= n_layers || (h.roles[(size_t) l] & bit) != 0) {
            err = "prism.hadamard folds " + n.s + "; this engine rotates the routed experts' inputs only, once each";
            return false;
        }
        h.roles[(size_t) l] |= bit;
    }
    for (int64_t l = 0; l < n_layers; ++l)
        if (((h.roles[(size_t) l] & 1) != 0) != ((h.roles[(size_t) l] & 2) != 0)) {
            err = "layer " + std::to_string(l) + ": prism.hadamard folds only one of gate and up, which share an input";
            return false;
        }
    if (mode == "explicit") {
        const lamina::MetaValue* sw = meta.get("prism.hadamard.sign_widths");
        const lamina::MetaValue* sv = meta.get("prism.hadamard.sign_values");
        if (sw == nullptr || sv == nullptr || sw->type != MetaType::ARRAY || sv->type != MetaType::ARRAY ||
            sv->items.size() != sv->count) {
            err = "prism.hadamard: no sign vectors";
            return false;
        }
        uint64_t at = 0;
        for (const lamina::MetaValue& w : sw->items) {
            const uint64_t width = w.u;
            if (width == 0 || width % (uint64_t) block != 0 || at + width > sv->count || h.signs.count((int) width) != 0) {
                err = "prism.hadamard: sign widths and values that do not fit";
                return false;
            }
            std::vector<float>& v = h.signs[(int) width];
            v.resize((size_t) width);
            for (uint64_t i = 0; i < width; ++i) {
                const int64_t s = (int64_t) sv->items[(size_t) (at + i)].u;
                if (s != 1 && s != -1) {
                    err = "prism.hadamard: a sign that is not +1 or -1";
                    return false;
                }
                v[(size_t) i] = (float) s;
            }
            at += width;
        }
        if (at != sv->count) {
            err = "prism.hadamard: sign widths and values that do not fit";
            return false;
        }
    }
    h.block = block;
    return true;
}

/// FORM: the float form the engine reads a pack tensor in (name without `blk.N.`); -1 = none, served from the
/// GGUF when quantized, written as stored when a float.
int form_of(const std::string& name) {
    struct F {
        const char* suffix;
        uint32_t type;
    };
    static const F forms[] = {
        {"ffn_gate_inp.weight", kBf16},   {"ffn_gate_inp_shexp.weight", kBf16}, {"hc_attn_down.weight", kBf16},
        {"hc_attn_up.weight", kBf16},     {"hc_attn_inject.weight", kBf16},     {"hc_ffn_down.weight", kBf16},
        {"hc_ffn_up.weight", kBf16},      {"hc_ffn_inject.weight", kBf16},      {"output_hc_down.weight", kBf16},
        {"output_hc_up.weight", kBf16},   {"indexer.k_proj.weight", kBf16},     {"indexer.q_proj.weight", kBf16},
        {"ssm_alpha.weight", kBf16},      {"ssm_beta.weight", kBf16},           {"ple_key.weight", kBf16},
        {"ple_value.weight", kBf16},      {"ple_conv1d.weight", kF16},          {"attn_q_norm.weight", kF32},
        {"attn_k_norm.weight", kF32},     {"hc_attn_norm.weight", kF32},        {"hc_ffn_norm.weight", kF32},
        {"output_hc_norm.weight", kF32},  {"indexer.q_norm.weight", kF32},      {"indexer.k_norm.weight", kF32},
        {"ple_norm_conv.weight", kF32},   {"ple_norm_key.weight", kF32},        {"ple_norm_query.weight", kF32},
        {"ssm_a", kF32},                  {"ssm_conv1d.weight", kF32},          {"ssm_dt.bias", kF32},
        {"ssm_norm.weight", kF32},
    };
    std::string s = name;   // re.sub(r"^blk\.\d+\.", "", name)
    if (starts_with(name, "blk.")) {
        size_t i = 4;
        while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
        if (i > 4 && i < name.size() && name[i] == '.') s = name.substr(i + 1);
    }
    for (const F& f : forms)
        if (s == f.suffix) return (int) f.type;
    return -1;
}

/// MMVQ_TYPES: NativeHead's and NativeDense's (native_mmvq_supported), the types of the dense projections and the PLE
/// key the engine serves from the GGUF (the prefill dequantizes each of them, dequant_f16)
const uint32_t kMmvqTypes[] = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29};

/// native_problem: why the engine cannot serve the quantized tensor `name` from the GGUF, or "".  EMBED_TYPES are
/// NativeEmbed's (iq_supported).
std::string native_problem(const std::string& name, uint32_t type) {
    static const uint32_t embed[] = {16, 17, 18, 20, 21, 22, 23, 19, 29, 42, 11, 12, 13, 14, 7, 8};
    static const char* const dense[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
                                        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
                                        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    bool served = false;
    if (name == "token_embd.weight") {
        if (one_of(type, embed)) return {};
        served = true;
    } else if (name == "output.weight" || name == "blk.1.ple_key.weight" ||
               (starts_with(name, "blk.") &&
                std::any_of(std::begin(dense), std::end(dense), [&](const char* d) { return ends_with(name, d); }))) {
        if (one_of(type, kMmvqTypes)) return {};
        served = true;
    }
    return served ? name + " is " + type_str(type) + ", a type the engine has no kernels for"
                  : name + " is " + type_str(type) + ": the engine reads it neither from the GGUF nor in a float form";
}

/// Whether this build dequantizes `type` (ggml's own dequantizers, the ones llama.cpp reads the type with).
bool dequantizable(uint32_t type) {
#if defined(LAMINA_NATIVE_EXPERTS)
    uint64_t be = 0, bb = 0;
    if (type >= (uint32_t) GGML_TYPE_COUNT || !block_of(type, be, bb)) return false;
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
    return tr->to_float != nullptr && (uint64_t) tr->blck_size == be && (uint64_t) tr->type_size == bb;
#else
    (void) type;
    return false;
#endif
}

/// `n` values of a `type` tensor's whole blocks at `src` as f32: floats exactly, quantized ones through ggml.
void decode(uint32_t type, const uint8_t* src, size_t n, float* out) {
    switch (type) {
        case kF32:
            std::memcpy(out, src, n * 4);
            return;
        case kF16:
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                out[i] = lamina::kernels::f32_from_f16(h);
            }
            return;
        case kBf16:
            for (size_t i = 0; i < n; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                out[i] = lamina::kernels::f32_from_bf16(h);
            }
            return;
        default:
#if defined(LAMINA_NATIVE_EXPERTS)
            ggml_get_type_traits((ggml_type) type)->to_float(src, out, (int64_t) n);
#endif
            return;
    }
}

/// `n` values in the piece's engine form at `out` (BF16 and F16 round to nearest-even, as gguf-py's quantize(BF16)
/// and numpy's astype(float16) do), or false with the refusal.
bool encode(const PackImage::Piece& p, const float* v, size_t n, uint8_t* out, std::string& err) {
    switch (p.conv) {
        case PackImage::Conv::ToBf16: {
            // the refusal is decided after the loop, so the loop vectorizes (an exit inside it took 2.2x as long)
            unsigned nonfinite = 0;
            for (size_t i = 0; i < n; ++i) {
                nonfinite |= !(std::fabs(v[i]) <= FLT_MAX);
                const uint16_t h = lamina::kernels::bf16_from_f32(v[i]);
                std::memcpy(out + i * 2, &h, 2);
            }
            if (nonfinite) {
                err = p.type == kF32 ? p.name + " is F32 with non-finite values; refused rather than converted"
                                     : p.name + ": cannot convert non-finite weights to BF16";
                return false;
            }
            return true;
        }
        case PackImage::Conv::ToF16:
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(v[i])) {
                    err = p.name + ": cannot convert non-finite weights to F16";
                    return false;
                }
                const uint16_t h = lamina::kernels::f16_from_f32(v[i]);
                if ((h & 0x7FFFu) == 0x7C00u) {
                    err = p.name + ": a value is beyond F16's range, which the engine reads it as";
                    return false;
                }
                std::memcpy(out + i * 2, &h, 2);
            }
            return true;
        case PackImage::Conv::ToF32:
            std::memcpy(out, v, n * 4);
            return true;
        case PackImage::Conv::Copy:
            break;
    }
    return false;
}

bool read_shard(const std::vector<std::string>& shards, std::vector<std::FILE*>& files, size_t s, uint64_t off,
                void* dst, size_t n, std::string& err) {
    std::FILE*& f = files[s];
    if (f == nullptr && (f = std::fopen(shards[s].c_str(), "rb")) == nullptr) {
        err = "cannot open " + shards[s];
        return false;
    }
#if defined(_WIN32)
    const bool sought = _fseeki64(f, (long long) off, SEEK_SET) == 0;
#else
    const bool sought = fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
    if (!sought) {
        err = shards[s] + ": seek failed";
        return false;
    }
    if (std::fread(dst, 1, n, f) != n) {
        err = shards[s] + ": short read";
        return false;
    }
    return true;
}

/// Bytes [rel, rel + n) of piece `p` in dense.bin into `out`: a copy, or the elements that cover them converted
/// (their source blocks read whole, a chunk at a time).
bool piece_bytes(const PackImage::Piece& p, const std::vector<std::string>& shards, std::vector<std::FILE*>& files,
                 uint64_t rel, size_t n, uint8_t* out, std::string& err) {
    if (p.conv == PackImage::Conv::Copy) return read_shard(shards, files, p.shard, p.src_off + rel, out, n, err);
    const uint64_t dsz = p.conv == PackImage::Conv::ToF32 ? 4 : 2;
    uint64_t be = 1, bb = 4;
    block_of(p.type, be, bb);
    const uint64_t e_end = (rel + n + dsz - 1) / dsz;
    std::vector<uint8_t> src, part;
    std::vector<float> vals;
    for (uint64_t c0 = rel / dsz; c0 < e_end;) {
        const uint64_t c1 = std::min<uint64_t>(e_end, (c0 / kChunk + 1) * kChunk);
        const uint64_t b0 = c0 / be, b1 = (c1 + be - 1) / be;
        src.resize((size_t) ((b1 - b0) * bb));
        vals.resize((size_t) ((b1 - b0) * be));
        if (!read_shard(shards, files, p.shard, p.src_off + b0 * bb, src.data(), src.size(), err)) return false;
        decode(p.type, src.data(), vals.size(), vals.data());
        const float* v = vals.data() + (c0 - b0 * be);
        // the bytes of elements [c0, c1) that the range asks for: all of them but at its two ends
        const uint64_t lo = std::max<uint64_t>(c0 * dsz, rel), hi = std::min<uint64_t>(c1 * dsz, rel + n);
        if (lo == c0 * dsz && hi == c1 * dsz) {
            if (!encode(p, v, (size_t) (c1 - c0), out + (lo - rel), err)) return false;
        } else {
            part.resize((size_t) ((c1 - c0) * dsz));
            if (!encode(p, v, (size_t) (c1 - c0), part.data(), err)) return false;
            std::memcpy(out + (lo - rel), part.data() + (lo - c0 * dsz), (size_t) (hi - lo));
        }
        c0 = c1;
    }
    return true;
}

std::string file_name(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace

bool PackImage::build(const std::string& gguf_first_shard, PackImage& out, std::string& err) {
    PackImage img;
    img.source_ = file_name(gguf_first_shard);
    try {
        // ---- the shards: every one (a missing one is an error), the split keys checked, each tensor sized and inside
        // its file
        img.shards_ = lamina::gguf_split_paths(gguf_first_shard);
        const lamina::GgufModel model(img.shards_);
        for (size_t s = 0; s < model.size(); ++s) {
            const GgufFile& g = model.shard(s);
            const uint64_t payload = g.file_size() - g.data_start();
            for (const TensorInfo& t : g.tensors()) {
                uint64_t size = 0;
                if (!expected_bytes(t, size) || t.offset > payload || size > payload - t.offset) {
                    err = file_name(g.path()) + ": unsupported or truncated tensor " + t.name;
                    return false;
                }
            }
        }

        // ---- the expert table first, so a model that cannot be packed is refused before the rest
        int64_t n_layers = 0;
        bool experts = false;
        for (size_t s = 0; s < model.size(); ++s)
            for (const TensorInfo& t : model.shard(s).tensors()) {
                if (!is_expert(t.name)) continue;
                const size_t dot = t.name.find('.', 4);
                const std::string num = t.name.substr(4, dot - 4);
                if (num.empty() || num.size() > 6 || num.find_first_not_of("0123456789") != std::string::npos) {
                    err = t.name + ": an expert tensor without a layer number";
                    return false;
                }
                experts = true;
                n_layers = std::max<int64_t>(n_layers, std::atoll(num.c_str()) + 1);
            }
        if (!experts) {
            err = "the model has no expert tensors (blk.N.ffn_{gate,up,down}_exps.weight)";
            return false;
        }
        // router rows = experts kept (pruned models ship fewer than 512)
        int64_t n_expert = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            const std::string name = "blk." + std::to_string(l) + ".ffn_gate_inp.weight";
            const TensorInfo* r = model.find(name);
            if (r == nullptr || r->shape.size() < 2 || r->shape[1] == 0) {
                err = "layer " + std::to_string(l) + ": missing " + name + " (the router, [n_embd, n_expert])";
                return false;
            }
            if (l == 0) n_expert = (int64_t) r->shape[1];
            if ((int64_t) r->shape[1] != n_expert) {
                err = "the routers disagree on the expert count; a per-layer pruned model cannot be packed";
                return false;
            }
        }
        static const char* const roles[3] = {"gate", "up", "down"};
        std::string lines;
        uint64_t total = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            const TensorInfo* ts[3] = {};
            std::string missing;
            for (int r = 0; r < 3; ++r) {
                const std::string name = "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight";
                if ((ts[r] = model.find(name)) == nullptr) missing += (missing.empty() ? "" : ", ") + name;
            }
            if (!missing.empty()) {
                err = "layer " + std::to_string(l) + ": missing " + missing;
                return false;
            }
            uint64_t per[3] = {};
            for (int r = 0; r < 3; ++r) {
                uint64_t bytes = 0;
                if (!expected_bytes(*ts[r], bytes) || ts[r]->shape.size() != 3 ||
                    (int64_t) ts[r]->shape[2] != n_expert) {
                    err = "layer " + std::to_string(l) + ": an expert tensor is not [*, *, " + std::to_string(n_expert) +
                          "] of whole blocks";
                    return false;
                }
                per[r] = bytes / (uint64_t) n_expert;
            }
            if (per[0] != per[1] || ts[0]->type != ts[1]->type) {
                err = "layer " + std::to_string(l) + ": gate and up differ in type";
                return false;
            }
            if (ts[0]->type == kNvfp4 || ts[2]->type == kNvfp4) {
                // the GPU expert kernels have no NVFP4, and the GGUF keeps these experts' scales out of their blobs
                err = "layer " + std::to_string(l) + ": NVFP4 experts, which this engine has no GPU kernels for";
                return false;
            }
            // a format past ggml's own only where the engine has kernels: its trellis types
            for (int r = 0; r < 3; ++r)
                if (ts[r]->type > 42 && !lamina::kernels::cpu::tq_supported((int) ts[r]->type)) {
                    err = "layer " + std::to_string(l) + ": experts in " + type_str(ts[r]->type) +
                          ", a type this engine has no kernels for";
                    return false;
                }
            const uint64_t blob = per[0] + per[1] + per[2];
            lines += std::to_string(l) + " " + std::to_string(ts[0]->type) + " " + std::to_string(ts[2]->type) + " " +
                     std::to_string(total) + " " + std::to_string(blob) + "\n";
            total += blob * (uint64_t) n_expert;
        }
        img.experts_ = "# lamina native experts v3: layer gu_type d_type offset blob_bytes (n_expert " +
                       std::to_string(n_expert) + ", total " + std::to_string(total) + "; the engine takes the "
                       "experts from " + img.source_ + " by tensor name)\n" + lines;
        if (!read_hadamard(model.meta(), n_layers, img.hadamard_, err)) return false;
        // the kernels: trellis experts rotated, and only they
        for (int64_t l = 0; l < n_layers; ++l) {
            const uint32_t gt = model.find("blk." + std::to_string(l) + ".ffn_gate_exps.weight")->type;
            const uint32_t dt = model.find("blk." + std::to_string(l) + ".ffn_down_exps.weight")->type;
            const bool tg = lamina::kernels::cpu::tq_supported((int) gt), td = lamina::kernels::cpu::tq_supported((int) dt);
            if (tg != img.hadamard_.gu(l) || td != img.hadamard_.down(l)) {
                err = "layer " + std::to_string(l) + ": " + type_str(gt) + " / " + type_str(dt) + " experts" +
                      (img.hadamard_.any() ? " folded otherwise than their formats" : " without prism.hadamard's rotation") +
                      "; this engine rotates the trellis formats, and only them";
                return false;
            }
        }
        if (img.hadamard_.any() && (img.hadamard_.signs.count((int) lamina::kernels::cpu::H) == 0 ||
                                    img.hadamard_.signs.count((int) lamina::kernels::cpu::FF) == 0)) {
            err = "prism.hadamard: this engine's rotation takes sign vectors of the experts' widths";
            return false;
        }

        // ---- the index: every non-expert tensor in the shards' order.  Quantized ones the engine reads from the
        // GGUF are rows of their shape; the floats it reads from the pack go into dense.bin in the form it reads them
        // (FORM), converted where stored otherwise - dequantized and rounded too, the only form an ordinary quant's
        // quantized projections have for this engine.
        struct Todo {
            const TensorInfo* t;
            size_t shard;
            bool native;
            int form;
        };
        std::vector<Todo> todo;
        std::vector<std::string> problems;
        for (size_t s = 0; s < model.size(); ++s)
            for (const TensorInfo& t : model.shard(s).tensors()) {
                if (is_expert(t.name) || t.name == "per_layer_token_embd.weight") continue;   // the PLE table
                if (t.shape.size() > 2) {
                    err = "tensor " + t.name + " has " + std::to_string(t.shape.size()) +
                          " dimensions; the index holds two";
                    return false;
                }
                const int form = form_of(t.name);
                const bool native =
                    !is_float(t.type) && (form < 0 || (t.name == "blk.1.ple_key.weight" && one_of(t.type, kMmvqTypes)));
                if (native) {
                    std::string why = native_problem(t.name, t.type);
                    if (!why.empty()) problems.push_back(std::move(why));
                } else if (!is_float(t.type) && !dequantizable(t.type)) {
                    problems.push_back(t.name + " is " + type_str(t.type) + ", which this engine cannot dequantize to "
                                       "the " + type_str((uint32_t) form) + " it reads");
                }
                todo.push_back({&t, s, native, form});
            }
        // refused before anything is read
        if (!problems.empty()) {
            for (size_t i = 0; i < problems.size() && i < 8; ++i) err += (i ? "; " : "") + problems[i];
            if (problems.size() > 8) err += "; ... and " + std::to_string(problems.size() - 8) + " more";
            return false;
        }
        // the rows' arena offsets (dst_off) at the alignment, the pool their sum; dense.bin's pieces at the same
        // alignment
        std::string rows;
        uint64_t at = 0, pool = 0;
        for (const Todo& d : todo) {
            const TensorInfo& t = *d.t;
            const std::string shape = std::to_string(t.shape[0]) + " " + std::to_string(t.shape.size() > 1 ? t.shape[1] : 0);
            if (d.native) {
                ++img.served_;
                rows += t.name + " 0 0 0 0 " + std::to_string(pool) + " 0 " + shape + " 8 0 32 0 0 0 0 0 0 0\n";
                continue;
            }
            // as stored when that is the form (or there is none), else converted to the form; index kinds 4 BF16,
            // 5 F16, 2 F32
            Piece p;
            p.name = t.name;
            p.shard = d.shard;
            p.type = t.type;
            expected_bytes(t, p.src_bytes);
            p.src_off = model.shard(d.shard).data_start() + t.offset;
            p.off = at;
            const uint32_t to = d.form < 0 ? t.type : (uint32_t) d.form;
            p.conv = to == t.type ? Conv::Copy : to == kBf16 ? Conv::ToBf16 : to == kF16 ? Conv::ToF16 : Conv::ToF32;
            p.bytes = p.conv == Conv::Copy ? p.src_bytes : t.elements() * (p.conv == Conv::ToF32 ? 4 : 2);
            const char kind = to == kBf16 ? '4' : to == kF16 ? '5' : '2';
            const std::string bytes = std::to_string(p.bytes);
            rows += t.name + " 0 " + kind + " " + std::to_string(p.off) + " " + bytes + " " + std::to_string(pool) +
                    " " + bytes + " " + shape + " 0 0 1 0 0 0 0 0 0 0\n";
            const uint64_t padded = (p.bytes + kAlign - 1) / kAlign * kAlign;
            at += padded;
            pool += padded;
            img.pieces_.push_back(std::move(p));
        }
        img.dense_bytes_ = at;
        img.index_ = "# lamina pack index v3 -- computed by the engine (native experts) from " + img.source_ +
                     "\n# align " + std::to_string(kAlign) + " pool " + std::to_string(pool) + " tensors " +
                     std::to_string(todo.size()) + "\n" + rows;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    out = std::move(img);
    return true;
}

bool PackImage::read(uint64_t offset, size_t n, void* dst, std::string& err) const {
    if (offset > dense_bytes_ || n > dense_bytes_ - offset) {
        err = where() + ": dense.bin bytes " + std::to_string(offset) + " + " + std::to_string(n) +
              " are past its end (" + std::to_string(dense_bytes_) + " B)";
        return false;
    }
    std::vector<std::FILE*> files(shards_.size(), nullptr);
    struct Closer {
        std::vector<std::FILE*>& f;
        ~Closer() {
            for (std::FILE* x : f)
                if (x) std::fclose(x);
        }
    } closer{files};
    uint8_t* o = (uint8_t*) dst;
    uint64_t at = offset;
    size_t left = n;
    // the first piece that ends past `offset`; between pieces (and after the last) the padding is zeros
    auto it = std::partition_point(pieces_.begin(), pieces_.end(),
                                   [&](const Piece& p) { return p.off + p.bytes <= offset; });
    while (left > 0) {
        if (it == pieces_.end() || at < it->off) {
            const size_t z = (size_t) std::min<uint64_t>(left, (it == pieces_.end() ? dense_bytes_ : it->off) - at);
            std::memset(o, 0, z);
            o += z;
            at += z;
            left -= z;
            continue;
        }
        const size_t take = (size_t) std::min<uint64_t>(left, it->bytes - (at - it->off));
        if (!piece_bytes(*it, shards_, files, at - it->off, take, o, err)) return false;
        o += take;
        at += take;
        left -= take;
        ++it;
    }
    return true;
}

}  // namespace lamina::core
