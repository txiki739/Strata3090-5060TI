// include/lamina/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in the arena.
//
// The canonical layout (the default, before a pack's is read) has one blob size for every layer, `BLOB`, in the
// Lamina Q2_0 expert form.  The pack computed from a model's GGUF (core::PackImage) keeps each expert's raw GGUF
// slices, so the blob size and the formats change from layer to layer; its `native_experts.txt` says how.
// Everything that touches an expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window -
// asks this table.
#pragma once

#include "lamina/kernels/cpu/expert.hpp"
#include "lamina/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace lamina::kernels::cpu {

struct ExpertLayout {
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    std::vector<NativeFmt> fmt;           ///< per layer (native packs)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    int version = 0;                      ///< native_experts.txt's header version (0 = none given)
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< every layer's blobs

    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; LAMINA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// Whether this CPU (and its OS) runs the AVX2 kernels (AVX, AVX2, FMA, F16C): the floor of every expert kernel
/// (q2_avx2.cpp, iq_avx2.cpp, and ggml-cpu in the portable build).  LAMINA_FORCE_AVX2 does not change it.
bool cpu_avx2_ok();
/// Whether this CPU (and its OS) runs AVX (Sandy Bridge and newer): the AVX1 router dot (kq_avx1.cpp) and ggml-cpu in
/// a LAMINA_ISA_FLOOR=avx build.  The other expert rows then run on ggml-cpu (experimental, slow).
bool cpu_avx1_ok();
/// LAMINA_FORCE_ISA=avx2|avx|sse caps the probes above (3 = no cap, 2 = AVX2, 1 = AVX, 0 = SSE4.2), so the older-CPU
/// paths run on any PC: tests and the experimental builds' checks.  ggml-cpu keeps what it was compiled for.
int cpu_isa_cap();
/// The experimental older-CPU build this engine is ("avx", "sse4.2"), or "" for the normal build (AVX2 floor).
const char* isa_floor_build();
/// LAMINA_IQ256_GATHER, the AVX-2 i-quant kernels' gathered grid decode (iq_avx2.cpp): -1 unset (auto, see
/// cpu_gather_fast_here), 0 the scalar decode on every core, 1 the gathered one on every core.
int iq256_gather_setting();
/// Whether this CPU's performance cores gather the IQ grid entries faster than they assemble them from scalar loads:
/// an Intel CPU with AVX-VNNI, i.e. Alder Lake / Sapphire Rapids or newer, and not one of the E-core-only parts.
/// The older Intel cores gather slowly (Haswell, Broadwell) or under the Downfall (GDS) microcode (Skylake to Tiger
/// Lake), AMD Zen 2/3 gather slowly, and Zen 4/5 run the AVX-512 kernels.  LAMINA_FORCE_ISA answers no.
bool cpu_gather_fast();
/// cpu_gather_fast() and the CALLING THREAD runs on a performance core: CPUID 1Ah core type 40h on a hybrid CPU (its
/// E-cores, 20h, gather slower than they assemble).  Probed once per thread: the pool pins each worker to one core.
bool cpu_gather_fast_here();
/// Whether the AVX-2 expert kernels take their AVX-VNNI forms (vpdpwssd / vpdpbusd: Alder Lake, Sapphire Rapids and
/// later, P- and E-cores alike; the same integer sums): cpu_avx2_ok() and CPUID 7.1:EAX[4], not capped by
/// LAMINA_FORCE_ISA, and not an older-CPU build (LAMINA_ISA_FLOOR, which stays on its floor and AVX2).
/// LAMINA_NO_AVXVNNI=1 answers no.
bool cpu_avxvnni_ok();
/// Whether a native pack's layer of this ggml type runs on Lamina's own Q2_0 kernels (q2_rows_any's AVX-512, AVX2 and
/// AVX rungs).  On a CPU without AVX (an older-CPU build: ggml-cpu compiled for its SSE; LAMINA_FORCE_ISA=sse) a Q2_0
/// layer takes ggml-cpu's own Q2_0 vec_dot like every other native type: slower, but it runs.
inline bool q2_native_kernels(int type) { return type == 42 && cpu_avx1_ok(); }
/// The CPU's brand string (CPUID 0x80000002..4), for messages; "unknown" when it has none.
std::string cpu_name();
/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (canonical Q2_0 until `expert_layout_load_text` reads a pack's).
const ExpertLayout& expert_layout();
/// Reads the native_experts.txt text of the pack computed from the model's GGUF (core::PackImage) into the
/// process-wide layout.  Versions up to kExpertLayoutVersion are read; a newer one is refused.
bool expert_layout_load_text(const std::string& native_experts, int64_t n_layers, int64_t n_expert,
                             std::string& err);
/// The newest native_experts.txt this engine reads: v3, each layer's expert formats, offset and blob size.
inline constexpr int kExpertLayoutVersion = 3;

}  // namespace lamina::kernels::cpu
