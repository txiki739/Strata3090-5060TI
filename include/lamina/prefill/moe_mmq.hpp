// include/lamina/prefill/moe_mmq.hpp - prompt-speed plan step 2b: the prompt path's experts through llama.cpp's MMQ
// kernels (ggml-cuda mmq.cuh, MIT): the weights stay quantized and the activations are rounded to q8_1, the
// products run on int8 tensor cores.  The dequantize-to-FP16 + cuBLAS path wrote ~10 MB of FP16 per expert and
// multiplied in FP16; this reads the ~1.4-2 MB expert once.  A group of experts is multiplied in one launch per
// product, each expert's matrices read where its blob lies (Product::tab), or gathered into one buffer (`gather_*`).
#pragma once

#include <cstddef>
#include <cstdint>

namespace lamina::prefill::mmq {

/// This build has the MMQ path (the ggml sources were available to the build).
bool built();
/// MMQ covers this ggml type (the i-quants and Q2_0 of ISTA-DASLab's files, the K-quants, Q5_0, Q5_1 and Q8_0 of
/// unsloth's; IQ1_M is not covered).
bool supported(int ggml_type);
/// `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of `w_rows` rows
/// that fits the card's shared memory - the test its tile choice makes, which aborts the process ("J_best=0") when
/// nothing fits.  false (said once per type) keeps that product off MMQ (the Qwen3.6 path asks it).
bool fits(int ggml_type, int64_t w_rows);
/// Bytes of one expert's gate+up ([2*n_ff, n_embd]) or down ([n_embd, n_ff]) weights in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);
/// The same from FP16 rows (`ld` halves apart, cols a multiple of 4): llama.cpp's quantizer reading FP16, the bytes
/// of quantize() over those rows in FP32.
/// `signs` (`cols` floats on this device, Hadamard-folded experts: hadamard.hpp): each 128 values rotated first, signs and all.
void quantize_f16(const uint16_t* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
                  void* stream, const float* signs = nullptr);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).  With `tab` (n host
/// pointers) expert e's matrix is read at tab[e] instead (`w` and `expert_bytes` unused): only where tab_takes says
/// so, and the same sums in the same order, so the same bits.
struct Product {
    const void* w = nullptr;
    const void* const* tab = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
};

/// Product::tab can read [w_rows, w_cols] matrices of `ggml_type` in place: this build has its pointer-table MMQ
/// for the type (moe_mmq_tab.cuh: the gate/up and down types of the configs' files) and the rows are a multiple of
/// 128.  The matrix needs tab_overread bytes after it.
bool tab_takes(int ggml_type, int64_t w_rows, int64_t w_cols);
/// Bytes past a matrix of w_cols columns that MMQ's k loop reads on the current device: its steps (256 values for
/// the configs' types) run past each row's end, the last row's past the matrix (down's 640 columns: 128 values,
/// Q8_0 136 bytes).  They meet zero activations, so any finite values do (zero_after).  0 for gate/up's 2,560.
size_t tab_overread(int ggml_type, int64_t w_cols);
/// Zeroes `bytes` (a multiple of 4) at each of the n <= kGatherGroupMax addresses (4-byte aligned): the bytes a
/// product reads past each matrix it reads in place.
void zero_after(uint8_t* const* at, int n, size_t bytes, void* stream);

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);
    /// A dense product: dst[t, r] (`ld_dst` floats a row) = row r of `w` ([rows, cols] of `ggml_type`) . row t of `xq`
    /// (quantize / quantize_f16 of `T` rows of `cols`).
    void run_dense(const void* w, int ggml_type, int64_t rows, int64_t cols, const void* xq, int64_t T, float* dst,
                   int64_t ld_dst, void* stream);

private:
    void* ctx_ = nullptr;
};

/// A GGUF-native expert (gate at `gate`, up at `up`, down at `down`, each its GGUF rows) into a group buffer's
/// slot: gate rows then up rows at `gu_dst`, down at `d_dst`.
void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream);
/// gather_native for up to kGatherGroupMax experts in one launch: expert q's blob (`blob[q]`: gate at +0, up at +up_off,
/// down at +down_off) to gu_dst + q * gu_stride and d_dst + q * d_stride, the bytes gather_native would put there.
/// Every pointer, offset and size 16-byte aligned, else false (nothing launched).
constexpr int kGatherGroupMax = 16;
bool gather_native_group(const uint8_t* const* blob, int n, size_t up_off, size_t gu_half_bytes, size_t down_off,
                         size_t d_bytes, void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream);

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1: the Lamina pack) or split
/// (gate k, up n_ff + k: GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);

/// swiglu() and quantize() of its h (n_ff a multiple of 4) in one kernel, their bytes: H's q8_1 rows for the down
/// product without the FP32 rows between.
void swiglu_quantize(const float* gu, void* xq, int ggml_type, int64_t rows, int64_t n_ff, bool interleaved, void* stream, const float* signs = nullptr);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

}  // namespace lamina::prefill::mmq
