// include/lamina/core/pack_image.hpp - the native pack, computed from the model's GGUF at load.
//
// The pack is what the engine reads besides the GGUF itself: index.txt (one row per non-expert tensor: a float
// tensor's place in dense.bin, or the shape of a quantized one the engine serves from the GGUF), dense.bin (the float
// tensors in the form the engine reads them: as stored, or converted per the FORM table) and native_experts.txt (each
// layer's expert formats and blob size).  All three follow from the GGUF's tensor directory and its float tensors:
// the two texts are made here, and dense.bin as a list of pieces (a tensor's bytes in its shard and the conversion
// they take) whose bytes `read` makes on demand - nothing is written to disk.
//
// native_experts.txt is v3 without GGUF offsets (the engine finds each expert tensor by name, so a layer split across
// shards needs no shard column); a float tensor stored in another form than the engine reads is converted, rounding
// to nearest-even (an ordinary quant's Q8_0 projections have no other form the engine reads; GSQ-RCO files need
// none), and an F32 PLE conv narrowed to F16 here (index kind 5: the values the loader's kind-3 narrowing would make).
//
// CPU only, and building reads the shards' headers only; the float tensors are read when `read` asks for them.
#pragma once

#include "lamina/core/weights.hpp"
#include "lamina/kernels/cpu/hadamard.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lamina::core {

class PackImage : public PackContents {
public:
    /// How a piece's bytes are made from its tensor's bytes in the GGUF.
    enum class Conv : uint8_t {
        Copy,     ///< as stored
        ToBf16,   ///< F32 / F16 / a quantized type -> BF16, nearest-even (refuses non-finite values)
        ToF16,    ///< F32 / BF16 / a quantized type -> F16, nearest-even (refuses non-finite and out-of-range values)
        ToF32,    ///< F16 / BF16 widened (exact), or a quantized type's dequantized values
    };
    /// One float tensor in dense.bin: its index row's bytes, padded with zeros to the next 64 B.
    struct Piece {
        std::string name;
        size_t shard = 0;         ///< into shards()
        uint32_t type = 0;        ///< the tensor's GGML type in the GGUF
        Conv conv = Conv::Copy;
        uint64_t src_off = 0;     ///< absolute, in the shard's file
        uint64_t src_bytes = 0;
        uint64_t off = 0;         ///< in dense.bin (the row's src_off)
        uint64_t bytes = 0;       ///< in dense.bin (the row's src_bytes)
        /// A quantized tensor in the GGUF (not F32 0, F16 1 or BF16 30): its values are dequantized and rounded to
        /// the form.
        bool dequantized() const { return conv != Conv::Copy && type != 0 && type != 1 && type != 30; }
    };

    /// The pack of the model whose first shard is `gguf_first_shard` (the other shards are found by name; the index
    /// and expert headers name this file).  False with `err` for a model the engine cannot run: a tensor it has no
    /// form or kernels for, NVFP4 or Hadamard-folded experts, a truncated file.
    static bool build(const std::string& gguf_first_shard, PackImage& out, std::string& err);

    const std::string& index_text() const override { return index_; }
    const std::string& native_experts_text() const { return experts_; }
    /// prism.hadamard: the folded experts' rotation (block 0: none), for cpu::hadamard_set before the layout is read
    const lamina::kernels::cpu::HadamardSpec& hadamard() const { return hadamard_; }
    /// dense.bin's bytes [offset, offset + n): the pieces' bytes converted as they are read, the padding zeros.
    /// Opens the shards it needs for each call, so concurrent calls are safe.
    bool read(uint64_t offset, size_t n, void* dst, std::string& err) const override;
    std::string where() const override { return "the pack computed from " + source_; }

    const std::vector<Piece>& pieces() const { return pieces_; }
    const std::vector<std::string>& shards() const { return shards_; }
    uint64_t dense_bytes() const { return dense_bytes_; }
    /// index.txt's rows: the tensors served from the GGUF (shape only) and those in dense.bin.
    size_t served() const { return served_; }

private:
    std::string source_;              ///< the first shard's file name
    std::vector<std::string> shards_;
    lamina::kernels::cpu::HadamardSpec hadamard_;
    std::string index_, experts_;
    std::vector<Piece> pieces_;       ///< in dense.bin's order
    uint64_t dense_bytes_ = 0;
    size_t served_ = 0;
};

}  // namespace lamina::core
