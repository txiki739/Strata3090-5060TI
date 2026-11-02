// include/lamina/kernels/cpu/hadamard.hpp - the activation side of Hadamard-folded experts.
//
// Gyro-L stores its routed experts in a block-Hadamard-rotated basis: the encoder replaced each expert weight W by
// W' = W * diag(s) * H_b^T per block of b input columns, H_b the normalized Sylvester-Walsh-Hadamard matrix (entry
// (-1)^popcount(r & c) / sqrt(b)) and s a +/-1 sign vector of the input width.  The GGUF says so in its `prism.hadamard.*`
// metadata.  H_b being orthonormal and symmetric, W x = W' (H_b (s * x)): every activation that enters a folded weight, the
// gate/up input x and the down input h, is rotated before it is quantized.  Skipped on any path it gives plausible garbage,
// so core::PackImage refuses a file whose rotation this engine cannot apply.
#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace lamina::kernels::cpu {

struct HadamardSpec {
    int block = 0;                             ///< 0: no Hadamard-folded experts
    std::map<int, std::vector<float>> signs;   ///< width -> +1/-1 per input value (none: identity signs)
    std::vector<uint8_t> roles;                ///< per layer: bit 0 gate, bit 1 up, bit 2 down
    bool any() const { return block > 0; }
    bool gu(int64_t layer) const { return block > 0 && (roles[(size_t) layer] & 3u) != 0; }
    bool down(int64_t layer) const { return block > 0 && (roles[(size_t) layer] & 4u) != 0; }
};

/// The process-wide spec: set before the expert layout is read (empty for every other file).
const HadamardSpec& hadamard_spec();
void hadamard_set(HadamardSpec h);
/// The sign vector of `width` in the process-wide spec, or nullptr (identity signs).  Valid until the next hadamard_set.
const float* hadamard_signs(int width);

/// In place: x[i] *= signs[i] (when signs is not null), then each block of `block` values -> H_block x.
/// `n` must be a multiple of `block`.
void hadamard_rotate(float* x, int n, int block, const float* signs);

}  // namespace lamina::kernels::cpu
