// src/kernels/cpu/hadamard.cpp - see include/lamina/kernels/cpu/hadamard.hpp.
#include "lamina/kernels/cpu/hadamard.hpp"

#include <cmath>
#include <utility>

namespace lamina::kernels::cpu {
namespace {
HadamardSpec g_spec;
}

const HadamardSpec& hadamard_spec() { return g_spec; }
void hadamard_set(HadamardSpec h) { g_spec = std::move(h); }

const float* hadamard_signs(int width) {
    const auto it = g_spec.signs.find(width);
    return it == g_spec.signs.end() ? nullptr : it->second.data();
}

void hadamard_rotate(float* x, int n, int block, const float* signs) {
    if (signs)
        for (int i = 0; i < n; ++i) x[i] *= signs[i];
    const float scale = 1.0f / std::sqrt((float) block);
    for (int b0 = 0; b0 < n; b0 += block) {
        float* v = x + b0;
        // the in-place fast Walsh-Hadamard transform in natural (Sylvester) order: entry (-1)^popcount(r & c)
        for (int len = 1; len < block; len <<= 1)
            for (int i = 0; i < block; i += len << 1)
                for (int j = i; j < i + len; ++j) {
                    const float a = v[j], c = v[j + len];
                    v[j] = a + c;
                    v[j + len] = a - c;
                }
        for (int i = 0; i < block; ++i) v[i] *= scale;
    }
}

}  // namespace lamina::kernels::cpu
