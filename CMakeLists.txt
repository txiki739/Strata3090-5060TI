# Lamina Engine - top-level build.
#
# The engine (`lamina`) and its GPU tools need LAMINA_ENABLE_CUDA and NVIDIA GPUs of compute capability 8.0 or
# newer; without CUDA only the CPU-side libraries, their tools and tests build.
cmake_minimum_required(VERSION 3.24)
project(lamina VERSION 0.1.43.1 LANGUAGES CXX)
add_compile_definitions(LAMINA_VERSION="${PROJECT_VERSION}")

# A build for other PCs (the Docker image): ggml's CPU backend for any AVX2 CPU instead of this machine's (the
# engine's own AVX-512 kernels are per-file and chosen at run time either way), and the MSVC runtime linked in, so
# no Visual C++ redistributable is needed.
option(LAMINA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(LAMINA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(LAMINA_ENABLE_CUDA "Build the CUDA targets" OFF)
option(LAMINA_BUILD_TESTS "Build the test suite" OFF)
option(LAMINA_WERROR "Treat compiler warnings as errors" OFF)

if(LAMINA_ENABLE_CUDA)
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Plan v0.3 P6: developed and measured on sm_120 (RTX 50); the kernels need sm_80 or newer (tf32 mma in
    # the attention scorer, bf16 math), so RTX 30 (86) and RTX 40 (89) build too.  Older cards are refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    if(_base LESS 80)
      message(FATAL_ERROR
        "Lamina needs an NVIDIA GPU of compute capability 8.0 or newer (RTX 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.")
    endif()
    # #220 / #224: RTX 50 engines built with CUDA 12.8 crashed in the prompt path on Linux (13.0 and newer did not)
    if(_base GREATER_EQUAL 120 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(WARNING "Building for sm_${_base} (RTX 50) with CUDA ${CMAKE_CUDA_COMPILER_VERSION}: engines built with "
                      "CUDA 12.8 crashed on long prompts on these cards (issues #220, #224). Use CUDA 13.0 or newer.")
    endif()
  endforeach()
  message(STATUS "Lamina: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(lamina_warnings INTERFACE)
target_compile_options(lamina_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  # C4324: a struct padded after an alignas member (the CPU pool's activation images), as intended; the CUDA
  # sources that include them (the Qwen3.6 path) see it through the host compiler.
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/wd4324>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/wd4324>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>
  # C4211 comes from the host stubs nvcc generates for kernels with internal linkage (anonymous namespaces).
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/wd4211>)
# fopen, sscanf and getenv are standard C; MSVC deprecates them in favour of its own _s variants.
target_compile_definitions(lamina_warnings INTERFACE $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>)
if(LAMINA_WERROR)
  target_compile_options(lamina_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# ---- lamina_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(lamina_artifact INTERFACE)
target_include_directories(lamina_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lamina_artifact INTERFACE lamina_warnings)

add_executable(lamina-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(lamina-gguf PRIVATE lamina_artifact)

add_executable(lamina-dequant src/artifact/dequant.cpp)
target_link_libraries(lamina-dequant PRIVATE lamina_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(lamina-dequant PRIVATE LAMINA_DEQUANT_SELFTEST)

# ---- lamina_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(lamina_plan INTERFACE)
target_include_directories(lamina_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lamina_plan INTERFACE lamina_warnings)

add_executable(lamina-plan src/plan/plan_main.cpp)
target_link_libraries(lamina-plan PRIVATE lamina_plan)

# ---- lamina_core (CUDA) ---------------------------------------------------
# Only built when CUDA is on, so the targets above keep working on a machine with no toolkit.  The arch guard
# above has already refused anything older than sm_80 by the time this runs.
if(LAMINA_ENABLE_CUDA)
  add_library(lamina_core STATIC src/core/pinned.cu src/platform/memory.cpp src/core/weights.cpp src/core/layout.cpp
                                 src/platform/console_quit.cpp)
  target_include_directories(lamina_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # CUDA::cudart is PUBLIC because host translation units call the runtime: the .cu sources get cudart through the
  # CUDA language rules, a .cpp does not.
  target_link_libraries(lamina_core PUBLIC lamina_plan lamina_warnings CUDA::cudart)

  # ---- lamina_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `lamina_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(lamina_kernels STATIC src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_mma.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/draft_proxy.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(lamina_kernels PUBLIC lamina_core lamina_artifact)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(lamina_kernels PRIVATE third_party/ggml third_party/ggml-tq)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_mma.cu src/kernels/cuda/native_bf16.cu
    src/kernels/cuda/native_gr_norm.cu
    src/kernels/cuda/native_moe.cu
    src/kernels/cuda/native_gr_postops.cu
    src/kernels/cuda/native_router.cu
    src/kernels/cuda/native_qsa.cu
    src/kernels/cuda/native_rope.cu
    src/kernels/cuda/native_ple_postops.cu
    src/kernels/cuda/native_qsa_indexer.cu
    src/kernels/cuda/draft_proxy.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")

  # ---- lamina_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `lamina_core`, because it calls kernels and `lamina_kernels`
  # already links `lamina_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(lamina_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp
    src/core/prompt_cache.cpp src/core/adaptive_tier.cpp src/core/second_gpu.cpp src/core/arena_place.cpp
    src/core/split_head.cpp src/core/vmm.cpp src/core/gpu_select.cpp src/core/pack_image.cpp)
  target_include_directories(lamina_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(lamina_engine PUBLIC lamina_core lamina_kernels lamina_kernels_cpu CUDA::cudart ${CMAKE_DL_LIBS})

  # `lamina generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(lamina_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/experts.cpp
                                    src/prefill/prefill.cpp src/prefill/split.cpp)
  target_include_directories(lamina_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(lamina_prefill PUBLIC lamina_engine CUDA::cublas CUDA::cudart)

  # the engine as a library: the lamina binary's main (src/program/main.cpp) and lamina-server's engine thread, which
  # speaks the --serve protocol through in-process channels (serve_io.hpp)
  add_library(lamina_serve_io STATIC src/program/serve_io.cpp)
  target_include_directories(lamina_serve_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_library(lamina_program STATIC src/program/generate.cpp)
  target_include_directories(lamina_program PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(lamina_program PUBLIC lamina_engine lamina_prefill lamina_spec lamina_serve_io CUDA::cudart)
  add_executable(lamina src/program/main.cpp)
  target_link_libraries(lamina PRIVATE lamina_program)
  # the hyper-connection read's fused up projection + mix against cuBLAS + gr_mix_r at every sub-chunk size (--bench)
  add_executable(gr_upmix_parity src/prefill/gr_upmix_parity.cu)
  target_link_libraries(gr_upmix_parity PRIVATE lamina_prefill CUDA::cudart)
  add_test(NAME gr_upmix_parity COMMAND gr_upmix_parity)
  set_tests_properties(gr_upmix_parity PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(adaptive_policy_test tests/core/adaptive_policy_test.cpp)
  target_include_directories(adaptive_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(adaptive_policy_test PRIVATE lamina_artifact)
  # #477: the learned expert profile (--expert-profile-save): its order and make_profile.py's file format (CPU only)
  add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
  target_link_libraries(expert_profile_save_test PRIVATE lamina_engine)
  # the prompt cache's checkpoint retention (spread_victim) against least recent use (CPU only)
  add_executable(prompt_cache_spread_test tests/core/prompt_cache_spread_test.cpp)
  target_link_libraries(prompt_cache_spread_test PRIVATE lamina_engine)
  # the conversation cache's ownership: retention, the starvation regression, and the park's budgets (CPU only)
  add_executable(conversation_cache_test tests/core/conversation_cache_test.cpp)
  target_link_libraries(conversation_cache_test PRIVATE lamina_engine)
  if(LAMINA_BUILD_TESTS)
    add_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
    add_test(NAME prompt_cache_spread_test COMMAND prompt_cache_spread_test)
    add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
    add_test(NAME adaptive_policy_test COMMAND adaptive_policy_test)
    add_test(NAME deterministic_adapt_help COMMAND lamina --deterministic-adapt --help)
    set_tests_properties(deterministic_adapt_help PROPERTIES
                         PASS_REGULAR_EXPRESSION "--deterministic-adapt")
  endif()

  # A kernel test: a program calling the CUDA runtime, linked with lamina_kernels (and LIBS), and the ctest test
  # running it with ARGS (none with NO_TEST). A host .cpp gets the toolkit's headers and cudart explicitly: enabling
  # the CUDA language gives them to .cu sources only.
  function(lamina_kernel_test name source)
    cmake_parse_arguments(PARSE_ARGV 2 T "NO_TEST" "" "LIBS;ARGS")
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(${name} PRIVATE lamina_kernels ${T_LIBS} CUDA::cudart)
    if(NOT T_NO_TEST)
      add_test(NAME ${name} COMMAND ${name} ${T_ARGS})
    endif()
  endfunction()

  # the coalesced Q8_0 dequant against the per-block kernel: the same bytes in BF16, FP16 and FP32
  lamina_kernel_test(dequant_q8_0_identity src/kernels/dequant_q8_0_identity.cu)
  lamina_kernel_test(iq_parity src/kernels/iq_parity.cpp NO_TEST)
  lamina_kernel_test(shared_expert_parity src/kernels/shared_expert_parity.cpp ARGS --selftest)
  lamina_kernel_test(gr_parity src/kernels/gr_parity.cpp ARGS --selftest)
  lamina_kernel_test(draft_proxy_parity src/kernels/draft_proxy_parity.cpp ARGS --selftest)
  lamina_kernel_test(gdn_parity src/kernels/gdn_parity.cpp ARGS --selftest)
  lamina_kernel_test(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp ARGS --selftest)
  lamina_kernel_test(sampler_parity src/kernels/sampler_parity.cpp ARGS --selftest)
  set_tests_properties(sampler_parity PROPERTIES ENVIRONMENT "LAMINA_OLD_SAMPLER=0;LAMINA_SAMPLER_ONE_BLOCK=0")
  # the same fixtures on the other two sampled paths: the one-block kernel and the 0.1.20 kernel
  add_test(NAME sampler_parity_one_block COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_one_block PROPERTIES ENVIRONMENT "LAMINA_OLD_SAMPLER=0;LAMINA_SAMPLER_ONE_BLOCK=1")
  add_test(NAME sampler_parity_old COMMAND sampler_parity --selftest)
  set_tests_properties(sampler_parity_old PROPERTIES ENVIRONMENT "LAMINA_OLD_SAMPLER=1;LAMINA_SAMPLER_ONE_BLOCK=0")

  lamina_kernel_test(rope_parity src/kernels/rope_parity.cpp ARGS --selftest)
  lamina_kernel_test(quantize_act_parity src/kernels/quantize_act_parity.cpp ARGS --selftest)
  lamina_kernel_test(s_gemv_parity src/kernels/s_gemv_parity.cpp ARGS --selftest)
  lamina_kernel_test(elementwise_parity src/kernels/elementwise_parity.cpp ARGS --selftest)
  lamina_kernel_test(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp ARGS --selftest)
  lamina_kernel_test(qsa_parity src/kernels/qsa_parity.cpp ARGS --selftest)

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `lamina_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  lamina_kernel_test(kv_q8_parity src/kernels/kv_q8_parity.cpp)

  # The verify window's batched kernels vs the per-token kernel sequences they replace, bitwise (synthetic data).
  lamina_kernel_test(verify_parity src/kernels/verify_parity.cpp LIBS lamina_prefill lamina_kernels_cpu ARGS --selftest)

  # The QSA block scores and top-k against the kernels they replaced, kept in the test: bitwise (synthetic data).
  lamina_kernel_test(qsa_select_parity src/kernels/qsa_select_parity.cu)
  # The MMVQ of 2..8 tokens from interleaved activations against native_mmvq: bitwise (synthetic matrices).
  lamina_kernel_test(mmvq_il_parity src/kernels/mmvq_il_parity.cu)

  # native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
  # bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
  lamina_kernel_test(mmvq_multi_parity src/kernels/mmvq_multi_parity.cpp)

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE lamina_core CUDA::cudart)
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(ple_reader_test PRIVATE lamina_kernels CUDA::cudart)
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)
endif()

# ---- lamina_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `lamina_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(lamina_spec STATIC src/spec/suffix_drafter.cpp)
target_include_directories(lamina_spec PUBLIC include)
add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
target_link_libraries(suffix_drafter_test PRIVATE lamina_spec)
add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
if(LAMINA_BUILD_TESTS)
  # the f32 -> bf16 rounding against ggml's, NaN included, over every f32 bit pattern (header-only, CPU)
  add_executable(bf16_bits_test src/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  add_test(NAME bf16_bits_test COMMAND bf16_bits_test)
endif()

add_library(lamina_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/hadamard.cpp)
target_include_directories(lamina_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
if(MSVC)
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
# AVX-VNNI (Alder Lake, Zen 5 and later): the AVX-2 expert kernels' vpdpwssd / vpdpbusd copies (q2_avx2.cpp,
# iq_avx2.cpp; *_rows.inl), taken at run time where cpu_avxvnni_ok(); the same integer sums.  GCC and Clang compile
# only those functions with target("avxvnni"), MSVC takes the intrinsics under /arch:AVX2.  A compiler without them
# (GCC < 11, Clang < 12) builds the files without the copies.  (Upstream's, b211ba17.)
include(CheckCXXSourceCompiles)
if(MSVC)
  set(CMAKE_REQUIRED_FLAGS "/arch:AVX2")
else()
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c")
endif()
check_cxx_source_compiles([=[
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avxvnni")))
#endif
static __m256i dots(__m256i a, __m256i b) { return _mm256_dpbusd_avx_epi32(_mm256_dpwssd_avx_epi32(a, a, b), a, b); }
int main() {
    volatile int x = 1;
    return _mm256_extract_epi32(dots(_mm256_set1_epi32(x), _mm256_set1_epi32(x)), 0) == 0;
}
]=] LAMINA_HAVE_AVXVNNI_INTRINSICS)
unset(CMAKE_REQUIRED_FLAGS)
if(LAMINA_HAVE_AVXVNNI_INTRINSICS)
  set(_lamina_avxvnni LAMINA_AVXVNNI=1)
else()
  set(_lamina_avxvnni LAMINA_AVXVNNI=0)
endif()
set_property(SOURCE src/kernels/cpu/q2_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_lamina_avxvnni})
find_package(Threads REQUIRED)
target_link_libraries(lamina_kernels_cpu PUBLIC Threads::Threads)
# The expert pool against serial runs; `--stress` needs no pack (synthetic experts, idle gaps that put the workers
# to sleep).
add_executable(pool_test src/kernels/cpu/pool_test.cpp)
target_link_libraries(pool_test PRIVATE lamina_kernels_cpu)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DLAMINA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(LAMINA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(LAMINA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(LAMINA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(LAMINA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(LAMINA_GGML_DIR)
    add_subdirectory(${LAMINA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(lamina_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(lamina_llamacpp)
    if(NOT lamina_llamacpp_POPULATED)
      FetchContent_Populate(lamina_llamacpp)
    endif()
    add_subdirectory(${lamina_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # lamina-server's parts from the same checkout (its root skips the ggml added above): libllama for the tokenizer
  # (the vocabulary of the model's GGUF, loaded without its tensors), llama-common for the Jinja chat templates,
  # and the vendored cpp-httplib, nlohmann/json, stb and sheredom/subprocess
  if(LAMINA_GGML_DIR)
    set(_lamina_llama_root ${LAMINA_GGML_DIR})
  else()
    set(_lamina_llama_root ${lamina_llamacpp_SOURCE_DIR})
  endif()
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
  # lamina-server's outgoing HTTPS: WinHTTP on Windows (no OpenSSL DLLs beside it), OpenSSL through cpp-httplib elsewhere
  if(WIN32)
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
  endif()
  foreach(_o LLAMA_BUILD_TESTS LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_SERVER LLAMA_BUILD_APP
             LLAMA_ALL_WARNINGS LLAMA_TOOLS_INSTALL LLAMA_TESTS_INSTALL)
    set(${_o} OFF CACHE BOOL "" FORCE)
  endforeach()
  add_subdirectory(${_lamina_llama_root} ${CMAKE_BINARY_DIR}/llama EXCLUDE_FROM_ALL)
  # part of lamina_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(lamina_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                           src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/kq_avx2.cpp
                                           src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp
                                           src/kernels/cpu/tq_cpu.cpp src/kernels/cpu/tq_avx2.cpp
                                           src/kernels/cpu/tq_avx512.cpp)
  target_include_directories(lamina_kernels_cpu PRIVATE third_party/ggml third_party/ggml-tq)
  if(MSVC)
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    # older CPUs' kernels (AVX1, called behind cpu_avx1_ok() && !cpu_avx2_ok()): no FMA, no 256-bit integer ops
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp src/kernels/cpu/tq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
    set_source_files_properties(src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/tq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels/cpu/q2_avx1.cpp src/kernels/cpu/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS
                                "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
  endif()
  set_property(SOURCE src/kernels/cpu/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS ${_lamina_avxvnni})
  target_link_libraries(lamina_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(lamina_kernels_cpu PUBLIC LAMINA_NATIVE_EXPERTS=1)
  if(TARGET ple_reader_test)   # the PLE table's rows against ggml's to_float
    target_link_libraries(ple_reader_test PRIVATE ggml-base)
    target_compile_definitions(ple_reader_test PRIVATE LAMINA_NATIVE_EXPERTS=1)
  endif()
  # the AVX-2 expert kernels' variants (iq_avx2.cpp, q2_avx2.cpp; with AVX-VNNI where the CPU has it) bit for bit
  # against each other at 1..8 tokens and against ggml-cpu's dot, on random blocks; --bench: ms per expert on one
  # thread, the weights streamed from DRAM
  add_executable(iq_avx2_parity src/kernels/cpu/iq_avx2_parity.cpp)
  target_link_libraries(iq_avx2_parity PRIVATE lamina_kernels_cpu)
  add_test(NAME iq_avx2_parity COMMAND iq_avx2_parity)
  # the AVX-2 Q8_K activation quantizer against ggml's, byte for byte (--bench: its time)
  add_executable(q8k_quant_parity src/kernels/cpu/q8k_quant_parity.cpp)
  target_link_libraries(q8k_quant_parity PRIVATE lamina_kernels_cpu)
  add_test(NAME q8k_quant_parity COMMAND q8k_quant_parity)
  if(LAMINA_ENABLE_CUDA)
    # llama.cpp's MMQ kernels (ggml-cuda, MIT) for the prompt path's experts, from the same checkout as ggml: mmq.cuh's
    # instances for the expert types of ISTA-DASLab's and unsloth's files and the quantizer, with the few host
    # symbols of ggml-cuda.cu they use in src/prefill/ggml_cuda_host.cu
    if(LAMINA_GGML_DIR)
      set(_lamina_gg ${LAMINA_GGML_DIR}/ggml)
    else()
      set(_lamina_gg ${lamina_llamacpp_SOURCE_DIR}/ggml)
    endif()
    set(_lamina_mmq ${_lamina_gg}/src/ggml-cuda)
    # Gyro-L's trellis types: an overlay of the pinned ggml-cuda headers, written at configure time
    # (cmake/lamina_ggml_tq_overlay.cmake); off with a message on another llama.cpp tree (its prompts then read in FP16)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/lamina_ggml_tq_overlay.cmake)
    set(_lamina_tq_dir ${CMAKE_BINARY_DIR}/ggml-tq-overlay)
    get_filename_component(_lamina_tq_llama ${_lamina_gg} DIRECTORY)
    lamina_ggml_tq_overlay(${_lamina_tq_llama} ${_lamina_tq_dir})
    set(_lamina_quantize_cu ${_lamina_mmq}/quantize.cu)
    if(LAMINA_TQ_OVERLAY_OK)
      set(_lamina_quantize_cu ${_lamina_tq_dir}/quantize.cu)   # it includes mmq.cuh
    endif()
    add_library(lamina_mmq STATIC src/prefill/moe_mmq.cu src/prefill/ggml_cuda_host.cu ${_lamina_quantize_cu})
    foreach(_t q2_0 iq1_s iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q4_k q5_k q6_k q5_0 q5_1 q8_0)
      target_sources(lamina_mmq PRIVATE ${_lamina_mmq}/template-instances/mmq-instance-${_t}.cu)
    endforeach()
    if(LAMINA_TQ_OVERLAY_OK)
      foreach(_t tqp5 tqp6 tqp7)
        target_sources(lamina_mmq PRIVATE ${_lamina_tq_dir}/template-instances/mmq-instance-${_t}.cu)
      endforeach()
      target_include_directories(lamina_mmq BEFORE PRIVATE ${_lamina_tq_dir} ${_lamina_tq_dir}/include)
      target_compile_definitions(lamina_mmq PRIVATE LAMINA_MMQ_TQP=1)
    endif()
    # MMQ with a pointer per expert (src/prefill/moe_mmq_tab.cuh, upstream's copy of mmq.cuh's scheduler): the prompt
    # experts read in place.  The copy mirrors mmq.cuh from mul_mat_q_process_tile through mul_mat_q_case; another
    # llama.cpp tree (another fingerprint of that text) leaves it out, and the experts are gathered as before.  The
    # gate/up and down types of ISTA-DASLab's IQ3_S and unsloth's files, one TU each.
    file(READ ${_lamina_mmq}/mmq.cuh _lamina_mmq_src)
    string(REPLACE "\r\n" "\n" _lamina_mmq_src "${_lamina_mmq_src}")
    string(FIND "${_lamina_mmq_src}" "static __device__ __forceinline__ void mul_mat_q_process_tile" _lamina_tab_a)
    string(FIND "${_lamina_mmq_src}" "#define DECL_MMQ_CASE" _lamina_tab_b)
    set(_lamina_tab_hash "")
    if(_lamina_tab_a GREATER -1 AND _lamina_tab_b GREATER _lamina_tab_a)
      math(EXPR _lamina_tab_len "${_lamina_tab_b} - ${_lamina_tab_a}")
      string(SUBSTRING "${_lamina_mmq_src}" ${_lamina_tab_a} ${_lamina_tab_len} _lamina_tab_region)
      string(SHA256 _lamina_tab_hash "${_lamina_tab_region}")
    endif()
    unset(_lamina_mmq_src)
    unset(_lamina_tab_region)
    if(_lamina_tab_hash STREQUAL "36e4dc716c504470ec09cc0852d29a6ac5eace257f6e5e72db3805ca4f043eb0")   # 3cf03257
      set(_lamina_tab_cuh ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/moe_mmq_tab.cuh)
      set(_lamina_tab_types iq2_s iq3_xxs iq3_s iq4_xs q4_k q5_k q2_0 iq4_nl q5_0 q5_1 q8_0)
      if(LAMINA_TQ_OVERLAY_OK)
        list(APPEND _lamina_tab_types tqp5 tqp6 tqp7)
      endif()
      foreach(_t ${_lamina_tab_types})
        string(TOUPPER ${_t} _lamina_tab_T)
        configure_file(src/prefill/mmq_tab_instance.cu.in ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu @ONLY)
        target_sources(lamina_mmq PRIVATE ${CMAKE_BINARY_DIR}/mmq_tab/mmq-tab-${_t}.cu)
      endforeach()
      target_compile_definitions(lamina_mmq PRIVATE LAMINA_MMQ_TAB=1)
    else()
      message(STATUS "pointer-table MMQ left out: mmq.cuh's scheduler differs from the mirrored one (${_lamina_tab_hash})")
    endif()
    target_include_directories(lamina_mmq PRIVATE ${_lamina_gg}/include ${_lamina_gg}/src ${_lamina_mmq}
                               PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(lamina_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
    if(MSVC)
      target_compile_options(lamina_mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
    target_link_libraries(lamina_mmq PUBLIC ggml-base CUDA::cudart)
    target_link_libraries(lamina_prefill PUBLIC lamina_mmq)
    target_link_libraries(lamina_engine PUBLIC lamina_mmq)   # the verify window's projections at 7-8 tokens
    # the prompt path's routed-expert products by format at the engine's shapes (a bench, not a test)
    add_executable(mmq_expert_bench src/prefill/mmq_expert_bench.cpp)
    target_include_directories(mmq_expert_bench PRIVATE ${_lamina_gg}/include)
    target_link_libraries(mmq_expert_bench PRIVATE lamina_mmq lamina_kernels CUDA::cudart)
    # the prompt experts' q8_1 inputs in one kernel each, byte for byte the pairs they replace (--bench: both forms)
    add_executable(mmq_quant_parity src/prefill/mmq_quant_parity.cu)
    target_include_directories(mmq_quant_parity PRIVATE ${_lamina_gg}/include)
    target_link_libraries(mmq_quant_parity PRIVATE lamina_prefill lamina_mmq CUDA::cudart)
    add_test(NAME mmq_quant_parity COMMAND mmq_quant_parity)
    # the prompt experts' gate/up product read in place (the pointer-table MMQ) byte for byte the gathered one
    add_executable(mmq_tab_parity src/prefill/mmq_tab_parity.cu)
    target_include_directories(mmq_tab_parity PRIVATE ${_lamina_gg}/include)
    target_link_libraries(mmq_tab_parity PRIVATE lamina_mmq CUDA::cudart)
    add_test(NAME mmq_tab_parity COMMAND mmq_tab_parity)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(native_expert_parity PRIVATE lamina_engine lamina_kernels lamina_kernels_cpu lamina_artifact CUDA::cudart)
    add_test(NAME native_expert_parity_k_quants COMMAND native_expert_parity --synthetic q4_K/q5_1 q5_K/q8_0)
    add_test(NAME native_expert_parity_q5_0 COMMAND native_expert_parity --synthetic iq4_xs/q5_0 iq3_xxs/q5_0 iq2_s/q5_0)
    add_test(NAME native_expert_parity_trellis COMMAND native_expert_parity --synthetic tqp5/tqp5 tqp6/tqp6 tqp7/tqp7)
    # --expert-math exact's expf: the CPU pool's (lamina_kernels_cpu) against the GPU's, over every float
    add_executable(swiglu_expf_parity src/kernels/swiglu_expf_parity.cu)
    target_link_libraries(swiglu_expf_parity PRIVATE lamina_kernels_cpu CUDA::cudart)
    add_test(NAME swiglu_expf_parity COMMAND swiglu_expf_parity)
    # The grouped native expert kernels against the ones they replaced, kept in the test: bitwise (synthetic experts).
    add_executable(expert_grouped_parity src/kernels/expert_grouped_parity.cu)
    target_include_directories(expert_grouped_parity PRIVATE third_party/ggml third_party/ggml-tq src/kernels/cuda)
    target_link_libraries(expert_grouped_parity PRIVATE lamina_kernels lamina_kernels_cpu CUDA::cudart)
    add_test(NAME expert_grouped_parity COMMAND expert_grouped_parity)

    # Qwen3.6-35B-A3B (GGUF architecture qwen35moe): its own engine path, a library `lamina` switches to for such a
    # model (the family switch at the top of its main) and lamina-q36 wraps on its own; the experts that do not fit
    # in VRAM run on the CPU pool.  lamina-q36ref: the reference path tools/qwen35/ checks against llama.cpp.
    add_library(lamina_qwen35 STATIC src/qwen35/q36_engine.cu)
    target_include_directories(lamina_qwen35 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                               PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(lamina_qwen35 PRIVATE lamina_prefill lamina_kernels lamina_kernels_cpu lamina_warnings
                          lamina_serve_io CUDA::cudart)
    target_compile_definitions(lamina_qwen35 PRIVATE LAMINA_Q36_CPU_POOL=1)
    add_executable(lamina-q36 src/qwen35/q36_main.cpp)
    target_link_libraries(lamina-q36 PRIVATE lamina_qwen35 CUDA::cudart)
    target_link_libraries(lamina_program PUBLIC lamina_qwen35)
    target_compile_definitions(lamina_program PRIVATE LAMINA_HAVE_QWEN35=1)
    add_executable(lamina-q36ref src/qwen35/q36ref.cu)
    target_include_directories(lamina-q36ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(lamina-q36ref PRIVATE lamina_kernels lamina_warnings CUDA::cudart)
  endif()
endif()

if(LAMINA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
  target_link_libraries(dequant_bf16_test PRIVATE lamina_kernels lamina_artifact CUDA::cudart)

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE lamina_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE lamina_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME pool_stress COMMAND pool_test --stress)
endif()

if(LAMINA_BUILD_TESTS)
  enable_testing()
endif()

# ---- lamina-server ----------------------------------------------------------
# The HTTP server: the OpenAI, Anthropic and Responses APIs and the web app, with the engine on a thread of its own
# (lamina_program through in-process channels). lamina_text is its text layer (no GPU): the tokenizer, the chat
# template (Jinja, vendored from llama.cpp's common/jinja), the request conversions and the output parser;
# lamina-check runs it on JSON-lines cases (scratch/lamina/p6_check.py compares them with the Python server's).
if(TARGET cpp-httplib AND TARGET ggml-base)
  file(GLOB _lamina_jinja CONFIGURE_DEPENDS src/server/vendor/jinja/*.cpp)
  add_library(lamina_jinja STATIC ${_lamina_jinja} src/server/vendor/common/json.cpp
                                  src/server/vendor/common/unicode.cpp)
  target_include_directories(lamina_jinja SYSTEM PUBLIC src/server/vendor PRIVATE src/server/vendor/common)
  target_link_libraries(lamina_jinja PUBLIC nlohmann ggml-base)
  if(MSVC)
    target_compile_options(lamina_jinja PRIVATE /w /utf-8 /bigobj)
  else()
    target_compile_options(lamina_jinja PRIVATE -w)
  endif()
  # llama.cpp's pre-tokenizer split (the qwen35 pattern has a custom implementation there)
  add_library(lamina_unicode STATIC ${_lamina_llama_root}/src/unicode.cpp ${_lamina_llama_root}/src/unicode-data.cpp)
  target_include_directories(lamina_unicode PUBLIC ${_lamina_llama_root})
  if(MSVC)
    target_compile_options(lamina_unicode PRIVATE /w /utf-8)
  else()
    target_compile_options(lamina_unicode PRIVATE -w)
  endif()
  add_library(lamina_text STATIC src/server/json.cpp src/server/log.cpp src/server/tokenizer.cpp
                                 src/server/frontend.cpp)
  target_include_directories(lamina_text PUBLIC src/server ${CMAKE_CURRENT_SOURCE_DIR}/tools/vision)
  target_link_libraries(lamina_text PUBLIC lamina_jinja lamina_unicode nlohmann ggml-base lamina_warnings)
  add_executable(lamina-check src/server/check.cpp)
  target_link_libraries(lamina-check PRIVATE lamina_text)
  # a tool call's parameter values against the tool's schema, whole and streamed (CPU only)
  add_executable(tool_args_test tests/server/tool_args_test.cpp)
  target_link_libraries(tool_args_test PRIVATE lamina_text)
  add_test(NAME tool_args_test COMMAND tool_args_test)
  if(TARGET lamina_program)
    # the web app inside the binary (cmake/embed_web.cmake)
    file(GLOB_RECURSE _lamina_web CONFIGURE_DEPENDS src/server/web/*)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/web_assets.cpp
      COMMAND ${CMAKE_COMMAND} -DWEB_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src/server/web
              -DOUT=${CMAKE_CURRENT_BINARY_DIR}/web_assets.cpp -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_web.cmake
      DEPENDS ${_lamina_web} cmake/embed_web.cmake
      COMMENT "Embedding the web app")
    add_library(lamina_server_core STATIC src/server/api.cpp src/server/config.cpp src/server/engine.cpp
                src/server/engine_batch.cpp src/server/http_client.cpp src/server/http_server.cpp src/server/images.cpp
                src/server/keepawake.cpp src/server/mcp.cpp src/server/net.cpp src/server/process.cpp
                src/server/prometheus.cpp src/server/replicas.cpp src/server/responses.cpp src/server/runconfig.cpp
                src/server/service.cpp src/server/service_run.cpp src/server/structured.cpp src/server/telemetry.cpp
                src/server/vision.cpp ${CMAKE_CURRENT_BINARY_DIR}/web_assets.cpp)
    target_include_directories(lamina_server_core PUBLIC src/server ${_lamina_llama_root}/vendor
                               ${_lamina_llama_root}/vendor/cpp-httplib)
    target_link_libraries(lamina_server_core PUBLIC lamina_text cpp-httplib vendor-hash lamina_program lamina_warnings)
    if(WIN32)
      target_link_libraries(lamina_server_core PUBLIC winhttp ws2_32 ole32 windowscodecs shell32)
    else()
      find_package(Threads REQUIRED)
      target_link_libraries(lamina_server_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
    endif()
    add_executable(lamina-server src/server/main.cpp)
    target_link_libraries(lamina-server PRIVATE lamina_server_core)
    # cmake --install <build> --component lamina --prefix <dir>: the programs in <dir>/bin, on Windows with the CUDA
    # runtime beside them (lamina-vision installs from tools/vision's own build; the component leaves out the install
    # rules of llama.cpp's libraries)
    install(TARGETS lamina lamina-server RUNTIME DESTINATION bin COMPONENT lamina)
    if(WIN32)
      file(GLOB _lamina_cuda_dlls "${CUDAToolkit_BIN_DIR}/cudart64_*.dll" "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
           "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll" "${CUDAToolkit_BIN_DIR}/x64/cudart64_*.dll"
           "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll" "${CUDAToolkit_BIN_DIR}/x64/cublasLt64_*.dll")
      install(FILES ${_lamina_cuda_dlls} DESTINATION bin COMPONENT lamina)
    endif()
  endif()
endif()

# Windows: every program here runs with UTF-8 as its code page (cmake/utf8.manifest)
if(MSVC)
  get_property(_lamina_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_t IN LISTS _lamina_targets)
    get_target_property(_type ${_t} TYPE)
    if(_type STREQUAL "EXECUTABLE")
      target_sources(${_t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/utf8.manifest)
    endif()
  endforeach()
endif()
