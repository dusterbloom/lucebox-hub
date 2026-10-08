// LUCE_QWEN_LAUNCH_CUT building block: fuse a plain RMS_NORM * gamma producer
// directly with its Q8_1 activation quantization, eliminating one of the
// 256 launches/tok `quantize_q8_1` family (see bench/exact/LAUNCH-INVENTORY.md).
//
// This header/translation unit is self-contained and is NOT wired into the
// ggml-cuda.cu dispatch loop yet -- it ships a tested, bit-identical kernel
// ready for a follow-up wiring change once the exact uncovered call sites are
// confirmed on-box (PRODUCER_Q8 already covers the GDN-tail (36/tok) and HC
// (95/tok) producer sites; this targets the generic pre-norm RMS_NORM*gamma
// sites that still fall through to a separate quantize_q8_1 launch).
//
// Gated by env LUCE_QWEN_LAUNCH_CUT=1 once wired; until then this file only
// adds new symbols and changes no existing code path.
#pragma once

#include "common.cuh"

#include <cstdint>

// Host launcher for the fused kernel, for the common decode-time shape:
// one row per (row, channel, sample) grid cell, `mul` is a per-column gamma
// vector with the same [ncols, nrows, nchannels, nsamples] broadcast shape as
// `x` (no fastmodulo broadcast support -- callers with a differently-shaped
// gamma must broadcast it to this layout first). `out_q8` receives ncols/32
// contiguous block_q8_1 structs per row, same layout `quantize_row_q8_1_cuda`
// would produce for a single contiguous F32 row of `ncols` elements.
void ggml_cuda_rms_norm_mul_q8_1_cuda(
        const float * x, const float * mul, float * dst, block_q8_1 * out_q8,
        int ncols, int nrows, int nchannels, int nsamples,
        int64_t stride_row, int64_t stride_channel, int64_t stride_sample,
        int64_t mul_stride_row, int64_t mul_stride_channel, int64_t mul_stride_sample,
        float eps, cudaStream_t stream);

// Standalone bit-identity test hook (bench/exact/test_launch_cut_bitexact.cpp):
// runs the fused kernel (A) and a from-scratch reproduction of the unfused
// two-kernel baseline -- rms_norm_f32<block,do_multiply=true> (norm.cu) then
// quantize_q8_1 (quantize.cu) -- (B) on identical inputs and returns both F32
// norm outputs and both Q8_1 byte buffers for the caller to memcmp.
extern "C" GGML_BACKEND_API int ggml_cuda_test_launch_cut_rms_norm_mul_q8_1(
        const float * x, const float * gamma,
        float * dst_a, void * q8_a, float * dst_b, void * q8_b,
        int ncols, int nrows, float eps, void * raw_stream);
