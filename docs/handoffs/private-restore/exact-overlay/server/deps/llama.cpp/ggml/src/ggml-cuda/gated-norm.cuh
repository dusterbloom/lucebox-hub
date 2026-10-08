#pragma once
#include "common.cuh"

// Fused gated RMS norm for the qwen4exp gated-delta-net tail: rms_norm(x) * gamma * sigmoid(z), written as F16 so the
// following Q8_0 -> F16 GEMM reads it directly. Same arithmetic as rms_norm_f32<256, true> + unary_mul(sigmoid) + the
// MMB F16 activation conversion, so it is bit-exact against that chain.
void ggml_cuda_op_gated_rms_norm_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Decode producer for an ordinary F32 gated RMS-norm result. `out_q8` is the
// same block_q8_1 byte layout emitted by quantize_row_q8_1_cuda, ready for the
// following MMVQ. The caller owns and keys the buffer in the per-eval memo.
void ggml_cuda_gated_rms_norm_q8_1(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * gamma, const ggml_tensor * z,
        ggml_tensor * dst, block_q8_1 * out_q8, float eps);

// Private emitted-byte fixture entry point. Exact x==dst is accepted; shifted
// or any q8/input/output overlap is rejected before launching the real kernel.
extern "C" GGML_BACKEND_API int ggml_cuda_test_gdn_q8_producer(
        const float * x, const float * gamma, const float * z, float * dst,
        block_q8_1 * q8, int gamma_rows, float eps, void * stream);
