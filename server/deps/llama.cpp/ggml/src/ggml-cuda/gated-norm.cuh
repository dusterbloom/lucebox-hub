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

// Parity-test entry point: runs the block-per-row (wave == 0) or wave-per-row (wave != 0) gated norm family on raw
// device pointers with the op's stride contract (gamma == NULL selects the sigmoid(z) * x gate).
extern "C" { GGML_BACKEND_API int ggml_cuda_test_gated_rms_norm_f16(int wave,
        const float * x, const float * gamma, const float * z, void * dst,
        int ncols, int nh, int T, int64_t x_s1, int64_t x_s2, int64_t z_sh, int64_t z_st, int gamma_rows, float eps,
        void * stream); }

// GGML_OP_GDN_TAIL (fusion-design.md K4): same arithmetic as ggml_cuda_op_gated_rms_norm_f16 but F32 output --
// the decode ssm_out consumer needs the un-rounded activation (the f16-gated GEMM path is prefill-only,
// mmb_min_t() == 512). Rows <= 128 (GDN's D=128) only.
bool ggml_cuda_gdn_tail_shape_ok(int64_t ncols);
void ggml_cuda_op_gdn_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
