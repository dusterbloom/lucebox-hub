#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Invalidate the per-forward rope cos/sin tables (call once per graph compute).
void ggml_cuda_rope_begin_graph();

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// ROPE (multi-section, F32) -> PERMUTE -> CONT in one kernel: the rotation is written straight into cont's layout.
void ggml_cuda_op_rope_permuted(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * cont);

// RMS_NORM -> MUL -> ROPE (multi-section, F32, per-token path) [-> PERMUTE -> CONT] in one kernel, bit-identical to
// the separate kernels (qwen4exp attention Q/K and indexer q). cont may be null. LUCE_NORM_ROPE_LEGACY=1 disables.
bool ggml_cuda_norm_rope_enabled();
void ggml_cuda_op_norm_rope(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul, ggml_tensor * rope,
                            ggml_tensor * cont);
