#pragma once

#include "common.cuh"

void ggml_cuda_op_ds4_moe_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_ds4_moe_combine_shared_gate(ggml_backend_cuda_context &, ggml_tensor *, const ggml_tensor *, const ggml_tensor *);
