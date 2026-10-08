#pragma once
#include "common.cuh"

// GGML_OP_MOE_ROUTE (fusion-design.md K5): fused router GEMV + softmax top-k + shexp gate, one launch.
// Validated against server/test/bench/moe_route_kernels.cu's mr_route_kernel (Experiment 2 round 2).
bool ggml_cuda_moe_route_shape_ok(int64_t N, int64_t NE, int64_t NU, int64_t T);
void ggml_cuda_op_moe_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
