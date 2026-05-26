// qwen3moe_verify_graph.h — shared types for the Qwen3-MoE verify graph builder.
//
// Qwen3MoeVerifyGraphResult must be visible to both the builder
// (qwen3moe_verify_graph.cpp) and the consumer (qwen3moe_dflash_target.cpp).

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

struct Qwen3MoeWeights;
struct Qwen3MoeCache;

// Return value of build_qwen3moe_verify_graph: named tensor handles so the
// caller can fill inputs, compute, and read outputs without knowing graph
// internals.
struct Qwen3MoeVerifyGraphResult {
    ggml_context *            ctx         = nullptr;
    ggml_cgraph *             gf          = nullptr;
    ggml_tensor *             inp         = nullptr;   // [hidden, n_tokens] F32
    ggml_tensor *             positions   = nullptr;   // [n_tokens] I32
    ggml_tensor *             attn_mask   = nullptr;   // [kv_len, n_tokens] F16
    ggml_tensor *             argmax      = nullptr;   // [n_tokens] I32 argmax
    std::vector<ggml_tensor*> captures;               // one per capture layer
};

// Build the verify graph: multi-token forward with all-position logits and
// activation captures at the requested layer IDs.
//
// `kv_start` — number of KV tokens already committed before this call.
// `capture_ids` — which layer indices to capture (after MoE residual add).
//
// Returns true on success; the caller must call ggml_free(out.ctx) after use.
bool build_qwen3moe_verify_graph(
        Qwen3MoeVerifyGraphResult & out,
        const Qwen3MoeWeights     & w,
        Qwen3MoeCache             & cache,
        ggml_backend_t              backend,
        int                         n_tokens,
        int                         kv_start,
        const std::vector<int>    & capture_ids);

} // namespace dflash::common
