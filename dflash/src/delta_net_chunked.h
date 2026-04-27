#pragma once
// Chunked gated delta-net graph builder.
// See src/delta_net_chunked.cpp for the algorithm and history.

#include <ggml.h>

namespace dflash27b {

struct DeltaNetChunkedResult {
    ggml_tensor * output;     // [S_v, H_v, n_tokens, n_seqs]
    ggml_tensor * new_state;  // [S_v, S_v, H_v, n_seqs]
    ggml_tensor * sl_mask;    // [CS, CS, n_chunks, H_v*n_seqs] input: strictly-lower mask (set before compute)
};

// Chain-only, no-tree variant. Caller passes q/k/v/g/b/s in the same shape as
// ggml_gated_delta_net expects. Returns the per-token output and the final
// recurrent state as two separate tensors (unlike the fused kernel which packs
// them into one dst tensor).
//
// gf: the compute graph being built; required (non-null) when ssm_inter_cap is
//   non-null so that per-token capture ggml_cpy ops are registered as graph
//   nodes. May be nullptr when ssm_inter_cap is also nullptr.
//
// ssm_inter_cap: optional persistent f16 buffer of shape
//   [S_v, S_v, H_v, max_verify_tokens] (cache.ssm_intermediate[il]).
//   When non-null, the sequential GDA path emits an in-graph ggml_cpy after
//   each token update to write the per-step state (f32) into slot t of the
//   f16 capture buffer. Used by the fast-rollback spec-decode path.
//   Ignored (nullptr) when using the KDA chunked path.
DeltaNetChunkedResult build_delta_net_chunked(
        ggml_context * ctx0,
        ggml_cgraph  * gf,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * g,
        ggml_tensor  * b,
        ggml_tensor  * s,
        ggml_tensor  * ssm_inter_cap = nullptr);

} // namespace dflash27b
