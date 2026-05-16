// qwen36_mtp_graph.h — CUDA cgraph for Qwen3.6 MTP head step forward.

#pragma once

#include "qwen36_mtp.h"

struct ggml_tensor;
struct ggml_context;
struct ggml_cgraph;
struct ggml_gallocr;
typedef struct ggml_gallocr * ggml_gallocr_t;
struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace dflash27b {
namespace mtp {

// Step graph state for a single MTP head's per-call forward.  Inputs are
// set via ggml_backend_tensor_set; the output (out_x_normed) is read back
// to host and projected through the target's shared LM head.
struct Qwen36MtpStepGraph {
    ggml_context * ctx           = nullptr;
    ggml_cgraph  * gf            = nullptr;
    ggml_gallocr_t alloc         = nullptr;

    // Inputs.
    ggml_tensor * inp_embed   = nullptr;   // [n_embd]   f32 — pre-embedded cur_token
    ggml_tensor * inp_h_prev  = nullptr;   // [n_embd]   f32 — backbone hidden h_{base_pos-1}
    ggml_tensor * inp_pos     = nullptr;   // [4]        i32 — MROPE positions (p,p,p,0)

    // Outputs.
    ggml_tensor * out_x_normed = nullptr;  // [n_embd]   f32 — post shared_head_norm hidden
};

void qwen36_mtp_step_graph_free(Qwen36MtpStepGraph & sg);

// Batched warmup graph: writes K/V to head_kv slots [slot_start, slot_start+n_tokens)
// in a single backend pass.  Replaces the host-side per-position CPU loop with
// one ggml cgraph using the same quant-aware matmul kernels the step graph uses.
struct Qwen36MtpWarmGraph {
    ggml_context * ctx           = nullptr;
    ggml_cgraph  * gf            = nullptr;
    ggml_gallocr_t alloc         = nullptr;

    ggml_tensor * inp_embed_seq  = nullptr;   // [n_embd, n_tokens] f32 — pre-embedded tokens
    ggml_tensor * inp_h_seq      = nullptr;   // [n_embd, n_tokens] f32 — backbone hiddens
    ggml_tensor * inp_pos        = nullptr;   // [4 * n_tokens]     i32 — MROPE positions
};

void qwen36_mtp_warm_graph_free(Qwen36MtpWarmGraph & sg);

// Build the warmup graph for n_tokens prefill positions writing to slots
// [slot_start, slot_start + n_tokens) of head_k_cache / head_v_cache.
bool build_qwen36_mtp_warm_graph(
        Qwen36MtpWarmGraph & sg,
        const Qwen36MtpHeadWeights & head,
        ggml_tensor * head_k_cache,
        ggml_tensor * head_v_cache,
        ggml_backend_t backend,
        int n_embd,
        int n_head_kv,
        int key_len,
        int val_len,
        int n_rot,
        int rope_sections[4],
        float rope_freq_base,
        float rms_eps,
        int slot_start,
        int n_tokens);

// Build the head's step graph for a SPECIFIC (draft_pos, kv_len) pair.
//   - head_k_cache / head_v_cache: per-head KV cache tensors on the backbone
//     backend; layout [head_dim, n_ctx, n_head_kv] matching the backbone's
//     cache_k / cache_v layout.
//   - draft_pos: slot index in head_k/v_cache where this call's K/V are
//     written.  Must equal base_pos for the current step (head's slot
//     convention; warmup writes slots [1, n_prompt], step writes slot
//     base_pos+h).
//   - kv_len: number of slots the flash attention attends over
//     ([0, kv_len)); must be >= draft_pos+1 for causal correctness.
// Returns false on allocation failure.
bool build_qwen36_mtp_step_graph(
        Qwen36MtpStepGraph & sg,
        const Qwen36MtpHeadWeights & head,
        ggml_tensor * head_k_cache,
        ggml_tensor * head_v_cache,
        ggml_backend_t backend,
        int n_embd,
        int n_head,
        int n_head_kv,
        int key_len,
        int val_len,
        int ffn_len,
        int n_rot,
        int rope_sections[4],
        float rope_freq_base,
        float rms_eps,
        int draft_pos,
        int kv_len);

}  // namespace mtp
}  // namespace dflash27b
