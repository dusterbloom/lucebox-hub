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
// to host (or, when the fused LM-head path is wired, out_argmax_token /
// out_topk_* are read instead and out_x_normed is unused).
//
// Phase B+: graphs are cached in Qwen36MtpModule keyed by draft_pos so
// the per-call rebuild cost (ggml_init + DAG construction + gallocr) is
// amortised across the lifetime of a generation.  Each cached graph is
// built with its own (draft_pos, kv_len) — kv_len defaults to draft_pos+1
// for the autoregressive chain pattern.  `topk_k` records the top-K size
// the fused LM-head outputs were built for (0 = no fused LM head).
struct Qwen36MtpStepGraph {
    ggml_context * ctx           = nullptr;
    ggml_cgraph  * gf            = nullptr;
    ggml_gallocr_t alloc         = nullptr;

    // Build-time keys (used to detect invalidation).
    int  draft_pos  = -1;
    int  kv_len     = -1;
    int  fa_window  = 0;
    int  topk_k     = 0;
    bool fused_lm_head = false;

    // Inputs.
    ggml_tensor * inp_embed   = nullptr;   // [n_embd]   f32 — pre-embedded cur_token
    ggml_tensor * inp_h_prev  = nullptr;   // [n_embd]   f32 — backbone hidden h_{base_pos-1}
    ggml_tensor * inp_pos     = nullptr;   // [4]        i32 — MROPE positions (p,p,p,0)

    // Outputs.
    ggml_tensor * out_x_normed     = nullptr;  // [n_embd]   f32 — post shared_head_norm hidden
    // Fused LM-head outputs (only populated when build is called with a
    // non-null lm_head_weight).  out_argmax_token holds the i32 argmax of
    // the logits; out_logits is exposed so the host can compute log-softmax
    // for top-K emission without re-running the LM head matmul.
    ggml_tensor * out_argmax_token = nullptr;  // [1]        i32
    ggml_tensor * out_logits       = nullptr;  // [n_vocab]  f32 — full logits (optional)
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
//     cache_k / cache_v layout.  Phase B+ stores these as F16 (was F32);
//     the F32 K/V emitted by the head's projections are cast on the fly
//     by ggml_cpy.
//   - draft_pos: slot index in head_k/v_cache where this call's K/V are
//     written.  Must equal base_pos for the current step (head's slot
//     convention; warmup writes slots [1, n_prompt], step writes slot
//     base_pos+h).
//   - kv_len: number of slots the flash attention attends over
//     ([fa_kv_lo, kv_len)); must be >= draft_pos+1 for causal correctness.
//     fa_window: if > 0, restrict FA to slots [kv_len - fa_window, kv_len).
//     If <= 0, attend the full [0, kv_len).  Mirrors the backbone's causal
//     window so the head sees the same active context as the target.
//   - lm_head_weight (optional): if non-null, append `mul_mat(W, x_normed)
//     -> argmax` to the graph so the host avoids a hidden -> logits round
//     trip per call.  When non-null, sg.out_argmax_token and sg.out_logits
//     are populated; otherwise sg.out_x_normed is the sole output.
//   - lm_head_topk: if > 0 and lm_head_weight is non-null, also expose the
//     raw logits as a graph output so the caller can compute log-softmax
//     for top-K on the host without a second matmul.  K is recorded on the
//     graph state for cache invalidation.
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
        int kv_len,
        int fa_window           = 0,
        ggml_tensor * lm_head_weight = nullptr,
        int lm_head_topk        = 0);

}  // namespace mtp
}  // namespace dflash27b
