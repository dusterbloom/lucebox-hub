// Qwen3MoeBackend forward-graph builder — STUB.
//
// Phase A TODO (next session): port qwen3/qwen3_graph.cpp attention block
// 1:1 (Q/K/V proj, q_norm, k_norm, RoPE NEOX, BF16 cache view+cpy,
// flash_attn_ext, output proj, residual), then implement the Qwen3-MoE
// FFN block following the pattern in
// dflash/src/gemma4/gemma4_graph.cpp::build_gemma4_moe_block:
//
//   ffn_in = rms_norm(cur) * ffn_norm
//   logits = mul_mat(ffn_gate_inp, ffn_in)          // [n_expert, n_tokens]
//   probs  = soft_max(logits)
//   sel    = top_k(probs, n_expert_used)
//   w_k    = get_rows(reshape3d(probs,1,n_expert,n_tok), sel)  // [n_used, n_tok]
//   // if norm_topk_prob: w_k = w_k / sum(w_k) per token
//   cur3d  = reshape3d(ffn_in, n_embd, 1, n_tokens)
//   gate_e = mul_mat_id(ffn_gate_exps, cur3d, sel)  // [n_ff_exp, n_used, n_tok]
//   up_e   = mul_mat_id(ffn_up_exps,   cur3d, sel)
//   gu     = silu(gate_e) * up_e                    // SwiGLU (NOT GELU)
//   exps   = mul_mat_id(ffn_down_exps, gu, sel)     // [n_embd, n_used, n_tok]
//   routed = sum_i (exps[:,i,:] * w_k[i,:])         // weighted sum over experts
//   cur    = cur + routed
//
// NOTE differences from Gemma4 MoE:
//   - SR2AM-30B has NO shared expert. ffn_gate/up/down are absent.
//   - SwiGLU (silu), not GELU.
//   - Separate gate_exps / up_exps tensors (NOT fused like Gemma4).
//   - Qwen3 uses norm_topk_prob=true (renormalize top-k probs).

#include "qwen3moe_internal.h"

#include <cstdio>

// Placeholder so the translation unit isn't empty for the linker.
namespace dflash::common {
void qwen3moe_graph_stub_marker() {
    // Intentionally empty.
}
}  // namespace dflash::common
