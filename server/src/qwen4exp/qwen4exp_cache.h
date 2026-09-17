// Qwen4ExpCache — KV + gated-delta-net state for Qwen3.8-Flash-Next.
//
// Hybrid cache: the 12 full-attention layers own a standard K/V cache, the 36
// linear-attention layers own a fixed recurrent state (independent of context)
// plus a short depthwise-conv history. Shapes match the GGUF tensor layout:
//   attn_qkv [n_embd, 10240]  -> conv_channels = d_inner + 2*n_group*d_state
//   head_dim 256, n_head_kv 2, n_v_heads 48, d_state 128

#pragma once

#include "qwen4exp_internal.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

namespace dflash::common {

struct Qwen4ExpCache {
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    int       max_ctx  = 0;
    int       cur_pos  = 0;
    ggml_type kv_type  = GGML_TYPE_F16;

    std::vector<int> full_layer_ids;    // size = 12
    std::vector<int> linear_layer_ids;  // size = 36

    // Full attention: [head_dim, max_ctx, n_head_kv] (flash_attn_ext layout).
    std::vector<ggml_tensor *> attn_k;  // size = n_full
    std::vector<ggml_tensor *> attn_v;

    // Gated delta net: ssm_state [S_v, S_v, H_v] f32;
    //                  conv_state [kernel-1, conv_channels] f32.
    std::vector<ggml_tensor *> ssm_state;   // size = n_linear
    std::vector<ggml_tensor *> conv_state;

    // Per-layer embedding (PLE) conv history, one per PLE layer:
    // [ple_hist, hc_dim] f32 where ple_hist = (ple_conv_kernel-1)*ple_ngram_size.
    std::vector<ggml_tensor *> ple_conv_state;
    std::vector<int> ple_layer_ids;

    // Rolling window of the last (ple_ngram_size - 1) token ids, oldest first,
    // for the host-side PLE n-gram hash across decode steps.
    std::vector<int32_t> ple_prev;
};

bool create_qwen4exp_cache(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           int max_ctx, ggml_type kv_type, Qwen4ExpCache & out);

void free_qwen4exp_cache(Qwen4ExpCache & c);

// Zero the recurrent state and conv history and reset cur_pos. KV is left
// intact; callers that need a clean sequence also reset cur_pos themselves.
void reset_qwen4exp_state(ggml_backend_t backend, Qwen4ExpCache & c);

}  // namespace dflash::common
