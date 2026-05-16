// Qwen35DFlashTarget — DFlashTarget adapter for qwen35 hybrid models.

#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "step_graph.h"
#include "common/attn_masks.h"

namespace dflash27b {

Qwen35DFlashTarget::~Qwen35DFlashTarget() {
    step_graph_destroy(proj_sg_);
}

Qwen35DFlashTarget::Qwen35DFlashTarget(
        TargetWeights & w,
        TargetCache & cache,
        ggml_backend_t backend,
        StepGraph & sg,
        int kq_stride_pad,
        int fa_window)
    : w_(w), cache_(cache), backend_(backend), sg_(sg),
      kq_stride_pad_(kq_stride_pad), fa_window_(fa_window) {
    capture_ids_.assign(w.capture_layer_ids,
                        w.capture_layer_ids + w.n_capture_layers);
}

bool Qwen35DFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax) {
    const int n_tokens = (int)tokens.size();
    if (n_tokens <= 0) return false;

    const int hidden = w_.n_embd;
    const bool need_mask = (kq_stride_pad_ > KQ_MASK_PAD) || (n_tokens > 1);

    if (!build_target_step(sg_, w_, cache_, backend_,
                           /*kv_start=*/base_pos, n_tokens,
                           need_mask, /*capture=*/true,
                           /*capture_delta_intermediate=*/false,
                           fa_window_,
                           /*last_token_logits_only=*/false,
                           kq_stride_pad_,
                           /*capture_all_norm_hidden=*/capture_hidden_seq_)) {
        std::fprintf(stderr, "[Qwen35DFlashTarget] build_target_step failed: base_pos=%d n_tokens=%d\n",
                     base_pos, n_tokens);
        return false;
    }

    // Embed input tokens and fill positions.
    std::vector<float> embed((size_t)n_tokens * hidden);
    if (!w_.embedder.embed(tokens.data(), n_tokens, embed.data())) return false;
    ggml_backend_tensor_set(sg_.inp_embed, embed.data(), 0,
                            sizeof(float) * embed.size());

    // Qwen35 uses interleaved positions: 4 ints per token.
    std::vector<int32_t> pos(4 * n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        pos[4 * i + 0] = base_pos + i;
        pos[4 * i + 1] = base_pos + i;
        pos[4 * i + 2] = base_pos + i;
        pos[4 * i + 3] = 0;
    }
    ggml_backend_tensor_set(sg_.positions, pos.data(), 0,
                            sizeof(int32_t) * pos.size());

    // Populate the causal attention mask. The mask buffer is freshly allocated
    // by build_target_step (uninitialized memory); without this set, attention
    // reads garbage and ggml_argmax over the resulting logits returns -1 for
    // non-last positions, breaking the chain runner's recommit path.
    if (sg_.attn_mask) {
        const int win_start = (fa_window_ > 0 && base_pos > fa_window_)
                                  ? (base_pos - fa_window_) : 0;
        const int kv_len = base_pos + n_tokens - win_start;
        std::vector<uint16_t> mask_buf;
        build_causal_mask(mask_buf, kv_len, n_tokens, base_pos,
                          kq_stride_pad_, win_start);
        ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    }

    auto st = ggml_backend_graph_compute(backend_, sg_.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[Qwen35DFlashTarget] graph_compute failed: base_pos=%d n_tokens=%d status=%d\n",
                     base_pos, n_tokens, (int)st);
        return false;
    }

    // Read argmax from last token.
    if (n_tokens == 1) {
        ggml_backend_tensor_get(sg_.argmax_tokens, &last_tok, 0, sizeof(int32_t));
    } else {
        ggml_backend_tensor_get(sg_.argmax_tokens, &last_tok,
                                (size_t)(n_tokens - 1) * sizeof(int32_t),
                                sizeof(int32_t));
    }

    if (all_argmax) {
        all_argmax->resize(n_tokens);
        ggml_backend_tensor_get(sg_.argmax_tokens, all_argmax->data(), 0,
                                sizeof(int32_t) * n_tokens);
    }

    // Copy the last token's post-norm hidden to a CPU buffer so the MTP module
    // can call last_hidden() before the next graph_compute overwrites it.
    if (sg_.last_norm_hidden) {
        last_hidden_cpu_.resize(hidden);
        ggml_backend_tensor_get(sg_.last_norm_hidden, last_hidden_cpu_.data(),
                                0, sizeof(float) * hidden);
    }

    // Copy the full [n_tokens, n_embd] post-norm hidden sequence so the
    // Qwen3.6 MTP warm_head_kv() can read per-position hiddens during prefill.
    if (sg_.all_norm_hidden) {
        last_hidden_seq_cpu_.resize((size_t)n_tokens * hidden);
        ggml_backend_tensor_get(sg_.all_norm_hidden, last_hidden_seq_cpu_.data(),
                                0, sizeof(float) * (size_t)n_tokens * hidden);
        last_hidden_seq_n_         = n_tokens;
        last_verify_chunk_start_   = base_pos;
    } else {
        last_hidden_seq_n_ = 0;
    }

    cache_.cur_pos = base_pos + n_tokens;
    return true;
}

bool Qwen35DFlashTarget::snapshot_kv() {
    snapshot_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::restore_kv() {
    restore_ssm_state(cache_);
    return true;
}

bool Qwen35DFlashTarget::is_eos(int token) const {
    return is_eos_tok(token, w_);
}

bool Qwen35DFlashTarget::embed_tokens(const int32_t * tokens, int n,
                                       float * out) const {
    return w_.embedder.embed(tokens, n, out);
}

bool Qwen35DFlashTarget::project_hidden_to_tokens(
        const float * hidden,
        int n_tokens,
        std::vector<int32_t> & tokens_out) {
    if (n_tokens <= 0) return false;

    if (!build_lm_head_projection_step(proj_sg_, w_, backend_, n_tokens)) {
        return false;
    }

    ggml_backend_tensor_set(proj_sg_.hidden_input, hidden, 0,
                            sizeof(float) * (size_t)n_tokens * w_.n_embd);

    auto st = ggml_backend_graph_compute(backend_, proj_sg_.gf);
    if (st != GGML_STATUS_SUCCESS) return false;

    tokens_out.resize(n_tokens);
    ggml_backend_tensor_get(proj_sg_.argmax_tokens, tokens_out.data(), 0,
                            sizeof(int32_t) * n_tokens);
    return true;
}

int Qwen35DFlashTarget::mask_token_id() const {
    return w_.mask_token_id;
}

const std::vector<int> & Qwen35DFlashTarget::capture_layer_ids() const {
    return capture_ids_;
}

}  // namespace dflash27b
