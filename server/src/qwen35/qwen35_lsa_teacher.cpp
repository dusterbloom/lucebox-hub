#include "qwen35_lsa_teacher.h"

#include "attn_masks.h"

#include <vector>

namespace dflash::common {

void destroy_qwen35_lsa_teacher_step(Qwen35LsaTeacherStep & step) {
    if (step.alloc) {
        ggml_gallocr_free(step.alloc);
        step.alloc = nullptr;
    }
    if (step.ctx) {
        ggml_free(step.ctx);
        step.ctx = nullptr;
    }
    step.gf = nullptr;
    step.inp_embed = nullptr;
    step.positions = nullptr;
    step.attn_mask = nullptr;
    step.outputs = {};
    step.capture_config = {};
    step.kv_start = 0;
    step.n_tokens = 0;
    step.kq_stride_pad = KQ_MASK_PAD;
}

bool build_qwen35_lsa_teacher_step(
    Qwen35LsaTeacherStep & step,
    const TargetWeights & weights,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    const Qwen35LsaCaptureConfig & config,
    int kq_stride_pad,
    std::string & error) {
    destroy_qwen35_lsa_teacher_step(step);
    error.clear();
    if (!backend || kv_start < 0 || n_tokens <= 0 ||
        kv_start + n_tokens > cache.max_ctx ||
        kq_stride_pad < KQ_MASK_PAD) {
        error = "Qwen LSA teacher-step geometry is invalid";
        return false;
    }

    ggml_init_params params{};
    params.mem_size = 512 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    step.ctx = ggml_init(params);
    if (!step.ctx) {
        error = "Qwen LSA teacher graph context allocation failed";
        return false;
    }

    step.kv_start = kv_start;
    step.n_tokens = n_tokens;
    step.kq_stride_pad = kq_stride_pad;
    step.capture_config = config;
    step.inp_embed = ggml_new_tensor_3d(
        step.ctx, GGML_TYPE_F32, weights.n_embd, n_tokens, 1);
    step.positions =
        ggml_new_tensor_1d(step.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(step.inp_embed, "lsa_teacher_embed");
    ggml_set_name(step.positions, "lsa_teacher_positions");
    ggml_set_input(step.inp_embed);
    ggml_set_input(step.positions);

    if (n_tokens > 1) {
        const int kv_pad = align_up(kv_start + n_tokens, kq_stride_pad);
        const int q_pad = align_up(n_tokens, KQ_MASK_PAD);
        step.attn_mask = ggml_new_tensor_2d(
            step.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(step.attn_mask, "lsa_teacher_mask");
        ggml_set_input(step.attn_mask);
    }

    step.gf = ggml_new_graph_custom(step.ctx, 16384, false);
    QwenGraphInputs inputs{};
    inputs.inp_embed = step.inp_embed;
    inputs.positions = step.positions;
    inputs.attn_mask = step.attn_mask;
    inputs.n_tokens = n_tokens;
    inputs.kv_start = kv_start;
    inputs.last_token_logits_only = true;
    if (!configure_qwen35_lsa_capture(weights, config, inputs, error)) {
        destroy_qwen35_lsa_teacher_step(step);
        return false;
    }

    step.outputs =
        build_qwen35_graph(step.ctx, step.gf, weights, cache, inputs);
    if (!step.outputs.logits || !step.outputs.lsa_hidden) {
        error = "Qwen LSA teacher graph did not expose required outputs";
        destroy_qwen35_lsa_teacher_step(step);
        return false;
    }
    ggml_set_output(step.outputs.logits);
    ggml_build_forward_expand(step.gf, step.outputs.logits);

    step.alloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!step.alloc || !ggml_gallocr_alloc_graph(step.alloc, step.gf)) {
        error = "Qwen LSA teacher graph allocation failed";
        destroy_qwen35_lsa_teacher_step(step);
        return false;
    }
    return true;
}

bool execute_qwen35_lsa_teacher_step(
    Qwen35LsaTeacherStep & step,
    const TargetWeights & weights,
    TargetCache & cache,
    ggml_backend_t backend,
    const int32_t * token_ids,
    Qwen35LsaCaptureBatch & batch,
    std::string & error) {
    batch = {};
    error.clear();
    if (!step.ctx || !step.gf || !step.alloc || !backend || !token_ids ||
        step.n_tokens <= 0) {
        error = "Qwen LSA teacher step is not ready";
        return false;
    }

    std::vector<float> embeddings(
        static_cast<size_t>(weights.n_embd) * step.n_tokens);
    if (!weights.embedder.embed(
            token_ids, step.n_tokens, embeddings.data())) {
        error = "Qwen LSA teacher token embedding failed";
        return false;
    }
    ggml_backend_tensor_set(step.inp_embed, embeddings.data(), 0,
                            embeddings.size() * sizeof(float));

    std::vector<int32_t> positions(
        static_cast<size_t>(4) * step.n_tokens, 0);
    for (int token = 0; token < step.n_tokens; ++token) {
        const int32_t position = step.kv_start + token;
        positions[4 * token + 0] = position;
        positions[4 * token + 1] = position;
        positions[4 * token + 2] = position;
    }
    ggml_backend_tensor_set(step.positions, positions.data(), 0,
                            positions.size() * sizeof(int32_t));

    if (step.attn_mask) {
        std::vector<uint16_t> mask;
        build_causal_mask(mask, step.kv_start + step.n_tokens,
                          step.n_tokens, step.kv_start,
                          step.kq_stride_pad, 0,
                          static_cast<int>(step.attn_mask->ne[0]));
        ggml_backend_tensor_set(step.attn_mask, mask.data(), 0,
                                mask.size() * sizeof(uint16_t));
    }

    const ggml_status status =
        ggml_backend_graph_compute(backend, step.gf);
    if (status != GGML_STATUS_SUCCESS) {
        error = "Qwen LSA teacher graph compute failed";
        return false;
    }

    if (!read_qwen35_lsa_capture(
            weights, step.capture_config, step.outputs,
            step.n_tokens, batch, error)) {
        return false;
    }
    cache.cur_pos = step.kv_start + step.n_tokens;
    return true;
}

}  // namespace dflash::common
