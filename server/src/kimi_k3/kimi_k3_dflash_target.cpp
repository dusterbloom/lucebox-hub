#include "kimi_k3_dflash_target.h"

#include "common/dflash_feature_ring.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace dflash::common {

KimiK3DFlashTarget::KimiK3DFlashTarget(
        KimiK3Weights & weights,
        KimiK3Cache & cache,
        ggml_backend_t backend,
        DraftFeatureMirror & feature_ring,
        std::vector<int> capture_layer_ids,
        int mask_token_id,
        bool fast_rollback,
        MoeHybridStreamEngine * stream_engine,
        KimiK3RoutedOutputProvider * routed_output_provider)
    : weights_(weights),
      cache_(cache),
      backend_(backend),
      feature_ring_(feature_ring),
      capture_layer_ids_(std::move(capture_layer_ids)),
      mask_token_id_(mask_token_id),
      fast_rollback_(fast_rollback),
      stream_engine_(stream_engine),
      routed_output_provider_(routed_output_provider) {}

KimiK3DFlashTarget::~KimiK3DFlashTarget() {
    step_graph_destroy(projection_graph_);
}

bool KimiK3DFlashTarget::sync_captures(
        const KimiK3ForwardResult & result, int base_pos, int n_tokens) {
    const size_t row_values = static_cast<size_t>(weights_.n_embd) * n_tokens;
    if (result.captured_hidden.size() !=
        row_values * capture_layer_ids_.size()) {
        std::fprintf(stderr,
            "[kimi-k3-dspark] capture shape mismatch: got=%zu expected=%zu\n",
            result.captured_hidden.size(),
            row_values * capture_layer_ids_.size());
        return false;
    }
    for (size_t i = 0; i < capture_layer_ids_.size(); ++i) {
        if (!copy_host_capture_slice_to_draft_ring(
                feature_ring_, static_cast<int>(i), base_pos, n_tokens,
                result.captured_hidden.data() + i * row_values, row_values)) {
            std::fprintf(stderr,
                "[kimi-k3-dspark] feature-ring copy failed at capture %zu\n",
                i);
            return false;
        }
    }
    return true;
}

bool KimiK3DFlashTarget::forward_token(
        int32_t token, int position, std::vector<float> & logits) {
    KimiK3ForwardOptions options;
    options.capture_layer_ids = &capture_layer_ids_;
    options.read_logits = true;
    options.read_argmax = false;
    options.routed_output_provider = routed_output_provider_;
    KimiK3ForwardResult result;
    if (!kimi_k3_forward(
            backend_, weights_, cache_, {token}, position, options, result,
            stream_engine_) || !sync_captures(result, position, 1)) {
        return false;
    }
    logits = std::move(result.logits);
    committed_logits_ = logits;
    return true;
}

bool KimiK3DFlashTarget::copy_committed_logits(
        std::vector<float> & logits) const {
    if (committed_logits_.size() != static_cast<size_t>(weights_.n_vocab)) {
        return false;
    }
    logits = committed_logits_;
    return true;
}

bool KimiK3DFlashTarget::verify_batch(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<int32_t> * all_argmax,
        bool capture_ssm_intermediates) {
    if (tokens.empty()) return false;

    // The production verifier is the qualified q=7 / V=8 exact path. Legacy
    // restore+replay is kept exact by replaying its shorter accepted span one
    // row at a time rather than admitting a second batched arithmetic path.
    if (capture_ssm_intermediates && tokens.size() == 8) {
        KimiK3ForwardOptions options;
        options.capture_layer_ids = &capture_layer_ids_;
        options.capture_replay = true;
        options.read_logits = true;
        options.read_argmax = true;
        options.routed_output_provider = routed_output_provider_;
        options.exact_multirow_core = true;
        KimiK3ForwardResult result;
        if (!kimi_k3_forward(
                backend_, weights_, cache_, tokens, base_pos, options, result,
                stream_engine_) || result.argmax.size() != tokens.size() ||
            result.logits.size() !=
                static_cast<size_t>(weights_.n_vocab) * tokens.size() ||
            !sync_captures(result, base_pos, static_cast<int>(tokens.size()))) {
            return false;
        }
        pending_verify_logits_ = std::move(result.logits);
        pending_verify_rows_ = static_cast<int>(tokens.size());
        last_tok = result.argmax.back();
        if (all_argmax) *all_argmax = std::move(result.argmax);
        return true;
    }

    // Width8-capable tails take the exact grouped path above. Other V<8
    // requests execute only their active rows in exact causal order.
    std::vector<int32_t> argmax;
    argmax.reserve(tokens.size());
    std::vector<float> verify_logits;
    verify_logits.reserve(static_cast<size_t>(weights_.n_vocab) * tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        KimiK3ForwardOptions options;
        options.capture_layer_ids = &capture_layer_ids_;
        options.read_logits = true;
        options.read_argmax = true;
        options.routed_output_provider = routed_output_provider_;
        KimiK3ForwardResult result;
        const int position = base_pos + static_cast<int>(i);
        if (!kimi_k3_forward(
                backend_, weights_, cache_, {tokens[i]}, position, options,
                result, stream_engine_) || result.argmax.size() != 1 ||
            result.logits.size() != static_cast<size_t>(weights_.n_vocab) ||
            !sync_captures(result, position, 1)) {
            return false;
        }
        argmax.push_back(result.argmax.front());
        verify_logits.insert(
            verify_logits.end(), result.logits.begin(), result.logits.end());
    }
    if (capture_ssm_intermediates) {
        pending_verify_logits_ = std::move(verify_logits);
        pending_verify_rows_ = static_cast<int>(tokens.size());
    } else {
        committed_logits_.assign(
            verify_logits.end() - weights_.n_vocab, verify_logits.end());
    }
    last_tok = argmax.back();
    if (all_argmax) *all_argmax = std::move(argmax);
    return true;
}

int KimiK3DFlashTarget::preferred_physical_verify_width(
        int logical_width, int max_width) const {
    // Width8 beats scalar verify+replay for measured terminal spans of four
    // or more rows. The common loop never accepts or commits the suffix.
    return exact_fast_rollback() && max_width == 8 && logical_width >= 4 &&
            cache_.cur_pos + max_width <= cache_.max_ctx
        ? max_width : logical_width;
}

bool KimiK3DFlashTarget::snapshot_kv() {
    pending_verify_logits_.clear();
    pending_verify_rows_ = 0;
    return kimi_k3_replay_snapshot(backend_, cache_);
}

bool KimiK3DFlashTarget::restore_kv() {
    pending_verify_logits_.clear();
    pending_verify_rows_ = 0;
    return kimi_k3_replay_restore(backend_, cache_);
}

bool KimiK3DFlashTarget::supports_fast_rollback() const {
    return fast_rollback_ && cache_.max_verify_tokens >= 8 &&
           cache_.snapshot_valid && cache_.replay_valid;
}

bool KimiK3DFlashTarget::exact_fast_rollback() const {
    return fast_rollback_ && cache_.max_verify_tokens >= 8;
}

bool KimiK3DFlashTarget::rollback_to(int base_pos, int commit_n) {
    if (commit_n <= 0 || commit_n > pending_verify_rows_ ||
        pending_verify_logits_.size() !=
            static_cast<size_t>(weights_.n_vocab) * pending_verify_rows_ ||
        !kimi_k3_replay_commit(
            backend_, weights_, cache_, base_pos, commit_n)) {
        return false;
    }
    const auto begin = pending_verify_logits_.begin() +
        static_cast<std::ptrdiff_t>(commit_n - 1) * weights_.n_vocab;
    committed_logits_.assign(begin, begin + weights_.n_vocab);
    pending_verify_logits_.clear();
    pending_verify_rows_ = 0;
    return true;
}

bool KimiK3DFlashTarget::is_eos(int token) const {
    return token == weights_.eos_token_id;
}

bool KimiK3DFlashTarget::embed_tokens(
        const int32_t * tokens, int n, float * out) const {
    if (!tokens || n <= 0 || !out) return false;
    std::vector<float> hidden(
        static_cast<size_t>(weights_.n_embd) * static_cast<size_t>(n));
    if (!kimi_k3_read_token_embeddings_on_host(
            weights_, std::vector<int32_t>(tokens, tokens + n), hidden) ||
        hidden.size() != static_cast<size_t>(weights_.n_embd) * n) {
        return false;
    }
    std::copy(hidden.begin(), hidden.end(), out);
    return true;
}

bool KimiK3DFlashTarget::build_projection_graph(int n_tokens) {
    StepGraph & graph = projection_graph_;
    step_graph_free(graph);
    ggml_init_params params{};
    params.mem_size = 4ull * 1024ull * 1024ull;
    params.no_alloc = true;
    graph.ctx = ggml_init(params);
    if (!graph.ctx) return false;
    graph.gf = ggml_new_graph_custom(graph.ctx, 256, false);
    graph.hidden_input = ggml_new_tensor_2d(
        graph.ctx, GGML_TYPE_F32, weights_.n_embd, n_tokens);
    ggml_set_input(graph.hidden_input);
    graph.logits = ggml_mul_mat(
        graph.ctx, weights_.output, graph.hidden_input);
    graph.argmax_tokens = ggml_argmax(graph.ctx, graph.logits);
    ggml_set_output(graph.logits);
    ggml_set_output(graph.argmax_tokens);
    ggml_build_forward_expand(graph.gf, graph.logits);
    ggml_build_forward_expand(graph.gf, graph.argmax_tokens);
    if (!graph.alloc) {
        graph.alloc = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
    }
    return graph.alloc && ggml_gallocr_alloc_graph(graph.alloc, graph.gf);
}

bool KimiK3DFlashTarget::project_hidden_to_tokens(
        const float * hidden, int n_tokens,
        std::vector<int32_t> & tokens_out) {
    if (!hidden || n_tokens <= 0 || !build_projection_graph(n_tokens)) {
        return false;
    }
    ggml_backend_tensor_set(
        projection_graph_.hidden_input, hidden, 0,
        sizeof(float) * static_cast<size_t>(weights_.n_embd) * n_tokens);
    if (ggml_backend_graph_compute(backend_, projection_graph_.gf) !=
        GGML_STATUS_SUCCESS) {
        return false;
    }
    tokens_out.resize(static_cast<size_t>(n_tokens));
    ggml_backend_tensor_get(
        projection_graph_.argmax_tokens, tokens_out.data(), 0,
        sizeof(int32_t) * tokens_out.size());
    return true;
}

bool KimiK3DFlashTarget::project_hidden_to_logits(
        const float * hidden, int n_tokens, std::vector<float> & logits_out) {
    if (!hidden || n_tokens <= 0 || !build_projection_graph(n_tokens)) {
        return false;
    }
    ggml_backend_tensor_set(
        projection_graph_.hidden_input, hidden, 0,
        sizeof(float) * static_cast<size_t>(weights_.n_embd) * n_tokens);
    if (ggml_backend_graph_compute(backend_, projection_graph_.gf) !=
        GGML_STATUS_SUCCESS) {
        return false;
    }
    logits_out.resize(static_cast<size_t>(weights_.n_vocab) * n_tokens);
    ggml_backend_tensor_get(
        projection_graph_.logits, logits_out.data(), 0,
        sizeof(float) * logits_out.size());
    return true;
}

ggml_tensor * KimiK3DFlashTarget::lm_head_tensor() { return weights_.output; }
ggml_tensor * KimiK3DFlashTarget::gpu_embd_table() { return weights_.tok_embd; }
ggml_backend_t KimiK3DFlashTarget::fused_head_backend() { return backend_; }
int KimiK3DFlashTarget::hidden_size() const { return weights_.n_embd; }
int KimiK3DFlashTarget::mask_token_id() const { return mask_token_id_; }
const std::vector<int> & KimiK3DFlashTarget::capture_layer_ids() const {
    return capture_layer_ids_;
}

} // namespace dflash::common
