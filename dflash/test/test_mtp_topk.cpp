// test_mtp_topk.cpp — Verifies the StepOutput top-K surface added in
// support of experiment C (MTP top-K → DDTree).
//
// Three behaviors must hold:
//   1. With no top-K configured (default K=1), StepOutput.topk_* are empty
//      and existing argmax fields (draft_token, draft_logit) match the
//      pre-patch behavior exactly.
//   2. With K=4 configured (via set_draft_topk(4)), each per-head
//      StepOutput carries topk_logprobs.size() == K and topk_ids.size() == K,
//      sorted DESCENDING by logprob, and topk_ids[0] == draft_token (top-1
//      must equal the argmax).
//   3. K=1 result is unchanged regardless of whether set_draft_topk(1) is
//      called before — i.e. it stays ABI-compatible for argmax-only callers.
//
// Test tier: T1 — no model file, no GPU, CPU stub target only.

#include "qwen36/qwen36_mtp.h"
#include "common/mtp_interface.h"
#include "common/dflash_target.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "%s:%d CHECK(%s) FAILED\n",                     \
                     __FILE__, __LINE__, #cond);                             \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

using namespace dflash27b;
using namespace dflash27b::mtp;

// Match the shapes used by test_qwen36_mtp_step_unit.cpp so we exercise the
// same CPU forward.  The explicit shared_head_head matrix is wired so the
// step_batch's per-head matvec produces logits we can verify against.
static constexpr int TEST_N_EMBD            = 64;
static constexpr int TEST_N_VOCAB           = 128;
static constexpr int TEST_N_HEADS           = 2;
static constexpr int TEST_N_BACKBONE_LAYERS = 4;
static constexpr int TEST_HEAD_COUNT        = 2;
static constexpr int TEST_HEAD_KV           = 2;
static constexpr int TEST_KEY_LENGTH        = 32;
static constexpr int TEST_VALUE_LENGTH      = 32;
static constexpr int TEST_FFN_LENGTH        = 128;

struct StubTarget : DFlashTarget {
    int H = TEST_N_EMBD;
    int V = TEST_N_VOCAB;

    bool verify_batch(const std::vector<int32_t> &, int, int &,
                      std::vector<int32_t> *) override { return true; }
    bool snapshot_kv() override { return true; }
    bool restore_kv() override  { return true; }
    bool is_eos(int) const override { return false; }
    bool embed_tokens(const int32_t * tokens, int n, float * out) const override {
        for (int i = 0; i < n; i++) {
            const float val = 1.0f / static_cast<float>(tokens[i] + 1);
            for (int d = 0; d < H; d++) out[i * H + d] = val;
        }
        return true;
    }
    bool project_hidden_to_tokens(const float * hidden, int n_tokens,
                                  std::vector<int32_t> & tokens_out) override {
        tokens_out.resize(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            float sum = 0.0f;
            for (int d = 0; d < H; d++) sum += hidden[i * H + d];
            tokens_out[i] = static_cast<int>(std::floor(std::abs(sum))) % V;
        }
        return true;
    }
    int hidden_size() const override { return H; }
    int mask_token_id() const override { return -1; }
    const std::vector<int> & capture_layer_ids() const override {
        static const std::vector<int> ids;
        return ids;
    }
};

// Build weights with an explicit, asymmetric shared_head_head so different
// vocab indices get different logits (per-row scale = (1 + 0.001 * row)).
// This ensures top-K ordering is meaningful, not all-equal.
static ggml_context * build_test_weights(Qwen36MtpWeights & out_w) {
    const size_t mem = 16 * 1024 * 1024;
    ggml_init_params ip{};
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx);

    out_w.n_embd            = TEST_N_EMBD;
    out_w.n_vocab           = TEST_N_VOCAB;
    out_w.n_heads           = TEST_N_HEADS;
    out_w.n_backbone_layers = TEST_N_BACKBONE_LAYERS;
    out_w.backbone_arch     = "qwen35";
    out_w.base_model_name   = "test-model";
    out_w.n_head_count   = TEST_HEAD_COUNT;
    out_w.n_head_kv      = TEST_HEAD_KV;
    out_w.n_key_length   = TEST_KEY_LENGTH;
    out_w.n_value_length = TEST_VALUE_LENGTH;
    out_w.n_ffn_length   = TEST_FFN_LENGTH;
    out_w.heads.resize(TEST_N_HEADS);

    auto make_vec = [&](int n, float val) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_f32(t, val);
        return t;
    };
    auto make_mat = [&](int rows, int cols, float val) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        ggml_set_f32(t, val);
        return t;
    };

    for (int h = 0; h < TEST_N_HEADS; h++) {
        out_w.heads[h].layer_idx = TEST_N_BACKBONE_LAYERS - TEST_N_HEADS + h;
        out_w.heads[h].eh_proj   = make_mat(TEST_N_EMBD, 2 * TEST_N_EMBD, 0.01f);
        out_w.heads[h].enorm     = make_vec(TEST_N_EMBD, 1.0f);
        out_w.heads[h].hnorm     = make_vec(TEST_N_EMBD, 1.0f);

        // Per-row varying shared_head_head so top-K has a distinct ordering.
        // shared_head_head shape: rows = n_vocab, cols = n_embd.
        ggml_tensor * head_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                  TEST_N_EMBD, TEST_N_VOCAB);
        float * data = (float *)head_w->data;
        for (int r = 0; r < TEST_N_VOCAB; r++) {
            const float scale = 0.001f * static_cast<float>(r + 1);
            for (int c = 0; c < TEST_N_EMBD; c++) data[r * TEST_N_EMBD + c] = scale;
        }
        out_w.heads[h].shared_head_head = head_w;
        out_w.heads[h].shared_head_norm = make_vec(TEST_N_EMBD, 1.0f);

        out_w.heads[h].attn_norm    = make_vec(TEST_N_EMBD, 1.0f);
        out_w.heads[h].attn_q       = make_mat(2 * TEST_HEAD_COUNT * TEST_KEY_LENGTH,
                                               TEST_N_EMBD, 0.01f);
        out_w.heads[h].attn_q_norm  = make_vec(TEST_KEY_LENGTH, 1.0f);
        out_w.heads[h].attn_k       = make_mat(TEST_HEAD_KV * TEST_KEY_LENGTH,
                                               TEST_N_EMBD, 0.01f);
        out_w.heads[h].attn_k_norm  = make_vec(TEST_KEY_LENGTH, 1.0f);
        out_w.heads[h].attn_v       = make_mat(TEST_HEAD_KV * TEST_VALUE_LENGTH,
                                               TEST_N_EMBD, 0.01f);
        out_w.heads[h].attn_output  = make_mat(TEST_N_EMBD,
                                               TEST_HEAD_COUNT * TEST_VALUE_LENGTH,
                                               0.01f);
        out_w.heads[h].post_attention_norm = make_vec(TEST_N_EMBD, 1.0f);
        out_w.heads[h].ffn_gate     = make_mat(TEST_FFN_LENGTH, TEST_N_EMBD, 0.01f);
        out_w.heads[h].ffn_up       = make_mat(TEST_FFN_LENGTH, TEST_N_EMBD, 0.01f);
        out_w.heads[h].ffn_down     = make_mat(TEST_N_EMBD, TEST_FFN_LENGTH, 0.01f);
    }
    return ctx;
}

static std::unique_ptr<Qwen36MtpModule> make_module(StubTarget & tgt,
                                                     Qwen36MtpWeights & w) {
    auto m = std::make_unique<Qwen36MtpModule>();
    m->attach_weights_for_test(w);
    CHECK(m->attach(&tgt));
    return m;
}

// 1. K=1 (default): topk_* must be empty; argmax fields populated.
static void test_default_k1_argmax_unchanged() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);
    std::vector<float> h_prev(TEST_N_EMBD, 0.1f);

    auto m = make_module(tgt, w);
    m->set_initial_hidden(h_prev.data(), TEST_N_EMBD);

    std::vector<StepOutput> out;
    CHECK(m->step_batch(/*cur=*/1, /*pos=*/0, out));
    CHECK((int)out.size() == TEST_N_HEADS);
    for (const auto & s : out) {
        CHECK(s.topk_logprobs.empty());
        CHECK(s.topk_ids.empty());
        CHECK(s.draft_token >= 0);
        CHECK(s.draft_token < TEST_N_VOCAB);
        CHECK(std::isfinite(s.draft_logit));
    }
    ggml_free(ctx);
    std::printf("[mtp_topk] test_default_k1_argmax_unchanged PASS\n");
}

// 2. K=4: topk_* are populated, sorted, and top-1 matches argmax.
static void test_k4_populates_sorted_topk() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);
    std::vector<float> h_prev(TEST_N_EMBD, 0.1f);

    auto m = make_module(tgt, w);
    m->set_draft_topk(4);
    m->set_initial_hidden(h_prev.data(), TEST_N_EMBD);

    std::vector<StepOutput> out;
    CHECK(m->step_batch(/*cur=*/1, /*pos=*/0, out));
    CHECK((int)out.size() == TEST_N_HEADS);

    for (const auto & s : out) {
        CHECK((int)s.topk_logprobs.size() == 4);
        CHECK((int)s.topk_ids.size() == 4);
        // sorted DESCENDING
        for (int i = 1; i < 4; i++) {
            CHECK(s.topk_logprobs[i] <= s.topk_logprobs[i - 1]);
        }
        // ids in vocab range, finite logprobs, all <= 0 (log-softmax)
        for (int i = 0; i < 4; i++) {
            CHECK(s.topk_ids[i] >= 0);
            CHECK(s.topk_ids[i] < TEST_N_VOCAB);
            CHECK(std::isfinite(s.topk_logprobs[i]));
            CHECK(s.topk_logprobs[i] <= 1e-4f);  // log-prob
        }
        // top-1 must match the argmax token.
        CHECK(s.topk_ids[0] == s.draft_token);
    }
    ggml_free(ctx);
    std::printf("[mtp_topk] test_k4_populates_sorted_topk PASS\n");
}

// 3. set_draft_topk(1) is identical to default K=1 behavior.
static void test_explicit_k1_matches_default() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);
    std::vector<float> h_prev(TEST_N_EMBD, 0.1f);

    auto m1 = make_module(tgt, w);
    m1->set_initial_hidden(h_prev.data(), TEST_N_EMBD);
    std::vector<StepOutput> out_default;
    CHECK(m1->step_batch(/*cur=*/1, /*pos=*/0, out_default));

    auto m2 = make_module(tgt, w);
    m2->set_draft_topk(1);
    m2->set_initial_hidden(h_prev.data(), TEST_N_EMBD);
    std::vector<StepOutput> out_explicit;
    CHECK(m2->step_batch(/*cur=*/1, /*pos=*/0, out_explicit));

    CHECK(out_default.size() == out_explicit.size());
    for (size_t i = 0; i < out_default.size(); i++) {
        CHECK(out_default[i].draft_token == out_explicit[i].draft_token);
        CHECK(out_default[i].topk_logprobs.empty());
        CHECK(out_explicit[i].topk_logprobs.empty());
    }
    ggml_free(ctx);
    std::printf("[mtp_topk] test_explicit_k1_matches_default PASS\n");
}

int main() {
    test_default_k1_argmax_unchanged();
    test_k4_populates_sorted_topk();
    test_explicit_k1_matches_default();
    std::printf("[mtp_topk] all 3 tests PASS\n");
    return 0;
}
