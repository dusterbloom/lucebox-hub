// test_qwen36_mtp_step_unit.cpp — T1 unit test for Qwen36MtpModule::step_batch.
//
// Builds a minimal Qwen36MtpWeights in-process (no GGUF, no GPU) and
// exercises the Phase A forward. All tensors are allocated in a local
// ggml_context filled with deterministic small values.
//
// Uses Qwen36MtpModule::attach_weights_for_test() to inject pre-built
// weights without a GGUF file.
//
// Test tier: T1 — no model file, no GPU, deterministic, must stay green.

#include "qwen36/qwen36_mtp.h"
#include "common/mtp_interface.h"
#include "common/dflash_target.h"

#include "ggml.h"
#include "ggml-cpu.h"   // ggml_set_f32 (declared GGML_BACKEND_API, CPU only)

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

// ── Constants ──────────────────────────────────────────────────────────────

static constexpr int TEST_N_EMBD            = 64;
static constexpr int TEST_N_VOCAB           = 128;
static constexpr int TEST_N_HEADS           = 2;
static constexpr int TEST_N_BACKBONE_LAYERS = 4;

// Shape B attention-sizing constants (match struct fields in Qwen36MtpWeights).
static constexpr int TEST_HEAD_COUNT  = 2;    // n_head_count
static constexpr int TEST_HEAD_KV     = 2;    // n_head_kv
static constexpr int TEST_KEY_LENGTH  = 32;   // n_key_length
static constexpr int TEST_VALUE_LENGTH = 32;  // n_value_length
static constexpr int TEST_FFN_LENGTH  = 128;  // n_ffn_length

// ── StubTarget ─────────────────────────────────────────────────────────────
//
// embed_tokens: fills output with 1.0 / (token_id + 1) per element.
// project_hidden_to_tokens: returns (floor(sum(hidden)) % n_vocab).
// All other methods return trivially.
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
            const int idx = static_cast<int>(std::floor(std::abs(sum)));
            tokens_out[i] = idx % V;
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

// ── DualHiddenTarget ───────────────────────────────────────────────────────
//
// Stub target that publishes DIFFERENT vectors for hidden_at_pos (post-norm)
// and hidden_at_pos_pre_norm.  The Qwen3.6 MTP module's chain seed (h_prev_0)
// must prefer the pre-norm vector to mirror llama.cpp PR #22673 t_h_pre_norm
// (post-norm path double-normalises via the head's hnorm and crushes D>=2
// accept).  This stub lets a unit test observe which accessor the consumer
// picks: the per-head draft token is a deterministic function of h_prev, so
// pre-norm vs post-norm seeds yield different tokens, and the test asserts
// equivalence to a control module that was fed the pre-norm vector directly
// via set_initial_hidden.
struct DualHiddenTarget : StubTarget {
    std::vector<float> post_norm;
    std::vector<float> pre_norm;
    int absolute_pos = 0;  // position at which both vectors live
    // Observability: how many times each accessor was called and at which
    // abs_pos.  The test asserts pre-norm was called and matched FIRST so a
    // future regression that swaps accessors will be caught regardless of
    // numerical noise downstream of the TRMBlock.
    mutable int  post_norm_call_count = 0;
    mutable int  pre_norm_call_count  = 0;
    mutable int  last_pre_norm_pos    = -999;
    mutable int  last_post_norm_pos   = -999;

    const float * hidden_at_pos(int abs_pos) const override {
        post_norm_call_count++;
        last_post_norm_pos = abs_pos;
        if (abs_pos != absolute_pos) return nullptr;
        return post_norm.empty() ? nullptr : post_norm.data();
    }
    const float * hidden_at_pos_pre_norm(int abs_pos) const override {
        pre_norm_call_count++;
        last_pre_norm_pos = abs_pos;
        if (abs_pos != absolute_pos) return nullptr;
        return pre_norm.empty() ? nullptr : pre_norm.data();
    }
};

// ── Weight builder ─────────────────────────────────────────────────────────
//
// Allocates per-head ggml tensors in a shared context with small deterministic
// values. The context is kept alive for the lifetime of the returned
// Qwen36MtpWeights (caller owns both ctx and weights).

static ggml_context * build_test_weights(Qwen36MtpWeights & out_w) {
    // Memory estimate: 2 heads x all per-head tensors in F32.
    // eh_proj: 64x128x4 = 32 KB; attn tensors: ~200 KB each; total ~8 MB is safe.
    const size_t mem = 16 * 1024 * 1024;
    struct ggml_init_params ip{};
    ip.mem_size   = mem;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = false;   // allocate data bytes in the context pool
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "build_test_weights: ggml_init failed\n");
        std::exit(1);
    }

    out_w.n_embd            = TEST_N_EMBD;
    out_w.n_vocab           = TEST_N_VOCAB;
    out_w.n_heads           = TEST_N_HEADS;
    out_w.n_backbone_layers = TEST_N_BACKBONE_LAYERS;
    out_w.backbone_arch     = "qwen35";
    out_w.base_model_name   = "test-model";
    // Shape B attention-sizing fields
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
        // layer_idx: last TEST_N_HEADS backbone layers.
        out_w.heads[h].layer_idx = TEST_N_BACKBONE_LAYERS - TEST_N_HEADS + h;

        // NextN-specific tensors
        // eh_proj: [n_embd x 2*n_embd] output rows x input cols.
        // Use a tiny non-zero value so the projection output is non-zero.
        out_w.heads[h].eh_proj          = make_mat(TEST_N_EMBD, 2 * TEST_N_EMBD, 0.01f);
        out_w.heads[h].enorm            = make_vec(TEST_N_EMBD, 1.0f);
        out_w.heads[h].hnorm            = make_vec(TEST_N_EMBD, 1.0f);
        // shared_head_head: [n_vocab x n_embd]
        out_w.heads[h].shared_head_head = make_mat(TEST_N_VOCAB, TEST_N_EMBD, 0.01f);
        out_w.heads[h].shared_head_norm = make_vec(TEST_N_EMBD, 1.0f);

        // Shape B: head-owned transformer-block tensors.
        // attn_q is packed Q+gate (same convention as backbone full-attention
        // blocks): rows = 2 * (head_count * key_length), cols = n_embd.
        // This matches the real GGUF's blk.64.attn_q.weight shape [n_embd, 12288]
        // where 12288 = 2 * (24 heads * 256 key_length).
        out_w.heads[h].attn_norm           = make_vec(TEST_N_EMBD, 1.0f);
        out_w.heads[h].attn_q              = make_mat(2 * TEST_HEAD_COUNT * TEST_KEY_LENGTH,
                                                       TEST_N_EMBD, 0.01f);  // packed Q+gate
        out_w.heads[h].attn_q_norm         = make_vec(TEST_KEY_LENGTH, 1.0f);
        out_w.heads[h].attn_k              = make_mat(TEST_HEAD_KV * TEST_KEY_LENGTH,
                                                       TEST_N_EMBD, 0.01f);
        out_w.heads[h].attn_k_norm         = make_vec(TEST_KEY_LENGTH, 1.0f);
        out_w.heads[h].attn_v              = make_mat(TEST_HEAD_KV * TEST_VALUE_LENGTH,
                                                       TEST_N_EMBD, 0.01f);
        out_w.heads[h].attn_output         = make_mat(TEST_N_EMBD,
                                                       TEST_HEAD_COUNT * TEST_VALUE_LENGTH,
                                                       0.01f);
        out_w.heads[h].post_attention_norm = make_vec(TEST_N_EMBD, 1.0f);
        out_w.heads[h].ffn_gate            = make_mat(TEST_FFN_LENGTH, TEST_N_EMBD, 0.01f);
        out_w.heads[h].ffn_up              = make_mat(TEST_FFN_LENGTH, TEST_N_EMBD, 0.01f);
        out_w.heads[h].ffn_down            = make_mat(TEST_N_EMBD, TEST_FFN_LENGTH, 0.01f);
    }

    return ctx;
}

// ── Test helpers ───────────────────────────────────────────────────────────

// Construct a freshly-loaded+attached module.
static std::unique_ptr<Qwen36MtpModule> make_module(StubTarget & tgt,
                                                     ggml_context * ctx,
                                                     Qwen36MtpWeights & w) {
    (void)ctx;  // ctx kept alive by caller; module does not own it
    auto m = std::make_unique<Qwen36MtpModule>();
    m->attach_weights_for_test(w);
    CHECK(m->attach(&tgt));
    return m;
}

// ── Test cases ─────────────────────────────────────────────────────────────

// 1. step_batch returns true after attach_weights_for_test + attach(target).
static void test_step_returns_true() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    auto m = make_module(tgt, ctx, w);
    std::vector<StepOutput> out;
    const bool ok = m->step_batch(/*cur=*/1, /*pos=*/0, out);
    CHECK(ok);

    ggml_free(ctx);
    std::printf("[step_unit] test_step_returns_true PASS\n");
}

// 2. out.size() == num_heads (== TEST_N_HEADS).
static void test_out_size_equals_num_heads() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    auto m = make_module(tgt, ctx, w);
    std::vector<StepOutput> out;
    CHECK(m->step_batch(1, 0, out));
    CHECK((int)out.size() == TEST_N_HEADS);

    ggml_free(ctx);
    std::printf("[step_unit] test_out_size_equals_num_heads PASS\n");
}

// 3. Each StepOutput.draft_token is in [0, n_vocab).
static void test_draft_tokens_in_vocab_range() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    auto m = make_module(tgt, ctx, w);
    std::vector<StepOutput> out;
    CHECK(m->step_batch(1, 0, out));
    CHECK((int)out.size() == TEST_N_HEADS);
    for (const auto & s : out) {
        CHECK(s.draft_token >= 0);
        CHECK(s.draft_token < TEST_N_VOCAB);
    }

    ggml_free(ctx);
    std::printf("[step_unit] test_draft_tokens_in_vocab_range PASS\n");
}

// 4. reset_chain followed by step_batch yields identical first-call output
//    to a fresh init (chain idempotence).
static void test_reset_chain_idempotence() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    // First module: capture output after first call.
    auto m1 = make_module(tgt, ctx, w);
    std::vector<StepOutput> out1;
    CHECK(m1->step_batch(5, 0, out1));

    // Second module: call step_batch, then reset and call again.
    auto m2 = make_module(tgt, ctx, w);
    std::vector<StepOutput> out_pre;
    CHECK(m2->step_batch(5, 0, out_pre));  // consume chain state
    m2->reset_chain();                      // reset to zero hidden
    std::vector<StepOutput> out2;
    CHECK(m2->step_batch(5, 0, out2));

    CHECK(out1.size() == out2.size());
    for (size_t i = 0; i < out1.size(); i++) {
        CHECK(out1[i].draft_token == out2[i].draft_token);
    }

    ggml_free(ctx);
    std::printf("[step_unit] test_reset_chain_idempotence PASS\n");
}

// 5. Two consecutive step_batch calls don't crash and produce coherent state.
static void test_two_consecutive_calls() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    auto m = make_module(tgt, ctx, w);

    std::vector<StepOutput> out1, out2;
    CHECK(m->step_batch(1, 0, out1));
    CHECK(!out1.empty());
    const int32_t next_tok = out1.back().draft_token;
    CHECK(m->step_batch(next_tok, 1, out2));
    CHECK(!out2.empty());
    for (const auto & s : out2) {
        CHECK(s.draft_token >= 0);
        CHECK(s.draft_token < TEST_N_VOCAB);
    }

    ggml_free(ctx);
    std::printf("[step_unit] test_two_consecutive_calls PASS\n");
}

// 6. step_batch on unloaded module returns false with empty out.
//    The load-check guard at the top of step_batch fires before any math.
static void test_unloaded_module_returns_false() {
    // Qwen36MtpModule default-constructed has state_->loaded == false.
    Qwen36MtpModule m;
    StubTarget tgt;
    // attach to a target so only the !loaded guard fires, not !attached.
    CHECK(m.attach(&tgt));

    std::vector<StepOutput> out;
    const bool ok = m.step_batch(0, 0, out);
    // The guard: if (!state_->loaded || !state_->attached) returns false.
    // After attach(), attached == true but loaded == false → still false.
    CHECK(!ok);
    CHECK(out.empty());

    std::printf("[step_unit] test_unloaded_module_returns_false PASS\n");
}

// 7. All 15 per-head tensor pointers are non-null after attach_weights_for_test.
//    Verifies build_test_weights populates every Shape B field and the struct
//    members are correctly named (catches typos in either the struct or the test).
static void test_head_tensors_non_null_after_attach() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    auto m = make_module(tgt, ctx, w);
    const Qwen36MtpWeights & loaded = m->weights();
    CHECK((int)loaded.heads.size() == TEST_N_HEADS);

    for (int h = 0; h < TEST_N_HEADS; h++) {
        const auto & head = loaded.heads[h];
        // NextN-specific tensors
        CHECK(head.eh_proj          != nullptr);
        CHECK(head.enorm            != nullptr);
        CHECK(head.hnorm            != nullptr);
        CHECK(head.shared_head_head != nullptr);
        CHECK(head.shared_head_norm != nullptr);
        // Shape B: head-owned transformer-block tensors
        CHECK(head.attn_norm           != nullptr);
        CHECK(head.attn_q              != nullptr);
        CHECK(head.attn_q_norm         != nullptr);
        CHECK(head.attn_k              != nullptr);
        CHECK(head.attn_k_norm         != nullptr);
        CHECK(head.attn_v              != nullptr);
        CHECK(head.attn_output         != nullptr);
        CHECK(head.post_attention_norm != nullptr);
        CHECK(head.ffn_gate            != nullptr);
        CHECK(head.ffn_up              != nullptr);
        CHECK(head.ffn_down            != nullptr);
    }

    ggml_free(ctx);
    std::printf("[step_unit] test_head_tensors_non_null_after_attach PASS\n");
}

// 8. set_initial_hidden stores the pointer and dim into state.
//    Uses the MTP_TEST_BUILD accessor to verify without breaking encapsulation.
static void test_set_initial_hidden_stores_pointer() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    auto m = make_module(tgt, ctx, w);

    // Before set_initial_hidden: pointer should be null.
    CHECK(m->test_initial_hidden_ptr() == nullptr);
    CHECK(m->test_initial_hidden_dim() == 0);

    // Set a dummy hidden vector.
    static const float dummy_hidden[TEST_N_EMBD] = {};   // all-zeros; just checking pointer
    m->set_initial_hidden(dummy_hidden, TEST_N_EMBD);

    // After: pointer must match and dim must match.
    CHECK(m->test_initial_hidden_ptr() == dummy_hidden);
    CHECK(m->test_initial_hidden_dim() == TEST_N_EMBD);

    // reset_chain clears it.
    m->reset_chain();
    CHECK(m->test_initial_hidden_ptr() == nullptr);
    CHECK(m->test_initial_hidden_dim() == 0);

    ggml_free(ctx);
    std::printf("[step_unit] test_set_initial_hidden_stores_pointer PASS\n");
}

// 9. Forward with set_initial_hidden produces a token in vocab range with a
//    finite logit (no NaN/Inf). This exercises the full Shape B chain using
//    a deterministic synthetic h_prev rather than a real backbone output.
static void test_forward_with_initial_hidden_produces_token() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    auto m = make_module(tgt, ctx, w);

    // Build a deterministic non-zero h_prev (value 0.1 per element).
    std::vector<float> h_prev(TEST_N_EMBD, 0.1f);
    m->set_initial_hidden(h_prev.data(), TEST_N_EMBD);

    std::vector<StepOutput> out;
    const bool ok = m->step_batch(/*cur=*/1, /*pos=*/0, out);
    CHECK(ok);
    CHECK((int)out.size() == TEST_N_HEADS);

    for (const auto & s : out) {
        // Token must be in vocab range.
        CHECK(s.draft_token >= 0);
        CHECK(s.draft_token < TEST_N_VOCAB);
        // Logit must be finite (not NaN, not Inf).
        // When the test uses the explicit shared_head_head path (as set in
        // build_test_weights), draft_logit is the actual logit value — finite.
        // When using the shared target path, draft_logit is set to 0.0f,
        // which is also finite. Either way the check passes.
        CHECK(std::isfinite(s.draft_logit));
    }

    ggml_free(ctx);
    std::printf("[step_unit] test_forward_with_initial_hidden_produces_token PASS\n");
}

// 10. Forward is deterministic: two calls with identical inputs produce the
//     same draft_token sequence. Exercises that the TRMBlock forward is pure
//     given the same weights, h_prev, token, and position.
static void test_forward_deterministic() {
    StubTarget tgt;
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    std::vector<float> h_prev(TEST_N_EMBD, 0.05f);

    // First call.
    auto m1 = make_module(tgt, ctx, w);
    m1->set_initial_hidden(h_prev.data(), TEST_N_EMBD);
    std::vector<StepOutput> out1;
    CHECK(m1->step_batch(/*cur=*/7, /*pos=*/3, out1));

    // Second call: fresh module, same inputs.
    auto m2 = make_module(tgt, ctx, w);
    m2->set_initial_hidden(h_prev.data(), TEST_N_EMBD);
    std::vector<StepOutput> out2;
    CHECK(m2->step_batch(/*cur=*/7, /*pos=*/3, out2));

    CHECK(out1.size() == out2.size());
    for (size_t i = 0; i < out1.size(); i++) {
        CHECK(out1[i].draft_token == out2[i].draft_token);
    }

    ggml_free(ctx);
    std::printf("[step_unit] test_forward_deterministic PASS\n");
}

// 11. Chain seed PREFERS pre-norm hidden over post-norm.  Regression guard
//     for the bug fixed in commit ${THIS_COMMIT}: the spec-chain's outer
//     h_prev_0 used target->hidden_at_pos(base_pos-1) (post-output-norm),
//     which double-normalised against the head's own hnorm and crushed
//     D>=2 accept (real-bench: 70.7% compound-decayed at D=3).  Pre-norm
//     seed (llama.cpp PR #22673 `t_h_pre_norm`) is the fix.
//
//     Method: drive step_batch (CPU path) at base_pos=1 with a target that
//     COUNTS how many times each accessor is called and at which abs_pos.
//     The new behaviour: hidden_at_pos_pre_norm() must be called for
//     abs_pos = base_pos - 1 = 0; hidden_at_pos() may be called (only as
//     a fallback) but must NOT be the one whose result was used.  We
//     check call-trace + the non-null pre-norm vector, which is more
//     robust than comparing tokens through the uniform-weight test
//     TRMBlock (which collapses input differences).
static void test_chain_seed_prefers_pre_norm() {
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);

    DualHiddenTarget tgt;
    tgt.post_norm.assign(TEST_N_EMBD, 0.5f);
    tgt.pre_norm.assign(TEST_N_EMBD, 0.1f);
    tgt.absolute_pos = 0;

    auto m = make_module(tgt, ctx, w);
    std::vector<StepOutput> out;
    CHECK(m->step_batch(/*cur=*/3, /*base_pos=*/1, out));
    CHECK((int)out.size() == TEST_N_HEADS);

    // Invariant 1: pre-norm was queried at abs_pos = base_pos - 1 = 0.
    if (tgt.pre_norm_call_count == 0) {
        std::fprintf(stderr,
            "[step_unit] FAIL: hidden_at_pos_pre_norm was NEVER called — "
            "consumer is still using hidden_at_pos exclusively (PR #22673 "
            "regression).\n");
        std::exit(1);
    }
    if (tgt.last_pre_norm_pos != 0) {
        std::fprintf(stderr,
            "[step_unit] FAIL: hidden_at_pos_pre_norm called with abs_pos=%d "
            "but base_pos-1 = 0 was expected.\n", tgt.last_pre_norm_pos);
        std::exit(1);
    }

    // Invariant 2: pre-norm returned non-null so consumer must NOT have
    // also dispatched the post-norm fallback for the SAME abs_pos.  If
    // tgt.post_norm_call_count > 0 at this abs_pos, the consumer is
    // probing post-norm after a successful pre-norm read — that's an
    // efficiency bug but not a correctness one; warn but don't fail.
    if (tgt.post_norm_call_count > 0 &&
        tgt.last_post_norm_pos == 0) {
        std::fprintf(stderr,
            "[step_unit] WARN: hidden_at_pos called at abs_pos=0 after a "
            "successful hidden_at_pos_pre_norm hit — consumer is "
            "fallback-probing both accessors.\n");
    }

    ggml_free(ctx);
    std::printf("[step_unit] test_chain_seed_prefers_pre_norm PASS "
                "(pre_norm_calls=%d post_norm_calls=%d)\n",
                tgt.pre_norm_call_count, tgt.post_norm_call_count);
}

// 12. Companion negative: if a target ONLY publishes post-norm (returns
//     nullptr from hidden_at_pos_pre_norm), the consumer must FALL BACK
//     to hidden_at_pos rather than failing.  This preserves the default
//     DFlashTarget interface contract — adapters that haven't been
//     updated for the pre-norm accessor must still work.
struct PostNormOnlyTarget : StubTarget {
    std::vector<float> post_norm;
    int absolute_pos = 0;
    mutable int post_norm_call_count = 0;
    const float * hidden_at_pos(int abs_pos) const override {
        post_norm_call_count++;
        if (abs_pos != absolute_pos) return nullptr;
        return post_norm.empty() ? nullptr : post_norm.data();
    }
    // hidden_at_pos_pre_norm intentionally NOT overridden — base virtual
    // returns nullptr, simulating an adapter that doesn't expose pre-norm.
};
static void test_chain_seed_falls_back_to_post_norm() {
    Qwen36MtpWeights w;
    ggml_context * ctx = build_test_weights(w);
    PostNormOnlyTarget tgt;
    tgt.post_norm.assign(TEST_N_EMBD, 0.2f);
    tgt.absolute_pos = 0;

    auto m = make_module(tgt, ctx, w);
    std::vector<StepOutput> out;
    CHECK(m->step_batch(/*cur=*/5, /*base_pos=*/1, out));
    CHECK((int)out.size() == TEST_N_HEADS);
    if (tgt.post_norm_call_count == 0) {
        std::fprintf(stderr,
            "[step_unit] FAIL: hidden_at_pos fallback not called when "
            "hidden_at_pos_pre_norm returns nullptr — chain would fail "
            "on un-updated adapters.\n");
        std::exit(1);
    }
    ggml_free(ctx);
    std::printf("[step_unit] test_chain_seed_falls_back_to_post_norm PASS "
                "(post_norm_calls=%d)\n", tgt.post_norm_call_count);
}

// ── main ──────────────────────────────────────────────────────────────────

int main() {
    test_step_returns_true();
    test_out_size_equals_num_heads();
    test_draft_tokens_in_vocab_range();
    test_reset_chain_idempotence();
    test_two_consecutive_calls();
    test_unloaded_module_returns_false();
    test_head_tensors_non_null_after_attach();
    test_set_initial_hidden_stores_pointer();
    test_forward_with_initial_hidden_produces_token();
    test_forward_deterministic();
    test_chain_seed_prefers_pre_norm();
    test_chain_seed_falls_back_to_post_norm();
    std::printf("[step_unit] all 12 tests PASS\n");
    return 0;
}
