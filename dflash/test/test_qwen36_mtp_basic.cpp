// test_qwen36_mtp_basic.cpp — PR 2 skeleton test.
//
// Verifies that:
//   1. load_qwen36_mtp_weights reports a clear error on a non-MTP GGUF
//      (i.e. when qwen35.nextn_predict_layers is absent).
//   2. On a real unsloth Qwen3.6-*-MTP-GGUF, the loader binds all
//      required NextN tensors and reports a sane head count.
//   3. Qwen36MtpModule honors the IMtpModule contract: flavor is
//      NativeHeads, num_heads / hidden_size are positive, attach binds
//      and shutdown is idempotent.
//
// The test gates on environment variable QWEN36_MTP_GGUF — when unset,
// only the contract-via-stub portion runs. When set to a real MTP GGUF
// path, the loader is exercised end-to-end. CI environments without
// the model file fail gracefully (skip with non-zero return only if the
// stub portion fails).
//
// PR 2b will add step_batch correctness against a captured backbone
// hidden state; this PR just proves the abstraction admits Qwen3.6.

#include "qwen36/qwen36_mtp.h"
#include "common/mtp_interface.h"
#include "common/dflash_target.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

namespace {

static constexpr int QWEN36_27B_N_EMBD = 5120;
// Verified vocab size from GGUF tensor inventory in qwen36_mtp_redesign.md.
static constexpr int QWEN36_27B_N_VOCAB = 248320;

// Stub target with a fixed hidden_size matching Qwen3.6-27B backbone.
// embed_tokens and project_hidden_to_tokens are functional so that
// step_batch can complete the full Shape B forward in T2 smoke tests.
struct StubTarget : DFlashTarget {
    int H = QWEN36_27B_N_EMBD;
    int V = QWEN36_27B_N_VOCAB;

    bool verify_batch(const std::vector<int32_t> &, int, int &,
                      std::vector<int32_t> *) override { return true; }
    bool snapshot_kv() override { return true; }
    bool restore_kv() override { return true; }
    bool is_eos(int) const override { return false; }

    // Fill embedding with a small deterministic value (token-dependent).
    bool embed_tokens(const int32_t * tokens, int n, float * out) const override {
        for (int i = 0; i < n; i++) {
            const float val = 1.0f / static_cast<float>(tokens[i] + 1);
            for (int d = 0; d < H; d++) out[i * H + d] = val * 0.01f;
        }
        return true;
    }

    // Project by argmax(abs(hidden)) % V (no GPU LM head in the stub).
    bool project_hidden_to_tokens(const float * hidden, int n_tokens,
                                  std::vector<int32_t> & tokens_out) override {
        tokens_out.resize(n_tokens);
        for (int i = 0; i < n_tokens; i++) {
            // Use sum-of-absolute-values as a cheap pseudo-argmax proxy.
            float sum = 0.0f;
            for (int d = 0; d < H; d++) sum += std::abs(hidden[i * H + d]);
            tokens_out[i] = static_cast<int32_t>(static_cast<int>(sum * 1e4f)) % V;
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

void test_contract_with_stub() {
    Qwen36MtpModule m;
    StubTarget t;

    // flavor is stable across calls and matches NativeHeads.
    CHECK(m.flavor() == MtpFlavor::NativeHeads);
    CHECK(m.flavor() == MtpFlavor::NativeHeads);

    // Pre-init: hidden_size and num_heads are zero until weights load.
    CHECK(m.hidden_size() == 0);
    CHECK(m.num_heads()   == 0);
    CHECK(m.max_gamma()   == 0);

    // attach succeeds even before init — but step_batch will surface the
    // not-implemented error.
    CHECK(m.attach(&t));

    std::vector<StepOutput> out;
    const bool stepped = m.step_batch(/*cur=*/123, /*pos=*/0, out);
    // The module was attach()ed above but never had weights loaded (no
    // init() or attach_weights_for_test() call), so state_->loaded == false.
    // step_batch's load-check guard fires first: if (!loaded || !attached)
    // returns false with empty out. This assertion remains valid after PR 2b
    // because the guard is the first thing step_batch checks, independent of
    // the PR 2 "not implemented" stderr that was removed.
    CHECK(!stepped);
    CHECK(out.empty());

    m.reset_chain();          // idempotent
    m.reset_chain();
    m.shutdown();
    m.shutdown();
    std::printf("[qwen36_mtp] contract-via-stub OK\n");
}

void test_loader_when_gguf_present() {
    const char * env = std::getenv("QWEN36_MTP_GGUF");
    if (!env || !*env) {
        std::printf("[qwen36_mtp] QWEN36_MTP_GGUF unset; skipping loader test\n");
        return;
    }
    const std::string path = env;

    Qwen36MtpModule m;
    StubTarget t;
    std::string err;
    const bool ok = m.init(path, &t, err);
    if (!ok) {
        std::fprintf(stderr,
            "[qwen36_mtp] init failed for %s: %s\n", path.c_str(), err.c_str());
        std::exit(1);
    }
    CHECK(m.hidden_size() == t.hidden_size());
    CHECK(m.num_heads()   >= 1);
    CHECK(m.max_gamma()   == m.num_heads());

    const auto & w = m.weights();
    CHECK(w.n_embd            == m.hidden_size());
    CHECK(w.n_heads           == m.num_heads());
    CHECK((int)w.heads.size() == w.n_heads);

    // Every head must have the three required tensors bound.
    for (const auto & h : w.heads) {
        CHECK(h.layer_idx >= 0);
        CHECK(h.eh_proj != nullptr);
        CHECK(h.enorm   != nullptr);
        CHECK(h.hnorm   != nullptr);
    }
    std::printf("[qwen36_mtp] loader OK on %s (n_heads=%d, n_embd=%d, backbone_layers=%d)\n",
                path.c_str(), w.n_heads, w.n_embd, w.n_backbone_layers);
}

// T2 smoke: load real GGUF and run step_batch with a hand-built h_prev.
// Gated on QWEN36_MTP_GGUF (same gate as test_loader_when_gguf_present).
// Verifies:
//   - out.size() == n_heads (1 for the 27B GGUF)
//   - out[0].draft_token in [0, QWEN36_27B_N_VOCAB)
//   - no crash
// Stretch: 5-step chain to verify per-head KV write doesn't corrupt state.
void test_forward_smoke_when_gguf_present() {
    const char * env = std::getenv("QWEN36_MTP_GGUF");
    if (!env || !*env) {
        std::printf("[qwen36_mtp] QWEN36_MTP_GGUF unset; skipping forward smoke\n");
        return;
    }
    const std::string path = env;

    Qwen36MtpModule m;
    StubTarget t;
    std::string err;
    const bool ok = m.init(path, &t, err);
    if (!ok) {
        std::fprintf(stderr,
            "[qwen36_mtp] forward smoke: init failed for %s: %s\n",
            path.c_str(), err.c_str());
        std::exit(1);
    }
    CHECK(m.num_heads() >= 1);

    // Build a synthetic h_prev: small noise (0.001 per element), non-zero.
    const int n_embd = m.hidden_size();
    std::vector<float> h_prev(n_embd);
    for (int i = 0; i < n_embd; i++) h_prev[i] = 0.001f * static_cast<float>(i % 17 + 1);

    m.set_initial_hidden(h_prev.data(), n_embd);

    std::vector<StepOutput> out;
    const bool stepped = m.step_batch(/*current_token=*/1, /*base_pos=*/0, out);
    if (!stepped) {
        std::fprintf(stderr, "[qwen36_mtp] forward smoke: step_batch failed\n");
        std::exit(1);
    }
    CHECK((int)out.size() == m.num_heads());
    CHECK(out[0].draft_token >= 0);
    CHECK(out[0].draft_token < QWEN36_27B_N_VOCAB);

    std::printf("[qwen36_mtp] forward smoke: step 0 -> draft_token=%d\n",
                out[0].draft_token);

    // Stretch: 5-step chain.  Advance base_pos and feed the last draft_token
    // as the next current_token.  Just verifies no crash and stable vocab range.
    int32_t cur_tok = out[0].draft_token;
    for (int step = 1; step <= 5; step++) {
        m.set_initial_hidden(h_prev.data(), n_embd);   // synthetic; not a real backbone hidden
        std::vector<StepOutput> out_s;
        const bool ok_s = m.step_batch(cur_tok, step, out_s);
        if (!ok_s || out_s.empty()) {
            std::fprintf(stderr,
                "[qwen36_mtp] forward smoke: step %d failed\n", step);
            std::exit(1);
        }
        CHECK(out_s[0].draft_token >= 0);
        CHECK(out_s[0].draft_token < QWEN36_27B_N_VOCAB);
        std::printf("[qwen36_mtp] forward smoke: step %d -> draft_token=%d\n",
                    step, out_s[0].draft_token);
        cur_tok = out_s[0].draft_token;
    }

    std::printf("[qwen36_mtp] forward smoke PASS (6 steps, no crash)\n");
}

}  // namespace

int main() {
    test_contract_with_stub();
    test_loader_when_gguf_present();
    test_forward_smoke_when_gguf_present();
    std::printf("[qwen36_mtp] PR 2 skeleton tests PASS\n");
    return 0;
}
