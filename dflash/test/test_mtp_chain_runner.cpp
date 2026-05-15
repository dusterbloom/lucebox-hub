// test_mtp_chain_runner.cpp — Exercises MtpChainRunner γ-loop with
// fake implementations of IMtpModule and DFlashTarget.
//
// Covers:
//   - γ=1 baseline: each iter emits exactly one bonus token
//   - γ=2 full-accept: all draft tokens match target argmax
//   - γ=2 full-reject: first draft mismatches → only bonus emitted
//   - γ=3 mid-chain reject: prefix accepted, tail rolled back
//   - EOS mid-chain: emission stops at the EOS token
//   - n_gen cap: termination at the requested generation count
//   - NativeHeads flavor produces the same observable behavior as
//     ExternalDrafter for the same predicted sequence

#include "common/mtp_chain_runner.h"
#include "common/mtp_interface.h"
#include "common/dflash_target.h"
#include "common/model_backend.h"
#include "common/sampler.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace dflash27b;
using namespace dflash27b::mtp;

// Release builds strip <cassert>'s CHECK(). Use CHECK() which always
// fires; failure aborts with non-zero exit so the test harness notices.
#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "%s:%d CHECK(%s) FAILED\n",                     \
                     __FILE__, __LINE__, #cond);                             \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

namespace {

// ── Fake DFlashTarget driven by a scripted "true argmax" sequence ──────
//
// target_argmax_[base_pos + i] is the value verify_batch returns for the
// argmax AFTER seeing candidate[i] at base_pos+i. Tests prepopulate this
// to drive specific accept/reject patterns.
struct ScriptedTarget : DFlashTarget {
    int H = 8;
    int eos_id = 2;
    std::vector<int32_t> target_argmax;  // indexed by absolute position
    int kv_pos = 0;
    std::vector<int32_t> snap_kv;        // snapshot stack (just kv_pos)
    int snap_count = 0;

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax) override {
        if (all_argmax) all_argmax->clear();
        for (size_t i = 0; i < tokens.size(); i++) {
            const int pos = base_pos + (int)i;
            if (pos >= (int)target_argmax.size()) {
                // Out-of-script: pad with cur_tok+1 deterministically
                target_argmax.resize(pos + 1, tokens[i] + 1);
            }
            if (all_argmax) all_argmax->push_back(target_argmax[pos]);
        }
        last_tok = all_argmax ? all_argmax->back() : -1;
        kv_pos = base_pos + (int)tokens.size();
        return true;
    }
    bool snapshot_kv() override {
        snap_kv.push_back(kv_pos); snap_count++; return true;
    }
    bool restore_kv() override {
        if (snap_kv.empty()) return false;
        kv_pos = snap_kv.back(); snap_kv.pop_back(); return true;
    }
    bool is_eos(int token) const override { return token == eos_id; }
    bool embed_tokens(const int32_t *, int, float *) const override { return true; }
    bool project_hidden_to_tokens(const float *, int,
                                  std::vector<int32_t> &) override { return true; }
    int hidden_size() const override { return H; }
    int mask_token_id() const override { return -1; }
    const std::vector<int> & capture_layer_ids() const override {
        static const std::vector<int> ids;
        return ids;
    }
};

// ── Fake ExternalDrafter driven by a scripted draft function ──────────
struct ScriptedExternalMtp : IExternalDrafterMtp {
    int H = 8;
    int gmax = 4;
    std::vector<int> donors_{0};
    // For position p and γ-step g, return draft tokens.
    // Default: cur + 1 + g (matches identity-like progression).
    std::function<int32_t(int32_t cur, int base_pos, int gamma_index)> script;
    int max_gamma() const override { return gmax; }
    int hidden_size() const override { return H; }
    bool attach(DFlashTarget *) override { return true; }
    void reset_chain() override {}
    void shutdown() override {}
    bool step(const StepInput & in, StepOutput & out) override {
        out.draft_token = script
            ? script(in.current_token, in.base_pos, in.gamma_index)
            : in.current_token + 1 + in.gamma_index;
        out.draft_logit = 1.0f;
        out.next_hidden.assign(H, 0.0f);
        return true;
    }
    const std::vector<int> & donor_layers() const override { return donors_; }
    bool enable_target_hidden_capture(bool, int) override { return true; }
    void set_capture_row(int) override {}
    bool consume_captured_hidden(float *, int) override { return true; }
};

// ── Fake NativeHeads — emits k drafts per step_batch call ─────────────
struct ScriptedNativeMtp : INativeMtp {
    int H = 8;
    int heads = 2;
    std::function<std::vector<int32_t>(int32_t cur, int base_pos)> script;
    int max_gamma() const override { return heads; }
    int hidden_size() const override { return H; }
    bool attach(DFlashTarget *) override { return true; }
    void reset_chain() override {}
    void shutdown() override {}
    int num_heads() const override { return heads; }
    bool step_batch(int32_t cur, int base_pos,
                    std::vector<StepOutput> & out) override {
        out.clear();
        std::vector<int32_t> drafts;
        if (script) drafts = script(cur, base_pos);
        else for (int i = 0; i < heads; i++) drafts.push_back(cur + 1 + i);
        for (auto d : drafts) {
            StepOutput s; s.draft_token = d; out.push_back(std::move(s));
        }
        return true;
    }
};

GenerateRequest make_req(int n_gen, bool stream = false) {
    GenerateRequest r;
    r.n_gen = n_gen;
    r.stream = stream;
    return r;
}

DaemonIO null_io() { DaemonIO io; io.stream_fd = -1; return io; }

// ── Test cases ─────────────────────────────────────────────────────────

void test_gamma1_baseline() {
    ScriptedTarget tgt;
    // target predicts: 101 -> 102 -> 103 -> 104
    tgt.target_argmax = { 102, 103, 104, 105, 106, 107 };
    ScriptedExternalMtp mtp;
    // drafter agrees: emits target_argmax[base_pos+gamma_index]
    mtp.script = [&](int32_t /*cur*/, int base_pos, int g) {
        return tgt.target_argmax[base_pos + g];
    };
    mtp.attach(&tgt);

    MtpChainRunner runner(mtp, tgt, SamplerCfg{});
    auto res = runner.run(make_req(4), null_io(),
                          /*last_prefill_token=*/101, /*committed_pos=*/0,
                          /*gamma=*/1);
    CHECK(res.ok);
    CHECK(res.tokens.size() == 4);
    CHECK(res.tokens[0] == 102);
    CHECK(res.tokens[1] == 103);
    CHECK(res.tokens[2] == 104);
    CHECK(res.tokens[3] == 105);
    std::printf("[runner] γ=1 baseline OK (tokens=%zu)\n", res.tokens.size());
}

void test_gamma2_full_accept() {
    ScriptedTarget tgt;
    tgt.target_argmax = { 200, 201, 202, 203, 204, 205, 206, 207 };
    ScriptedExternalMtp mtp;
    mtp.script = [&](int32_t /*cur*/, int base_pos, int g) {
        return tgt.target_argmax[base_pos + g];
    };
    mtp.attach(&tgt);

    MtpChainRunner runner(mtp, tgt, SamplerCfg{});
    auto res = runner.run(make_req(6), null_io(), /*cur=*/100, /*pos=*/0,
                          /*gamma=*/2);
    CHECK(res.ok);
    CHECK(res.tokens.size() == 6);
    // Each iter accepts both drafts + 1 bonus → 3 tokens/iter → 2 iters.
    CHECK(runner.stats().total_iters == 2);
    CHECK(runner.stats().total_accepted == 4);
    CHECK(runner.stats().total_emitted == 6);
    std::printf("[runner] γ=2 full-accept OK (iters=%d accepted=%d)\n",
                runner.stats().total_iters, runner.stats().total_accepted);
}

void test_gamma2_full_reject() {
    ScriptedTarget tgt;
    tgt.target_argmax = { 999, 998, 997, 996, 995, 994, 993, 992 };
    ScriptedExternalMtp mtp;
    // drafter always wrong
    mtp.script = [&](int32_t /*cur*/, int /*base_pos*/, int /*g*/) {
        return 42;
    };
    mtp.attach(&tgt);

    MtpChainRunner runner(mtp, tgt, SamplerCfg{});
    auto res = runner.run(make_req(3), null_io(), /*cur=*/100, /*pos=*/0,
                          /*gamma=*/2);
    CHECK(res.ok);
    CHECK(res.tokens.size() == 3);
    // First draft rejects → accept_n=0, emit only 1 bonus per iter.
    // 3 iters to produce 3 tokens.
    CHECK(runner.stats().total_iters == 3);
    CHECK(runner.stats().total_accepted == 0);
    CHECK(runner.stats().total_emitted == 3);
    std::printf("[runner] γ=2 full-reject OK (iters=%d accepted=%d)\n",
                runner.stats().total_iters, runner.stats().total_accepted);
}

void test_gamma3_mid_chain_reject() {
    ScriptedTarget tgt;
    // base_pos=0,1,2,3 → target argmax: 10, 11, 12, 13
    tgt.target_argmax = { 10, 11, 12, 13, 14, 15, 16, 17 };
    ScriptedExternalMtp mtp;
    // drafter: matches argmax for g=0,1; wrong for g=2
    mtp.script = [&](int32_t /*cur*/, int base_pos, int g) {
        if (g == 2) return 99999;
        return tgt.target_argmax[base_pos + g];
    };
    mtp.attach(&tgt);

    MtpChainRunner runner(mtp, tgt, SamplerCfg{});
    auto res = runner.run(make_req(3), null_io(), /*cur=*/0, /*pos=*/0,
                          /*gamma=*/3);
    CHECK(res.ok);
    // accept_n=2, bonus=target_argmax[2]=12 → 3 tokens in one iter.
    CHECK(res.tokens.size() == 3);
    CHECK(res.tokens[0] == 10);
    CHECK(res.tokens[1] == 11);
    CHECK(res.tokens[2] == 12);
    CHECK(runner.stats().total_iters == 1);
    CHECK(runner.stats().total_accepted == 2);
    std::printf("[runner] γ=3 mid-chain-reject OK\n");
}

void test_eos_mid_chain() {
    ScriptedTarget tgt;
    tgt.eos_id = 2;
    // target argmax: 1, 2(EOS), 3, ...
    tgt.target_argmax = { 1, 2, 3, 4, 5, 6 };
    ScriptedExternalMtp mtp;
    mtp.script = [&](int32_t /*cur*/, int base_pos, int g) {
        return tgt.target_argmax[base_pos + g];
    };
    mtp.attach(&tgt);

    MtpChainRunner runner(mtp, tgt, SamplerCfg{});
    auto res = runner.run(make_req(5), null_io(), /*cur=*/0, /*pos=*/0,
                          /*gamma=*/3);
    CHECK(res.ok);
    // First iter: drafts = [1, 2(EOS), 3], target argmax matches all.
    // accept_n=3, bonus=target_argmax[3]=4, but emission halts at EOS.
    // Expected tokens: [1, 2]; loop exits.
    CHECK(res.tokens.size() == 2);
    CHECK(res.tokens.back() == 2);
    std::printf("[runner] EOS-mid-chain OK\n");
}

void test_n_gen_cap() {
    ScriptedTarget tgt;
    tgt.target_argmax = { 10, 11, 12, 13, 14, 15, 16, 17 };
    ScriptedExternalMtp mtp;
    mtp.script = [&](int32_t /*cur*/, int base_pos, int g) {
        return tgt.target_argmax[base_pos + g];
    };
    mtp.attach(&tgt);

    MtpChainRunner runner(mtp, tgt, SamplerCfg{});
    // γ=3 but n_gen=2 — runner must clamp drafts to fit.
    auto res = runner.run(make_req(2), null_io(), /*cur=*/0, /*pos=*/0,
                          /*gamma=*/3);
    CHECK(res.ok);
    CHECK(res.tokens.size() == 2);
    std::printf("[runner] n_gen cap OK (tokens=%zu)\n", res.tokens.size());
}

void test_native_flavor_equivalence() {
    ScriptedTarget tgt;
    tgt.target_argmax = { 50, 51, 52, 53, 54, 55 };
    ScriptedNativeMtp mtp;
    mtp.heads = 2;
    mtp.script = [&](int32_t /*cur*/, int base_pos) {
        return std::vector<int32_t>{ tgt.target_argmax[base_pos],
                                     tgt.target_argmax[base_pos + 1] };
    };
    mtp.attach(&tgt);

    MtpChainRunner runner(mtp, tgt, SamplerCfg{});
    auto res = runner.run(make_req(6), null_io(), /*cur=*/0, /*pos=*/0,
                          /*gamma=*/2);
    CHECK(res.ok);
    CHECK(res.tokens.size() == 6);
    // Full-accept on every iter: 2 drafts + 1 bonus = 3 tokens/iter, 2 iters.
    CHECK(runner.stats().total_iters == 2);
    CHECK(runner.stats().total_accepted == 4);
    std::printf("[runner] NativeHeads equivalence OK\n");
}

}  // namespace

int main() {
    test_gamma1_baseline();
    test_gamma2_full_accept();
    test_gamma2_full_reject();
    test_gamma3_mid_chain_reject();
    test_eos_mid_chain();
    test_n_gen_cap();
    test_native_flavor_equivalence();
    std::printf("[runner] all PASS\n");
    return 0;
}
