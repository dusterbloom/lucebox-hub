// Unit test driving the extraction of MTP orchestration from
// qwen35_backend.cpp into dflash/src/common/mtp_orchestrator.
//
// Howard's PR #214 review (CHANGES_REQUESTED) asked for MTP logic to live
// in /common so any ModelBackend that supports MTP can leverage it. This
// test pins the public surface of the new common helper and proves it
// handles the trivial guard cases — null backend, no MTP support — before
// any real backend is wired through it.
//
// Plain int main(), assert-based, mirrors test_kv_quant.cpp style.

#include "common/mtp_orchestrator.h"
#include "common/model_backend.h"
#include "common/mtp_interface.h"
#include "common/dflash_target.h"
#include "common/mtp_tree_runner.h"
#include "common/mtp_chain_runner.h"
#include "common/ddtree.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// Minimal stub backend: implements every pure-virtual but every operation
// is a no-op or returns false. Lets us exercise the orchestrator's guard
// paths without loading any GPU weights.
struct StubBackend : public dflash27b::ModelBackend {
    bool supports_mtp_value = false;

    void print_ready_banner() const override {}
    bool park(const std::string &)   override { return true; }
    bool unpark(const std::string &) override { return true; }
    bool is_target_parked() const override { return false; }
    dflash27b::GenerateResult generate(const dflash27b::GenerateRequest &,
                                       const dflash27b::DaemonIO &) override {
        return {};
    }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int  snapshot_cur_pos(int) const override { return 0; }
    dflash27b::GenerateResult restore_and_generate(int,
                                                   const dflash27b::GenerateRequest &,
                                                   const dflash27b::DaemonIO &) override {
        return {};
    }
    bool handle_compress(const std::string &,
                         const dflash27b::DaemonIO &) override { return false; }
    void free_drafter() override {}
    bool supports_mtp() const override { return supports_mtp_value; }
    dflash27b::mtp::IMtpModule * mtp() override { return nullptr; }
    void shutdown() override {}
};

// Mock NativeHeads MTP module for the "interface is generic" test (T4).
// Records calls to prove the orchestrator invokes the right virtuals.
struct StubMtpModule : public dflash27b::mtp::INativeMtp {
    int reset_chain_calls = 0;
    int set_initial_hidden_calls = 0;
    int max_gamma() const override { return 3; }
    int effective_gamma_value = 3;
    int  effective_gamma() const override { return effective_gamma_value; }
    void set_effective_gamma(int g) override { effective_gamma_value = g; }
    int hidden_size() const override { return 4; }
    bool attach(dflash27b::DFlashTarget *) override { return true; }
    void reset_chain() override { reset_chain_calls++; }
    void shutdown() override {}
    int num_heads() const override { return 3; }
    bool step_batch(int32_t, int,
                    std::vector<dflash27b::mtp::StepOutput> &) override { return true; }
    void set_initial_hidden(const float *, int) override { set_initial_hidden_calls++; }
};

// Mock DFlashTarget that fails verify_batch — proves the orchestrator reached
// the prefill loop using only abstract DFlashTarget methods.
struct StubTarget : public dflash27b::DFlashTarget {
    int verify_batch_calls = 0;
    int hidden_size() const override { return 4; }
    bool verify_batch(const std::vector<int32_t> &, int, int &,
                      std::vector<int32_t> *) override {
        verify_batch_calls++;
        return false;
    }
    bool verify_tree(const std::vector<int32_t> &, const dflash27b::DDTree &,
                     int, std::vector<int32_t> &, std::vector<float> *) override { return false; }
    bool snapshot_kv() override { return false; }
    bool restore_kv() override { return false; }
    bool restore_kv_at_dfs(const std::vector<int> &) override { return false; }
    bool restore_kv_at_chain(int) override { return false; }
    void capture_topology_for_chain(int, int) override {}
    bool is_eos(int) const override { return false; }
    bool embed_tokens(const int32_t *, int, float *) const override { return false; }
    bool project_hidden_to_tokens(const float *, int,
                                   std::vector<int32_t> &) override { return false; }
    bool project_hidden_to_logits(const float *, int,
                                   std::vector<float> &, int &) override { return false; }
    int  mask_token_id() const override { return 0; }
    const std::vector<int> & capture_layer_ids() const override {
        static std::vector<int> empty;
        return empty;
    }
    ggml_backend * backend() const override { return nullptr; }
    ggml_tensor * lm_head_weight() const override { return nullptr; }
    int fa_window() const override { return 0; }
};

// Richer stub backend that returns valid mtp() + dflash_target() pointers,
// proving the orchestrator drives a generic ModelBackend without any
// Qwen35-specific cast or include.
struct FullStubBackend : public StubBackend {
    StubMtpModule mtp_module;
    StubTarget    target;
    FullStubBackend() { supports_mtp_value = true; }
    dflash27b::mtp::IMtpModule * mtp() override { return &mtp_module; }
    dflash27b::DFlashTarget *    dflash_target() override { return &target; }
};

}  // namespace

// ─── T1: null backend pointer ───────────────────────────────────────────────

static void t1_null_backend() {
    dflash27b::GenerateRequest req;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(nullptr, req, io);
    assert(!res.ok);
    assert(res.error.find("backend") != std::string::npos);
    std::puts("T1 null_backend PASS");
}

// ─── T2: backend that does NOT support MTP — orchestrator declines cleanly ──

static void t2_backend_without_mtp() {
    StubBackend b;
    b.supports_mtp_value = false;
    dflash27b::GenerateRequest req;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    assert(res.error.find("mtp") != std::string::npos
        || res.error.find("MTP") != std::string::npos);
    std::puts("T2 backend_without_mtp PASS");
}

// ─── T3: empty prompt — orchestrator declines with an explicit error ────────

static void t3_empty_prompt() {
    StubBackend b;
    b.supports_mtp_value = true;
    dflash27b::GenerateRequest req;
    req.n_gen = 8;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    assert(res.error.find("prompt") != std::string::npos);
    std::puts("T3 empty_prompt PASS");
}

// ─── T4: orchestrator drives a generic ModelBackend through abstract
//        interfaces only (proves logic in common/ is not Qwen35-specific).
//        Module needs to expose dflash_target() in the abstract — currently
//        ModelBackend::dflash_target() lives on the base. ───────────────

static void t4_generic_backend_dispatch() {
    FullStubBackend b;
    dflash27b::GenerateRequest req;
    req.prompt = {1, 2, 3, 4};
    req.n_gen = 4;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    // Reached verify_batch (which stub fails) — proves orchestrator depends
    // only on ModelBackend / DFlashTarget / IMtpModule abstractions.
    assert(res.error.find("verify_batch") != std::string::npos);
    assert(b.target.verify_batch_calls >= 1);
    std::puts("T4 generic_backend_dispatch PASS");
}

// ─── T5/T6/T8 scaffolding: recording arena-aware mock ──────────────────────

// Mock NativeHeads MTP that records (path_id, depth) for every
// step_batch_arena call.  Emits deterministic top-K logprobs so DDTree
// builds reproducibly.
struct ArenaRecorderMtp : public dflash27b::mtp::INativeMtp {
    std::vector<std::pair<int,int>> arena_calls;  // (path_id, depth)
    int chain_calls = 0;                          // step_batch (chain-mode)
    int reset_chain_calls = 0;
    int gamma_max_ = 3;
    int K_emit = 2;
    int hidden_ = 4;

    int  max_gamma() const override { return gamma_max_; }
    int  effective_gamma_value = 3;
    int  effective_gamma() const override { return effective_gamma_value; }
    void set_effective_gamma(int g) override { effective_gamma_value = g; }
    int  hidden_size() const override { return hidden_; }
    bool attach(dflash27b::DFlashTarget *) override { return true; }
    void reset_chain() override { reset_chain_calls++; arena_calls.clear(); }
    void shutdown() override {}
    int  num_heads() const override { return 1; }

    static void fill_topk(dflash27b::mtp::StepOutput & o, int K,
                          int32_t seed_tok) {
        o.topk_ids.assign(K, 0);
        o.topk_logprobs.assign(K, 0.0f);
        for (int k = 0; k < K; ++k) {
            o.topk_ids[k] = seed_tok * 10 + k;
            o.topk_logprobs[k] = -(float)k - 0.1f;
        }
        o.draft_token = o.topk_ids[0];
        o.draft_logit = o.topk_logprobs[0];
    }

    bool step_batch(int32_t cur, int /*base_pos*/,
                    std::vector<dflash27b::mtp::StepOutput> & out) override {
        chain_calls++;
        out.clear();
        dflash27b::mtp::StepOutput o;
        fill_topk(o, K_emit, cur);
        out.push_back(std::move(o));
        return true;
    }

    bool step_batch_arena(int32_t parent, int path_id, int depth, int K,
                          std::vector<dflash27b::mtp::StepOutput> & out) override {
        arena_calls.emplace_back(path_id, depth);
        out.clear();
        dflash27b::mtp::StepOutput o;
        fill_topk(o, std::max(K, K_emit), parent);
        // Force per-path distinction so depth0 topk siblings are unique
        // across path slots.
        o.topk_ids[0]   = parent * 100 + path_id * 10 + depth;
        o.draft_token   = o.topk_ids[0];
        out.push_back(std::move(o));
        return true;
    }
};

// Tree-verify-capable target: simulates a small accept window so the runner
// exercises the verify -> follow_verified_tree -> restore_kv_at_dfs path.
struct ArenaRecorderTarget : public dflash27b::DFlashTarget {
    int verify_batch_calls = 0;
    int verify_tree_calls  = 0;
    int restore_kv_at_dfs_calls = 0;
    int snapshot_kv_calls  = 0;
    int hidden_size() const override { return 4; }
    bool verify_batch(const std::vector<int32_t> & toks, int, int & last,
                      std::vector<int32_t> * argmax) override {
        verify_batch_calls++;
        // Echo argmax = identity so the chain runner accepts everything,
        // and last is the final token in the candidate.
        last = toks.empty() ? -1 : toks.back();
        if (argmax) {
            argmax->resize(toks.size());
            for (size_t i = 0; i < toks.size(); ++i) (*argmax)[i] = toks[i];
        }
        return true;
    }
    bool verify_tree(const std::vector<int32_t> & flat, const dflash27b::DDTree &,
                     int, std::vector<int32_t> & argmax,
                     std::vector<float> *) override {
        verify_tree_calls++;
        argmax.resize(flat.size());
        for (size_t i = 0; i < flat.size(); ++i) argmax[i] = flat[i];
        return true;
    }
    bool snapshot_kv() override { snapshot_kv_calls++; return true; }
    bool restore_kv() override { return true; }
    bool restore_kv_at_dfs(const std::vector<int> &) override {
        restore_kv_at_dfs_calls++;
        return true;
    }
    bool restore_kv_at_chain(int) override { return true; }
    void capture_topology_for_chain(int, int) override {}
    bool is_eos(int) const override { return false; }
    bool embed_tokens(const int32_t *, int, float *) const override { return false; }
    bool project_hidden_to_tokens(const float *, int,
                                   std::vector<int32_t> &) override { return false; }
    bool project_hidden_to_logits(const float *, int,
                                   std::vector<float> &, int &) override { return false; }
    int  mask_token_id() const override { return 0; }
    const std::vector<int> & capture_layer_ids() const override {
        static std::vector<int> empty; return empty;
    }
    ggml_backend * backend() const override { return nullptr; }
    ggml_tensor *  lm_head_weight() const override { return nullptr; }
    int fa_window() const override { return 0; }
};

// ─── T5 arena_slot_uniqueness ─────────────────────────────────────────────
// Run the tree runner with B=2, gamma=3 over a short request and assert
// every (path_id, depth) pair recorded by step_batch_arena is unique.

static void t5_arena_slot_uniqueness() {
    ArenaRecorderMtp    mtp;
    ArenaRecorderTarget tgt;
    dflash27b::SamplerCfg sampler;
    dflash27b::mtp::MtpTreeRunner runner(mtp, tgt, sampler, /*B=*/2, /*K=*/2);

    dflash27b::GenerateRequest req;
    req.n_gen = 4;       // enough to drive a couple of iters
    req.stream = false;
    dflash27b::DaemonIO io;

    auto res = runner.run(req, io, /*last_prefill_token=*/7,
                          /*committed_pos=*/0, /*gamma=*/3);
    assert(res.ok);
    assert(!mtp.arena_calls.empty());

    std::set<std::pair<int,int>> seen_in_iter;
    // Slot uniqueness applies *within* an iter, not across (arena resets
    // between iters).  We infer iter boundaries by reset_chain hooks not
    // being called, so just check that within any rolling iter-length
    // window the (path,depth) pairs are unique.  For B=2, gamma=3, one iter
    // has 2*3 = 6 calls — any duplicate inside that window is a bug.
    const int per_iter = 2 * 3;
    for (size_t base = 0; base < mtp.arena_calls.size(); base += per_iter) {
        std::set<std::pair<int,int>> seen;
        for (size_t i = base; i < std::min(base + per_iter, mtp.arena_calls.size()); ++i) {
            auto pd = mtp.arena_calls[i];
            assert(seen.insert(pd).second && "duplicate (path_id, depth) in iter");
        }
    }
    std::puts("T5 arena_slot_uniqueness PASS");
}

// ─── T6 chain_equivalence_B1 ──────────────────────────────────────────────
// With DFLASH27B_MTP_TREE_B=1 (default), the tree runner delegates verbatim
// to MtpChainRunner.  Run both directly and assert byte-identical outputs.

static void t6_chain_equivalence_b1() {
    // Chain runner reference.
    ArenaRecorderMtp    mtp_ref;
    ArenaRecorderTarget tgt_ref;
    dflash27b::SamplerCfg sampler;
    dflash27b::mtp::MtpChainRunner chain(mtp_ref, tgt_ref, sampler);
    dflash27b::GenerateRequest req;
    req.n_gen = 4; req.stream = false;
    dflash27b::DaemonIO io;
    auto ref = chain.run(req, io, /*last_prefill_token=*/7,
                         /*committed_pos=*/0, /*gamma=*/2);

    // Tree runner with B=1 — must short-circuit to chain.
    ArenaRecorderMtp    mtp_tree;
    ArenaRecorderTarget tgt_tree;
    dflash27b::mtp::MtpTreeRunner tree_run(mtp_tree, tgt_tree, sampler,
                                            /*B=*/1, /*K=*/2);
    auto got = tree_run.run(req, io, 7, 0, 2);

    assert(ref.ok && got.ok);
    assert(ref.tokens == got.tokens);
    // Pin: chain runner is the one driving arena recorder — but for B=1 the
    // tree runner delegates so its mtp recorder sees zero arena calls.
    assert(mtp_tree.arena_calls.empty());
    std::puts("T6 chain_equivalence_B1 PASS");
}

// ─── T7 ddtree_per_sibling_input ──────────────────────────────────────────
// Synthetic [L=3, B=2, K=2] yields a balanced 2-ary tree: 2+4+8 = 14
// non-root nodes (plus root) = 15 nodes total.

static void t7_ddtree_per_sibling_input() {
    const int L = 3, B = 2, K = 2;
    std::vector<float>   tlp((size_t)L * B * K, 0.0f);
    std::vector<int32_t> tid((size_t)L * B * K, 0);
    for (int d = 0; d < L; ++d)
        for (int b = 0; b < B; ++b)
            for (int k = 0; k < K; ++k) {
                const size_t i = (size_t)d * B * K + (size_t)b * K + k;
                tid[i] = d * 1000 + b * 100 + k;     // unique per (d,b,k)
                tlp[i] = -(float)k - 0.1f * d;
            }
    auto tree = dflash27b::build_ddtree_tree(tlp.data(), tid.data(),
                                              L, B, K, /*budget=*/64);
    // Expected: 2 + 4 + 8 = 14 non-root nodes, plus root = 15 total.
    assert(tree.n_nodes == 14);
    // Depth histogram: 2 at d=1, 4 at d=2, 8 at d=3.
    int d_count[4] = {0,0,0,0};
    for (int dep : tree.depths) {
        assert(dep >= 1 && dep <= 3);
        d_count[dep]++;
    }
    assert(d_count[1] == 2);
    assert(d_count[2] == 4);
    assert(d_count[3] == 8);
    std::puts("T7 ddtree_per_sibling_input PASS");
}

// ─── T8 arena_reset_between_iters ─────────────────────────────────────────
// Across 3 successive iters at B=2, gamma=3, every iter's first call must be
// (path_id=0, depth=0) — i.e. arena slot 0 is free at iter start.

static void t8_arena_reset_between_iters() {
    ArenaRecorderMtp    mtp;
    ArenaRecorderTarget tgt;
    dflash27b::SamplerCfg sampler;
    dflash27b::mtp::MtpTreeRunner runner(mtp, tgt, sampler, /*B=*/2, /*K=*/2);

    dflash27b::GenerateRequest req;
    req.n_gen = 12;  // ~3 iters at gamma=3 + bonus
    req.stream = false;
    dflash27b::DaemonIO io;

    auto res = runner.run(req, io, 7, 0, 3);
    assert(res.ok);
    // Per-iter call count is fixed B*gamma=6.  First call of each iter must
    // be (0,0).
    const int per_iter = 2 * 3;
    assert((int)mtp.arena_calls.size() >= per_iter * 2);
    for (size_t base = 0; base + (size_t)per_iter <= mtp.arena_calls.size();
         base += per_iter) {
        auto pd = mtp.arena_calls[base];
        assert(pd.first == 0 && pd.second == 0);
    }
    std::puts("T8 arena_reset_between_iters PASS");
}

int main() {
    t1_null_backend();
    t2_backend_without_mtp();
    t3_empty_prompt();
    t4_generic_backend_dispatch();
    t5_arena_slot_uniqueness();
    t6_chain_equivalence_b1();
    t7_ddtree_per_sibling_input();
    t8_arena_reset_between_iters();
    std::puts("ALL PASS");
    return 0;
}
