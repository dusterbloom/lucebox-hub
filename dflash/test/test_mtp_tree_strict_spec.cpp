// test_mtp_tree_strict_spec.cpp — Stage 3 correctness invariant.
//
// Strict-speculation equivalence: a speculative path that uses the target's
// posterior to accept/reject candidates MUST produce the SAME emitted token
// sequence as the target alone in greedy mode.  If chain mode and tree mode
// diverge for the same target + same prompt + greedy sampling, then either
// the tree-verify's logits are wrong, the accept walk is wrong, or the
// SSM/conv/KV rollback (Qwen35DFlashTarget::restore_kv_at_dfs) is wrong.
//
// This test runs on a mock target (no model GGUF needed, no GPU) so it can
// execute in CI.  The mock target's argmax depends on its internal SSM
// counter; if a tree-verify path forgets to restore state, the counter
// advances by N (the tree size) instead of commit_n (the accepted-path
// length) and subsequent argmaxes diverge from the chain-mode reference.
//
// Test tier: T1 — no model file, no GPU, CPU only.

#include "common/dflash_target.h"
#include "common/ddtree.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash27b;

#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "%s:%d CHECK(%s) FAILED\n",                     \
                     __FILE__, __LINE__, #cond);                             \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

namespace {

// ── DeterministicOracleTarget ─────────────────────────────────────────
//
// A mock target with a stateful "ssm_counter_" that is bumped once per
// token forwarded.  argmax(pos, tok) = (ssm_counter_ + 1) % vocab — i.e.
// the argmax depends on hidden state, not just on `tok`.  This is the
// minimum condition under which rollback correctness becomes observable:
// if tree-verify forwards N tokens but only `commit_n` are accepted, the
// counter must be restored to (counter_before + commit_n) before the next
// verify; otherwise it sits at (counter_before + N), and the next argmax
// is wrong.
//
// verify_tree models the real graph behavior: every DFS slot gets
// forwarded (counter += N), per-DFS-slot argmax is computed against the
// counter value AT that slot, and per-DFS-slot ssm_intermediate snapshots
// are stashed in an internal table for restore_kv_at_dfs to consume.
struct DeterministicOracleTarget : DFlashTarget {
    int H = 4;
    int V = 64;
    int eos_id = -1;  // no EOS in this test
    int ssm_counter_ = 0;
    // Stash of per-DFS counter values from the most recent verify_tree.
    // restore_kv_at_dfs reads this to roll back ssm_counter_.
    std::vector<int> last_tree_counter_states_;
    int last_tree_base_pos_ = -1;
    int last_tree_n_nodes_ = 0;

    // Snapshot stack for snapshot_kv/restore_kv (unused by this test).
    std::vector<int> snap_stack_;

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax) override {
        (void)base_pos;
        if (all_argmax) all_argmax->resize(tokens.size());
        for (size_t i = 0; i < tokens.size(); i++) {
            ssm_counter_++;
            const int32_t am = (int32_t)((ssm_counter_ + 1) % V);
            if (all_argmax) (*all_argmax)[i] = am;
        }
        last_tok = (int32_t)((ssm_counter_ + 1) % V);
        return true;
    }
    bool verify_tree(const std::vector<int32_t> & flat_tokens,
                     const DDTree & tree,
                     int base_pos,
                     std::vector<int32_t> & out_argmax,
                     std::vector<float> * out_logits) override {
        const int N = (int)flat_tokens.size();
        CHECK(N == 1 + tree.n_nodes);
        out_argmax.resize(N);
        // Reset per-tree snapshot table.
        last_tree_counter_states_.assign(N, 0);
        last_tree_base_pos_ = base_pos;
        last_tree_n_nodes_  = tree.n_nodes;
        // DFS walk: for slot i, the counter value AFTER processing this
        // slot's parent ancestry equals (ssm_counter_ at entry) + depth(i) + 1.
        // We don't bother with real DeltaNet state — we just mirror the
        // accept-path counter (depth d → counter += d+1).
        const int counter_at_entry = ssm_counter_;
        for (int i = 0; i < N; i++) {
            const int depth_i = (i == 0) ? 0 : tree.depths[i - 1];
            const int slot_counter = counter_at_entry + depth_i + 1;
            last_tree_counter_states_[i] = slot_counter;
            out_argmax[i] = (int32_t)((slot_counter + 1) % V);
        }
        // The "live" counter advances over the full DFS extent (mirrors
        // what build_target_step_tree does to the real SSM state).  This
        // is exactly the poisoning case Stage 3 must repair: if we don't
        // call restore_kv_at_dfs, the counter ends up too far ahead.
        ssm_counter_ = counter_at_entry + N;
        if (out_logits) out_logits->clear();
        return true;
    }
    bool restore_kv_at_dfs(const std::vector<int> & accepted_dfs) override {
        CHECK(!accepted_dfs.empty());
        CHECK(accepted_dfs[0] == 0);
        CHECK(last_tree_base_pos_ >= 0);
        const int deepest_dfs = accepted_dfs.back();
        CHECK(deepest_dfs >= 0 && deepest_dfs < (int)last_tree_counter_states_.size());
        ssm_counter_ = last_tree_counter_states_[deepest_dfs];
        return true;
    }
    bool snapshot_kv() override { snap_stack_.push_back(ssm_counter_); return true; }
    bool restore_kv() override {
        CHECK(!snap_stack_.empty());
        ssm_counter_ = snap_stack_.back(); snap_stack_.pop_back();
        return true;
    }
    bool is_eos(int t) const override { return eos_id >= 0 && t == eos_id; }
    bool embed_tokens(const int32_t *, int, float *) const override { return true; }
    bool project_hidden_to_tokens(const float *, int,
                                  std::vector<int32_t> &) override { return true; }
    int hidden_size() const override { return H; }
    int mask_token_id() const override { return -1; }
    const std::vector<int> & capture_layer_ids() const override {
        static const std::vector<int> ids; return ids;
    }
};

// ── Reference: pure chain mode ────────────────────────────────────────
//
// One token per verify_batch call. This is the "target alone in greedy
// mode" reference any speculative path must match byte-for-byte.
static std::vector<int32_t> run_chain_reference(
        DeterministicOracleTarget & tgt,
        int32_t prefill_token,
        int n_gen) {
    std::vector<int32_t> generated;
    int32_t cur = prefill_token;
    int base_pos = 0;
    while ((int)generated.size() < n_gen) {
        std::vector<int32_t> single{cur};
        int32_t last = -1;
        std::vector<int32_t> all_argmax;
        CHECK(tgt.verify_batch(single, base_pos, last, &all_argmax));
        generated.push_back(last);
        cur = last;
        base_pos++;
    }
    return generated;
}

// ── Mtp-topk DDTree mode emulating the test_dflash.cpp harness ───────
//
// Mirrors the structure of the qwen36-mtp[topk] harness branch at
// test_dflash.cpp:876-970 (post Stage 3 patch).  The drafter is a stub
// that proposes K=2 candidates per head: top-1 = the chain-reference
// continuation (so verify accepts it), top-2 = a deliberately-wrong
// sibling (always gets rejected by the oracle argmax → exercises the
// rollback path).
struct ChainSeededDrafter {
    int K;
    int L;  // num heads / chain depth per round
    // Per round: predict L positions of top-K. For correctness, predict
    // the chain-reference continuation at top-1 and a bogus sibling at
    // top-2.  We don't need real log-probs — DDTree only uses them for
    // best-first ordering; with chain_seed=false the test still passes
    // because the spine accepts.
    std::vector<int32_t> chain_argmax;  // future tokens, length >= base_pos+L

    void propose(int32_t cur, int base_pos,
                 std::vector<float> & out_logp,
                 std::vector<int32_t> & out_ids) {
        (void)cur;
        out_logp.assign((size_t)L * K, 0.0f);
        out_ids.assign((size_t)L * K, 0);
        for (int i = 0; i < L; i++) {
            // top-1: matches the chain-reference future (will be accepted).
            out_ids[(size_t)i * K + 0] = chain_argmax[base_pos + i];
            out_logp[(size_t)i * K + 0] = -0.1f;
            // top-2: a deliberately-wrong sibling (forces sibling DFS
            // slots to exist; if rollback is broken, those slots' KV
            // state leaks into the next iter).
            for (int k = 1; k < K; k++) {
                out_ids[(size_t)i * K + k] = (chain_argmax[base_pos + i] + 7 + k) % 64;
                out_logp[(size_t)i * K + k] = -1.0f - (float)k;
            }
        }
    }
};

static std::vector<int32_t> run_topk_tree_mode(
        DeterministicOracleTarget & tgt,
        int32_t prefill_token,
        int n_gen,
        int K, int ddtree_budget, bool chain_seed,
        const std::vector<int32_t> & oracle_chain) {
    ChainSeededDrafter drafter;
    drafter.K = K;
    drafter.L = 4;  // 4 heads
    drafter.chain_argmax = oracle_chain;

    std::vector<int32_t> generated;
    int32_t cur = prefill_token;
    int base_pos = 0;

    std::vector<float>   ddtree_logp;
    std::vector<int32_t> ddtree_ids;
    while ((int)generated.size() < n_gen) {
        drafter.propose(cur, base_pos, ddtree_logp, ddtree_ids);
        DDTree tree = build_ddtree(
            ddtree_logp.data(), ddtree_ids.data(),
            drafter.L, K, ddtree_budget, chain_seed);

        std::vector<int32_t> flat;
        flat.reserve(1 + tree.n_nodes);
        flat.push_back(cur);
        for (int i = 0; i < tree.n_nodes; i++) flat.push_back(tree.token_ids[i]);

        std::vector<int32_t> tree_argmax;
        CHECK(tgt.verify_tree(flat, tree, base_pos, tree_argmax, nullptr));

        int next_token = -1;
        int bonus_node_idx = 0;
        std::vector<int> accepted_path = follow_verified_tree(
            tree, tree_argmax.data(), next_token, &bonus_node_idx);
        const int accept_depth = (int)accepted_path.size();
        const int draft_depth  = std::max(0, accept_depth - 1);
        int committed_dfs_n = 1;  // root
        bool tt_cap = false;
        for (int i = 1; i < accept_depth; i++) {
            const int node_idx = accepted_path[i];
            const int32_t tok  = tree.token_ids[node_idx - 1];
            generated.push_back(tok);
            committed_dfs_n++;
            if ((int)generated.size() >= n_gen) { cur = tok; tt_cap = true; break; }
        }
        if (!tt_cap && next_token >= 0) {
            generated.push_back((int32_t)next_token);
            cur = (int32_t)next_token;
            base_pos += draft_depth + 1;
        } else if (!tt_cap) {
            base_pos += draft_depth;
        }
        if (tt_cap || (int)generated.size() >= n_gen) break;

        // Stage 3: critical rollback (the thing this test exists to
        // catch).  Comment this out and watch the test fail with a
        // mismatched token sequence.
        std::vector<int> commit_prefix(accepted_path.begin(),
                                       accepted_path.begin() + committed_dfs_n);
        CHECK(tgt.restore_kv_at_dfs(commit_prefix));
    }
    return generated;
}

// ── Test 1: strict equivalence with chain-seeded tree ────────────────
//
// With chain_seed=true, the DDTree always contains the top-1 spine,
// so the oracle's verify_tree always accepts at least the spine.
static void test_strict_equivalence_chain_seed() {
    DeterministicOracleTarget tgt_ref;
    auto ref = run_chain_reference(tgt_ref, /*prefill=*/7, /*n_gen=*/64);

    DeterministicOracleTarget tgt_topk;
    auto top = run_topk_tree_mode(tgt_topk, /*prefill=*/7, /*n_gen=*/64,
                                  /*K=*/2, /*ddtree_budget=*/6,
                                  /*chain_seed=*/true, ref);
    CHECK(ref.size() == top.size());
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i] != top[i]) {
            std::fprintf(stderr,
                "strict-spec MISMATCH at i=%zu: ref=%d top=%d\n"
                "  (this means tree-verify or rollback is broken)\n",
                i, ref[i], top[i]);
            std::exit(1);
        }
    }
    // End-state counter is informational only: when the loop bails on
    // n_gen-cap it skips the final rollback (mirrors the real harness
    // which doesn't need post-cap state), so topk's counter may sit at
    // (committed + bonus DFS extent) rather than `committed`.  The
    // token-sequence equality above is the strict-spec invariant.
    std::printf("[tree_strict_spec] chain_seed=true: %zu tokens identical "
                "(ref_counter=%d topk_counter=%d)\n",
                ref.size(), tgt_ref.ssm_counter_, tgt_topk.ssm_counter_);
}

// ── Test 2: strict equivalence with best-first (no chain seed) ───────
//
// With chain_seed=false the DDTree is pure best-first, so the spine may
// or may not be present.  Either way, the oracle's argmax-driven walk
// determines the accepted prefix, and rollback must put state back to
// match the chain reference's progression.
static void test_strict_equivalence_no_chain_seed() {
    DeterministicOracleTarget tgt_ref;
    auto ref = run_chain_reference(tgt_ref, /*prefill=*/7, /*n_gen=*/64);

    DeterministicOracleTarget tgt_topk;
    auto top = run_topk_tree_mode(tgt_topk, /*prefill=*/7, /*n_gen=*/64,
                                  /*K=*/2, /*ddtree_budget=*/6,
                                  /*chain_seed=*/false, ref);
    CHECK(ref.size() == top.size());
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i] != top[i]) {
            std::fprintf(stderr,
                "strict-spec (no chain seed) MISMATCH at i=%zu: ref=%d top=%d\n",
                i, ref[i], top[i]);
            std::exit(1);
        }
    }
    std::printf("[tree_strict_spec] chain_seed=false: %zu tokens identical "
                "(ref_counter=%d topk_counter=%d)\n",
                ref.size(), tgt_ref.ssm_counter_, tgt_topk.ssm_counter_);
}

// ── Test 3: regression — confirm broken rollback would be caught ─────
//
// Wire a deliberately-broken "no-op restore_kv_at_dfs" target and verify
// that it produces a DIFFERENT sequence from the chain reference.  This
// proves the test has bite: without rollback, the assertion in tests 1
// and 2 would fail loudly.  (We invert the assertion here — divergence
// is the expected outcome.)
struct BrokenRollbackTarget : DeterministicOracleTarget {
    bool restore_kv_at_dfs(const std::vector<int> & accepted_dfs) override {
        (void)accepted_dfs;
        // Intentionally do nothing: leak the over-advanced counter.
        return true;
    }
};
static void test_broken_rollback_diverges() {
    DeterministicOracleTarget tgt_ref;
    auto ref = run_chain_reference(tgt_ref, /*prefill=*/7, /*n_gen=*/64);

    BrokenRollbackTarget tgt_bad;
    auto bad = run_topk_tree_mode(tgt_bad, /*prefill=*/7, /*n_gen=*/64,
                                  /*K=*/2, /*ddtree_budget=*/6,
                                  /*chain_seed=*/true, ref);
    // Must diverge (otherwise the test has no signal).
    bool any_mismatch = false;
    for (size_t i = 0; i < ref.size() && i < bad.size(); i++) {
        if (ref[i] != bad[i]) { any_mismatch = true; break; }
    }
    if (!any_mismatch && ref.size() == bad.size()) {
        std::fprintf(stderr,
            "[tree_strict_spec] FAIL: broken-rollback path DID NOT diverge\n"
            "  → strict-spec test is not actually testing the rollback.\n");
        std::exit(1);
    }
    std::printf("[tree_strict_spec] broken-rollback diverges (control) OK\n");
}

}  // namespace

int main() {
    test_strict_equivalence_chain_seed();
    test_strict_equivalence_no_chain_seed();
    test_broken_rollback_diverges();
    std::printf("[tree_strict_spec] all 3 tests PASS\n");
    return 0;
}
