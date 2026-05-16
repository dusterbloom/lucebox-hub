// test_mtp_chain_strict_spec.cpp — Phase A correctness invariant for
// the new INativeMtp::step_chain entry point.
//
// Strict-speculation equivalence: drafting at chain_depth=D must propose
// the SAME D tokens as D successive depth=1 calls in which each iter
// uses the previous iter's post-shared_head_norm hidden as h_prev.  A
// broken chain implementation that re-feeds the initial h_prev for every
// iteration (a common refactor regression) will diverge from the
// reference at iter 1 and later.
//
// Test tier: T1 — no model file, no GPU, CPU only.
//
// This test does NOT exercise Qwen36MtpModule's GPU body (no backend in
// CI). Instead it uses deterministic stub INativeMtp subclasses that
// model the chain transformation in closed form: a "good" chain advances
// its internal state per iter; a "broken" chain re-feeds the iter-0
// h_prev every step.  The runner-level invariant is the same either way.

#include "common/mtp_interface.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash27b;
using namespace dflash27b::mtp;

#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "%s:%d CHECK(%s) FAILED\n",                     \
                     __FILE__, __LINE__, #cond);                             \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

namespace {

// Deterministic hidden transform: a one-token "module" whose state is a
// single float (stand-in for the post-shared_head_norm hidden).  The
// transform is:
//
//   h_next  = 0.5 * h_prev + 0.25 * embed(cur_token)
//   token   = floor(h_next * 100) mod V    (deterministic, depends on h)
//
// where embed(tok) = sin(tok * 0.1) + cos(tok * 0.05) is a closed-form
// stand-in for the backbone embedding lookup.  Cheap, stateless,
// torture-tests whether the chain is actually threading h_prev through.

static constexpr int kVocab = 64;

static double fake_embed(int32_t tok) {
    const double t = (double)tok;
    return std::sin(t * 0.1) + std::cos(t * 0.05);
}

static int32_t hidden_to_token(double h) {
    // Map an arbitrary float to a vocab index.  abs+floor+mod is
    // deterministic and exercises every byte of the float (so any drift
    // in h_prev propagates to a token mismatch).
    long long bits = (long long)(h * 100.0);
    if (bits < 0) bits = -bits;
    return (int32_t)(bits % kVocab);
}

// ── GoodChainMtp ──────────────────────────────────────────────────────
// step_chain advances h_prev between iterations like Qwen36MtpModule's
// new Phase A path: iter h>0 uses the post-norm hidden written by iter
// h-1.  step_batch returns a single draft (the iter-0 result), so the
// default INativeMtp::step_chain (= step_batch clamped to depth=1) would
// produce ONLY the first token of a Good chain.
struct GoodChainMtp : INativeMtp {
    int max_gamma() const override { return 8; }
    int hidden_size() const override { return 4; }
    bool attach(DFlashTarget *) override { return true; }
    void reset_chain() override {}
    void shutdown() override {}

    int num_heads() const override { return 1; }

    bool step_batch(int32_t cur, int /*base_pos*/,
                    std::vector<StepOutput> & out) override {
        // Single-draft fallback: iter-0 only.
        out.clear();
        const double h0 = 0.0;  // first call has no prior hidden
        const double h_next = 0.5 * h0 + 0.25 * fake_embed(cur);
        StepOutput so;
        so.draft_token = hidden_to_token(h_next);
        so.draft_logit = (float)h_next;
        out.push_back(so);
        return true;
    }

    // The good Phase A behaviour: thread the post-norm hidden between
    // iterations.  Mirrors the contract of Qwen36MtpModule::step_chain.
    bool step_chain(int32_t cur, int /*base_pos*/, int chain_depth,
                    std::vector<StepOutput> & out) override {
        out.clear();
        if (chain_depth <= 0) chain_depth = 1;
        double h_prev = 0.0;  // iter 0 has no prior hidden (per Qwen3.6 contract)
        int32_t cur_tok = cur;
        for (int it = 0; it < chain_depth; it++) {
            const double h_next = 0.5 * h_prev + 0.25 * fake_embed(cur_tok);
            StepOutput so;
            so.draft_token = hidden_to_token(h_next);
            so.draft_logit = (float)h_next;
            out.push_back(so);
            h_prev  = h_next;          // <-- the critical advance
            cur_tok = so.draft_token;  // <-- advance current token too
        }
        return true;
    }
};

// ── BrokenRefeedMtp ───────────────────────────────────────────────────
// Same transform, but iter h>0 re-feeds h_prev = 0 every time (the
// "forgot to advance" bug catcher).  The argmax progression diverges
// from the good chain at iter 1 — but only if the test runner is
// actually inspecting more than one iter's output.  Failure of this
// divergence assertion would mean the test has no teeth.
struct BrokenRefeedMtp : INativeMtp {
    int max_gamma() const override { return 8; }
    int hidden_size() const override { return 4; }
    bool attach(DFlashTarget *) override { return true; }
    void reset_chain() override {}
    void shutdown() override {}

    int num_heads() const override { return 1; }

    bool step_batch(int32_t cur, int /*base_pos*/,
                    std::vector<StepOutput> & out) override {
        out.clear();
        const double h_next = 0.5 * 0.0 + 0.25 * fake_embed(cur);
        StepOutput so;
        so.draft_token = hidden_to_token(h_next);
        so.draft_logit = (float)h_next;
        out.push_back(so);
        return true;
    }

    bool step_chain(int32_t cur, int /*base_pos*/, int chain_depth,
                    std::vector<StepOutput> & out) override {
        out.clear();
        if (chain_depth <= 0) chain_depth = 1;
        int32_t cur_tok = cur;
        for (int it = 0; it < chain_depth; it++) {
            // BUG: always pretend h_prev is zero (never threads the
            // previous iter's output).  cur_tok still advances, so the
            // first draft matches the good chain — but draft 1, 2, ...
            // diverge.
            const double h_next = 0.5 * 0.0 + 0.25 * fake_embed(cur_tok);
            StepOutput so;
            so.draft_token = hidden_to_token(h_next);
            so.draft_logit = (float)h_next;
            out.push_back(so);
            cur_tok = so.draft_token;
        }
        return true;
    }
};

// ── Drive an INativeMtp at a given chain depth ────────────────────────
static std::vector<int32_t> drive(INativeMtp & mtp,
                                  int32_t prefill_tok,
                                  int chain_depth) {
    std::vector<StepOutput> outs;
    CHECK(mtp.step_chain(prefill_tok, /*base_pos=*/0, chain_depth, outs));
    std::vector<int32_t> toks;
    toks.reserve(outs.size());
    for (auto & so : outs) toks.push_back(so.draft_token);
    return toks;
}

// ── Test 1: depth=1 equivalence ───────────────────────────────────────
// GoodChainMtp at depth=1 must equal GoodChainMtp's step_batch fallback
// (single token).  This pins the no-regression contract for callers
// today (gamma=1 paths must behave identically).
static void test_depth1_byte_identical() {
    GoodChainMtp good;
    auto d1 = drive(good, /*prefill=*/7, /*chain_depth=*/1);
    CHECK(d1.size() == 1);

    // Independently compute the step_batch fallback.
    std::vector<StepOutput> sb_out;
    CHECK(good.step_batch(7, 0, sb_out));
    CHECK(sb_out.size() == 1);
    CHECK(sb_out[0].draft_token == d1[0]);
    std::printf("[chain_strict_spec] depth=1 byte-identical: token=%d OK\n", d1[0]);
}

// ── Test 2: depth=3 reference equals iterated depth=1 (good) ──────────
// Drive the Good chain at depth=3 in a single call; compare with three
// successive depth=1-with-manual-advance calls (i.e. emulate the chain
// by hand at the test level).  They must match exactly.
static std::vector<int32_t> reference_chain(int32_t prefill_tok, int depth) {
    std::vector<int32_t> out;
    out.reserve(depth);
    double h_prev = 0.0;
    int32_t cur = prefill_tok;
    for (int it = 0; it < depth; it++) {
        const double h_next = 0.5 * h_prev + 0.25 * fake_embed(cur);
        const int32_t tok = hidden_to_token(h_next);
        out.push_back(tok);
        h_prev = h_next;
        cur    = tok;
    }
    return out;
}

static void test_good_chain_matches_reference() {
    GoodChainMtp good;
    auto got = drive(good, /*prefill=*/7, /*chain_depth=*/3);
    auto ref = reference_chain(/*prefill=*/7, /*depth=*/3);
    CHECK(got.size() == ref.size());
    for (size_t i = 0; i < ref.size(); i++) {
        if (got[i] != ref[i]) {
            std::fprintf(stderr,
                "[chain_strict_spec] good-chain MISMATCH at i=%zu: got=%d ref=%d\n",
                i, got[i], ref[i]);
            std::exit(1);
        }
    }
    std::printf("[chain_strict_spec] depth=3 good-chain matches reference: "
                "[%d, %d, %d] OK\n", got[0], got[1], got[2]);
}

// ── Test 3: BrokenRefeedMtp diverges from reference ───────────────────
// The control case — if this assertion fires we know the test has bite.
// (If a future refactor accidentally produces the same tokens for both
// implementations, the test silently passes; this assertion guards
// against that.)
static void test_broken_chain_diverges() {
    BrokenRefeedMtp bad;
    auto got = drive(bad, /*prefill=*/7, /*chain_depth=*/3);
    auto ref = reference_chain(/*prefill=*/7, /*depth=*/3);
    CHECK(got.size() == ref.size());

    // Iter 0 must match (BrokenRefeed and Good agree at iter 0 because
    // both start from h_prev=0 — the divergence appears from iter 1).
    CHECK(got[0] == ref[0]);

    bool any_mismatch = false;
    for (size_t i = 1; i < ref.size(); i++) {
        if (got[i] != ref[i]) { any_mismatch = true; break; }
    }
    if (!any_mismatch) {
        std::fprintf(stderr,
            "[chain_strict_spec] FAIL: broken-refeed chain DID NOT diverge "
            "from reference — test has no teeth.\n");
        std::exit(1);
    }
    std::printf("[chain_strict_spec] broken-refeed diverges (control) OK "
                "[good=%d,%d,%d  bad=%d,%d,%d]\n",
                ref[0], ref[1], ref[2], got[0], got[1], got[2]);
}

// ── Test 4: default step_chain forwards to step_batch ─────────────────
// Sanity check on the base-class default — INativeMtp::step_chain with
// no override must forward to step_batch and return ALL emitted drafts
// (preserves pre-Phase-A multi-head semantics).  A module that emits N
// heads per step_batch call returns N drafts regardless of chain_depth.
struct DefaultMultiHeadMtp : INativeMtp {
    int max_gamma() const override { return 4; }
    int hidden_size() const override { return 4; }
    bool attach(DFlashTarget *) override { return true; }
    void reset_chain() override {}
    void shutdown() override {}
    int num_heads() const override { return 2; }
    bool step_batch(int32_t cur, int, std::vector<StepOutput> & out) override {
        out.clear();
        for (int h = 0; h < 2; h++) {
            StepOutput so;
            so.draft_token = cur + 1 + h;
            so.draft_logit = 1.0f;
            out.push_back(so);
        }
        return true;
    }
};
static void test_default_step_chain_forwards_to_step_batch() {
    DefaultMultiHeadMtp def;
    std::vector<StepOutput> outs;
    // chain_depth=4 requested, but the default impl simply forwards to
    // step_batch which emits num_heads=2 drafts.  No clamping.
    CHECK(def.step_chain(/*cur=*/41, /*base_pos=*/0, /*depth=*/4, outs));
    CHECK(outs.size() == 2);
    CHECK(outs[0].draft_token == 42);
    CHECK(outs[1].draft_token == 43);
    std::printf("[chain_strict_spec] default step_chain forwards to step_batch "
                "(returned %zu drafts) OK\n", outs.size());
}

}  // namespace

int main() {
    test_depth1_byte_identical();
    test_good_chain_matches_reference();
    test_broken_chain_diverges();
    test_default_step_chain_forwards_to_step_batch();
    std::printf("[chain_strict_spec] all 4 tests PASS\n");
    return 0;
}
