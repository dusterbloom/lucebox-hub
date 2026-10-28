#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/prefill_helpers.h"

namespace {
struct KvflashLiveRestoreFixture {};
using Toks = std::vector<int32_t>;
Toks seq(int n, int base = 0) {
    Toks t((size_t)n);
    for (int i = 0; i < n; i++) t[(size_t)i] = base + i;
    return t;
}
const std::vector<int> kRungs = {1024, 2048, 4096};
}

using namespace luce::common;

TEST_CASE(KvflashLiveRestoreFixture, identical_prompt_resumes_below_the_end) {
    // History = prompt + generated answer; same prompt again.
    Toks hist = seq(5000);
    Toks prompt = seq(4500);
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 4500);
    CHECK(d.C == 4096);
    CHECK(d.replay_from == 4096);
    CHECK(d.discard_above == 4500);
}

TEST_CASE(KvflashLiveRestoreFixture, rung_at_prompt_end_still_replays_last_token) {
    Toks hist = seq(4096);
    Toks prompt = seq(4096);
    auto d = kvflash_live_restore_decide(prompt, hist, 4096, kRungs);
    CHECK(d.L == 4096);
    CHECK(d.C == 2048);   // 4096 would leave nothing to prefill
    CHECK(d.replay_from == 2048);
}

TEST_CASE(KvflashLiveRestoreFixture, extended_prompt_new_turn) {
    Toks hist = seq(4300);              // prompt 4100 + 200 generated
    Toks prompt = seq(4300);
    for (int i = 0; i < 300; i++) prompt.push_back(90000 + i);   // new turn
    auto d = kvflash_live_restore_decide(prompt, hist, 4300, kRungs);
    CHECK(d.L == 4300);
    CHECK(d.C == 4096);
    CHECK(d.replay_from == 4096);
}

TEST_CASE(KvflashLiveRestoreFixture, divergence_in_the_middle) {
    Toks hist = seq(5000);
    Toks prompt = seq(5000);
    prompt[3000] = -7;
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 3000);
    CHECK(d.C == 2048);
    CHECK(d.discard_above == 3000);   // the 4096 rung is stale
}

TEST_CASE(KvflashLiveRestoreFixture, divergence_below_first_rung) {
    Toks hist = seq(5000);
    Toks prompt = seq(5000);
    prompt[500] = -7;
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 500);
    CHECK(d.C == -1);
    CHECK(d.replay_from == 0);
    CHECK(d.discard_above == 500);
}

TEST_CASE(KvflashLiveRestoreFixture, lcp_exactly_on_a_rung) {
    Toks hist = seq(5000);
    Toks prompt = seq(5000);
    prompt[2048] = -7;   // tokens [0,2048) match
    auto d = kvflash_live_restore_decide(prompt, hist, 5000, kRungs);
    CHECK(d.L == 2048);
    CHECK(d.C == 2048);
    CHECK(d.replay_from == 2048);
}

TEST_CASE(KvflashLiveRestoreFixture, empty_history_and_empty_ladder) {
    auto d = kvflash_live_restore_decide(seq(3000), {}, 0, kRungs);
    CHECK(d.L == 0);
    CHECK(d.C == -1);
    CHECK(d.replay_from == 0);
    auto e = kvflash_live_restore_decide(seq(3000), seq(3000), 3000, {});
    CHECK(e.L == 3000);
    CHECK(e.C == -1);
    auto f = kvflash_live_restore_decide({}, seq(10), 10, kRungs);
    CHECK(f.C == -1);
}

TEST_CASE(KvflashLiveRestoreFixture, history_capped_at_cur_pos) {
    // History longer than what the KV holds: tokens past cur_pos are unproven.
    Toks hist = seq(6000);
    Toks prompt = seq(6000);
    auto d = kvflash_live_restore_decide(prompt, hist, 3000, kRungs);
    CHECK(d.L == 3000);
    CHECK(d.C == 2048);
}

// ---- S9: generation-end checkpoint at an off-grid position ----------------

TEST_CASE(KvflashLiveRestoreFixture, gen_rung_off_grid_is_picked_when_prompt_extends_history) {
    // G = 4300 is not a multiple of 512; the next turn re-renders the whole
    // history and appends a new user turn.
    const std::vector<int> rungs = {1024, 2048, 4096, 4300};
    Toks hist = seq(4301);                       // 4300 in KV + 1 pending token
    Toks prompt = seq(4301);
    for (int i = 0; i < 80; i++) prompt.push_back(90000 + i);
    auto d = kvflash_live_restore_decide(prompt, hist, 4300, rungs);
    CHECK(d.L == 4300);
    CHECK(d.C == 4300);
    CHECK(d.replay_from == 4300);
    CHECK((int)prompt.size() - d.C == 81);       // only the new tokens replay
    CHECK(d.discard_above == 4300);              // the gen rung survives
}

TEST_CASE(KvflashLiveRestoreFixture, gen_rung_dropped_when_template_rewrites_the_answer) {
    const std::vector<int> rungs = {1024, 2048, 4096, 4300};
    Toks hist = seq(4300);
    Toks prompt = seq(4400);
    prompt[4250] = -5;                           // the re-rendered answer differs
    auto d = kvflash_live_restore_decide(prompt, hist, 4300, rungs);
    CHECK(d.L == 4250);
    CHECK(d.C == 4096);
    CHECK(d.discard_above == 4250);              // 4300 is stale and gets erased
}

TEST_CASE(KvflashLiveRestoreFixture, gen_ckpt_position_needs_history_to_cover_cur_pos) {
    CHECK(kvflash_gen_ckpt_pos(4300, 4301) == 4300);   // one pending token is fine
    CHECK(kvflash_gen_ckpt_pos(4300, 4300) == 4300);
    CHECK(kvflash_gen_ckpt_pos(4300, 4000) == -1);     // history lacks generated ids
    CHECK(kvflash_gen_ckpt_pos(0, 10) == -1);
    CHECK(kvflash_gen_ckpt_pos(-3, 10) == -1);
}

TEST_CASE(KvflashLiveRestoreFixture, early_stop_inside_a_verified_block_needs_a_state_repair) {
    CHECK(spec_early_stop_needs_repair(4, 2, false));    // EOS or cancel mid-block
    CHECK(spec_early_stop_needs_repair(4, 1, false));
    CHECK(!spec_early_stop_needs_repair(4, 4, false));   // whole block emitted (EOS last)
    CHECK(!spec_early_stop_needs_repair(1, 1, false));   // plain decode step
    CHECK(!spec_early_stop_needs_repair(4, 2, true));    // budget close rewinds on its own
}

// ---- spec decode: committed ids are appended to the history ---------------

TEST_CASE(KvflashLiveRestoreFixture, history_append_takes_generated_ids_after_entry) {
    Toks hist = seq(100);
    const Toks out = {7, 8, 9, 10};              // out_tokens, nothing before entry
    kvflash_history_set_generated(hist, 100, out, 0);
    CHECK(hist.size() == 104);
    CHECK(hist[100] == 7);
    CHECK(hist[103] == 10);
}

TEST_CASE(KvflashLiveRestoreFixture, history_append_is_idempotent_over_ar_fallback) {
    // The AR fallback inside spec decode already pushed some ids; resyncing
    // from out_tokens must not duplicate them.
    Toks hist = seq(100);
    hist.push_back(7); hist.push_back(8);
    const Toks out = {7, 8, 9};
    kvflash_history_set_generated(hist, 100, out, 0);
    CHECK(hist.size() == 103);
    CHECK(hist[102] == 9);
}

TEST_CASE(KvflashLiveRestoreFixture, history_append_skips_tokens_before_entry) {
    Toks hist = seq(50);
    const Toks out = {1, 2, 3, 4};               // 2 came from an earlier phase
    kvflash_history_set_generated(hist, 50, out, 2);
    CHECK(hist.size() == 52);
    CHECK(hist[50] == 3);
    CHECK(hist[51] == 4);
}

TEST_CASE(KvflashLiveRestoreFixture, history_append_nothing_generated_truncates_to_base) {
    Toks hist = seq(60);
    const Toks out = {};
    kvflash_history_set_generated(hist, 55, out, 0);
    CHECK(hist.size() == 55);
}

TEST_CASE(KvflashLiveRestoreFixture, turn_rung_catches_the_final_answer_rerender) {
    // Previous prompt Pend = 5000, turn rung at Pend-4 = 4996, gen ckpt G = 5300.
    // The new prompt re-renders the old assistant turn without the think block,
    // so it matches the history only up to 4996 and then diverges.
    Toks hist = seq(5300);
    Toks prompt = seq(4996);
    for (int i = 0; i < 400; i++) prompt.push_back(90000 + i);
    const std::vector<int> rungs = {1024, 4608, 4996, 5300};
    auto d = kvflash_live_restore_decide(prompt, hist, 5300, rungs);
    CHECK(d.L == 4996);
    CHECK(d.C == 4996);
    CHECK(d.replay_from == 4996);
    CHECK(d.discard_above == 4996);   // the gen rung 5300 is dropped
    // without the turn rung it falls back to the grid
    auto e = kvflash_live_restore_decide(prompt, hist, 5300, {1024, 4608, 5300});
    CHECK(e.C == 4608);
}

// ---- Stage S (PREREG_G3C): spec step commit, gen rung, QK pooling ---------

// S-U1: one spec step. committed = position at the step start, accept_n draft
// tokens accepted (the seed counts), has_bonus = the target's extra token,
// budget_left clamps what may be committed, eos/cancel_index = index inside
// the committed block where the emit loop stops (-1 = never), and a budget
// close ends the block at close_index (-1 = no close).
TEST_CASE(KvflashLiveRestoreFixture, spec_step_whole_block_no_stop) {
    auto s = spec_step_commit(1000, 5, true, 100, -1, -1, -1);
    CHECK(s.commit_n == 6);                  // 5 accepted + bonus
    CHECK(s.emitted == 6);
    CHECK(!s.needs_repair);
    CHECK(s.state_pos == 1006);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_step_bonus_is_clamped_by_the_budget) {
    auto a = spec_step_commit(1000, 5, true, 5, -1, -1, -1);
    CHECK(a.commit_n == 5);                  // bonus dropped
    CHECK(a.emitted == 5);
    CHECK(!a.needs_repair);
    auto b = spec_step_commit(1000, 5, true, 3, -1, -1, -1);
    CHECK(b.commit_n == 3);                  // accepted tokens beyond budget dropped
    CHECK(b.state_pos == 1003);
    CHECK(spec_step_commit(1000, 5, false, 100, -1, -1, -1).commit_n == 5);  // fast rollback: no bonus
}

TEST_CASE(KvflashLiveRestoreFixture, spec_step_eos_at_first_middle_last) {
    auto first = spec_step_commit(1000, 5, true, 100, 0, -1, -1);
    CHECK(first.commit_n == 6);
    CHECK(first.emitted == 1);               // EOS itself is emitted
    CHECK(first.needs_repair);
    CHECK(first.state_pos == 1001);          // not 1006
    auto mid = spec_step_commit(1000, 5, true, 100, 3, -1, -1);
    CHECK(mid.emitted == 4);
    CHECK(mid.needs_repair);
    CHECK(mid.state_pos == 1004);
    auto last = spec_step_commit(1000, 5, true, 100, 5, -1, -1);
    CHECK(last.emitted == 6);
    CHECK(!last.needs_repair);               // EOS is the last committed token
    CHECK(last.state_pos == 1006);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_step_cancel_at_first_middle_last) {
    auto first = spec_step_commit(2000, 8, true, 100, -1, 0, -1);
    CHECK(first.emitted == 1);
    CHECK(first.needs_repair);
    CHECK(first.state_pos == 2001);
    auto mid = spec_step_commit(2000, 8, true, 100, -1, 4, -1);
    CHECK(mid.emitted == 5);
    CHECK(mid.needs_repair);
    auto last = spec_step_commit(2000, 8, true, 100, -1, 8, -1);
    CHECK(last.emitted == 9);
    CHECK(!last.needs_repair);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_step_stop_index_past_the_commit_is_ignored) {
    // An EOS or cancel index at or beyond commit_n is not inside this block.
    auto s = spec_step_commit(1000, 3, false, 100, 3, 7, -1);
    CHECK(s.commit_n == 3);
    CHECK(s.emitted == 3);
    CHECK(!s.needs_repair);
    // Budget clamp moves the block end in front of an EOS that was inside the unclamped block.
    auto t = spec_step_commit(1000, 5, true, 2, 4, -1, -1);
    CHECK(t.commit_n == 2);
    CHECK(t.emitted == 2);
    CHECK(!t.needs_repair);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_step_earliest_of_eos_and_cancel_wins) {
    auto s = spec_step_commit(1000, 8, true, 100, 6, 2, -1);
    CHECK(s.emitted == 3);
    CHECK(s.needs_repair);
    auto t = spec_step_commit(1000, 8, true, 100, 1, 5, -1);
    CHECK(t.emitted == 2);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_step_budget_close_repairs_by_itself) {
    auto s = spec_step_commit(1000, 5, true, 100, -1, -1, 2);
    CHECK(s.commit_n == 6);
    CHECK(s.emitted == 3);                   // close token is the third emitted
    CHECK(!s.needs_repair);                  // the budget-close path has its own repair
    CHECK(s.state_pos == 1003);
    // A close before an EOS: the close ends the block first.
    auto t = spec_step_commit(1000, 5, true, 100, 4, -1, 1);
    CHECK(t.emitted == 2);
    CHECK(!t.needs_repair);
}

TEST_CASE(KvflashLiveRestoreFixture, spec_step_plain_decode_step_never_repairs) {
    auto s = spec_step_commit(1000, 1, false, 100, 0, -1, -1);
    CHECK(s.commit_n == 1);
    CHECK(s.emitted == 1);
    CHECK(!s.needs_repair);
}

// S-U2: generation-end rung G, spec turn ended on EOS vs an AR turn.
TEST_CASE(KvflashLiveRestoreFixture, gen_rung_spec_eos_history_vs_ar_pending_token) {
    const int prompt = 4000, answer = 300;
    // Spec ending on EOS: KV holds prompt + answer including the EOS row.
    const int spec_cur_pos = prompt + answer;
    const size_t spec_hs = (size_t)(prompt + answer);          // hs == cp
    // AR: the final token is pending, not in KV.
    const int ar_cur_pos = prompt + answer - 1;
    const size_t ar_hs = (size_t)(prompt + answer);            // hs == cp + 1
    const int g_spec = kvflash_gen_ckpt_pos(spec_cur_pos, spec_hs);
    const int g_ar = kvflash_gen_ckpt_pos(ar_cur_pos, ar_hs);
    CHECK(g_spec == 4300);
    CHECK(g_ar == 4299);
    CHECK(g_spec - g_ar == 1);               // G differs by one token: tests must not compare it to AR
    // History one short of cur_pos (junk or missing id) refuses the checkpoint.
    CHECK(kvflash_gen_ckpt_pos(spec_cur_pos, spec_hs - 1) == -1);
}

TEST_CASE(KvflashLiveRestoreFixture, gen_rung_after_early_stop_repair_matches_emitted_ids) {
    // Block of 6 committed, EOS at index 2: the repair leaves cur_pos at
    // committed + emitted, and the history holds exactly those ids.
    auto s = spec_step_commit(4000, 5, true, 100, 2, -1, -1);
    Toks hist = seq(4000);
    const Toks out = {11, 12, 13};
    kvflash_history_set_generated(hist, 4000, out, 0);
    CHECK((int)hist.size() == s.state_pos);
    CHECK(kvflash_gen_ckpt_pos(s.state_pos, hist.size()) == s.state_pos);
}

// S-U3: QK pooling after a spec step.
TEST_CASE(KvflashLiveRestoreFixture, qk_pool_range_covers_sealed_chunks_only) {
    auto r = kvflash_qk_pool_range(3, 64 * 5 + 10, 64);
    CHECK(r.from == 3);
    CHECK(r.to == 5);                        // chunks 3 and 4 are sealed, 5 is open
    auto none = kvflash_qk_pool_range(5, 64 * 5 + 63, 64);
    CHECK(none.from == none.to);
    auto back = kvflash_qk_pool_range(7, 64 * 5, 64);   // pooled index never moves back
    CHECK(back.from == back.to);
    CHECK(back.to == 7);
}

TEST_CASE(KvflashLiveRestoreFixture, qk_pool_spec_step_sealing_a_chunk_advances_the_index) {
    const int ct = 64;
    int upto = 2;
    // A spec step from committed=190 that commits 6 tokens crosses chunk 2 -> 3 boundary at 192.
    int committed = 190;
    auto s = spec_step_commit(committed, 5, true, 100, -1, -1, -1);
    auto r = kvflash_qk_pool_range(upto, s.state_pos, ct);
    CHECK(r.from == 2);
    CHECK(r.to == 3);
    upto = r.to;
    // Next step stays inside chunk 3: nothing to pool.
    committed = s.state_pos;
    auto s2 = spec_step_commit(committed, 5, true, 100, -1, -1, -1);
    auto r2 = kvflash_qk_pool_range(upto, s2.state_pos, ct);
    CHECK(r2.from == r2.to);
}

TEST_CASE(KvflashLiveRestoreFixture, qk_pool_a_16_token_block_seals_at_most_one_chunk) {
    // The pooled chunk must still be resident at seal time: with 64-token
    // chunks and blocks of at most 16 tokens a step never skips a chunk.
    for (int c = 0; c < 200; c++) {
        auto r = kvflash_qk_pool_range(c / 64, c + 16, 64);
        CHECK(r.to - r.from <= 1);
    }
}
