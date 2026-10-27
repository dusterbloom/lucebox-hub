// Unit test for the LUCE_KVFLASH_PREFILL_RESELECT pure helpers (no GPU).

#include "CppUnitTestFramework.hpp"

#include "kvflash_prefill_reselect.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashPrefillReselectFixture {};
}

TEST_CASE(KvflashPrefillReselectFixture, empty_cuts_gives_end_minus_1) {
    CHECK(kvflash_prefill_reselect_point({}, 0, 1000) == 999);
}
TEST_CASE(KvflashPrefillReselectFixture, last_cut_inside_wins) {
    CHECK(kvflash_prefill_reselect_point({100, 900}, 0, 1000) == 900);
}
TEST_CASE(KvflashPrefillReselectFixture, cut_at_end_falls_back) {
    CHECK(kvflash_prefill_reselect_point({100, 1000}, 0, 1000) == 999);
}
TEST_CASE(KvflashPrefillReselectFixture, cut_at_or_before_offset_falls_back) {
    CHECK(kvflash_prefill_reselect_point({500}, 500, 1000) == 999);
    CHECK(kvflash_prefill_reselect_point({400}, 500, 1000) == 999);
}
TEST_CASE(KvflashPrefillReselectFixture, one_token_prefill_is_disabled) {
    CHECK(kvflash_prefill_reselect_point({}, 999, 1000) == -1);
    CHECK(kvflash_prefill_reselect_point({}, 0, 1) == -1);
}
TEST_CASE(KvflashPrefillReselectFixture, cuts_with_inserts_sorted) {
    CHECK((kvflash_cuts_with({100, 900}, 500) == std::vector<int>{100, 500, 900}));
    CHECK((kvflash_cuts_with({}, 7) == std::vector<int>{7}));
    CHECK((kvflash_cuts_with({100, 900}, 900) == std::vector<int>{100, 900}));
    CHECK((kvflash_cuts_with({100, 900}, -1) == std::vector<int>{100, 900}));
}

TEST_CASE(KvflashPrefillReselectFixture, mode2_ignores_cuts_and_uses_end_minus_1) {
    CHECK(kvflash_prefill_reselect_point({100, 900}, 0, 1000, 2) == 999);
    CHECK(kvflash_prefill_reselect_point({}, 0, 1000, 2) == 999);
}
TEST_CASE(KvflashPrefillReselectFixture, mode2_one_token_prefill_is_disabled) {
    CHECK(kvflash_prefill_reselect_point({}, 999, 1000, 2) == -1);
    CHECK(kvflash_prefill_reselect_point({500}, 999, 1000, 2) == -1);
}
TEST_CASE(KvflashPrefillReselectFixture, mode1_is_the_default) {
    CHECK(kvflash_prefill_reselect_point({100, 900}, 0, 1000, 1) == 900);
}

TEST_CASE(KvflashPrefillReselectFixture, mode3_point_is_end_minus_1_like_mode2) {
    CHECK(kvflash_prefill_reselect_point({100, 900}, 0, 1000, 3) == 999);
    CHECK(kvflash_prefill_reselect_point({}, 999, 1000, 3) == -1);
}
TEST_CASE(KvflashPrefillReselectFixture, mode3_plan_runs_with_one_chunk_reserve) {
    const KvFlashMode3Plan p = kvflash_mode3_plan(16, 1000);
    CHECK(p.run);
    CHECK(p.r == 999);
    CHECK(p.reserve_chunks == 1);
}
TEST_CASE(KvflashPrefillReselectFixture, mode3_plan_skips_one_token_prefill) {
    CHECK(!kvflash_mode3_plan(999, 1000).run);   // end-1 == kv_offset: no prior state to snapshot
    CHECK(!kvflash_mode3_plan(0, 1).run);
    CHECK(kvflash_mode3_plan(0, 2).run);
}

TEST_CASE(KvflashPrefillReselectFixture, max_abs_diff_and_argmax) {
    CHECK(kvflash_max_abs_diff({1.f, 2.f, 3.f}, {1.f, 2.5f, 2.f}) == 1.0f);
    CHECK(kvflash_max_abs_diff({1.f, 2.f}, {1.f, 2.f}) == 0.0f);
    CHECK(kvflash_max_abs_diff({1.f}, {1.f, 2.f}) < 0.0f);   // size mismatch: -1
    CHECK(kvflash_argmax({0.f, 5.f, 3.f}) == 1);
    CHECK(kvflash_argmax({}) == -1);
}

TEST_CASE(KvflashPrefillReselectFixture, last_user_turn_finds_the_last_im_start_user) {
    // 1=im_start 2=user 3=assistant 9=other
    CHECK(kvflash_last_user_turn({9, 1, 2, 9, 1, 3, 9, 1, 2, 9, 9}, 1, 2) == 7);
    CHECK(kvflash_last_user_turn({9, 1, 3, 9}, 1, 2) == -1);       // no user turn
    CHECK(kvflash_last_user_turn({1}, 1, 2) == -1);
    CHECK(kvflash_last_user_turn({1, 2}, -1, 2) == -1);            // unknown id
}
TEST_CASE(KvflashPrefillReselectFixture, mode4_plan_uses_last_user_turn_clamped_to_prefill_start) {
    const std::vector<int> toks = {9, 1, 2, 9, 1, 3, 9, 1, 2, 9, 9};
    KvFlashMode4Plan p = kvflash_mode4_plan(toks, 100, 1, 2);
    CHECK(p.run);
    CHECK(p.r == 107);                       // kv_offset + local index 7
    CHECK(p.reserve_chunks == -1);           // derived from the tokens left to prefill
    // No user turn in the (delta) tokens: fall back to the prefill start.
    p = kvflash_mode4_plan({9, 1, 3, 9}, 100, 1, 2);
    CHECK(p.run);
    CHECK(p.r == 100);
    // Cold prefill with the user turn at the very start: R4 = 0 (zero state).
    p = kvflash_mode4_plan({1, 2, 9}, 0, 1, 2);
    CHECK(p.run);
    CHECK(p.r == 0);
    CHECK(!kvflash_mode4_plan({}, 0, 1, 2).run);
}

TEST_CASE(KvflashPrefillReselectFixture, format_ranges_collapses_runs) {
    CHECK(kvflash_format_ranges({}) == "");
    CHECK(kvflash_format_ranges({0, 3, 4, 5, 6, 7, 8, 9, 12}) == "0,3-9,12");
    CHECK(kvflash_format_ranges({1, 2, 5}) == "1-2,5");
    CHECK(kvflash_format_ranges({7}) == "7");
}

TEST_CASE(KvflashPrefillReselectFixture, pool_fits_predicate_edges) {
    // 250 chunks + 6 reserve == 256 blocks: still fits (strict no-op).
    CHECK(kvflash_pool_fits(250 * 64, 64, 6, 256));
    // One more reserve chunk, or one more token past the chunk edge: the pool is full.
    CHECK(!kvflash_pool_fits(250 * 64, 64, 7, 256));
    CHECK(!kvflash_pool_fits(250 * 64 + 1, 64, 6, 256));
    CHECK(kvflash_pool_fits(10, 64, 1, 256));
    CHECK(kvflash_pool_fits(0, 64, 0, 256));
    CHECK(!kvflash_pool_fits(100, 0, 1, 256));   // unusable chunk size: never claim it fits
}

// ---- per-slot recall plan (P2-01) --------------------------------------------
namespace {
// The pre-P1 derivation of qwen35_backend.cpp kvflash_prefill_apply_on, kept verbatim as the oracle for k = 0.
struct LegacyRecall { int tail, first_window, first_chunk; std::vector<int> picked; };
LegacyRecall legacy_recall(const std::vector<float> & s, int Q, int cur, int recall_chunks, int unit, int reserve,
                           int pin_reserve, int pinned) {
    LegacyRecall r;
    const int k_units = recall_chunks / unit;
    r.first_chunk = kvflash_recall_first_chunk(pinned);
    r.tail = kvflash_recall_tail_chunks(Q, recall_chunks, reserve, pin_reserve);
    r.first_window = cur - r.tail;
    r.picked = kvflash_recall_pick(s, r.first_window, k_units, unit, r.first_chunk);
    return r;
}
std::vector<float> ramp(int n) { std::vector<float> s((size_t)n); for (int i = 0; i < n; i++) s[(size_t)i] = (float)((i * 37) % 101); return s; }
KvFlashRecallIn slot_in(int Q, int cur, int k, int recall = 240, int reserve = 3) {
    KvFlashRecallIn in;
    in.n_pool_blocks = Q; in.cur_chunk = cur; in.recall_chunks = recall; in.unit = 10; in.reserve = reserve;
    in.pin_reserve_chunks = k; in.pinned_chunks = k; in.sink_chunks = 1; in.min_tail_chunks = 4; in.slot_pins = k > 0;
    return in;
}
}  // namespace

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_with_zero_pins_is_byte_identical_to_the_legacy_derivation) {
    for (int Q : {64, 128, 256, 300})
        for (int recall : {0, 10, 100, 240, 400})
            for (int reserve : {1, 3, 40})
                for (int pin_reserve : {0, 7})            // the serial path's extra capacity
                    for (int pinned : {0, 5}) {           // and its explicit pins
                        const int cur = Q + 90;
                        const auto s = ramp(cur);
                        KvFlashRecallIn in;
                        in.n_pool_blocks = Q; in.cur_chunk = cur; in.recall_chunks = recall; in.unit = 10;
                        in.reserve = reserve; in.pin_reserve_chunks = pin_reserve; in.pinned_chunks = pinned;
                        in.slot_pins = false; in.sink_chunks = 1; in.min_tail_chunks = 4;
                        const KvFlashRecallPlan p = kvflash_recall_plan(s, in);
                        const LegacyRecall l = legacy_recall(s, Q, cur, recall, 10, reserve, pin_reserve, pinned);
                        CHECK(p.tail == l.tail);
                        CHECK(p.first_window == l.first_window);
                        CHECK(p.first_chunk == l.first_chunk);
                        CHECK(p.picked == l.picked);
                        CHECK(!p.clamped);
                    }
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_slot_with_k_zero_matches_the_legacy_derivation) {
    const auto s = ramp(400);
    const KvFlashRecallPlan p = kvflash_recall_plan(s, slot_in(256, 400, 0));
    const LegacyRecall l = legacy_recall(s, 256, 400, 240, 10, 3, 0, 0);
    CHECK(p.tail == l.tail);
    CHECK(p.picked == l.picked);
    CHECK(p.first_chunk == 10);
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_two_slots_with_different_k) {
    const auto s = ramp(400);
    const KvFlashRecallPlan a = kvflash_recall_plan(s, slot_in(256, 400, 0, 100));
    const KvFlashRecallPlan b = kvflash_recall_plan(s, slot_in(256, 400, 18, 100));
    // candidate exclusion: slot b never picks a block that starts inside its 18 pinned chunks
    CHECK(a.first_chunk == 10);
    CHECK(b.first_chunk == 18);
    for (int blk : b.picked) CHECK(blk * 10 >= 18);
    // the pinned chunks come off the history budget: the tail shrinks by exactly k
    CHECK(b.tail == a.tail - 18);
    CHECK(b.first_window == a.first_window + 18);
    // retained picks: 10 requested, 10 retained, no clamp, and the plan fits Q with headroom
    CHECK((int)b.picked.size() == 10);
    CHECK(!b.clamped);
    CHECK(18 + b.tail + 10 * 10 + 3 + 2 <= 256);
    CHECK(b.tail >= 4);
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_saturation_clamps_recall_rather_than_the_tail_only) {
    // the review's arithmetic: Q = 256, k = 26, RECALL_BLOCKS 24 -> 240 chunks asked; 26 + 240 + 3 + 1 + 4 + 2 > 256
    const auto s = ramp(600);
    const KvFlashRecallIn in = slot_in(256, 600, 26, 240, 3);
    const KvFlashRecallPlan p = kvflash_recall_plan(s, in);
    CHECK(p.clamped);
    CHECK(p.fits);
    CHECK(p.k_units * 10 <= 256 - 1 - 4 - 26 - 3 - 2);
    CHECK(p.k_units == 22);                                   // 220 chunks: the most that fits
    CHECK(p.tail >= 4);                                       // the tail window is still there
    CHECK(26 + p.tail + p.k_units * 10 + 3 + 2 <= 256);
    CHECK((int)p.picked.size() <= p.k_units);
    // the old formula would have asked for 240 picks and a zero tail
    const LegacyRecall l = legacy_recall(s, 256, 600, 240, 10, 3, 26, 26);
    CHECK(l.tail == 0);
    // retained picks: every pick survives the Q-candidate reselect because pins + tail + picks + reserve fit
    CHECK(26 + p.tail + (int)p.picked.size() * 10 + 3 + 2 <= 256);
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_pins_that_leave_no_room_drop_the_recall_and_say_so) {
    const auto s = ramp(100);
    KvFlashRecallPlan p = kvflash_recall_plan(s, slot_in(40, 100, 30, 100, 3));   // 1 + 4 + 30 + 3 + 2 = 40: zero room
    CHECK(p.fits);
    CHECK(p.k_units == 0);
    CHECK(p.picked.empty());
    p = kvflash_recall_plan(s, slot_in(30, 100, 30, 100, 3));                      // pins alone exceed the pool
    CHECK(!p.fits);
    CHECK(p.k_units == 0);
    CHECK(p.tail == 0);
}

TEST_CASE(KvflashPrefillReselectFixture, recall_plan_unit_other_than_ten) {
    const auto s = ramp(600);
    KvFlashRecallIn in = slot_in(256, 600, 26, 240, 3);
    in.unit = 5;
    const KvFlashRecallPlan p = kvflash_recall_plan(s, in);
    CHECK(p.clamped);
    CHECK(p.k_units * 5 <= 256 - 1 - 4 - 26 - 3 - 2);
    for (int blk : p.picked) CHECK(blk * 5 >= 26);
}

TEST_CASE(KvflashPrefillReselectFixture, dump_line_carries_picks_residency_and_scores) {
    CHECK(kvflash_dump_line(3, 100, 40, 10, {7, 2}, {0, 1, 7}, {0.5f, 1.25f}) ==
          "{\"seq\":3,\"c\":100,\"fw\":40,\"u\":10,\"picked\":[7,2],\"res\":[0,1,7],\"sc\":[0.5,1.25]}");
    CHECK(kvflash_dump_line(0, 0, 0, 10, {}, {}, {}) == "{\"seq\":0,\"c\":0,\"fw\":0,\"u\":10,\"picked\":[],\"res\":[],\"sc\":[]}");
}

TEST_CASE(KvflashPrefillReselectFixture, plane_norms_are_the_l2_norm_of_each_layer_head_plane) {
    const float q[6] = {3.f, 4.f, 0.f, 0.f, 1.f, 0.f};   // 3 planes of head_dim 2
    const auto n = kvflash_q_plane_norms(q, 3, 2);
    CHECK(n.size() == 3); CHECK(n[0] == 5.0f); CHECK(n[1] == 0.0f); CHECK(n[2] == 1.0f);
}
