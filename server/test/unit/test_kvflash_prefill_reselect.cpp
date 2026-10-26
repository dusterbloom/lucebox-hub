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
