#include "CppUnitTestFramework.hpp"
#include <functional>
#include "../../src/common/kvflash_pager.h"
#include "../../src/qwen35/prefill_helpers.h"

namespace {
struct KvflashSpecSerialFixture {};
}

using namespace luce::common;

// F1: DDTree commits every accepted token before it checks termination and
// never settles, so on EOS/cancel/budget end it finishes one position past AR.
// DDTree is opt-in and no launcher uses it: --ddtree with KVFlash fails closed.
TEST_CASE(KvflashSpecSerialFixture, ddtree_with_kvflash_is_refused) {
    CHECK(!kvflash_ddtree_refusal(true, "4096", nullptr).empty());      // --kvflash 4096
    CHECK(!kvflash_ddtree_refusal(true, "auto", nullptr).empty());
    CHECK(!kvflash_ddtree_refusal(true, "", "8192").empty());           // LUCE_KVFLASH env
    CHECK(!kvflash_ddtree_refusal(true, "", "auto").empty());
}

TEST_CASE(KvflashSpecSerialFixture, ddtree_alone_or_kvflash_alone_is_accepted) {
    CHECK(kvflash_ddtree_refusal(true, "", nullptr).empty());
    CHECK(kvflash_ddtree_refusal(true, "", "0").empty());               // pool 0 = off
    CHECK(kvflash_ddtree_refusal(false, "4096", nullptr).empty());
    CHECK(kvflash_ddtree_refusal(false, "", "8192").empty());
}

TEST_CASE(KvflashSpecSerialFixture, ddtree_refusal_is_one_line_naming_both_flags) {
    const std::string m = kvflash_ddtree_refusal(true, "4096", nullptr);
    CHECK(m.find("--ddtree") != std::string::npos);
    CHECK(m.find("--kvflash") != std::string::npos);
    CHECK(m.find('\n') == std::string::npos);
}

// F2: a plain-decode (adaptive one-token) step used to skip the snapshot when
// the budget was settled and ample. io.emit can still end the request on that
// step (cancel, stop string), and the repair restores the snapshot. Every
// terminal one-token step must repair to AR's state: the emitted token is
// pending, cur_pos stays at the step start. The loop now snapshots every step.
TEST_CASE(KvflashSpecSerialFixture, one_token_step_cancelled_after_emit_ends_in_ars_state) {
    const int committed = 700;
    // The step verified the seed (pending at 700) and emitted it; the request
    // ends there. AR ends in the same place: the emitted token is pending, its
    // row not consumed, cur_pos 700.
    const auto s = spec_step_settle(committed, /*commit_n=*/1, /*emitted=*/1,
                                    /*budget_close_fired=*/false, /*terminal=*/true);
    CHECK(s.repair);                      // needs the pre-step snapshot
    CHECK(s.replay_n == 0);
    CHECK(s.new_committed == committed);  // not committed + 1
}

TEST_CASE(KvflashSpecSerialFixture, one_token_step_that_keeps_going_needs_no_repair) {
    const auto s = spec_step_settle(700, 1, 1, false, false);
    CHECK(!s.repair);
    CHECK(s.new_committed == 701);
}

// F3/F4/F5: a verify span never contains a reselect point except at its first
// row, and never reaches into a chunk that is not allocated yet except p0's
// own. The step starting at the reselect point then reselects first, exactly
// where AR does, from the query of the last row before it.
namespace {
const std::function<bool(int)> kNoneAllocated = [](int) { return false; };
const std::function<bool(int)> kAllAllocated = [](int) { return true; };
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_without_kvflash_or_points_is_the_proposal) {
    CHECK(kvflash_verify_span_cap(100, 16, {}, 0, kNoneAllocated) == 16);
    CHECK(kvflash_verify_span_cap(100, 1, {}, 0, kNoneAllocated) == 1);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_stops_before_a_reselect_point) {
    // point at p0 + 5: rows 0..4 only; the row at the point starts the next step.
    CHECK(kvflash_verify_span_cap(60, 16, {65}, 0, kAllAllocated) == 5);
    // point at p0 + n: the whole span fits, the point is the next step's first row.
    CHECK(kvflash_verify_span_cap(60, 5, {65}, 0, kAllAllocated) == 5);
    // point at p0 + 1: a single row.
    CHECK(kvflash_verify_span_cap(60, 16, {61}, 0, kAllAllocated) == 1);
    // point beyond the span: no clamp.
    CHECK(kvflash_verify_span_cap(60, 4, {65}, 0, kAllAllocated) == 4);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_a_point_at_p0_is_the_first_row_and_does_not_clamp) {
    // The step reselects before row 0, then runs the whole span.
    CHECK(kvflash_verify_span_cap(64, 16, {64}, 0, kAllAllocated) == 16);
    // Points before p0 are history.
    CHECK(kvflash_verify_span_cap(70, 16, {64}, 0, kAllAllocated) == 16);
    // The nearest point after p0 wins.
    CHECK(kvflash_verify_span_cap(64, 16, {64, 70, 80}, 0, kAllAllocated) == 6);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_n_one_is_never_clamped_below_one) {
    CHECK(kvflash_verify_span_cap(60, 1, {61}, 64, kNoneAllocated) == 1);
    CHECK(kvflash_verify_span_cap(447, 1, {}, 64, kNoneAllocated) == 1);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_does_not_allocate_a_chunk_ahead_of_ar) {
    // 64-token chunks, 448-token pool = 7 chunks (0..6). p0 = 447 is the last
    // row of chunk 6, chunk 7 is not allocated: AR would not touch it for one
    // more token, so the span is one row.
    const std::function<bool(int)> upto6 = [](int c) { return c <= 6; };
    CHECK(kvflash_verify_span_cap(447, 16, {}, 64, upto6) == 1);
    // p0 at a chunk start: its own chunk is allocated by the first row, as AR does.
    CHECK(kvflash_verify_span_cap(448, 16, {}, 64, upto6) == 16);
    // p0 mid-chunk, the span ends inside the chunk: no clamp.
    CHECK(kvflash_verify_span_cap(400, 16, {}, 64, upto6) == 16);
    // The next chunk is already allocated (rows from an earlier rejected span): no clamp.
    const std::function<bool(int)> upto7 = [](int c) { return c <= 7; };
    CHECK(kvflash_verify_span_cap(447, 16, {}, 64, upto7) == 16);
    // The crossing needs a new chunk 3 rows in.
    CHECK(kvflash_verify_span_cap(445, 16, {}, 64, upto6) == 3);
}

TEST_CASE(KvflashSpecSerialFixture, span_cap_takes_the_tighter_of_point_and_chunk) {
    const std::function<bool(int)> upto6 = [](int c) { return c <= 6; };
    CHECK(kvflash_verify_span_cap(445, 16, {447}, 64, upto6) == 2);
    CHECK(kvflash_verify_span_cap(445, 16, {460}, 64, upto6) == 3);
}

// Reselect points as positions: AR reselects after emitting generated token r
// (r % tau == 0), before forwarding it; token r sits at p0 + (r - g0 - 1) for a
// step at p0 that has emitted g0 tokens (the seed is token g0 + 1).
TEST_CASE(KvflashSpecSerialFixture, reselect_positions_map_generated_index_to_position) {
    // hist0 100, tau 64: r = 64 is the only point up to 70. g0 = 60, p0 = 160.
    CHECK(kvflash_reselect_positions(100, 60, 160, 16, 64) == std::vector<int>({163}));
    // Nothing in range.
    CHECK(kvflash_reselect_positions(100, 0, 100, 16, 64).empty());
    // r == g0 + 1 is the seed's own point: handled before the step, not listed.
    CHECK(kvflash_reselect_positions(100, 63, 163, 16, 64).empty());
    CHECK(kvflash_reselect_positions(100, 0, 100, 16, 0).empty());   // tau off
}

// q_cap holds the query of the last consumed row. A spec step stages the
// query of every verify row; after acceptance only the accepted rows count,
// so q_cap must equal what AR holds after the same number of steps: the query
// of position (new_committed - 1). Rejected rows never reach q_cap.
TEST_CASE(KvflashSpecSerialFixture, q_stage_row_picks_the_last_consumed_position) {
    // verify at p0 = 100, 8 rows (positions 100..107).
    CHECK(kvflash_q_stage_row(100, 8, 101) == 0);      // 1 row consumed: new_committed 101
    CHECK(kvflash_q_stage_row(100, 8, 104) == 3);      // accepted 4
    CHECK(kvflash_q_stage_row(100, 8, 108) == 7);      // whole block
    CHECK(kvflash_q_stage_row(100, 8, 100) == -1);     // nothing consumed: q_cap stays
    CHECK(kvflash_q_stage_row(100, 8, 109) == -1);     // beyond the staged rows
    CHECK(kvflash_q_stage_row(-1, 0, 50) == -1);       // nothing staged
}

TEST_CASE(KvflashSpecSerialFixture, q_cap_after_k_of_n_accepted_equals_q_cap_after_k_ar_steps) {
    const int n = 8, per_row = 6, p0 = 100;
    // AR: one query per step; q_cap after step at position p holds q(p).
    auto query = [&](int pos, int i) { return (float)(pos * 10 + i); };
    std::vector<float> stage((size_t)n * per_row);
    for (int r = 0; r < n; r++) for (int i = 0; i < per_row; i++) stage[(size_t)r * per_row + i] = query(p0 + r, i);
    for (int k = 1; k <= n; k++) {
        std::vector<float> ar_q_cap(per_row);
        for (int i = 0; i < per_row; i++) ar_q_cap[i] = query(p0 + k - 1, i);   // after k AR steps
        std::vector<float> spec_q_cap(per_row, -1.0f);
        const int row = kvflash_q_stage_row(p0, n, p0 + k);
        CHECK(row == k - 1);
        std::copy(stage.begin() + (size_t)row * per_row, stage.begin() + (size_t)(row + 1) * per_row,
                  spec_q_cap.begin());
        CHECK(spec_q_cap == ar_q_cap);
    }
}
