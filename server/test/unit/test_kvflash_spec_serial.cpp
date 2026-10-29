#include "CppUnitTestFramework.hpp"
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
