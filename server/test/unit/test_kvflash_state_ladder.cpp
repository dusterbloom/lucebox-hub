#include "CppUnitTestFramework.hpp"
#include "../../src/qwen35/prefill_helpers.h"

namespace {
struct KvflashLadderFixture {};
using Pos = std::vector<int>;
}

using namespace luce::common;

TEST_CASE(KvflashLadderFixture, positions_head_stride_and_prompt_end) {
    // head 1300 -> 1024; every 8192; grid point <= end (20000 -> 19968).
    CHECK((kvflash_state_ladder_positions(1300, 20000, 512, 8192) == Pos{1024, 8192, 16384, 19968}));
    // prompt end exactly on the grid; no duplicate with the stride rung.
    CHECK((kvflash_state_ladder_positions(1024, 16384, 512, 8192) == Pos{1024, 8192, 16384}));
    // no head known: that rung is left out.
    CHECK((kvflash_state_ladder_positions(-1, 9000, 512, 8192) == Pos{8192, 8704}));
    // head below one grid cell rounds to 0: left out. Short prompt: nothing.
    CHECK((kvflash_state_ladder_positions(100, 300, 512, 8192) == Pos{}));
    // head beyond the prompt is dropped.
    CHECK((kvflash_state_ladder_positions(5000, 2000, 512, 8192) == Pos{1536}));
}

TEST_CASE(KvflashLadderFixture, pick_greatest_not_above_L) {
    CHECK(kvflash_ladder_pick({}, 5000) == -1);
    const Pos c{1024, 8192, 16384};
    CHECK(kvflash_ladder_pick(c, 1000) == -1);     // below every C
    CHECK(kvflash_ladder_pick(c, 1024) == 1024);   // exactly on C
    CHECK(kvflash_ladder_pick(c, 8191) == 1024);   // between
    CHECK(kvflash_ladder_pick(c, 8192) == 8192);
    CHECK(kvflash_ladder_pick(c, 99999) == 16384); // above all
    CHECK(kvflash_ladder_pick(c, 0) == -1);
}

TEST_CASE(KvflashLadderFixture, evict_nothing_within_budget) {
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 2}, {16384, 3}};
    CHECK(kvflash_ladder_evict(e, 300, 100, 1024).empty());
    CHECK(kvflash_ladder_evict({}, 0, 100, -1).empty());
}

TEST_CASE(KvflashLadderFixture, evict_lru_under_budget_protecting_head_and_newest) {
    // budget holds 3 of 5. Head (1024) and newest (tick 9) are kept;
    // LRU among the rest goes first.
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 5}, {16384, 2}, {24576, 4}, {32768, 9}};
    CHECK((kvflash_ladder_evict(e, 300, 100, 1024) == Pos{16384, 24576}));
}

TEST_CASE(KvflashLadderFixture, pick_prefers_an_off_grid_gen_rung) {
    const Pos c{1024, 4096, 4300};   // 4300 = generation end, off the 512 grid
    CHECK(kvflash_ladder_pick(c, 4299) == 4096);
    CHECK(kvflash_ladder_pick(c, 4300) == 4300);
    CHECK(kvflash_ladder_pick(c, 9000) == 4300);
}

TEST_CASE(KvflashLadderFixture, evict_keeps_the_newest_gen_rung) {
    // The gen rung (4300) is the newest; even with a budget of zero it stays,
    // and the older grid rungs go LRU first, the head last of all.
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {4096, 2}, {8192, 3}, {4300, 9}};
    CHECK((kvflash_ladder_evict(e, 200, 100, 1024) == Pos{4096, 8192}));
    CHECK((kvflash_ladder_evict(e, 0, 100, 1024) == Pos{4096, 8192}));
}

TEST_CASE(KvflashLadderFixture, evict_stops_when_only_protected_remain) {
    const std::vector<KvflashLadderEntry> e{{1024, 1}, {8192, 2}, {16384, 3}};
    // budget fits one; two protected (head, newest) survive regardless.
    CHECK((kvflash_ladder_evict(e, 100, 100, 1024) == Pos{8192}));
    CHECK((kvflash_ladder_evict(e, 0, 100, 1024) == Pos{8192}));
    // no head protection when head is unknown: only the newest is protected.
    CHECK((kvflash_ladder_evict(e, 0, 100, -1) == Pos{1024, 8192}));
}
