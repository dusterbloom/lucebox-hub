// C1.3-04: the pure side of the exact per-slot state ladder and the
// deterministic prefill cuts (D1). The slab capture/restore itself is device
// code (box test); here: the cut rule, the slot-keyed rung store and the
// rewind plan.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_ladder.h"

#include <algorithm>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotLadderFixture {};
std::vector<int> positions(const KvFlashSlotLadder & l, int slot) { return l.positions(slot); }
}  // namespace

// ── D1: cuts depend only on position and U ───────────────────────────────

TEST_CASE(KvflashSlotLadderFixture, cuts_are_multiples_of_u_inside_the_range_and_the_end_closes_it) {
    CHECK(kvflash_cut_sequence(0, 1300, 512) == (std::vector<int>{512, 1024, 1300}));
    CHECK(kvflash_cut_sequence(100, 1300, 512) == (std::vector<int>{512, 1024, 1300}));   // start offset only shortens the first batch
    CHECK(kvflash_cut_sequence(512, 1024, 512) == (std::vector<int>{1024}));
    CHECK(kvflash_cut_sequence(512, 600, 512) == (std::vector<int>{600}));
    CHECK(kvflash_cut_sequence(7, 7, 512).empty());
    CHECK(kvflash_cut_sequence(9, 7, 512).empty());
    CHECK(kvflash_next_cut(0, 100000, 512) == 512);
    CHECK(kvflash_next_cut(511, 100000, 512) == 512);
    CHECK(kvflash_next_cut(512, 100000, 512) == 1024);
    CHECK(kvflash_next_cut(1000, 1010, 512) == 1010);
}

TEST_CASE(KvflashSlotLadderFixture, resuming_at_any_cut_gives_the_same_remaining_cuts) {
    // Idle, mixed and cancelled steps all resume at a recorded boundary: the
    // continuation must be the suffix of the uninterrupted sequence.
    for (int start : {0, 1, 300, 512, 777}) {
        const auto full = kvflash_cut_sequence(start, 4000, 512);
        for (size_t i = 0; i < full.size(); i++) {
            const auto rest = kvflash_cut_sequence(full[i], 4000, 512);
            CHECK(rest == std::vector<int>(full.begin() + (long)i + 1, full.end()));
        }
        for (int c : full) if (c != 4000) CHECK(c % 512 == 0);
    }
}

// ── the slot-keyed rung store ────────────────────────────────────────────

TEST_CASE(KvflashSlotLadderFixture, a_slot_keeps_at_most_its_cap_evicting_its_own_lru_but_never_the_head_or_the_newest) {
    KvFlashSlotLadder l(/*max_rungs_per_slot=*/3);
    l.begin(0, /*generation=*/1, /*head_pos=*/512);
    CHECK(l.capture(0, 1, 512).empty());
    CHECK(l.capture(0, 1, 1024).empty());
    CHECK(l.capture(0, 1, 1536).empty());
    CHECK(positions(l, 0) == (std::vector<int>{512, 1024, 1536}));
    const auto ev = l.capture(0, 1, 2048);              // over the cap: the oldest non-head goes
    CHECK(ev == std::vector<int>{1024});
    CHECK(positions(l, 0) == (std::vector<int>{512, 1536, 2048}));
    l.pick(0, 1, 1700);                                  // using a rung refreshes it (LRU)
    const auto ev2 = l.capture(0, 1, 2560);
    CHECK(ev2 == std::vector<int>{2048});                // 1536 was just used; 2048 is now the oldest non-head, non-newest
}

TEST_CASE(KvflashSlotLadderFixture, the_newest_rung_survives_a_cap_of_one_and_a_recapture_does_not_duplicate) {
    KvFlashSlotLadder l(1);
    l.begin(0, 1, 512);
    l.capture(0, 1, 512);
    CHECK(l.capture(0, 1, 1024).empty());               // only the head and the newest are left: over the cap, nothing lost
    CHECK(positions(l, 0) == (std::vector<int>{512, 1024}));
    CHECK(l.capture(0, 1, 1024).empty());                // same position again: refreshed, not duplicated
    CHECK(positions(l, 0) == (std::vector<int>{512, 1024}));
}

TEST_CASE(KvflashSlotLadderFixture, one_slots_pressure_never_evicts_another_slots_rungs) {
    KvFlashSlotLadder l(2);
    l.begin(0, 1, -1);
    l.begin(1, 1, -1);
    l.capture(1, 1, 512);
    l.capture(1, 1, 1024);
    for (int p = 512; p <= 8192; p += 512) l.capture(0, 1, p);
    CHECK(positions(l, 1) == (std::vector<int>{512, 1024}));
    CHECK(positions(l, 0).size() == 2);
}

TEST_CASE(KvflashSlotLadderFixture, a_stale_generation_neither_captures_nor_restores) {
    KvFlashSlotLadder l(4);
    l.begin(2, 5, -1);
    l.capture(2, 5, 512);
    CHECK(l.pick(2, 5, 700) == 512);
    CHECK(l.pick(2, 4, 700) == -1);                      // the old request
    CHECK(l.pick(2, 6, 700) == -1);                      // a request that has not begun
    CHECK(l.capture(2, 4, 1024).empty());
    CHECK(positions(l, 2) == std::vector<int>{512});     // the stale capture was refused
    l.begin(2, 6, -1);                                   // a new request on the slot drops everything of the old one
    CHECK(positions(l, 2).empty());
    CHECK(l.pick(2, 5, 700) == -1);
    CHECK(l.pick(2, 6, 700) == -1);
    l.capture(2, 6, 512);
    l.release(2);
    CHECK(positions(l, 2).empty());
    CHECK(l.pick(2, 6, 700) == -1);
}

TEST_CASE(KvflashSlotLadderFixture, pick_returns_the_greatest_rung_at_or_below_the_position) {
    KvFlashSlotLadder l(8);
    l.begin(0, 1, -1);
    for (int p : {512, 1024, 2048}) l.capture(0, 1, p);
    CHECK(l.pick(0, 1, 2048) == 2048);
    CHECK(l.pick(0, 1, 2047) == 1024);
    CHECK(l.pick(0, 1, 511) == -1);
    CHECK(l.has(0, 1, 1024));
    CHECK(!l.has(0, 1, 1025));
}

// ── rewind ───────────────────────────────────────────────────────────────

TEST_CASE(KvflashSlotLadderFixture, rewind_to_an_exact_rung_restores_without_replay) {
    KvFlashSlotLadder l(8);
    l.begin(0, 1, -1);
    for (int p : {512, 1024, 1536}) l.capture(0, 1, p);
    const KvFlashRewindPlan r = l.plan_rewind(0, 1, 1024, 512);
    CHECK(r.kind == KvFlashRewindKind::Exact);
    CHECK(r.rung == 1024);
    CHECK(r.replay_from == 1024);
    CHECK(r.replay_cuts.empty());
}

TEST_CASE(KvflashSlotLadderFixture, rewind_between_rungs_restores_the_newest_lower_rung_and_replays_on_the_cuts) {
    KvFlashSlotLadder l(8);
    l.begin(0, 1, -1);
    for (int p : {512, 1024, 1536}) l.capture(0, 1, p);
    const KvFlashRewindPlan r = l.plan_rewind(0, 1, 1300, 512);
    CHECK(r.kind == KvFlashRewindKind::Replay);
    CHECK(r.rung == 1024);
    CHECK(r.replay_from == 1024);
    CHECK(r.replay_cuts == (std::vector<int>{1300}));                  // [1024, 1300)
    const KvFlashRewindPlan r2 = l.plan_rewind(0, 1, 3000, 512);       // beyond every rung
    CHECK(r2.rung == 1536);
    CHECK(r2.replay_cuts == (std::vector<int>{2048, 2560, 3000}));
}

TEST_CASE(KvflashSlotLadderFixture, rewind_below_every_rung_starts_from_a_zero_state) {
    KvFlashSlotLadder l(8);
    l.begin(0, 1, -1);
    l.capture(0, 1, 512);
    const KvFlashRewindPlan r = l.plan_rewind(0, 1, 300, 512);
    CHECK(r.kind == KvFlashRewindKind::FromScratch);
    CHECK(r.rung == -1);
    CHECK(r.replay_from == 0);
    CHECK(r.replay_cuts == (std::vector<int>{300}));
    const KvFlashRewindPlan stale = l.plan_rewind(0, 9, 600, 512);     // wrong generation: nothing restorable
    CHECK(stale.kind == KvFlashRewindKind::FromScratch);
}

TEST_CASE(KvflashSlotLadderFixture, committing_a_rewind_invalidates_every_rung_after_the_target) {
    KvFlashSlotLadder l(8);
    l.begin(0, 1, -1);
    for (int p : {512, 1024, 1536, 2048}) l.capture(0, 1, p);
    l.commit_rewind(0, 1, 1300);
    CHECK(positions(l, 0) == (std::vector<int>{512, 1024}));
    CHECK(l.pick(0, 1, 5000) == 1024);                                 // nothing from the abandoned future survives
    l.commit_rewind(0, 1, 1024);                                       // exact: the rung at the target stays
    CHECK(positions(l, 0) == (std::vector<int>{512, 1024}));
    l.commit_rewind(0, 2, 0);                                          // stale generation: no effect
    CHECK(positions(l, 0) == (std::vector<int>{512, 1024}));
}
