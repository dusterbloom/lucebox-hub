// KvFlashStepDriver (C1.3-05): preflight + plan + executor over real slot pagers
// (no-tensor path, CPU). The test plays the engine: per step it names the rows
// each slot appends, then reads the table columns the graph would be given.
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"
#include "../../src/common/kvflash_step_driver.h"

#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashStepDriverFixture {};

constexpr int kN = 2, kQ = 12, kChunk = 64;

struct Slot : KvFlashSlotPager {
    Slot() { set_pinned_host_backing(pager_pinned_host_available()); }
};

struct World {
    KvFlashArena arena;
    std::vector<std::unique_ptr<Slot>> s;
    std::unique_ptr<KvFlashStepDriver> drv;
    explicit World(int headroom = 8) {
        arena.set_host_wait([] {});
        std::vector<KvFlashSlotPager *> raw;
        for (int i = 0; i < kN; i++) {
            s.push_back(std::make_unique<Slot>());
            KvFlashConfig c; c.chunk_tokens = kChunk; c.sink_chunks = 1; c.tail_window_chunks = 2;
            arena.add(*s.back());
            if (!s.back()->attach_slice(i, kN, kQ, c, {}, {})) throw std::runtime_error("attach failed");
            raw.push_back(s.back().get());
        }
        drv = std::make_unique<KvFlashStepDriver>(raw, arena, headroom);
    }
};
}  // namespace

TEST_CASE(KvflashStepDriverFixture, fitting_slots_run_pins_expire_and_the_table_is_stale_only_when_the_epoch_moved) {
    World w;
    int32_t col[kQ * 4];
    auto r = w.drv->run_step({{0, 0, 8 * kChunk}, {1, 0, 8 * kChunk}}, nullptr);   // two aligned 512 cuts
    CHECK(r.ok);
    CHECK((r.ran == std::vector<int>{0, 1}));
    for (int i = 0; i < kN; i++) {
        CHECK(w.s[i]->resident_blocks() == 8);
        CHECK(!w.s[i]->is_pinned(0));                             // EndStep expired the step pins
        CHECK(w.drv->table_stale(i));
        w.drv->write_table(i, col, 8, 16);
        CHECK(!w.drv->table_stale(i));
        CHECK(col[0] == w.s[i]->block_of(0) * 4);
    }
    r = w.drv->run_step({{0, 8 * kChunk, 1}}, nullptr);           // the first row of a new chunk
    CHECK(r.ok);
    CHECK(w.drv->table_stale(0));
    w.drv->write_table(0, col, 9, 16);
    r = w.drv->run_step({{0, 8 * kChunk + 1, 1}}, nullptr);       // inside the resident chunk: nothing to upload
    CHECK(r.ok);
    CHECK(!w.drv->table_stale(0));
    CHECK(!w.drv->table_stale(1));                                // the peer did not run
    CHECK(w.drv->scorer_calls() == 0);
}

TEST_CASE(KvflashStepDriverFixture, a_slot_that_does_not_fit_idles_while_its_page_out_settles_and_peers_run) {
    World w;
    REQUIRE(w.drv->run_step({{0, 0, kQ * kChunk}, {1, 0, kChunk}}, nullptr).ok);   // slot 0 holds its whole slice
    const int before = w.s[0]->resident_blocks();
    REQUIRE(before == kQ);
    auto r = w.drv->run_step({{0, kQ * kChunk, kChunk}, {1, kChunk, kChunk}}, nullptr);
    CHECK((r.ran == std::vector<int>{1}));                        // slot 0 needs a block that is not free: idle
    CHECK(w.s[0]->resident_blocks() == kQ - 1);                   // only the recovery page-out was queued
    CHECK(w.s[1]->resident_blocks() == 2);                        // the peer was not held back
    r = w.drv->run_step({{0, kQ * kChunk, kChunk}}, nullptr);     // the recovery page-out settled at step start
    CHECK(r.ok);
    CHECK((r.ran == std::vector<int>{0}));
    CHECK(w.s[0]->block_of(kQ) >= 0);
    int holes = 0;
    for (int c = 0; c < kQ; c++) holes += w.s[0]->block_of(c) < 0;
    CHECK(holes == 1);                                            // LRU evicted exactly one chunk
    CHECK(w.drv->scorer_calls() == 0);
}

TEST_CASE(KvflashStepDriverFixture, abort_expires_the_step_pins_of_every_slot) {
    World w;
    REQUIRE(w.drv->run_step({{0, 0, 3 * kChunk}}, nullptr).ok);
    w.s[0]->pin_step_range(0, kChunk);
    REQUIRE(w.s[0]->is_pinned(0));
    w.drv->abort_step();
    CHECK(!w.s[0]->is_pinned(0));
}

TEST_CASE(KvflashStepDriverFixture, zero_scorer_calls_over_a_run_that_evicts_and_decodes) {
    World w;
    int64_t pos[kN] = {0, 0};
    for (int step = 0; step < 40; step++) {
        std::vector<KvFlashStepWork> work;
        for (int i = 0; i < kN; i++) work.push_back({i, pos[i], kChunk});
        const auto r = w.drv->run_step(work, nullptr);
        REQUIRE(r.ok);
        for (int i : r.ran) pos[i] += work[(size_t)i].tokens;
    }
    CHECK(pos[0] > kQ * kChunk);                                  // it ran past the quota: eviction happened
    CHECK(w.drv->scorer_calls() == 0);                            // C1.3 is LRU only (D6)
}

TEST_CASE(KvflashStepDriverFixture, preflight_keeps_slots_with_enough_completed_blocks_in_order) {
    const std::vector<KvFlashPreflightSlot> in = {{3, 8, 8}, {1, 7, 8}, {0, 0, 1}, {2, 5, 1}};
    CHECK((kvflash_preflight_step(in) == std::vector<int>{3, 2}));
    CHECK(kvflash_preflight_step({}).empty());
    CHECK((kvflash_preflight_step({{4, 0, 0}}) == std::vector<int>{4}));   // a slot that needs nothing always runs
}

TEST_CASE(KvflashStepDriverFixture, lru_victims_are_oldest_first_and_skip_sink_tail_and_pinned_chunks) {
    World w;
    REQUIRE(w.drv->run_step({{0, 0, 8 * kChunk}}, nullptr).ok);       // chunks 0..7, last_use ascending
    auto v = w.s[0]->lru_victims(3);
    CHECK((v == std::vector<int>{1, 2, 3}));                          // 0 is the sink, 5..7 are the tail window
    w.s[0]->pin_step_range(kChunk, 2 * kChunk);                       // chunk 1 pinned
    CHECK((w.s[0]->lru_victims(2) == std::vector<int>{2, 3}));
    CHECK((w.s[0]->lru_victims(99) == std::vector<int>{2, 3, 4}));   // 5..7 are the tail window, 0 the sink, 1 pinned
    CHECK(w.s[0]->lru_victims(0).empty());
}

TEST_CASE(KvflashStepDriverFixture, settle_pages_a_full_slot_down_to_the_usable_quota_and_never_past_the_headroom) {
    World w(4);                                                       // usable quota 12 - 4 = 8
    REQUIRE(w.drv->run_step({{0, 0, kQ * kChunk}, {1, 0, 3 * kChunk}}, nullptr).ok);
    CHECK(w.s[0]->resident_blocks() == kQ);
    CHECK(w.drv->settle({0, 1}) == 4);                                // slot 0: 12 -> 8; slot 1 is under quota
    CHECK(w.s[0]->resident_blocks() == 8);
    CHECK(w.s[1]->resident_blocks() == 3);
    CHECK(w.drv->settle({0, 1}) == 0);                                // idempotent
    auto r = w.drv->run_step({{0, kQ * kChunk, 4 * kChunk}}, nullptr);   // the settled page-outs complete at step start
    CHECK((r.ran == std::vector<int>{0}));                            // a 4-chunk cut fits at once
    CHECK(w.drv->scorer_calls() == 0);
}

TEST_CASE(KvflashStepDriverFixture, a_cut_that_lacks_several_blocks_queues_them_all_in_one_idle_step) {
    World w(4);
    REQUIRE(w.drv->run_step({{0, 0, kQ * kChunk}}, nullptr).ok);
    auto r = w.drv->run_step({{0, kQ * kChunk, 4 * kChunk}}, nullptr);
    CHECK(r.ran.empty());                                             // full slice: the cut idles
    CHECK(w.s[0]->resident_blocks() == kQ - 4);                       // but all four missing blocks are on their way out
    r = w.drv->run_step({{0, kQ * kChunk, 4 * kChunk}}, nullptr);
    CHECK((r.ran == std::vector<int>{0}));                            // one step later it runs, not four
}
