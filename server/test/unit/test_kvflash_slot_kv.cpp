// KvFlashSlotKv (C1.3-05): the SlotKv over slot pagers, no-tensor path (CPU).
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"
#include "../../src/common/concurrency/kvflash_slot_kv.h"
#include "../../src/common/concurrency/seq_slot_manager.h"
#include "../../src/common/kvflash_arena.h"

#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotKvFixture {};

constexpr int kN = 2, kQ = 12, kChunk = 64;
constexpr int64_t kMaxCtx = 4096, kChunkBytes = 1000;

struct Slot : KvFlashSlotPager {
    Slot() { set_pinned_host_backing(pager_pinned_host_available()); }
};

struct World {
    KvFlashArena arena;
    std::vector<std::unique_ptr<Slot>> s;
    KvFlashHostLedger ledger;
    std::unique_ptr<KvFlashSlotKv> kv;
    explicit World(int64_t host_budget = 1000000) : ledger(host_budget, 0) {
        arena.set_host_wait([] {});
        std::vector<KvFlashSlotPager *> raw;
        for (int i = 0; i < kN; i++) {
            s.push_back(std::make_unique<Slot>());
            KvFlashConfig c; c.chunk_tokens = kChunk; c.sink_chunks = 1; c.tail_window_chunks = 2;
            arena.add(*s.back());
            if (!s.back()->attach_slice(i, kN, kQ, c, {}, {})) throw std::runtime_error("attach failed");
            raw.push_back(s.back().get());
        }
        KvFlashSlotKvConfig cfg; cfg.quota_chunks = kQ; cfg.headroom_blocks = 2; cfg.max_ctx = kMaxCtx; cfg.chunk_bytes = kChunkBytes;
        kv = std::make_unique<KvFlashSlotKv>(raw, ledger, cfg);
    }
};
}  // namespace

TEST_CASE(KvflashSlotKvFixture, acquire_gives_distinct_slots_then_exhausts_and_release_recycles) {
    World w;
    PagedKvSequenceHandle a, b, c;
    CHECK(w.kv->acquire_reserved(1, 256, a) == PagedKvStatus::Ok);
    CHECK(w.kv->acquire_reserved(2, 256, b) == PagedKvStatus::Ok);
    CHECK(a.slot != b.slot);
    CHECK(w.kv->acquire_reserved(3, 256, c) == PagedKvStatus::SequenceSlotsExhausted);
    CHECK(w.kv->acquire_reserved(1, 256, c) == PagedKvStatus::DuplicateRequest);
    CHECK(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(w.kv->release(a) == PagedKvStatus::StaleHandle);       // a handle outlives nothing
    CHECK(w.kv->acquire_reserved(3, 256, c) == PagedKvStatus::Ok);
    CHECK(c.slot == a.slot);
    CHECK(c.generation != a.generation);
    CHECK(w.kv->owned_block_count(a, *std::make_unique<uint32_t>()) == PagedKvStatus::StaleHandle);
}

TEST_CASE(KvflashSlotKvFixture, host_ledger_decides_busy_vs_permanent_and_release_gives_it_back) {
    World w(/*budget=*/10 * kChunkBytes);                         // ten chunks of host backing in total
    PagedKvSequenceHandle a, b;
    CHECK(w.kv->acquire_reserved(1, 6 * kChunk, a) == PagedKvStatus::Ok);           // 6 chunks reserved
    CHECK(w.ledger.used_bytes() == 6 * kChunkBytes);
    CHECK(w.kv->acquire_reserved(2, 6 * kChunk, b) == PagedKvStatus::BlocksExhausted);   // transient: BusyHost
    CHECK(w.kv->acquire_reserved(2, 11 * kChunk, b) == PagedKvStatus::InvalidArgument);  // permanent: HostTooSmall
    CHECK(w.kv->acquire_reserved(2, (uint32_t)kMaxCtx + kChunk + 1, b) == PagedKvStatus::InvalidArgument);   // PromptTooLong
    CHECK(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(w.ledger.used_bytes() == 0);
    CHECK(w.kv->acquire_reserved(2, 6 * kChunk, b) == PagedKvStatus::Ok);
}

TEST_CASE(KvflashSlotKvFixture, append_rows_stay_inside_the_slots_own_slice) {
    World w;
    PagedKvSequenceHandle a, b;
    REQUIRE(w.kv->acquire_reserved(1, 512, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->acquire_reserved(2, 512, b) == PagedKvStatus::Ok);
    for (PagedKvSequenceHandle h : {a, b}) {
        const PagedKvAppendResult r = w.kv->append(h, 200);
        REQUIRE(r);
        REQUIRE(r.write_slots.size() == 200);
        const KvFlashSlice sl = w.s[h.slot]->slice();
        for (uint32_t i = 0; i < 200; ++i) {
            CHECK(r.write_slots[i].logical_position == i);
            CHECK((int64_t)r.write_slots[i].physical_token_index >= sl.row_lo);
            CHECK((int64_t)r.write_slots[i].physical_token_index < sl.row_hi);
            CHECK((int)r.write_slots[i].physical_token_index == w.s[h.slot]->slot_of(i));
        }
    }
}

TEST_CASE(KvflashSlotKvFixture, full_slice_queues_a_page_out_and_succeeds_after_completion) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, (kQ + 4) * kChunk, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, kQ * kChunk));                        // slice full: Q chunks resident
    const PagedKvAppendResult blocked = w.kv->append(a, kChunk);
    CHECK(blocked.status == PagedKvStatus::BlocksExhausted);      // victim paged out, block in flight
    CHECK(blocked.write_slots.empty());
    CHECK(w.s[a.slot]->resident_blocks() <= kQ);
    CHECK(w.arena.complete_all());
    const PagedKvAppendResult ok = w.kv->append(a, kChunk);       // retry: same positions, nothing was consumed
    REQUIRE(ok);
    CHECK(ok.first.logical_position == (uint32_t)(kQ * kChunk));
    int null_entries = 0;
    for (int c = 0; c < kQ + 1; ++c) null_entries += w.s[a.slot]->block_of(c) < 0;
    CHECK(null_entries >= 1);                                     // the evicted chunk is a hole (-1) in the table column
    uint32_t owned = 0;
    CHECK(w.kv->owned_block_count(a, owned) == PagedKvStatus::Ok);
    CHECK(owned == (uint32_t)kQ + 4);                             // owns its reservation; evicted history is still owned
}

TEST_CASE(KvflashSlotKvFixture, append_beyond_the_reserved_capacity_is_refused) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 2 * kChunk, a) == PagedKvStatus::Ok);
    CHECK(w.kv->append(a, 2 * kChunk));
    CHECK(w.kv->append(a, 1).status == PagedKvStatus::BlocksExhausted);
    CHECK(w.kv->reserve_capacity(a, 2 * kChunk) == PagedKvStatus::Ok);
    CHECK(w.kv->reserve_capacity(a, 3 * kChunk) == PagedKvStatus::BlocksExhausted);
}

TEST_CASE(KvflashSlotKvFixture, rollback_and_clear_rewind_the_pager) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 6 * kChunk, a) == PagedKvStatus::Ok);
    const PagedKvAppendResult first = w.kv->append(a, 130);
    REQUIRE(first);
    CHECK(w.s[a.slot]->resident_blocks() == 3);
    CHECK(w.kv->rollback_append(a, 70) == PagedKvStatus::Ok);     // 60 tokens left: one chunk
    CHECK(w.s[a.slot]->resident_blocks() == 1);
    CHECK(w.kv->rollback_append(a, 61) == PagedKvStatus::InvalidArgument);
    const PagedKvAppendResult again = w.kv->append(a, 10);
    REQUIRE(again);
    CHECK(again.first.logical_position == 60);
    CHECK(w.kv->clear(a) == PagedKvStatus::Ok);
    CHECK(w.s[a.slot]->resident_blocks() == 0);
    CHECK(w.kv->append(a, 1).first.logical_position == 0);
}

TEST_CASE(KvflashSlotKvFixture, seq_slot_manager_runs_over_it_unchanged) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    CHECK(mgr.slot_count() == kN);
    const std::vector<int32_t> prompt(300, 1);
    const SeqEngine::AdmitResult r1 = mgr.admit(1, prompt, SamplerCfg{});
    const SeqEngine::AdmitResult r2 = mgr.admit(2, prompt, SamplerCfg{});
    REQUIRE(r1.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(r2.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(r1.slot != r2.slot);
    CHECK(mgr.admit(3, prompt, SamplerCfg{}).status == SeqEngine::AdmitResult::Status::busy);
    for (const auto & r : {r1, r2}) {
        const SeqSlotManager::PrefillChunk ch = mgr.append_prefill(r.slot, 300);
        REQUIRE(ch.ok);
        REQUIRE(ch.rows.size() == 300);
        const KvFlashSlice sl = w.s[(size_t)r.slot]->slice();      // manager slot == kv slot: same first-free order
        for (int64_t row : ch.rows) { CHECK(row >= sl.row_lo); CHECK(row < sl.row_hi); }
    }
    mgr.retire(r1.slot);
    CHECK(w.ledger.reservation_count() == 1);
    CHECK(mgr.admit(4, prompt, SamplerCfg{}).status == SeqEngine::AdmitResult::Status::admitted);
}

TEST_CASE(KvflashSlotKvFixture, failed_ubatch_rolls_back_to_the_last_committed_boundary) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, (kQ + 4) * kChunk, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, 130));                                // committed
    const PagedKvAppendResult ubatch = w.kv->append(a, 100);      // the ubatch whose graph then fails
    REQUIRE(ubatch);
    CHECK(w.s[a.slot]->resident_blocks() == 4);
    CHECK(w.kv->rollback_append(a, 100) == PagedKvStatus::Ok);
    CHECK(w.s[a.slot]->resident_blocks() == 3);
    CHECK(w.s[a.slot]->block_of(3) < 0);                          // its table entry is null again
    const PagedKvAppendResult again = w.kv->append(a, 100);
    REQUIRE(again);
    CHECK(again.first.logical_position == 130);
    CHECK(again.first.physical_token_index == ubatch.first.physical_token_index);   // same rows, nothing leaked
}

TEST_CASE(KvflashSlotKvFixture, retire_frees_the_slice_the_reservation_and_nulls_the_table_column) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, 512, a) == PagedKvStatus::Ok);
    REQUIRE(w.kv->append(a, 300));
    int32_t col[5 * 4];
    w.s[a.slot]->fill_block_table(col, 5, 16);
    CHECK(col[0] >= 0);
    CHECK(w.kv->release(a) == PagedKvStatus::Ok);
    CHECK(w.s[a.slot]->resident_blocks() == 0);
    CHECK(w.s[a.slot]->stats().host_bytes == 0);
    CHECK(w.ledger.reservation_count() == 0);
    for (int32_t & e : col) e = 7;
    w.s[a.slot]->fill_block_table(col, 5, 16);
    for (int32_t e : col) CHECK(e == -1);                         // the column is all null, not stale
}

TEST_CASE(KvflashSlotKvFixture, kvflash_slots_are_not_parkable) {
    World w;
    CHECK(!w.kv->parkable());
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    CHECK(!mgr.parkable());
    const SeqEngine::AdmitResult r = mgr.admit(1, std::vector<int32_t>(300, 1), SamplerCfg{});
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(mgr.append_prefill(r.slot, 300).ok);
    CHECK(!mgr.detach_kv(r.slot));                                // parking would clear the slice
    CHECK(!mgr.evict_for_recompute(r.slot, 5));
    CHECK(mgr.is_active(r.slot));
}

TEST_CASE(KvflashSlotKvFixture, resident_prefix_covers_refuses_an_evicted_or_unmaterialized_prefix) {
    World w;
    PagedKvSequenceHandle a;
    REQUIRE(w.kv->acquire_reserved(1, (kQ + 4) * kChunk, a) == PagedKvStatus::Ok);
    const KvFlashSlotPager & p = *w.s[a.slot];
    REQUIRE(w.kv->append(a, 3 * kChunk));
    CHECK(p.resident_prefix_covers(0));
    CHECK(p.resident_prefix_covers(1));
    CHECK(p.resident_prefix_covers(3 * kChunk - 5));              // a partial last chunk counts whole
    CHECK(!p.resident_prefix_covers(3 * kChunk + 1));             // chunk 3 does not exist yet
    REQUIRE(w.kv->append(a, (kQ - 3) * kChunk));                  // slice full
    CHECK(p.resident_prefix_covers(kQ * kChunk));
    CHECK(w.kv->append(a, kChunk).status == PagedKvStatus::BlocksExhausted);   // queues a page-out
    CHECK(w.arena.complete_all());
    REQUIRE(w.kv->append(a, kChunk));
    int hole = -1;
    for (int c = 0; c < kQ + 1 && hole < 0; ++c) if (p.block_of(c) < 0) hole = c;
    REQUIRE(hole > 0);                                            // the sink (chunk 0) is never the victim
    CHECK(p.resident_prefix_covers(hole * kChunk));               // everything before the hole is still covered
    CHECK(!p.resident_prefix_covers(hole * kChunk + 1));
    CHECK(!p.resident_prefix_covers((kQ + 1) * kChunk));
}

TEST_CASE(KvflashSlotKvFixture, the_total_token_bound_sets_the_host_reservation_and_the_slot_capacity_for_one_acquire) {
    World w;
    PagedKvSequenceHandle a, b;
    w.kv->set_next_total_tokens(7 * kChunk);
    REQUIRE(w.kv->acquire_reserved(1, 2 * kChunk, a) == PagedKvStatus::Ok);
    CHECK(w.ledger.used_bytes() == 7 * kChunkBytes);              // prompt 2 + new 5 chunks of backing
    CHECK(w.kv->reserve_capacity(a, 7 * kChunk) == PagedKvStatus::Ok);       // decode may grow to the bound
    CHECK(w.kv->reserve_capacity(a, 7 * kChunk + 1) == PagedKvStatus::BlocksExhausted);
    REQUIRE(w.kv->acquire_reserved(2, 2 * kChunk, b) == PagedKvStatus::Ok);  // the bound was one-shot
    CHECK(w.ledger.used_bytes() == 9 * kChunkBytes);
    CHECK(w.kv->reserve_capacity(b, 3 * kChunk) == PagedKvStatus::BlocksExhausted);
}

TEST_CASE(KvflashSlotKvFixture, a_decode_page_past_max_ctx_is_not_a_prompt_too_long) {
    World w;
    PagedKvSequenceHandle a;
    // prompt == max_ctx; the engine's capacity request adds the next decode page
    CHECK(w.kv->acquire_reserved(1, (uint32_t)kMaxCtx + kChunk, a) == PagedKvStatus::Ok);
}

TEST_CASE(KvflashSlotKvFixture, evicted_history_stays_owned_so_reserve_decode_needs_no_extra_blocks) {
    World w;
    SeqSlotManager slots(*w.kv, (int)kMaxCtx);
    w.kv->set_next_total_tokens(2048 + 1984);
    const auto a = slots.admit(1, std::vector<int32_t>(2048, 7), SamplerCfg{});
    REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    PagedKvSequenceHandle h = slots.slot(a.slot).handle;
    uint32_t owned = 0;
    REQUIRE(w.kv->owned_block_count(h, owned) == PagedKvStatus::Ok);
    CHECK(owned == 63);                                           // 4032 tokens of reservation, not the resident count
    CHECK(w.s[a.slot]->resident_blocks() == 0);
    CHECK(w.kv->free_block_count() == w.kv->physical_block_count() - 63);
}

TEST_CASE(KvflashSlotKvFixture, a_bound_that_fills_max_ctx_fits_an_exact_host_budget_and_does_not_count_the_decode_page_twice) {
    World w(/*budget=*/64 * kChunkBytes);                         // exactly max_ctx worth of backing
    SeqSlotManager slots(*w.kv, (int)kMaxCtx);
    w.kv->set_next_total_tokens(300 + 3796);
    const auto a = slots.admit(1, std::vector<int32_t>(300, 7), SamplerCfg{});
    REQUIRE(a.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(w.ledger.used_bytes() == 64 * kChunkBytes);
    CHECK(w.kv->free_block_count() <= w.kv->physical_block_count());
}
