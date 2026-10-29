// Pure unit tests for the Track C slot-pager arithmetic (kvflash_slot_pager.h):
// slice geometry, host-budget admission (C0 section 4, R1, R5) and the step
// plan (C0 R3: out before in, one fence per step). No ggml, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_pager.h"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotPlanFixture {};

constexpr int64_t kMiB = 1024 * 1024;
constexpr int64_t kChunkBytes = 17 * kMiB / 8;   // 2.125 MiB, q8_0 qwen35 (C0 section 0)

KvFlashAdmitInput base_admit() {
    KvFlashAdmitInput in;
    in.slot_free = true;
    in.quota_chunks = 352;
    in.headroom_blocks = 8;
    in.prompt_tokens = 1000;
    in.max_new_tokens = 24;               // 1024 tokens = 16 chunks
    in.max_ctx = 131072;
    in.chunk_tokens = 64;
    in.chunk_bytes = kChunkBytes;
    in.ladder_bytes = 0;
    in.transient_reserve_bytes = 0;
    in.host_used_bytes = 0;
    in.host_budget_bytes = 16 * kChunkBytes;
    return in;
}

int count_kind(const std::vector<KvFlashOp> & ops, KvFlashOpKind k) {
    return (int)std::count_if(ops.begin(), ops.end(), [&](const KvFlashOp & o) { return o.kind == k; });
}
int last_index(const std::vector<KvFlashOp> & ops, KvFlashOpKind k) {
    for (int i = (int)ops.size() - 1; i >= 0; i--) if (ops[(size_t)i].kind == k) return i;
    return -1;
}
int first_index(const std::vector<KvFlashOp> & ops, KvFlashOpKind k) {
    for (int i = 0; i < (int)ops.size(); i++) if (ops[(size_t)i].kind == k) return i;
    return -1;
}
}  // namespace

TEST_CASE(KvflashSlotPlanFixture, arena_sizing_derives_the_pool_and_refuses_a_different_kv_pool_tokens) {
    const KvFlashArenaSizing a = kvflash_arena_sizing(4, 352, 0);
    CHECK(a.error.empty());
    CHECK(a.pool_tokens == 4 * 352 * 64);                          // N x Q x 64
    CHECK(a.rows == kvflash_arena_rows(4, 352));                   // ctx_alloc: the pool plus the scratch block
    const KvFlashArenaSizing same = kvflash_arena_sizing(4, 352, 4 * 352 * 64);
    CHECK(same.error.empty());
    CHECK(same.rows == a.rows);
    const KvFlashArenaSizing diff = kvflash_arena_sizing(4, 352, 4096);
    CHECK(diff.rows == 0);
    CHECK(diff.error.find("--kv-pool-tokens 4096") != std::string::npos);
    CHECK(diff.error.find("90112") != std::string::npos);          // names the derived value
    CHECK(!kvflash_arena_sizing(0, 352, 0).error.empty());
    CHECK(!kvflash_arena_sizing(4, 0, 0).error.empty());
    CHECK(!kvflash_arena_sizing(8, 1 << 24, 0).error.empty());    // rows do not fit the int ctx_alloc
}

TEST_CASE(KvflashSlotPlanFixture, slices_tile_the_arena_exactly) {
    const int N = 4, Q = 352;                       // C0 section 1: Q = 22,528 tokens
    REQUIRE(kvflash_arena_rows(N, Q) == (int64_t)N * Q * 64 + 64);   // N x Q x 64 + scratch block
    int64_t next = 0;
    for (int s = 0; s < N; s++) {
        const KvFlashSlice sl = kvflash_slice(s, Q);
        CHECK(sl.row_lo == next);                   // no gap, no overlap
        CHECK(sl.row_hi - sl.row_lo == 22528);
        CHECK(sl.block_hi - sl.block_lo == Q);
        CHECK(sl.row_lo == (int64_t)sl.block_lo * 64);
        next = sl.row_hi;
    }
    CHECK(next == (int64_t)N * Q * 64);
    CHECK(kvflash_scratch_row(N, Q) == next);       // the scratch block starts where the slices end
}

TEST_CASE(KvflashSlotPlanFixture, every_row_belongs_to_exactly_one_slot_or_scratch) {
    const int N = 3, Q = 8;
    for (int64_t r = 0; r < kvflash_arena_rows(N, Q); r++) {
        int owners = 0, owner = -1;
        for (int s = 0; s < N; s++) {
            const KvFlashSlice sl = kvflash_slice(s, Q);
            if (r >= sl.row_lo && r < sl.row_hi) { owners++; owner = s; }
        }
        CHECK(owners <= 1);
        CHECK(kvflash_slot_of_row(r, N, Q) == (owners == 1 ? owner : -1));
    }
    CHECK(kvflash_slot_of_row(-1, N, Q) == -1);
    CHECK(kvflash_slot_of_row(kvflash_arena_rows(N, Q), N, Q) == -1);
}

TEST_CASE(KvflashSlotPlanFixture, quota_floor_is_the_pager_minimum) {
    KvFlashConfig cfg;                              // sink 1, tail 4: 7 chunks
    CHECK(kvflash_quota_ok(7, cfg));
    CHECK(!kvflash_quota_ok(6, cfg));
    CHECK(!kvflash_quota_ok(0, cfg));
    CHECK(kvflash_quota_ok(352, cfg));
    CHECK(kvflash_quota_chunks(22528, 64) == 352);
    CHECK(kvflash_quota_chunks(22529, 64) == 0);    // not a whole number of chunks
}

TEST_CASE(KvflashSlotPlanFixture, admission_ok_when_everything_fits) {
    CHECK(kvflash_admit(base_admit()) == KvFlashAdmit::Ok);
}

TEST_CASE(KvflashSlotPlanFixture, admission_host_need_counts_prompt_plus_new_tokens_in_whole_chunks) {
    KvFlashAdmitInput in = base_admit();
    CHECK(kvflash_host_need_bytes(in) == 16 * kChunkBytes);          // 1024 tokens
    in.max_new_tokens = 25;                                          // 1025 tokens -> 17 chunks
    CHECK(kvflash_host_need_bytes(in) == 17 * kChunkBytes);
    in.ladder_bytes = 3 * 150 * kMiB;
    CHECK(kvflash_host_need_bytes(in) == 17 * kChunkBytes + 3 * 150 * kMiB);
    // 128K prompt at q8_0: 2048 chunks = 4.25 GiB (R5 serialize figure).
    in = base_admit();
    in.prompt_tokens = 131072; in.max_new_tokens = 0;
    CHECK(kvflash_host_need_bytes(in) == 2048 * kChunkBytes);
    CHECK(kvflash_host_need_bytes(in) == 4352 * kMiB);
}

TEST_CASE(KvflashSlotPlanFixture, admission_host_budget_boundary_is_exact) {
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes;                         // exactly the need
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    in.host_budget_bytes -= 1;                                       // the request alone cannot ever fit
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in = base_admit();
    in.host_used_bytes = 1;                                          // one byte already taken by others: transient
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
}

TEST_CASE(KvflashSlotPlanFixture, admission_counts_ladder_and_global_transient_reserve) {
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes + 100;
    in.transient_reserve_bytes = 100;                                // fits exactly
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    in.transient_reserve_bytes = 101;
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);          // permanent: nothing else is reserved
    in.host_used_bytes = 2; in.transient_reserve_bytes = 100; in.host_budget_bytes += 1;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);              // fits alone, waits for others
    in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes + 100;
    in.ladder_bytes = 101;
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
}

TEST_CASE(KvflashSlotPlanFixture, admission_refuses_busy_slot_and_overlong_prompt_with_distinct_verdicts) {
    KvFlashAdmitInput in = base_admit();
    in.slot_free = false;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusySlot);
    in = base_admit();
    in.prompt_tokens = in.max_ctx + 1;                               // R1: prompt <= max_ctx
    CHECK(kvflash_admit(in) == KvFlashAdmit::PromptTooLong);
    in.prompt_tokens = in.max_ctx;
    in.max_new_tokens = 0;
    in.host_budget_bytes = 1LL << 40;
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    // A permanent refusal outranks a transient one.
    in.prompt_tokens = in.max_ctx + 1;
    in.slot_free = false;
    CHECK(kvflash_admit(in) == KvFlashAdmit::PromptTooLong);
}

TEST_CASE(KvflashSlotPlanFixture, step_plan_puts_every_out_before_every_in_with_one_fence) {
    std::vector<KvFlashStepSlot> st(2);
    st[0].slot = 0; st[0].page_out = {5, 6}; st[0].page_in = {1}; st[0].alloc_tokens = 512;
    st[1].slot = 1; st[1].page_out = {9};    st[1].page_in = {2, 3}; st[1].alloc_tokens = 1;
    const auto ops = kvflash_plan_step(st);
    CHECK(count_kind(ops, KvFlashOpKind::PageOut) == 3);
    CHECK(count_kind(ops, KvFlashOpKind::PageIn) == 3);
    CHECK(count_kind(ops, KvFlashOpKind::Fence) == 1);
    CHECK(last_index(ops, KvFlashOpKind::PageOut) < first_index(ops, KvFlashOpKind::PageIn));
    CHECK(last_index(ops, KvFlashOpKind::PageIn) < first_index(ops, KvFlashOpKind::AllocSpan));
    CHECK(last_index(ops, KvFlashOpKind::AllocSpan) < first_index(ops, KvFlashOpKind::TableDelta));
    CHECK(last_index(ops, KvFlashOpKind::TableDelta) < first_index(ops, KvFlashOpKind::Fence));
    CHECK(ops[ops.size() - 2].kind == KvFlashOpKind::Fence);         // the graph runs after the fence
    CHECK(ops.back().kind == KvFlashOpKind::EndStep);
    // No free block: the page-ins reuse blocks the page-outs free, so one
    // completion phase sits between the last out and the first in.
    REQUIRE(count_kind(ops, KvFlashOpKind::CompleteOut) == 1);
    CHECK(last_index(ops, KvFlashOpKind::PageOut) < first_index(ops, KvFlashOpKind::CompleteOut));
    CHECK(first_index(ops, KvFlashOpKind::CompleteOut) < first_index(ops, KvFlashOpKind::PageIn));
}

TEST_CASE(KvflashSlotPlanFixture, step_plan_pins_step_chunks_before_any_alloc) {
    std::vector<KvFlashStepSlot> st(2);
    st[0].slot = 0; st[0].pin_chunks = {7, 8}; st[0].alloc_tokens = 128;
    st[1].slot = 1; st[1].pin_chunks = {3};    st[1].alloc_tokens = 1;
    const auto ops = kvflash_plan_step(st);
    CHECK(count_kind(ops, KvFlashOpKind::Pin) == 3);
    CHECK(last_index(ops, KvFlashOpKind::Pin) < first_index(ops, KvFlashOpKind::AllocSpan));
    // R3(b): pins precede the page work too, or an in-flight chunk could be a victim.
    std::vector<KvFlashStepSlot> st2(1);
    st2[0].slot = 0; st2[0].pin_chunks = {4}; st2[0].page_out = {1}; st2[0].alloc_tokens = 1;
    const auto ops2 = kvflash_plan_step(st2);
    CHECK(last_index(ops2, KvFlashOpKind::Pin) < first_index(ops2, KvFlashOpKind::PageOut));
}

TEST_CASE(KvflashSlotPlanFixture, step_plan_without_page_work_has_no_fence) {
    std::vector<KvFlashStepSlot> st(2);
    st[0].slot = 0; st[0].alloc_tokens = 1; st[0].free_blocks = 1;   // decode row into a free block: no paging
    st[1].slot = 1; st[1].alloc_tokens = 1; st[1].free_blocks = 1;
    const auto ops = kvflash_plan_step(st);
    CHECK(count_kind(ops, KvFlashOpKind::Fence) == 0);
    CHECK(count_kind(ops, KvFlashOpKind::AllocSpan) == 2);
    CHECK(count_kind(ops, KvFlashOpKind::TableDelta) == 2);
    const auto none = kvflash_plan_step({});
    REQUIRE(none.size() == 1);
    CHECK(none[0].kind == KvFlashOpKind::EndStep);
}

TEST_CASE(KvflashSlotPlanFixture, step_plan_orders_by_slot_and_skips_idle_slots) {
    std::vector<KvFlashStepSlot> st(3);
    st[0].slot = 2; st[0].alloc_tokens = 1;
    st[1].slot = 0; st[1].alloc_tokens = 1;
    st[2].slot = 1;                                                  // nothing to do
    const auto ops = kvflash_plan_step(st);
    std::vector<int> alloc_slots;
    for (const auto & o : ops) if (o.kind == KvFlashOpKind::AllocSpan) alloc_slots.push_back(o.slot);
    CHECK((alloc_slots == std::vector<int>{0, 2}));
    for (const auto & o : ops) CHECK(o.slot != 1);
}

TEST_CASE(KvflashSlotPlanFixture, step_plan_runs_at_most_one_scorer_call_after_the_fence) {
    std::vector<KvFlashStepSlot> st(3);
    st[0].slot = 0; st[0].page_in = {1}; st[0].alloc_tokens = 1;
    st[1].slot = 1; st[1].alloc_tokens = 1; st[1].wants_score = true;
    st[2].slot = 2; st[2].alloc_tokens = 1; st[2].wants_score = true;
    KvFlashStepDeferred deferred;
    const auto ops = kvflash_plan_step(st, &deferred);
    CHECK(count_kind(ops, KvFlashOpKind::Score) == 1);
    REQUIRE(count_kind(ops, KvFlashOpKind::Fence) == 1);             // the page-in is page work: scoring needs the fence
    CHECK(ops[ops.size() - 2].kind == KvFlashOpKind::Score);
    CHECK(ops[ops.size() - 2].slot == 1);                            // lowest slot first
    CHECK(first_index(ops, KvFlashOpKind::Score) > last_index(ops, KvFlashOpKind::Fence));
    CHECK((deferred.score == std::vector<int>{2}));
}

TEST_CASE(KvflashSlotPlanFixture, end_step_is_always_present_and_always_last) {
    std::vector<std::vector<KvFlashStepSlot>> cases;
    cases.push_back({});                                             // empty step
    { KvFlashStepSlot a; a.slot = 0; cases.push_back({a}); }         // idle slot
    { KvFlashStepSlot a; a.slot = 0; a.alloc_tokens = 1; cases.push_back({a}); }
    { KvFlashStepSlot a; a.slot = 1; a.page_out = {1}; a.page_in = {2}; a.pin_chunks = {3}; a.alloc_tokens = 1; a.wants_score = true;
      KvFlashStepSlot b; b.slot = 0; b.wants_score = true; cases.push_back({a, b}); }
    for (const auto & c : cases) {
        const auto ops = kvflash_plan_step(c);
        CHECK(count_kind(ops, KvFlashOpKind::EndStep) == 1);
        REQUIRE(!ops.empty());
        CHECK(ops.back().kind == KvFlashOpKind::EndStep);
    }
}

TEST_CASE(KvflashSlotPlanFixture, headroom_shrinks_the_usable_quota_and_keeps_the_pager_floor) {
    CHECK(kvflash_usable_chunks(352, 8) == 344);
    KvFlashConfig cfg;                                               // floor 7 chunks
    CHECK(kvflash_headroom_ok(352, 8, cfg));
    CHECK(kvflash_headroom_ok(15, 8, cfg));                          // usable 7: exactly the floor
    CHECK(!kvflash_headroom_ok(15, 9, cfg));                         // usable 6
    CHECK(!kvflash_headroom_ok(352, -1, cfg));
    CHECK(kvflash_headroom_ok(7, 0, cfg));
}

TEST_CASE(KvflashSlotPlanFixture, page_outs_are_capped_at_the_headroom_and_the_rest_wait) {
    std::vector<KvFlashStepSlot> st(2);
    st[0].slot = 0; st[0].headroom_blocks = 3; st[0].page_out = {10, 11, 12};        // at the cap
    st[1].slot = 1; st[1].headroom_blocks = 3; st[1].page_out = {20, 21, 22, 23, 24}; // two over
    st[1].page_in = {5};
    KvFlashStepDeferred d;
    const auto ops = kvflash_plan_step(st, &d);
    std::vector<int> out0, out1;
    for (const auto & o : ops) if (o.kind == KvFlashOpKind::PageOut) (o.slot == 0 ? out0 : out1).push_back(o.arg);
    CHECK((out0 == std::vector<int>{10, 11, 12}));                   // at the cap: all go
    CHECK((out1 == std::vector<int>{20, 21, 22}));                   // over the cap: first three go
    REQUIRE(d.page_out.size() == 2);
    CHECK((d.page_out[0].slot == 1 && d.page_out[0].arg == 23));
    CHECK((d.page_out[1].slot == 1 && d.page_out[1].arg == 24));     // deferred in order, to the next step
    // The deferred outs fit the next step's headroom.
    std::vector<KvFlashStepSlot> next(1);
    next[0].slot = 1; next[0].headroom_blocks = 3;
    for (const auto & o : d.page_out) next[0].page_out.push_back(o.arg);
    KvFlashStepDeferred d2;
    const auto ops2 = kvflash_plan_step(next, &d2);
    CHECK(count_kind(ops2, KvFlashOpKind::PageOut) == 2);
    CHECK(d2.page_out.empty());
}

TEST_CASE(KvflashSlotPlanFixture, zero_headroom_defers_every_page_out_and_default_is_uncapped) {
    std::vector<KvFlashStepSlot> st(1);
    st[0].slot = 0; st[0].page_out = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CHECK(count_kind(kvflash_plan_step(st), KvFlashOpKind::PageOut) == 10);
    st[0].headroom_blocks = 0;
    KvFlashStepDeferred d;
    const auto ops = kvflash_plan_step(st, &d);
    CHECK(count_kind(ops, KvFlashOpKind::PageOut) == 0);
    CHECK(count_kind(ops, KvFlashOpKind::Fence) == 0);               // no page work left to fence
    CHECK(d.page_out.size() == 10);
}

TEST_CASE(KvflashSlotPlanFixture, admission_with_headroom_refuses_a_quota_below_the_floor) {
    KvFlashAdmitInput in = base_admit();
    in.quota_chunks = 15; in.headroom_blocks = 8;                    // usable 7: ok
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    in.headroom_blocks = 9;                                          // usable 6
    CHECK(kvflash_admit(in) == KvFlashAdmit::BadQuota);
    in.slot_free = false; in.prompt_tokens = in.max_ctx + 1;         // a bad quota outranks everything
    CHECK(kvflash_admit(in) == KvFlashAdmit::BadQuota);
    // The host bound is the same with and without headroom.
    in = base_admit();
    const int64_t need = kvflash_host_need_bytes(in);
    in.headroom_blocks = 0;
    CHECK(kvflash_host_need_bytes(in) == need);
}

// ── Review follow-ups: fence decision, completion phase, pin conflict ────────

TEST_CASE(KvflashSlotPlanFixture, allocation_that_must_page_gets_a_fence_with_empty_page_lists) {
    // A full slice appends a chunk: alloc_span pages a victim out by itself.
    KvFlashStepSlot full; full.slot = 0; full.alloc_tokens = 64; full.free_blocks = 0; full.alloc_new_chunks = 1;
    CHECK(count_kind(kvflash_plan_step({full}), KvFlashOpKind::Fence) == 1);
    // The same step with a free block does not page.
    KvFlashStepSlot room = full; room.free_blocks = 1;
    CHECK(count_kind(kvflash_plan_step({room}), KvFlashOpKind::Fence) == 0);
    // A prefill ubatch of 512 rows spans at most 9 chunks; 8 free blocks are not enough, 9 are.
    KvFlashStepSlot big; big.slot = 0; big.alloc_tokens = 512; big.free_blocks = 8;
    CHECK(count_kind(kvflash_plan_step({big}), KvFlashOpKind::Fence) == 1);
    big.free_blocks = 9;
    CHECK(count_kind(kvflash_plan_step({big}), KvFlashOpKind::Fence) == 0);
    // A caller that says nothing gets the safe side: a fence.
    KvFlashStepSlot silent; silent.slot = 0; silent.alloc_tokens = 1;
    CHECK(count_kind(kvflash_plan_step({silent}), KvFlashOpKind::Fence) == 1);
    // A page-in takes a block too: it can page out inside the call.
    KvFlashStepSlot in; in.slot = 0; in.page_in = {3}; in.free_blocks = 0;
    CHECK(count_kind(kvflash_plan_step({in}), KvFlashOpKind::Fence) == 1);
    // Two slots: one pages inside alloc, the other does not; still ONE fence.
    KvFlashStepSlot other = room; other.slot = 1;
    CHECK(count_kind(kvflash_plan_step({full, other}), KvFlashOpKind::Fence) == 1);
}

TEST_CASE(KvflashSlotPlanFixture, out_then_in_in_deferred_mode_gets_one_completion_phase_between) {
    // Full slice: page a victim out, page an old chunk back in, append a row.
    KvFlashStepSlot s; s.slot = 0; s.page_out = {5}; s.page_in = {2}; s.alloc_tokens = 1;
    s.free_blocks = 0; s.alloc_new_chunks = 0;
    const auto ops = kvflash_plan_step({s});
    REQUIRE(count_kind(ops, KvFlashOpKind::CompleteOut) == 1);
    CHECK(count_kind(ops, KvFlashOpKind::Fence) == 1);               // CompleteOut is not a second fence
    CHECK(last_index(ops, KvFlashOpKind::PageOut) < first_index(ops, KvFlashOpKind::CompleteOut));
    CHECK(first_index(ops, KvFlashOpKind::CompleteOut) < first_index(ops, KvFlashOpKind::PageIn));
    CHECK(last_index(ops, KvFlashOpKind::TableDelta) < first_index(ops, KvFlashOpKind::Fence));
    // Enough free blocks: nothing reuses a block the outs free, no completion phase.
    s.free_blocks = 1;
    CHECK(count_kind(kvflash_plan_step({s}), KvFlashOpKind::CompleteOut) == 0);
    // No page-out this step: nothing to complete.
    KvFlashStepSlot only_in; only_in.slot = 0; only_in.page_in = {2}; only_in.free_blocks = 0;
    CHECK(count_kind(kvflash_plan_step({only_in}), KvFlashOpKind::CompleteOut) == 0);
    // A cross-slot case: slot 1 pages out, slot 0 needs a block: completion still precedes every page-in.
    KvFlashStepSlot a; a.slot = 0; a.page_in = {1}; a.free_blocks = 0;
    KvFlashStepSlot b; b.slot = 1; b.page_out = {9}; b.free_blocks = 5;
    const auto ops2 = kvflash_plan_step({a, b});
    CHECK(count_kind(ops2, KvFlashOpKind::CompleteOut) == 0);        // slices are private: slot 1's out frees nothing for slot 0
}

TEST_CASE(KvflashSlotPlanFixture, a_page_out_of_a_chunk_pinned_this_step_is_deferred_not_emitted) {
    KvFlashStepSlot s; s.slot = 0; s.pin_chunks = {2}; s.page_out = {2, 3}; s.free_blocks = 4;
    KvFlashStepDeferred d;
    const auto ops = kvflash_plan_step({s}, &d);
    std::vector<int> outs;
    for (const auto & o : ops) if (o.kind == KvFlashOpKind::PageOut) outs.push_back(o.arg);
    CHECK((outs == std::vector<int>{3}));
    REQUIRE(d.page_out.size() == 1);
    CHECK((d.page_out[0].slot == 0 && d.page_out[0].arg == 2));
    // Without a deferral sink the conflicting out is still never emitted.
    CHECK(count_kind(kvflash_plan_step({s}), KvFlashOpKind::PageOut) == 1);
    // A pin on another slot's chunk 2 does not block this slot.
    KvFlashStepSlot t; t.slot = 1; t.page_out = {2}; t.free_blocks = 4;
    CHECK(count_kind(kvflash_plan_step({s, t}), KvFlashOpKind::PageOut) == 2);
}

// ── C1.3-01: the host reservation ledger and the permanent refusal ─────────

TEST_CASE(KvflashSlotPlanFixture, a_request_that_can_never_fit_is_refused_not_queued) {
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes - 1;                     // alone it exceeds the whole budget
    in.host_used_bytes = 0;
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in.host_used_bytes = 5 * kChunkBytes;                            // same with others present: still permanent
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in.host_budget_bytes = 100 * kChunkBytes;                        // fits alone, others hold it: transient
    in.host_used_bytes = 90 * kChunkBytes;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
    // A permanent refusal outranks a busy slot; a bad quota and an overlong prompt outrank it.
    in.host_budget_bytes = 1; in.host_used_bytes = 0; in.slot_free = false;
    CHECK(kvflash_admit(in) == KvFlashAdmit::HostTooSmall);
    in.prompt_tokens = in.max_ctx + 1;
    CHECK(kvflash_admit(in) == KvFlashAdmit::PromptTooLong);
}

TEST_CASE(KvflashSlotPlanFixture, the_ledger_reserves_at_admit_and_releases_on_every_exit) {
    KvFlashHostLedger led(40 * kChunkBytes, 0);
    KvFlashAdmitInput in = base_admit();                             // needs 16 chunks
    in.host_budget_bytes = 0;                                        // the ledger owns the budget and used bytes
    CHECK(led.admit(1, in) == KvFlashAdmit::Ok);
    CHECK(led.used_bytes() == 16 * kChunkBytes);
    CHECK(led.admit(2, in) == KvFlashAdmit::Ok);
    CHECK(led.used_bytes() == 32 * kChunkBytes);
    CHECK(led.admit(3, in) == KvFlashAdmit::BusyHost);               // 48 > 40, but it fits alone
    CHECK(led.used_bytes() == 32 * kChunkBytes);                     // a refusal reserves nothing
    CHECK(led.reservation_count() == 2);
    led.release(1);                                                  // finish
    CHECK(led.admit(3, in) == KvFlashAdmit::Ok);
    led.release(2);                                                  // cancel
    led.release(3);                                                  // error / abort_step
    CHECK(led.used_bytes() == 0);
    CHECK(led.reservation_count() == 0);
    led.release(3);                                                  // idempotent
    led.release(99);                                                 // unknown id
    CHECK(led.used_bytes() == 0);
}

TEST_CASE(KvflashSlotPlanFixture, the_ledger_never_double_counts_a_request_id_and_counts_the_transient_once) {
    KvFlashHostLedger led(40 * kChunkBytes, 8 * kChunkBytes);        // 8 chunks of global transient
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 0; in.transient_reserve_bytes = 0;
    CHECK(led.admit(1, in) == KvFlashAdmit::Ok);
    CHECK(led.admit(1, in) == KvFlashAdmit::BusySlot);               // same id twice: refused, not charged again
    CHECK(led.used_bytes() == 16 * kChunkBytes);
    CHECK(led.admit(2, in) == KvFlashAdmit::Ok);                     // 16 + 16 + 8 transient = 40: exact fit
    CHECK(led.admit(3, in) == KvFlashAdmit::BusyHost);               // the transient was charged once, not per slot
    KvFlashAdmitInput small = in; small.prompt_tokens = 500; small.max_new_tokens = 12;   // 8 chunks
    CHECK(led.admit(5, small) == KvFlashAdmit::BusyHost);            // 32 + 8 + 8 > 40: only the transient makes the difference
    in.prompt_tokens = 100000;
    CHECK(led.admit(4, in) == KvFlashAdmit::HostTooSmall);           // bigger than the whole budget
    CHECK(led.reservation_count() == 2);
}

TEST_CASE(KvflashSlotPlanFixture, auto_budget_is_split_across_slots_and_the_reserve_is_held_once) {
    KvFlashAutoBudget b; b.free_bytes = 10000; b.reserve_bytes = 2000; b.bytes_per_token = 10;
    CHECK(kvflash_budget_per_slot(b, 1).free_bytes == 10000);
    CHECK(kvflash_budget_per_slot(b, 4).free_bytes == 2000 + 8000 / 4);
    KvFlashAutoBudget low; low.free_bytes = 1000; low.reserve_bytes = 2000;
    CHECK(kvflash_budget_per_slot(low, 4).free_bytes == 1000);     // nothing above the reserve: unchanged
}

// ── Eviction stage 0 (F2): the park staging reservation ────────────────
// Qwen3.8-27B q8_0 geometry (BYTES_PER_SESSION.md): slab 156,893,184 B, chunk 2,228,224 B.
namespace {
constexpr int64_t kSlab = 156893184;
constexpr int64_t kMaxCtx = 135168;
constexpr int64_t kNSlabTensors = 96, kHeader = 4096;   // small terms; the test tolerates them

// A short sticky session (2K prompt + 256 new) with a bundle dir, the reservation as init_kvflash_slots builds it.
KvFlashAdmitInput sticky_bundle_admit(bool staged_by_size) {
    KvFlashAdmitInput in = base_admit();
    in.prompt_tokens = 2048; in.max_new_tokens = 256; in.max_ctx = kMaxCtx;
    const KvFlashParkStaging st = kvflash_park_staging(kSlab, kNSlabTensors, kHeader, 64, kChunkBytes);
    in.ladder_bytes = kvflash_ladder_bytes(kSlab, false, true, false, st);
    if (staged_by_size) { in.ladder_bytes = kvflash_ladder_bytes(kSlab, false, true, true, st); in.stage_chunk_bytes = st.per_chunk_bytes; }
    else                in.ladder_bytes += kvflash_park_staging_max_bytes(st, kMaxCtx, 64);
    return in;
}
}

TEST_CASE(KvflashSlotPlanFixture, f2_staging_at_max_ctx_is_about_5_GiB_per_session) {
    const KvFlashParkStaging st = kvflash_park_staging(kSlab, kNSlabTensors, kHeader, 64, kChunkBytes);
    const int64_t at_max = st.fixed_bytes + 2112 * st.per_chunk_bytes;   // nc_max = 135168 / 64
    CHECK(kvflash_park_staging_max_bytes(st, kMaxCtx, 64) == at_max);
    CHECK(at_max > 4900 * kMiB);                      // 4.97 GiB computed by the design note
    CHECK(at_max < 5500 * kMiB);                      // +2 slabs (the third rung, one live and one staged pack): 5.1 GiB
    // measured parks are 448-478 MiB (C1_4_PROGRESS.md:89): the reservation is >10x the truth
    CHECK(at_max > 10 * 478 * kMiB);
}

TEST_CASE(KvflashSlotPlanFixture, f2_today_the_default_budget_holds_two_short_sessions) {
    const KvFlashAdmitInput in = sticky_bundle_admit(false);
    CHECK(kvflash_host_need_bytes(in) > 5200 * kMiB);   // ~5.3 GiB for a 2K session
    CHECK(kvflash_host_need_bytes(in) < 5800 * kMiB);   // 5.6 GiB with the third rung
}

TEST_CASE(KvflashSlotPlanFixture, f2_staging_sized_by_the_session_fits_every_slot) {
    const KvFlashAdmitInput in = sticky_bundle_admit(true);
    // 2304 tokens = 36 chunks: backing + both rungs + fixed staging + 36 chunks of staging, about 0.6 GiB x 2 + slabs
    CHECK(kvflash_host_need_bytes(in) < 1500 * kMiB);   // +2 slabs (the third rung)
    // never below what the park really holds (a 478 MiB bundle)
    CHECK(kvflash_host_need_bytes(in) > 478 * kMiB);
}

TEST_CASE(KvflashSlotPlanFixture, f2_staging_by_size_grows_with_the_session) {
    KvFlashAdmitInput in = sticky_bundle_admit(true);
    const int64_t small = kvflash_host_need_bytes(in);
    in.prompt_tokens = 32768;
    CHECK(kvflash_host_need_bytes(in) > small);
    // a full max_ctx session still reserves its whole backing and staging: the bound is the session, not max_ctx for everyone
    in.prompt_tokens = kMaxCtx; in.max_new_tokens = 0;
    CHECK(kvflash_host_need_bytes(in) > kvflash_host_need_bytes(sticky_bundle_admit(false)));
}

TEST_CASE(KvflashSlotPlanFixture, f2_ledger_reports_the_reservation_and_the_refused_need_for_the_stage0_log) {
    KvFlashHostLedger led(4 * kChunkBytes, 0);
    KvFlashAdmitInput in = base_admit();
    in.prompt_tokens = 128; in.max_new_tokens = 0;                // 2 chunks
    CHECK(led.admit(1, in) == KvFlashAdmit::Ok);
    CHECK(led.reserved_of(1) == 2 * kChunkBytes);
    CHECK(led.last_refused_need_bytes() == 0);
    in.prompt_tokens = 192;                                       // 3 chunks: 2 + 3 > 4
    CHECK(led.admit(2, in) == KvFlashAdmit::BusyHost);
    CHECK(led.last_refused_need_bytes() == 3 * kChunkBytes);
    CHECK(led.budget_bytes() == 4 * kChunkBytes);
    led.release(1);
    CHECK(led.admit(2, in) == KvFlashAdmit::Ok);
    CHECK(led.last_refused_need_bytes() == 0);
}

// ── review P1 #1: a resume keeps the reservation while the last park is still staged ──────────────────────
TEST_CASE(KvflashSlotPlanFixture, a_reserve_floor_holds_the_reservation_at_what_the_slot_held) {
    KvFlashAdmitInput in = base_admit();
    in.prompt_tokens = 128; in.max_new_tokens = 0;                    // 2 chunks
    CHECK(kvflash_host_need_bytes(in) == 2 * kChunkBytes);
    in.reserve_floor_bytes = 40 * kChunkBytes;                        // the 40-chunk park still being written
    CHECK(kvflash_host_need_bytes(in) == 40 * kChunkBytes);
    in.prompt_tokens = 64 * 60;                                       // a larger turn is charged its own size
    CHECK(kvflash_host_need_bytes(in) == 60 * kChunkBytes);
}

// ── v5: the third rung ─────────────────────────────────────────────────
TEST_CASE(KvflashSlotPlanFixture, v5_the_third_rung_adds_exactly_two_slabs_to_the_ladder_charge_and_the_host_need) {
    const KvFlashParkStaging st = kvflash_park_staging(kSlab, kNSlabTensors, kHeader, 64, kChunkBytes);
    // the two-rung charge: 2 live rungs, and a staging of 4 slabs (end snapshot, end pack, two rung packs) + the small terms
    const int64_t small = 48 + 16 * kNSlabTensors + kHeader + kChunkBytes;
    const int64_t two_rungs = 2 * kSlab + 4 * kSlab + small;
    CHECK(st.fixed_bytes == 5 * kSlab + small);                                   // one more rung pack in the staging
    CHECK(kvflash_ladder_bytes(kSlab, false, true, true, st) - two_rungs == 2 * kSlab);   // one more live rung, one more staged pack
    KvFlashAdmitInput now = sticky_bundle_admit(true), before = now;
    before.ladder_bytes = two_rungs;
    CHECK(kvflash_host_need_bytes(now) - kvflash_host_need_bytes(before) == 2 * kSlab);   // the admission / host-ledger estimate
    CHECK(kvflash_ladder_bytes(kSlab, true, true, false, st) == 4 * kSlab);       // a recall turn's snapshot beside the three rungs
    CHECK(kvflash_ladder_bytes(kSlab, false, false, true, st) == st.fixed_bytes);  // not sticky: no rungs
    CHECK(kvflash_ladder_bytes(kSlab, false, true, false, st) == 3 * kSlab);       // no bundle dir: no staging
}
