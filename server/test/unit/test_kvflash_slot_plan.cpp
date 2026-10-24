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
    in.host_budget_bytes -= 1;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
    in = base_admit();
    in.host_used_bytes = 1;                                          // one byte already taken
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
}

TEST_CASE(KvflashSlotPlanFixture, admission_counts_ladder_and_global_transient_reserve) {
    KvFlashAdmitInput in = base_admit();
    in.host_budget_bytes = 16 * kChunkBytes + 100;
    in.transient_reserve_bytes = 100;                                // fits exactly
    CHECK(kvflash_admit(in) == KvFlashAdmit::Ok);
    in.transient_reserve_bytes = 101;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
    in.transient_reserve_bytes = 0;
    in.ladder_bytes = 101;
    CHECK(kvflash_admit(in) == KvFlashAdmit::BusyHost);
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
    CHECK(ops.back().kind == KvFlashOpKind::Fence);                  // the graph runs after the fence
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
    st[0].slot = 0; st[0].alloc_tokens = 1;                          // decode row, no paging
    st[1].slot = 1; st[1].alloc_tokens = 1;
    const auto ops = kvflash_plan_step(st);
    CHECK(count_kind(ops, KvFlashOpKind::Fence) == 0);
    CHECK(count_kind(ops, KvFlashOpKind::AllocSpan) == 2);
    CHECK(count_kind(ops, KvFlashOpKind::TableDelta) == 2);
    CHECK(kvflash_plan_step({}).empty());
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
    std::vector<int> deferred;
    const auto ops = kvflash_plan_step(st, &deferred);
    CHECK(count_kind(ops, KvFlashOpKind::Score) == 1);
    CHECK(ops.back().kind == KvFlashOpKind::Score);
    CHECK(ops.back().slot == 1);                                     // lowest slot first
    CHECK(first_index(ops, KvFlashOpKind::Score) > last_index(ops, KvFlashOpKind::Fence));
    CHECK((deferred == std::vector<int>{2}));
}
