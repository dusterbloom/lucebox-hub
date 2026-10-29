#include "CppUnitTestFramework.hpp"
#include <functional>
#include "../../src/common/kvflash_pager.h"
#include "../../src/qwen35/prefill_helpers.h"
#include "../../src/qwen35/qwen35_daemon.h"
#include "../../src/qwen35moe/qwen35moe_daemon.h"
#include <cstdlib>
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "kvflash_pager_test_host.h"

namespace {
struct KvflashSpecSerialFixture {};
}

using namespace luce::common;

// F1: DDTree commits every accepted token before it checks termination and
// never settles, so on EOS/cancel/budget end it finishes one position past AR.
// --ddtree with KVFlash fails closed. Launchers do add it (the container
// entrypoint, harness/clients/common.sh), so the check sits in luce_server
// startup and in both legacy daemons.
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

// F5: the seed's own chunk is exempt from the span cap (the first row always
// allocates it, as AR does), but AR allocates it only after it emitted the
// seed. A request that ends on that emission leaves the pager untouched in AR;
// a spec step that verifies first allocates (and evicts) before the emit loop,
// and the terminal repair restores recurrent state only.
namespace {
constexpr int kF5Pool = 512, kF5Chunk = 64;

struct F5Pager {
    ggml_context *        ctx = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    TestPager             pager;

    F5Pager() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < 2; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kF5Pool, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kF5Pool, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        KvFlashConfig c;
        c.pool_tokens = kF5Pool;
        c.chunk_tokens = kF5Chunk;
        c.sink_chunks = 1;
        c.tail_window_chunks = 4;
        pager.attach(c, k, v);
    }
    ~F5Pager() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    F5Pager(const F5Pager &) = delete;
    F5Pager & operator=(const F5Pager &) = delete;

    // Full pool: chunks 0..7 resident, the next position opens chunk 8.
    void fill() { pager.alloc_span(0, kF5Pool); }
    std::string state() const {
        return pager.debug_resident(16) + " cur=" + std::to_string(pager.cur_chunk());
    }
    bool alloc_row(int pos) { return pager.slot_for(pos) >= 0; }
};
}

TEST_CASE(KvflashSpecSerialFixture, terminal_seed_at_a_new_chunk_leaves_the_pager_as_ar_does) {
    const int p0 = kF5Pool;   // first row of chunk 8, pool full
    int remaining = 1;        // the seed is the last token the request may emit

    F5Pager ar;               // AR: emit, stop, never forward
    ar.fill();
    const std::string ar_before = ar.state();
    remaining--;
    if (remaining > 0) ar.alloc_row(p0);
    CHECK(ar.state() == ar_before);

    // The old spec order: verify (allocate) first, emit after.
    F5Pager old_order;
    old_order.fill();
    old_order.alloc_row(p0);
    CHECK(old_order.state() != ar_before);   // the hazard: allocation + eviction stay

    // The spec step as the loop runs it.
    F5Pager spec;
    spec.fill();
    remaining = 1;
    const bool emit_first = kvflash_seed_emits_before_verify(
        p0, kF5Chunk, [&](int c) { return spec.pager.is_resident(c); });
    if (emit_first) {
        remaining--;
        if (remaining <= 0) { /* ends here, as AR */ }
        else spec.alloc_row(p0);
    } else {
        spec.alloc_row(p0);   // verify
        remaining--;
    }
    CHECK(spec.state() == ar_before);
}

TEST_CASE(KvflashSpecSerialFixture, seed_emits_first_only_where_it_opens_an_unallocated_chunk) {
    const std::function<bool(int)> upto7 = [](int c) { return c <= 7; };
    CHECK(kvflash_seed_emits_before_verify(512, 64, upto7));     // opens chunk 8
    CHECK(!kvflash_seed_emits_before_verify(511, 64, upto7));    // last row of chunk 7
    CHECK(!kvflash_seed_emits_before_verify(100, 64, upto7));    // mid chunk
    CHECK(!kvflash_seed_emits_before_verify(512, 64, [](int) { return true; }));  // already allocated
    CHECK(!kvflash_seed_emits_before_verify(512, 0, upto7));     // no KVFlash
}

// F1: the legacy daemon (test_dflash --daemon --ddtree, launched by
// pflash/dflash_client.py with LUCE_KVFLASH inherited) reaches the backend
// without passing luce_server's startup check. It must refuse the same
// combination with exit 2 before it loads anything: the bogus target path
// would make init() return 1, so 2 proves the refusal ran first.
namespace {
struct ScopedKvflashEnv {
    std::string old; bool had;
    explicit ScopedKvflashEnv(const char * v) {
        const char * o = std::getenv("LUCE_KVFLASH");
        had = o != nullptr; if (had) old = o;
        if (v) setenv("LUCE_KVFLASH", v, 1); else unsetenv("LUCE_KVFLASH");
    }
    ~ScopedKvflashEnv() { if (had) setenv("LUCE_KVFLASH", old.c_str(), 1); else unsetenv("LUCE_KVFLASH"); }
};
Qwen35DaemonArgs daemon_args(bool ddtree) {
    Qwen35DaemonArgs a;
    a.target_path = "/nonexistent/no-such-target.gguf";
    a.ddtree_mode = ddtree;
    return a;
}
}

TEST_CASE(KvflashSpecSerialFixture, legacy_daemon_refuses_ddtree_with_kvflash_before_init) {
    ScopedKvflashEnv env("4096");
    CHECK(run_qwen35_daemon(daemon_args(true)) == 2);
    CHECK(run_qwen35moe_daemon(daemon_args(true)) == 2);
}

TEST_CASE(KvflashSpecSerialFixture, legacy_daemon_without_the_combination_is_not_refused) {
    {
        ScopedKvflashEnv env("4096");                    // kvflash alone: reaches init, which fails (1)
        CHECK(run_qwen35_daemon(daemon_args(false)) == 1);
    }
    {
        ScopedKvflashEnv env(nullptr);                   // ddtree alone
        CHECK(run_qwen35_daemon(daemon_args(true)) == 1);
    }
}

// P2: reselection scores the history AR has at that point, and AR applies the
// thinking budget before it records a token. A forced close that coincides
// with a reselect point replaces the seed, so the scorer must see close[0].
TEST_CASE(KvflashSpecSerialFixture, reselect_history_holds_the_budgeted_seed_like_ar) {
    BudgetHook hook;
    hook.close_token_ids = {9, 8};
    hook.marker_token_ids = {8};
    hook.hard_limit_remaining = 3;
    ThinkingBudget spec_budget(hook, 3);     // remaining 3: the next token is force-closed
    ThinkingBudget ar_budget(hook, 3);

    const int32_t seed = 5;
    const int32_t ar_recorded = ar_budget.apply(seed).token;   // AR: apply, then record, then reselect
    CHECK(ar_recorded == 9);
    CHECK(kvflash_reselect_seed_token(spec_budget, seed) == ar_recorded);
    // The real accounting stays at emission.
    CHECK(spec_budget.remaining() == 3);
    CHECK(!spec_budget.forced_close());
    CHECK(spec_budget.apply(seed).token == 9);
}

TEST_CASE(KvflashSpecSerialFixture, reselect_history_keeps_the_seed_when_the_budget_does_not_replace_it) {
    BudgetHook hook;
    hook.close_token_ids = {9, 8};
    hook.marker_token_ids = {8};
    hook.hard_limit_remaining = 3;
    ThinkingBudget b(hook, 100);
    CHECK(kvflash_reselect_seed_token(b, 5) == 5);
    CHECK(b.remaining() == 100);
    ThinkingBudget off;                      // no hook
    CHECK(kvflash_reselect_seed_token(off, 5) == 5);
    ThinkingBudget closing(hook, 100);       // the natural marker passes through unchanged
    CHECK(kvflash_reselect_seed_token(closing, 8) == 8);
}
