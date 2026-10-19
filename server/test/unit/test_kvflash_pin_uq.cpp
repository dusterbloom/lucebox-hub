// Unit tests for the two gated KVFlash selector changes (no model, CPU ggml only):
//   LUCE_KVFLASH_PREAMBLE_ROWS  pin the system+tools preamble outside the history budget
//   LUCE_KVFLASH_UQ_QUERIES     the latest human question as extra selector queries
// Both default OFF; the OFF cases must reproduce today's numbers exactly.

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"

#include "kvflash_exact_fast.h"
#include "kvflash_gpu_mass.h"
#include "kvflash_mass_synth.h"
#include "kvflash_pager.h"
#include "kvflash_prefill_reselect.h"
#include "kvflash_qk_dump.h"
#include "kvflash_uq.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPinUqFixture {};

// Token ids used by the synthetic prompts.
constexpr int32_t S = 100, U = 101, SYS = 102, NL = 103, END = 104, TR = 105, A = 106;

struct PagerHarness {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    PagerHarness() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < 2; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~PagerHarness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};
void must(bool ok) { if (!ok) throw std::runtime_error("pin_uq helper precondition failed"); }
}  // namespace

// ---- preamble end ----------------------------------------------------------
TEST_CASE(KvflashPinUqFixture, preamble_end_system_with_tools_first) {
    // <|im_start|>system \n [tools...] <|im_end|> \n <|im_start|>user ...
    const std::vector<int32_t> ids = {S, SYS, NL, 7, 8, 9, END, NL, S, U, NL, 5, END};
    CHECK(kvflash_preamble_end(ids, S, SYS) == 8);
}
TEST_CASE(KvflashPinUqFixture, preamble_end_no_system_is_zero) {
    CHECK(kvflash_preamble_end({S, U, NL, 5, END, NL}, S, SYS) == 0);   // user first
    CHECK(kvflash_preamble_end({}, S, SYS) == 0);                       // empty
    CHECK(kvflash_preamble_end({7, S, SYS, NL, 1, S, U}, S, SYS) == 0); // system not at the start
    CHECK(kvflash_preamble_end({S, SYS, NL, 7, END, NL}, S, SYS) == 0); // system only: no next turn
    CHECK(kvflash_preamble_end({S, SYS, NL, 7, S, U}, -1, SYS) == 0);   // unknown ids
    CHECK(kvflash_preamble_end({S, SYS, NL, 7, S, U}, S, -1) == 0);
}

// ---- pool sizing -----------------------------------------------------------
TEST_CASE(KvflashPinUqFixture, preamble_reserve_rounds_up_to_a_chunk) {
    CHECK(kvflash_preamble_reserve_tokens(0, 64) == 0);
    CHECK(kvflash_preamble_reserve_tokens(-5, 64) == 0);
    CHECK(kvflash_preamble_reserve_tokens(1, 64) == 64);
    CHECK(kvflash_preamble_reserve_tokens(64, 64) == 64);
    CHECK(kvflash_preamble_reserve_tokens(7168, 64) == 7168);
    CHECK(kvflash_preamble_reserve_tokens(7169, 64) == 7232);
    CHECK(kvflash_preamble_reserve_tokens(100, 0) == 0);
}
TEST_CASE(KvflashPinUqFixture, pool_with_preamble_adds_reserve_and_clamps) {
    CHECK(kvflash_pool_with_preamble(8192, 0, 131072, 64) == 8192);        // off: untouched
    CHECK(kvflash_pool_with_preamble(0, 7168, 131072, 64) == 0);           // KVFlash off stays off
    CHECK(kvflash_pool_with_preamble(8192, 7168, 131072, 64) == 15360);
    CHECK(kvflash_pool_with_preamble(8192, 7168, 10000, 64) == 9984);      // max_ctx clamp, 256 multiple
    CHECK(kvflash_pool_with_preamble(8192, 128, 131072, 64) == 8448);      // keeps the pool's 256-token stride
    CHECK(kvflash_pool_with_preamble(8192, 7168, 4096, 64) == 8192);       // max_ctx below the pool: never shrinks it
}
TEST_CASE(KvflashPinUqFixture, preamble_rows_env) {
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", nullptr);
        CHECK(kvflash_preamble_rows_from_env() == 0);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", "7168");
        CHECK(kvflash_preamble_rows_from_env() == 7168);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", "-3");
        CHECK(kvflash_preamble_rows_from_env() == 0);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", "junk");
        CHECK(kvflash_preamble_rows_from_env() == 0);
    }
}

// ---- pin rows / recall floor / budget -------------------------------------
TEST_CASE(KvflashPinUqFixture, pin_rows_never_exceed_the_reserve) {
    CHECK(kvflash_preamble_pin_rows(500, 1000) == 500);
    CHECK(kvflash_preamble_pin_rows(1500, 1000) == 1000);
    CHECK(kvflash_preamble_pin_rows(0, 1000) == 0);
    CHECK(kvflash_preamble_pin_rows(500, 0) == 0);
    CHECK(kvflash_preamble_exceeds(1500, 1000));
    CHECK(!kvflash_preamble_exceeds(1000, 1000));
    CHECK(!kvflash_preamble_exceeds(0, 0));
}
TEST_CASE(KvflashPinUqFixture, pin_chunks_counts_the_covered_chunks) {
    CHECK(kvflash_pin_chunks(0, 7168, 64) == 0);
    CHECK(kvflash_pin_chunks(7000, 0, 64) == 0);          // feature off
    CHECK(kvflash_pin_chunks(7000, 7168, 64) == 110);     // ceil(7000 / 64)
    CHECK(kvflash_pin_chunks(9000, 7168, 64) == 112);     // clipped to the reserve
}
TEST_CASE(KvflashPinUqFixture, recall_pick_never_returns_pinned_chunks) {
    // 400 chunks; the pinned preamble (chunks 0-34) scores loudest.
    std::vector<float> sc(400, 1.0f);
    for (int c = 0; c < 35; c++) sc[(size_t)c] = 100.0f;
    const int pinned = 35, unit = 10;
    const auto ids = kvflash_recall_pick(sc, 300, 50, unit, kvflash_recall_first_chunk(pinned));
    REQUIRE(!ids.empty());
    for (int b : ids) CHECK(b * unit >= pinned);
    // Without pins the loud preamble blocks (>= block 1) are picked, which is what the floor prevents.
    const auto naive = kvflash_recall_pick(sc, 300, 50, unit);
    CHECK(naive.size() > ids.size());
    // No pins: the floor is today's 10 and the pick is byte-identical.
    CHECK(kvflash_recall_first_chunk(0) == 10);
    CHECK(kvflash_recall_first_chunk(7) == 10);
    CHECK((kvflash_recall_pick(sc, 300, 50, unit, kvflash_recall_first_chunk(0)) == naive));
}
TEST_CASE(KvflashPinUqFixture, history_budget_is_unchanged_by_the_pin_reserve) {
    const int cpb = 10;
    for (int k : {1, 6, 10}) {
        for (int reserve : {2, 9, 30}) {
            const int off = kvflash_recall_tail_chunks(256, cpb * k, reserve, 0);
            CHECK(off == std::max(0, 256 - 1 - cpb * k - reserve - 1));   // today's formula, unchanged
            for (int rc : {1, 112, 200}) CHECK(kvflash_recall_tail_chunks(256 + rc, cpb * k, reserve, rc) == off);
        }
    }
    CHECK(kvflash_recall_tail_chunks(20, 60, 5, 0) == 0);   // clamps at zero
}

// ---- pager: pin, truncate, re-pin ------------------------------------------
TEST_CASE(KvflashPinUqFixture, pin_preamble_truncate_below_then_repin) {
    PagerHarness h;
    KvFlashConfig cfg;
    cfg.pool_tokens = 512; cfg.chunk_tokens = 64; cfg.sink_chunks = 1; cfg.tail_window_chunks = 4;
    KvFlashPager p;
    must(p.attach(cfg, h.k, h.v));
    must(p.alloc_span(0, 6 * 64));
    // chunk 0 is also the sink; a 64-row reserve pins exactly one chunk.
    CHECK(kvflash_pin_preamble(p, 300, 64) == 1);
    CHECK(p.is_pinned(0));
    CHECK(!p.is_pinned(1));
    p.truncate(0);                       // below the preamble: pins are clipped away
    CHECK(!p.is_pinned(0));
    must(p.alloc_span(0, 3 * 64));
    CHECK(kvflash_pin_preamble(p, 300, 64) == 1);   // re-applied
    CHECK(p.is_pinned(0));
    p.reset();                           // request start: reset clears pins
    CHECK(!p.is_pinned(0));
    CHECK(kvflash_pin_preamble(p, 300, 64) == 1);
    CHECK(kvflash_pin_preamble(p, 0, 64) == 0);    // no preamble: nothing pinned
    CHECK(kvflash_pin_preamble(p, 300, 0) == 0);   // feature off: nothing pinned
}

// ---- last human turn --------------------------------------------------------
TEST_CASE(KvflashPinUqFixture, last_human_turn_picks_the_last_non_tool_user_turn) {
    //            0  1    2   3 4 5  6   7  8 9  10 11 12 13 14
    std::vector<int32_t> ids = {S, SYS, NL, 7, END, NL, S, U, NL, 21, 22, 23, END, NL, S};
    ids.insert(ids.end(), {A, NL, 31, END, NL});                      // assistant turn (idx 14..)
    const auto t = kvflash_last_human_turn(ids, S, U, NL, END, TR);
    REQUIRE(t.valid());
    CHECK(t.lo == 9);
    CHECK(t.hi == 12);
}
TEST_CASE(KvflashPinUqFixture, last_human_turn_skips_tool_results) {
    // human question, assistant tool call, user turn that is a tool result, assistant header
    const std::vector<int32_t> ids = {S, U, NL, 21, 22, END, NL,
                                      S, A, NL, 31, END, NL,
                                      S, U, NL, TR, 41, END, NL,
                                      S, A, NL};
    const auto t = kvflash_last_human_turn(ids, S, U, NL, END, TR);
    REQUIRE(t.valid());
    CHECK(t.lo == 3);
    CHECK(t.hi == 5);
}
TEST_CASE(KvflashPinUqFixture, last_human_turn_several_human_turns_gives_the_last) {
    const std::vector<int32_t> ids = {S, U, NL, 1, END, NL, S, A, NL, 2, END, NL, S, U, NL, 3, 4, 5, END, NL, S, A, NL};
    const auto t = kvflash_last_human_turn(ids, S, U, NL, END, TR);
    CHECK(t.lo == 15);
    CHECK(t.hi == 18);
}
TEST_CASE(KvflashPinUqFixture, last_human_turn_none) {
    CHECK(!kvflash_last_human_turn({}, S, U, NL, END, TR).valid());
    CHECK(!kvflash_last_human_turn({S, SYS, NL, 7, END, NL}, S, U, NL, END, TR).valid());           // no user turn
    CHECK(!kvflash_last_human_turn({S, U, NL, TR, 9, END}, S, U, NL, END, TR).valid());             // only a tool result
    CHECK(!kvflash_last_human_turn({S, U, NL, 9, 9}, S, U, NL, END, TR).valid());                   // unterminated
    CHECK(!kvflash_last_human_turn({S, U, NL, 9, END}, -1, U, NL, END, TR).valid());                // unknown ids
    // Unknown tool_response id: tool results cannot be told apart, the turn counts as human.
    CHECK(kvflash_last_human_turn({S, U, NL, TR, 9, END}, S, U, NL, END, -1).valid());
}

// ---- capture plan, batch take, store ---------------------------------------
TEST_CASE(KvflashPinUqFixture, capture_plan_covers_the_window_or_says_why_not) {
    const KvFlashHumanTurn t{100, 300};
    auto p = kvflash_uq_plan(t, 64, 0, 1000);
    CHECK(p.capture); CHECK(p.n == 64); CHECK(p.win_lo == 236); CHECK(p.cut == 300);
    p = kvflash_uq_plan(KvFlashHumanTurn{100, 130}, 64, 0, 1000);   // short question: only its own tokens
    CHECK(p.capture); CHECK(p.n == 30); CHECK(p.win_lo == 100);
    p = kvflash_uq_plan(t, 64, 250, 1000);                           // window starts before the prefill
    CHECK(!p.capture);
    p = kvflash_uq_plan(t, 64, 0, 299);                              // prefill ends before the question does
    CHECK(!p.capture);
    p = kvflash_uq_plan(t, 0, 0, 1000);                              // N = 0: feature off
    CHECK(!p.capture); CHECK(p.n == 0);
    p = kvflash_uq_plan(KvFlashHumanTurn{}, 64, 0, 1000);            // no human turn
    CHECK(!p.capture);
}
TEST_CASE(KvflashPinUqFixture, batch_take_intersects_window_with_captured_rows) {
    // Batch [512, 1024) captured its last 64 rows = positions [960, 1024); window [980, 1024).
    auto t = kvflash_uq_batch_take(512, 512, 64, 980, 1024);
    CHECK(t.first_row == 20); CHECK(t.count == 44);
    // Window ends inside the batch (forced cut at hi): batch [768, 900) with hi = 900.
    t = kvflash_uq_batch_take(768, 132, 64, 850, 900);
    CHECK(t.first_row == 14); CHECK(t.count == 50);
    // No overlap.
    CHECK(kvflash_uq_batch_take(0, 512, 64, 980, 1024).count == 0);
    CHECK(kvflash_uq_batch_take(1024, 512, 64, 980, 1024).count == 0);
    // Short batch: captured rows = batch tokens.
    t = kvflash_uq_batch_take(1000, 24, 24, 980, 1024);
    CHECK(t.first_row == 0); CHECK(t.count == 24);
}
TEST_CASE(KvflashPinUqFixture, store_key_and_invalidation) {
    std::vector<int32_t> ids(1000);
    for (int i = 0; i < 1000; i++) ids[(size_t)i] = i % 97;
    KvFlashUqStore st;
    CHECK(st.empty());
    CHECK(!kvflash_uq_matches(st, ids, 300));
    st.lo = 270; st.hi = 300; st.n = 2; st.hash = kvflash_uq_hash(ids, 300); st.rows.assign(8, 1.0f);
    CHECK(kvflash_uq_matches(st, ids, 300));
    CHECK(!kvflash_uq_matches(st, ids, 301));            // different hi
    auto other = ids; other[299] ^= 1;
    CHECK(!kvflash_uq_matches(st, other, 300));          // hash of ids[0,hi) differs
    auto later = ids; later[300] ^= 1;
    CHECK(kvflash_uq_matches(st, later, 300));           // only ids[0,hi) matters
    CHECK(!kvflash_uq_matches(st, std::vector<int32_t>(ids.begin(), ids.begin() + 299), 300));
    // history truncated below hi drops the store; at or above keeps it
    KvFlashUqStore keep = st;
    kvflash_uq_truncate(keep, 300);
    CHECK(!keep.empty());
    kvflash_uq_truncate(st, 299);
    CHECK(st.empty());
    CHECK(st.rows.empty());
    // revalidate against a history: a cold prefill with a different prefix drops it
    KvFlashUqStore s2; s2.lo = 270; s2.hi = 300; s2.n = 2; s2.hash = kvflash_uq_hash(ids, 300); s2.rows.assign(8, 1.0f);
    kvflash_uq_revalidate(s2, ids);
    CHECK(!s2.empty());
    kvflash_uq_revalidate(s2, other);
    CHECK(s2.empty());
}
TEST_CASE(KvflashPinUqFixture, uq_queries_env_off_by_default) {
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_UQ_QUERIES", nullptr);
        luce_test::ScopedEnvVar d("KVFLASH_QK_DUMP", nullptr);
        luce_test::ScopedEnvVar m("LUCE_KVFLASH_EXACT_DUMP_MQ", nullptr);
        luce_test::ScopedEnvVar q("LUCE_KVFLASH_SELECT_QUERIES", nullptr);
        CHECK(kvflash_uq_queries() == 0);
        CHECK(kvflash_q_capture_all_tokens() == 0);      // today's value: nothing captured
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_UQ_QUERIES", "64");
        luce_test::ScopedEnvVar d("KVFLASH_QK_DUMP", nullptr);
        luce_test::ScopedEnvVar m("LUCE_KVFLASH_EXACT_DUMP_MQ", nullptr);
        luce_test::ScopedEnvVar q("LUCE_KVFLASH_SELECT_QUERIES", "16");
        CHECK(kvflash_uq_queries() == 64);
        CHECK(kvflash_q_capture_all_tokens() == 64);
    }
}

// ---- scorer: an extra aligned query raises that chunk -----------------------
TEST_CASE(KvflashPinUqFixture, extra_query_raises_its_chunk_in_the_cpu_selector) {
    SynthCfg c;
    c.L = 2; c.HK = 2; c.G = 3; c.D = 64; c.CT = 16; c.NB = 24; c.KT = GGML_TYPE_Q8_0;
    c.sealed = 100; c.N = 8; c.keep_newest = 4; c.tail = 16;
    Synth s;
    s.build(c);
    REQUIRE(s.n_hot >= 8 && s.n_hot < 13);              // hot j and j + 8 would share query j % 8; j = 5 has no partner
    const int hot = 5;                                  // planted chunk 10 + 7*5 = 45, driven by query 5 alone
    const int chunk = 10 + 7 * hot;
    KvFlashExactKeys keys;
    keys.n_chunks = c.sealed; keys.chunk_tokens = c.CT; keys.type = c.KT;
    keys.seg = [&](int ch, int l, int h) -> const uint8_t * {
        return s.blk[(size_t)ch] >= 0 ? s.pool_seg(s.blk[(size_t)ch], l, h) : s.host_seg(ch, l, h);
    };
    auto select = [&](const std::vector<const float *> & qs) {
        std::vector<std::vector<float>> outs;
        std::vector<KvFlashMassPlanes> planes;
        must(kvflash_exact_mass_scores_fast_multi(keys, s.dims(), qs, outs, 0, nullptr, &planes));
        return kvflash_select_scores(planes);
    };
    // Base queries: all but #5. The stored user-question row is query #5.
    std::vector<const float *> base;
    for (int n = 0; n < c.N; n++) if (n != hot) base.push_back(s.qs.data() + (size_t)n * s.QS);
    KvFlashUqStore st;
    st.n = 1; st.rows.assign(s.qs.begin() + (long)hot * (long)s.QS, s.qs.begin() + (long)(hot + 1) * (long)s.QS);

    std::vector<const float *> with_uq = base;
    kvflash_uq_append_queries(with_uq, st, s.QS);
    REQUIRE(with_uq.size() == base.size() + 1);
    const auto sel_base = select(base), sel_uq = select(with_uq);
    CHECK(sel_uq[(size_t)chunk] > sel_base[(size_t)chunk] * 1.5f);
    // Monotone: max over queries never lowers any chunk.
    for (size_t i = 0; i < sel_base.size(); i++) CHECK(sel_uq[i] >= sel_base[i]);
    // The recall pick with the extra query now contains the chunk's block, the base pick does not.
    const int unit = 2, fwc = c.sealed - c.tail;
    const auto pb = kvflash_recall_pick(sel_base, fwc, 7, unit), pu = kvflash_recall_pick(sel_uq, fwc, 7, unit);
    auto has = [&](const std::vector<int> & v) { for (int b : v) if (b * unit <= chunk && chunk < (b + 1) * unit) return true; return false; };
    CHECK(has(pu));
    CHECK(!has(pb));
    // N = 0 (empty store): the query list is untouched, byte for byte.
    KvFlashUqStore none;
    std::vector<const float *> same = base;
    kvflash_uq_append_queries(same, none, s.QS);
    CHECK((same == base));
}

// ---- GPU scorer VRAM guard (pure) ---------------------------------------------
TEST_CASE(KvflashPinUqFixture, gpu_mass_fits_guard_matches_the_scorer_rule) {
    const size_t MiB = (size_t)1 << 20, margin = 512 * MiB;
    CHECK(kvflash_gpu_mass_fits(100 * MiB, 0, 1000 * MiB, 8000 * MiB, margin));
    CHECK(!kvflash_gpu_mass_fits(600 * MiB, 0, 1000 * MiB, 8000 * MiB, margin));        // 600 + 512 > 1000
    CHECK(kvflash_gpu_mass_fits(600 * MiB, 200 * MiB, 1000 * MiB, 8000 * MiB, margin)); // the held buffer is already out of free
    CHECK(kvflash_gpu_mass_fits(600 * MiB, 700 * MiB, 10 * MiB, 8000 * MiB, margin));   // need <= held
    CHECK(kvflash_gpu_mass_fits(9999 * MiB, 0, 0, 0, margin));                          // unknown total: no guard
    // 64 queries more cost real bytes: 128 queries need more than 64 on the same geometry.
    KvFlashQkDims d; d.n_layers = 16; d.n_q_heads = 24; d.n_kv_heads = 4; d.head_dim = 256;
    CHECK(kvflash_gpu_mass_bytes(d, GGML_TYPE_Q4_0, 64, 256, 300, 128) > kvflash_gpu_mass_bytes(d, GGML_TYPE_Q4_0, 64, 256, 300, 64));
}
