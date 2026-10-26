// C1.3-02: the pure decisions around N KVFlash sessions on one engine:
// when the seq engine is built (including explicit --max-concurrency 1),
// which architectures and kernels may run it, and the LRU-only profile that
// replaces the serial recall defaults. No GPU, no startup.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_concurrent_gate.h"

using namespace luce::common;

namespace {
struct KvflashConcurrentGateFixture {};
}  // namespace

TEST_CASE(KvflashConcurrentGateFixture, seq_engine_is_wanted_for_n_gt_1_explicit_1_or_a_kvflash_arena_when_paged) {
    SeqRouteInput in;
    in.paged = true;
    CHECK(!seq_engine_wanted(in));                       // default N=1: the single-sequence path, as before
    in.n_slots = 2;
    CHECK(seq_engine_wanted(in));
    in.n_slots = 1; in.max_concurrency_set = true;       // explicit --max-concurrency 1
    CHECK(seq_engine_wanted(in));
    in.max_concurrency_set = false; in.kvflash_arena = true;
    CHECK(seq_engine_wanted(in));
    in.paged = false;                                    // never without paged attention
    CHECK(!seq_engine_wanted(in));
    in.n_slots = 8; in.max_concurrency_set = true;
    CHECK(!seq_engine_wanted(in));
}

TEST_CASE(KvflashConcurrentGateFixture, only_dense_qwen35_may_run_the_concurrent_arena) {
    KvFlashConcurrentRequest r;
    r.arch = "qwen35";
    CHECK(!kvflash_concurrent_refusal(r).has_value());
    for (const char * a : {"qwen35moe", "qwen3", "gemma4", "laguna", "deepseek4", ""}) {
        r.arch = a;
        const auto m = kvflash_concurrent_refusal(r);
        REQUIRE(m.has_value());
        CHECK(m->find("qwen35") != std::string::npos);
    }
}

TEST_CASE(KvflashConcurrentGateFixture, wmma_draft_chain_tree_and_pflash_are_refused_with_a_message_each) {
    auto base = [] { KvFlashConcurrentRequest r; r.arch = "qwen35"; return r; };
    {
        auto r = base(); r.wmma_env = true;               // until the kernel excludes null blocks
        const auto m = kvflash_concurrent_refusal(r);
        REQUIRE(m.has_value());
        CHECK(m->find("PAGED_WMMA") != std::string::npos);
        r.wmma_null_exclusion_ready = true;               // lifted by the gate-lift commit, not before
        CHECK(!kvflash_concurrent_refusal(r).has_value());
    }
    { auto r = base(); r.draft = true;  const auto m = kvflash_concurrent_refusal(r); REQUIRE(m.has_value()); CHECK(m->find("draft") != std::string::npos); }
    { auto r = base(); r.tree = true;   const auto m = kvflash_concurrent_refusal(r); REQUIRE(m.has_value()); CHECK(m->find("tree") != std::string::npos); }
    { auto r = base(); r.pflash = true; const auto m = kvflash_concurrent_refusal(r); REQUIRE(m.has_value()); CHECK(m->find("PFlash") != std::string::npos); }
    { auto r = base(); r.images = true; const auto m = kvflash_concurrent_refusal(r); REQUIRE(m.has_value()); CHECK(m->find("image") != std::string::npos); }
}

TEST_CASE(KvflashConcurrentGateFixture, the_moe_check_runs_before_a_subclass_can_erase_the_request) {
    // Qwen35MoeBackend::post_kvflash_init_gate may disable the pool and return true.
    // The request-level refusal looks at the REQUESTED combination, so an all-hot
    // placement that would drop the pool still refuses.
    KvFlashConcurrentRequest r;
    r.arch = "qwen35moe";
    r.kvflash_requested = true;
    r.pool_would_be_disabled_by_placement = true;
    const auto m = kvflash_concurrent_refusal(r);
    REQUIRE(m.has_value());
    r.kvflash_auto = true;                                // auto request, same verdict
    CHECK(kvflash_concurrent_refusal(r).has_value());
}

TEST_CASE(KvflashConcurrentGateFixture, the_concurrent_profile_is_lru_only_by_default_and_rejects_c2_requests) {
    KvFlashProfileRequest p;                              // plain --kvflash: serial defaults would install QK, recall, ...
    const KvFlashProfile ok = kvflash_resolve_concurrent_profile(p);
    CHECK(ok.ok);
    CHECK(ok.policy == "lru-only");
    CHECK(ok.suppress_serial_defaults);                   // the recall env defaults must not be installed
    CHECK(!ok.log_line.empty());
    CHECK(ok.log_line.find("lru-only") != std::string::npos);
    for (int i = 0; i < 3; i++) {                         // C2 pieces, alone or on top of a full recall request
        for (int with_recall = 0; with_recall < 2; with_recall++) {
            KvFlashProfileRequest q;
            if (with_recall) { q.explicit_qk_policy = q.explicit_recall = q.exact_scorer = true; q.reselect_mode = 4; q.select_queries = 64; }
            switch (i) {
                case 0: q.explicit_uq = true; break;
                case 1: q.explicit_preamble_pin = true; break;
                case 2: q.explicit_tau = true; break;
            }
            const KvFlashProfile bad = kvflash_resolve_concurrent_profile(q);
            CHECK(!bad.ok);
            CHECK(!bad.error.empty());
            CHECK(bad.error.find("C2") != std::string::npos);
        }
    }
}

TEST_CASE(KvflashConcurrentGateFixture, the_recall_profile_needs_every_explicit_piece_and_installs_no_default) {
    KvFlashProfileRequest full;                           // C1.4: qk + mode 4 + exact scorer + queries + recall blocks, GPU scoring optional
    full.explicit_qk_policy = full.explicit_recall = full.exact_scorer = true;
    full.reselect_mode = 4; full.select_queries = 64;
    const KvFlashProfile ok = kvflash_resolve_concurrent_profile(full);
    CHECK(ok.ok);
    CHECK(ok.policy == "recall");
    CHECK(ok.suppress_serial_defaults);
    CHECK(ok.log_line.find("recall") != std::string::npos);
    full.explicit_gpu_score = true;
    CHECK(kvflash_resolve_concurrent_profile(full).ok);
    for (int i = 0; i < 7; i++) {                         // a partial request is refused, not completed with defaults
        KvFlashProfileRequest q = full;
        switch (i) {
            case 0: q = {}; q.explicit_qk_policy = true; break;
            case 1: q = {}; q.explicit_recall = true; break;
            case 2: q = {}; q.explicit_gpu_score = true; break;
            case 3: q = {}; q.reselect_mode = 4; break;
            case 4: q.reselect_mode = 3; break;                // mode 3 is not per-slot recall
            case 5: q.exact_scorer = false; break;
            case 6: q.select_queries = 0; break;
        }
        const KvFlashProfile bad = kvflash_resolve_concurrent_profile(q);
        CHECK(!bad.ok);
        CHECK(bad.error.find("together") != std::string::npos);
    }
}

TEST_CASE(KvflashConcurrentGateFixture, test_switch_opens_only_on_exactly_1) {
    const char * old = std::getenv("LUCE_KVFLASH_CONCURRENT_TEST");
    unsetenv("LUCE_KVFLASH_CONCURRENT_TEST");
    CHECK(!luce::common::kvflash_concurrent_test_switch());
    setenv("LUCE_KVFLASH_CONCURRENT_TEST", "0", 1);
    CHECK(!luce::common::kvflash_concurrent_test_switch());
    setenv("LUCE_KVFLASH_CONCURRENT_TEST", "1", 1);
    CHECK(luce::common::kvflash_concurrent_test_switch());
    if (old) setenv("LUCE_KVFLASH_CONCURRENT_TEST", old, 1); else unsetenv("LUCE_KVFLASH_CONCURRENT_TEST");
}
