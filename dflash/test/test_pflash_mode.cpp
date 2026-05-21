// Unit tests for PFlash mode wiring bugs (no GPU, no model files required).
//
// P0: env-fallback must NOT override explicit PFlashMode::OFF from request path.
// P1a: compute_anchor_hits outer loop must not exit when buffer fills (recall regression).
// P1b: AUTO mode L_compress default must be 32768 not 8192.
// P2: unknown extra_body.pflash_mode string must return error, not silently fall through.
//
// Build: cmake --build . --target test_pflash_mode
// Run:   ./test_pflash_mode

#include "qwen3/qwen3_drafter.h"
#include "server/http_server.h"
#include "common/model_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <optional>
#include <vector>

using namespace dflash27b;

// ─── Test framework ────────────────────────────────────────────────────

static int test_failures = 0;
static int test_count    = 0;

#define TEST_ASSERT(expr) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    } \
} while (0)

#define TEST_ASSERT_MSG(expr, msg) do { \
    test_count++; \
    if (!(expr)) { \
        test_failures++; \
        std::fprintf(stderr, "  FAIL: %s:%d: %s — %s\n", __FILE__, __LINE__, #expr, msg); \
    } \
} while (0)

#define RUN_TEST(fn) do { \
    std::fprintf(stderr, "  %s ...", #fn); \
    int before = test_failures; \
    fn(); \
    if (test_failures == before) std::fprintf(stderr, " ok\n"); \
    else std::fprintf(stderr, "\n"); \
} while (0)

// ═══════════════════════════════════════════════════════════════════════
// P0 — env-fallback must not override explicit OFF from request path
// ═══════════════════════════════════════════════════════════════════════
//
// resolve_pflash_mode(PFlashMode) is the extracted free function that
// implements the mode-resolution step in drafter_score_and_compress.
// When mode_param == OFF, the production bug allows DFLASH_PFLASH_MODE=always
// to upgrade the mode to ALWAYS. The fix: remove the env-fallback entirely.

static void test_p0_env_does_not_override_off() {
    // Set env to "always" — the bug upgrades OFF to ALWAYS.
    setenv("DFLASH_PFLASH_MODE", "always", 1);
    PFlashMode resolved = resolve_pflash_mode(PFlashMode::OFF);
    unsetenv("DFLASH_PFLASH_MODE");
    // OFF must stay OFF regardless of env when the caller explicitly passed OFF.
    TEST_ASSERT_MSG(resolved == PFlashMode::OFF,
                    "env DFLASH_PFLASH_MODE=always must not override explicit OFF");
}

static void test_p0_env_does_not_override_auto() {
    // Even AUTO must not be changed by env (env is only for shell callers who pass OFF).
    setenv("DFLASH_PFLASH_MODE", "off", 1);
    PFlashMode resolved = resolve_pflash_mode(PFlashMode::AUTO);
    unsetenv("DFLASH_PFLASH_MODE");
    TEST_ASSERT_MSG(resolved == PFlashMode::AUTO,
                    "env DFLASH_PFLASH_MODE must not downgrade explicit AUTO");
}

static void test_p0_legacy_skip_drafter_still_works() {
    // DFLASH_SKIP_DRAFTER env is the legacy back-compat path: when mode is OFF
    // AND no DFLASH_PFLASH_MODE env is set, drafter_score_and_compress checks
    // DFLASH_SKIP_DRAFTER. The fixed resolve_pflash_mode returns OFF; the caller
    // is responsible for the DFLASH_SKIP_DRAFTER check on the OFF branch.
    // This test verifies resolve_pflash_mode(OFF) == OFF even when SKIP_DRAFTER is set.
    setenv("DFLASH_SKIP_DRAFTER", "1", 1);
    PFlashMode resolved = resolve_pflash_mode(PFlashMode::OFF);
    unsetenv("DFLASH_SKIP_DRAFTER");
    TEST_ASSERT_MSG(resolved == PFlashMode::OFF,
                    "DFLASH_SKIP_DRAFTER must not affect mode resolution");
}

// ═══════════════════════════════════════════════════════════════════════
// P1a — compute_anchor_hits outer loop recall regression
// ═══════════════════════════════════════════════════════════════════════
//
// craft_ids builds a 200-token body where:
//   - query n-gram at position q0 (token 150) has 9 body matches (fills 16-buf)
//   - a later query n-gram at position q0+4 has 1 body match at position 90
//
// Pre-refactor (per-q 8-cap): inner scan caps at 8/q, outer loop keeps going.
// Current code (total < max_hits_buf on outer): stops at q0 after 16 hits,
// dropping the later q-gram match at body position 90.
//
// We test at the chunk-coverage level: with chunk_size=32, body pos 90 is in
// chunk 2. The test verifies chunk 2 is forced after compute_anchor_hits.

static void test_p1a_anchor_hits_recall_regression() {
    // Pattern token IDs (must not conflict with fill tokens)
    constexpr int32_t A = 1001, B = 1002, C = 1003, D = 1004; // first query n-gram
    constexpr int32_t E = 2001, F = 2002, G = 2003, H = 2004; // second query n-gram
    constexpr int32_t FILL = 5;

    const int S = 200;
    const int query_tokens = 50;  // last 50 tokens are query
    const int q0 = S - query_tokens; // = 150

    std::vector<int32_t> ids(S, FILL);

    // Place first query n-gram at q0..q0+3
    ids[q0]   = A; ids[q0+1] = B; ids[q0+2] = C; ids[q0+3] = D;
    // Place second query n-gram at q0+4..q0+7
    ids[q0+4] = E; ids[q0+5] = F; ids[q0+6] = G; ids[q0+7] = H;

    // Place 9 body matches for the first n-gram (positions 0..8*4, step 4)
    for (int i = 0; i < 9; ++i) {
        int pos = i * 10; // positions 0, 10, 20, ..., 80
        if (pos + 4 <= q0 - 4) {
            ids[pos]   = A; ids[pos+1] = B; ids[pos+2] = C; ids[pos+3] = D;
        }
    }
    // Place 1 body match for the second n-gram at position 90
    ids[90] = E; ids[91] = F; ids[92] = G; ids[93] = H;

    const int max_hits_per_q = 8;
    const int max_hits_buf   = 16;
    int hit_pos[16] = {0};
    const int total = compute_anchor_hits(ids, S, query_tokens,
                                          max_hits_per_q, max_hits_buf, hit_pos);

    // With the bug (outer loop exits at total==max_hits_buf), the match at
    // position 90 is never recorded. With the fix, total > 0 and at least one
    // hit_pos[] value should be 90.
    bool found_90 = false;
    for (int i = 0; i < total; ++i) {
        if (hit_pos[i] == 90) { found_90 = true; break; }
    }
    TEST_ASSERT_MSG(found_90,
        "body match at pos 90 (second query n-gram) must not be dropped when buffer fills");
}

// ═══════════════════════════════════════════════════════════════════════
// P1b — AUTO mode L_compress default must be 32768
// ═══════════════════════════════════════════════════════════════════════
//
// The drafter forward costs 14-16s at 32K (plan §2). AUTO should only skip
// drafter when S >= 32768. Current default 8192 causes over-aggressive bypass
// at mid-context (e.g. S=10000).

static void test_p1b_auto_threshold_default_32k() {
    // With the correct default, S=10000 with anchor_hits>0 should NOT activate
    // the skip path (10000 < 32768). We test resolve_auto_skip() directly.
    // resolve_auto_skip(S, anchor_hits) returns true when AUTO would skip the drafter.
    unsetenv("DFLASH_PFLASH_L_COMPRESS");  // ensure default is used
    const int S_mid = 10000;
    const int anchor_hits = 1;
    bool should_skip = resolve_auto_skip(S_mid, anchor_hits);
    TEST_ASSERT_MSG(!should_skip,
        "AUTO must not skip drafter at S=10000 with default L_compress=32768");
}

static void test_p1b_auto_threshold_at_32k() {
    unsetenv("DFLASH_PFLASH_L_COMPRESS");
    const int S_long = 32768;
    const int anchor_hits = 1;
    bool should_skip = resolve_auto_skip(S_long, anchor_hits);
    TEST_ASSERT_MSG(should_skip,
        "AUTO must skip drafter at S=32768 with anchor_hits>0 (default L_compress=32768)");
}

static void test_p1b_auto_no_skip_without_anchors() {
    unsetenv("DFLASH_PFLASH_L_COMPRESS");
    // Even at S > L_compress, no anchors -> no skip (Louver-style recall floor)
    bool should_skip = resolve_auto_skip(40000, 0);
    TEST_ASSERT_MSG(!should_skip,
        "AUTO must not skip drafter when anchor_hits==0 (no retrieval signal)");
}

// ═══════════════════════════════════════════════════════════════════════
// P2 — unknown extra_body.pflash_mode must return nullopt (not silently ignore)
// ═══════════════════════════════════════════════════════════════════════
//
// parse_pflash_mode_str(str) returns std::optional<PFlashMode>.
// nullopt means "invalid string" → caller should return 400.
// The bug: unknown strings silently return the server-wide config.

static void test_p2_valid_strings_parse() {
    auto off    = parse_pflash_mode_str("off");
    auto auto_  = parse_pflash_mode_str("auto");
    auto always = parse_pflash_mode_str("always");
    TEST_ASSERT(off.has_value() && *off == PFlashMode::OFF);
    TEST_ASSERT(auto_.has_value() && *auto_ == PFlashMode::AUTO);
    TEST_ASSERT(always.has_value() && *always == PFlashMode::ALWAYS);
}

static void test_p2_unknown_string_returns_nullopt() {
    auto bad1 = parse_pflash_mode_str("alway");   // typo
    auto bad2 = parse_pflash_mode_str("true");    // wrong type
    auto bad3 = parse_pflash_mode_str("");         // empty
    auto bad4 = parse_pflash_mode_str("OFF");      // wrong case
    TEST_ASSERT_MSG(!bad1.has_value(), "typo 'alway' must return nullopt");
    TEST_ASSERT_MSG(!bad2.has_value(), "'true' must return nullopt");
    TEST_ASSERT_MSG(!bad3.has_value(), "empty string must return nullopt");
    TEST_ASSERT_MSG(!bad4.has_value(), "'OFF' (uppercase) must return nullopt");
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

int main() {
    std::fprintf(stderr, "══════════════════════════════════════════\n");
    std::fprintf(stderr, " PFlash Mode Wiring Tests\n");
    std::fprintf(stderr, "══════════════════════════════════════════\n");

    std::fprintf(stderr, "\n── P0: env-fallback must not override explicit OFF ──\n");
    RUN_TEST(test_p0_env_does_not_override_off);
    RUN_TEST(test_p0_env_does_not_override_auto);
    RUN_TEST(test_p0_legacy_skip_drafter_still_works);

    std::fprintf(stderr, "\n── P1a: anchor-hits recall regression ──\n");
    RUN_TEST(test_p1a_anchor_hits_recall_regression);

    std::fprintf(stderr, "\n── P1b: AUTO threshold default 32K ──\n");
    RUN_TEST(test_p1b_auto_threshold_default_32k);
    RUN_TEST(test_p1b_auto_threshold_at_32k);
    RUN_TEST(test_p1b_auto_no_skip_without_anchors);

    std::fprintf(stderr, "\n── P2: unknown pflash_mode string rejected ──\n");
    RUN_TEST(test_p2_valid_strings_parse);
    RUN_TEST(test_p2_unknown_string_returns_nullopt);

    std::fprintf(stderr, "\n══════════════════════════════════════════\n");
    std::fprintf(stderr, " Results: %d assertions, %d failures\n",
                 test_count, test_failures);
    std::fprintf(stderr, "══════════════════════════════════════════\n");

    if (test_failures) { std::fprintf(stderr, "FAILED\n"); return 1; }
    std::fprintf(stderr, "ALL PASSED\n");
    return 0;
}
