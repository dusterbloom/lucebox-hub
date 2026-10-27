// KVFlash 2.0 sessions P1a: the preamble store core (SESSIONS_DESIGN.md 3.2), pure, no GPU.
// Chunk-prefix hashing, the boundary rule, refcount + LRU within a budget, and the capture bound (P2-04).

#include "CppUnitTestFramework.hpp"

#include "kvflash_preamble_store.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPreambleStoreFixture {};

constexpr int kCt = 64;
constexpr int64_t kChunkB = 2228224;     // 2.125 MiB: the q8_0 chunk of the 27B geometry
constexpr int64_t kSlabB = 156893184;    // 149.6 MiB

// A prompt of `n` tokens: `shared` leading tokens common to every session, the rest unique per `tag`.
std::vector<int32_t> prompt(int n, int shared, int tag) {
    std::vector<int32_t> v((size_t)n);
    for (int i = 0; i < n; i++) v[(size_t)i] = i < shared ? 1000 + i : 100000 * tag + i;
    return v;
}
std::vector<uint64_t> hashes(const std::vector<int32_t> & ids, int e_role) {
    return kvflash_preamble_hashes(ids, kvflash_preamble_ec(e_role, kCt) / kCt, kCt);
}
}  // namespace

// ---- hashing and the boundary ----------------------------------------------
TEST_CASE(KvflashPreambleStoreFixture, ec_is_chunk_aligned_and_never_above_e_role) {
    CHECK(kvflash_preamble_ec(0, kCt) == 0);
    CHECK(kvflash_preamble_ec(63, kCt) == 0);
    CHECK(kvflash_preamble_ec(64, kCt) == 64);
    CHECK(kvflash_preamble_ec(1143, kCt) == 1088);   // unaligned E_role: floor
    CHECK(kvflash_preamble_ec(-5, kCt) == 0);
}

TEST_CASE(KvflashPreambleStoreFixture, hashes_chain_over_chunks_and_are_prefix_stable) {
    const auto a = prompt(1300, 1300, 1);
    const auto h = kvflash_preamble_hashes(a, 18, kCt);
    REQUIRE(h.size() == 18);
    const auto h9 = kvflash_preamble_hashes(a, 9, kCt);
    for (int i = 0; i < 9; i++) CHECK(h[(size_t)i] == h9[(size_t)i]);   // H_k does not depend on later chunks
    CHECK(h[0] != h[1]);
    auto b = a;
    b[100] ^= 1;                                                         // chunk 1 differs
    const auto hb = kvflash_preamble_hashes(b, 18, kCt);
    CHECK(hb[0] == h[0]);
    for (int i = 1; i < 18; i++) CHECK(hb[(size_t)i] != h[(size_t)i]);   // the chain carries the difference forward
    CHECK(kvflash_preamble_hashes(a, 100, kCt).size() == 20);            // asking for more chunks than ids hold clamps
}

TEST_CASE(KvflashPreambleStoreFixture, shared_chunks_is_the_longest_aligned_common_prefix) {
    // two sessions share 9 chunks + 10 tokens, then differ; E_role of each is unaligned and longer than the shared part
    const auto a = prompt(1300, 586, 1), b = prompt(1350, 586, 2);
    CHECK(kvflash_preamble_shared_chunks(a, b, kCt, 1280) == 9);
    CHECK(kvflash_preamble_shared_chunks(a, b, kCt, 4 * kCt) == 4);      // capped by E_c
    CHECK(kvflash_preamble_shared_chunks(a, a, kCt, 1280) == 20);
    CHECK(kvflash_preamble_shared_chunks(a, {}, kCt, 1280) == 0);
}

TEST_CASE(KvflashPreambleStoreFixture, decide_boundary_across_sessions_sharing_a_prefix) {
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    const auto a = prompt(1300, 586, 1), b = prompt(1350, 586, 2);
    const int e_a = 1143, e_b = 1200;                                    // unaligned E_role
    // session A: nothing stored -> cold, capture at E_c(A) = 17 chunks
    auto d = kvflash_preamble_decide(st, a, e_a, kCt, "x", /*k_max=*/26);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 17);
    REQUIRE(st.capture(hashes(a, e_a), "x") == KvFlashPreambleStore::Stored);
    // A again: full hit
    d = kvflash_preamble_decide(st, a, e_a, kCt, "x", 26);
    CHECK(d.kind == KvFlashPreambleDecision::Attach);
    CHECK(d.k == 17);
    // B shares only 9 chunks with the entry of 17: partial hit -> cold admit, capture at the shared cut 9
    d = kvflash_preamble_decide(st, b, e_b, kCt, "x", 26);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 9);
    REQUIRE(st.capture(kvflash_preamble_hashes(b, 9, kCt), "x") == KvFlashPreambleStore::Stored);
    // B now hits its own shorter entry at 9; a third session sharing the same 9 chunks attaches it too
    d = kvflash_preamble_decide(st, b, 9 * kCt, kCt, "x", 26);
    CHECK(d.kind == KvFlashPreambleDecision::Attach);
    CHECK(d.k == 9);
    const auto c = prompt(1400, 586, 3);
    d = kvflash_preamble_decide(st, c, 700, kCt, "x", 26);               // E_c = 10, shares 9
    CHECK(d.kind == KvFlashPreambleDecision::Attach);
    CHECK(d.k == 9);
}

TEST_CASE(KvflashPreambleStoreFixture, decide_needs_two_chunks_and_matching_exec_id) {
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    const auto a = prompt(1300, 1300, 1);
    CHECK(kvflash_preamble_decide(st, a, 127, kCt, "x", 26).kind == KvFlashPreambleDecision::Cold);     // E_c = 1 chunk
    CHECK(kvflash_preamble_decide(st, a, 128, kCt, "x", 26).kind == KvFlashPreambleDecision::Capture);  // E_c >= 128
    REQUIRE(st.capture(hashes(a, 1143), "x") == KvFlashPreambleStore::Stored);
    // another execution identity neither attaches nor reuses the entry as a partial hit
    auto d = kvflash_preamble_decide(st, a, 1143, kCt, "y", 26);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 17);
}

TEST_CASE(KvflashPreambleStoreFixture, hit_above_this_slots_k_max_goes_cold_and_captures_shorter) {
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    const auto a = prompt(1700, 1700, 1);
    REQUIRE(st.capture(hashes(a, 1600), "x") == KvFlashPreambleStore::Stored);     // 25 chunks
    auto d = kvflash_preamble_decide(st, a, 1600, kCt, "x", /*k_max=*/12);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);                              // the entry's slab is at 25, not 12
    CHECK(d.k == 12);
    d = kvflash_preamble_decide(st, a, 1600, kCt, "x", /*k_max=*/1);
    CHECK(d.kind == KvFlashPreambleDecision::Cold);                                 // k_max < 2: nothing worth keeping
    d = kvflash_preamble_decide(st, a, 1600, kCt, "x", /*k_max=*/0);
    CHECK(d.kind == KvFlashPreambleDecision::Cold);
}

// ---- refcount and LRU --------------------------------------------------------
TEST_CASE(KvflashPreambleStoreFixture, lru_evicts_oldest_unreferenced_entry_to_fit) {
    const int64_t one = 4 * kChunkB + kSlabB;
    KvFlashPreambleStore st(2 * one + 1, kSlabB, kChunkB);   // two entries of 4 chunks fit
    const auto a = prompt(400, 0, 1), b = prompt(400, 0, 2), c = prompt(400, 0, 3);
    REQUIRE(st.capture(kvflash_preamble_hashes(a, 4, kCt), "x") == KvFlashPreambleStore::Stored);
    REQUIRE(st.capture(kvflash_preamble_hashes(b, 4, kCt), "x") == KvFlashPreambleStore::Stored);
    CHECK(st.count() == 2);
    CHECK(st.used_bytes() == 2 * one);
    st.touch(kvflash_preamble_hashes(a, 4, kCt).back(), "x");   // a is now the most recent
    REQUIRE(st.capture(kvflash_preamble_hashes(c, 4, kCt), "x") == KvFlashPreambleStore::Stored);
    CHECK(st.count() == 2);
    CHECK(st.find(kvflash_preamble_hashes(b, 4, kCt).back(), "x") == nullptr);   // b was the LRU
    CHECK(st.find(kvflash_preamble_hashes(a, 4, kCt).back(), "x") != nullptr);
    CHECK(st.used_bytes() <= st.budget_bytes());
}

TEST_CASE(KvflashPreambleStoreFixture, positive_refcount_is_never_evicted) {
    const int64_t one = 4 * kChunkB + kSlabB;
    KvFlashPreambleStore st(2 * one, kSlabB, kChunkB);
    const auto a = prompt(400, 0, 1), b = prompt(400, 0, 2), c = prompt(400, 0, 3);
    const auto ha = kvflash_preamble_hashes(a, 4, kCt), hb = kvflash_preamble_hashes(b, 4, kCt);
    REQUIRE(st.capture(ha, "x") == KvFlashPreambleStore::Stored);   // a is the LRU
    REQUIRE(st.capture(hb, "x") == KvFlashPreambleStore::Stored);
    REQUIRE(st.acquire(ha.back(), "x"));                                  // a is bound to a slot
    REQUIRE(st.capture(kvflash_preamble_hashes(c, 4, kCt), "x") == KvFlashPreambleStore::Stored);
    CHECK(st.find(ha.back(), "x") != nullptr);                            // the older but protected entry stays
    CHECK(st.find(hb.back(), "x") == nullptr);                            // the unprotected one goes
    CHECK(st.release(ha.back(), "x"));
    CHECK(!st.release(ha.back(), "x"));                                   // refcount never goes below zero
    CHECK(!st.acquire(12345, "x"));                                       // unknown hash
}

TEST_CASE(KvflashPreambleStoreFixture, protected_entries_exhaust_the_budget_and_decline_new_captures) {
    const int64_t one = 4 * kChunkB + kSlabB;
    KvFlashPreambleStore st(3 * one, kSlabB, kChunkB);
    std::vector<std::vector<uint64_t>> hs;
    for (int t = 1; t <= 3; t++) {
        hs.push_back(kvflash_preamble_hashes(prompt(400, 0, t), 4, kCt));
        REQUIRE(st.capture(hs.back(), "x") == KvFlashPreambleStore::Stored);
        REQUIRE(st.acquire(hs.back().back(), "x"));
    }
    CHECK(st.protected_bytes() == 3 * one);
    CHECK(st.free_for_capture() == 0);
    const auto d = kvflash_preamble_hashes(prompt(400, 0, 4), 4, kCt);
    CHECK(st.capture(d, "x") == KvFlashPreambleStore::Declined);
    CHECK(st.count() == 3);                                          // nothing was evicted for a capture that could not fit
    CHECK(st.declined() == 1);
    // one slot lets go: the next capture fits again
    REQUIRE(st.release(hs[1].back(), "x"));
    CHECK(st.capture(d, "x") == KvFlashPreambleStore::Stored);
    CHECK(st.find(hs[1].back(), "x") == nullptr);
}

TEST_CASE(KvflashPreambleStoreFixture, budget_below_one_slab_declines_every_capture) {
    KvFlashPreambleStore st(kSlabB - 1, kSlabB, kChunkB);
    CHECK(st.capture(kvflash_preamble_hashes(prompt(400, 0, 1), 4, kCt), "x") == KvFlashPreambleStore::Declined);
    CHECK(st.count() == 0);
    CHECK(st.used_bytes() == 0);
    // and the decision does not ask for a capture it knows is refused
    const auto a = prompt(400, 400, 1);
    const int64_t kmax = kvflash_capture_bound(256, 1, 4, 2, 24, st.free_for_capture(), kSlabB, kChunkB);
    CHECK(kmax == 0);
    CHECK(kvflash_preamble_decide(st, a, 400, kCt, "x", (int)kmax).kind == KvFlashPreambleDecision::Cold);
}

TEST_CASE(KvflashPreambleStoreFixture, duplicate_capture_is_idempotent) {
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    const auto h = kvflash_preamble_hashes(prompt(400, 0, 1), 4, kCt);
    CHECK(st.capture(h, "x") == KvFlashPreambleStore::Stored);
    const int64_t used = st.used_bytes();
    CHECK(st.capture(h, "x") == KvFlashPreambleStore::Exists);
    CHECK(st.used_bytes() == used);
    CHECK(st.count() == 1);
}

// ---- capture bound (P2-04) ---------------------------------------------------
TEST_CASE(KvflashPreambleStoreFixture, capture_bound_is_the_pin_range_guard_and_the_recall_budget) {
    // sink 1 + tail 4 + k + reserve 2 + recall 24 + 2 spare <= Q = 256  ->  k <= 223
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 24, int64_t(64) << 30, kSlabB, kChunkB) == 223);
    // Q = 256, no recall: k <= 247
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 0, int64_t(64) << 30, kSlabB, kChunkB) == 247);
}

TEST_CASE(KvflashPreambleStoreFixture, capture_bound_is_capped_by_the_reserve_rows) {
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 24, int64_t(64) << 30, kSlabB, kChunkB, 96) == 96);
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 24, int64_t(64) << 30, kSlabB, kChunkB, 500) == 223);   // the pool binds first
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 24, int64_t(64) << 30, kSlabB, kChunkB, 0) == 0);
}

TEST_CASE(KvflashPreambleStoreFixture, store_with_reserve_zero_is_refused) {
    CHECK(!kvflash_preamble_store_refusal(6144).size());
    const std::string msg = kvflash_preamble_store_refusal(0);
    CHECK(msg.find("LUCE_KVFLASH_PREAMBLE_ROWS") != std::string::npos);
    CHECK(!kvflash_preamble_store_refusal(-1).empty());
}

TEST_CASE(KvflashPreambleStoreFixture, pin_k_is_the_capture_k_of_an_empty_store) {
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    for (int kmax : {0, 1, 7, 8, 9, 100})
        for (int e_role : {0, 100, 512, 700, 1500, 2600}) {
            const auto a = prompt(e_role + 40, e_role, 1);
            const auto d = kvflash_preamble_decide(st, a, e_role, kCt, "x", kmax, 512);
            CHECK(kvflash_preamble_pin_k(e_role, kCt, 512, kmax) == (d.kind == KvFlashPreambleDecision::Capture ? d.k : 0));
        }
}

TEST_CASE(KvflashPreambleStoreFixture, capture_bound_small_q_and_prefix_larger_than_q) {
    CHECK(kvflash_capture_bound(16, 1, 4, 2, 0, int64_t(64) << 30, kSlabB, kChunkB) == 7);
    CHECK(kvflash_capture_bound(8, 1, 4, 2, 0, int64_t(64) << 30, kSlabB, kChunkB) == 0);    // nothing left after protection
    CHECK(kvflash_capture_bound(3, 1, 4, 2, 24, int64_t(64) << 30, kSlabB, kChunkB) == 0);   // Q below the guard itself
    // a system+tools prefix of 40 chunks in a Q = 16 slot: the bound caps capture at 7, never 40
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    const auto a = prompt(2600, 2600, 1);
    const auto d = kvflash_preamble_decide(st, a, 2560, kCt, "x",
                                           kvflash_capture_bound(16, 1, 4, 2, 0, st.free_for_capture(), kSlabB, kChunkB));
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 7);
}

TEST_CASE(KvflashPreambleStoreFixture, capture_bound_is_limited_by_the_store_budget) {
    // 10 chunks + slab fit exactly; one byte less fits 9
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 0, 10 * kChunkB + kSlabB, kSlabB, kChunkB) == 10);
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 0, 10 * kChunkB + kSlabB - 1, kSlabB, kChunkB) == 9);
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 0, kSlabB - 1, kSlabB, kChunkB) == 0);        // below one slab
    CHECK(kvflash_capture_bound(256, 1, 4, 2, 0, -5, kSlabB, kChunkB) == 0);                // negative free space
}

TEST_CASE(KvflashPreambleStoreFixture, store_free_for_capture_excludes_protected_but_not_evictable) {
    const int64_t one = 4 * kChunkB + kSlabB;
    KvFlashPreambleStore st(3 * one, kSlabB, kChunkB);
    const auto h1 = kvflash_preamble_hashes(prompt(400, 0, 1), 4, kCt), h2 = kvflash_preamble_hashes(prompt(400, 0, 2), 4, kCt);
    REQUIRE(st.capture(h1, "x") == KvFlashPreambleStore::Stored);
    REQUIRE(st.capture(h2, "x") == KvFlashPreambleStore::Stored);
    REQUIRE(st.acquire(h1.back(), "x"));
    CHECK(st.free_for_capture() == 2 * one);   // h2 is unprotected, so evictable room counts
}

// ---- the cut grid: E_c and k are multiples of the prefill slice -----------------------------------------
// A cut off the slice grid changes the prefill's segment shapes and so the float rounding (S0: max logit diff 3.08 at
// E_c 384). On the grid the legacy prefill, the capture cut and an attach are one computation.
constexpr int kSlice = 512;

TEST_CASE(KvflashPreambleStoreFixture, ec_rounds_down_to_the_slice_when_asked) {
    CHECK(kvflash_preamble_ec(1143, kCt, kSlice) == 1024);
    CHECK(kvflash_preamble_ec(1024, kCt, kSlice) == 1024);
    CHECK(kvflash_preamble_ec(1023, kCt, kSlice) == 512);
    CHECK(kvflash_preamble_ec(511, kCt, kSlice) == 0);
    CHECK(kvflash_preamble_ec(1143, kCt, 0) == 1088);               // no slice: today's chunk floor
    CHECK(kvflash_preamble_ec(1143, kCt, kCt) == 1088);
}

TEST_CASE(KvflashPreambleStoreFixture, decide_captures_and_attaches_on_the_slice_grid_only) {
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    const auto a = prompt(1700, 1143, 1);
    auto d = kvflash_preamble_decide(st, a, 1143, kCt, "x", 26, kSlice);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 16);                                               // 1024 tokens, not 17 chunks
    REQUIRE(st.capture(kvflash_preamble_hashes(a, 16, kCt), "x") == KvFlashPreambleStore::Stored);
    d = kvflash_preamble_decide(st, a, 1143, kCt, "x", 26, kSlice);
    CHECK(d.kind == KvFlashPreambleDecision::Attach);
    CHECK(d.k == 16);
    // a session sharing 700 tokens (10 chunks) with the entry: the capture cut is 8 chunks (512), not 10
    const auto b = prompt(1700, 700, 2);
    d = kvflash_preamble_decide(st, b, 1143, kCt, "x", 26, kSlice);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 8);
    // below one slice of shared tokens there is no capture
    const auto c = prompt(1700, 500, 3);
    d = kvflash_preamble_decide(st, c, 1143, kCt, "x", 26, kSlice);
    CHECK(d.kind == KvFlashPreambleDecision::Cold);
    CHECK(d.k == 0);
    CHECK(kvflash_preamble_decide(st, prompt(1700, 1700, 4), 500, kCt, "x", 26, kSlice).kind == KvFlashPreambleDecision::Cold);   // E_role < 512
    // the slot's bound rounds down to the grid too
    KvFlashPreambleStore fresh(int64_t(2) << 30, kSlabB, kChunkB);
    d = kvflash_preamble_decide(fresh, a, 1143, kCt, "x", 12, kSlice);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 8);
    CHECK(kvflash_preamble_decide(fresh, a, 1143, kCt, "x", 7, kSlice).kind == KvFlashPreambleDecision::Cold);
}
