// KVFlash 2.0 sessions P1a: the preamble store core (SESSIONS_DESIGN.md 3.2), pure, no GPU.
// Chunk-prefix hashing, the boundary rule, refcount + LRU within a budget, and the capture bound (P2-04).

#include "CppUnitTestFramework.hpp"

#include "kvflash_preamble_store.h"
#include "kvflash_prefill_reselect.h"

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
    // B shares only 9 chunks with the entry of 17: k is B's own E_c (18), never the shared cut; the store's
    // contents decide attach versus capture, not k
    d = kvflash_preamble_decide(st, b, e_b, kCt, "x", 26);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 18);
    REQUIRE(st.capture(kvflash_preamble_hashes(b, 18, kCt), "x") == KvFlashPreambleStore::Stored);
    d = kvflash_preamble_decide(st, b, e_b, kCt, "x", 26);
    CHECK(d.kind == KvFlashPreambleDecision::Attach);
    CHECK(d.k == 18);
    const auto c = prompt(1400, 586, 3);
    d = kvflash_preamble_decide(st, c, 700, kCt, "x", 26);               // E_c = 10, an entry shares 9: still a capture at 10
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 10);
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

// ---- the reserve refusal ------------------------------------------------------
TEST_CASE(KvflashPreambleStoreFixture, store_with_reserve_zero_is_refused) {
    CHECK(!kvflash_preamble_store_refusal(6144).size());
    const std::string msg = kvflash_preamble_store_refusal(0);
    CHECK(msg.find("LUCE_KVFLASH_PREAMBLE_ROWS") != std::string::npos);
    CHECK(!kvflash_preamble_store_refusal(-1).empty());
}

TEST_CASE(KvflashPreambleStoreFixture, pin_k_is_the_decide_k_on_an_empty_store_and_on_one_holding_another_entry) {
    KvFlashPreambleStore st(int64_t(2) << 30, kSlabB, kChunkB);
    // the S0 shape: A1's entry (16 chunks) is in the store, A2 shares ~770 tokens with it and has its own E_role
    const auto a1 = prompt(1378, 1046, 1);
    REQUIRE(st.capture(kvflash_preamble_hashes(a1, 16, kCt), "x") == KvFlashPreambleStore::Stored);
    const auto a2 = prompt(1597, 770, 2);
    const auto d2 = kvflash_preamble_decide(st, a2, 1254, kCt, "x", 96, 512);
    CHECK(d2.kind == KvFlashPreambleDecision::Capture);
    CHECK(d2.k == kvflash_preamble_pin_k(1254, kCt, 512, 96));
    CHECK(d2.k == 16);
    const auto d1 = kvflash_preamble_decide(st, a1, 1041, kCt, "x", 96, 512);
    CHECK(d1.kind == KvFlashPreambleDecision::Attach);
    CHECK(d1.k == kvflash_preamble_pin_k(1041, kCt, 512, 96));
    for (int kmax : {0, 1, 7, 8, 9, 100})
        for (int e_role : {0, 100, 512, 700, 1500, 2600}) {
            const auto a = prompt(e_role + 40, e_role, 1);
            const auto d = kvflash_preamble_decide(st, a, e_role, kCt, "x", kmax, 512);
            CHECK(kvflash_preamble_pin_k(e_role, kCt, 512, kmax) == d.k);
        }
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
    // sessions sharing 700 or 500 tokens with the entry still capture their own 16 chunks: k ignores the store
    for (int shared : {700, 500}) {
        d = kvflash_preamble_decide(st, prompt(1700, shared, 2), 1143, kCt, "x", 26, kSlice);
        CHECK(d.kind == KvFlashPreambleDecision::Capture);
        CHECK(d.k == 16);
    }
    CHECK(kvflash_preamble_decide(st, prompt(1700, 1700, 4), 500, kCt, "x", 26, kSlice).kind == KvFlashPreambleDecision::Cold);   // E_role < 512
    // the slot's bound rounds down to the grid too
    KvFlashPreambleStore fresh(int64_t(2) << 30, kSlabB, kChunkB);
    d = kvflash_preamble_decide(fresh, a, 1143, kCt, "x", 12, kSlice);
    CHECK(d.kind == KvFlashPreambleDecision::Capture);
    CHECK(d.k == 8);
    CHECK(kvflash_preamble_decide(fresh, a, 1143, kCt, "x", 7, kSlice).kind == KvFlashPreambleDecision::Cold);
}
