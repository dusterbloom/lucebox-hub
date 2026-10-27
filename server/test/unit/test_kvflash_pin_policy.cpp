// KVFlash 2.0 sessions: the pin policy (amendment 6). k is a function of the preamble alone: min(E_c, ROWS)/ct, whatever the
// replay reserve, the recall chunks or the history length. Pure and CPU only (the review's reproduction, no tensors).

#include "CppUnitTestFramework.hpp"

#include "kvflash_preamble_binder.h"
#include "kvflash_prefill_reselect.h"
#include "kvflash_slot_pager.h"
#include "kvflash_slot_turn.h"

#include <climits>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPinPolicyFixture {};
constexpr int ct = 64, Q = 352, rows = 6144, cut = 512, IM = 7, SYS = 8, USR = 9;

std::vector<int32_t> long_prompt() {
    std::vector<int32_t> p(30128, 1000);
    p[0] = IM; p[1] = SYS; p[6144] = IM; p[6145] = USR; p[30000] = IM; p[30001] = USR; p[30120] = IM; p[30121] = 10;
    return p;
}
int derive(const std::vector<int32_t> & ids) { return kvflash_derive_pin_k(ids, IM, SYS, ct, cut, rows); }
int replay_reserve(const std::vector<int32_t> & ids) { return kvflash_slot_turn_plan(ids, 0, IM, USR, ct, Q, cut, 64, true).reserve_chunks; }
std::vector<int> pinned(const KvFlashSlotPager & p) { std::vector<int> v; for (int c = 0; c < Q; c++) if (p.is_pinned(c)) v.push_back(c); return v; }
std::vector<int> resident(const KvFlashSlotPager & p) { std::vector<int> v; for (int c = 0; c < Q; c++) if (p.is_resident(c)) v.push_back(c); return v; }
KvFlashSlotPager fresh() {
    KvFlashSlotPager p;
    p.attach_slice(0, 1, Q, KvFlashConfig{}, {}, {});
    p.set_deferred_free(false);
    return p;
}
}  // namespace

TEST_CASE(KvflashPinPolicyFixture, k_ignores_the_replay_reserve_the_question_length_and_the_generated_reply) {
    const auto initial = long_prompt();
    auto longer = initial;
    longer.insert(longer.begin() + 30120, 2432, 1000);
    auto history = initial;
    history.insert(history.end(), 4096, 1000);
    // the real turn planner's replay reserve moves with the tail (3, 41, 67 chunks), k does not
    CHECK(replay_reserve(initial) == 3);
    CHECK(replay_reserve(longer) == 41);
    CHECK(replay_reserve(history) == 67);
    CHECK(derive(initial) == 96);
    CHECK(derive(longer) == 96);
    CHECK(derive(history) == 96);
}

TEST_CASE(KvflashPinPolicyFixture, k_is_capped_by_the_rows_and_floors_to_the_cut_grid) {
    auto p = long_prompt();
    CHECK(kvflash_derive_pin_k(p, IM, SYS, ct, cut, 3000) == 40);   // 3000 / 64 = 46 chunks, 5 cuts of 8 chunks
    CHECK(kvflash_derive_pin_k(p, IM, SYS, ct, cut, 0) == 0);
    p.assign(2000, 1000); p[0] = IM; p[1] = SYS; p[700] = IM; p[701] = USR;
    CHECK(derive(p) == 8);                                          // E_role 700 -> E_c 512
}

TEST_CASE(KvflashPinPolicyFixture, a_restore_with_the_entry_without_it_and_with_the_store_off_holds_the_same_pins_and_residency) {
    const auto history = [] { auto h = long_prompt(); h.insert(h.end(), 4096, 1000); return h; }();
    const int k = derive(history);
    KvFlashSlotPager before = fresh();
    REQUIRE(before.alloc_span(0, k * ct));
    int bound_k = k;
    REQUIRE(kvflash_repin_preamble(before, bound_k).accepted == k);
    REQUIRE(before.alloc_span(k * ct, (int)history.size() - k * ct));
    const auto blob = before.serialize();
    KvFlashPreambleStore st(INT64_MAX / 2, 1, 1);
    const auto hashes = kvflash_preamble_hashes(history, k, ct);
    REQUIRE(st.capture(hashes, "x") == KvFlashPreambleStore::Stored);
    KvFlashPreambleBinder bd(st, "x", 1);
    KvFlashSlotPager with_entry = fresh(), no_entry = fresh(), store_off = fresh();
    for (auto * p : {&with_entry, &no_entry, &store_off}) { REQUIRE(p->deserialize(blob.data(), blob.size())); p->set_deferred_free(false); }
    REQUIRE(bd.bind(0, hashes.back(), k));
    CHECK(bd.ensure(0, with_entry) == k);                           // the entry present: bound at the derived k
    int kn = derive(history), ks = derive(history);                 // absent / store off: the same pure function
    CHECK(kvflash_repin_preamble(no_entry, kn).accepted == k);
    CHECK(kvflash_repin_preamble(store_off, ks).accepted == k);
    for (auto * p : {&with_entry, &no_entry, &store_off}) p->reselect_recent();
    CHECK(pinned(with_entry) == pinned(no_entry));
    CHECK(pinned(no_entry) == pinned(store_off));
    CHECK(resident(with_entry) == resident(no_entry));
    CHECK(resident(no_entry) == resident(store_off));
    CHECK(with_entry.is_resident(32));                              // the chunk the old cap (32) let go
}

TEST_CASE(KvflashPinPolicyFixture, the_fallback_pins_follow_decline_rewind_reuse_with_exact_indices) {
    const auto prompt = long_prompt();
    KvFlashSlotPager p = fresh();
    int k = derive(prompt);                                         // a capture the store declined: the pins are the same k
    REQUIRE(p.alloc_span(0, 7000));
    CHECK(kvflash_repin_preamble(p, k).accepted == 96);
    std::vector<int> want;
    for (int c = 0; c < 96; c++) want.push_back(c);
    CHECK(pinned(p) == want);
    // a rewind that keeps the prefix keeps the pins; one below it releases them (k = 0, nothing left pinned)
    p.truncate((int64_t)96 * ct);
    CHECK(kvflash_repin_preamble(p, k).accepted == 96);
    CHECK(pinned(p) == want);
    p.truncate(10 * ct);
    CHECK(kvflash_repin_preamble(p, k).released);
    CHECK(k == 0);
    CHECK(pinned(p).empty());
    // the slot is reused: a shorter preamble pins exactly its own chunks, an absent one pins nothing
    p.reset();
    std::vector<int32_t> shorter(2000, 1000);
    shorter[0] = IM; shorter[1] = SYS; shorter[700] = IM; shorter[701] = USR;
    int k2 = derive(shorter);
    REQUIRE(p.alloc_span(0, 1500));
    CHECK(kvflash_repin_preamble(p, k2).accepted == 8);
    CHECK(pinned(p) == (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    p.reset();
    std::vector<int32_t> none(2000, 1000);
    int k3 = derive(none);
    CHECK(k3 == 0);
    REQUIRE(p.alloc_span(0, 1500));
    CHECK(kvflash_repin_preamble(p, k3).accepted == 0);
    CHECK(pinned(p).empty());
}

TEST_CASE(KvflashPinPolicyFixture, the_reserve_log_counts_every_slot_once) {
    // 2 slots x 6144 tokens x 98304 bytes/token = 1152 MiB; one slot is half of it
    CHECK(kvflash_preamble_reserve_mib(2, 6144, 98304) == 1152.0);
    CHECK(kvflash_preamble_reserve_mib(1, 6144, 98304) == 576.0);
    CHECK(kvflash_preamble_reserve_mib(2, 0, 98304) == 0.0);
}
