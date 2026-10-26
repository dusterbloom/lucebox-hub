// KVFlash 2.0 sessions P1b: every engine path keeps the preamble pins (P2-02), CPU only. The allocator
// (KvFlashSlotKv over slot pagers) owns the re-pin through the binder it is given, so these tests never call
// the re-pin themselves: they drive the operations each engine path performs and read pins and residency.
//
//   capture-on-miss   cold prefill to the cut, capture (binds + pins), the rest of the prompt
//   attach            attach blob + adopt, resume admit at 64k, prefill the suffix
//   restart           adopt of a full session blob, then the bound session's rebind
//   rung rewind       truncate_parked to a checkpoint
//   mode-4 rewind     rewind_prefill of a live turn to R
//   failed admission  a resume or cold admit the host ledger refuses
//   cold slot reuse   a bound parked session's slot taken by a new session
//   recompute/clear   clear / release / drop of the slot
// and the admit through a recording engine: a second session with the same system + tools attaches and
// prefills only from E_c.

#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "../../src/common/concurrency/kvflash_slot_kv.h"
#include "../../src/common/concurrency/seq_slot_manager.h"
#include "../../src/common/kvflash_arena.h"
#include "../../src/common/kvflash_preamble_binder.h"

#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPreamblePathsFixture {};

constexpr int kN = 2, kQ = 12, kChunk = 64, kK = 3, kHead = kK * kChunk + 10;   // kHead: the shared system+tools head of prompt_of
constexpr int64_t kMaxCtx = 4096, kChunkBytes = 1000;

struct Slot : KvFlashSlotPager {
    Slot() { set_pinned_host_backing(pager_pinned_host_available()); }
};

struct World {
    KvFlashArena arena;
    std::vector<std::unique_ptr<Slot>> s;
    KvFlashHostLedger ledger;
    std::unique_ptr<KvFlashSlotKv> kv;
    KvFlashPreambleStore store{1 << 30, 100, 0};
    KvFlashPreambleBinder binder{store, "exec", kN};
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
        kv->set_binder(&binder);                                  // the production wiring, nothing else
    }
};

std::vector<int32_t> prompt_of(int n, int salt = 0) {
    std::vector<int32_t> v((size_t)n);
    for (int i = 0; i < n; i++) v[(size_t)i] = i < kHead ? i * 3 + 5 : 100000 + salt * 1000 + i;   // a shared head, a tail per session
    return v;
}

SeqEngine::AdmitResult admit_cold(World & w, SeqSlotManager & mgr, uint64_t id, const std::vector<int32_t> & prompt, int slot = -1) {
    w.kv->set_next_total_tokens((int64_t)prompt.size() + 60);
    if (slot >= 0) w.kv->set_next_slot(slot);
    return mgr.admit(id, prompt, SamplerCfg{});
}

// Prefill `n` more tokens in the pool's own cuts, completing the page-outs the full slice queues.
bool prefill(World & w, SeqSlotManager & mgr, int slot, int n) {
    while (n > 0) {
        const int c = std::min(n, kChunk);
        auto r = mgr.append_prefill(slot, c);
        if (!r.ok) { if (!w.arena.complete_all()) return false; r = mgr.append_prefill(slot, c); if (!r.ok) return false; }
        n -= c;
    }
    return true;
}

void must(bool ok) { if (!ok) throw std::runtime_error("paths test precondition failed"); }

// Pins and residency of chunks [0, k) on `slot`, and what recall reads.
bool bound(World & w, int slot, int k) {
    bool ok = w.binder.binding(slot).k == k && w.binder.pins(slot) == k;
    for (int c = 0; c < k; c++) ok = ok && w.s[(size_t)slot]->is_pinned(c) && w.s[(size_t)slot]->is_resident(c);
    return ok;
}
bool unbound(World & w, int slot, int k_was) {
    bool ok = w.binder.binding(slot).k == 0 && w.binder.pins(slot) == 0;
    for (int c = 0; c < k_was; c++) ok = ok && !w.s[(size_t)slot]->is_pinned(c);
    return ok;
}
// Later pressure: fill the slice past its pool; the bound chunks must survive every eviction.
bool pressure(World & w, SeqSlotManager & mgr, int slot, int k) {
    const int len = mgr.slot(slot).cur_pos;
    return prefill(w, mgr, slot, (kQ + 4) * kChunk - len) && bound(w, slot, k);
}

struct Captured { uint64_t hash = 0; int slot = -1; std::vector<int32_t> prompt; };
// A session that prefilled its first kK chunks, captured the entry and went on to a full prompt.
Captured capture_on_miss(World & w, SeqSlotManager & mgr, uint64_t id, int total, int salt = 0) {
    Captured c;
    c.prompt = prompt_of(total, salt);
    const auto r = admit_cold(w, mgr, id, c.prompt);
    must(r.status == SeqEngine::AdmitResult::Status::admitted);
    c.slot = r.slot;
    must(prefill(w, mgr, c.slot, kK * kChunk));
    const auto cap = kvflash_preamble_capture(w.binder, c.slot, w.kv->pager(c.slot), c.prompt, kK, kK, {1, 2, 3});
    must(cap.kind == KvFlashPreambleCapture::Stored);
    c.hash = cap.hash;
    return c;
}
}  // namespace

TEST_CASE(KvflashPreamblePathsFixture, capture_on_miss_pins_the_prefix_and_it_survives_the_rest_of_the_prompt) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured c = capture_on_miss(w, mgr, 1, 3000);
    CHECK(bound(w, c.slot, kK));
    CHECK(w.store.find(c.hash, "exec")->refcount == 1);
    CHECK(pressure(w, mgr, c.slot, kK));
}

TEST_CASE(KvflashPreamblePathsFixture, attach_pins_the_prefix_resumes_at_64k_and_it_survives_the_suffix) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    const int b_slot = 1 - a.slot;
    std::vector<uint8_t> slab;
    REQUIRE(kvflash_preamble_attach(*w.kv, w.binder, b_slot, 0x77, a.hash, kK, slab));
    CHECK(slab == std::vector<uint8_t>({1, 2, 3}));
    CHECK(bound(w, b_slot, kK));                                 // the adopt itself left it pinned
    CHECK(w.store.find(a.hash, "exec")->refcount == 2);
    const auto p = prompt_of(2600, 1);
    w.kv->set_next_total_tokens((int64_t)p.size() + 60);
    w.kv->set_next_resume(b_slot, kK * kChunk);
    const auto r = mgr.admit_at(b_slot, 2, p, SamplerCfg{}, kK * kChunk);
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    CHECK(mgr.slot(b_slot).cur_pos == kK * kChunk);              // prefills only from E_c
    CHECK(bound(w, b_slot, kK));
    CHECK(pressure(w, mgr, b_slot, kK));
}

TEST_CASE(KvflashPreamblePathsFixture, an_attach_the_pager_cannot_pin_falls_back_to_cold_and_leaves_the_slot_free) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    // k = 9: sink 1 + tail 2 + 9 + 2 > 12, the pager refuses the span
    KvFlashPreambleStore & st = w.store;
    const auto ids = prompt_of(12 * kChunk);
    const auto big = kvflash_preamble_hashes(ids, 9, kChunk);
    REQUIRE(st.capture(big, "exec") == KvFlashPreambleStore::Stored);
    std::vector<uint8_t> hdr = w.kv->pager(a.slot).blob_header(9);   // the header of the captured pager says 3 chunks: patch the count
    hdr[8] = 9; hdr[9] = hdr[10] = hdr[11] = 0;
    REQUIRE(st.set_payload(big.back(), "exec", hdr, {}, {1}));
    std::vector<uint8_t> slab;
    CHECK(!kvflash_preamble_attach(*w.kv, w.binder, 1 - a.slot, 0x78, big.back(), 9, slab));
    CHECK(!w.kv->parked(1 - a.slot));
    CHECK(w.s[(size_t)(1 - a.slot)]->n_chunks() == 0);
    CHECK(unbound(w, 1 - a.slot, 9));
    CHECK(st.find(big.back(), "exec")->refcount == 0);
    CHECK(w.ledger.reservation_count() == 1);                    // only the live capturing session's
}

TEST_CASE(KvflashPreamblePathsFixture, restart_adopts_a_full_session_and_the_rebind_pins_it_again) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    REQUIRE(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    CHECK(bound(w, a.slot, kK));                                 // parked-gpu: still bound
    const std::vector<uint8_t> blob = w.s[(size_t)a.slot]->serialize();
    // "restart": the pagers are fresh, the store is adopted from disk, the binding comes from meta.json
    World w2;
    REQUIRE(w2.store.capture(kvflash_preamble_hashes(hist, kK, kChunk), "exec") == KvFlashPreambleStore::Stored);
    REQUIRE(w2.kv->adopt_parked(a.slot, 0x99, 1200, blob.data(), blob.size()) == PagedKvStatus::Ok);
    CHECK(unbound(w2, a.slot, kK));                              // deserialize cleared the pins, nothing bound yet
    CHECK(kvflash_preamble_rebind(*w2.kv, w2.binder, a.slot, hist, a.hash, kK));
    CHECK(bound(w2, a.slot, kK));
    CHECK(w2.store.find(a.hash, "exec")->refcount == 1);
    // a binding that names another prefix than the history's is not honoured
    World w3;
    REQUIRE(w3.store.capture(kvflash_preamble_hashes(hist, kK, kChunk), "exec") == KvFlashPreambleStore::Stored);
    REQUIRE(w3.kv->adopt_parked(a.slot, 0x99, 1200, blob.data(), blob.size()) == PagedKvStatus::Ok);
    auto other = hist;
    other[5] ^= 1;
    CHECK(!kvflash_preamble_rebind(*w3.kv, w3.binder, a.slot, other, a.hash, kK));
    CHECK(unbound(w3, a.slot, kK));
    // and one whose entry the store no longer holds binds nothing
    World w4;
    REQUIRE(w4.kv->adopt_parked(a.slot, 0x99, 1200, blob.data(), blob.size()) == PagedKvStatus::Ok);
    CHECK(!kvflash_preamble_rebind(*w4.kv, w4.binder, a.slot, hist, a.hash, kK));
    CHECK(unbound(w4, a.slot, kK));
}

TEST_CASE(KvflashPreamblePathsFixture, a_rung_rewind_of_a_parked_session_keeps_the_binding_down_to_the_prefix_and_releases_below) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    REQUIRE(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    REQUIRE(w.kv->truncate_parked(a.slot, 700));
    CHECK(bound(w, a.slot, kK));
    REQUIRE(w.kv->truncate_parked(a.slot, kK * kChunk));         // exactly the prefix: still held
    CHECK(bound(w, a.slot, kK));
    REQUIRE(w.kv->truncate_parked(a.slot, kChunk + 10));         // below the attached prefix
    CHECK(unbound(w, a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 0);
}

TEST_CASE(KvflashPreamblePathsFixture, a_mode_4_rewind_keeps_the_pins_and_the_recall_count_and_below_the_prefix_releases) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    REQUIRE(prefill(w, mgr, a.slot, 1500 - kK * kChunk));
    REQUIRE(mgr.rewind_prefill(a.slot, 900));                    // R: the first pass is rewound
    CHECK(bound(w, a.slot, kK));                                 // what kvflash_prefill_apply_on reads is still k
    CHECK(pressure(w, mgr, a.slot, kK));
    REQUIRE(mgr.rewind_prefill(a.slot, kChunk));                 // a rewind below the attached prefix
    CHECK(unbound(w, a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 0);
}

TEST_CASE(KvflashPreamblePathsFixture, a_failed_admission_leaves_no_stale_binding_or_pins) {
    World w(/*host budget: 14 chunks of backing*/ 14 * kChunkBytes);
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 640);
    REQUIRE(prefill(w, mgr, a.slot, 640 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    // a resume the host ledger refuses: the session stays parked and bound as it was
    w.kv->set_next_total_tokens(20 * kChunk);
    w.kv->set_next_resume(a.slot, 640);
    CHECK(mgr.admit_at(a.slot, 2, prompt_of(900), SamplerCfg{}, 640).status != SeqEngine::AdmitResult::Status::admitted);
    CHECK(w.kv->parked(a.slot));
    CHECK(bound(w, a.slot, kK));
    // a cold admit into the same slot that the ledger refuses: the parked session was dropped, its binding with it
    w.kv->set_next_total_tokens(20 * kChunk);
    w.kv->set_next_slot(a.slot);
    CHECK(mgr.admit_at(a.slot, 3, prompt_of(900), SamplerCfg{}, 0).status != SeqEngine::AdmitResult::Status::admitted);
    CHECK(!w.kv->parked(a.slot));
    CHECK(unbound(w, a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 0);
}

TEST_CASE(KvflashPreamblePathsFixture, cold_slot_reuse_releases_the_old_binding_and_the_new_session_binds_on_its_own) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 1200);
    REQUIRE(prefill(w, mgr, a.slot, 1200 - kK * kChunk));
    mgr.commit_prefill(a.slot);
    std::vector<int32_t> hist;
    REQUIRE(mgr.park(a.slot, hist));
    const auto r = admit_cold(w, mgr, 2, prompt_of(2000, 4), a.slot);   // the slot's session is replaced
    REQUIRE(r.status == SeqEngine::AdmitResult::Status::admitted);
    REQUIRE(r.slot == a.slot);
    CHECK(unbound(w, a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 0);
    REQUIRE(prefill(w, mgr, a.slot, kK * kChunk));
    const auto cap = kvflash_preamble_capture(w.binder, a.slot, w.kv->pager(a.slot), prompt_of(2000, 4), kK, kK, {1});
    CHECK(cap.kind == KvFlashPreambleCapture::Exists);           // same head: the entry exists, the new session binds to it
    CHECK(bound(w, a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 1);
    CHECK(pressure(w, mgr, a.slot, kK));
}

TEST_CASE(KvflashPreamblePathsFixture, clear_release_and_drop_give_the_refcount_back_across_two_slots_sharing_one_entry) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    const Captured a = capture_on_miss(w, mgr, 1, 3000);
    std::vector<uint8_t> slab;
    REQUIRE(kvflash_preamble_attach(*w.kv, w.binder, 1 - a.slot, 0x77, a.hash, kK, slab));
    CHECK(w.store.find(a.hash, "exec")->refcount == 2);
    // the live slot is cleared (a recompute starts over): its binding goes, the parked slot's stays
    CHECK(w.kv->clear(mgr.slot(a.slot).handle) == PagedKvStatus::Ok);
    CHECK(unbound(w, a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 1);
    mgr.retire(a.slot);                                          // release: reset + ledger release
    CHECK(unbound(w, a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 1);
    CHECK(bound(w, 1 - a.slot, kK));
    w.kv->drop_parked(1 - a.slot);                               // evict / drop of the other
    CHECK(unbound(w, 1 - a.slot, kK));
    CHECK(w.store.find(a.hash, "exec")->refcount == 0);
}

// ---- the admit through a recording engine (the style of the scheduler's MetaRecordingEngine) ----------------
namespace {
// What Qwen35SeqEngine::admit_bounded does about the preamble, on the CPU allocator: decide, attach or cold
// admit, capture at the cut. It records where each prefill started.
struct RecordingEngine {
    World & w;
    SeqSlotManager & mgr;
    struct Admit { int slot = -1; int prefill_from = -1; KvFlashPreambleDecision::Kind kind = KvFlashPreambleDecision::Cold; };
    std::vector<Admit> log;
    int e_role_of(const std::vector<int32_t> & ids) const { (void)ids; return kHead; }   // the system block's end in prompt_of
    Admit admit(uint64_t id, const std::vector<int32_t> & prompt) {
        Admit a;
        const int e_role = e_role_of(prompt);
        const auto d = kvflash_preamble_decide(w.store, prompt, e_role, kChunk, "exec", /*k_max=*/kK);
        a.kind = d.kind;
        int slot = 0;
        while (slot < kN && (mgr.is_active(slot) || w.kv->parked(slot))) slot++;
        if (slot == kN) return a;
        a.slot = slot;
        if (d.kind == KvFlashPreambleDecision::Attach) {
            std::vector<uint8_t> slab;
            const auto hashes = kvflash_preamble_hashes(prompt, d.k, kChunk);
            if (kvflash_preamble_attach(*w.kv, w.binder, slot, 0x1000 + (uint64_t)slot, hashes.back(), d.k, slab)) {
                w.kv->set_next_total_tokens((int64_t)prompt.size() + 60);
                w.kv->set_next_resume(slot, (uint32_t)d.k * kChunk);
                const auto r = mgr.admit_at(slot, id, prompt, SamplerCfg{}, d.k * kChunk);
                if (r.status == SeqEngine::AdmitResult::Status::admitted) { a.prefill_from = d.k * kChunk; log.push_back(a); return a; }
                w.kv->drop_parked(slot);
            }
        }
        const auto r = admit_cold(w, mgr, id, prompt, slot);
        a.slot = r.slot; a.prefill_from = 0;
        if (d.kind == KvFlashPreambleDecision::Capture) {   // the capture cut at 64k: prefill to it, capture, go on
            prefill(w, mgr, r.slot, d.k * kChunk);
            kvflash_preamble_capture(w.binder, r.slot, w.kv->pager(r.slot), prompt, d.k, d.k, {1, 2, 3});
        }
        log.push_back(a);
        return a;
    }
};
}  // namespace

TEST_CASE(KvflashPreamblePathsFixture, a_second_session_with_the_same_system_and_tools_attaches_and_prefills_only_from_e_c) {
    World w;
    SeqSlotManager mgr(*w.kv, (int)kMaxCtx);
    RecordingEngine eng{w, mgr, {}};
    const auto first = eng.admit(1, prompt_of(1500, 1));         // E_role = 202 -> E_c = 192: three chunks, captured on miss
    CHECK(first.kind == KvFlashPreambleDecision::Capture);
    CHECK(first.prefill_from == 0);
    CHECK(w.store.count() == 1);
    const auto second = eng.admit(2, prompt_of(1700, 2));        // same head, another suffix
    CHECK(second.kind == KvFlashPreambleDecision::Attach);
    CHECK(second.prefill_from == kK * kChunk);                    // only from E_c
    CHECK(second.slot != first.slot);
    CHECK(mgr.slot(second.slot).cur_pos == kK * kChunk);
    CHECK(w.binder.binding(second.slot).k == kK);
    CHECK(w.binder.pins(second.slot) == kK);
    CHECK(w.store.find(w.binder.binding(first.slot).hash, "exec")->refcount == 2);
    // a session with another system prompt shares nothing: cold, and it captures its own entry (store budget allows)
    auto other = prompt_of(1500, 3);
    for (int i = 0; i < 20; i++) other[(size_t)i] += 77;
    mgr.retire(first.slot);                                      // a free slot for it
    const auto third = eng.admit(3, other);
    CHECK(third.kind == KvFlashPreambleDecision::Capture);
    CHECK(third.prefill_from == 0);
    CHECK(w.store.count() == 2);
}
