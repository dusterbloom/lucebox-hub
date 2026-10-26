// KVFlash 2.0 sessions P1b: the preamble store's engine-side pieces, CPU only: entry payloads and eviction
// reports, the exec_id, the binder (refcount per slot, the pin count recall reads), blob assembly for an
// attach, the bounded capture, persistence under preambles/ with the execution-identity check, and the
// binding fields of a session's meta.json.

#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "kvflash_preamble_binder.h"
#include "kvflash_session_store.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPreambleBinderFixture {};

constexpr int kCt = 64;
constexpr int64_t kSlab = 1000, kChunk = 100, kBig = 1 << 30;

std::vector<int32_t> ids_of(int n, int salt = 0) {
    std::vector<int32_t> v((size_t)n);
    for (int i = 0; i < n; i++) v[(size_t)i] = i * 7 + 1 + salt;
    return v;
}
std::vector<uint64_t> path_of(const std::vector<int32_t> & ids, int k) { return kvflash_preamble_hashes(ids, k, kCt); }

// A scratch directory under the build's cwd, never /tmp.
struct Scratch {
    std::string dir;
    Scratch() {
        dir = "kvflash_preamble_binder_scratch_" + std::to_string((long)::getpid());
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
    }
    ~Scratch() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
};

// A fake pager with the surface the binder uses.
struct FakePager {
    int ct = kCt, held = 10;
    std::vector<int> pinned;
    int chunk_tokens() const { return ct; }
    int n_chunks() const { return held; }
    void pin_range(int64_t, int64_t hi) { pinned.assign((size_t)((hi + ct - 1) / ct), 1); }
    bool is_pinned(int c) const { return c < (int)pinned.size(); }
    void unpin_all() { pinned.clear(); }
    void synchronize_paging() const {}
    size_t chunk_bytes() const { return (size_t)kChunk; }
    std::vector<uint8_t> blob_header(int k) const { return {(uint8_t)k, 0}; }
    void read_chunk(int c, uint8_t * dst) const { std::fill(dst, dst + kChunk, (uint8_t)(c + 1)); }
};

void must(bool ok) { if (!ok) throw std::runtime_error("binder test precondition failed"); }
}  // namespace

TEST_CASE(KvflashPreambleBinderFixture, a_payload_belongs_to_its_entry_and_eviction_is_reported) {
    KvFlashPreambleStore st(2 * (3 * kChunk + kSlab), kSlab, kChunk);
    const auto a = path_of(ids_of(3 * kCt), 3), b = path_of(ids_of(3 * kCt, 5), 3), c = path_of(ids_of(3 * kCt, 9), 3);
    must(st.capture(a, "x") == KvFlashPreambleStore::Stored);
    CHECK(st.set_payload(a.back(), "x", {1, 2}, {3, 4, 5}, {6}));
    CHECK(!st.set_payload(a.back(), "other-exec", {}, {}, {}));        // never under another identity
    const auto * e = st.find(a.back(), "x");
    REQUIRE(e != nullptr);
    CHECK(e->hdr == std::vector<uint8_t>({1, 2}));
    CHECK(e->chunks == std::vector<uint8_t>({3, 4, 5}));
    CHECK(e->slab == std::vector<uint8_t>({6}));
    must(st.capture(b, "x") == KvFlashPreambleStore::Stored);
    must(st.capture(c, "x") == KvFlashPreambleStore::Stored);          // evicts the least recently used: a
    const auto gone = st.take_evicted();
    REQUIRE(gone.size() == 1u);
    CHECK(gone[0].hash == a.back());
    CHECK(gone[0].exec_id == "x");
    CHECK(st.take_evicted().empty());                                  // reported once
    CHECK(st.find(a.back(), "x") == nullptr);
}

TEST_CASE(KvflashPreambleBinderFixture, erase_refuses_a_protected_entry) {
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    const auto a = path_of(ids_of(2 * kCt), 2);
    must(st.capture(a, "x") == KvFlashPreambleStore::Stored);
    must(st.acquire(a.back(), "x"));
    CHECK(!st.erase(a.back(), "x"));
    CHECK(st.count() == 1);
    must(st.release(a.back(), "x"));
    CHECK(st.erase(a.back(), "x"));
    CHECK(st.count() == 0);
    CHECK(st.used_bytes() == 0);
}

TEST_CASE(KvflashPreambleBinderFixture, exec_id_changes_with_every_part_and_its_order) {
    const std::string base = kvflash_preamble_exec_id({"m|1|2", "cuda", "512", "0"});
    CHECK(base == kvflash_preamble_exec_id({"m|1|2", "cuda", "512", "0"}));
    CHECK(base != kvflash_preamble_exec_id({"m|1|3", "cuda", "512", "0"}));
    CHECK(base != kvflash_preamble_exec_id({"m|1|2", "hip", "512", "0"}));
    CHECK(base != kvflash_preamble_exec_id({"m|1|2", "cuda", "256", "0"}));
    CHECK(base != kvflash_preamble_exec_id({"m|1|2", "cuda", "0", "512"}));
    CHECK(base != kvflash_preamble_exec_id({"m|1|2", "cuda", "5120"}));  // no concatenation ambiguity
    CHECK(base.size() == 16u);
}

TEST_CASE(KvflashPreambleBinderFixture, refcount_follows_bind_and_unbind_and_two_slots_share_one_entry) {
    KvFlashPreambleStore st(2 * (3 * kChunk + kSlab), kSlab, kChunk);
    KvFlashPreambleBinder bd(st, "x", 2);
    const auto a = path_of(ids_of(3 * kCt), 3);
    must(st.capture(a, "x") == KvFlashPreambleStore::Stored);
    CHECK(bd.bind(0, a.back(), 3));
    CHECK(bd.binding(0).k == 3);
    CHECK(st.find(a.back(), "x")->refcount == 1);
    CHECK(bd.bind(1, a.back(), 3));
    CHECK(st.find(a.back(), "x")->refcount == 2);
    bd.unbind(0);                                                      // evict / drop of one slot's session
    CHECK(bd.binding(0).k == 0);
    CHECK(st.find(a.back(), "x")->refcount == 1);
    bd.unbind(0);                                                      // idempotent: never goes below the other slot's count
    CHECK(st.find(a.back(), "x")->refcount == 1);
    // the entry is still protected from the LRU while slot 1 holds it
    const auto b = path_of(ids_of(3 * kCt, 5), 3), c = path_of(ids_of(3 * kCt, 9), 3);
    must(st.capture(b, "x") == KvFlashPreambleStore::Stored);
    must(st.capture(c, "x") == KvFlashPreambleStore::Stored);          // evicts b (a is protected)
    CHECK(st.find(a.back(), "x") != nullptr);
    CHECK(st.find(b.back(), "x") == nullptr);
    bd.unbind(1);
    CHECK(st.find(a.back(), "x")->refcount == 0);
}

TEST_CASE(KvflashPreambleBinderFixture, rebinding_a_slot_releases_the_old_entry_and_an_unknown_entry_binds_nothing) {
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    KvFlashPreambleBinder bd(st, "x", 1);
    const auto a = path_of(ids_of(3 * kCt), 3), b = path_of(ids_of(3 * kCt, 5), 3);
    must(st.capture(a, "x") == KvFlashPreambleStore::Stored);
    must(st.capture(b, "x") == KvFlashPreambleStore::Stored);
    must(bd.bind(0, a.back(), 3));
    CHECK(bd.bind(0, b.back(), 3));
    CHECK(st.find(a.back(), "x")->refcount == 0);
    CHECK(st.find(b.back(), "x")->refcount == 1);
    CHECK(!bd.bind(0, 12345, 3));                                      // no such entry: no binding, the old one released
    CHECK(bd.binding(0).k == 0);
    CHECK(st.find(b.back(), "x")->refcount == 0);
    CHECK(!bd.bind(5, a.back(), 3));                                   // no such slot
}

TEST_CASE(KvflashPreambleBinderFixture, ensure_pins_records_the_count_and_a_released_binding_gives_its_refcount_back) {
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    KvFlashPreambleBinder bd(st, "x", 1);
    const auto a = path_of(ids_of(3 * kCt), 3);
    must(st.capture(a, "x") == KvFlashPreambleStore::Stored);
    FakePager p;
    must(bd.bind(0, a.back(), 3));
    CHECK(bd.ensure(0, p) == 3);
    CHECK(bd.pins(0) == 3);
    p.held = 2;                                                        // the slot's length fell below the prefix
    CHECK(bd.ensure(0, p) == 0);
    CHECK(bd.pins(0) == 0);
    CHECK(bd.binding(0).k == 0);
    CHECK(st.find(a.back(), "x")->refcount == 0);                      // released with the binding
    FakePager q;
    CHECK(bd.ensure(0, q) == 0);                                       // no binding: nothing to pin
}

namespace {
struct Harness {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    Harness() {
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
    ~Harness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};
// Two pager blobs of k chunks equal except the f32 score of each ledger entry.
bool same_but_scores(std::vector<uint8_t> a, std::vector<uint8_t> b, int k, size_t chunk_bytes) {
    constexpr size_t kHdr = 8 + 9 * 4;
    if (a.size() != b.size() || a.size() != kHdr + (size_t)k * (chunk_bytes + 8)) return false;
    for (int c = 0; c < k; c++)
        for (size_t i = 4; i < 8; i++) a[kHdr + (size_t)k * chunk_bytes + (size_t)c * 8 + i] = b[kHdr + (size_t)k * chunk_bytes + (size_t)c * 8 + i] = 0;
    return a == b;
}
KvFlashConfig cfg8() {
    KvFlashConfig c;
    c.pool_tokens = 512; c.chunk_tokens = 64; c.sink_chunks = 1; c.tail_window_chunks = 2;
    return c;
}
}  // namespace

TEST_CASE(KvflashPreambleBinderFixture, an_attach_blob_is_the_serialize_of_the_captured_pager_and_a_restore_matches) {
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 5 * 64));
    // distinct bytes in the pool so the comparison means something
    for (ggml_tensor * t : h.k) { std::vector<uint8_t> junk(ggml_nbytes(t)); for (size_t i = 0; i < junk.size(); i++) junk[i] = (uint8_t)(i * 31 + 7); ggml_backend_tensor_set(t, junk.data(), 0, junk.size()); }
    const auto ids = ids_of(5 * kCt);
    KvFlashPreambleStore st(kBig, kSlab, (int64_t)p.chunk_bytes());
    KvFlashPreambleBinder bd(st, "x", 1);
    const auto r = kvflash_preamble_capture(bd, 0, p, ids, 3, 3, std::vector<uint8_t>{9, 9, 9});
    REQUIRE(r.kind == KvFlashPreambleCapture::Stored);
    CHECK(r.k == 3);
    const auto * e = st.find(r.hash, "x");
    REQUIRE(e != nullptr);
    CHECK(e->slab == std::vector<uint8_t>({9, 9, 9}));
    CHECK(e->chunks.size() == 3 * p.chunk_bytes());
    // the same bytes the pager would hand over: header and K/V exact, the ledger's flags exact; the scores
    // (recency, in a pager that served a turn) are zero in an entry, as they are in a fresh pager
    CHECK(same_but_scores(kvflash_preamble_blob(*e, p.chunk_bytes()), p.serialize(3), 3, p.chunk_bytes()));
    CHECK(bd.binding(0).k == 3);                                               // the capturing slot is bound and pinned
    CHECK(bd.pins(0) == 3);
    // and the blob restores into a fresh pager, K/V bytes identical
    Harness h2; TestPager q;
    must(q.attach(cfg8(), h2.k, h2.v));
    const auto blob = kvflash_preamble_blob(*e, p.chunk_bytes());
    REQUIRE(q.deserialize(blob.data(), blob.size(), 3));
    CHECK(same_but_scores(q.serialize(), p.serialize(3), 3, p.chunk_bytes()));
}

TEST_CASE(KvflashPreambleBinderFixture, capture_is_refused_past_the_bound_and_when_the_pager_lacks_the_chunks) {
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    KvFlashPreambleBinder bd(st, "x", 1);
    FakePager p;                                                       // 10 chunks held
    const auto ids = ids_of(12 * kCt);
    const auto past = kvflash_preamble_capture(bd, 0, p, ids, 6, /*k_max=*/5, {});
    CHECK(past.kind == KvFlashPreambleCapture::Refused);
    CHECK(st.count() == 0);
    CHECK(bd.binding(0).k == 0);
    CHECK(kvflash_preamble_capture(bd, 0, p, ids, 1, 5, {}).kind == KvFlashPreambleCapture::Refused);   // one chunk is not worth an entry
    p.held = 3;
    CHECK(kvflash_preamble_capture(bd, 0, p, ids, 4, 5, {}).kind == KvFlashPreambleCapture::Refused);   // chunks 3 does not exist
    CHECK(kvflash_preamble_capture(bd, 0, p, ids_of(2 * kCt), 3, 5, {}).kind == KvFlashPreambleCapture::Refused);   // ids shorter than k chunks
    CHECK(st.count() == 0);
}

TEST_CASE(KvflashPreambleBinderFixture, a_second_capture_of_the_same_prefix_only_binds_and_a_declined_one_binds_nothing) {
    KvFlashPreambleStore st(3 * kChunk + kSlab, kSlab, kChunk);        // room for one entry of 3 chunks
    KvFlashPreambleBinder bd(st, "x", 2);
    FakePager p;
    const auto ids = ids_of(6 * kCt);
    REQUIRE(kvflash_preamble_capture(bd, 0, p, ids, 3, 5, {1}).kind == KvFlashPreambleCapture::Stored);
    CHECK(kvflash_preamble_capture(bd, 1, p, ids, 3, 5, {1}).kind == KvFlashPreambleCapture::Exists);
    CHECK(st.find(path_of(ids, 3).back(), "x")->refcount == 2);
    const auto other = ids_of(6 * kCt, 3);
    FakePager q;
    CHECK(kvflash_preamble_capture(bd, 1, q, other, 3, 5, {1}).kind == KvFlashPreambleCapture::Declined);   // the protected entry leaves no room
    CHECK(st.count() == 1);
    CHECK(bd.binding(1).k == 3);                                       // its earlier binding is untouched by a declined capture
}

TEST_CASE(KvflashPreambleBinderFixture, an_entry_persists_under_preambles_and_reloads_under_its_exec_id) {
    Scratch sc;
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    const auto a = path_of(ids_of(3 * kCt), 3);
    must(st.capture(a, "exec-A") == KvFlashPreambleStore::Stored);
    must(st.set_payload(a.back(), "exec-A", {1, 2, 3}, std::vector<uint8_t>(3 * kChunk, 7), {4, 5}));
    CHECK(kvflash_preamble_save(sc.dir, *st.find(a.back(), "exec-A")));
    KvFlashPreambleStore st2(kBig, kSlab, kChunk);
    const KvFlashPreambleLoad rep = kvflash_preamble_load_all(sc.dir, st2, "exec-A", (size_t)kChunk);
    CHECK(rep.loaded == 1);
    CHECK(rep.foreign == 0);
    const auto * e = st2.find(a.back(), "exec-A");
    REQUIRE(e != nullptr);
    CHECK(e->k == 3);
    CHECK(e->path == a);
    CHECK(e->hdr == std::vector<uint8_t>({1, 2, 3}));
    CHECK(e->chunks == std::vector<uint8_t>(3 * kChunk, 7));
    CHECK(e->slab == std::vector<uint8_t>({4, 5}));
    CHECK(e->refcount == 0);
}

TEST_CASE(KvflashPreambleBinderFixture, reload_under_another_exec_id_is_refused_and_the_file_stays) {
    Scratch sc;
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    const auto a = path_of(ids_of(3 * kCt), 3);
    must(st.capture(a, "exec-A") == KvFlashPreambleStore::Stored);
    must(st.set_payload(a.back(), "exec-A", {1}, std::vector<uint8_t>(3 * kChunk, 7), {4}));
    must(kvflash_preamble_save(sc.dir, *st.find(a.back(), "exec-A")));
    KvFlashPreambleStore st2(kBig, kSlab, kChunk);
    const KvFlashPreambleLoad rep = kvflash_preamble_load_all(sc.dir, st2, "exec-B", (size_t)kChunk);
    CHECK(rep.loaded == 0);
    CHECK(rep.foreign == 1);
    CHECK(st2.count() == 0);
    CHECK(st2.find(a.back(), "exec-B") == nullptr);
    // still there for the server that wrote it
    KvFlashPreambleStore st3(kBig, kSlab, kChunk);
    CHECK(kvflash_preamble_load_all(sc.dir, st3, "exec-A", (size_t)kChunk).loaded == 1);
}

TEST_CASE(KvflashPreambleBinderFixture, a_torn_or_resized_entry_file_is_never_loaded) {
    Scratch sc;
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    const auto a = path_of(ids_of(3 * kCt), 3);
    must(st.capture(a, "x") == KvFlashPreambleStore::Stored);
    must(st.set_payload(a.back(), "x", {1}, std::vector<uint8_t>(3 * kChunk, 7), {4}));
    must(kvflash_preamble_save(sc.dir, *st.find(a.back(), "x")));
    std::string file;
    for (const auto & d : std::filesystem::recursive_directory_iterator(sc.dir)) if (d.is_regular_file()) file = d.path().string();
    REQUIRE(!file.empty());
    const auto sz = std::filesystem::file_size(file);
    {   // a flipped byte in the chunk bytes
        std::FILE * f = std::fopen(file.c_str(), "r+b");
        std::fseek(f, (long)sz - 40, SEEK_SET);
        int c = std::fgetc(f);
        std::fseek(f, (long)sz - 40, SEEK_SET);
        std::fputc(c ^ 0xff, f);
        std::fclose(f);
    }
    KvFlashPreambleStore st2(kBig, kSlab, kChunk);
    const auto rep = kvflash_preamble_load_all(sc.dir, st2, "x", (size_t)kChunk);
    CHECK(rep.loaded == 0);
    CHECK(rep.bad == 1);
    std::filesystem::resize_file(file, sz / 2);                        // and a torn tail
    CHECK(kvflash_preamble_load_all(sc.dir, st2, "x", (size_t)kChunk).bad == 1);
    // a chunk size that is not this server's
    must(kvflash_preamble_save(sc.dir, *st.find(a.back(), "x")));
    CHECK(kvflash_preamble_load_all(sc.dir, st2, "x", (size_t)kChunk + 1).bad == 1);
    CHECK(st2.count() == 0);
}

TEST_CASE(KvflashPreambleBinderFixture, load_declines_entries_that_do_not_fit_the_budget_and_remove_deletes_the_file) {
    Scratch sc;
    KvFlashPreambleStore st(kBig, kSlab, kChunk);
    for (int salt : {0, 5}) {
        const auto a = path_of(ids_of(3 * kCt, salt), 3);
        must(st.capture(a, "x") == KvFlashPreambleStore::Stored);
        must(st.set_payload(a.back(), "x", {1}, std::vector<uint8_t>(3 * kChunk, 7), {4}));
        must(kvflash_preamble_save(sc.dir, *st.find(a.back(), "x")));
    }
    KvFlashPreambleStore small(1 * (3 * kChunk + kSlab), kSlab, kChunk);   // one entry's worth
    const auto rep = kvflash_preamble_load_all(sc.dir, small, "x", (size_t)kChunk);
    CHECK(rep.loaded == 1);
    CHECK(rep.declined == 1);
    const auto a0 = path_of(ids_of(3 * kCt), 3);
    kvflash_preamble_remove(sc.dir, a0.back(), "x");
    KvFlashPreambleStore st2(kBig, kSlab, kChunk);
    CHECK(kvflash_preamble_load_all(sc.dir, st2, "x", (size_t)kChunk).loaded == 1);   // only the other one is left
}

TEST_CASE(KvflashPreambleBinderFixture, the_binding_round_trips_through_a_sessions_meta) {
    KvFlashSessionEntry e;
    e.sid = "s"; e.client = "c"; e.preamble = kvflash_hex16(0xabcdef0123456789ULL); e.preamble_chunks = 18;
    KvFlashSessionEntry back;
    REQUIRE(kvflash_meta_unpack(kvflash_meta_pack(e), back));
    CHECK(back.preamble == e.preamble);
    CHECK(back.preamble_chunks == 18);
    KvFlashSessionEntry none;
    none.sid = "s";
    REQUIRE(kvflash_meta_unpack(kvflash_meta_pack(none), back));
    CHECK(back.preamble.empty());                                      // a P0 meta, or a session with no binding
    CHECK(back.preamble_chunks == 0);
}
