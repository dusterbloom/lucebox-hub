// Unit tests for KvFlashPager serialize/deserialize, floor_to_chunk, critical
// chunk pinning and the per-chunk score ledger (kvflash_pager.h). CPU ggml
// backend, no GPU: the cache tensors live in a CPU buffer, so every code path
// that moves bytes between pool and host backing is exercised for real.
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerSerdeFixture {};

constexpr int kHeadDim = 4, kPool = 512, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
// KVFLASH2 blob header: 8-byte magic + 9 x uint32.
constexpr size_t kHdr = 8 + 9 * 4;
constexpr size_t kLedgerEntry = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// pool 512 / chunk 64 = 8 blocks; sink 1 + tail 4 + 2 = 7 <= 8 blocks.
struct Harness {
    ggml_context *          ctx = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    explicit Harness(ggml_type vtype = GGML_TYPE_F16, int pool = kPool) {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kNLayer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kNLayer; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, pool, kNHeadKv));
            v.push_back(ggml_new_tensor_3d(ctx, vtype, kHeadDim, pool, kNHeadKv));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Harness() {
        if (buf)     ggml_backend_buffer_free(buf);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    Harness(const Harness &) = delete;
    Harness & operator=(const Harness &) = delete;
};

KvFlashConfig make_cfg() {
    KvFlashConfig c;
    c.pool_tokens = kPool;
    c.chunk_tokens = kChunk;
    c.sink_chunks = 1;
    c.tail_window_chunks = 4;
    return c;
}

// REQUIRE is a fixture member; free helpers throw instead (the framework
// reports an exception as a test failure).
void must(bool ok) { if (!ok) throw std::runtime_error("kvflash serde helper precondition failed"); }

// Every (tensor, head) segment that belongs to block `b`, in a fixed order.
template <class F>
void for_each_segment(Harness & h, int block, F && fn) {
    for (int l = 0; l < kNLayer; l++) {
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * t = kv == 0 ? h.k[l] : h.v[l];
            for (int hh = 0; hh < kNHeadKv; hh++) {
                const size_t off = (size_t)block * kChunk * t->nb[1] + (size_t)hh * t->nb[2];
                fn(t, off, (size_t)kChunk * t->nb[1]);
            }
        }
    }
}

// Deterministic per-(logical chunk, seed, byte offset in chunk) pattern. `i` is
// the offset across the whole chunk, so every segment gets different bytes (a
// linear pattern with a short period repeated identically in each segment).
uint8_t pat(int c, int seed, size_t i) {
    uint32_t x = (uint32_t)c * 2654435761u ^ (uint32_t)seed * 40503u ^ (uint32_t)i * 2246822519u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return (uint8_t)(x >> 24) | 1u;   // never zero: a zeroed segment is detected too
}

void stamp_chunk(Harness & h, const KvFlashPager & p, int c, int seed) {
    const int block = p.block_of(c);
    must(block >= 0);
    size_t n = 0;
    for_each_segment(h, block, [&](ggml_tensor * t, size_t off, size_t len) {
        std::vector<uint8_t> b(len);
        for (size_t i = 0; i < len; i++) b[i] = pat(c, seed, n + i);
        ggml_backend_tensor_set(t, b.data(), off, len);
        n += len;
    });
}

bool chunk_matches(Harness & h, const KvFlashPager & p, int c, int seed) {
    const int block = p.block_of(c);
    if (block < 0) return false;
    size_t n = 0;
    bool ok = true;
    for_each_segment(h, block, [&](ggml_tensor * t, size_t off, size_t len) {
        std::vector<uint8_t> b(len);
        ggml_backend_tensor_get(t, b.data(), off, len);
        for (size_t i = 0; i < len; i++) ok = ok && b[i] == pat(c, seed, n + i);
        n += len;
    });
    return ok;
}

// A: 6 chunks (0..5) stamped, chunks 2 and 3 paged out, chunk 2 recalled into a
// different block (non-identity), chunk 3 left on host. Scores 10*c on 0..5.
void build_source(Harness & A, KvFlashPager & pa, int seed) {
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pa.alloc_span(0, 6 * kChunk));
    for (int c = 0; c < 6; c++) stamp_chunk(A, pa, c, seed);
    must(pa.page_out(2));
    must(pa.page_out(3));
    must(pa.page_in(2));
    pa.synchronize_paging();
    for (int c = 0; c < 6; c++) pa.set_chunk_score(c, 10.0f * (float)c);
}

} // namespace

TEST_CASE(KvflashPagerSerdeFixture, floor_to_chunk_aligns_down) {
    CHECK(floor_to_chunk(0, 64) == 0);
    CHECK(floor_to_chunk(63, 64) == 0);
    CHECK(floor_to_chunk(64, 64) == 64);
    CHECK(floor_to_chunk(65, 64) == 64);
    CHECK(floor_to_chunk(127, 64) == 64);
    CHECK(floor_to_chunk(34077, 64) == 34048);
}

TEST_CASE(KvflashPagerSerdeFixture, roundtrip_preserves_residency_bytes_and_host_copies) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 1);
    REQUIRE(pa.block_of(2) >= 0 && pa.block_of(2) != 2);  // relocated, not identity
    REQUIRE(!pa.is_resident(3));                          // paged out at serialize time

    const std::vector<uint8_t> blob = pa.serialize();
    REQUIRE(blob.size() == kHdr + 6 * (size_t)(kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2) +
                           6 * kLedgerEntry);
    REQUIRE(std::memcmp(blob.data(), "2HSALFVK", 8) == 0);  // "KVFLASH2" little-endian u64

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    // Pre-existing state must be discarded by deserialize.
    REQUIRE(pb.alloc_span(0, 2 * kChunk));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));

    REQUIRE(pb.n_chunks() == 6);
    for (int c = 0; c < 6; c++) CHECK(pb.is_resident(c) == pa.is_resident(c));
    CHECK(!pb.is_resident(3));
    CHECK(pb.resident_blocks() == 5);
    for (int c : {0, 1, 2, 4, 5}) CHECK(chunk_matches(B, pb, c, 1));
    // Positions: logical -> slot lands inside the chunk's block, in order.
    for (int c : {0, 1, 2, 4, 5}) {
        for (int i : {0, 17, kChunk - 1}) {
            CHECK(pb.slot_of((int64_t)c * kChunk + i) == pb.block_of(c) * kChunk + i);
        }
    }
    CHECK(pb.slot_of(3 * kChunk) == -1);
    // The host copy of chunk 3 survived: recalling it yields the source bytes.
    REQUIRE(pb.page_in(3));
    pb.synchronize_paging();
    CHECK(chunk_matches(B, pb, 3, 1));
    // Source pager is untouched by serialize().
    CHECK(!pa.is_resident(3));
    CHECK(pa.n_chunks() == 6);
}

TEST_CASE(KvflashPagerSerdeFixture, roundtrip_of_all_host_backed_chunks) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 2);
    for (int c : {0, 1, 2, 4, 5}) REQUIRE(pa.page_out(c));
    REQUIRE(pa.resident_blocks() == 0);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.resident_blocks() == 0);
    CHECK(pb.n_chunks() == 6);
    for (int c = 0; c < 6; c++) {
        REQUIRE(pb.page_in(c));
        pb.synchronize_paging();
        CHECK(chunk_matches(B, pb, c, 2));
    }
}

TEST_CASE(KvflashPagerSerdeFixture, partial_serialize_keeps_only_first_chunks) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 3);
    const std::vector<uint8_t> full = pa.serialize();
    const std::vector<uint8_t> part = pa.serialize(4);
    REQUIRE(part.size() < full.size());

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(part.data(), part.size()));
    CHECK(pb.n_chunks() == 4);
    for (int c : {0, 1, 2}) { CHECK(pb.is_resident(c)); CHECK(chunk_matches(B, pb, c, 3)); }
    CHECK(!pb.is_resident(3));
    CHECK(pb.block_of(4) == -1);
    CHECK(pb.slot_of(4 * kChunk) == -1);
    REQUIRE(pb.page_in(3));
    pb.synchronize_paging();
    CHECK(chunk_matches(B, pb, 3, 3));

    // max_chunks beyond n_chunks and the default both mean "everything".
    CHECK(pa.serialize(100) == full);
    CHECK(pa.serialize(-1) == full);
    // max_chunks == 0 is a valid, empty snapshot.
    const std::vector<uint8_t> none = pa.serialize(0);
    CHECK(none.size() == kHdr);
    REQUIRE(pb.deserialize(none.data(), none.size()));
    CHECK(pb.n_chunks() == 0);
    CHECK(pb.resident_blocks() == 0);
}

TEST_CASE(KvflashPagerSerdeFixture, ledger_scores_roundtrip) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 4);
    pa.set_chunk_score(4, -3.5f);
    CHECK(pa.chunk_score(5) == 50.0f);
    CHECK(pa.chunk_score(99) == kNegInf);   // out of range: unset sentinel
    pa.set_chunk_score(99, 1.0f);           // out of range: ignored, no crash
    pa.set_chunk_score(-1, 1.0f);

    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    for (int c = 0; c < 6; c++) CHECK(pb.chunk_score(c) == pa.chunk_score(c));
    CHECK(pb.chunk_score(3) == 30.0f);      // host-backed chunk keeps its score
    CHECK(pb.chunk_score(4) == -3.5f);
    // A chunk never scored stays at the unset sentinel through the roundtrip.
    Harness C;
    TestPager pc;
    REQUIRE(pc.attach(make_cfg(), C.k, C.v));
    REQUIRE(pc.alloc_span(0, 2 * kChunk));
    const std::vector<uint8_t> blob2 = pc.serialize();
    REQUIRE(pb.deserialize(blob2.data(), blob2.size()));
    CHECK(pb.chunk_score(0) == kNegInf);
    CHECK(pb.chunk_score(1) == kNegInf);
}

TEST_CASE(KvflashPagerSerdeFixture, reset_pager_is_identity_so_dense_snapshots_ignore_stale_mapping) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 3);
    REQUIRE(!pa.is_identity());   // previous request left a paged mapping
    pa.reset();
    CHECK(pa.is_identity());
    CHECK(pa.n_chunks() == 0);
}

TEST_CASE(KvflashPagerSerdeFixture, serialized_dtype_is_recorded) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 5);
    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.serialized_kv_k_type() == (uint32_t)GGML_TYPE_F16);
}

TEST_CASE(KvflashPagerSerdeFixture, deserialize_rejects_a_changed_v_dtype_and_the_old_format) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 7);
    const std::vector<uint8_t> blob = pa.serialize();

    // Same K, V changed F16 -> BF16: identical byte geometry, different values.
    Harness B(GGML_TYPE_BF16);
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    CHECK(!pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.n_chunks() == 0);

    // A blob in the previous format (KVFLASH1, 8-field header) is refused.
    auto old = blob;
    std::memcpy(old.data(), "1HSALFVK", 8);
    Harness C;
    TestPager pc;
    REQUIRE(pc.attach(make_cfg(), C.k, C.v));
    CHECK(!pc.deserialize(old.data(), old.size()));
    CHECK(pc.deserialize(blob.data(), blob.size()));
}

TEST_CASE(KvflashPagerSerdeFixture, malformed_blobs_are_rejected_without_touching_state) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 6);
    const std::vector<uint8_t> good = pa.serialize();

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 3 * kChunk));
    for (int c = 0; c < 3; c++) stamp_chunk(B, pb, c, 9);
    const uint64_t epoch0 = pb.epoch();

    auto rejected = [&](const std::vector<uint8_t> & blob) {
        return !pb.deserialize(blob.data(), blob.size());
    };
    auto untouched = [&]() {
        bool ok = pb.n_chunks() == 3 && pb.epoch() == epoch0;
        for (int c = 0; c < 3; c++) ok = ok && pb.is_resident(c) && chunk_matches(B, pb, c, 9);
        return ok;
    };

    // Empty / null / short at every structural boundary.
    CHECK(!pb.deserialize(nullptr, 0));
    CHECK(rejected(std::vector<uint8_t>{}));
    for (size_t len : {size_t(1), size_t(7), size_t(8), kHdr - 1, kHdr, kHdr + 1,
                       good.size() / 2, good.size() - 1}) {
        CHECK(rejected(std::vector<uint8_t>(good.begin(), good.begin() + len)));
    }
    CHECK(untouched());

    { auto b = good; b[0] ^= 0xFF;  CHECK(rejected(b)); }     // wrong magic
    { auto b = good; b[7] ^= 0x01;  CHECK(rejected(b)); }     // wrong magic (last byte)
    { auto b = good; b[8 + 4] ^= 0xFF; CHECK(rejected(b)); }  // chunk_tokens mismatch
    { auto b = good; b[8 + 8] ^= 0xFF; CHECK(rejected(b)); }  // n_head_kv mismatch
    { auto b = good; b[8 + 12] ^= 0xFF; CHECK(rejected(b)); } // k_seg_bytes mismatch
    { auto b = good; b[8 + 20] ^= 0xFF; CHECK(rejected(b)); } // chunk_bytes mismatch
    { auto b = good; b[8 + 24] ^= 0x01; CHECK(rejected(b)); } // KV dtype mismatch
    { auto b = good; b[8 + 32] ^= 0x01; CHECK(rejected(b)); } // V dtype mismatch
    { auto b = good; b[8 + 28] = 7;     CHECK(rejected(b)); } // unknown ledger flag
    { auto b = good; std::memset(&b[8], 0xFF, 4); CHECK(rejected(b)); }  // nc = 0xFFFFFFFF (overflow)
    { auto b = good; b[8] = 7;          CHECK(rejected(b)); } // nc claims more chunks than the blob holds
    {   // ledger claims more resident chunks (40) than the pool has blocks (8)
        const size_t cb = (good.size() - kHdr - 6 * kLedgerEntry) / 6;
        std::vector<uint8_t> b(good.begin(), good.begin() + kHdr);
        b[8] = 40;  // nc = 40 (little-endian u32)
        b.resize(kHdr + 40 * cb + 40 * kLedgerEntry, 0);
        for (size_t c = 0; c < 40; c++) b[kHdr + 40 * cb + c * kLedgerEntry] = 1;
        CHECK(rejected(b));
    }
    CHECK(untouched());

    // Trailing bytes after a valid blob are tolerated; the blob still loads.
    { auto b = good; b.push_back(0xAB); CHECK(pb.deserialize(b.data(), b.size())); }
    CHECK(pb.n_chunks() == 6);
}

TEST_CASE(KvflashPagerSerdeFixture, pin_range_is_half_open_and_pinned_chunks_are_never_evicted) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));

    // [64, 128) is exactly chunk 1; chunk 2 (token 128) is outside.
    p.pin_range(1 * kChunk, 2 * kChunk);
    CHECK(!p.is_pinned(0));
    CHECK(p.is_pinned(1));
    CHECK(!p.is_pinned(2));
    // A range ending mid-chunk pins that chunk too; empty / inverted is a no-op.
    p.unpin_all();
    p.pin_range(kChunk, kChunk + 1);
    CHECK(p.is_pinned(1));
    CHECK(!p.is_pinned(2));
    p.unpin_all();
    p.pin_range(2 * kChunk, 2 * kChunk);
    p.pin_range(3 * kChunk, 2 * kChunk);
    p.pin_range(-5, 0);
    for (int c = 0; c < 8; c++) CHECK(!p.is_pinned(c));
    CHECK(!p.is_pinned(-1));
    CHECK(!p.is_pinned(1000));

    // Deadlock guard: sink 1 + tail 4 + pinned + 2 must fit in 8 blocks, so
    // at most one chunk can be pinned. A 2-chunk span is refused whole.
    p.pin_range(1 * kChunk, 3 * kChunk);
    CHECK(!p.is_pinned(1));
    CHECK(!p.is_pinned(2));
    p.pin_range(1 * kChunk, 2 * kChunk);
    CHECK(p.is_pinned(1));
    p.pin_range(2 * kChunk, 3 * kChunk);   // a second pin would deadlock: refused
    CHECK(!p.is_pinned(2));
    p.pin_range(1 * kChunk, 2 * kChunk);   // re-pinning an already pinned chunk is fine
    CHECK(p.is_pinned(1));

    // Drive eviction far past the pool: chunk 1 stays resident throughout,
    // with both LRU and score-driven victim selection.
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) p.score_hook = [](int c) { return c == 1 ? -1e9f : (float)c; };
        for (int c = 8 + pass * 8; c < 16 + pass * 8; c++) {
            REQUIRE(p.alloc_span(c * kChunk, kChunk));
            CHECK(p.is_resident(1));
            CHECK(p.is_resident(0));  // sink
        }
    }
    // reselect() with a score hook that hates chunk 1 still keeps it.
    p.score_hook = [](int c) { return c == 1 ? -1e9f : (float)c; };
    p.reselect();
    CHECK(p.is_resident(1));

    // reset() clears pins.
    p.reset();
    CHECK(!p.is_pinned(1));
}

TEST_CASE(KvflashPagerSerdeFixture, no_new_api_use_keeps_eviction_unchanged) {
    // Without pins or scores the victim is the oldest unprotected chunk.
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    REQUIRE(p.alloc_span(8 * kChunk, kChunk));   // evicts chunk 1 (sink 0 protected)
    CHECK(p.is_resident(0));
    CHECK(!p.is_resident(1));
    CHECK(p.is_resident(8));
}

TEST_CASE(KvflashPagerSerdeFixture, attach_clears_pins_from_a_previous_attach) {
    // 16-block pool: the deadlock guard allows 9 pins (1 sink + 4 tail + 9 + 2 <= 16).
    Harness Big(GGML_TYPE_F16, 2 * kPool);
    KvFlashConfig big = make_cfg();
    big.pool_tokens = 2 * kPool;
    TestPager p;
    REQUIRE(p.attach(big, Big.k, Big.v));
    p.pin_range(1 * kChunk, 9 * kChunk);   // chunks 1..8
    REQUIRE(p.is_pinned(1));
    REQUIRE(p.is_pinned(8));

    // Re-attach to the 8-block pool: stale pins would protect every chunk and
    // starve allocation (sink + 7 pinned = the whole pool, no victim left).
    Harness Small;
    REQUIRE(p.attach(make_cfg(), Small.k, Small.v));
    for (int c = 0; c < 9; c++) CHECK(!p.is_pinned(c));
    CHECK(p.alloc_span(0, 12 * kChunk));
}

TEST_CASE(KvflashPagerSerdeFixture, pin_range_rejects_positions_beyond_the_chunk_index_range) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    // Chunk index 2^31 used to narrow to a negative int, skip the resize and
    // write pinned_ out of bounds.
    const int64_t c31 = (int64_t)1 << 31;
    p.pin_range(c31 * kChunk, c31 * kChunk + 1);
    p.pin_range(c31 * kChunk - 1, c31 * kChunk + 1);
    // Huge span: bounded scan, refused whole (it could never fit the pool).
    p.pin_range(0, (int64_t)1 << 40);
    p.pin_range(0, std::numeric_limits<int64_t>::max());
    // In int range but absurd: must not allocate a gigabyte of pin flags.
    p.pin_range((int64_t)1 << 36, ((int64_t)1 << 36) + 1);
    for (int c : {0, 1, 7, 1 << 20, 1 << 30, std::numeric_limits<int>::max()}) CHECK(!p.is_pinned(c));
    CHECK(!p.is_pinned(-1));
    // Valid pins still work afterwards.
    p.pin_range(1 * kChunk, 2 * kChunk);
    CHECK(p.is_pinned(1));
}

TEST_CASE(KvflashPagerSerdeFixture, never_materialized_holes_stay_holes_through_a_roundtrip) {
    Harness A;
    TestPager pa;
    REQUIRE(pa.attach(make_cfg(), A.k, A.v));
    REQUIRE(pa.alloc_span(0, 2 * kChunk));
    for (int c = 0; c < 2; c++) stamp_chunk(A, pa, c, 8);
    REQUIRE(pa.alloc_span(6 * kChunk, 1));   // jump: chunks 2..5 are holes, never written
    stamp_chunk(A, pa, 6, 8);
    REQUIRE(pa.n_chunks() == 7);
    for (int c = 2; c < 6; c++) REQUIRE(pa.block_of(c) == -1);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    // A hole is not recallable zero KV, and holds no host copy.
    for (int c = 2; c < 6; c++) CHECK(!pb.page_in(c));
    CHECK(pb.stats().host_bytes == 3 * (int64_t)(blob.size() - kHdr - 7 * kLedgerEntry) / 7);
    // reselect() must not pick them either, however well they score.
    pb.score_hook = [](int c) { return c >= 2 && c < 6 ? 1e6f : (float)c; };
    pb.reselect();
    for (int c = 2; c < 6; c++) CHECK(!pb.is_resident(c));
    for (int c : {0, 1, 6}) { CHECK(pb.is_resident(c)); CHECK(chunk_matches(B, pb, c, 8)); }
    // The hole survives a second serialize unchanged.
    pb.score_hook = nullptr;
    CHECK(pb.serialize() == blob);

    // A blob claiming a resident chunk that was never materialized is corrupt.
    auto bad = blob;
    const size_t cb = (blob.size() - kHdr - 7 * kLedgerEntry) / 7;
    bad[kHdr + 7 * cb + 3 * kLedgerEntry] = 1;       // chunk 3 "resident"...
    bad[kHdr + 7 * cb + 3 * kLedgerEntry + 1] = 0;   // ...but not materialized
    CHECK(!pb.deserialize(bad.data(), bad.size()));
}

// The fill pattern must tell segments apart: a segment is 512 bytes here, so a
// pattern that repeats every 256 bytes made whole K/V/head/layer segments
// interchangeable and a swapped-segment bug invisible.
TEST_CASE(KvflashPagerSerdeFixture, fill_pattern_detects_swapped_segments) {
    Harness A;
    TestPager pa;
    REQUIRE(pa.attach(make_cfg(), A.k, A.v));
    REQUIRE(pa.alloc_span(0, kChunk));
    struct Seg { ggml_tensor * t; size_t off, len; };
    std::vector<Seg> segs;
    for_each_segment(A, pa.block_of(0), [&](ggml_tensor * t, size_t off, size_t len) {
        segs.push_back({t, off, len});
    });
    REQUIRE(segs.size() == (size_t)(kNLayer * 2 * kNHeadKv));
    // Swap segments i and j in place, check, swap back.
    auto swapped_is_caught = [&](size_t i, size_t j) {
        stamp_chunk(A, pa, 0, 3);
        std::vector<uint8_t> a(segs[i].len), b(segs[j].len);
        ggml_backend_tensor_get(segs[i].t, a.data(), segs[i].off, a.size());
        ggml_backend_tensor_get(segs[j].t, b.data(), segs[j].off, b.size());
        ggml_backend_tensor_set(segs[i].t, b.data(), segs[i].off, b.size());
        ggml_backend_tensor_set(segs[j].t, a.data(), segs[j].off, a.size());
        return !chunk_matches(A, pa, 0, 3);
    };
    stamp_chunk(A, pa, 0, 3);
    REQUIRE(chunk_matches(A, pa, 0, 3));
    for (size_t i = 0; i < segs.size(); i++)
        for (size_t j = i + 1; j < segs.size(); j++)
            CHECK(swapped_is_caught(i, j));   // head, K<->V and layer swaps alike
}

TEST_CASE(KvflashPagerSerdeFixture, pageable_host_backing_pages_out_and_in_bit_exact) {
    // Explicit pageable mode: the path hosted CI (no pinned memory) relies on.
    Harness A;
    KvFlashPager p;
    p.set_pinned_host_backing(false);
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 3 * kChunk));
    for (int c = 0; c < 3; c++) stamp_chunk(A, p, c, 11);
    REQUIRE(p.page_out(1));
    REQUIRE(!p.is_resident(1));
    CHECK(p.stats().host_bytes > 0);
    REQUIRE(p.page_in(1));
    p.synchronize_paging();
    CHECK(chunk_matches(A, p, 1, 11));
    const std::vector<uint8_t> blob = p.serialize();
    Harness B;
    KvFlashPager q;
    q.set_pinned_host_backing(false);
    REQUIRE(q.attach(make_cfg(), B.k, B.v));
    REQUIRE(q.deserialize(blob.data(), blob.size()));
    for (int c = 0; c < 3; c++) CHECK(chunk_matches(B, q, c, 11));
    q.truncate(kChunk);   // frees pageable host buffers; reset frees the rest
    CHECK(q.n_chunks() == 1);
}

TEST_CASE(KvflashPagerSerdeFixture, blob_fields_are_little_endian) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 2 * kChunk));
    p.set_chunk_score(1, 1.0f);   // IEEE-754 0x3f800000
    const std::vector<uint8_t> blob = p.serialize();
    const size_t cb = (blob.size() - kHdr - 2 * kLedgerEntry) / 2;
    const uint8_t nc_le[4] = {2, 0, 0, 0}, chunk_tokens_le[4] = {kChunk, 0, 0, 0};
    CHECK(std::memcmp(&blob[8], nc_le, 4) == 0);
    CHECK(std::memcmp(&blob[12], chunk_tokens_le, 4) == 0);
    const uint8_t one_le[4] = {0x00, 0x00, 0x80, 0x3f};
    CHECK(std::memcmp(&blob[kHdr + 2 * cb + kLedgerEntry + 4], one_le, 4) == 0);
}

TEST_CASE(KvflashPagerSerdeFixture, reselect_recalls_top_scored_host_chunk_and_keeps_tail) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 12 * kChunk));       // 8 blocks: 4 chunks go to host
    int cold = -1;
    for (int c = 1; c < 8 && cold < 0; c++) if (!p.is_resident(c)) cold = c;
    REQUIRE(cold > 0);
    p.score_hook = [cold](int c) { return c == cold ? 1e9f : (float)-c; };
    CHECK(p.reselect() > 0);
    CHECK(p.is_resident(cold));
    CHECK(p.is_resident(0));                      // sink
    for (int c = 8; c < 12; c++) CHECK(p.is_resident(c));   // tail window
}

TEST_CASE(KvflashPagerSerdeFixture, reselect_tail_override_protects_only_n_chunks) {
    auto run = [](int tail_override) {
        Harness A;
        TestPager p;
        if (!p.attach(make_cfg(), A.k, A.v) || !p.alloc_span(0, 12 * kChunk)) return std::vector<int>{};
        // Tail chunks score lowest; old chunks score by index.
        p.score_hook = [](int c) { return c >= 8 ? -1000.0f + c : (float)c; };
        if (tail_override < 0) p.reselect(); else p.reselect(tail_override);
        std::vector<int> res;
        for (int c = 0; c < 12; c++) if (p.is_resident(c)) res.push_back(c);
        return res;
    };
    // Default (cfg tail = 4): 8..11 protected.
    const std::vector<int> def = run(-1);
    REQUIRE(!def.empty());
    for (int c = 8; c < 12; c++) CHECK(std::find(def.begin(), def.end(), c) != def.end());
    // Override 1: protects c > cur-1-1, i.e. the current chunk 11 plus 10; 8, 9 lose to the higher-scored old chunks.
    const std::vector<int> two = run(1);
    CHECK(std::find(two.begin(), two.end(), 10) != two.end());
    CHECK(std::find(two.begin(), two.end(), 11) != two.end());
    CHECK(std::find(two.begin(), two.end(), 8) == two.end());
    CHECK(std::find(two.begin(), two.end(), 9) == two.end());
    CHECK(std::find(two.begin(), two.end(), 0) != two.end());   // sink
}

TEST_CASE(KvflashPagerSerdeFixture, host_k_seg_points_at_the_paged_out_k_rows) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    for (int c = 0; c < 8; c++) stamp_chunk(A, p, c, 3);
    REQUIRE(p.alloc_span(8 * kChunk, 2 * kChunk));       // pages two chunks out
    int cold = -1;
    for (int c = 1; c < 8 && cold < 0; c++) if (!p.is_resident(c)) cold = c;
    REQUIRE(cold > 0);
    CHECK(p.host_k_seg(0, 0, 0) == nullptr);              // resident: no host copy to read
    CHECK(p.cur_chunk() == 9);
    CHECK(p.n_pool_blocks() == 8);
    const size_t len = p.k_seg_bytes();
    CHECK(len == (size_t)kChunk * kHeadDim * 2);
    for (int l = 0; l < kNLayer; l++) {
        for (int h = 0; h < kNHeadKv; h++) {
            const uint8_t * s = p.host_k_seg(cold, l, h);
            REQUIRE(s != nullptr);
            const size_t n = (size_t)((l * 2 + 0) * kNHeadKv + h) * len;   // stamp order: layer, K|V, head
            bool same = true;
            for (size_t i = 0; i < len; i++) same = same && s[i] == pat(cold, 3, n + i);
            CHECK(same);
        }
    }
}

TEST_CASE(KvflashPagerSerdeFixture, reselect_with_full_tail_keeps_a_non_recent_resident_set_exactly) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 12 * kChunk));
    // Move the pool to a resident set that is NOT the newest chunks.
    p.score_hook = [](int c) { return (float)-c; };
    p.reselect(0);                                  // sink + current chunk + oldest-scored-best chunks 1..6
    std::vector<int> before;
    for (int c = 0; c < 12; c++) if (p.is_resident(c)) before.push_back(c);
    REQUIRE(before.size() == 8);
    CHECK(!p.is_resident(8));                       // a recent chunk is out, older ones are in
    CHECK(p.is_resident(2));
    // Every chunk protected: more tied candidates than blocks. Residents win the ties.
    p.score_hook = [](int c) { return (float)c; };
    CHECK(p.reselect(12) == 0);
    std::vector<int> after;
    for (int c = 0; c < 12; c++) if (p.is_resident(c)) after.push_back(c);
    CHECK(after == before);
}

namespace {
// Block map after: alloc 12 chunks, install the base hook, let `mid` run
// (it may install a "reselect" hook), then truncate (refill_after_truncate_
// ranks the host chunks by whatever hook is installed then).
template <class Mid>
std::vector<int> blocks_after_truncate(Mid && mid) {
    Harness A;
    TestPager p;
    must(p.attach(make_cfg(), A.k, A.v));
    must(p.alloc_span(0, 12 * kChunk));
    p.score_hook = [](int c) { return (float)-c; };     // base policy: older host chunks first
    mid(p);
    p.truncate(10 * kChunk);                             // frees two blocks; host chunks refill them
    std::vector<int> blocks;
    for (int c = 0; c < p.n_chunks(); c++) blocks.push_back(p.block_of(c));
    return blocks;
}
}

TEST_CASE(KvflashPagerSerdeFixture, scoped_score_hook_restores_the_base_hook_so_truncate_recalls_alike) {
    const std::vector<int> ref = blocks_after_truncate([](KvFlashPager &) {});
    const std::vector<int> scoped = blocks_after_truncate([](KvFlashPager & p) {
        ScopedScoreHook scope(p);
        p.score_hook = [](int c) { return (float)c; };   // the reselect's hook: newest host chunks first
    });
    const std::vector<int> leaked = blocks_after_truncate([](KvFlashPager & p) {
        p.score_hook = [](int c) { return (float)c; };   // no restore: what apply did before
    });
    CHECK(scoped == ref);
    CHECK(leaked != ref);                                // the test can see the contamination
}

TEST_CASE(KvflashPagerSerdeFixture, scoped_score_hook_restores_an_empty_hook_too) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    {
        ScopedScoreHook scope(p);
        p.score_hook = [](int c) { return (float)c; };
        CHECK((bool)p.score_hook);
    }
    CHECK(!p.score_hook);
}

// Mode 4 depends on this: without a hook, truncating back to a position and
// appending the same tokens again evicts the same chunks as the first time.
TEST_CASE(KvflashPagerSerdeFixture, truncate_back_restores_residency_and_replays_the_same_evictions) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    auto resident = [&]() {
        std::vector<int> r;
        for (int c = 0; c < 32; c++) if (p.is_resident(c)) r.push_back(c);
        return r;
    };
    REQUIRE(p.alloc_span(0, 6 * kChunk));
    const std::vector<int> pre = resident();
    std::vector<int> first_grown;
    for (int pass = 0; pass < 2; pass++) {
        // Append past the pool so old chunks are evicted to host.
        for (int c = 6; c < 14; c++) REQUIRE(p.alloc_span(c * kChunk, kChunk));
        const std::vector<int> grown = resident();
        if (pass == 0) CHECK(grown != pre);
        if (pass == 0) first_grown = grown;
        else CHECK(grown == first_grown);                    // same appends evict the same chunks
        p.truncate(6 * kChunk);                              // rewind: free the new chunks, recall evicted ones
        CHECK(resident() == pre);                            // residency equals the pre-append residency
    }
    // A mid-chunk rewind point (the question starts inside a chunk) behaves the same.
    p.truncate(0);
    REQUIRE(p.alloc_span(0, 5 * kChunk + 10));
    const std::vector<int> pre2 = resident();
    for (int c = 6; c < 14; c++) REQUIRE(p.alloc_span(c * kChunk, kChunk));
    p.truncate(5 * kChunk + 10);
    CHECK(resident() == pre2);
}

// ── truncate() layout contract ───────────────────────────────────────────
// truncate() promises "the result matches a pager that only ever saw P
// tokens". Pool attention reduces over physical slots in order, so the
// logical-to-physical map (chunks_[c].block) is part of that result: the same
// content in a different layout rounds differently. Scope: only when the kept
// prefix fits the pool (ceil(P/chunk) <= n_blocks). When it does not fit, the
// layout is history-dependent by design and is not tested here.
namespace {
constexpr int kPoolBlocks = kPool / kChunk;   // 8

std::vector<int> layout_of(const KvFlashPager & p, int n_chunks) {
    std::vector<int> v;
    for (int c = 0; c < n_chunks; c++) v.push_back(p.block_of(c));
    return v;
}

// What a pager that only ever saw P tokens gives: the prefix map, then the
// blocks handed out for M more chunks appended after it.
struct FreshMaps { std::vector<int> prefix, after_append; bool ok = false; };

FreshMaps fresh_maps(int P, int M) {
    FreshMaps f;
    Harness A;
    TestPager p;
    if (!p.attach(make_cfg(), A.k, A.v)) return f;
    if (!p.alloc_span(0, P)) return f;
    const int keep = (P + kChunk - 1) / kChunk;
    f.prefix = layout_of(p, keep);
    if (!p.alloc_span((int64_t)keep * kChunk, (int64_t)M * kChunk)) return f;
    f.after_append = layout_of(p, keep + M);
    f.ok = true;
    return f;
}

// truncate(P) on `p`, then append M chunks; same two maps as fresh_maps().
FreshMaps truncated_maps(KvFlashPager & p, int P, int M) {
    FreshMaps f;
    p.truncate(P);
    const int keep = (P + kChunk - 1) / kChunk;
    f.prefix = layout_of(p, keep);
    if (!p.alloc_span((int64_t)keep * kChunk, (int64_t)M * kChunk)) return f;
    f.after_append = layout_of(p, keep + M);
    f.ok = true;
    return f;
}

// Append chunks [from, to) one at a time (the prefill pattern).
bool append_chunks(KvFlashPager & p, int from, int to) {
    for (int c = from; c < to; c++)
        if (!p.alloc_span((int64_t)c * kChunk, kChunk)) return false;
    return true;
}
}

// Control: a pager that never overflowed. Must already pass.
TEST_CASE(KvflashPagerSerdeFixture, truncate_fitting_prefix_gives_fresh_layout_without_overflow) {
    for (int P : {3 * kChunk, 5 * kChunk + 10}) {
        const FreshMaps fresh = fresh_maps(P, 2);
        REQUIRE(fresh.ok);
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 7));                 // 7 <= 8 blocks: never evicts
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);
        CHECK(got.after_append == fresh.after_append);
    }
}

TEST_CASE(KvflashPagerSerdeFixture, truncate_fitting_prefix_gives_fresh_layout_after_reselect) {
    for (int P : {3 * kChunk, 5 * kChunk + 10}) {
        const FreshMaps fresh = fresh_maps(P, 2);
        REQUIRE(fresh.ok);
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 14));                // past the pool: LRU evicts and pages out
        // A reselect with a score hook permutes residency and layout.
        p.score_hook = [](int c) { return (float)-c; };
        p.reselect(0);
        p.score_hook = nullptr;
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);               // block of every chunk c < keep
        CHECK(got.after_append == fresh.after_append);   // later appends hand out the same blocks
    }
}

TEST_CASE(KvflashPagerSerdeFixture, truncate_fitting_prefix_gives_fresh_layout_after_any_lru_history) {
    const int P = 3 * kChunk;
    const FreshMaps fresh = fresh_maps(P, 2);
    REQUIRE(fresh.ok);
    {   // History A: a long straight append, LRU only.
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 16));
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);
        CHECK(got.after_append == fresh.after_append);
    }
    {   // History B: a different length with an interleaved truncate, LRU only.
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 12));
        p.truncate(9 * kChunk);                          // 9 > 8 blocks: layout is history-dependent here
        REQUIRE(append_chunks(p, 9, 15));
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);
        CHECK(got.after_append == fresh.after_append);
    }
}

// ── truncate() identity relayout moves real bytes ────────────────────────
namespace {
// Append chunks [from, to) one at a time and stamp each with `seed`.
bool append_stamped(Harness & h, KvFlashPager & p, int from, int to, int seed) {
    for (int c = from; c < to; c++) {
        if (!p.alloc_span((int64_t)c * kChunk, kChunk)) return false;
        stamp_chunk(h, p, c, seed);
    }
    return true;
}

bool all_match(Harness & h, KvFlashPager & p, int n, int seed) {
    p.synchronize_paging();
    for (int c = 0; c < n; c++)
        if (!chunk_matches(h, p, c, seed)) return false;
    return true;
}
}

// Codex's case: 8 blocks, chunks 0-5, then 6-13, then truncate to 6 chunks.
// The retained map used to be [0,7,1,2,3,4]; every retained chunk must now sit
// in block c with its own bytes, and the next appends get blocks 6, 7.
TEST_CASE(KvflashPagerSerdeFixture, truncate_relayout_codex_case_moves_bytes_to_identity_blocks) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(append_stamped(A, p, 0, 6, 1));
    REQUIRE(append_stamped(A, p, 6, 14, 1));
    p.truncate(6 * kChunk);
    CHECK(layout_of(p, 6) == (std::vector<int>{0, 1, 2, 3, 4, 5}));
    CHECK(all_match(A, p, 6, 1));
    REQUIRE(append_chunks(p, 6, 8));
    CHECK(layout_of(p, 8) == (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    CHECK(all_match(A, p, 6, 1));
}

// A pure swap (a cycle with no free block to start from) plus a partial last
// chunk: bytes survive, rows >= P are zero, and the vacated blocks are zero.
TEST_CASE(KvflashPagerSerdeFixture, truncate_relayout_resolves_a_swap_cycle_without_losing_bytes) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(append_stamped(A, p, 0, 8, 2));
    REQUIRE(p.page_out(1));
    REQUIRE(p.page_out(2));
    REQUIRE(p.page_in(1));
    REQUIRE(p.page_in(2));
    p.synchronize_paging();
    REQUIRE(p.block_of(1) == 2);
    REQUIRE(p.block_of(2) == 1);                         // chunks 1 and 2 swapped
    p.truncate(5 * kChunk);
    CHECK(layout_of(p, 5) == (std::vector<int>{0, 1, 2, 3, 4}));
    CHECK(all_match(A, p, 5, 2));
    for (int b = 5; b < 8; b++) {                        // vacated blocks hold zeros
        bool zero = true;
        for_each_segment(A, b, [&](ggml_tensor * t, size_t off, size_t len) {
            std::vector<uint8_t> v(len);
            ggml_backend_tensor_get(t, v.data(), off, len);
            for (uint8_t x : v) zero = zero && x == 0;
        });
        CHECK(zero);
    }
}

// ── recency pool before the mode-4 scoring pass ──────────────────────────
// truncate(C) with C not fitting keeps every current resident, so the previous
// request's recalled (hook-picked) blocks stay resident into the next request's
// scoring pass. reselect_recent() (LUCE_KVFLASH_SCORE_POOL_RECENT=1) restores the
// pool a pager that only ever saw C tokens under plain LRU holds.
namespace {
std::vector<bool> resident_set(const KvFlashPager & p, int n) {
    std::vector<bool> v;
    for (int c = 0; c < n; c++) v.push_back(p.is_resident(c));
    return v;
}

// Previous request: 14 chunks, a hook-picked old block (2) recalled, hook gone.
void leave_stale_pick(KvFlashPager & p) {
    must(append_chunks(p, 0, 14));
    p.score_hook = [](int c) { return c == 2 ? 100.0f : (float)c; };
    p.reselect(0);
    p.score_hook = nullptr;
}
}

TEST_CASE(KvflashPagerSerdeFixture, recent_pool_after_truncate_equals_fresh_lru_pager) {
    const int C = 10 * kChunk;                         // 10 chunks > 8 blocks
    Harness F;
    TestPager fresh;
    REQUIRE(fresh.attach(make_cfg(), F.k, F.v));
    REQUIRE(append_chunks(fresh, 0, 10));
    const std::vector<bool> want = resident_set(fresh, 10);

    {   // Flag off: the stale pick survives truncate (the bug the flag fixes).
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        leave_stale_pick(p);
        p.truncate(C);
        CHECK(p.is_resident(2));
        CHECK(resident_set(p, 10) != want);
    }
    {   // Flag on: recency reselect gives the fresh LRU resident set, and no hook is left.
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        leave_stale_pick(p);
        p.truncate(C);
        p.reselect_recent();
        CHECK(!p.score_hook);
        CHECK(resident_set(p, 10) == want);
    }
}

// A prefix that fits is fully resident (relayout_identity_): the recency step has
// nothing to do, so applying it on every pooled request changes no request.
TEST_CASE(KvflashPagerSerdeFixture, reselect_recent_on_a_fitting_prefix_is_a_noop) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    leave_stale_pick(p);
    p.truncate(6 * kChunk);                            // 6 chunks <= 8 blocks
    const std::vector<bool> before = resident_set(p, 6);
    CHECK(p.reselect_recent() == 0);
    CHECK(resident_set(p, 6) == before);
    CHECK(!p.score_hook);
}

// A bundle adoption admits a chunk count first: deserialize refuses a blob of
// another count before it touches any state, and the streamed serialize gives
// the serialize() bytes without the whole-blob copy.
TEST_CASE(KvflashPagerSerdeFixture, deserialize_checks_the_expected_chunk_count_first_and_serialize_streams_the_same_bytes) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 1);
    const std::vector<uint8_t> blob = pa.serialize();
    CHECK(pa.serialized_bytes(6) == blob.size());
    CHECK(pa.serialized_bytes(2) == kHdr + 2 * (size_t)(kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2) + 2 * kLedgerEntry);
    std::vector<uint8_t> streamed;
    REQUIRE(pa.serialize_to([&](const uint8_t * d, size_t n) { streamed.insert(streamed.end(), d, d + n); return true; }));
    CHECK(streamed == blob);
    CHECK(!pa.serialize_to([&](const uint8_t *, size_t) { return false; }));   // a sink failure is reported

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 2 * kChunk));
    CHECK(!pb.deserialize(blob.data(), blob.size(), 5));
    CHECK(pb.n_chunks() == 2);                                                 // untouched
    CHECK(pb.is_resident(0) && pb.is_resident(1));
    REQUIRE(pb.deserialize(blob.data(), blob.size(), 6));
    CHECK(pb.n_chunks() == 6);
}
