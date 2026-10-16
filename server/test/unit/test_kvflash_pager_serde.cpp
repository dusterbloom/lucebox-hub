// Unit tests for KvFlashPager serialize/deserialize, floor_to_chunk, critical
// chunk pinning and the per-chunk score ledger (kvflash_pager.h). CPU ggml
// backend, no GPU: the cache tensors live in a CPU buffer, so every code path
// that moves bytes between pool and host backing is exercised for real.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"

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
// KVFLASH1 blob header: 8-byte magic + 8 x uint32.
constexpr size_t kHdr = 8 + 8 * 4;
constexpr size_t kLedgerEntry = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// pool 512 / chunk 64 = 8 blocks; sink 1 + tail 4 + 2 = 7 <= 8 blocks.
struct Harness {
    ggml_context *          ctx = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    Harness() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kNLayer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kNLayer; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, kPool, kNHeadKv));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, kPool, kNHeadKv));
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

// Deterministic per-(logical chunk, seed) byte pattern, never all-zero.
uint8_t pat(int c, int seed, size_t i) { return (uint8_t)(c * 53 + seed * 29 + i * 31 + 7); }

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
    KvFlashPager pa;
    build_source(A, pa, 1);
    REQUIRE(pa.block_of(2) >= 0 && pa.block_of(2) != 2);  // relocated, not identity
    REQUIRE(!pa.is_resident(3));                          // paged out at serialize time

    const std::vector<uint8_t> blob = pa.serialize();
    REQUIRE(blob.size() == kHdr + 6 * (size_t)(kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2) +
                           6 * kLedgerEntry);
    REQUIRE(std::memcmp(blob.data(), "1HSALFVK", 8) == 0);  // "KVFLASH1" little-endian u64

    Harness B;
    KvFlashPager pb;
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
    KvFlashPager pa;
    build_source(A, pa, 2);
    for (int c : {0, 1, 2, 4, 5}) REQUIRE(pa.page_out(c));
    REQUIRE(pa.resident_blocks() == 0);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness B;
    KvFlashPager pb;
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
    KvFlashPager pa;
    build_source(A, pa, 3);
    const std::vector<uint8_t> full = pa.serialize();
    const std::vector<uint8_t> part = pa.serialize(4);
    REQUIRE(part.size() < full.size());

    Harness B;
    KvFlashPager pb;
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
    KvFlashPager pa;
    build_source(A, pa, 4);
    pa.set_chunk_score(4, -3.5f);
    CHECK(pa.chunk_score(5) == 50.0f);
    CHECK(pa.chunk_score(99) == kNegInf);   // out of range: unset sentinel
    pa.set_chunk_score(99, 1.0f);           // out of range: ignored, no crash
    pa.set_chunk_score(-1, 1.0f);

    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    for (int c = 0; c < 6; c++) CHECK(pb.chunk_score(c) == pa.chunk_score(c));
    CHECK(pb.chunk_score(3) == 30.0f);      // host-backed chunk keeps its score
    CHECK(pb.chunk_score(4) == -3.5f);
    // A chunk never scored stays at the unset sentinel through the roundtrip.
    Harness C;
    KvFlashPager pc;
    REQUIRE(pc.attach(make_cfg(), C.k, C.v));
    REQUIRE(pc.alloc_span(0, 2 * kChunk));
    const std::vector<uint8_t> blob2 = pc.serialize();
    REQUIRE(pb.deserialize(blob2.data(), blob2.size()));
    CHECK(pb.chunk_score(0) == kNegInf);
    CHECK(pb.chunk_score(1) == kNegInf);
}

TEST_CASE(KvflashPagerSerdeFixture, serialized_dtype_is_recorded) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 5);
    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.serialized_kv_k_type() == (uint32_t)GGML_TYPE_F16);
}

TEST_CASE(KvflashPagerSerdeFixture, malformed_blobs_are_rejected_without_touching_state) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 6);
    const std::vector<uint8_t> good = pa.serialize();

    Harness B;
    KvFlashPager pb;
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
    KvFlashPager p;
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
    KvFlashPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    REQUIRE(p.alloc_span(8 * kChunk, kChunk));   // evicts chunk 1 (sink 0 protected)
    CHECK(p.is_resident(0));
    CHECK(!p.is_resident(1));
    CHECK(p.is_resident(8));
}

TEST_CASE(KvflashPagerSerdeFixture, reselect_recalls_top_scored_host_chunk_and_keeps_tail) {
    Harness A;
    KvFlashPager p;
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
        KvFlashPager p;
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
    KvFlashPager p;
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
    KvFlashPager p;
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
    KvFlashPager p;
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
    KvFlashPager p;
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
    KvFlashPager p;
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
    KvFlashPager p;
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
        KvFlashPager p;
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
        KvFlashPager p;
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
        KvFlashPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 16));
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);
        CHECK(got.after_append == fresh.after_append);
    }
    {   // History B: a different length with an interleaved truncate, LRU only.
        Harness A;
        KvFlashPager p;
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
    KvFlashPager p;
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
    KvFlashPager p;
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
