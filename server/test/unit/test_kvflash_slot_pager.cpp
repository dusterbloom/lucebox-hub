// Unit tests for KvFlashSlotPager (kvflash_slot_pager.h): the existing pager
// confined to its slice [sQ, (s+1)Q) chunks of one shared KV allocation
// (Track C, C0 R1). Logic cases run on the no-tensor path; byte cases use CPU
// ggml tensors, so no GPU is needed.
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"
#include "../../src/common/kvflash_slot_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashSlotPagerFixture {};

constexpr int kHeadDim = 4, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
constexpr int kN = 3, kQ = 8;                        // 3 slots, 8 chunks each (floor is 7)

// One arena: N x Q x 64 rows plus the scratch block, all layers, K and V.
struct Arena {
    ggml_context *          ctx = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    explicit Arena(int n_slots = kN, int rows_override = 0) {
        const int rows = rows_override ? rows_override : (int)kvflash_arena_rows(n_slots, kQ);
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kNLayer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kNLayer; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, rows, kNHeadKv));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, rows, kNHeadKv));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        for (ggml_tensor * t : k) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
        for (ggml_tensor * t : v) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    ~Arena() {
        if (buf)     ggml_backend_buffer_free(buf);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    Arena(const Arena &) = delete;
    Arena & operator=(const Arena &) = delete;
};

struct Slot : KvFlashSlotPager {
    Slot() { set_pinned_host_backing(pager_pinned_host_available()); }
};

KvFlashConfig make_cfg() {
    KvFlashConfig c;
    c.chunk_tokens = kChunk;
    c.sink_chunks = 1;
    c.tail_window_chunks = 4;
    return c;
}

void must(bool ok) { if (!ok) throw std::runtime_error("kvflash slot pager helper precondition failed"); }

// No-tensor slot pagers over a notional arena.
struct Fleet {
    std::vector<std::unique_ptr<Slot>> s;
    explicit Fleet(int n = kN) {
        for (int i = 0; i < n; i++) {
            s.push_back(std::make_unique<Slot>());
            must(s.back()->attach_slice(i, n, kQ, make_cfg(), {}, {}));
        }
    }
    Slot & operator[](int i) { return *s[(size_t)i]; }
};

bool row_in_slice(int64_t row, int slot) {
    const KvFlashSlice sl = kvflash_slice(slot, kQ);
    return row >= sl.row_lo && row < sl.row_hi;
}

// Append tokens [from, to) to a slot; every returned row must stay in its slice.
bool append(Slot & p, int slot, int from, int to) {
    for (int pos = from; pos < to; pos++) {
        const int64_t row = p.slot_for(pos);
        if (row < 0 || !row_in_slice(row, slot)) return false;
    }
    return true;
}

std::vector<int> blocks_of(const Slot & p) {
    std::vector<int> b;
    for (int c = 0; c < p.n_chunks(); c++) b.push_back(p.global_block_of(c));
    return b;
}

// Bytes of arena rows [row_lo, row_hi) across every tensor and head.
std::vector<uint8_t> snapshot(Arena & a, int64_t row_lo, int64_t row_hi) {
    std::vector<uint8_t> out;
    for (int kv = 0; kv < 2; kv++) {
        for (ggml_tensor * t : kv == 0 ? a.k : a.v) {
            for (int h = 0; h < kNHeadKv; h++) {
                const size_t off = (size_t)row_lo * t->nb[1] + (size_t)h * t->nb[2];
                const size_t len = (size_t)(row_hi - row_lo) * t->nb[1];
                const size_t at = out.size();
                out.resize(at + len);
                ggml_backend_tensor_get(t, out.data() + at, off, len);
            }
        }
    }
    return out;
}
std::vector<uint8_t> snapshot_slice(Arena & a, int slot) {
    const KvFlashSlice sl = kvflash_slice(slot, kQ);
    return snapshot(a, sl.row_lo, sl.row_hi);
}
std::vector<uint8_t> snapshot_scratch(Arena & a) {
    const int64_t r = kvflash_scratch_row(kN, kQ);
    return snapshot(a, r, r + kChunk);
}

uint8_t pat(int c, int seed, size_t i) {
    uint32_t x = (uint32_t)c * 2654435761u ^ (uint32_t)seed * 40503u ^ (uint32_t)i * 2246822519u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return (uint8_t)(x >> 24) | 1u;   // never zero
}

template <class F>
void for_each_segment(Arena & a, int64_t first_row, F && fn) {
    for (int l = 0; l < kNLayer; l++) {
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * t = kv == 0 ? a.k[(size_t)l] : a.v[(size_t)l];
            for (int h = 0; h < kNHeadKv; h++)
                fn(t, (size_t)first_row * t->nb[1] + (size_t)h * t->nb[2], (size_t)kChunk * t->nb[1]);
        }
    }
}

// The pattern depends on (chunk, seed), not on the slot: the same content in
// two different slices must produce the same blob.
void stamp_chunk(Arena & a, const Slot & p, int c, int seed) {
    const int64_t row = p.slot_of((int64_t)c * kChunk);
    must(row >= 0);
    size_t n = 0;
    for_each_segment(a, row, [&](ggml_tensor * t, size_t off, size_t len) {
        std::vector<uint8_t> b(len);
        for (size_t i = 0; i < len; i++) b[i] = pat(c, seed, n + i);
        ggml_backend_tensor_set(t, b.data(), off, len);
        n += len;
    });
}
bool chunk_matches(Arena & a, const Slot & p, int c, int seed) {
    const int64_t row = p.slot_of((int64_t)c * kChunk);
    if (row < 0) return false;
    size_t n = 0;
    bool ok = true;
    for_each_segment(a, row, [&](ggml_tensor * t, size_t off, size_t len) {
        std::vector<uint8_t> b(len);
        ggml_backend_tensor_get(t, b.data(), off, len);
        for (size_t i = 0; i < len; i++) ok = ok && b[i] == pat(c, seed, n + i);
        n += len;
    });
    return ok;
}

// 6 chunks stamped, chunks 2 and 3 paged out, chunk 2 recalled into another block.
void build_source(Arena & a, Slot & p, int slot, int seed) {
    must(p.attach_slice(slot, kN, kQ, make_cfg(), a.k, a.v));
    must(p.alloc_span(0, 6 * kChunk));
    for (int c = 0; c < 6; c++) stamp_chunk(a, p, c, seed);
    must(p.page_out(2));
    must(p.page_out(3));
    must(p.page_in(2));
    p.synchronize_paging();
    for (int c = 0; c < 6; c++) p.set_chunk_score(c, 10.0f * (float)c);
}

}  // namespace

TEST_CASE(KvflashSlotPagerFixture, attach_slice_refuses_a_misfit) {
    Slot p;
    CHECK(p.attach_slice(0, kN, kQ, make_cfg(), {}, {}));
    CHECK(!p.attach_slice(kN, kN, kQ, make_cfg(), {}, {}));       // slot out of range
    CHECK(!p.attach_slice(-1, kN, kQ, make_cfg(), {}, {}));
    CHECK(!p.attach_slice(0, kN, 6, make_cfg(), {}, {}));          // below sink + tail + 2
    Arena small(kN, (kN - 1) * kQ * kChunk);                       // one slice short, no scratch
    CHECK(!p.attach_slice(kN - 1, kN, kQ, make_cfg(), small.k, small.v));
    Arena fits;
    CHECK(p.attach_slice(kN - 1, kN, kQ, make_cfg(), fits.k, fits.v));
}

TEST_CASE(KvflashSlotPagerFixture, quota_is_exact_and_rows_are_the_slice) {
    Fleet f;
    for (int s = 0; s < kN; s++) {
        Slot & p = f[s];
        CHECK(p.pool_tokens() == kQ * kChunk);
        CHECK(p.n_pool_blocks() == kQ);
        REQUIRE(append(p, s, 0, kQ * kChunk));                     // exactly Q chunks
        CHECK(p.resident_blocks() == kQ);
        std::vector<int> blocks = blocks_of(p);
        std::sort(blocks.begin(), blocks.end());
        const KvFlashSlice sl = kvflash_slice(s, kQ);
        for (int i = 0; i < kQ; i++) CHECK(blocks[(size_t)i] == sl.block_lo + i);   // all of it, nothing else
    }
    // One chunk past the quota evicts inside the slice; residency never exceeds Q.
    REQUIRE(append(f[1], 1, kQ * kChunk, (kQ + 1) * kChunk));
    CHECK(f[1].resident_blocks() == kQ);
    CHECK(f[1].stats().page_outs == 1);
}

TEST_CASE(KvflashSlotPagerFixture, victims_never_cross_slots) {
    Fleet f;
    REQUIRE(append(f[0], 0, 0, kQ * kChunk));
    REQUIRE(append(f[2], 2, 0, kQ * kChunk));
    const std::vector<int> b0 = blocks_of(f[0]), b2 = blocks_of(f[2]);
    // Slot 1 streams 5x its quota, with a score hook that prefers other slots' chunk ids.
    f[1].score_hook = [](int c) { return (float)-c; };
    REQUIRE(append(f[1], 1, 0, 5 * kQ * kChunk));
    CHECK(f[1].stats().page_outs >= 4 * kQ);
    CHECK(f[0].stats().page_outs == 0);
    CHECK(f[2].stats().page_outs == 0);
    CHECK(blocks_of(f[0]) == b0);                                  // untouched, same blocks
    CHECK(blocks_of(f[2]) == b2);
    // Recall inside slot 1 (reselect) also stays in the slice.
    f[1].score_hook = [](int c) { return (float)c; };
    f[1].reselect(0);
    for (int c = 0; c < f[1].n_chunks(); c++) {
        const int g = f[1].global_block_of(c);
        if (g >= 0) CHECK(g >= kvflash_slice(1, kQ).block_lo && g < kvflash_slice(1, kQ).block_hi);
    }
    CHECK(blocks_of(f[0]) == b0);
}

TEST_CASE(KvflashSlotPagerFixture, pins_hold_per_slot) {
    Fleet f;
    f[0].pin_range(2 * kChunk, 3 * kChunk);                        // chunk 2, slot 0 only
    CHECK(f[0].is_pinned(2));
    CHECK(!f[1].is_pinned(2));
    REQUIRE(append(f[0], 0, 0, 30 * kChunk));
    REQUIRE(append(f[1], 1, 0, 30 * kChunk));
    CHECK(f[0].is_resident(2));                                    // the pin held through 30 chunks
    CHECK(!f[1].is_resident(2));                                   // same load, no pin: evicted
    // Resetting slot 1 drops only slot 1's pins.
    f[1].pin_range(5 * kChunk, 6 * kChunk);
    f[1].reset();
    CHECK(!f[1].is_pinned(5));
    CHECK(f[0].is_pinned(2));
}

TEST_CASE(KvflashSlotPagerFixture, truncate_is_per_slot) {
    Fleet f;
    for (int s = 0; s < kN; s++) REQUIRE(append(f[s], s, 0, 12 * kChunk));
    const std::vector<int> b0 = blocks_of(f[0]), b2 = blocks_of(f[2]);
    const int64_t out0 = f[0].stats().page_outs, out2 = f[2].stats().page_outs;
    f[1].truncate(3 * kChunk + 10);
    CHECK(f[1].n_chunks() == 4);
    CHECK(blocks_of(f[0]) == b0);
    CHECK(blocks_of(f[2]) == b2);
    CHECK(f[0].stats().page_outs == out0);
    CHECK(f[2].stats().page_outs == out2);
    // The kept prefix is identity inside the slice, and later appends stay in it.
    const KvFlashSlice sl = kvflash_slice(1, kQ);
    for (int c = 0; c < 4; c++) CHECK(f[1].slot_of((int64_t)c * kChunk) == sl.row_lo + (int64_t)c * kChunk);
    REQUIRE(append(f[1], 1, 3 * kChunk + 10, 14 * kChunk));
    f[1].truncate(0);
    CHECK(f[1].n_chunks() == 0);
    CHECK(f[1].resident_blocks() == 0);
    CHECK(blocks_of(f[0]) == b0);
}

TEST_CASE(KvflashSlotPagerFixture, serialize_roundtrips_per_slot_and_the_blob_is_slot_independent) {
    Arena A;
    Slot pa;
    build_source(A, pa, 1, 7);
    REQUIRE(!pa.is_resident(3));
    const std::vector<uint8_t> blob = pa.serialize();

    // Same content built in slot 0 of another arena: byte-identical blob.
    Arena C;
    Slot pc;
    build_source(C, pc, 0, 7);
    CHECK(pc.serialize() == blob);

    // Restore into slot 1 of a fresh arena: bytes land in slice 1 only.
    Arena B;
    Slot pb;
    REQUIRE(pb.attach_slice(1, kN, kQ, make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.n_chunks() == 6);
    for (int c : {0, 1, 2, 4, 5}) {
        CHECK(chunk_matches(B, pb, c, 7));
        CHECK(row_in_slice(pb.slot_of((int64_t)c * kChunk), 1));
    }
    CHECK(!pb.is_resident(3));
    REQUIRE(pb.page_in(3));                                        // host copy survived the trip
    pb.synchronize_paging();
    CHECK(chunk_matches(B, pb, 3, 7));
    CHECK(row_in_slice(pb.slot_of(3 * kChunk), 1));
    const std::vector<uint8_t> zero_slice(snapshot_slice(B, 0).size(), 0);
    CHECK(snapshot_slice(B, 0) == zero_slice);
    CHECK(snapshot_slice(B, 2) == zero_slice);
    CHECK(snapshot_scratch(B) == std::vector<uint8_t>(snapshot_scratch(B).size(), 0));

    // A blob is not tied to the slot it came from: restore it into slot 2.
    Slot p2;
    REQUIRE(p2.attach_slice(2, kN, kQ, make_cfg(), B.k, B.v));
    REQUIRE(p2.deserialize(blob.data(), blob.size()));
    for (int c : {0, 1, 2, 4, 5}) {
        CHECK(chunk_matches(B, p2, c, 7));
        CHECK(row_in_slice(p2.slot_of((int64_t)c * kChunk), 2));
    }
    CHECK(chunk_matches(B, pb, 0, 7));                             // slot 1's rows still intact
}

TEST_CASE(KvflashSlotPagerFixture, freeing_or_resetting_one_slot_leaves_the_others_byte_identical) {
    Arena A;
    Slot p0, p1, p2;
    build_source(A, p0, 0, 1);
    build_source(A, p1, 1, 2);
    build_source(A, p2, 2, 3);
    const auto s0 = snapshot_slice(A, 0), s2 = snapshot_slice(A, 2), sc = snapshot_scratch(A);
    auto others_intact = [&]() {
        return snapshot_slice(A, 0) == s0 && snapshot_slice(A, 2) == s2 && snapshot_scratch(A) == sc;
    };
    CHECK(others_intact());

    // Heavy paging in slot 1: page-outs zero blocks, recalls rewrite them.
    for (int c = 6; c < 20; c++) {
        REQUIRE(p1.alloc_span(c * kChunk, kChunk));
        stamp_chunk(A, p1, c, 2);
    }
    p1.synchronize_paging();
    CHECK(others_intact());

    p1.truncate(2 * kChunk + 5);                                   // frees blocks, zeroes the tail rows
    p1.synchronize_paging();
    CHECK(others_intact());

    p1.reset();
    p1.zero_free_blocks();                                         // wipes the whole slice
    CHECK(others_intact());
    CHECK(snapshot_slice(A, 1) == std::vector<uint8_t>(snapshot_slice(A, 1).size(), 0));

    // Slot 1 is reusable: a new request, and the others still unchanged.
    REQUIRE(p1.alloc_span(0, 3 * kChunk));
    for (int c = 0; c < 3; c++) stamp_chunk(A, p1, c, 9);
    CHECK(chunk_matches(A, p1, 1, 9));
    CHECK(others_intact());
    CHECK(chunk_matches(A, p0, 0, 1));
    CHECK(chunk_matches(A, p2, 0, 3));
}
