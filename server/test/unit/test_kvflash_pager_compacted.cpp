// Unit tests for compacted chunks in the KvFlashPager (kvflash_pager.h): a
// compacted block is one ordinary pool chunk (64 rows) that stands for `span`
// logical chunks (640 tokens at span 10), carries beta per (layer, KV head,
// slot) for the kv_bias upload, and pages and serializes like any chunk.
// CPU ggml backend, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerCompactedFixture {};

constexpr int kHeadDim = 4, kPool = 512, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
constexpr size_t kHdr = 8 + 8 * 4;
constexpr size_t kChunkBytes = (size_t)kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2;

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

void must(bool ok) { if (!ok) throw std::runtime_error("kvflash compacted helper precondition failed"); }

// beta for (layer, head, slot), laid out as the pager takes it. Slot 16 of
// (layer 0, head 0) is exactly 0 on purpose: a live slot with beta 0.
float beta_at(int l, int h, int i) { return 0.5f * (float)i - 8.0f + 100.0f * l + 10.0f * h; }

std::vector<float> make_beta() {
    std::vector<float> b((size_t)kNLayer * kNHeadKv * kChunk);
    for (int l = 0; l < kNLayer; l++)
        for (int h = 0; h < kNHeadKv; h++)
            for (int i = 0; i < kChunk; i++) b[((size_t)l * kNHeadKv + h) * kChunk + i] = beta_at(l, h, i);
    return b;
}

uint8_t pat(int block, int t, int h, size_t i) { return (uint8_t)(block * 29 + t * 13 + h * 7 + i * 3 + 1); }

// Stamp every (tensor, head) segment of pool block `blk` with a block-specific pattern.
void stamp(Harness & hs, int blk) {
    int t = 0;
    for (int l = 0; l < kNLayer; l++) for (int kv = 0; kv < 2; kv++, t++) {
        ggml_tensor * x = kv == 0 ? hs.k[l] : hs.v[l];
        for (int h = 0; h < kNHeadKv; h++) {
            std::vector<uint8_t> b((size_t)kChunk * x->nb[1]);
            for (size_t i = 0; i < b.size(); i++) b[i] = pat(blk, t, h, i);
            ggml_backend_tensor_set(x, b.data(), (size_t)blk * kChunk * x->nb[1] + (size_t)h * x->nb[2], b.size());
        }
    }
}

// True when pool block `blk` holds the pattern stamped into block `from`.
bool holds(Harness & hs, int blk, int from) {
    int t = 0;
    for (int l = 0; l < kNLayer; l++) for (int kv = 0; kv < 2; kv++, t++) {
        ggml_tensor * x = kv == 0 ? hs.k[l] : hs.v[l];
        for (int h = 0; h < kNHeadKv; h++) {
            std::vector<uint8_t> b((size_t)kChunk * x->nb[1]);
            ggml_backend_tensor_get(x, b.data(), (size_t)blk * kChunk * x->nb[1] + (size_t)h * x->nb[2], b.size());
            for (size_t i = 0; i < b.size(); i++) if (b[i] != pat(from, t, h, i)) return false;
        }
    }
    return true;
}

// Chunks 0..5 resident in identity blocks, chunk 2..4 compacted into one block (span 3).
void build(Harness & hs, KvFlashPager & p) {
    must(p.attach(make_cfg(), hs.k, hs.v));
    must(p.alloc_span(0, 6 * kChunk));
    for (int b = 0; b < 6; b++) stamp(hs, b);
    must(p.mark_compacted(2, 3, make_beta()));
}

std::vector<float> bias_of(const KvFlashPager & p, int layer) {
    std::vector<float> b((size_t)kPool * kNHeadKv, 12345.f);
    p.fill_kv_bias(layer, b.data());
    return b;
}

// Expected bias of `layer`: beta on the columns of `blk` (-1 = none), 0 elsewhere.
std::vector<float> expect_bias(int layer, int blk) {
    std::vector<float> b((size_t)kPool * kNHeadKv, 0.0f);
    if (blk >= 0)
        for (int h = 0; h < kNHeadKv; h++)
            for (int i = 0; i < kChunk; i++) b[(size_t)h * kPool + blk * kChunk + i] = beta_at(layer, h, i);
    return b;
}

} // namespace

TEST_CASE(KvflashPagerCompactedFixture, fill_kv_bias_puts_beta_in_compacted_columns_and_0_in_exact_ones) {
    Harness hs;
    KvFlashPager p;
    build(hs, p);

    CHECK(p.any_compacted());
    CHECK(p.is_compacted(2));
    CHECK(p.block_of(2) == 2);
    for (int l = 0; l < kNLayer; l++) {
        const auto got = bias_of(p, l);
        const auto want = expect_bias(l, 2);
        CHECK(got == want);   // every column written: beta on block 2's 64 columns per head, 0 elsewhere
    }
    CHECK(bias_of(p, 0)[(size_t)0 * kPool + 2 * kChunk + 16] == 0.0f);   // live slot with beta exactly 0
}

TEST_CASE(KvflashPagerCompactedFixture, logical_positions_count_chunks_not_rows) {
    Harness hs;
    KvFlashPager p;
    build(hs, p);

    CHECK(p.n_chunks() == 6);                 // 384 logical tokens, 4 pool blocks
    CHECK(p.resident_blocks() == 4);
    CHECK(p.compacted_span(2) == 3);
    CHECK(p.is_covered(3));
    CHECK(p.is_covered(4));
    CHECK(!p.is_covered(2));
    CHECK(!p.is_covered(5));
    CHECK(p.slot_of(2 * kChunk + 5) == -1);   // no per-token slot inside a compacted block
    CHECK(p.slot_of(3 * kChunk) == -1);
    CHECK(p.slot_of(5 * kChunk + 1) == 5 * kChunk + 1);
    CHECK(p.slot_for(3 * kChunk) == -1);      // sealed positions are not writable
    CHECK(!p.is_identity());
    CHECK(!p.identity_prefix_covers(6 * kChunk));

    // Slot positions: a compacted row stands somewhere inside its 192 positions,
    // in order, so the causal mask sees it for every later query.
    std::vector<int32_t> pos(kPool);
    p.fill_slot_pos(pos.data());
    for (int i = 0; i < kChunk; i++) CHECK(pos[2 * kChunk + i] == 2 * kChunk + i * 3);
    for (int i = 0; i < kChunk; i++) CHECK(pos[3 * kChunk + i] == -1);
    CHECK(pos[5 * kChunk] == 5 * kChunk);

    // The shared mask: compacted block open, released blocks closed.
    std::vector<uint16_t> m(kPool);
    p.fill_slot_mask(m.data());
    CHECK(m[2 * kChunk] == 0x0000);
    CHECK(m[3 * kChunk] == 0xFC00);
    CHECK(m[4 * kChunk + 63] == 0xFC00);
    CHECK(m[5 * kChunk] == 0x0000);

    // The append head continues at logical position 384, in the next free block.
    CHECK(p.slot_for(6 * kChunk) >= 0);
    CHECK(p.n_chunks() == 7);
    CHECK(p.resident_blocks() == 5);
}

TEST_CASE(KvflashPagerCompactedFixture, mark_compacted_rejects_bad_geometry_and_changes_nothing) {
    Harness hs;
    KvFlashPager p;
    must(p.attach(make_cfg(), hs.k, hs.v));
    must(p.alloc_span(0, 6 * kChunk));
    const uint64_t e0 = p.epoch();
    const auto beta = make_beta();

    CHECK(!p.mark_compacted(2, 0, beta));                      // empty span
    CHECK(!p.mark_compacted(4, 3, beta));                      // runs past the materialized chunks
    CHECK(!p.mark_compacted(2, 1, std::vector<float>(10)));    // wrong beta size
    must(p.page_out(3));
    CHECK(!p.mark_compacted(3, 1, beta));                      // the head chunk must be resident
    must(p.page_in(3));
    must(p.mark_compacted(2, 2, beta));
    const uint64_t e1 = p.epoch();
    CHECK(e1 > e0);
    CHECK(!p.mark_compacted(1, 2, beta));                      // overlaps the compacted block
    CHECK(!p.mark_compacted(3, 1, beta));                      // inside it
    CHECK(p.epoch() == e1);
    CHECK(p.resident_blocks() == 5);
}

TEST_CASE(KvflashPagerCompactedFixture, truncate_inside_a_compacted_chunk_drops_it_and_returns_its_start) {
    {   // inside the block (chunk 3 of the 2..4 block)
        Harness hs; KvFlashPager p; build(hs, p);
        CHECK(p.truncate(3 * kChunk + 10) == 2 * kChunk);
        CHECK(p.n_chunks() == 2);
        CHECK(!p.any_compacted());
        CHECK(p.resident_blocks() == 2);
        CHECK(bias_of(p, 0) == expect_bias(0, -1));
        CHECK(bias_of(p, 1) == expect_bias(1, -1));
    }
    {   // exactly at its start
        Harness hs; KvFlashPager p; build(hs, p);
        CHECK(p.truncate(2 * kChunk) == 2 * kChunk);
        CHECK(p.n_chunks() == 2);
        CHECK(!p.any_compacted());
    }
    {   // at its end the block survives whole
        Harness hs; KvFlashPager p; build(hs, p);
        CHECK(p.truncate(5 * kChunk) == 5 * kChunk);
        CHECK(p.n_chunks() == 5);
        CHECK(p.is_compacted(2));
        CHECK(bias_of(p, 1) == expect_bias(1, 2));
    }
    {   // exact chunks and out-of-range cuts return P unchanged
        Harness hs; KvFlashPager p; build(hs, p);
        CHECK(p.truncate(5 * kChunk + 7) == 5 * kChunk + 7);
        CHECK(p.is_compacted(2));
        CHECK(p.truncate(10 * kChunk) == 10 * kChunk);
        CHECK(p.truncate(-5) == 0);
        CHECK(p.n_chunks() == 0);
        CHECK(!p.any_compacted());
    }
}

TEST_CASE(KvflashPagerCompactedFixture, page_out_and_page_in_keep_beta_and_the_flag) {
    Harness hs;
    KvFlashPager p;
    build(hs, p);

    CHECK(p.page_out(2));
    CHECK(p.is_compacted(2));
    CHECK(p.compacted_span(2) == 3);
    CHECK(p.is_covered(3));
    CHECK(bias_of(p, 0) == expect_bias(0, -1));   // the block left the pool: its columns are 0 (mask closes them)

    // Take the block it left, so the recall lands somewhere else.
    CHECK(p.slot_for(6 * kChunk) >= 0);
    CHECK(p.page_in(2));
    p.synchronize_paging();
    const int blk = p.block_of(2);
    CHECK(blk >= 0);
    CHECK(blk != 2);
    CHECK(p.is_compacted(2));
    CHECK(bias_of(p, 0) == expect_bias(0, blk));
    CHECK(bias_of(p, 1) == expect_bias(1, blk));
    CHECK(holds(hs, blk, 2));                     // the 64 compacted rows came back bit-exact
}

TEST_CASE(KvflashPagerCompactedFixture, serialize_then_deserialize_keeps_flags_and_beta) {
    Harness hs;
    KvFlashPager p;
    build(hs, p);
    const std::vector<uint8_t> blob = p.serialize();

    Harness hs2;
    KvFlashPager q;
    must(q.attach(make_cfg(), hs2.k, hs2.v));
    CHECK(q.deserialize(blob.data(), blob.size()));
    CHECK(q.n_chunks() == 6);
    CHECK(q.is_compacted(2));
    CHECK(q.compacted_span(2) == 3);
    CHECK(q.is_covered(3));
    CHECK(q.is_covered(4));
    CHECK(q.resident_blocks() == 4);
    const int blk = q.block_of(2);
    CHECK(blk >= 0);
    CHECK(bias_of(q, 0) == expect_bias(0, blk));
    CHECK(bias_of(q, 1) == expect_bias(1, blk));
    CHECK(holds(hs2, blk, 2));
    CHECK(q.serialize() == blob);                 // a second round trip is a fixed point

    // A compacted chunk that sat on host stays there and keeps its beta.
    must(p.page_out(2));
    const std::vector<uint8_t> cold = p.serialize();
    Harness hs3;
    KvFlashPager r;
    must(r.attach(make_cfg(), hs3.k, hs3.v));
    CHECK(r.deserialize(cold.data(), cold.size()));
    CHECK(r.is_compacted(2));
    CHECK(r.block_of(2) == -1);
    CHECK(r.page_in(2));
    r.synchronize_paging();
    CHECK(bias_of(r, 1) == expect_bias(1, r.block_of(2)));
    CHECK(holds(hs3, r.block_of(2), 2));

    // A cut that would split the block stops at its start.
    const std::vector<uint8_t> cut = q.serialize(4);
    Harness hs4;
    KvFlashPager s;
    must(s.attach(make_cfg(), hs4.k, hs4.v));
    CHECK(s.deserialize(cut.data(), cut.size()));
    CHECK(s.n_chunks() == 2);
    CHECK(!s.any_compacted());
}

TEST_CASE(KvflashPagerCompactedFixture, a_bad_compaction_trailer_is_rejected_without_touching_state) {
    Harness hs;
    KvFlashPager p;
    build(hs, p);
    std::vector<uint8_t> blob = p.serialize();

    Harness hs2;
    KvFlashPager q;
    must(q.attach(make_cfg(), hs2.k, hs2.v));
    must(q.alloc_span(0, 2 * kChunk));
    const uint64_t e = q.epoch();

    // First trailer entry is (c0, span): push span beyond the blob's chunks.
    const size_t trailer = kHdr + 6 * kChunkBytes + 6 * 8;
    REQUIRE(blob.size() > trailer + 12);
    std::vector<uint8_t> bad = blob;
    const uint32_t huge = 99;
    std::memcpy(bad.data() + trailer + 12, &huge, 4);   // magic, count, c0, then span
    CHECK(!q.deserialize(bad.data(), bad.size()));
    CHECK(q.epoch() == e);
    CHECK(q.n_chunks() == 2);
    CHECK(!q.any_compacted());

    std::vector<uint8_t> trunc(blob.begin(), blob.end() - 4);   // beta cut short
    CHECK(!q.deserialize(trunc.data(), trunc.size()));
    CHECK(q.n_chunks() == 2);
}

TEST_CASE(KvflashPagerCompactedFixture, with_no_compacted_chunk_the_mask_and_the_blob_are_unchanged) {
    Harness hs;
    KvFlashPager p;
    must(p.attach(make_cfg(), hs.k, hs.v));
    must(p.alloc_span(0, 6 * kChunk));

    CHECK(!p.any_compacted());                    // so the engine never attaches kv_bias
    CHECK(p.is_identity());
    CHECK(bias_of(p, 0) == expect_bias(0, -1));

    std::vector<uint16_t> m(kPool);
    p.fill_slot_mask(m.data());
    std::vector<uint16_t> want(kPool, 0xFC00);
    for (int i = 0; i < 6 * kChunk; i++) want[(size_t)i] = 0x0000;
    CHECK(std::memcmp(m.data(), want.data(), kPool * 2) == 0);

    CHECK(p.serialize().size() == kHdr + 6 * kChunkBytes + 6 * 8);   // no trailer
}
