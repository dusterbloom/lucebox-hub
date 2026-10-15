// Unit tests for the KvFlashPager region B (compacted segments, kv_bias
// columns, per-layer ragged extents; kvflash_pager.h). CPU ggml backend, no
// GPU. Layers are ragged on purpose: layer 0 carries 2 B blocks, layer 1 four.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerRegionBFixture {};

constexpr int kHeadDim = 4, kPool = 512, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
constexpr int kBTokens[kNLayer] = {2 * kChunk, 4 * kChunk};
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// b_scale 0 = today's layout (region A only); 1 = A + ragged B_L per layer.
struct Harness {
    ggml_context *          ctx = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    explicit Harness(int b_scale = 1) {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kNLayer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kNLayer; l++) {
            const int rows = kPool + b_scale * kBTokens[l];
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, rows, kNHeadKv));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, rows, kNHeadKv));
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

void must(bool ok) { if (!ok) throw std::runtime_error("kvflash region B helper precondition failed"); }

// Live slots per (layer, head): layer 0 needs 2 blocks, layer 1 one.
const std::vector<int> kCounts = {70, 10, 64, 5};

std::vector<std::vector<float>> make_beta() {
    std::vector<std::vector<float>> beta;
    for (int lh = 0; lh < kNLayer * kNHeadKv; lh++) {
        std::vector<float> b((size_t)kCounts[(size_t)lh]);
        for (size_t i = 0; i < b.size(); i++) b[i] = 0.25f * (float)i - 3.0f + (float)lh;
        beta.push_back(std::move(b));
    }
    beta[0][12] = 0.0f;   // a live slot whose beta is exactly 0
    return beta;
}

// Chunks 0..5 in region A, then [2*kChunk, 4*kChunk) replaced by a segment.
void build(Harness & h, KvFlashPager & p) {
    must(p.attach(make_cfg(), h.k, h.v));
    must(p.alloc_span(0, 6 * kChunk));
    must(p.insert_compacted(2 * kChunk, 4 * kChunk, kCounts, make_beta()));
}

std::vector<float> bias_of(const KvFlashPager & p, int layer) {
    std::vector<float> b((size_t)p.layer_cols(layer) * kNHeadKv, 12345.f);
    p.fill_kv_bias(layer, b.data());
    return b;
}

// Rows of B block `blk` in layer `l`, one byte pattern per (tensor, head).
template <class F>
void for_each_b_segment(Harness & h, const KvFlashPager & p, int l, int blk, F && fn) {
    for (int kv = 0; kv < 2; kv++) {
        ggml_tensor * t = kv == 0 ? h.k[l] : h.v[l];
        for (int hh = 0; hh < kNHeadKv; hh++)
            fn(t, (size_t)p.b_row(l, blk) * t->nb[1] + (size_t)hh * t->nb[2],
               (size_t)kChunk * t->nb[1]);
    }
}

uint8_t pat(int l, int blk, size_t i) { return (uint8_t)(l * 61 + blk * 17 + i * 31 + 5); }

void stamp_b(Harness & h, const KvFlashPager & p) {
    for (const auto & s : p.segments())
        for (int l = 0; l < kNLayer; l++)
            for (int blk : s.blocks[(size_t)l]) {
                size_t n = 0;
                for_each_b_segment(h, p, l, blk, [&](ggml_tensor * t, size_t off, size_t len) {
                    std::vector<uint8_t> b(len);
                    for (size_t i = 0; i < len; i++) b[i] = pat(l, blk, n + i);
                    ggml_backend_tensor_set(t, b.data(), off, len);
                    n += len;
                });
            }
}

} // namespace

TEST_CASE(KvflashPagerRegionBFixture, insert_compacted_fills_kv_bias_columns) {
    Harness h;
    KvFlashPager p;
    build(h, p);

    REQUIRE(p.segments().size() == 1);
    CHECK(p.segments()[0].lo == 2 * kChunk);
    CHECK(p.segments()[0].hi == 4 * kChunk);
    CHECK(p.segments()[0].blocks[0].size() == 2);
    CHECK(p.segments()[0].blocks[1].size() == 1);
    // The segment's positions get no slot; its region A blocks were released.
    CHECK(p.slot_of(2 * kChunk + 3) == -1);
    CHECK(p.slot_for(3 * kChunk) == -1);
    CHECK(p.resident_blocks() == 4);
    CHECK(!p.is_identity());

    const auto beta = make_beta();
    for (int l = 0; l < kNLayer; l++) {
        const int cols = kPool + kBTokens[l];
        REQUIRE(p.layer_cols(l) == cols);
        const std::vector<float> b = bias_of(p, l);
        for (int hh = 0; hh < kNHeadKv; hh++) {
            const float * row = b.data() + (size_t)hh * cols;
            for (int c = 0; c < kPool; c++) CHECK(row[c] == 0.0f);
            const auto & bt = beta[(size_t)(l * kNHeadKv + hh)];
            for (int s = 0; s < kBTokens[l]; s++) {
                const float want = s < (int)bt.size() ? bt[(size_t)s] : kNegInf;
                CHECK(row[kPool + s] == want);
            }
        }
    }
    CHECK(bias_of(p, 0)[kPool + 12] == 0.0f);   // a live slot with beta exactly 0
}

TEST_CASE(KvflashPagerRegionBFixture, insert_beyond_b_capacity_changes_nothing) {
    Harness h;
    KvFlashPager p;
    REQUIRE(p.attach(make_cfg(), h.k, h.v));
    REQUIRE(p.alloc_span(0, 6 * kChunk));
    const uint64_t epoch = p.epoch();
    // 3 blocks needed in layer 0, which has 2.
    std::vector<int> counts = {130, 10, 64, 5};
    std::vector<std::vector<float>> beta;
    for (int n : counts) beta.emplace_back((size_t)n, 1.0f);
    CHECK(!p.insert_compacted(2 * kChunk, 4 * kChunk, counts, beta));
    CHECK(p.segments().empty());
    CHECK(p.resident_blocks() == 6);
    CHECK(p.epoch() == epoch);
}

TEST_CASE(KvflashPagerRegionBFixture, truncate_inside_segment_drops_it_and_reports_lo) {
    Harness h;
    KvFlashPager p;
    build(h, p);

    CHECK(p.truncate(6 * kChunk) == 6 * kChunk);     // at the end: no-op
    CHECK(p.truncate(4 * kChunk) == 4 * kChunk);     // at hi: the segment is kept
    CHECK(p.segments().size() == 1);

    CHECK(p.truncate(3 * kChunk + 5) == 2 * kChunk);  // inside: dropped, replay from lo
    CHECK(p.segments().empty());
    CHECK(p.n_chunks() == 2);
    CHECK(p.b_free_blocks(0) == 2);
    CHECK(p.b_free_blocks(1) == 4);
    for (int l = 0; l < kNLayer; l++) {
        const std::vector<float> b = bias_of(p, l);
        for (int hh = 0; hh < kNHeadKv; hh++)
            for (int s = 0; s < kBTokens[l]; s++)
                CHECK(b[(size_t)hh * p.layer_cols(l) + kPool + s] == kNegInf);
    }
    // Chunks below lo are untouched and the freed A blocks are reusable.
    CHECK(p.is_resident(0));
    CHECK(p.is_resident(1));
    REQUIRE(p.alloc_span(2 * kChunk, 2 * kChunk));
    CHECK(p.is_resident(2));
    CHECK(p.is_resident(3));
}

TEST_CASE(KvflashPagerRegionBFixture, truncate_at_lo_drops_whole_segment) {
    Harness h;
    KvFlashPager p;
    build(h, p);
    CHECK(p.truncate(2 * kChunk) == 2 * kChunk);
    CHECK(p.segments().empty());
    CHECK(p.b_free_blocks(0) == 2);
}

TEST_CASE(KvflashPagerRegionBFixture, serialize_roundtrip_keeps_segments_and_b_blocks) {
    Harness A;
    KvFlashPager pa;
    build(A, pa);
    stamp_b(A, pa);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 3 * kChunk));   // discarded by deserialize
    REQUIRE(pb.deserialize(blob.data(), blob.size()));

    REQUIRE(pb.segments().size() == 1);
    const auto & sa = pa.segments()[0];
    const auto & sb = pb.segments()[0];
    CHECK(sb.lo == sa.lo);
    CHECK(sb.hi == sa.hi);
    CHECK(sb.counts == sa.counts);
    CHECK(sb.beta == sa.beta);
    CHECK(pb.n_chunks() == 6);
    CHECK(pb.slot_of(2 * kChunk) == -1);
    CHECK(pb.resident_blocks() == 4);
    for (int l = 0; l < kNLayer; l++) {
        REQUIRE(sb.blocks[(size_t)l].size() == sa.blocks[(size_t)l].size());
        CHECK(bias_of(pb, l) == bias_of(pa, l));
        for (size_t i = 0; i < sa.blocks[(size_t)l].size(); i++) {
            size_t n = 0;
            for_each_b_segment(B, pb, l, sb.blocks[(size_t)l][i], [&](ggml_tensor * t, size_t off, size_t len) {
                std::vector<uint8_t> got(len);
                ggml_backend_tensor_get(t, got.data(), off, len);
                for (size_t j = 0; j < len; j++)
                    CHECK(got[j] == pat(l, sa.blocks[(size_t)l][i], n + j));
                n += len;
            });
        }
    }
    // The restored pager truncates like the source.
    CHECK(pb.truncate(3 * kChunk) == 2 * kChunk);
    CHECK(pb.b_free_blocks(1) == 4);
}

TEST_CASE(KvflashPagerRegionBFixture, serialize_cut_inside_a_segment_stops_at_lo) {
    Harness A;
    KvFlashPager pa;
    build(A, pa);
    const std::vector<uint8_t> blob = pa.serialize(3);   // 3 chunks would split [2,4)
    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.n_chunks() == 2);
    CHECK(pb.segments().empty());
}

TEST_CASE(KvflashPagerRegionBFixture, deserialize_rejects_blob_when_b_is_too_small) {
    Harness A;
    KvFlashPager pa;
    build(A, pa);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness small(0);   // no region B at all
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), small.k, small.v));
    REQUIRE(pb.alloc_span(0, 2 * kChunk));
    CHECK(!pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.n_chunks() == 2);          // untouched
    CHECK(pb.is_resident(0));
}

TEST_CASE(KvflashPagerRegionBFixture, empty_b_leaves_region_a_masks_byte_identical) {
    Harness plain(0), withb(1);
    KvFlashPager pp, pw;
    REQUIRE(pp.attach(make_cfg(), plain.k, plain.v));
    REQUIRE(pw.attach(make_cfg(), withb.k, withb.v));
    for (KvFlashPager * p : {&pp, &pw}) {
        REQUIRE(p->alloc_span(0, 8 * kChunk));
        REQUIRE(p->alloc_span(8 * kChunk, 2 * kChunk));   // evicts
        REQUIRE(p->page_in(1));
        p->synchronize_paging();
    }
    std::vector<uint16_t> mp(kPool), mw(kPool);
    std::vector<int32_t> sp(kPool), sw(kPool);
    pp.fill_slot_mask(mp.data());
    pw.fill_slot_mask(mw.data());
    pp.fill_slot_pos(sp.data());
    pw.fill_slot_pos(sw.data());
    CHECK(std::memcmp(mp.data(), mw.data(), mp.size() * 2) == 0);
    CHECK(std::memcmp(sp.data(), sw.data(), sp.size() * 4) == 0);
    CHECK(pp.epoch() == pw.epoch());
    CHECK(pp.is_identity() == pw.is_identity());
    // B allocated but empty: every B column is -inf, region A columns are 0.
    for (int l = 0; l < kNLayer; l++) {
        const std::vector<float> b = bias_of(pw, l);
        const int cols = pw.layer_cols(l);
        for (int hh = 0; hh < kNHeadKv; hh++)
            for (int c = 0; c < cols; c++)
                CHECK(b[(size_t)hh * cols + c] == (c < kPool ? 0.0f : kNegInf));
    }
    CHECK(pp.layer_cols(0) == kPool);   // no B: just region A
}

TEST_CASE(KvflashPagerRegionBFixture, fill_b_slot_mask_opens_only_blocks_a_segment_uses) {
    Harness h;
    KvFlashPager p;
    build(h, p);   // layer 0 uses B blocks 0-1, layer 1 block 0

    constexpr uint16_t F16_ZERO = 0x0000, F16_NEG_INF = 0xFC00;
    const int width = kPool + kBTokens[1];
    std::vector<uint16_t> m((size_t)width, F16_NEG_INF);
    p.fill_slot_mask(m.data());
    const std::vector<uint16_t> a_only = m;
    p.fill_b_slot_mask(m.data());
    for (int c = 0; c < kPool; c++) CHECK(m[(size_t)c] == a_only[(size_t)c]);   // region A untouched
    for (int c = kPool; c < width; c++)
        CHECK(m[(size_t)c] == (c < kPool + 2 * kChunk ? F16_ZERO : F16_NEG_INF));

    // No segment: B stays closed, the mask is the region A mask.
    REQUIRE(p.truncate(2 * kChunk) == 2 * kChunk);
    std::vector<uint16_t> e((size_t)width, F16_NEG_INF);
    p.fill_slot_mask(e.data());
    p.fill_b_slot_mask(e.data());
    for (int c = kPool; c < width; c++) CHECK(e[(size_t)c] == F16_NEG_INF);
}
