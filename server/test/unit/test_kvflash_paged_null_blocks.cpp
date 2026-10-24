// CPU reference for paged attention over a KVFlash block table with null (-1)
// entries (Track C, C0 R4). Block = chunk = 64 tokens; an evicted chunk is one
// -1 entry and its rows are skipped exactly. The paged reference walks the
// table block by block with an online softmax (the shape of the kernel's loop);
// the dense reference gathers the resident rows in logical order and does a
// two-pass softmax in double. They must agree within float tolerance. This pins
// the semantics the CUDA/HIP kernels must match; it does not run a kernel.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_pager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPagedNullBlocksFixture {};

constexpr int D = 32, N_HEAD = 4, N_HEAD_KV = 2, kBlock = 64;
constexpr int kN = 3, kQ = 8, kMaxEntries = 32;
constexpr float kTol = 2.0e-6f;

float val(int kind, int head, int pos, int d) {   // deterministic, ~[-1, 1]
    uint32_t x = ((uint32_t)kind * 7919u + (uint32_t)head * 104729u + (uint32_t)pos * 15485863u + (uint32_t)d * 32452843u) * 2654435761u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
    return ((x & 0xffff) / 32768.0f) - 1.0f;
}

struct Arena {
    std::vector<float> k, v;                      // [kv_head][arena_row][D]
    int rows;
    float * K(int h, int64_t r) { return &k[((size_t)h * rows + (size_t)r) * D]; }
    float * V(int h, int64_t r) { return &v[((size_t)h * rows + (size_t)r) * D]; }
};

struct World {
    std::vector<std::unique_ptr<KvFlashSlotPager>> s;
    Arena a;
    World() {
        KvFlashConfig cfg;
        for (int i = 0; i < kN; i++) {
            s.push_back(std::make_unique<KvFlashSlotPager>());
            s.back()->set_pinned_host_backing(false);
            s.back()->attach_slice(i, kN, kQ, cfg, {}, {});
        }
        a.rows = (int)kvflash_arena_rows(kN, kQ);
        a.k.assign((size_t)N_HEAD_KV * a.rows * D, 0.0f);
        a.v.assign((size_t)N_HEAD_KV * a.rows * D, 0.0f);
    }
    // Append logical tokens [0, n): write each token's K/V at the row the pager hands out.
    void fill(int slot, int n) {
        for (int pos = 0; pos < n; pos++) {
            const int64_t r = s[(size_t)slot]->slot_for(pos);
            for (int h = 0; h < N_HEAD_KV; h++)
                for (int d = 0; d < D; d++) {
                    a.K(h, r)[d] = val(0, h, pos, d);
                    a.V(h, r)[d] = val(1, h, pos, d);
                }
        }
    }
};

std::vector<float> query(int qpos) {
    std::vector<float> q((size_t)N_HEAD * D);
    for (int h = 0; h < N_HEAD; h++) for (int d = 0; d < D; d++) q[(size_t)h * D + d] = val(2, h, qpos, d);
    return q;
}

// Walk the table; -1 entries are skipped; causal clamp at qpos (inclusive).
std::vector<float> paged_ref(Arena & a, const std::vector<int32_t> & table, int kv_len, int qpos,
                             const std::vector<float> & q) {
    std::vector<float> out((size_t)N_HEAD * D, 0.0f);
    const float scale = 1.0f / std::sqrt((float)D);
    const int last = std::min(kv_len - 1, qpos);
    for (int h = 0; h < N_HEAD; h++) {
        const int kvh = h / (N_HEAD / N_HEAD_KV);
        float m = -INFINITY, den = 0.0f;
        std::vector<float> acc(D, 0.0f);
        for (int b = 0; b * kBlock <= last; b++) {
            const int32_t e = table[(size_t)b];
            if (e < 0) continue;                  // null block: exact skip
            for (int t = 0; t < kBlock && b * kBlock + t <= last; t++) {
                const int64_t row = (int64_t)e * kBlock + t;
                float s = 0.0f;
                for (int d = 0; d < D; d++) s += q[(size_t)h * D + d] * a.K(kvh, row)[d];
                s *= scale;
                const float nm = std::max(m, s);
                const float corr = std::isfinite(m) ? std::exp(m - nm) : 0.0f;
                const float p = std::exp(s - nm);
                for (int d = 0; d < D; d++) acc[d] = acc[d] * corr + p * a.V(kvh, row)[d];
                den = den * corr + p;
                m = nm;
            }
        }
        if (den > 0.0f) for (int d = 0; d < D; d++) out[(size_t)h * D + d] = acc[d] / den;
    }
    return out;
}

// Dense attention over the resident rows, gathered in logical order.
std::vector<float> dense_ref(Arena & a, const KvFlashSlotPager & p, int kv_len, int qpos,
                             const std::vector<float> & q, int * n_rows) {
    std::vector<float> out((size_t)N_HEAD * D, 0.0f);
    const int last = std::min(kv_len - 1, qpos);
    std::vector<int64_t> rows;
    for (int pos = 0; pos <= last; pos++) {
        const int64_t r = p.slot_of(pos);
        if (r >= 0) rows.push_back(r);
    }
    *n_rows = (int)rows.size();
    for (int h = 0; h < N_HEAD; h++) {
        const int kvh = h / (N_HEAD / N_HEAD_KV);
        std::vector<double> sc(rows.size());
        double m = -1e300;
        for (size_t i = 0; i < rows.size(); i++) {
            double s = 0;
            for (int d = 0; d < D; d++) s += (double)q[(size_t)h * D + d] * a.K(kvh, rows[i])[d];
            sc[i] = s / std::sqrt((double)D);
            m = std::max(m, sc[i]);
        }
        double den = 0;
        for (double & s : sc) { s = std::exp(s - m); den += s; }
        for (size_t i = 0; i < rows.size(); i++)
            for (int d = 0; d < D; d++) out[(size_t)h * D + d] += (float)(sc[i] / den * a.V(kvh, rows[i])[d]);
    }
    return out;
}

float max_abs_diff(const std::vector<float> & x, const std::vector<float> & y) {
    float d = 0;
    for (size_t i = 0; i < x.size(); i++) d = std::max(d, std::fabs(x[i] - y[i]));
    return d;
}
}  // namespace

TEST_CASE(KvflashPagedNullBlocksFixture, gapped_table_matches_dense_attention_over_the_resident_rows) {
    World w;
    w.fill(0, 6 * kBlock);                        // neighbours with their own rows
    w.fill(2, 7 * kBlock);
    const int kv_len = 20 * kBlock;
    w.fill(1, kv_len);                            // 12 chunks evicted: stale rows stay in the arena
    KvFlashSlotPager & p = *w.s[1];
    std::vector<int32_t> table((size_t)kMaxEntries, -1);
    p.fill_block_table(table.data(), kMaxEntries);
    int nulls = 0, longest = 0, run = 0;
    for (int b = 0; b < 20; b++) { if (table[(size_t)b] < 0) { nulls++; run++; longest = std::max(longest, run); } else run = 0; }
    REQUIRE(nulls == 20 - kQ);
    REQUIRE(longest >= 4);                        // a stretch of consecutive null blocks

    float worst = 0;
    // Decode at the end, causal prefill rows in the resident tail, a mid-sequence row,
    // and a row inside a block that is itself null.
    for (int qpos : {kv_len - 1, kv_len - 70, kv_len - 200, 10 * kBlock + 5, 3 * kBlock}) {
        const std::vector<float> q = query(qpos);
        int n_rows = 0;
        const auto d = dense_ref(w.a, p, kv_len, qpos, q, &n_rows);
        const auto g = paged_ref(w.a, table, kv_len, qpos, q);
        REQUIRE(n_rows > 0);
        const float diff = max_abs_diff(d, g);
        worst = std::max(worst, diff);
        CHECK(diff <= kTol);
    }
    std::printf("[kvflash] paged-null vs dense: max abs diff %.3e (tol %.1e)\n", worst, kTol);
}

TEST_CASE(KvflashPagedNullBlocksFixture, a_stale_row_in_a_null_block_does_not_leak_into_the_result) {
    World w;
    const int kv_len = 20 * kBlock;
    w.fill(1, kv_len);
    KvFlashSlotPager & p = *w.s[1];
    std::vector<int32_t> table((size_t)kMaxEntries, -1);
    p.fill_block_table(table.data(), kMaxEntries);
    const int qpos = kv_len - 1;
    const std::vector<float> q = query(qpos);
    int n_rows = 0;
    const auto want = dense_ref(w.a, p, kv_len, qpos, q, &n_rows);   // independent oracle
    // Poison every row of the slice that no table entry names with huge values.
    const KvFlashSlice sl = kvflash_slice(1, kQ);
    std::vector<bool> named((size_t)kQ, false);
    for (int b = 0; b < kMaxEntries; b++) if (table[(size_t)b] >= 0) named[(size_t)(table[(size_t)b] - sl.block_lo)] = true;
    for (int blk = 0; blk < kQ; blk++) {
        if (named[(size_t)blk]) continue;
        for (int h = 0; h < N_HEAD_KV; h++) for (int t = 0; t < kBlock; t++)
            for (int d = 0; d < D; d++) { w.a.K(h, (int64_t)(sl.block_lo + blk) * kBlock + t)[d] = 1e6f; w.a.V(h, (int64_t)(sl.block_lo + blk) * kBlock + t)[d] = 1e6f; }
    }
    CHECK(max_abs_diff(want, paged_ref(w.a, table, kv_len, qpos, q)) <= kTol);
}

TEST_CASE(KvflashPagedNullBlocksFixture, an_all_null_table_gives_zero_output_like_dense_with_no_rows) {
    World w;
    std::fill(w.a.k.begin(), w.a.k.end(), 1.0f);        // stale rows everywhere: a null must not read them
    std::fill(w.a.v.begin(), w.a.v.end(), 1.0f);
    std::vector<int32_t> table((size_t)kMaxEntries, -1);
    const std::vector<float> q = query(100);
    const auto g = paged_ref(w.a, table, 20 * kBlock, 100, q);
    CHECK(max_abs_diff(g, std::vector<float>(g.size(), 0.0f)) == 0.0f);
}
