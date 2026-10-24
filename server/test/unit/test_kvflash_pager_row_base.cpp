// KvFlashConfig::row_base: a pager pool that is a slice of a larger KV
// allocation. Block ids stay pool-local; rows and tensor bytes are offset.
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPagerRowBaseFixture {};
constexpr int kChunk = 64, kPoolChunks = 8, kBaseChunks = 5, kTotalChunks = 14;

KvFlashConfig cfg_at(int base_chunks) {
    KvFlashConfig c;
    c.pool_tokens = kPoolChunks * kChunk;
    c.row_base = base_chunks * kChunk;
    return c;
}
}  // namespace

TEST_CASE(KvflashPagerRowBaseFixture, rows_and_block_ids_are_both_arena_space) {
    TestPager p;
    REQUIRE(p.attach(cfg_at(kBaseChunks), {}, {}));
    CHECK(p.row_base() == kBaseChunks * kChunk);
    REQUIRE(p.alloc_span(0, 3 * kChunk));
    for (int c = 0; c < 3; c++) {
        CHECK(p.block_of(c) >= kBaseChunks && p.block_of(c) < kBaseChunks + kPoolChunks);
        CHECK(p.slot_of((int64_t)c * kChunk + 7) == p.block_of(c) * kChunk + 7);
    }
    CHECK(p.slot_of(3 * kChunk) == -1);
}

TEST_CASE(KvflashPagerRowBaseFixture, attach_refuses_bad_base_and_short_tensors) {
    TestPager p;
    KvFlashConfig c = cfg_at(kBaseChunks);
    c.row_base = 5;                                   // not a chunk multiple
    CHECK(!p.attach(c, {}, {}));
    c.row_base = -kChunk;
    CHECK(!p.attach(c, {}, {}));

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip{};
    ip.mem_size = 8 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, (kBaseChunks + kPoolChunks) * kChunk - 1, 2);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, (kBaseChunks + kPoolChunks) * kChunk - 1, 2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    CHECK(!p.attach(cfg_at(kBaseChunks), {k}, {v}));  // one row short of base + pool
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
}

TEST_CASE(KvflashPagerRowBaseFixture, page_out_and_in_move_bytes_at_the_offset_rows) {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_init_params ip{};
    ip.mem_size = 8 * ggml_tensor_overhead();
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kTotalChunks * kChunk, 2);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kTotalChunks * kChunk, 2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (ggml_tensor * t : {k, v}) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));

    TestPager p;
    REQUIRE(p.attach(cfg_at(kBaseChunks), {k}, {v}));
    REQUIRE(p.alloc_span(0, 2 * kChunk));
    const int64_t row = p.slot_of(0);
    REQUIRE(row >= kBaseChunks * kChunk);
    const size_t len = (size_t)kChunk * k->nb[1];
    std::vector<uint8_t> pat(len, 0xA5);
    ggml_backend_tensor_set(k, pat.data(), (size_t)row * k->nb[1], len);
    REQUIRE(p.page_out(0));
    p.synchronize_paging();
    std::vector<uint8_t> got(len, 1);
    ggml_backend_tensor_get(k, got.data(), (size_t)row * k->nb[1], len);
    CHECK(got == std::vector<uint8_t>(len, 0));        // zeroed in the offset block
    REQUIRE(p.page_in(0));
    p.synchronize_paging();
    ggml_backend_tensor_get(k, got.data(), (size_t)p.slot_of(0) * k->nb[1], len);
    CHECK(got == pat);
    // Nothing below the base was touched.
    std::vector<uint8_t> below((size_t)kBaseChunks * kChunk * k->nb[1], 1);
    ggml_backend_tensor_get(k, below.data(), 0, below.size());
    CHECK(below == std::vector<uint8_t>(below.size(), 0));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
}
