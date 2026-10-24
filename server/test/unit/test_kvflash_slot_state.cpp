// C1.3-04: byte-exact capture/restore of ONE slot's recurrent slabs (SSM + conv)
// out of the shared multi-slot cache tensors. CPU tensors here; the device path
// is the same ggml_backend_tensor_get/set on the same byte ranges.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_state.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotStateFixture {};

constexpr int kSlots = 3, kLayers = 2;

struct Cache {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> ssm, conv;
    explicit Cache(int conv_slots = kSlots) {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kLayers * 2 + 4) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kLayers; l++) {
            ssm.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, 8, 4, kSlots));   // [S_v, S_v, H_v, n_slots]
            conv.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 10, conv_slots));    // [(k-1), channels, n_slots]
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) throw std::runtime_error("alloc");
    }
    ~Cache() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); if (backend) ggml_backend_free(backend); }
    Cache(const Cache &) = delete;
    Cache & operator=(const Cache &) = delete;
    void fill(int seed) {
        for (auto * set : {&ssm, &conv})
            for (ggml_tensor * t : *set) {
                std::vector<uint8_t> b(ggml_nbytes(t));
                for (size_t i = 0; i < b.size(); i++) b[i] = (uint8_t)(seed * 31 + i * 7 + (i >> 5));
                ggml_backend_tensor_set(t, b.data(), 0, b.size());
            }
    }
};

std::vector<uint8_t> snapshot(const std::vector<ggml_tensor *> & v) {
    std::vector<uint8_t> out;
    for (ggml_tensor * t : v) { const size_t at = out.size(); out.resize(at + ggml_nbytes(t)); ggml_backend_tensor_get(t, out.data() + at, 0, ggml_nbytes(t)); }
    return out;
}
}  // namespace

TEST_CASE(KvflashSlotStateFixture, capture_then_restore_is_byte_exact_for_that_slot_and_touches_no_other) {
    Cache c;
    c.fill(1);
    const size_t n = kvflash_slot_state_bytes(c.ssm, c.conv);
    CHECK(n == (size_t)kLayers * (8 * 8 * 4 + 3 * 10) * sizeof(float));
    std::vector<uint8_t> saved;
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 1, saved));
    CHECK(saved.size() == n);
    const uint64_t h0 = kvflash_slot_state_hash(saved);
    // Slot 1 moves on (the abandoned future); the whole cache changes.
    c.fill(2);
    std::vector<uint8_t> now;
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 1, now));
    CHECK(kvflash_slot_state_hash(now) != h0);
    REQUIRE(kvflash_slot_state_restore(c.ssm, c.conv, 1, saved));
    std::vector<uint8_t> back;
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 1, back));
    CHECK(back == saved);                                  // bit-exact
    CHECK(kvflash_slot_state_hash(back) == h0);
    // Slots 0 and 2 are byte-identical to before the restore.
    for (int s : {0, 2}) {
        std::vector<uint8_t> a;
        REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, s, a));
        Cache ref;
        ref.fill(2);
        std::vector<uint8_t> b;
        REQUIRE(kvflash_slot_state_capture(ref.ssm, ref.conv, s, b));
        CHECK(a == b);
    }
}

TEST_CASE(KvflashSlotStateFixture, a_restore_into_another_slot_lands_in_that_slot_only) {
    Cache c;
    c.fill(3);
    std::vector<uint8_t> s0;
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 0, s0));
    c.fill(4);
    std::vector<uint8_t> before1, before0;
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 1, before1));
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 0, before0));
    REQUIRE(kvflash_slot_state_restore(c.ssm, c.conv, 2, s0));   // slot 0's state into slot 2
    std::vector<uint8_t> got2, got1, got0;
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 2, got2));
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 1, got1));
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 0, got0));
    CHECK(got2 == s0);
    CHECK(got1 == before1);
    CHECK(got0 == before0);
}

TEST_CASE(KvflashSlotStateFixture, bad_slot_or_blob_is_refused_and_changes_nothing) {
    Cache c;
    c.fill(5);
    const auto before_s = snapshot(c.ssm), before_c = snapshot(c.conv);
    std::vector<uint8_t> blob(kvflash_slot_state_bytes(c.ssm, c.conv), 0xAB), out;
    CHECK(!kvflash_slot_state_capture(c.ssm, c.conv, -1, out));
    CHECK(!kvflash_slot_state_capture(c.ssm, c.conv, kSlots, out));
    CHECK(!kvflash_slot_state_restore(c.ssm, c.conv, kSlots, blob));
    CHECK(!kvflash_slot_state_restore(c.ssm, c.conv, -1, blob));
    blob.pop_back();
    CHECK(!kvflash_slot_state_restore(c.ssm, c.conv, 1, blob));   // wrong size
    CHECK(snapshot(c.ssm) == before_s);
    CHECK(snapshot(c.conv) == before_c);
    CHECK(kvflash_slot_state_bytes({}, {}) == 0);
}

TEST_CASE(KvflashSlotStateFixture, the_blob_is_the_slot_slabs_in_tensor_order) {
    Cache c;
    c.fill(7);
    std::vector<uint8_t> blob;
    REQUIRE(kvflash_slot_state_capture(c.ssm, c.conv, 2, blob));
    size_t at = 0;
    for (auto * set : {&c.ssm, &c.conv})
        for (ggml_tensor * t : *set) {
            const size_t axis_bytes = set == &c.ssm ? t->nb[3] : t->nb[2];
            std::vector<uint8_t> slab(axis_bytes);
            ggml_backend_tensor_get(t, slab.data(), 2 * axis_bytes, axis_bytes);
            REQUIRE(at + slab.size() <= blob.size());
            CHECK(std::equal(slab.begin(), slab.end(), blob.begin() + (long)at));
            at += slab.size();
        }
    CHECK(at == blob.size());
}

TEST_CASE(KvflashSlotStateFixture, a_restore_that_cannot_fit_every_tensor_writes_nothing) {
    Cache c(/*conv_slots=*/2);                              // the conv tensors have no slot 2
    c.fill(8);
    const auto before_s = snapshot(c.ssm), before_c = snapshot(c.conv);
    std::vector<uint8_t> blob(kvflash_slot_state_bytes(c.ssm, c.conv), 0x5A);
    CHECK(!kvflash_slot_state_restore(c.ssm, c.conv, 2, blob));
    CHECK(snapshot(c.ssm) == before_s);                     // the ssm slabs, which would fit, were not written either
    CHECK(snapshot(c.conv) == before_c);
}
