// KVFlash 2.0 sessions P1a: the centralized re-pin helper (P2-02). A binding {k chunks pinned from chunk 0}
// must hold after every pager operation that resets or truncates: the helper re-applies it, checks how many
// pins the pager accepted and releases the binding when it cannot hold. CPU ggml backend, no GPU.

#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "kvflash_preamble_store.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPreambleRepinFixture {};

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
void must(bool ok) { if (!ok) throw std::runtime_error("repin helper precondition failed"); }
KvFlashConfig cfg8() {   // 8 blocks of 64 tokens
    KvFlashConfig c;
    c.pool_tokens = 512; c.chunk_tokens = 64; c.sink_chunks = 1; c.tail_window_chunks = 2;
    return c;
}
}  // namespace

TEST_CASE(KvflashPreambleRepinFixture, no_binding_is_a_no_op) {
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 4 * 64));
    int k = 0;
    const KvFlashRepin r = kvflash_repin_preamble(p, k);
    CHECK(r.accepted == 0);
    CHECK(!r.released);
    CHECK(!p.is_pinned(0));
}

TEST_CASE(KvflashPreambleRepinFixture, pins_k_chunks_and_reports_the_accepted_count) {
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 4 * 64));
    int k = 3;
    const KvFlashRepin r = kvflash_repin_preamble(p, k);
    CHECK(r.accepted == 3);
    CHECK(!r.released);
    CHECK(k == 3);
    for (int c = 0; c < 3; c++) CHECK(p.is_pinned(c));
    CHECK(!p.is_pinned(3));
    // idempotent
    CHECK(kvflash_repin_preamble(p, k).accepted == 3);
}

TEST_CASE(KvflashPreambleRepinFixture, adopt_rollback_clear_release_adopt_keeps_pins_and_residency) {
    // the lifecycle of kvflash_slot_kv.h: deserialize / reset clear every pin, truncate clips dropped chunks
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 6 * 64));
    int k = 3;
    must(kvflash_repin_preamble(p, k).accepted == 3);
    // adopt: serialize, reset (clears pins), deserialize (reset inside)
    const std::vector<uint8_t> blob = p.serialize();
    p.reset();
    for (int c = 0; c < 3; c++) CHECK(!p.is_pinned(c));
    must(p.deserialize(blob.data(), blob.size()));
    for (int c = 0; c < 3; c++) CHECK(!p.is_pinned(c));              // deserialize cleared them
    CHECK(kvflash_repin_preamble(p, k).accepted == 3);
    // rollback: truncate to 4 chunks keeps the binding; the helper makes sure of it
    p.truncate(4 * 64);
    CHECK(kvflash_repin_preamble(p, k).accepted == 3);
    // pressure: grow past the pool; the pinned chunks stay resident
    must(p.alloc_span(4 * 64, 12 * 64));
    for (int c = 0; c < 3; c++) { CHECK(p.is_pinned(c)); CHECK(p.is_resident(c)); }
    // clear: reset drops pins, the length is zero, the binding is stale -> released, nothing pinned
    p.reset();
    const KvFlashRepin r = kvflash_repin_preamble(p, k);
    CHECK(r.released);
    CHECK(r.accepted == 0);
    CHECK(k == 0);
    for (int c = 0; c < 3; c++) CHECK(!p.is_pinned(c));
}

TEST_CASE(KvflashPreambleRepinFixture, rewind_below_the_attached_prefix_releases_the_binding) {
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 5 * 64));
    int k = 3;
    must(kvflash_repin_preamble(p, k).accepted == 3);
    p.truncate(1 * 64 + 10);                                         // the length falls inside the prefix
    const KvFlashRepin r = kvflash_repin_preamble(p, k);
    CHECK(r.released);
    CHECK(k == 0);
    for (int c = 0; c < 3; c++) CHECK(!p.is_pinned(c));              // no stale pin on the surviving chunks either
}

TEST_CASE(KvflashPreambleRepinFixture, refused_pin_range_releases_the_binding_instead_of_assuming_success) {
    // sink 1 + tail 2 + k + 2 > 8 blocks: pin_range refuses k = 5; accepted is read back, not assumed
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 8 * 64));
    int k = 5;
    const KvFlashRepin r = kvflash_repin_preamble(p, k);
    CHECK(r.accepted == 0);
    CHECK(r.released);
    CHECK(k == 0);
    k = 3;                                                           // sink 1 + tail 2 + 3 + 2 = 8: the largest that fits
    CHECK(kvflash_repin_preamble(p, k).accepted == 3);
}

TEST_CASE(KvflashPreambleRepinFixture, fewer_pins_than_k_after_a_partial_state_releases) {
    // a fake pager that accepts fewer pins than asked
    struct Fake {
        int ct = 64, cur = 10, accepted = 2;
        std::vector<int> pinned;
        int chunk_tokens() const { return ct; }
        int cur_chunk() const { return cur; }
        void pin_range(int64_t, int64_t) { pinned.assign((size_t)accepted, 1); }
        bool is_pinned(int c) const { return c < (int)pinned.size(); }
        void unpin_all() { pinned.clear(); }
    } f;
    int k = 4;
    const KvFlashRepin r = kvflash_repin_preamble(f, k);
    CHECK(r.accepted == 0);   // a released binding has no pins to report
    CHECK(r.released);
    CHECK(k == 0);
    CHECK(f.pinned.empty());
}
