// Page-out / slot-reuse ordering on a real GPU. page_out() queues the D2H copy
// and the block zeroing on the pager's own stream and hands the block back at
// once; the decode step then writes the new K/V into the reused slot on the
// backend's (non-blocking) compute stream. Without an ordering between the two
// streams the new write can land before the queued copy (host archive gets the
// NEW bytes) or before the queued zeroing (the new row is wiped).
//
// The stress below appends chunk after chunk through a full pool, writes each
// new chunk on a separate non-blocking stream right after slot_for() (like the
// backend does), never syncs in between, and finally checks every chunk's bytes
// (host archive for paged-out chunks, pool for resident ones) via serialize().
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerCudaRaceFixture {};

constexpr int kRHeadDim = 128, kRPool = 512, kRNHeadKv = 8, kRNLayer = 8, kRChunk = 64;
constexpr size_t kRHdr = 8 + 8 * 4;

uint8_t rpat(int c, size_t i) { return (uint8_t)(c * 53 + i * 31 + 7 + (i >> 8) * 13); }

struct CudaHarness {
    ggml_context *        ctx = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    CudaHarness() {
        backend = ggml_backend_cuda_init(0);
        if (!backend) return;
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kRNLayer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kRNLayer; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kRHeadDim, kRPool, kRNHeadKv));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kRHeadDim, kRPool, kRNHeadKv));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~CudaHarness() {
        if (buf)     ggml_backend_buffer_free(buf);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    CudaHarness(const CudaHarness &) = delete;
    CudaHarness & operator=(const CudaHarness &) = delete;
};

// Queue chunk c's bytes (staging holds rpat in copy_chunk order) into `block`
// on stream `cs` from a pinned staging buffer. No host sync, no host work.
void write_chunk_async(CudaHarness & h, int block, uint8_t * staging, cudaStream_t cs) {
    size_t n = 0;
    for (int l = 0; l < kRNLayer; l++)
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * t = kv == 0 ? h.k[l] : h.v[l];
            for (int hh = 0; hh < kRNHeadKv; hh++) {
                const size_t off = (size_t)block * kRChunk * t->nb[1] + (size_t)hh * t->nb[2];
                const size_t len = (size_t)kRChunk * t->nb[1];
                cudaMemcpyAsync((uint8_t *)t->data + off, staging + n, len, cudaMemcpyHostToDevice, cs);
                n += len;
            }
        }
}

// Returns the number of chunks whose bytes differ from rpat after the run.
int run_round(int n_chunks) {
    CudaHarness h;
    KvFlashPager p;
    KvFlashConfig cfg;
    cfg.pool_tokens = kRPool;
    cfg.chunk_tokens = kRChunk;
    cfg.sink_chunks = 1;
    cfg.tail_window_chunks = 4;
    if (!p.attach(cfg, h.k, h.v)) return -1;

    cudaStream_t cs = nullptr;                       // the backend's compute stream is non-blocking
    cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
    std::vector<uint8_t *> staging;
    for (int c = 0; c < n_chunks; c++) {                     // all host work up front
        uint8_t * st = nullptr;
        cudaMallocHost(&st, p.chunk_bytes());
        for (size_t i = 0; i < p.chunk_bytes(); i++) st[i] = rpat(c, i);
        staging.push_back(st);
    }
    int bad = 0;
    for (int c = 0; c < n_chunks; c++) {
        const int slot = p.slot_for((int64_t)c * kRChunk);   // may page a victim out
        if (slot < 0) { bad = -1; break; }
        write_chunk_async(h, p.block_of(c), staging[(size_t)c], cs);   // new K/V, no wait on the pager
    }
    cudaStreamSynchronize(cs);
    p.synchronize_paging();
    cudaDeviceSynchronize();
    if (bad == 0) {
        const std::vector<uint8_t> blob = p.serialize();
        for (int c = 0; c < n_chunks; c++) {
            const uint8_t * src = blob.data() + kRHdr + (size_t)c * p.chunk_bytes();
            bool ok = true;
            for (size_t i = 0; i < p.chunk_bytes() && ok; i++) ok = src[i] == rpat(c, i);
            bad += !ok;
        }
    }
    for (uint8_t * s : staging) cudaFreeHost(s);
    cudaStreamDestroy(cs);
    return bad;
}

} // namespace

TEST_CASE(KvflashPagerCudaRaceFixture, reused_slot_write_is_ordered_after_queued_page_out) {
    if (ggml_backend_cuda_get_device_count() < 1) return;   // no GPU: nothing to stress
    const char * rounds_env = std::getenv("LUCE_KVFLASH_RACE_ROUNDS");
    const int rounds = rounds_env ? std::atoi(rounds_env) : 40;
    int corrupted_rounds = 0, corrupted_chunks = 0;
    for (int r = 0; r < rounds; r++) {
        const int bad = run_round(24);
        REQUIRE(bad >= 0);
        corrupted_rounds += bad > 0;
        corrupted_chunks += bad;
    }
    std::fprintf(stderr, "[race] rounds=%d corrupted_rounds=%d corrupted_chunks=%d\n",
                 rounds, corrupted_rounds, corrupted_chunks);
    CHECK(corrupted_chunks == 0);
}
