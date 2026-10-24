// WMMA paged attention two-mode differential test (RDNA4, gfx1201).
//
// Runs ggml_paged_attn_ext through the stage-1 WMMA kernel when
// LUCE_PAGED_WMMA=1 (route pinned by the launch counter) or through
// the V_DOT2 reference kernel when =0 (counter must stay 0), dumping the
// outputs to paged_attn_out_wmma.bin / paged_attn_out_vdot2.bin (one file per
// route, so the two CTest entries never truncate each other when run in
// parallel) for the compare_paged_attn.py comparator (max-abs-diff tolerance
// 2e-3 f16 / 3e-3 q8_0 / 6e-3 q4_0). The CPU backend
// ABORTS on GGML_OP_PAGED_ATTN, so the two GPU routes are compared against each
// other.
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int D = 256;
constexpr int Hq = 24;
constexpr int Hk = 4;
constexpr int BLOCK_SIZE = 16;

uint32_t g_rng = 0x9e3779b9u;
float next_float() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t)(g_rng >> 8) - 8388608) / 8388608.0f;
}

struct Case {
    const char * name;
    int n_rows;      // query rows
    int n_slots;     // block-table columns
    int kv_len;      // committed prefix length (per slot, may differ)
    ggml_type kv_type; // F16 / Q8_0 / Q4_0 paged pool
    bool mixed;      // ragged multi-slot + decode-row shape
    int qpos_stride = 1; // 1 = dense prefill chunk at end of prefix
                         // (extent kv_len-n_rows+r+1); >1 spans partitions
    // KVFlash block-table holes (C1.3-04). A hole is a -1 entry the kernel must
    // exclude from the softmax; the physical rows behind it hold poison.
    int holes = 0;       // 0 none, 1 whole 64-token chunks missing mid-sequence,
                         // 2 the last 8 chunks, 3 one long stretch (all-null
                         // partitions), 4 every entry (all-null context)
    int base_blocks = 0; // slot sits at a nonzero arena offset: physical = logical + base
};

// Is logical block b a hole? (a 64-token chunk is four 16-token blocks)
bool is_hole(int holes, int b, int live_blocks) {
    switch (holes) {
        case 1: return (b / 4) % 3 == 1;
        case 2: return b >= live_blocks - 32;
        case 3: return b >= 16 && b < live_blocks - 16;
        case 4: return true;
        default: return false;
    }
}

bool run_case(ggml_backend_t gpu, const Case & c, FILE * out) {
    const ggml_type kv_type = c.kv_type;
    const int pool_tokens = (c.kv_len + BLOCK_SIZE - 1 + 64) / BLOCK_SIZE * BLOCK_SIZE + c.base_blocks * BLOCK_SIZE;
    const int max_blocks  = pool_tokens / BLOCK_SIZE;

    ggml_init_params params = { 4u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, c.n_rows, Hq);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, kv_type, D, pool_tokens, Hk);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, kv_type, D, pool_tokens, Hk);
    ggml_tensor * block_table  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, c.n_slots);
    ggml_tensor * kv_seq_lens  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_slots);
    ggml_tensor * active_slot_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_rows);
    ggml_tensor * query_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_rows);

    for (ggml_tensor * t : { q, k, v, block_table, kv_seq_lens, active_slot_ids, query_positions }) {
        ggml_set_input(t);
    }

    ggml_tensor * out_t = ggml_paged_attn_ext(
        ctx, q, k, v, block_table, kv_seq_lens, active_slot_ids, query_positions,
        1.0f / std::sqrt((float) D), BLOCK_SIZE, c.kv_len);
    ggml_set_output(out_t);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_t);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // ── Data ──
    std::vector<float> qv(ggml_nelements(q));
    for (float & x : qv) x = next_float();
    ggml_backend_tensor_set(q, qv.data(), 0, qv.size() * sizeof(float));

    // Block table: logical block b of slot s -> physical block b + base (identity
    // mapping within the pool). Holes stay -1.
    std::vector<int32_t> bt((size_t) max_blocks * c.n_slots, -1);
    const int live_blocks = (c.kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int s = 0; s < c.n_slots; ++s) {
        for (int b = 0; b < live_blocks; ++b) {
            if (!is_hole(c.holes, b, live_blocks)) bt[(size_t) s * max_blocks + b] = b + c.base_blocks;
        }
    }
    const size_t n_kv_rows = (size_t) pool_tokens * Hk;
    std::vector<float> kvf(n_kv_rows * D);
    for (float & x : kvf) x = next_float();
    if (c.holes != 0 || c.base_blocks != 0) {
        // Every physical block the table does not name holds poison: a kernel
        // that reads or softmaxes an unnamed row is wrong by orders of magnitude.
        std::vector<bool> named((size_t) pool_tokens / BLOCK_SIZE, false);
        for (int32_t e : bt) {
            if (e >= 0) named[(size_t) e] = true;
        }
        for (size_t pb = 0; pb < named.size(); ++pb) {
            if (named[pb]) continue;
            for (int h = 0; h < Hk; ++h)
                for (int t = 0; t < BLOCK_SIZE; ++t)
                    for (int d = 0; d < D; ++d)
                        kvf[((size_t) h * pool_tokens + pb * BLOCK_SIZE + t) * D + d] = (d & 1) ? 3000.0f : -3000.0f;
        }
    }

    if (c.kv_type != GGML_TYPE_F16) {
        std::vector<uint8_t> kvq(ggml_nbytes(k));
        ggml_quantize_chunk(c.kv_type, kvf.data(), kvq.data(), 0,
                            (int64_t) n_kv_rows, D, nullptr);
        ggml_backend_tensor_set(k, kvq.data(), 0, kvq.size());
        ggml_backend_tensor_set(v, kvq.data(), 0, kvq.size());
    } else {
        std::vector<ggml_fp16_t> kvh(n_kv_rows * D);
        for (size_t i = 0; i < kvh.size(); ++i) kvh[i] = ggml_fp32_to_fp16(kvf[i]);
        ggml_backend_tensor_set(k, kvh.data(), 0, kvh.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, kvh.data(), 0, kvh.size() * sizeof(ggml_fp16_t));
    }

    ggml_backend_tensor_set(block_table, bt.data(), 0, bt.size() * sizeof(int32_t));

    std::vector<int32_t> ksl((size_t) c.n_slots);
    for (int s = 0; s < c.n_slots; ++s) {
        ksl[s] = c.kv_len; // slot 1 of mixed case stays short below
    }
    std::vector<int32_t> asi((size_t) c.n_rows);
    std::vector<int32_t> qpos((size_t) c.n_rows);
    if (c.mixed) {
        // 30 rows slot 0 (positions 0..29), 20 rows slot 1 (positions
        // 0..19), 14 rows slot 2 (positions 0..13), one decode row slot 3
        // (no causal position).
        ksl[1] = 128; ksl[2] = 256;
        int row = 0;
        for (int r = 0; r < 30; ++r) { asi[row] = 0; qpos[row] = r; ++row; }
        for (int r = 0; r < 20; ++r) { asi[row] = 1; qpos[row] = r; ++row; }
        for (int r = 0; r < 14; ++r) { asi[row] = 2; qpos[row] = r; ++row; }
        asi[row] = 3; qpos[row] = -1; // decode row
    } else {
        for (int r = 0; r < c.n_rows; ++r) {
            asi[r] = 0;
            // Dense cases are prefill chunks: the rows sit at the end of the
            // committed prefix, so they attend over the full context (not
            // just the first n_rows tokens). Sparse cases keep their stride
            // to span partitions.
            qpos[r] = c.qpos_stride == 1 ? (c.kv_len - c.n_rows + r)
                                         : (r * c.qpos_stride);
        }
    }
    ggml_backend_tensor_set(kv_seq_lens, ksl.data(), 0, ksl.size() * sizeof(int32_t));
    ggml_backend_tensor_set(active_slot_ids, asi.data(), 0, asi.size() * sizeof(int32_t));
    ggml_backend_tensor_set(query_positions, qpos.data(), 0, qpos.size() * sizeof(int32_t));

    // ── Run ──
    const size_t wmma_before = ggml_backend_cuda_get_paged_attn_wmma256_launch_count();
    const size_t part_before = ggml_backend_cuda_get_paged_attn_partitioned_launch_count();
    if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }
    const size_t wmma_launches = ggml_backend_cuda_get_paged_attn_wmma256_launch_count() - wmma_before;

    const size_t part_launches = ggml_backend_cuda_get_paged_attn_partitioned_launch_count() - part_before;
    // GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS must reach the dispatch on both routes:
    // >=2 takes the partials path, 1 the direct one.
    const char * fp = getenv("GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS");
    const int forced = fp ? atoi(fp) : 0;
    const bool part_ok = forced >= 2 ? part_launches >= 1 : forced == 1 ? part_launches == 0 : true;
    const bool expect_wmma = getenv("LUCE_PAGED_WMMA") != nullptr
                             && atoi(getenv("LUCE_PAGED_WMMA")) != 0;
    const bool route_ok = expect_wmma ? wmma_launches >= 1 : wmma_launches == 0;

    std::vector<float> out_data(ggml_nelements(out_t));
    ggml_backend_tensor_get(out_t, out_data.data(), 0, ggml_nbytes(out_t));

    bool finite = true;
    size_t n_bad = 0, first_bad = 0;
    for (size_t i = 0; i < out_data.size(); ++i) {
        if (!std::isfinite(out_data[i])) {
            if (finite) first_bad = i;
            finite = false;
            ++n_bad;
        }
    }
    if (!finite) std::printf("[paged-wmma] %-18s non-finite: %zu of %zu, first at flat index %zu (dims %lld x %lld x %lld)\n",
                             c.name, n_bad, out_data.size(), first_bad,
                             (long long) out_t->ne[0], (long long) out_t->ne[1], (long long) out_t->ne[2]);

    bool zero_ok = true;
    if (c.holes == 4) {                     // no named row: the output is exactly zero on either route
        for (float x : out_data) zero_ok = zero_ok && x == 0.0f;
    }

    std::fwrite(out_data.data(), sizeof(float), out_data.size(), out);
    std::printf("[paged-wmma] %-18s rows=%d slots=%d kv=%d kv_type=%s wmma=%zu part=%zu finite=%s route=%s\n",
                c.name, c.n_rows, c.n_slots, c.kv_len, ggml_type_name(c.kv_type),
                wmma_launches, part_launches, finite ? "yes" : "NO", route_ok && part_ok ? "ok" : "MISMATCH");
    if (!zero_ok) std::printf("[paged-wmma] %-18s FAIL: all-null context must give zero output\n", c.name);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return route_ok && part_ok && finite && zero_ok;
}

} // namespace

int main() {
#if defined(GGML_USE_HIP)
    // Distinguish "no device" (skip) from a device that exists but cannot be
    // queried (real failure), so CI does not report a runtime error as a skip.
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[paged-wmma] SKIP: no HIP device\n");
        return 77;
    }
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
        std::printf("[paged-wmma] FAIL: hipGetDeviceProperties failed\n");
        return 1;
    }
    if (std::strncmp(props.gcnArchName, "gfx12", 5) != 0) {
        std::printf("[paged-wmma] SKIP: requires gfx12 (RDNA4), got %s\n", props.gcnArchName);
        return 77;
    }
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) return 1;

    // One dump per route: the V_DOT2 and WMMA CTest entries share a working
    // directory, so a common filename would race under `ctest -j`.
    const bool wmma_route = getenv("LUCE_PAGED_WMMA") != nullptr
                            && atoi(getenv("LUCE_PAGED_WMMA")) != 0;
    // LUCE_PAGED_NULL_CASES=1 runs the KVFlash hole cases instead (own dumps, so
    // the comparator's global max is not shared with the plain cases). Run the
    // direct and the partitioned dispatch (GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS)
    // as separate processes: the choice is cached on first use.
    const bool null_cases = getenv("LUCE_PAGED_NULL_CASES") != nullptr
                            && atoi(getenv("LUCE_PAGED_NULL_CASES")) != 0;
    const char * out_name = null_cases ? (wmma_route ? "paged_attn_null_out_wmma.bin" : "paged_attn_null_out_vdot2.bin")
                                       : (wmma_route ? "paged_attn_out_wmma.bin" : "paged_attn_out_vdot2.bin");
    FILE * out = std::fopen(out_name, "wb");
    if (!out) return 1;

    const Case cases[] = {
        { "prefill512-f16",    64,  4,  512, GGML_TYPE_F16,  false },
        { "prefill512-q8",     64,  4,  512, GGML_TYPE_Q8_0, false },
        { "prefill512-q4",     64,  4,  512, GGML_TYPE_Q4_0, false },
        { "prefill8192-f16",   64,  4, 8192, GGML_TYPE_F16,  false },
        { "prefill8192-q8",    64,  4, 8192, GGML_TYPE_Q8_0, false },
        { "prefill8192-q4",    64,  4, 8192, GGML_TYPE_Q4_0, false },
        { "mixed8192-f16",     65,  4, 8192, GGML_TYPE_F16,  true  },
        { "mixed8192-q4",      65,  4, 8192, GGML_TYPE_Q4_0, true  },
        // Sparse causal positions: extents up to 2017 force two live
        // context partitions (PAGED_ATTN_BLOCKS_PER_PARTITION = 64 blocks),
        // regression-covering the partition-range clamp.
        { "sparse4096-f16",    64,  4, 4096, GGML_TYPE_F16,  false, 32 },
    };
    // KVFlash hole cases: fields after qpos_stride are {holes, base_blocks}.
    const Case null_set[] = {
        { "gap-mid-f16",      64, 4, 4096, GGML_TYPE_F16,  false, 1, 1, 0 },
        { "gap-mid-q8",       64, 4, 4096, GGML_TYPE_Q8_0, false, 1, 1, 0 },
        { "gap-mid-q4",       64, 4, 4096, GGML_TYPE_Q4_0, false, 1, 1, 0 },
        { "gap-tail-q8",      64, 4, 4096, GGML_TYPE_Q8_0, false, 1, 2, 0 },
        { "gap-long-q8",      64, 4, 8192, GGML_TYPE_Q8_0, false, 32, 3, 0 },   // all-null partitions
        { "allnull-q8",       64, 4, 4096, GGML_TYPE_Q8_0, false, 1, 4, 0 },    // every entry null: zero output
        { "gap-base-q8",      64, 4, 4096, GGML_TYPE_Q8_0, false, 1, 1, 37 },   // nonzero arena offset
        { "gap-mixed-q8",     65, 4, 4096, GGML_TYPE_Q8_0, true,  1, 1, 0 },    // ragged prefill + decode row
        { "gap-causal-q8",    64, 4, 4096, GGML_TYPE_Q8_0, false, 32, 1, 0 },   // queries below kv_len across holes
    };
    bool ok = true;
    if (null_cases) {
        for (const Case & c : null_set) ok = run_case(gpu, c, out) && ok;
    } else {
        for (const Case & c : cases) ok = run_case(gpu, c, out) && ok;
    }
    std::fclose(out);
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
#else
    std::printf("[paged-wmma] SKIP: HIP-only\n");
    return 77;
#endif
}
