// Bug #5 proof-of-teeth: set_rows -> [cast] -> get_rows -> flash_attn_ext on
// CUDA, byte-compared to a CPU F32 reference.
//
// Two modes (compile-time toggle):
//   default          : Kfa/Vfa = cast(get_rows(cache, idxs), F16)   -- the fix
//   -DMTP_SKIP_CAST  : Kfa/Vfa = get_rows(cache, idxs)              -- pre-fix
//
// If the F16 cache + cast path passes (<1e-2) and the F32 (no-cast) path
// blows up, the cast is correctness-critical: the CUDA FA-ext vec kernel
// reinterprets the F32 buffer as half2 and reads garbage. If both pass,
// the cast is unnecessary.

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

// ---------------------------------------------------------------------------
// Test config
// ---------------------------------------------------------------------------
// FA_MAX = 64 to satisfy FATTN_KQ_STRIDE (=64) so the CUDA FA-ext VEC kernel
// is picked -- that's the kernel the bug #5 fix targets. Only N_VALID rows
// carry real indices; the remaining tail is gathered from slot 0 then masked
// to -INF, mirroring the production gather/mask pattern.
static constexpr int D        = 64;   // head_dim
static constexpr int N_CTX    = 128;  // cache rows
static constexpr int HKV      = 2;    // n_head_kv
// H == HKV so gqa_opt_applies=false on Ampere -- forces FA-ext VEC kernel
// (the one the bug #5 fix targets).  With gqa_ratio>1 on sm_86 the dispatch
// picks MMA_F16 instead, which has its own F32-load path and would mask the
// bug.  See ggml-cuda/fattn.cu ggml_cuda_get_best_fattn_kernel.
static constexpr int H        = 2;    // n_head (Q) -- gqa_ratio=1 -> VEC
static constexpr int FA_MAX   = 64;   // KV rows attended (multiple of FATTN_KQ_STRIDE)
static constexpr int WRITE_IX = 5;    // i64 write slot
static constexpr int N_VALID  = 6;    // first 6 rows valid, rest masked

static int32_t make_read_idx(int i) {
    static const int seed[6] = {0,1,2,3,4,5};
    return (i < N_VALID) ? seed[i] : 0;
}

// ---------------------------------------------------------------------------
// CPU reference: same math, F32 throughout.
//
// State after step:
//   K_cache_after[head_dim, ctx, hkv]  -- F16 cache with Kcur written at WRITE_IX
//   K_read[h] = gather rows READ_IDX[*] from cache_after[:,:,h]   -> [D, FA_MAX]
//   For each Q-head q (GQA: hk = q * HKV / H):
//     scores[k] = (Q[:,q] . K_read[hk][:,k]) * scale + mask[k]
//     softmax over the N_VALID valid rows
//     out[:,q] = sum_k softmax_k * V_read[hk][:,k]
// ---------------------------------------------------------------------------

static void cpu_reference(
    const std::vector<float> & K_cache_f32,  // [D, N_CTX, HKV]
    const std::vector<float> & V_cache_f32,
    const std::vector<float> & Kcur,         // [D, 1, HKV]
    const std::vector<float> & Vcur,
    const std::vector<float> & Q,            // [D, H, 1]
    std::vector<float>       & out_f32)      // [D, H, 1]
{
    // 1. Apply set_rows: write Kcur/Vcur at slot WRITE_IX (broadcast across hkv).
    std::vector<float> Kc = K_cache_f32;
    std::vector<float> Vc = V_cache_f32;
    for (int h = 0; h < HKV; ++h) {
        for (int d = 0; d < D; ++d) {
            Kc[h * N_CTX * D + WRITE_IX * D + d] = Kcur[h * D + d];
            Vc[h * N_CTX * D + WRITE_IX * D + d] = Vcur[h * D + d];
        }
    }

    // 2. Gather + attention per Q-head (GQA).
    const float scale = 1.0f / std::sqrt((float)D);
    out_f32.assign((size_t)D * H, 0.0f);

    for (int q = 0; q < H; ++q) {
        const int hk = q * HKV / H;
        const float * Qq = Q.data() + q * D;

        // Gather K_read[D, FA_MAX], V_read[D, FA_MAX].
        float K_read[FA_MAX][D];
        float V_read[FA_MAX][D];
        for (int r = 0; r < FA_MAX; ++r) {
            int src = make_read_idx(r);
            const float * Krow = Kc.data() + hk * N_CTX * D + src * D;
            const float * Vrow = Vc.data() + hk * N_CTX * D + src * D;
            for (int d = 0; d < D; ++d) {
                K_read[r][d] = Krow[d];
                V_read[r][d] = Vrow[d];
            }
        }

        // Mask: first N_VALID rows = 0, rest = -INF.
        float scores[FA_MAX];
        float row_max = -std::numeric_limits<float>::infinity();
        for (int r = 0; r < FA_MAX; ++r) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += Qq[d] * K_read[r][d];
            float s = dot * scale + (r < N_VALID ? 0.0f : -std::numeric_limits<float>::infinity());
            scores[r] = s;
            if (s > row_max) row_max = s;
        }
        float sum_e = 0.0f;
        for (int r = 0; r < FA_MAX; ++r) {
            float e = std::isfinite(scores[r]) ? std::exp(scores[r] - row_max) : 0.0f;
            scores[r] = e;
            sum_e += e;
        }
        float inv = (sum_e > 0.0f) ? 1.0f / sum_e : 0.0f;
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int r = 0; r < FA_MAX; ++r) acc += scores[r] * inv * V_read[r][d];
            out_f32[q * D + d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------

static int run_gpu_test(ggml_backend_t backend, bool skip_cast, float & out_absmax)
{
    const size_t ctx_size = 64 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    // Caches (F16, fixed inputs) -- the production layout.
    ggml_tensor * Kcache = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, N_CTX, HKV);
    ggml_tensor * Vcache = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, N_CTX, HKV);
    ggml_set_input(Kcache);
    ggml_set_input(Vcache);

    // Kcur, Vcur reshaped to [D, 1, HKV] for set_rows (matches production).
    ggml_tensor * Kcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, HKV);
    ggml_tensor * Vcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, HKV);
    ggml_set_input(Kcur);
    ggml_set_input(Vcur);

    // i64 write_idx [1] -- broadcasts over HKV (ne[2] % c->ne[1] == 0).
    ggml_tensor * write_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    ggml_set_input(write_idx);

    // Q [D, n_batch=1, n_head=H] (F32) -- FA-ext convention.
    ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, 1, H);
    ggml_set_input(Q);

    // Read indices i32 [FA_MAX, HKV].
    ggml_tensor * read_idxs = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, FA_MAX, HKV);
    ggml_set_input(read_idxs);

    // Mask F16 [FA_MAX, 1].
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, FA_MAX, 1);
    ggml_set_input(mask);

    // ----- Build graph: set_rows -> [cast] -> get_rows -> flash_attn_ext -----
    ggml_tensor * k_after = ggml_set_rows(ctx, Kcache, Kcur, write_idx);
    ggml_tensor * v_after = ggml_set_rows(ctx, Vcache, Vcur, write_idx);

    ggml_tensor * Kgot = ggml_get_rows(ctx, k_after, read_idxs);  // F32 [D, FA_MAX, HKV]
    ggml_tensor * Vgot = ggml_get_rows(ctx, v_after, read_idxs);

    ggml_tensor * Kfa;
    ggml_tensor * Vfa;
    if (skip_cast) {
        // PRE-FIX: hand F32 directly to FA-ext.
        Kfa = Kgot;
        Vfa = Vgot;
    } else {
        // BUG-#5 FIX: cast to F16 before FA-ext.
        Kfa = ggml_cast(ctx, Kgot, GGML_TYPE_F16);
        Vfa = ggml_cast(ctx, Vgot, GGML_TYPE_F16);
    }

    // Q layout per ggml.h: [n_embd_k, n_batch, n_head].
    const float scale = 1.0f / std::sqrt((float)D);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, Q, Kfa, Vfa, mask, scale, 0.0f, 0.0f);
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, k_after);
    ggml_build_forward_expand(gf, v_after);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "gallocr alloc failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return 1;
    }

    // ----- Fill inputs (fixed seed) -----
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Kcache_f32((size_t)D * N_CTX * HKV);
    std::vector<float> Vcache_f32((size_t)D * N_CTX * HKV);
    for (auto & x : Kcache_f32) x = dist(rng);
    for (auto & x : Vcache_f32) x = dist(rng);

    std::vector<float> Kcur_buf((size_t)D * HKV);
    std::vector<float> Vcur_buf((size_t)D * HKV);
    for (auto & x : Kcur_buf) x = dist(rng);
    for (auto & x : Vcur_buf) x = dist(rng);

    std::vector<float> Q_buf((size_t)D * H);
    for (auto & x : Q_buf) x = dist(rng);

    // F16 cache upload.
    {
        std::vector<ggml_fp16_t> tmp(Kcache_f32.size());
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = ggml_fp32_to_fp16(Kcache_f32[i]);
        ggml_backend_tensor_set(Kcache, tmp.data(), 0, ggml_nbytes(Kcache));
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = ggml_fp32_to_fp16(Vcache_f32[i]);
        ggml_backend_tensor_set(Vcache, tmp.data(), 0, ggml_nbytes(Vcache));
    }

    // Round-trip F32 caches through F16 so the CPU reference matches what the
    // GPU actually sees (avoids a free 5e-4 mismatch from F32->F16 quant).
    for (auto & x : Kcache_f32) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    for (auto & x : Vcache_f32) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));

    ggml_backend_tensor_set(Kcur, Kcur_buf.data(), 0, ggml_nbytes(Kcur));
    ggml_backend_tensor_set(Vcur, Vcur_buf.data(), 0, ggml_nbytes(Vcur));
    ggml_backend_tensor_set(Q,    Q_buf.data(),    0, ggml_nbytes(Q));

    const int64_t wi = WRITE_IX;
    ggml_backend_tensor_set(write_idx, &wi, 0, sizeof(int64_t));

    // read_idxs: same column for each hkv.
    std::vector<int32_t> ridx((size_t)FA_MAX * HKV);
    for (int h = 0; h < HKV; ++h) {
        for (int i = 0; i < FA_MAX; ++i) ridx[h * FA_MAX + i] = make_read_idx(i);
    }
    ggml_backend_tensor_set(read_idxs, ridx.data(), 0, ggml_nbytes(read_idxs));

    // mask: N_VALID zeros, then -INF.
    std::vector<ggml_fp16_t> mask_buf(FA_MAX);
    for (int i = 0; i < FA_MAX; ++i) {
        mask_buf[i] = ggml_fp32_to_fp16(i < N_VALID ? 0.0f : -std::numeric_limits<float>::infinity());
    }
    ggml_backend_tensor_set(mask, mask_buf.data(), 0, ggml_nbytes(mask));

    // ----- Compute -----
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return 2;
    }

    // Reference (uses the round-tripped F16 cache values; cur stays F32 since
    // it goes through set_rows on an F16 destination -- ggml quantises on write,
    // so round-trip those too for an apples-to-apples compare).
    for (auto & x : Kcur_buf) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
    for (auto & x : Vcur_buf) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));

    std::vector<float> ref;
    cpu_reference(Kcache_f32, Vcache_f32, Kcur_buf, Vcur_buf, Q_buf, ref);

    // ----- Download GPU output -----
    const size_t n_out = ggml_nelements(out);
    std::vector<float> gpu(n_out);
    ggml_backend_tensor_get(out, gpu.data(), 0, n_out * sizeof(float));

    // FA-ext output layout: [v->ne[0], q->ne[2], q->ne[1], q->ne[3]]
    //                      = [D, H, 1, 1] -- same memory layout as ref [D, H].
    float absmax = 0.0f;
    bool any_nan = false;
    int worst_i = -1;
    for (size_t i = 0; i < n_out; ++i) {
        if (!std::isfinite(gpu[i])) { any_nan = true; break; }
        float d = std::fabs(gpu[i] - ref[i]);
        if (d > absmax) { absmax = d; worst_i = (int)i; }
    }
    out_absmax = absmax;

    // Print a few raw output samples so the with-cast vs no-cast diff is
    // visible byte-for-byte across runs.
    printf("[%s] absmax=%.6f nan=%s n_out=%zu worst_i=%d gpu=%.6f ref=%.6f\n",
           skip_cast ? "no-cast" : "with-cast",
           absmax, any_nan ? "YES" : "no", n_out, worst_i,
           worst_i >= 0 ? gpu[worst_i] : 0.0f,
           worst_i >= 0 ? ref[worst_i] : 0.0f);
    printf("  first8 gpu:");
    for (int i = 0; i < 8 && i < (int)n_out; ++i) printf(" %.6f", gpu[i]);
    printf("\n  first8 ref:");
    for (int i = 0; i < 8 && i < (int)n_out; ++i) printf(" %.6f", ref[i]);
    printf("\n");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return (any_nan || absmax >= 1e-2f) ? 3 : 0;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "CUDA backend not available\n");
        return 77;  // ctest "skip"
    }

#ifdef MTP_SKIP_CAST
    const bool skip_cast = true;
    const char * label = "MTP_SKIP_CAST=on (pre-fix path)";
#else
    const bool skip_cast = false;
    const char * label = "MTP_SKIP_CAST=off (with cast -- the fix)";
#endif
    printf("=== %s ===\n", label);

    float absmax = 0.0f;
    int rc = run_gpu_test(backend, skip_cast, absmax);

    ggml_backend_free(backend);

    printf("%s (absmax=%.5f, tol=1e-2)\n",
           rc == 0 ? "PASS" : "FAIL", absmax);
    return rc;
}
