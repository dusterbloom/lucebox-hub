// Routed MMB 128-row-tile threshold (Codex idea #1): mmb.cu assigns a BN_BIG=128-row
// tile to experts with >= THRESH rows, and one-or-more BN_SMALL=64-row tiles below that.
// Default THRESH was 128, so an expert with 65-127 routed rows took TWO 64-row tiles,
// each re-streaming the expert's quantized weights. This exercises the new single named
// threshold (mmb_tile128_min(), overridable via LUCE_MMB_TILE128_MIN) against the old
// value (128) and the new default (65), on both routed dispatchers:
//   - ggml_cuda_mul_mat_id_mmb      (plain routed MUL_MAT_ID: a Q5_1 "down" projection)
//   - ggml_cuda_mul_mat_id_mmb_glu  (fused gate/up SwiGLU: two Q4_K "gate"/"up" MUL_MAT_ID)
//
// What is proven: with E=512 experts, n_used=10 of them per token, T=4096 tokens, the
// routing below drives many experts into the 65-127-row band (printed below), so the
// GPU(THRESH=128) vs GPU(THRESH=65) comparison actually exercises the changed tile split.
// The per-output K-accumulation order is unchanged by the tile width (BN_SMALL and
// BN_BIG differ only in how many tokens are processed per kernel launch per expert, not
// in the K-reduction), so the two runs are expected to be BIT-IDENTICAL; a mismatch here
// would mean the threshold move changed more than tiling. A loose CPU reference run is
// also included as a sanity bound, same caveat as test_mmb_cublas_diff: the CPU path
// does not use the same bf16/quant intermediate, so it is not a bit-exact oracle.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint32_t g_state = 0xC0FFEEu;
static float rnd() {
    g_state = g_state * 1664525u + 1013904223u;
    return (float) ((g_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

static double maxdiff(const std::vector<float> & a, const std::vector<float> & b) {
    double md = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) md = std::fmax(md, std::fabs((double) a[i] - b[i]));
    return md;
}
static double rms(const std::vector<float> & a) {
    double s = 0;
    for (float v : a) s += (double) v * v;
    return a.empty() ? 0 : std::sqrt(s / a.size());
}

// Plain routed MUL_MAT_ID: dst[m, j, t] = sum_k w[k, m, ids[j,t]] * x[k, t].
static bool run_down(ggml_backend_t backend, ggml_type wtype, const std::vector<uint8_t> & wq,
                     const std::vector<float> & xf, const std::vector<int32_t> & idsh,
                     int64_t K, int64_t M, int64_t E, int64_t T, int64_t n_used,
                     std::vector<float> & out) {
    ggml_init_params ip{};
    ip.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false) + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, wtype, K, M, E);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    ggml_set_input(w); ggml_set_input(x); ggml_set_input(ids);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_set_output(dst);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    ggml_backend_tensor_set(ids, idsh.data(), 0, idsh.size() * sizeof(int32_t));
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "down: compute failed (%d)\n", (int) st);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.resize((size_t) M * n_used * T);
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Fused gate/up SwiGLU on two routed MUL_MAT_ID: glu[m,j,t] = silu(gate) * up.
static bool run_glu(ggml_backend_t backend, ggml_type wtype,
                    const std::vector<uint8_t> & wgate, const std::vector<uint8_t> & wup,
                    const std::vector<float> & xf, const std::vector<int32_t> & idsh,
                    int64_t K, int64_t M, int64_t E, int64_t T, int64_t n_used,
                    std::vector<float> & out) {
    ggml_init_params ip{};
    ip.mem_size = 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false) + 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * gw  = ggml_new_tensor_3d(ctx, wtype, K, M, E);
    ggml_tensor * uw  = ggml_new_tensor_3d(ctx, wtype, K, M, E);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    ggml_set_input(gw); ggml_set_input(uw); ggml_set_input(x); ggml_set_input(ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, gw, x, ids);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, uw, x, ids);
    ggml_tensor * glu  = ggml_swiglu_split(ctx, gate, up);
    ggml_set_output(glu);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(gf, glu);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(gw, wgate.data(), 0, wgate.size());
    ggml_backend_tensor_set(uw, wup.data(), 0, wup.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    ggml_backend_tensor_set(ids, idsh.data(), 0, idsh.size() * sizeof(int32_t));
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "glu: compute failed (%d)\n", (int) st);
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }
    out.resize((size_t) M * n_used * T);
    ggml_backend_tensor_get(glu, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    const int64_t K = 512, M = 256, E = 512, n_used = 10, T = 4096;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!cpu || !gpu) { std::fprintf(stderr, "backend init failed\n"); return 1; }
    if (!ggml_backend_cuda_qwen4exp_supported(gpu)) {
        std::fprintf(stderr, "[tile128] SKIP: qwen4exp profile is gfx1151 only\n");
        ggml_backend_free(cpu); ggml_backend_free(gpu);
        return 77;
    }
    ggml_backend_cuda_set_qwen4exp_profile(GGML_CUDA_QWEN4EXP_DEFAULT);

    // Routing: distinct experts within a token (10 fixed pairwise-distinct offsets mod
    // E=512), base rotated per token so that, averaged over T=4096 tokens, every expert
    // gets ~T*n_used/E = 80 rows -- squarely inside the 65-127 band this change targets.
    static const int32_t offs[10] = {0, 53, 107, 163, 211, 269, 311, 367, 419, 467};
    std::vector<int32_t> idsh((size_t) n_used * T);
    for (int64_t t = 0; t < T; ++t) {
        const int32_t base = (int32_t) ((t * 97) % E);
        for (int64_t j = 0; j < n_used; ++j) idsh[(size_t) t * n_used + j] = (base + offs[j]) % (int32_t) E;
    }
    std::vector<int> counts(E, 0);
    for (int32_t id : idsh) counts[id]++;
    int band = 0, min_c = counts[0], max_c = counts[0];
    for (int c : counts) { if (c >= 65 && c <= 127) ++band; min_c = std::min(min_c, c); max_c = std::max(max_c, c); }
    std::fprintf(stderr, "[tile128] routing: %d/%lld experts land in the 65-127 row band (min=%d max=%d)\n",
                 band, (long long) E, min_c, max_c);
    if (band < (int) E / 4) { std::fprintf(stderr, "[tile128] FAIL: routing too skewed to exercise the band\n"); return 1; }

    // Q5_1 "down" weight.
    const size_t row5_1 = ggml_row_size(GGML_TYPE_Q5_1, K);
    std::vector<float> wdf((size_t) K * M * E);
    for (auto & v : wdf) v = rnd();
    std::vector<uint8_t> wdq(row5_1 * M * E);
    {
        const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q5_1);
        for (int64_t r = 0; r < M * E; ++r) tr->from_float_ref(wdf.data() + r * K, wdq.data() + r * row5_1, K);
    }

    // Q4_K gate/up weights.
    const size_t row4k = ggml_row_size(GGML_TYPE_Q4_K, K);
    std::vector<float> wgf((size_t) K * M * E), wuf((size_t) K * M * E);
    for (auto & v : wgf) v = rnd();
    for (auto & v : wuf) v = rnd();
    std::vector<uint8_t> wgq(row4k * M * E), wuq(row4k * M * E);
    {
        const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q4_K);
        for (int64_t r = 0; r < M * E; ++r) {
            tr->from_float_ref(wgf.data() + r * K, wgq.data() + r * row4k, K);
            tr->from_float_ref(wuf.data() + r * K, wuq.data() + r * row4k, K);
        }
    }

    std::vector<float> xf((size_t) K * T);
    for (auto & v : xf) v = rnd();

    std::vector<float> down_cpu, down_128, down_65, glu_cpu, glu_128, glu_65;
    if (!run_down(cpu, GGML_TYPE_Q5_1, wdq, xf, idsh, K, M, E, T, n_used, down_cpu)) return 1;
    if (!run_glu(cpu, GGML_TYPE_Q4_K, wgq, wuq, xf, idsh, K, M, E, T, n_used, glu_cpu)) return 1;

    setenv("LUCE_MMB_TILE128_MIN", "128", 1);
    if (!run_down(gpu, GGML_TYPE_Q5_1, wdq, xf, idsh, K, M, E, T, n_used, down_128)) return 1;
    if (!run_glu(gpu, GGML_TYPE_Q4_K, wgq, wuq, xf, idsh, K, M, E, T, n_used, glu_128)) return 1;

    setenv("LUCE_MMB_TILE128_MIN", "65", 1);
    if (!run_down(gpu, GGML_TYPE_Q5_1, wdq, xf, idsh, K, M, E, T, n_used, down_65)) return 1;
    if (!run_glu(gpu, GGML_TYPE_Q4_K, wgq, wuq, xf, idsh, K, M, E, T, n_used, glu_65)) return 1;

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    const bool down_bitexact = down_128.size() == down_65.size() &&
        std::memcmp(down_128.data(), down_65.data(), down_128.size() * sizeof(float)) == 0;
    const bool glu_bitexact = glu_128.size() == glu_65.size() &&
        std::memcmp(glu_128.data(), glu_65.data(), glu_128.size() * sizeof(float)) == 0;

    const double down_r = rms(down_cpu), glu_r = rms(glu_cpu);
    std::fprintf(stderr, "[tile128] down (Q5_1): bit-identical(128 vs 65)=%s maxdiff=%.6g; "
                 "cpu sanity: maxdiff(128,cpu)=%.6g rel=%.4g rms=%.4g\n",
                 down_bitexact ? "YES" : "NO", maxdiff(down_128, down_65),
                 maxdiff(down_128, down_cpu), down_r > 0 ? maxdiff(down_128, down_cpu) / down_r : 0.0, down_r);
    std::fprintf(stderr, "[tile128] glu  (Q4_K): bit-identical(128 vs 65)=%s maxdiff=%.6g; "
                 "cpu sanity: maxdiff(128,cpu)=%.6g rel=%.4g rms=%.4g\n",
                 glu_bitexact ? "YES" : "NO", maxdiff(glu_128, glu_65),
                 maxdiff(glu_128, glu_cpu), glu_r > 0 ? maxdiff(glu_128, glu_cpu) / glu_r : 0.0, glu_r);

    // The quantized-vs-CPU-reference gap is a sanity bound only (CPU uses Q8_0/Q8_K
    // activation quantization internally for these vec_dot paths, same caveat as
    // test_mmb_cublas_diff), not an oracle for the threshold change itself -- both
    // THRESH=128 and THRESH=65 are compared to the SAME cpu reference and already
    // shown bit-identical to each other above, so this only guards against a
    // grossly broken dispatch (e.g. wrong operand order), not fine precision.
    // GLU's SwiGLU nonlinearity amplifies the quantization gap relative to the
    // plain dot product (down), hence the looser bound.
    const bool down_sane = std::isfinite(down_r) && down_r > 0 && maxdiff(down_128, down_cpu) < 0.05 * down_r;
    const bool glu_sane  = std::isfinite(glu_r)  && glu_r  > 0 && maxdiff(glu_128, glu_cpu) < 0.20 * glu_r;

    const bool ok = down_bitexact && glu_bitexact && down_sane && glu_sane;
    std::fprintf(stderr, "[tile128] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
