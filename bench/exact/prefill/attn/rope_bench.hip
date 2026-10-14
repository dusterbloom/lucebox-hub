// Prefill M-RoPE microbench (lucebox4, gfx1151): the per-token rope_multi_heads kernel with the inline FP64
// angle code (legacy, LUCE_ROPE_TABLE_LEGACY=1) vs the per-forward cos/sin table (rope_multi_table + table apply).
// Includes ours/rope.cu directly so the static launchers are callable. Shapes are the Qwen4exp prefill rope
// calls per layer: Q [256,24,T], K [256,2,T], indexer q [128,4,T], pooled [128,1,T/4]; n_dims 64, theta 1e7,
// sections {11,11,10,0}, positions section-major [s*T+i] (0..2 carry the position, 3 is zero).
// usage: rope_bench [-T 4136] [-n 50]
#include "ours/rope.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

bool ggml_cuda_qwen4exp_reference() { return false; }
extern "C" void ggml_rope_yarn_corr_dims(int, int, float, float, float, float dims[2]) { dims[0] = dims[1] = 0.0f; }

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)

struct Shape { const char * name; int ne00, ne01, T; };

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char ** argv) {
    int T = 4136, n = 50;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-T")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) n = atoi(argv[++i]);
    }
    const int n_dims = 64;
    const float freq_base = 1e7f, theta_scale = powf(freq_base, -2.0f / n_dims);
    (void) theta_scale;
    mrope_sections sections{{11, 11, 10, 0}};
    rope_corr_dims corr{{0.f, 0.f}};
    hipStream_t st; CHECK(hipStreamCreate(&st));
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));

    const Shape shapes[] = { {"Q", 256, 24, T}, {"K", 256, 2, T}, {"idx_q", 128, 4, T}, {"pooled", 128, 1, T / 4} };
    double legacy_layer = 0, table_layer = 0;
    for (const Shape & sh : shapes) {
        const int Tt = sh.T;
        std::vector<int32_t> hpos((size_t) 4 * Tt);
        for (int s = 0; s < 4; ++s) for (int i = 0; i < Tt; ++i) hpos[(size_t) s * Tt + i] = s < 3 ? i : 0;
        const size_t nel = (size_t) sh.ne00 * sh.ne01 * Tt;
        std::vector<float> hx(nel);
        uint32_t r = 12345u;
        for (float & v : hx) { r = r * 1664525u + 1013904223u; v = ((r >> 8) & 0xFFFF) / 32768.0f - 1.0f; }
        int32_t * pos; float * x; float * d0; float * d1;
        CHECK(hipMalloc(&pos, hpos.size() * 4)); CHECK(hipMemcpy(pos, hpos.data(), hpos.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&x, nel * 4)); CHECK(hipMemcpy(x, hx.data(), nel * 4, hipMemcpyHostToDevice));
        CHECK(hipMalloc(&d0, nel * 4)); CHECK(hipMalloc(&d1, nel * 4));
        const int s01 = sh.ne00, s02 = sh.ne00 * sh.ne01, s03 = s02 * Tt, nr = sh.ne01 * Tt;
        auto run = [&](float * dst, const void * node) {
            rope_multi_cuda<true, float>(x, dst, sh.ne00, sh.ne01, Tt, s01, s02, s03, s01, s02, s03, n_dims, nr, pos, node,
                1.0f, freq_base, 0.0f, 1.0f, corr, nullptr, sections, false, st);
        };
        int dummy;
        // correctness: legacy vs table (table built on the first call)
        run(d0, nullptr);
        ggml_cuda_rope_begin_graph();
        run(d1, &dummy);
        CHECK(hipStreamSynchronize(st));
        std::vector<float> h0(nel), h1(nel);
        CHECK(hipMemcpy(h0.data(), d0, nel * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(h1.data(), d1, nel * 4, hipMemcpyDeviceToHost));
        const bool same = memcmp(h0.data(), h1.data(), nel * 4) == 0;
        // timings: legacy per call; table build (fresh generation, one call = build + apply); table hit per call
        std::vector<float> tl, tb, th;
        for (int i = 0; i < n; ++i) {
            CHECK(hipEventRecord(e0, st)); run(d0, nullptr); CHECK(hipEventRecord(e1, st)); CHECK(hipEventSynchronize(e1));
            float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); tl.push_back(ms);
            ggml_cuda_rope_begin_graph();
            CHECK(hipEventRecord(e0, st)); run(d1, &dummy); CHECK(hipEventRecord(e1, st)); CHECK(hipEventSynchronize(e1));
            CHECK(hipEventElapsedTime(&ms, e0, e1)); tb.push_back(ms);
            CHECK(hipEventRecord(e0, st)); run(d1, &dummy); CHECK(hipEventRecord(e1, st)); CHECK(hipEventSynchronize(e1));
            CHECK(hipEventElapsedTime(&ms, e0, e1)); th.push_back(ms);
        }
        const double bytes = 2.0 * nel * 4;
        printf("%-7s [%d,%d,%d] legacy %.4f ms (%.0f GB/s)  table build+apply %.4f ms  table hit %.4f ms (%.0f GB/s)  bit-identical %s\n",
            sh.name, sh.ne00, sh.ne01, Tt, median(tl), bytes / median(tl) / 1e6, median(tb), median(th), bytes / median(th) / 1e6, same ? "yes" : "NO");
        legacy_layer += median(tl); table_layer += median(th);
        if (!same) { fprintf(stderr, "MISMATCH %s\n", sh.name); return 1; }
        CHECK(hipFree(pos)); CHECK(hipFree(x)); CHECK(hipFree(d0)); CHECK(hipFree(d1));
    }
    printf("per layer (4 calls): legacy %.3f ms, table hits %.3f ms; per 12-layer request legacy %.2f ms vs table %.2f ms + 2 builds\n",
        legacy_layer, table_layer, 12 * legacy_layer, 12 * table_layer);
    return 0;
}
