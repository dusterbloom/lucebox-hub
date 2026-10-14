// T=1 routed gate/up Q4_K GEMV microbench, gfx1151 (lucebox6).
// Shape: 512 experts, 10 routed, gate/up 2560 -> 640, one token.
// Arms: ours legacy (mmvq.cu variant 0, 8-warp), ours new (variant 1), header
// variants (rows x pairs grid), gufo mul_mat_vec_q_moe<Q4_K,2,true> (MIT).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1);} } while (0)

#include "common.cuh"
#include "mmvq-routed-gu.cuh"

extern "C" void ggml_backend_cuda_launch_mmvq_routed_gate_up(
        int type_, const void * up, const void * gate, const void * y_q8_1, const int32_t * ids, float * dst,
        int ncols_x, int nrows_x, int nchannels_dst, int variant, hipStream_t stream);

extern "C" void gufo_gated_q4k(const void * gate, const void * up, const void * y, const int32_t * ids, float * out,
                               int k, int rows, int experts_used, int input_stride, hipStream_t stream);

constexpr int K = 2560, M = 640, E = 512, NU = 10;
constexpr size_t ROW_BYTES = (size_t) (K / 256) * 144, EXPERT_BYTES = (size_t) M * ROW_BYTES, W_BYTES = (size_t) E * EXPERT_BYTES;
constexpr int Y_BLOCKS = K / 32;
static int g_layers = 4;

static uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; memcpy(&u, &h, 2); return u; }

struct Arm {
    const char * name;
    void (*launch)(int layer, int set, hipStream_t st);
};

struct Dev {
    std::vector<uint8_t *> wg, wu;      // per layer
    uint8_t * y;
    std::vector<int32_t *> ids;         // per id set
    float * out;
};
static Dev D;

static void launch_ours(int variant, int layer, int set, hipStream_t st) {
    ggml_backend_cuda_launch_mmvq_routed_gate_up(GGML_TYPE_Q4_K, D.wu[layer], D.wg[layer], D.y, D.ids[set], D.out, K, M, NU, variant, st);
}
static void launch_legacy(int l, int s, hipStream_t st) { launch_ours(0, l, s, st); }
static void launch_new(int l, int s, hipStream_t st) { launch_ours(1, l, s, st); }
static void launch_gufo(int l, int s, hipStream_t st) {
    gufo_gated_q4k(D.wg[l], D.wu[l], D.y, D.ids[s], D.out, K, M, NU, Y_BLOCKS, st);
}
template <int R, int P, bool SP = true>
static void launch_hdr(int l, int s, hipStream_t st) {
    launch_mul_mat_vec_q_routed_gate_up<GGML_TYPE_Q4_K, R, P, SP>(D.wg[l], D.wu[l], D.y, D.ids[s], D.out, K, M, NU, K / 256, M * (K / 256), M, 32, st);
}

int main(int argc, char ** argv) {
    int n_calls = 400, n_batches = 20;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n")) n_calls = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b")) n_batches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-L")) g_layers = atoi(argv[++i]);
    }
    hipStream_t st; CHECK(hipStreamCreate(&st));
    std::mt19937 rng(42);

    // Q4_K weights: d, dmin in a realistic range, random scales/qs.
    std::vector<uint8_t> w(W_BYTES);
    std::uniform_real_distribution<float> dd(0.0002f, 0.001f), dmd(0.00005f, 0.0004f);
    std::uniform_int_distribution<int> bd(0, 255);
    auto fill_w = [&](std::vector<uint8_t> & v) {
        uint8_t * p = v.data();
        for (size_t b = 0; b < v.size() / 144; ++b, p += 144) {
            const uint16_t d = f2h(dd(rng)), dm = f2h(dmd(rng)); memcpy(p, &d, 2); memcpy(p + 2, &dm, 2);
            for (int i = 4; i < 144; ++i) p[i] = (uint8_t) bd(rng);
        }
    };
    for (int l = 0; l < g_layers; ++l) {
        for (int which = 0; which < 2; ++which) {
            fill_w(w);
            uint8_t * d; CHECK(hipMalloc(&d, W_BYTES)); CHECK(hipMemcpy(d, w.data(), W_BYTES, hipMemcpyHostToDevice));
            (which ? D.wu : D.wg).push_back(d);
        }
    }
    // y: one q8_1 row (ds = {d, sum}, 32 int8).
    std::vector<uint8_t> y((size_t) Y_BLOCKS * 36);
    {
        std::uniform_int_distribution<int> qd(-127, 127);
        for (int b = 0; b < Y_BLOCKS; ++b) {
            int8_t * qs = (int8_t *) (y.data() + (size_t) b * 36 + 4);
            int sum = 0;
            for (int i = 0; i < 32; ++i) { qs[i] = (int8_t) qd(rng); sum += qs[i]; }
            const float d = 0.0123f;
            const uint16_t dh = f2h(d), sh = f2h(d * sum);
            memcpy(y.data() + (size_t) b * 36, &dh, 2); memcpy(y.data() + (size_t) b * 36 + 2, &sh, 2);
        }
        CHECK(hipMalloc(&D.y, y.size())); CHECK(hipMemcpy(D.y, y.data(), y.size(), hipMemcpyHostToDevice));
    }
    // id sets: one per layer (distinct experts), plus a masked set (two -1 routes) for correctness.
    const int n_sets = g_layers + 1;
    std::vector<std::vector<int32_t>> h_ids(n_sets);
    for (int s = 0; s < n_sets; ++s) {
        std::vector<int> perm(E); for (int e = 0; e < E; ++e) perm[e] = e; std::shuffle(perm.begin(), perm.end(), rng);
        h_ids[s].assign(perm.begin(), perm.begin() + NU);
        if (s == n_sets - 1) { h_ids[s][3] = -1; h_ids[s][7] = -1; }
        int32_t * d; CHECK(hipMalloc(&d, NU * 4)); CHECK(hipMemcpy(d, h_ids[s].data(), NU * 4, hipMemcpyHostToDevice));
        D.ids.push_back(d);
    }
    CHECK(hipMalloc(&D.out, (size_t) NU * M * 4));

    Arm arms[] = {
        {"legacy row_warps=8 (mmvq.cu v0)", launch_legacy},
        {"new 2-wave split (mmvq.cu v1)", launch_new},
        {"gufo mul_mat_vec_q_moe<Q4_K,2,true>", launch_gufo},
        {"hdr rows=1 pairs=1 scale_pair", launch_hdr<1, 1>},
        {"hdr rows=2 pairs=1 scale_pair", launch_hdr<2, 1>},
        {"hdr rows=4 pairs=1 scale_pair", launch_hdr<4, 1>},
        {"hdr rows=2 pairs=2 scale_pair", launch_hdr<2, 2>},
        {"hdr rows=1 pairs=2 scale_pair", launch_hdr<1, 2>},
        {"hdr rows=1 pairs=4 scale_pair", launch_hdr<1, 4>},
        {"hdr rows=1 pairs=1 branchy", launch_hdr<1, 1, false>},
        {"hdr rows=2 pairs=1 branchy", launch_hdr<2, 1, false>},
    };
    const int n_arms = sizeof(arms) / sizeof(arms[0]);

    // Correctness: every arm vs legacy, on every id set (incl. masked), layer 0.
    std::vector<std::vector<float>> ref(n_sets, std::vector<float>((size_t) NU * M));
    for (int s = 0; s < n_sets; ++s) {
        CHECK(hipMemset(D.out, 0xFF, (size_t) NU * M * 4));
        arms[0].launch(0, s, st); CHECK(hipStreamSynchronize(st));
        CHECK(hipMemcpy(ref[s].data(), D.out, (size_t) NU * M * 4, hipMemcpyDeviceToHost));
    }
    printf("%-40s %10s %10s %12s\n", "arm", "exact", "max_ulp", "max_abs");
    for (int a = 0; a < n_arms; ++a) {
        size_t exact = 0, total = 0; int max_ulp = 0; float max_abs = 0;
        std::vector<float> got((size_t) NU * M);
        for (int s = 0; s < n_sets; ++s) {
            CHECK(hipMemset(D.out, 0xFF, (size_t) NU * M * 4));
            arms[a].launch(0, s, st); CHECK(hipStreamSynchronize(st));
            CHECK(hipMemcpy(got.data(), D.out, (size_t) NU * M * 4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < got.size(); ++i) {
                ++total;
                int32_t ga, gb; memcpy(&ga, &got[i], 4); memcpy(&gb, &ref[s][i], 4);
                if (ga == gb) { ++exact; continue; }
                max_ulp = std::max(max_ulp, (int) std::min<int64_t>(1 << 30, std::llabs((int64_t) ga - gb)));
                max_abs = std::max(max_abs, std::fabs(got[i] - ref[s][i]));
            }
        }
        printf("%-40s %5zu/%5zu %10d %12.3e\n", arms[a].name, exact, total, max_ulp, max_abs);
    }

    // Timing: per-call events (median over n_calls, rotating layer+ids so MALL does not serve the weights),
    // and back-to-back batches of 47 calls (median of n_batches, per-call average).
    hipEvent_t ea, eb; CHECK(hipEventCreate(&ea)); CHECK(hipEventCreate(&eb));
    printf("\n%-40s %12s %12s %12s\n", "arm", "call_med_us", "call_min_us", "batch47_us");
    for (int a = 0; a < n_arms; ++a) {
        for (int i = 0; i < 20; ++i) arms[a].launch(i % g_layers, i % g_layers, st);
        CHECK(hipStreamSynchronize(st));
        std::vector<float> v;
        for (int i = 0; i < n_calls; ++i) {
            const int l = i % g_layers;
            CHECK(hipEventRecord(ea, st)); arms[a].launch(l, l, st); CHECK(hipEventRecord(eb, st)); CHECK(hipEventSynchronize(eb));
            float ms; CHECK(hipEventElapsedTime(&ms, ea, eb)); v.push_back(ms * 1000.f);
        }
        std::sort(v.begin(), v.end());
        std::vector<float> b;
        for (int r = 0; r < n_batches; ++r) {
            CHECK(hipEventRecord(ea, st));
            for (int i = 0; i < 47; ++i) { const int l = i % g_layers; arms[a].launch(l, l, st); }
            CHECK(hipEventRecord(eb, st)); CHECK(hipEventSynchronize(eb));
            float ms; CHECK(hipEventElapsedTime(&ms, ea, eb)); b.push_back(ms * 1000.f / 47.f);
        }
        std::sort(b.begin(), b.end());
        printf("%-40s %12.1f %12.1f %12.1f\n", arms[a].name, v[v.size() / 2], v[0], b[b.size() / 2]);
    }
    return 0;
}
