// HC combine + norm microbench (lucebox4, gfx1151): the Qwen4exp prefill hyper-connection combine kernels,
// per-stream / per-token-serial (hc_combine_norm_f32_b256, hc_combine_norm_moe_f32_b256<true>) vs the all-streams
// single-pass kernels (hc_combine_norm_f32_all<4>, hc_combine_norm_moe_f32_all<4,true>), vs gufo's HcCombine
// (f32 residual, f32 xn) and HcCombineMoeF16 (f32 or half residual, half xn + q8 tiles) (MIT).
// Includes ours/hc-cn.cu directly so the static kernels are callable. Shape: n_embd 2560, hc 4, n_used 10,
// T tokens; outputs of the prefill path: out_res f32, xn bf16 (in place, no f32), q8 tiles.
// Bit-exactness: old vs new memcmp on out_res / xn / q8 for both modes.
// usage: hccn_bench [-T 4136] [-n 30]
#include "ours/hc-cn.cu"

#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" bool ggml_are_same_shape(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] && a->ne[3] == b->ne[3];
}

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)

extern "C" void gufo_hc_combine(float * res, const float * block_out, const float * inject, const float * gamma, float * xn,
                                uint32_t n_tokens, uint32_t hidden, float eps, hipStream_t st);
extern "C" int  gufo_hc_combine_moe(void * res, const uint16_t * expert_out, const float * weights, const float * shared_out,
                                    const float * gate, uint32_t used, const float * inject, const float * gamma, uint16_t * xn,
                                    void * xn_q8, uint32_t n_tokens, uint32_t hidden, float eps, hipStream_t st, int res_half);

static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <typename F> static float time_ms(hipStream_t st, int n, F f) {
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    std::vector<float> t;
    for (int i = 0; i < n + 3; ++i) {
        CHECK(hipEventRecord(e0, st)); f(); CHECK(hipEventRecord(e1, st)); CHECK(hipEventSynchronize(e1));
        float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); if (i >= 3) t.push_back(ms);
    }
    return median(t);
}

template <typename T> static T * dev(const std::vector<T> & h) { T * d; CHECK(hipMalloc(&d, h.size() * sizeof(T))); CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); return d; }
template <typename T> static std::vector<T> host(const T * d, size_t n) { std::vector<T> h(n); CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
static uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; memcpy(&u, &h, 2); return u; }

int main(int argc, char ** argv) {
    int T = 4136, n = 30;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-T")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) n = atoi(argv[++i]);
    }
    const int E = 2560, HC = 4, NU = 10;
    const float s1 = 0.5f, b1 = 0.1f, s2 = 2.0f, b2 = 0.0f, eps = 1e-6f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> inject((size_t) T * HC), residual((size_t) T * HC * E), blk((size_t) T * E), gamma((size_t) HC * E);
    std::vector<float> moe_w((size_t) T * NU), shared((size_t) T * E), sh_logit(T);
    std::vector<uint16_t> down((size_t) T * NU * E);
    for (auto & x : inject) x = nd(rng);
    for (auto & x : residual) x = 4.0f * nd(rng);
    for (auto & x : blk) x = nd(rng);
    for (auto & x : gamma) x = 1.0f + 0.1f * nd(rng);
    for (auto & x : moe_w) x = 0.1f * fabsf(nd(rng));
    for (auto & x : shared) x = nd(rng);
    for (auto & x : sh_logit) x = nd(rng);
    for (auto & x : down) x = f2h(nd(rng));
    float * d_inj = dev(inject), * d_res = dev(residual), * d_blk = dev(blk), * d_gamma = dev(gamma);
    float * d_w = dev(moe_w), * d_sh = dev(shared), * d_shl = dev(sh_logit);
    uint16_t * d_down = dev(down);
    const size_t nres = (size_t) T * HC * E;
    const size_t q8_bytes = (size_t) ((T + 15) / 16) * (HC * E / 32) * 576 * 2;
    float * o_res[2]; uint16_t * o_xn[2]; int8_t * o_q8[2]; float * o_xnf;
    for (int i = 0; i < 2; ++i) { CHECK(hipMalloc(&o_res[i], nres * 4)); CHECK(hipMalloc(&o_xn[i], nres * 2)); CHECK(hipMalloc(&o_q8[i], q8_bytes)); }
    CHECK(hipMalloc(&o_xnf, nres * 4));
    uint16_t * h_res; CHECK(hipMalloc(&h_res, nres * 2));
    hipStream_t st; CHECK(hipStreamCreate(&st));
    const dim3 blockd(HC_CN_BLOCK2);

    auto old_cn = [&](int i) {
        hc_combine_norm_f32_b256<<<dim3(HC, T, 1), blockd, 0, st>>>(d_inj, d_res, d_blk, d_gamma, o_res[i], o_xnf, o_xn[i], false,
            nullptr, nullptr, nullptr, E, s1, b1, s2, b2, eps, o_q8[i], nullptr);
    };
    auto new_cn = [&](int i) {
        hc_combine_norm_f32_all<4><<<dim3(T, 1, 1), blockd, 0, st>>>(d_inj, d_res, d_blk, d_gamma, o_res[i], o_xnf, o_xn[i], false,
            E, s1, b1, s2, b2, eps, o_q8[i]);
    };
    auto old_moe = [&](int i) {
        hc_combine_norm_moe_f32_b256<true><<<dim3(T, 1, 1), blockd, 0, st>>>(d_inj, d_res, d_gamma, d_down, d_w, d_sh, d_shl,
            o_res[i], o_xnf, o_xn[i], false, E, HC, NU, E, (int64_t) NU * E, NU, E, s1, b1, s2, b2, eps, o_q8[i]);
    };
    auto new_moe = [&](int i) {
        hc_combine_norm_moe_f32_all<4, true><<<dim3(T, 1, 1), blockd, 0, st>>>(d_inj, d_res, d_gamma, d_down, d_w, d_sh, d_shl,
            o_res[i], o_xnf, o_xn[i], false, E, NU, E, (int64_t) NU * E, NU, E, s1, b1, s2, b2, eps, o_q8[i]);
    };
    // gufo's HcCombine updates res in place: copy the residual into the output first (outside the timed region
    // would be unfair for the memcpy, so time the kernel alone on a pre-copied buffer; the kernel overwrites it).
    auto gufo_cn = [&]() { gufo_hc_combine(o_res[1], d_blk, d_inj, d_gamma, o_xnf, T, E, eps, st); };
    auto gufo_moe_f32 = [&]() { gufo_hc_combine_moe(o_res[1], d_down, d_w, d_sh, d_shl, NU, d_inj, d_gamma, o_xn[1], o_q8[1], T, E, eps, st, 0); };
    auto gufo_moe_h   = [&]() { gufo_hc_combine_moe(h_res, d_down, d_w, d_sh, d_shl, NU, d_inj, d_gamma, o_xn[1], o_q8[1], T, E, eps, st, 1); };

    auto check = [&](const char * what) {
        CHECK(hipStreamSynchronize(st));
        auto r0 = host(o_res[0], nres), r1 = host(o_res[1], nres);
        auto x0 = host(o_xn[0], nres), x1 = host(o_xn[1], nres);
        auto q0 = host(o_q8[0], q8_bytes / 2), q1 = host(o_q8[1], q8_bytes / 2);
        const bool ok = !memcmp(r0.data(), r1.data(), nres * 4) && !memcmp(x0.data(), x1.data(), nres * 2) && !memcmp(q0.data(), q1.data(), q8_bytes / 2);
        double maxd = 0; for (size_t i = 0; i < nres; ++i) maxd = std::max(maxd, (double) fabsf(r0[i] - r1[i]));
        printf("%s: old vs new %s (res max |d| %.3g, xn %s, q8 %s)\n", what, ok ? "BIT-IDENTICAL" : "DIFFER", maxd,
               memcmp(x0.data(), x1.data(), nres * 2) ? "differ" : "same", memcmp(q0.data(), q1.data(), q8_bytes / 2) ? "differ" : "same");
    };
    CHECK(hipMemset(o_q8[0], 0, q8_bytes)); CHECK(hipMemset(o_q8[1], 0, q8_bytes));
    old_cn(0); new_cn(1); check("combine");
    CHECK(hipMemset(o_q8[0], 0, q8_bytes)); CHECK(hipMemset(o_q8[1], 0, q8_bytes));
    old_moe(0); new_moe(1); check("combine-moe");

    printf("T=%d n_embd=%d hc=%d n_used=%d, medians of %d (ms)\n", T, E, HC, NU, n);
    printf("combine     old %.3f  new %.3f  gufo(f32 res, f32 xn, no q8) %.3f\n",
           time_ms(st, n, [&] { old_cn(0); }), time_ms(st, n, [&] { new_cn(0); }), time_ms(st, n, gufo_cn));
    printf("combine-moe old %.3f  new %.3f  gufo f32-res %.3f  gufo half-res %.3f\n",
           time_ms(st, n, [&] { old_moe(0); }), time_ms(st, n, [&] { new_moe(0); }), time_ms(st, n, gufo_moe_f32), time_ms(st, n, gufo_moe_h));
    const double bytes_cn = (double) T * (HC * E * 4.0 * 2 + E * 4 + HC * E * 2 + HC * E * 1.125);
    const double bytes_moe = (double) T * (HC * E * 4.0 * 2 + NU * E * 2 + E * 4 + HC * E * 2 + HC * E * 1.125);
    printf("min traffic: combine %.0f MB, combine-moe %.0f MB (220 GB/s floor %.2f / %.2f ms)\n", bytes_cn / 1e6, bytes_moe / 1e6, bytes_cn / 220e9 * 1e3, bytes_moe / 220e9 * 1e3);
    return 0;
}
