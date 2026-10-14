// Routed MoE gate/up SwiGLU microbench, gfx1151 (lucebox6). One Qwen3.8-Flash-Next MoE layer:
// hidden K=2560, expert_ff M=640, E=512 experts, top-10, Q4_K gate/up.
// Ours: mmb-routed.cuh kernels through the production tile schedule (legacy two-launch and the new path);
// gufo: RoutedGatedF16Gemm (Q4_K pair kernel) linked from libgufo_qwen38_flash_next.a on identical inputs.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1);} } while (0)

#include "common.cuh"
#include "unary.cuh"

namespace {
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));
constexpr int MMB_BK = 64, MMB_NT = 256, MMB_LDS_STRIDE = MMB_BK + 8;
__device__ __forceinline__ uint16_t mmb_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
__device__ __forceinline__ uint32_t mmb_pack2(float a, float b) { return (uint32_t)mmb_f2bf(a) | ((uint32_t)mmb_f2bf(b) << 16); }
__constant__ int8_t mmb_kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ __forceinline__ float mmb_h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
}
#include "mmb-quant.cuh"
#include "mmb-q8f16.cuh"
namespace {
__device__ __forceinline__ void mmb_dq_row36(const uint4, const uint4, const uint32_t, uint32_t *) {}
#include "mmb-routed.cuh"
}

namespace gufo::models::qwen38_flash_next::rocm {
enum class WeightType : std::uint32_t { kF32 = 0, kF16 = 1, kQ5_1 = 7, kQ8_0 = 8, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kIQ4_NL = 20, kIQ3_S = 21, kIQ4_XS = 23, kBF16 = 30 };
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type, const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows, const std::int32_t* pad_bounds, const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out, std::size_t m, std::size_t k, hipStream_t stream);
}
namespace G = gufo::models::qwen38_flash_next::rocm;
namespace gufo::quant { const std::uint32_t * Iq3sGrid() { return nullptr; } }   // only the IQ3_S instantiations reach it

static uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float) h; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; memcpy(&f, &u, 4); return f; }

static int g_iters = 15;
template <typename F>
static double time_ms(F && launch, hipStream_t st) {
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) launch();
    CHECK(hipStreamSynchronize(st));
    std::vector<float> v;
    for (int i = 0; i < g_iters; ++i) {
        CHECK(hipEventRecord(a, st)); launch(); CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
        float ms; CHECK(hipEventElapsedTime(&ms, a, b)); v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b));
    return v[v.size() / 2];
}

constexpr int K = 2560, M = 640, E = 512, NU = 10;
constexpr double FLOOR_MS = 4.52;   // 2*T*NU*K*2M at 59.4 TF/s for T=4096 (scaled by T below)

struct Routing {
    std::vector<int32_t> ids;                        // [T][NU]
    std::vector<int32_t> counts, bounds, ids_src, ids_dst;   // ours (mm_ids order: by expert, then route index)
    std::vector<int32_t> pad_bounds, rows_tok, rows_slot;    // gufo (16-padded buckets)
    std::vector<int32_t> tiles128, tiles64;                  // gufo tile maps
    int n_rows = 0;
};

static Routing make_routing(int T, const char * ids_file, uint32_t seed) {
    Routing r; r.n_rows = T * NU; r.ids.resize((size_t) r.n_rows);
    if (ids_file) {
        FILE * f = fopen(ids_file, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", ids_file); exit(1); }
        if (fread(r.ids.data(), 4, r.ids.size(), f) != r.ids.size()) { fprintf(stderr, "short ids file\n"); exit(1); }
        fclose(f);
    } else {
        // Skewed synthetic top-10: expert weight exp(-3 * rank / E) over a random permutation of expert ids
        // (max/min weight ratio 20), every 20th rank dead (weight 0), sampled without replacement per token.
        std::mt19937 rng(seed);
        std::vector<int> perm(E); for (int e = 0; e < E; ++e) perm[e] = e; std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<double> w(E);
        for (int rk = 0; rk < E; ++rk) w[perm[rk]] = (rk % 20 == 19) ? 0.0 : exp(-3.0 * rk / E);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        for (int t = 0; t < T; ++t) {
            std::vector<double> ww = w;
            for (int s = 0; s < NU; ++s) {
                double tot = 0; for (int e = 0; e < E; ++e) tot += ww[e];
                double x = u01(rng) * tot, acc = 0; int pick = E - 1;
                for (int e = 0; e < E; ++e) { acc += ww[e]; if (x < acc) { pick = e; break; } }
                r.ids[(size_t) t * NU + s] = pick; ww[pick] = 0;
            }
        }
    }
    r.counts.assign(E, 0);
    for (int32_t e : r.ids) r.counts[e]++;
    r.bounds.assign(E + 1, 0);
    for (int e = 0; e < E; ++e) r.bounds[e + 1] = r.bounds[e] + r.counts[e];
    r.ids_src.resize(r.n_rows); r.ids_dst.resize(r.n_rows);
    r.pad_bounds.assign(E + 1, 0);
    for (int e = 0; e < E; ++e) r.pad_bounds[e + 1] = r.pad_bounds[e] + (r.counts[e] + 15) / 16 * 16;
    r.rows_tok.assign(r.pad_bounds[E], -1); r.rows_slot.assign(r.pad_bounds[E], -1);
    std::vector<int32_t> cur(E, 0);
    for (int route = 0; route < r.n_rows; ++route) {
        const int e = r.ids[route], pos = r.bounds[e] + cur[e], ppos = r.pad_bounds[e] + cur[e]; cur[e]++;
        r.ids_src[pos] = route / NU; r.ids_dst[pos] = route;
        r.rows_tok[ppos] = route / NU; r.rows_slot[ppos] = route;
    }
    for (int e = 0; e < E; ++e) {
        for (int j = 0; j < (r.counts[e] + 127) / 128; ++j) r.tiles128.push_back(e | (j << 16));
        const int padded = (r.counts[e] + 15) / 16 * 16;
        for (int j = 0; j < (padded + 63) / 64; ++j) r.tiles64.push_back(e | (j << 16));
    }
    std::vector<int32_t> sc = r.counts; std::sort(sc.begin(), sc.end());
    int zero = 0, over128 = 0, over256 = 0, band65 = 0; for (int c : sc) { zero += c == 0; over128 += c >= 128; over256 += c > 256; band65 += c >= 65 && c <= 127; }
    printf("routing: %s T=%d rows=%d experts: min=%d p50=%d mean=%.1f max=%d zero=%d 65-127=%d >=128=%d >256=%d; gufo tiles128=%zu tiles64=%zu\n",
        ids_file ? ids_file : "synthetic exp(-3 rank/E), 1/20 dead", T, r.n_rows, sc[0], sc[E / 2], (double) r.n_rows / E, sc[E - 1], zero, band65, over128, over256, r.tiles128.size(), r.tiles64.size());
    return r;
}

// Q4_K dequant exactly as mmb_dq_q4k_stripe (F32 d*sc, dm*mn; fma(ds, q, -ms); RNE to bf16), for the CPU spot check.
static float q4k_value_bf16(const uint8_t * blk, int k) {
    uint16_t d16, dm16; memcpy(&d16, blk, 2); memcpy(&dm16, blk + 2, 2);
    const float d = h2f(d16), dm = h2f(dm16);
    const uint8_t * sc = blk + 4; const uint8_t * qs = blk + 16;
    const int j = k / 32;   // sub-block
    uint32_t s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    const float ds = d * s, ms = dm * m;
    const int slice = k / 64, within = k % 64;
    const uint8_t q = qs[32 * slice + (within % 32)];
    const int v = within < 32 ? (q & 15) : (q >> 4);
    return bf2f(f2bf(fmaf(ds, (float) v, -ms)));
}

struct Dev {
    uint8_t * wg, * wu; uint16_t * xb, * xh; float * d; uint16_t * dh; __half * gout;
    int32_t * ids_src, * ids_dst, * bounds, * pad_bounds, * rows_tok, * rows_slot, * tiles128, * tiles64;
    uint32_t * desc_big, * desc_small;
};

#include "f16kern.cuh"
int main(int argc, char ** argv) {
    int T = 4096; const char * ids_file = nullptr; uint32_t seed = 1234; int t16k = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-T")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) g_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ids")) ids_file = argv[++i];
        else if (!strcmp(argv[i], "-seed")) seed = (uint32_t) atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [-T tokens] [-n iters] [-ids file] [-seed s]\n", argv[0]); return 1; }
    }
    (void) t16k;
    hipStream_t st; CHECK(hipStreamCreate(&st));
    std::mt19937 rng(42);
    Routing R = make_routing(T, ids_file, seed);
    const int n_rows = R.n_rows;
    const double flop = 2.0 * n_rows * K * 2.0 * M;
    const double floor_ms = flop / 59.4e12 * 1e3;

    // Q4_K weights: [E][M][K/256 blocks of 144 B]
    const size_t row_bytes = (size_t) (K / 256) * 144, expert_bytes = (size_t) M * row_bytes, w_bytes = (size_t) E * expert_bytes;
    std::vector<uint8_t> wg(w_bytes), wu(w_bytes);
    {
        std::uniform_real_distribution<float> dd(0.0002f, 0.001f), dmd(0.00005f, 0.0004f);
        std::uniform_int_distribution<int> bd(0, 255);
        for (std::vector<uint8_t> * w : {&wg, &wu}) {
            uint8_t * p = w->data();
            for (size_t b = 0; b < w_bytes / 144; ++b, p += 144) {
                const uint16_t d = f2h(dd(rng)), dm = f2h(dmd(rng)); memcpy(p, &d, 2); memcpy(p + 2, &dm, 2);
                for (int i = 4; i < 144; ++i) p[i] = (uint8_t) bd(rng);
            }
        }
    }
    // activations: bf16 values in [-1,1) that are also exact F16 (so gufo sees identical inputs)
    std::vector<uint16_t> xb((size_t) T * K), xh((size_t) T * K);
    { std::uniform_real_distribution<float> xd(-1.f, 1.f); for (size_t i = 0; i < xb.size(); ++i) { const uint16_t b = f2bf(xd(rng)); xb[i] = b; xh[i] = f2h(bf2f(b)); } }

    Dev D;
    CHECK(hipMalloc(&D.wg, w_bytes)); CHECK(hipMalloc(&D.wu, w_bytes));
    CHECK(hipMalloc(&D.xb, xb.size() * 2)); CHECK(hipMalloc(&D.xh, xh.size() * 2));
    CHECK(hipMalloc(&D.d, (size_t) n_rows * M * 4)); CHECK(hipMalloc(&D.dh, (size_t) n_rows * M * 2)); CHECK(hipMalloc(&D.gout, (size_t) n_rows * M * 2));
    CHECK(hipMalloc(&D.ids_src, n_rows * 4)); CHECK(hipMalloc(&D.ids_dst, n_rows * 4)); CHECK(hipMalloc(&D.bounds, (E + 1) * 4));
    CHECK(hipMalloc(&D.pad_bounds, (E + 1) * 4)); CHECK(hipMalloc(&D.rows_tok, R.rows_tok.size() * 4)); CHECK(hipMalloc(&D.rows_slot, R.rows_slot.size() * 4));
    CHECK(hipMalloc(&D.tiles128, R.tiles128.size() * 4)); CHECK(hipMalloc(&D.tiles64, R.tiles64.size() * 4));
    CHECK(hipMemcpy(D.wg, wg.data(), w_bytes, hipMemcpyHostToDevice)); CHECK(hipMemcpy(D.wu, wu.data(), w_bytes, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(D.xb, xb.data(), xb.size() * 2, hipMemcpyHostToDevice)); CHECK(hipMemcpy(D.xh, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(D.ids_src, R.ids_src.data(), n_rows * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(D.ids_dst, R.ids_dst.data(), n_rows * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(D.bounds, R.bounds.data(), (E + 1) * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(D.pad_bounds, R.pad_bounds.data(), (E + 1) * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(D.rows_tok, R.rows_tok.data(), R.rows_tok.size() * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(D.rows_slot, R.rows_slot.data(), R.rows_slot.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(D.tiles128, R.tiles128.data(), R.tiles128.size() * 4, hipMemcpyHostToDevice)); CHECK(hipMemcpy(D.tiles64, R.tiles64.data(), R.tiles64.size() * 4, hipMemcpyHostToDevice));

    // legacy schedule, exactly as ggml_cuda_mul_mat_id_mmb_glu
    constexpr int BN = 128, BN_SMALL = 64, THRESH = 128, WT = 32 + GGML_TYPE_Q4_K;
    const int nbig_max = n_rows / BN + E + 1, nsmall_max = E * ((THRESH + BN_SMALL - 1) / BN_SMALL) + 1;
    CHECK(hipMalloc(&D.desc_big, nbig_max * 4)); CHECK(hipMalloc(&D.desc_small, nsmall_max * 4));
    int nbig = 0, nsmall = 0;
    for (int e = 0; e < E; ++e) { const int c = R.counts[e]; if (c >= THRESH) nbig += (c + BN - 1) / BN; else nsmall += (c + BN_SMALL - 1) / BN_SMALL; }
    printf("T=%d iters=%d flop=%.1f G floor(59.4TF)=%.2f ms; legacy tiles: big(128)=%d/%d small(64)=%d/%d grid rows\n", T, g_iters, flop * 1e-9, floor_ms, nbig, nbig_max, nsmall, nsmall_max);
    printf("| kernel | ms | TFLOP/s | %% of floor | vs legacy F32/bf16 |\n|---|---:|---:|---:|---|\n");

    auto build_desc = [&] { mmb_build_desc2<<<1, 1024, 0, st>>>(D.bounds, D.desc_big, D.desc_small, E, nbig_max, nsmall_max, BN, BN_SMALL, THRESH); };
    auto legacy = [&](bool f32, bool tight) {
        dim3 gbig(M / 64, tight ? nbig : nbig_max), gsmall(M / 64, tight ? nsmall : nsmall_max);
        float * d = f32 ? D.d : nullptr; uint16_t * dh = D.dh;
        mmb_routed_glu_kernel<64, BN, 32, 32, WT><<<gbig, MMB_NT, 0, st>>>(D.wg, D.wu, expert_bytes, D.xb, d, dh, f32, D.ids_src, D.ids_dst, D.bounds, D.desc_big, M, K);
        mmb_routed_glu_kernel<64, BN_SMALL, 16, 32, WT><<<gsmall, MMB_NT, 0, st>>>(D.wg, D.wu, expert_bytes, D.xb, d, dh, f32, D.ids_src, D.ids_dst, D.bounds, D.desc_small, M, K);
    };
    std::vector<float> ref_f32((size_t) n_rows * M), out_f32((size_t) n_rows * M);
    std::vector<uint16_t> ref_bf16((size_t) n_rows * M), out_bf16((size_t) n_rows * M), gufo_h((size_t) n_rows * M);
    auto grab = [&](std::vector<float> & f, std::vector<uint16_t> & h) { CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(f.data(), D.d, f.size() * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(h.data(), D.dh, h.size() * 2, hipMemcpyDeviceToHost)); };
    auto report = [&](const char * name, double ms, const std::vector<float> * f, const std::vector<uint16_t> * h) {
        std::string cmp = "-";
        if (f) { bool ex = true; double maxabs = 0, maxrel = 0; for (size_t i = 0; i < f->size(); ++i) { const double dd = fabs((double) (*f)[i] - ref_f32[i]); if (dd != 0) ex = false; maxabs = std::max(maxabs, dd); maxrel = std::max(maxrel, dd / std::max(fabs((double) ref_f32[i]), 1e-2)); }
            cmp = ex ? "F32 bit-exact" : "F32 maxabs=" + std::to_string(maxabs) + " maxrel=" + std::to_string(maxrel); }
        if (h) { bool ex = true; double maxabs = 0, maxrel = 0; for (size_t i = 0; i < h->size(); ++i) { const double dd = fabs((double) bf2f((*h)[i]) - bf2f(ref_bf16[i])); if ((*h)[i] != ref_bf16[i]) ex = false; maxabs = std::max(maxabs, dd); maxrel = std::max(maxrel, dd / std::max(fabs((double) bf2f(ref_bf16[i])), 1e-2)); }
            cmp += ex ? "; bf16 bit-exact" : "; bf16 maxabs=" + std::to_string(maxabs) + " maxrel=" + std::to_string(maxrel); }
        printf("| %-44s | %7.3f | %5.1f | %5.1f%% | %s |\n", name, ms, flop / ms / 1e9, floor_ms / ms * 100.0, cmp.c_str()); fflush(stdout);
    };

    const bool f16_only = getenv("MOE_F16_ONLY") != nullptr;   // skip the other arms and the double reference
    if (!f16_only) {
    // reference: legacy, F32 + bf16 outputs
    CHECK(hipMemset(D.d, 0, (size_t) n_rows * M * 4)); CHECK(hipMemset(D.dh, 0, (size_t) n_rows * M * 2));
    build_desc();
    double ms = time_ms([&] { legacy(true, false); }, st);
    grab(ref_f32, ref_bf16);
    {   // CPU spot check of the reference
        double worst = 0;
        for (int n = 0; n < 48; ++n) {
            const int pos = (int) (rng() % n_rows), col = (int) (rng() % M);
            int e = 0; while (R.bounds[e + 1] <= pos) ++e;
            const int tok = R.ids_src[pos], dst = R.ids_dst[pos];
            const uint8_t * rg = wg.data() + (size_t) e * expert_bytes + (size_t) col * row_bytes, * ru = wu.data() + (size_t) e * expert_bytes + (size_t) col * row_bytes;
            double g = 0, u = 0;
            for (int k = 0; k < K; ++k) { const double x = bf2f(xb[(size_t) tok * K + k]); g += (double) q4k_value_bf16(rg + (k / 256) * 144, k % 256) * x; u += (double) q4k_value_bf16(ru + (k / 256) * 144, k % 256) * x; }
            const double want = g / (1.0 + exp(-g)) * u, got = ref_f32[(size_t) dst * M + col];
            worst = std::max(worst, fabs(want - got) / std::max(fabs(want), 1e-2));
        }
        printf("| %-44s | %7.3f | %5.1f | %5.1f%% | cpu spot maxrel=%.2e |\n", "ours legacy 2-launch (F32+bf16 out)", ms, flop / ms / 1e9, floor_ms / ms * 100.0, worst);
    }
    double dms = time_ms([&] { build_desc(); }, st);
    printf("| %-44s | %7.3f | | | desc build only |\n", "mmb_build_desc2", dms);
    CHECK(hipMemset(D.dh, 0, (size_t) n_rows * M * 2));
    ms = time_ms([&] { legacy(false, false); }, st); grab(out_f32, out_bf16); report("ours legacy 2-launch (bf16 out, in-situ)", ms, nullptr, &out_bf16);
    CHECK(hipMemset(D.dh, 0, (size_t) n_rows * M * 2));
    ms = time_ms([&] { legacy(false, true); }, st); grab(out_f32, out_bf16); report("ours legacy, tight grid (no empty descs)", ms, nullptr, &out_bf16);
    {   // per-launch split
        dim3 gbig(M / 64, nbig_max), gsmall(M / 64, nsmall_max);
        double b = time_ms([&] { mmb_routed_glu_kernel<64, BN, 32, 32, WT><<<gbig, MMB_NT, 0, st>>>(D.wg, D.wu, expert_bytes, D.xb, nullptr, D.dh, false, D.ids_src, D.ids_dst, D.bounds, D.desc_big, M, K); }, st);
        double s = time_ms([&] { mmb_routed_glu_kernel<64, BN_SMALL, 16, 32, WT><<<gsmall, MMB_NT, 0, st>>>(D.wg, D.wu, expert_bytes, D.xb, nullptr, D.dh, false, D.ids_src, D.ids_dst, D.bounds, D.desc_small, M, K); }, st);
        printf("| %-44s | %7.3f | | | big=%.3f small=%.3f |\n", "  legacy per launch", b + s, b, s);
    }

    // gufo pair kernel, 128- and 64-token maps
    {
        CHECK(hipMemset(D.gout, 0, (size_t) n_rows * M * 2));
        ms = time_ms([&] { if (!G::RoutedGatedF16Gemm(D.wg, D.wu, G::WeightType::kQ4_K, (const __half *) D.xh, D.tiles128, (uint32_t) R.tiles128.size(), 128, D.pad_bounds, D.rows_tok, D.rows_slot, D.gout, M, K, st)) { fprintf(stderr, "gufo 128 failed\n"); exit(1); } }, st);
        CHECK(hipDeviceSynchronize()); CHECK(hipMemcpy(gufo_h.data(), D.gout, gufo_h.size() * 2, hipMemcpyDeviceToHost));
        double maxabs = 0, maxrel = 0; size_t nonfinite = 0, worst_i = 0;
        for (size_t i = 0; i < gufo_h.size(); ++i) { const double gv = h2f(gufo_h[i]); if (!std::isfinite(gv)) { ++nonfinite; continue; } const double dd = fabs(gv - ref_f32[i]); if (dd > maxabs) { maxabs = dd; worst_i = i; } maxrel = std::max(maxrel, dd / std::max(fabs((double) ref_f32[i]), 1e-2)); }
        printf("| %-44s | %7.3f | %5.1f | %5.1f%% | F16 vs ours F32 maxabs=%.3g (row %zu col %zu: gufo %.4g ours %.4g) maxrel=%.3g nonfinite=%zu |\n", "gufo RoutedGatedF16Gemm Q4_K 128-tok tiles", ms, flop / ms / 1e9, floor_ms / ms * 100.0, maxabs, worst_i / M, worst_i % M, h2f(gufo_h[worst_i]), ref_f32[worst_i], maxrel, nonfinite);
        ms = time_ms([&] { if (!G::RoutedGatedF16Gemm(D.wg, D.wu, G::WeightType::kQ4_K, (const __half *) D.xh, D.tiles64, (uint32_t) R.tiles64.size(), 64, D.pad_bounds, D.rows_tok, D.rows_slot, D.gout, M, K, st)) { fprintf(stderr, "gufo 64 failed\n"); exit(1); } }, st);
        printf("| %-44s | %7.3f | %5.1f | %5.1f%% | |\n", "gufo RoutedGatedF16Gemm Q4_K 64-tok tiles", ms, flop / ms / 1e9, floor_ms / ms * 100.0);
    }

    // single-launch balanced kernel (mmb_routed_glu_q4k_kernel): one tile list, atomic work counter, grid sweep
    {
        const int n_max = n_rows / BN + E + 1;
        uint32_t * tiles; int * ctl; CHECK(hipMalloc(&tiles, n_max * 4)); CHECK(hipMalloc(&ctl, 8));
        int ntiles = 0; for (int e = 0; e < E; ++e) ntiles += (R.counts[e] + BN - 1) / BN;
        auto run1 = [&](int grid, bool f32) {
            mmb_build_tiles<<<1, 1024, 0, st>>>(D.bounds, tiles, ctl, E, n_max, BN);
            mmb_routed_glu_q4k_kernel<BN, WT><<<grid, MMB_NT, 0, st>>>(D.wg, D.wu, expert_bytes, D.xb, f32 ? D.d : nullptr, D.dh, f32, D.ids_src, D.ids_dst, D.bounds, tiles, ctl, M, K);
        };
        double tms = time_ms([&] { mmb_build_tiles<<<1, 1024, 0, st>>>(D.bounds, tiles, ctl, E, n_max, BN); }, st);
        printf("| %-44s | %7.3f | | | tiles=%d x %d m-slices |\n", "mmb_build_tiles", tms, ntiles, M / 64);
        CHECK(hipMemset(D.d, 0, (size_t) n_rows * M * 4)); CHECK(hipMemset(D.dh, 0, (size_t) n_rows * M * 2));
        ms = time_ms([&] { run1(80, true); }, st); grab(out_f32, out_bf16); report("ours single-launch (F32+bf16 out) grid 80", ms, &out_f32, &out_bf16);
        for (int grid : {40, 60, 80, 120, 160}) {
            CHECK(hipMemset(D.dh, 0, (size_t) n_rows * M * 2));
            char name[64]; snprintf(name, sizeof name, "ours single-launch bf16 out, grid %d", grid);
            ms = time_ms([&] { run1(grid, false); }, st); grab(out_f32, out_bf16); report(name, ms, nullptr, &out_bf16);
        }
    }

    }
#ifdef MOE_EXP
#include MOE_EXP
#endif
    return 0;
}
