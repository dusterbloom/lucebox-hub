// Selector-v2 GPU scorer derisk (optimizations/kvflash/GPU_SCORER.md). The exact chunk mass of every sealed
// chunk (softmax over all sealed keys, summed per 64-token chunk) for N queries x 384 (layer, q-head) planes, as stock
// ggml ops on the target backend, against the AVX2 CPU scorer (kvflash_exact_fast.h). Synthetic Qwen3.8-27B attention
// geometry: 16 layers, 4 KV heads x 6 q-heads, D 256, q4_0 (server default) or q8_0 K, 64-token chunks, a 16K pool holding 254 chunks (sink +
// recency), the rest on pinned host memory in the pager's layout. ~40 planted chunks of graded strength give the
// ranking a real order. Times gather, uploads, graph build, graph, download, host aggregation; checks scores and picks.
//   bench_kvflash_gpu_mass [sealed_chunks 514] [queries 64] [mmq|f32|single] [reps 5] [q4_0|q8_0]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "kvflash_exact_fast.h"
#include "kvflash_prefill_reselect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace luce::common;
using clk = std::chrono::steady_clock;
constexpr int L = 16, HK = 4, G = 6, HQ = HK * G, D = 256, CT = 64, POOL = 16384, NB = POOL / CT, PL = L * HQ;
constexpr int TAIL = 191, PICK_UNITS = 30, UNIT = 2;   // the logged mode-4 window at k=6; 60 chunks in units of 2

static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

struct Rng {   // xorshift, uniform with unit variance
    uint64_t s;
    float next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (float)((s >> 40) * (1.0 / (1 << 24))) * 3.4641016f - 1.7320508f; }
};

template <class F> static void par(int n, F f) {
    const int nt = std::max(1, std::min<int>(n, (int)std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int w = 0; w < nt; w++) th.emplace_back([&, w] { for (int i = w; i < n; i += nt) f(i); });
    for (auto & t : th) t.join();
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

int main(int argc, char ** argv) {
    const int sealed = argc > 1 ? atoi(argv[1]) : 514;
    const int N = argc > 2 ? atoi(argv[2]) : 64;
    const bool f32 = argc > 3 && std::string(argv[3]) == "f32", single = argc > 3 && std::string(argv[3]) == "single";
    const int reps = argc > 4 ? atoi(argv[4]) : 5;
    const ggml_type KT = argc > 5 && std::string(argv[5]) == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;   // server default q4_0
    const size_t RB = ggml_row_size(KT, D), SEG = (size_t)CT * RB, CHUNK = (size_t)L * HK * 2 * SEG;
    const size_t QS = (size_t)L * HQ * D;

    // Residency as the engine logs it: sink chunks 0-1 plus the newest chunks, 254 of 256 blocks, shuffled placement.
    std::vector<int> blk((size_t)sealed, -1), perm(NB);
    for (int b = 0; b < NB; b++) perm[(size_t)b] = b;
    std::shuffle(perm.begin(), perm.end(), std::mt19937_64(42));
    std::vector<int> res;
    for (int c = 0; c < sealed; c++) if (sealed <= NB - 2 || c < 2 || c >= sealed - (NB - 4)) res.push_back(c);
    for (size_t i = 0; i < res.size(); i++) blk[(size_t)res[i]] = perm[i];
    std::vector<int> ev;   // evicted, in staging order
    for (int c = 0; c < sealed; c++) if (blk[(size_t)c] < 0) ev.push_back(c);
    const int E = (int)ev.size(), fwc = sealed - TAIL;
    const int n_hot = std::max(0, std::min(40, (fwc - 10) / 7));
    std::printf("sealed %d (%d resident, %d evicted), N %d, %s K, %s, first window chunk %d, %d planted chunks\n",
                sealed, (int)res.size(), E, N, ggml_type_name(KT), f32 ? "f32 math" : single ? "single K + mask" : "mmq concat", fwc, n_hot);

    // Planted signal: hot chunk j = 10 + 7j in kv plane (j % L, j % HK), rows 0-7 along v_j; the queries n % 8 == j % 8
    // carry v_j on that group's 6 q-heads, a logit boost of 3 + 5 j / n_hot after the 1/16 scale.
    std::vector<std::vector<float>> v((size_t)n_hot, std::vector<float>(D));
    for (int j = 0; j < n_hot; j++) {
        Rng r{1234567ull + (uint64_t)j};
        double nn = 0;
        for (float & x : v[(size_t)j]) { x = r.next(); nn += x * x; }
        for (float & x : v[(size_t)j]) x /= (float)std::sqrt(nn);
    }
    auto hot_of = [&](int c) { return (c >= 10 && (c - 10) % 7 == 0 && (c - 10) / 7 < n_hot) ? (c - 10) / 7 : -1; };
    auto boost = [&](int j) { return 3.0f + 5.0f * (float)j / (float)std::max(1, n_hot); };

    std::vector<std::vector<uint8_t>> pool_img((size_t)L, std::vector<uint8_t>((size_t)HK * POOL * RB, 0));
    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::fprintf(stderr, "no CUDA backend\n"); return 1; }
    ggml_backend_buffer_t hbuf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), std::max<size_t>(1, (size_t)E * CHUNK));
    ggml_backend_buffer_t sbuf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), std::max<size_t>(1, (size_t)L * HK * E * SEG));
    uint8_t * host = (uint8_t *)ggml_backend_buffer_get_base(hbuf), * stage = (uint8_t *)ggml_backend_buffer_get_base(sbuf);
    std::vector<int> ev_idx((size_t)sealed, -1);
    for (int e = 0; e < E; e++) ev_idx[(size_t)ev[(size_t)e]] = e;
    auto host_seg = [&](int c, int l, int h) { return host + (size_t)ev_idx[(size_t)c] * CHUNK + (size_t)l * HK * 2 * SEG + (size_t)h * SEG; };
    auto pool_seg = [&](int c, int l, int h) { return pool_img[(size_t)l].data() + (size_t)h * POOL * RB + (size_t)blk[(size_t)c] * SEG; };

    const auto tg0 = clk::now();
    par(sealed, [&](int c) {
        std::vector<float> k((size_t)CT * D);
        for (int l = 0; l < L; l++)
            for (int h = 0; h < HK; h++) {
                Rng r{((uint64_t)c * L + l) * HK + h + 77};
                for (float & x : k) x = r.next();
                const int j = hot_of(c);
                if (j >= 0 && l == j % L && h == j % HK)
                    for (int t = 0; t < 8; t++)
                        for (int d = 0; d < D; d++) k[(size_t)t * D + d] += 2.0f * boost(j) * v[(size_t)j][(size_t)d];
                uint8_t * dst = blk[(size_t)c] >= 0 ? pool_seg(c, l, h) : host_seg(c, l, h);
                ggml_quantize_chunk(KT, k.data(), dst, 0, CT, D, nullptr);
            }
    });
    std::vector<float> qs((size_t)N * QS);
    par(N, [&](int n) {
        Rng r{999ull + (uint64_t)n};
        float * q = qs.data() + (size_t)n * QS;
        for (size_t i = 0; i < QS; i++) q[i] = r.next();
        for (int j = 0; j < n_hot; j++) {
            if (n % 8 != j % 8) continue;
            const int l = j % L, h = j % HK;
            for (int g = 0; g < G; g++)
                for (int d = 0; d < D; d++) q[((size_t)l * HQ + h * G + g) * D + d] += 8.0f * v[(size_t)j][(size_t)d];
        }
    });
    std::printf("synthetic data %.0f ms\n", ms(tg0, clk::now()));

    // CPU oracle: the shipped fast scorer, keys read in place.
    KvFlashQkDims dims;
    dims.n_layers = L; dims.n_q_heads = HQ; dims.n_kv_heads = HK; dims.head_dim = D;
    KvFlashExactKeys keys;
    keys.n_chunks = sealed; keys.chunk_tokens = CT; keys.type = KT;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * { return blk[(size_t)c] >= 0 ? pool_seg(c, l, h) : host_seg(c, l, h); };
    std::vector<const float *> qp;
    for (int n = 0; n < N; n++) qp.push_back(qs.data() + (size_t)n * QS);
    std::vector<std::vector<float>> outs_cpu;
    std::vector<KvFlashMassPlanes> planes_cpu;
    const auto tc0 = clk::now();
    kvflash_exact_mass_scores_fast_multi(keys, dims, qp, outs_cpu, 0, nullptr, &planes_cpu);
    const double cpu_ms = ms(tc0, clk::now());
    const auto sel_cpu = kvflash_select_scores(planes_cpu);
    const auto pick_cpu = kvflash_recall_pick(sel_cpu, fwc, PICK_UNITS, UNIT);

    // Device pool: L tensors [D, POOL, HK] of the K type, as cache_.attn_k.
    ggml_init_params pp{ggml_tensor_overhead() * (L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(pp);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, KT, D, POOL, HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    for (int l = 0; l < L; l++) ggml_backend_tensor_set(pool_k[(size_t)l], pool_img[(size_t)l].data(), 0, ggml_nbytes(pool_k[(size_t)l]));

    // Graph blocks: the pool's 256 blocks, then the E staged chunks; chunk_of = -1 masks free blocks.
    const int NBT = NB + E, T = NBT * CT;
    std::vector<int> chunk_of((size_t)NBT, -1);
    for (int c = 0; c < sealed; c++) if (blk[(size_t)c] >= 0) chunk_of[(size_t)blk[(size_t)c]] = c;
    for (int e = 0; e < E; e++) chunk_of[(size_t)(NB + e)] = ev[(size_t)e];
    std::vector<float> bias((size_t)T);
    for (int t = 0; t < T; t++) bias[(size_t)t] = chunk_of[(size_t)(t / CT)] >= 0 ? 0.0f : -INFINITY;

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    std::vector<float> out((size_t)PL * NBT * N);
    std::vector<double> t_gather, t_h2d, t_qup, t_build, t_graph, t_d2h, t_agg, t_total;
    std::vector<float> sel_gpu;
    std::vector<KvFlashMassPlanes> planes_gpu;
    std::vector<std::vector<float>> outs_gpu;
    size_t vram = 0;
    for (int rep = 0; rep < reps; rep++) {
        const auto a0 = clk::now();
        par(L, [&](int l) {
            for (int h = 0; h < HK; h++)
                for (int e = 0; e < E; e++)
                    std::memcpy(stage + (((size_t)l * HK + h) * E + e) * SEG, host_seg(ev[(size_t)e], l, h), SEG);
        });
        const auto a1 = clk::now();
        ggml_init_params ip{ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(1024, false), nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, false);
        ggml_tensor * q_all = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HQ, L, N);
        // single: one K tensor per layer, pool copied into its front (D2D in-graph), evicted rows H2D behind it;
        // the free-block bias rides in as the softmax mask. Otherwise: two matmuls, concat, bias add.
        ggml_tensor * kst = single ? ggml_new_tensor_4d(ctx, KT, D, T, HK, L)
                          : E ? ggml_new_tensor_4d(ctx, KT, D, E * CT, HK, L) : nullptr;
        ggml_tensor * b_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, T);
        ggml_tensor * o_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, PL, NBT, N);
        ggml_set_input(q_all); if (kst) ggml_set_input(kst); ggml_set_input(b_t); ggml_set_output(o_t);
        ggml_tensor * msk = single ? ggml_repeat(ctx, ggml_reshape_2d(ctx, b_t, T, 1), ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, G * N)) : nullptr;
        for (int l = 0; l < L; l++) {
            ggml_tensor * qv = ggml_view_4d(ctx, q_all, D, G, HK, N, q_all->nb[1], G * q_all->nb[1], q_all->nb[3], (size_t)l * q_all->nb[2]);
            ggml_tensor * qr = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, qv, 0, 1, 3, 2)), D, G * N, HK);
            ggml_tensor * lg = nullptr;
            if (single) {
                for (int h = 0; h < HK; h++)   // expanded before the matmul, so the copies run first
                    ggml_build_forward_expand(gf, ggml_cpy(ctx,
                        ggml_view_2d(ctx, pool_k[(size_t)l], D, POOL, pool_k[(size_t)l]->nb[1], (size_t)h * pool_k[(size_t)l]->nb[2]),
                        ggml_view_2d(ctx, kst, D, POOL, kst->nb[1], (size_t)l * kst->nb[3] + (size_t)h * kst->nb[2])));
                lg = ggml_mul_mat(ctx, ggml_view_3d(ctx, kst, D, T, HK, kst->nb[1], kst->nb[2], (size_t)l * kst->nb[3]), qr);
            } else {
                ggml_tensor * kp = pool_k[(size_t)l];
                if (f32) kp = ggml_cast(ctx, kp, GGML_TYPE_F32);
                lg = ggml_mul_mat(ctx, kp, qr);                                            // [POOL, G*N, HK]
                if (kst) {
                    ggml_tensor * ks = ggml_view_3d(ctx, kst, D, E * CT, HK, kst->nb[1], kst->nb[2], (size_t)l * kst->nb[3]);
                    if (f32) ks = ggml_cast(ctx, ks, GGML_TYPE_F32);
                    lg = ggml_concat(ctx, lg, ggml_mul_mat(ctx, ks, qr), 0);               // [T, G*N, HK]
                }
                lg = ggml_add(ctx, lg, b_t);
            }
            ggml_tensor * sm = ggml_soft_max_ext(ctx, lg, msk, 1.0f / std::sqrt((float)D), 0.0f);
            ggml_tensor * m = ggml_sum_rows(ctx, ggml_reshape_4d(ctx, sm, CT, NBT, G * N, HK));   // [1, NBT, G*N, HK]
            ggml_tensor * mp = ggml_permute(ctx, ggml_reshape_4d(ctx, m, NBT, G, N, HK), 2, 0, 3, 1);   // [G, HK, NBT, N]
            ggml_tensor * dst = ggml_view_4d(ctx, o_t, G, HK, NBT, N, G * sizeof(float), o_t->nb[1], o_t->nb[2], (size_t)l * HQ * sizeof(float));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, mp, dst));
        }
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "gallocr failed\n"); return 1; }
        vram = ggml_gallocr_get_buffer_size(galloc, 0);
        const auto a2 = clk::now();
        if (single)
            for (int l = 0; l < L && E; l++)
                for (int h = 0; h < HK; h++)
                    ggml_backend_tensor_set(kst, stage + ((size_t)l * HK + h) * E * SEG,
                                            (size_t)l * kst->nb[3] + (size_t)h * kst->nb[2] + (size_t)POOL * RB, (size_t)E * SEG);
        else if (kst) ggml_backend_tensor_set(kst, stage, 0, ggml_nbytes(kst));
        const auto a3 = clk::now();
        ggml_backend_tensor_set(q_all, qs.data(), 0, ggml_nbytes(q_all));
        ggml_backend_tensor_set(b_t, bias.data(), 0, ggml_nbytes(b_t));
        const auto a4 = clk::now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); return 1; }
        const auto a5 = clk::now();
        ggml_backend_tensor_get(o_t, out.data(), 0, ggml_nbytes(o_t));
        const auto a6 = clk::now();
        planes_gpu.assign((size_t)N, {});
        outs_gpu.assign((size_t)N, std::vector<float>((size_t)sealed, 0.0f));
        par(N, [&](int n) {
            auto & p = planes_gpu[(size_t)n];
            p.top.assign((size_t)sealed, 0.0f);
            p.max.assign((size_t)sealed, 0.0f);
            float tmp[PL];
            for (int gb = 0; gb < NBT; gb++) {
                const int c = chunk_of[(size_t)gb];
                if (c < 0) continue;
                const float * x = out.data() + ((size_t)n * NBT + gb) * PL;
                double s = 0;
                for (int i = 0; i < PL; i++) s += x[i];
                std::copy(x, x + PL, tmp);
                std::nth_element(tmp, tmp + kExactTopPlanes - 1, tmp + PL, std::greater<float>());
                double top = 0;
                for (int i = 0; i < kExactTopPlanes; i++) top += tmp[i];
                outs_gpu[(size_t)n][(size_t)c] = (float)(s / PL);
                p.top[(size_t)c] = (float)(top / kExactTopPlanes);
                p.max[(size_t)c] = *std::max_element(tmp, tmp + kExactTopPlanes);
            }
        });
        sel_gpu = kvflash_select_scores(planes_gpu);
        const auto a7 = clk::now();
        ggml_free(ctx);
        t_gather.push_back(ms(a0, a1)); t_build.push_back(ms(a1, a2)); t_h2d.push_back(ms(a2, a3)); t_qup.push_back(ms(a3, a4));
        t_graph.push_back(ms(a4, a5)); t_d2h.push_back(ms(a5, a6)); t_agg.push_back(ms(a6, a7)); t_total.push_back(ms(a0, a7));
    }
    const auto pick_gpu = kvflash_recall_pick(sel_gpu, fwc, PICK_UNITS, UNIT);

    double max_abs = 0, max_rel = 0, mean_rel = 0;
    for (int c = 0; c < sealed; c++) {
        const double a = sel_cpu[(size_t)c], b = sel_gpu[(size_t)c];
        max_abs = std::max(max_abs, std::fabs(a - b));
        if (a > 1e-9) max_rel = std::max(max_rel, std::fabs(a - b) / a);
    }
    for (int n = 0; n < N; n++)
        for (int c = 0; c < sealed; c++)
            if (outs_cpu[(size_t)n][(size_t)c] > 1e-9)
                mean_rel = std::max(mean_rel, (double)std::fabs(outs_cpu[(size_t)n][(size_t)c] - outs_gpu[(size_t)n][(size_t)c]) / outs_cpu[(size_t)n][(size_t)c]);
    int common = 0;
    for (int x : pick_gpu) common += std::count(pick_cpu.begin(), pick_cpu.end(), x) > 0;
    int hot_cpu = 0, hot_gpu = 0;
    for (int x : pick_cpu) hot_cpu += hot_of(UNIT * x) >= 0 || hot_of(UNIT * x + 1) >= 0;
    for (int x : pick_gpu) hot_gpu += hot_of(UNIT * x) >= 0 || hot_of(UNIT * x + 1) >= 0;
    std::printf("parity: select max|d| %.3g max rel %.3g; mean-plane max rel %.3g; picks %zu cpu %zu gpu, %d common, %s order; "
                "planted units picked cpu %d gpu %d\n", max_abs, max_rel, mean_rel, pick_cpu.size(), pick_gpu.size(), common,
                pick_cpu == pick_gpu ? "same" : "different", hot_cpu, hot_gpu);
    std::printf("cpu oracle %.1f ms\n", cpu_ms);
    std::printf("gpu ms (median of %d; first): gather %.1f build %.1f h2d %.1f (%.0f MB) q+bias %.1f (%.1f MB) graph %.1f d2h %.1f (%.1f MB) "
                "agg %.1f total %.1f (first %.1f); scratch %.0f MB\n", reps, median(t_gather), median(t_build), median(t_h2d),
                (double)L * HK * E * SEG / 1e6, median(t_qup), (double)N * QS * 4 / 1e6, median(t_graph), median(t_d2h),
                (double)out.size() * 4 / 1e6, median(t_agg), median(t_total), t_total.empty() ? 0.0 : t_total[0], (double)vram / 1e6);
    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(pbuf); ggml_backend_buffer_free(hbuf); ggml_backend_buffer_free(sbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
    return 0;
}
