// Selector-v2 GPU scorer bench (optimizations/kvflash/GPU_SCORER.md): the server's module (kvflash_gpu_mass.h) on
// the CUDA backend against the AVX2 CPU scorer (kvflash_exact_fast.h). Synthetic Qwen3.8-27B attention geometry:
// 16 layers, 4 KV heads x 6 q-heads, D 256, q4_0 (server default) or q8_0 K, 64-token chunks, a 16K pool holding
// 254 chunks (sink + recency), the rest on host memory in the pager's layout (test/unit/kvflash_mass_synth.h).
// Times the module's phases and checks scores and picks against the CPU oracle.
//   bench_kvflash_gpu_mass [sealed_chunks 514] [queries 64] [mmq] [reps 5] [q4_0|q8_0]
// The third argument is kept for the old command lines; the f32 / single-K variants were measured and rejected
// (GPU_SCORER.md), only the module's graph ("mmq": two matmuls, concat, bias) remains.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "kvflash_gpu_mass.h"
#include "kvflash_mass_synth.h"
#include "kvflash_prefill_reselect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace luce::common;
using clk = std::chrono::steady_clock;
constexpr int TAIL = 191, PICK_UNITS = 30, UNIT = 2;   // the logged mode-4 window at k=6; 60 chunks in units of 2

static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

int main(int argc, char ** argv) {
    SynthCfg cfg;
    cfg.sealed = argc > 1 ? atoi(argv[1]) : 514;
    cfg.N = argc > 2 ? atoi(argv[2]) : 64;
    const std::string mode = argc > 3 ? argv[3] : "mmq";
    const int reps = argc > 4 ? atoi(argv[4]) : 5;
    cfg.KT = argc > 5 && std::string(argv[5]) == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;   // server default q4_0
    cfg.tail = TAIL;
    if (mode != "mmq") std::fprintf(stderr, "note: mode '%s' was a measured-and-rejected variant; running the module's graph\n", mode.c_str());

    const auto tg0 = clk::now();
    Synth s;
    s.build(cfg);
    const int sealed = cfg.sealed, N = cfg.N, L = cfg.L, HK = cfg.HK, D = cfg.D, CT = cfg.CT, NB = cfg.NB;
    const int E = (int)s.ev.size(), fwc = s.first_window_chunk;
    std::printf("sealed %d (%d resident, %d evicted), N %d, %s K, mmq concat, first window chunk %d, %d planted chunks\n",
                sealed, sealed - E, E, N, ggml_type_name(cfg.KT), fwc, s.n_hot);
    std::printf("synthetic data %.0f ms\n", ms(tg0, clk::now()));

    // CPU oracle: the shipped fast scorer, keys read in place.
    const KvFlashQkDims dims = s.dims();
    KvFlashExactKeys keys;
    keys.n_chunks = sealed; keys.chunk_tokens = CT; keys.type = cfg.KT;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * { return s.blk[(size_t)c] >= 0 ? s.pool_seg(s.blk[(size_t)c], l, h) : s.host_seg(c, l, h); };
    const auto qp = s.qptrs();
    std::vector<std::vector<float>> outs_cpu;
    std::vector<KvFlashMassPlanes> planes_cpu;
    const auto tc0 = clk::now();
    kvflash_exact_mass_scores_fast_multi(keys, dims, qp, outs_cpu, 0, nullptr, &planes_cpu);
    const double cpu_ms = ms(tc0, clk::now());
    const auto sel_cpu = kvflash_select_scores(planes_cpu);
    const auto pick_cpu = kvflash_recall_pick(sel_cpu, fwc, PICK_UNITS, UNIT);

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { std::fprintf(stderr, "no CUDA backend\n"); return 1; }
    // Device pool: L tensors [D, POOL, HK] of the K type, as cache_.attn_k.
    ggml_init_params pp{ggml_tensor_overhead() * (L + 4), nullptr, true};
    ggml_context * pctx = ggml_init(pp);
    std::vector<ggml_tensor *> pool_k;
    for (int l = 0; l < L; l++) pool_k.push_back(ggml_new_tensor_3d(pctx, cfg.KT, D, (int64_t)NB * CT, HK));
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
    for (int l = 0; l < L; l++) ggml_backend_tensor_set(pool_k[(size_t)l], s.pool_img[(size_t)l].data(), 0, ggml_nbytes(pool_k[(size_t)l]));

    KvFlashGpuMass gm(be);
    std::vector<double> t_gather, t_build, t_h2d, t_graph, t_d2h, t_agg, t_total;
    std::vector<std::vector<float>> outs_gpu;
    std::vector<KvFlashMassPlanes> planes_gpu;
    auto host = [&](int c, int l, int h) { return s.host_seg(c, l, h); };
    for (int rep = 0; rep < reps; rep++) {
        KvFlashGpuMassTiming tm;
        std::string why;
        if (!gm.score(pool_k, s.blk, host, CT, dims, qp, outs_gpu, &planes_gpu, &tm, &why)) { std::fprintf(stderr, "gpu scorer failed: %s\n", why.c_str()); return 1; }
        t_gather.push_back(tm.gather); t_build.push_back(tm.build); t_h2d.push_back(tm.h2d); t_graph.push_back(tm.graph);
        t_d2h.push_back(tm.d2h); t_agg.push_back(tm.agg); t_total.push_back(tm.total);
    }
    const auto sel_gpu = kvflash_select_scores(planes_gpu);
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
    for (int x : pick_cpu) hot_cpu += s.hot_of(UNIT * x) >= 0 || s.hot_of(UNIT * x + 1) >= 0;
    for (int x : pick_gpu) hot_gpu += s.hot_of(UNIT * x) >= 0 || s.hot_of(UNIT * x + 1) >= 0;
    std::printf("parity: select max|d| %.3g max rel %.3g; mean-plane max rel %.3g; picks %zu cpu %zu gpu, %d common, %s order; "
                "planted units picked cpu %d gpu %d\n", max_abs, max_rel, mean_rel, pick_cpu.size(), pick_gpu.size(), common,
                pick_cpu == pick_gpu ? "same" : "different", hot_cpu, hot_gpu);
    std::printf("cpu oracle %.1f ms\n", cpu_ms);
    const double stage_mb = (double)L * HK * E * CT * ggml_row_size(cfg.KT, D) / 1e6;
    std::printf("gpu ms (median of %d; first): gather %.1f build %.1f h2d %.1f (%.0f MB) graph %.1f d2h %.1f (%.1f MB) "
                "agg %.1f total %.1f (first %.1f); scratch %.0f MB\n", reps, median(t_gather), median(t_build), median(t_h2d), stage_mb,
                median(t_graph), median(t_d2h), (double)s.PL * (NB + E) * N * 4 / 1e6, median(t_agg), median(t_total),
                t_total.empty() ? 0.0 : t_total[0], (double)gm.scratch_bytes() / 1e6);
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    ggml_backend_free(be);
    return 0;
}
