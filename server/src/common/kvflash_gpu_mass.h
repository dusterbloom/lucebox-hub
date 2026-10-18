// Selector-v2 exact chunk mass on the target backend (optimizations/kvflash/GPU_SCORER.md). Same quantity and the
// same outputs as kvflash_exact_mass_scores_fast_multi (kvflash_exact_fast.h): per (query, chunk) the mean over the
// (layer, q-head) planes, plus KvFlashMassPlanes{top,max}; the CPU needs ~1.7 s for 64 queries at 514 sealed chunks.
//
// One side graph per call, stock ggml ops only (ggml / ggml-backend / ggml-alloc; no ggml-cuda.h), so it runs on
// CUDA, HIP, Vulkan and, for the unit test, the ggml CPU backend:
//   keys     resident chunks are read in place (the pool tensors [D, pool tokens, Hkv] are graph leaves); evicted
//            chunks are gathered from their host copies into a pinned staging buffer, one 3D tensor [D, E*ct, Hkv]
//            per layer (a plain tensor, so the staging mul_mat can take MMQ like the pool one), uploaded once.
//   per layer  mul_mat(pool K, q) and mul_mat(staging K, q), q as [D, G*N, Hkv] (each kv head serves its G q heads);
//            concat; add a [T] bias (0 on blocks holding a sealed chunk, -inf on every other block: free, or holding
//            a chunk >= sealed as in the first pass of a turn); soft_max_ext at 1/sqrt(D); sum_rows over [ct, blocks].
//   host     one D2H of [planes, blocks, N]; per (query, chunk) mean, top-8 mean (nth_element) and max over planes.
// The ggml context, gallocr, pinned staging / output buffers persist across calls and grow on demand.
//
// WSL silently spills over-committed VRAM into system RAM (slow, no error), so the working set is estimated first
// and the call refuses (returns false) when it exceeds the device's free memory minus a margin.

#pragma once

#include "kvflash_exact_fast.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

struct KvFlashGpuMassTiming {   // ms
    double gather = 0, build = 0, h2d = 0, graph = 0, d2h = 0, agg = 0, total = 0;
};

// Pure: the chunk each graph block holds, -1 = masked. Blocks [0, n_pool) are the pool's; block n_pool + e is the
// e-th evicted chunk (in `evicted`, ascending). block_of[c] = pool block of sealed chunk c, -1 = evicted; pool blocks
// no sealed chunk maps to (free, or chunk >= sealed) stay -1. Empty result when block_of points outside the pool.
inline std::vector<int> kvflash_gpu_mass_chunk_of(const std::vector<int> & block_of, int n_pool, std::vector<int> & evicted) {
    evicted.clear();
    std::vector<int> chunk_of((size_t)std::max(n_pool, 0), -1);
    for (size_t c = 0; c < block_of.size(); c++) {
        const int b = block_of[c];
        if (b >= n_pool) return {};
        if (b >= 0) { if (chunk_of[(size_t)b] >= 0) return {}; chunk_of[(size_t)b] = (int)c; }
        else evicted.push_back((int)c);
    }
    for (int c : evicted) chunk_of.push_back(c);
    return chunk_of;
}

// Pure: device bytes of one call. Logits dominate: gallocr keeps ~2-3 [T, G*N, Hkv] f32 tensors live at once
// (measured 637 MB at 514 sealed, 64 queries, q4_0 against a 202 MB logits tensor); 3x is the conservative bound.
inline size_t kvflash_gpu_mass_bytes(const KvFlashQkDims & d, ggml_type kt, int ct, int n_blocks, int n_evicted, int nq) {
    const size_t group = (size_t)(d.n_q_heads / std::max(d.n_kv_heads, 1)), T = (size_t)(n_blocks + n_evicted) * ct;
    const size_t logits = T * group * nq * d.n_kv_heads * sizeof(float);
    const size_t staging = (size_t)d.n_layers * d.n_kv_heads * n_evicted * ct * ggml_row_size(kt, d.head_dim);
    const size_t q = (size_t)nq * d.n_layers * d.n_q_heads * d.head_dim * sizeof(float);
    const size_t out = (size_t)d.n_layers * d.n_q_heads * (n_blocks + n_evicted) * nq * sizeof(float);
    return 3 * logits + staging + q + out + T * sizeof(float);
}

class KvFlashGpuMass {
public:
    static constexpr size_t kMarginBytes = (size_t)512 << 20;   // VRAM kept free beside the scorer

    explicit KvFlashGpuMass(ggml_backend_t be) : be_(be) {}
    KvFlashGpuMass(const KvFlashGpuMass &) = delete;
    KvFlashGpuMass & operator=(const KvFlashGpuMass &) = delete;
    ~KvFlashGpuMass() {
        if (galloc_) ggml_gallocr_free(galloc_);
        if (stage_) ggml_backend_buffer_free(stage_);
        if (out_) ggml_backend_buffer_free(out_);
    }

    // pool_k[l]: [D, n_blocks * ct, Hkv] of the K type (cache_.attn_k). block_of[c] for c < sealed = block_of.size().
    // host_seg(c, l, h): the evicted chunk's ct rows for (layer, kv head), as KvFlashKSegFn. qs[i]: L*Hq*D floats.
    // Fills outs / planes like kvflash_exact_mass_scores_fast_multi. false (+ why): outs / planes are not valid.
    bool score(const std::vector<ggml_tensor *> & pool_k, const std::vector<int> & block_of, const KvFlashKSegFn & host_seg,
               int ct, const KvFlashQkDims & d, const std::vector<const float *> & qs,
               std::vector<std::vector<float>> & outs, std::vector<KvFlashMassPlanes> * planes,
               KvFlashGpuMassTiming * timing = nullptr, std::string * why = nullptr) {
        auto fail = [&](const std::string & m) { if (why) *why = m; return false; };
        using clk = std::chrono::steady_clock;
        auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        KvFlashGpuMassTiming tm;
        const auto t0 = clk::now();
        const int L = d.n_layers, HK = d.n_kv_heads, HQ = d.n_q_heads, D = d.head_dim, N = (int)qs.size(), sealed = (int)block_of.size();
        if (!be_) return fail("no backend");
        if (L <= 0 || HK <= 0 || HQ <= 0 || D <= 0 || HQ % HK || ct <= 0) return fail("bad dims");
        const int G = HQ / HK, PL = L * HQ;
        if (sealed <= 0) return fail("no sealed chunks");
        if ((int)pool_k.size() != L) return fail("pool tensor count != layers");
        for (ggml_tensor * t : pool_k)
            if (!t || t->ne[0] != D || t->ne[2] != HK || t->ne[1] % ct || t->type != pool_k[0]->type || t->ne[1] != pool_k[0]->ne[1])
                return fail("unexpected pool K tensor shape or type");
        for (const float * q : qs) if (!q) return fail("null query");
        const ggml_type kt = pool_k[0]->type;
        const int NB = (int)(pool_k[0]->ne[1] / ct);
        if (N == 0) { outs.clear(); if (planes) planes->clear(); return true; }
        std::vector<int> ev;
        const std::vector<int> chunk_of = kvflash_gpu_mass_chunk_of(block_of, NB, ev);
        if (chunk_of.empty()) return fail("block_of points outside the pool or maps two chunks to one block");
        const int E = (int)ev.size(), NBT = NB + E, T = NBT * ct;
        const size_t RB = ggml_row_size(kt, D), SEG = (size_t)ct * RB, QS = (size_t)L * HQ * D;
        const size_t stage_k = (size_t)L * HK * E * SEG, stage_total = stage_k + (size_t)N * QS * sizeof(float);
        const size_t out_bytes = (size_t)PL * NBT * N * sizeof(float);

        // Free-memory guard (a held gallocr buffer is already out of the device's free count).
        ggml_backend_dev_t dev = ggml_backend_get_device(be_);
        if (dev) {
            size_t fr = 0, tot = 0;
            ggml_backend_dev_memory(dev, &fr, &tot);
            const size_t need = kvflash_gpu_mass_bytes(d, kt, ct, NB, E, N);
            const size_t held = galloc_ ? ggml_gallocr_get_buffer_size(galloc_, 0) : 0;
            if (tot > 0 && need > held && need - held + kMarginBytes > fr)
                return fail("needs " + std::to_string(need >> 20) + " MiB (+" + std::to_string(kMarginBytes >> 20) + " MiB margin) but " +
                            std::to_string(fr >> 20) + " MiB are free");
        }
        if (!host_buft_) {
            host_buft_ = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            if (!host_buft_) host_buft_ = ggml_backend_get_default_buffer_type(be_);
            if (!host_buft_ || !ggml_backend_buft_is_host(host_buft_)) { host_buft_ = nullptr; return fail("no host-accessible buffer type"); }
        }
        if (!grow(stage_, stage_total)) return fail("pinned staging alloc failed");
        if (!grow(out_, out_bytes)) return fail("pinned output alloc failed");
        outs.assign((size_t)N, std::vector<float>((size_t)sealed, 0.0f));
        if (planes) planes->assign((size_t)N, KvFlashMassPlanes{std::vector<float>((size_t)sealed, 0.0f), std::vector<float>((size_t)sealed, 0.0f)});
        uint8_t * stage = (uint8_t *)ggml_backend_buffer_get_base(stage_);
        float * qbuf = (float *)(stage + stage_k);
        float * out = (float *)ggml_backend_buffer_get_base(out_);

        // Gather: evicted K into the staging layout [layer][kv head][e][row], and the queries behind it.
        std::atomic<bool> absent{false};
        par_(L, [&](int l) {
            for (int h = 0; h < HK; h++)
                for (int e = 0; e < E; e++) {
                    const uint8_t * s = host_seg(ev[(size_t)e], l, h);
                    if (!s) { absent = true; continue; }
                    std::memcpy(stage + (((size_t)l * HK + h) * E + e) * SEG, s, SEG);
                }
        });
        if (absent) return fail("missing host K segment");
        for (int n = 0; n < N; n++) std::memcpy(qbuf + (size_t)n * QS, qs[(size_t)n], QS * sizeof(float));
        std::vector<float> bias((size_t)T);
        for (int t = 0; t < T; t++) bias[(size_t)t] = chunk_of[(size_t)(t / ct)] >= 0 ? 0.0f : -INFINITY;
        const auto t1 = clk::now();

        // Graph.
        const int max_nodes = 32 * L + 64;
        ggml_init_params ip{ggml_tensor_overhead() * (size_t)(40 * L + 64) + ggml_graph_overhead_custom(max_nodes, false), nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return fail("ggml_init failed");
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);
        ggml_tensor * q_all = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HQ, L, N);
        ggml_tensor * b_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, T);
        ggml_tensor * o_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, PL, NBT, N);
        ggml_set_input(q_all); ggml_set_input(b_t); ggml_set_output(o_t);
        std::vector<ggml_tensor *> kst((size_t)L, nullptr);
        for (int l = 0; l < L; l++) {
            ggml_tensor * qv = ggml_view_4d(ctx, q_all, D, G, HK, N, q_all->nb[1], G * q_all->nb[1], q_all->nb[3], (size_t)l * q_all->nb[2]);
            ggml_tensor * qr = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, qv, 0, 1, 3, 2)), D, G * N, HK);
            ggml_tensor * lg = ggml_mul_mat(ctx, pool_k[(size_t)l], qr);                       // [NB*ct, G*N, HK]
            if (E) {
                kst[(size_t)l] = ggml_new_tensor_3d(ctx, kt, D, (int64_t)E * ct, HK);
                ggml_set_input(kst[(size_t)l]);
                lg = ggml_concat(ctx, lg, ggml_mul_mat(ctx, kst[(size_t)l], qr), 0);           // [T, G*N, HK]
            }
            lg = ggml_add(ctx, lg, b_t);
            ggml_tensor * sm = ggml_soft_max_ext(ctx, lg, nullptr, 1.0f / std::sqrt((float)D), 0.0f);
            ggml_tensor * m = ggml_sum_rows(ctx, ggml_reshape_4d(ctx, sm, ct, NBT, G * N, HK));   // [1, NBT, G*N, HK]
            ggml_tensor * mp = ggml_permute(ctx, ggml_reshape_4d(ctx, m, NBT, G, N, HK), 2, 0, 3, 1);   // [G, HK, NBT, N]
            ggml_tensor * dst = ggml_view_4d(ctx, o_t, G, HK, NBT, N, G * sizeof(float), o_t->nb[1], o_t->nb[2], (size_t)l * HQ * sizeof(float));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, mp, dst));
        }
        if (!galloc_) galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be_));
        if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf)) { ggml_free(ctx); return fail("graph allocation failed"); }
        const auto t2 = clk::now();

        for (int l = 0; l < L && E; l++)
            ggml_backend_tensor_set_async(be_, kst[(size_t)l], stage + (size_t)l * HK * E * SEG, 0, ggml_nbytes(kst[(size_t)l]));
        ggml_backend_tensor_set_async(be_, q_all, qbuf, 0, ggml_nbytes(q_all));
        ggml_backend_tensor_set_async(be_, b_t, bias.data(), 0, ggml_nbytes(b_t));
        ggml_backend_synchronize(be_);
        const auto t3 = clk::now();
        const bool ok = ggml_backend_graph_compute(be_, gf) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(be_);
        const auto t4 = clk::now();
        if (ok) { ggml_backend_tensor_get_async(be_, o_t, out, 0, ggml_nbytes(o_t)); ggml_backend_synchronize(be_); }
        const auto t5 = clk::now();
        ggml_free(ctx);
        if (!ok) return fail("graph compute failed");

        // Aggregate over the planes, per query in parallel.
        constexpr int K = kExactTopPlanes;
        const int Kp = std::min(K, PL);
        par_(N, [&](int n) {
            std::vector<float> tmp((size_t)PL);
            for (int gb = 0; gb < NBT; gb++) {
                const int c = chunk_of[(size_t)gb];
                if (c < 0) continue;
                const float * x = out + ((size_t)n * NBT + gb) * PL;
                double s = 0;
                for (int i = 0; i < PL; i++) s += x[i];
                outs[(size_t)n][(size_t)c] = (float)(s / PL);
                if (!planes) continue;
                std::copy(x, x + PL, tmp.begin());
                std::nth_element(tmp.begin(), tmp.begin() + (Kp - 1), tmp.end(), std::greater<float>());
                double top = 0;
                for (int i = 0; i < Kp; i++) top += tmp[(size_t)i];
                (*planes)[(size_t)n].top[(size_t)c] = (float)(top / Kp);
                (*planes)[(size_t)n].max[(size_t)c] = *std::max_element(tmp.begin(), tmp.begin() + Kp);
            }
        });
        const auto t6 = clk::now();
        tm.gather = ms(t0, t1); tm.build = ms(t1, t2); tm.h2d = ms(t2, t3); tm.graph = ms(t3, t4); tm.d2h = ms(t4, t5);
        tm.agg = ms(t5, t6); tm.total = ms(t0, t6);
        if (timing) *timing = tm;
        return true;
    }

    size_t scratch_bytes() const { return galloc_ ? ggml_gallocr_get_buffer_size(galloc_, 0) : 0; }

private:
    template <class F> static void par_(int n, F f) {
        const int nt = std::max(1, std::min<int>(n, (int)std::thread::hardware_concurrency()));
        std::vector<std::thread> th;
        for (int w = 1; w < nt; w++) th.emplace_back([&, w] { for (int i = w; i < n; i += nt) f(i); });
        for (int i = 0; i < n; i += nt) f(i);
        for (auto & t : th) t.join();
    }
    bool grow(ggml_backend_buffer_t & buf, size_t bytes) {
        if (buf && ggml_backend_buffer_get_size(buf) >= bytes) return true;
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        buf = ggml_backend_buft_alloc_buffer(host_buft_, std::max<size_t>(bytes, 1));
        return buf && ggml_backend_buffer_get_base(buf);
    }

    ggml_backend_t            be_;
    ggml_backend_buffer_type_t host_buft_ = nullptr;
    ggml_gallocr_t            galloc_ = nullptr;
    ggml_backend_buffer_t     stage_ = nullptr;   // pinned: evicted K staging, then the queries
    ggml_backend_buffer_t     out_ = nullptr;     // pinned: [planes, blocks, N] masses
};

}  // namespace luce::common
