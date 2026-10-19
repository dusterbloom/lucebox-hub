// Exact attention-mass chunk scorer (no GPU): for one query, the softmax mass
// each sealed chunk receives over ALL sealed chunks, averaged over the
// full-attention layers and query heads. Replaces the cosine-on-pooled-keys
// router with the quantity attention actually computes.
//
//   s_t(l,h)    = (q[l][h] . k_t[l][h / group]) * scale     scale = 1/sqrt(head_dim)
//   lse_c(l,h)  = logsumexp of s_t over the tokens of chunk c
//   Z(l,h)      = logsumexp of lse_c over chunks
//   score[c]    = mean over (l,h) of exp(lse_c - Z)
//
// scale matches kq_scale in qwen35_target_graph.cpp. q and K live in the same
// basis (post-RoPE, and the same FWHT rotation when the cache is rotated), so
// the dot products equal the attention graph's. K is read through ggml's type
// traits (to_float): f32/f16/q8_0/q4_0/...

#pragma once

#include "kvflash_qk.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

// seg(c, layer, kv_head) -> the chunk's K rows for that (layer, head):
// chunk_tokens contiguous rows of head_dim elements of `type`; nullptr = absent
// (the chunk then contributes nothing and scores 0).
using KvFlashKSegFn = std::function<const uint8_t *(int c, int l, int h)>;

struct KvFlashExactKeys {
    int           n_chunks     = 0;
    int           chunk_tokens = 0;
    ggml_type     type         = GGML_TYPE_F16;
    KvFlashKSegFn seg;
};

// Thread-summed CPU time per phase (ms), filled when a non-null pointer is passed.
struct KvFlashExactTiming {
    std::atomic<int64_t> dequant_ns{0}, dot_ns{0}, exp_ns{0};
};

// Several queries share one dequantization pass. qs[i]: n_layers*n_q_heads*
// head_dim floats. outs[i][c] = score of chunk c for qs[i].
inline bool kvflash_exact_mass_scores_multi(
    const KvFlashExactKeys & keys, const KvFlashQkDims & d,
    const std::vector<const float *> & qs, std::vector<std::vector<float>> & outs,
    int n_threads = 0, KvFlashExactTiming * tm = nullptr) {
    const int nq = (int)qs.size();
    const int C = keys.n_chunks, ct = keys.chunk_tokens, D = d.head_dim;
    const int group = d.n_kv_heads > 0 ? d.n_q_heads / d.n_kv_heads : 0;
    outs.assign((size_t)nq, std::vector<float>((size_t)std::max(C, 0), 0.0f));
    if (nq == 0 || C <= 0 || ct <= 0 || D <= 0 || group <= 0 || !keys.seg) return nq == 0;
    const auto * tt = ggml_get_type_traits(keys.type);
    if (keys.type != GGML_TYPE_F32 && (!tt || !tt->to_float)) return false;

    const int LH = d.n_layers * d.n_q_heads;
    const float scale = 1.0f / std::sqrt((float)D);
    const double NEG = -INFINITY;
    // lse[(i*C + c)*LH + l*Hq + hq]; absent chunks stay -inf.
    std::vector<double> lse((size_t)nq * C * LH, NEG);

    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, C);
    std::atomic<int> next{0};
    auto worker = [&]() {
        std::vector<float> rows((size_t)ct * D), logit((size_t)ct);
        for (int c; (c = next.fetch_add(1)) < C;) {
            bool absent = false;
            for (int l = 0; l < d.n_layers && !absent; l++) {
                for (int hk = 0; hk < d.n_kv_heads && !absent; hk++) {
                    const uint8_t * p = keys.seg(c, l, hk);
                    if (!p) { absent = true; break; }
                    const float * kr;
                    const auto td0 = std::chrono::steady_clock::now();
                    if (keys.type == GGML_TYPE_F32) kr = (const float *)p;
                    else { tt->to_float(p, rows.data(), (int64_t)ct * D); kr = rows.data(); }
                    if (tm) tm->dequant_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - td0).count();
                    for (int i = 0; i < nq; i++) {
                        for (int g = 0; g < group; g++) {
                            const int hq = hk * group + g;
                            const float * q = qs[(size_t)i] + ((size_t)l * d.n_q_heads + hq) * D;
                            const auto tq0 = std::chrono::steady_clock::now();
                            float mx = -INFINITY;
                            for (int t = 0; t < ct; t++) {
                                const float * k = kr + (size_t)t * D;
                                float dot = 0.0f;
                                for (int j = 0; j < D; j++) dot += q[j] * k[j];
                                logit[(size_t)t] = dot * scale;
                                mx = std::max(mx, logit[(size_t)t]);
                            }
                            const auto tq1 = std::chrono::steady_clock::now();
                            double s = 0.0;
                            for (int t = 0; t < ct; t++) s += std::exp((double)logit[(size_t)t] - mx);
                            if (tm) {
                                const auto tq2 = std::chrono::steady_clock::now();
                                tm->dot_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tq1 - tq0).count();
                                tm->exp_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(tq2 - tq1).count();
                            }
                            lse[((size_t)i * C + c) * LH + (size_t)l * d.n_q_heads + hq] = mx + std::log(s);
                        }
                    }
                }
            }
            if (absent)
                for (int i = 0; i < nq; i++)
                    for (int x = 0; x < LH; x++) lse[((size_t)i * C + c) * LH + x] = NEG;
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < n_threads; t++) th.emplace_back(worker);
    worker();
    for (auto & t : th) t.join();

    for (int i = 0; i < nq; i++) {
        for (int x = 0; x < LH; x++) {
            double mx = NEG;
            for (int c = 0; c < C; c++) mx = std::max(mx, lse[((size_t)i * C + c) * LH + x]);
            if (mx == NEG) continue;
            double Z = 0.0;
            for (int c = 0; c < C; c++) Z += std::exp(lse[((size_t)i * C + c) * LH + x] - mx);
            const double logZ = mx + std::log(Z);
            for (int c = 0; c < C; c++)
                outs[(size_t)i][(size_t)c] += (float)(std::exp(lse[((size_t)i * C + c) * LH + x] - logZ) / LH);
        }
    }
    return true;
}

inline bool kvflash_exact_mass_scores(const KvFlashExactKeys & keys, const KvFlashQkDims & d,
                                      const float * q, std::vector<float> & out,
                                      int n_threads = 0) {
    std::vector<std::vector<float>> o;
    const bool ok = kvflash_exact_mass_scores_multi(keys, d, {q}, o, n_threads);
    out = o.empty() ? std::vector<float>() : std::move(o[0]);
    return ok;
}

}  // namespace luce::common
