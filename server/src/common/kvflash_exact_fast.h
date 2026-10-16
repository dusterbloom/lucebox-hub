// Fast exact attention-mass chunk scorer: same quantity as
// kvflash_exact_mass_scores (kvflash_exact_mass.h), computed with a fused
// q4_0 dequant + AVX2/FMA dot (one dequantized key row feeds every q head of
// its kv group), a vectorized float exp for the in-chunk logsumexp, and the
// same double-precision cross-chunk softmax. Falls back to scalar code when
// AVX2 is unavailable. Keys of other types go through ggml's to_float.
//
// Differences from the reference are float rounding only (dot order, expf):
// per-chunk scores agree to ~1e-6 relative; see test_kvflash_exact_mass.cpp.

#pragma once

#include "kvflash_exact_mass.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KVFLASH_FAST_X86 1
#endif

namespace luce::common {
namespace fastmass {

constexpr int kQ4Block = 32;
constexpr int kMaxGroup = 6;

// Row dequant, bit-identical to ggml's dequantize_row_q4_0: (nib - 8) * d.
inline void dequant_row_q4_0_scalar(const uint8_t * p, float * out, int D) {
    for (int b = 0; b < D / kQ4Block; b++, p += 18) {
        uint16_t h; std::memcpy(&h, p, 2);
        const float d = ggml_fp16_to_fp32((ggml_fp16_t)h);
        for (int j = 0; j < 16; j++) {
            out[b * kQ4Block + j]      = (float)((int)(p[2 + j] & 0xF) - 8) * d;
            out[b * kQ4Block + j + 16] = (float)((int)(p[2 + j] >> 4) - 8) * d;
        }
    }
}

// max_g = number of q heads sharing the row; logits[g][t] = dot(q_g, row_t).
inline void dots_scalar(const float * row, const float * const * qg, int G, int D, float * out /*[G]*/) {
    for (int g = 0; g < G; g++) {
        float s = 0.0f;
        for (int j = 0; j < D; j++) s += qg[g][j] * row[j];
        out[g] = s;
    }
}

// lse = mx + log(sum exp(x - mx)); returns the float sum, x already minus nothing.
inline double lse_scalar(const float * x, int n) {
    float mx = -INFINITY;
    for (int t = 0; t < n; t++) mx = std::max(mx, x[t]);
    double s = 0.0;
    for (int t = 0; t < n; t++) s += std::exp((double)x[t] - mx);
    return mx + std::log(s);
}

#ifdef KVFLASH_FAST_X86
#define KVFLASH_AVX2 __attribute__((target("avx2,fma,f16c")))

KVFLASH_AVX2 inline __m256 nib8_to_f32(__m128i v, __m256 eight, __m256 d) {
    return _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)), eight), d);
}

KVFLASH_AVX2 inline void dequant_row_q4_0_avx2(const uint8_t * p, float * out, int D) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m256 eight = _mm256_set1_ps(8.0f);
    for (int b = 0; b < D / kQ4Block; b++, p += 18, out += kQ4Block) {
        uint16_t h; std::memcpy(&h, p, 2);
        const __m256 d = _mm256_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))));
        const __m128i x  = _mm_loadu_si128((const __m128i *)(p + 2));
        const __m128i lo = _mm_and_si128(x, m4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(x, 4), m4);
        _mm256_storeu_ps(out,      nib8_to_f32(lo, eight, d));
        _mm256_storeu_ps(out + 8,  nib8_to_f32(_mm_srli_si128(lo, 8), eight, d));
        _mm256_storeu_ps(out + 16, nib8_to_f32(hi, eight, d));
        _mm256_storeu_ps(out + 24, nib8_to_f32(_mm_srli_si128(hi, 8), eight, d));
    }
}

KVFLASH_AVX2 inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// G q heads against one dequantized row (D % 8 == 0).
template <int G>
KVFLASH_AVX2 inline void dots_avx2(const float * row, const float * const * qg, int D, float * out) {
    __m256 acc[G];
    for (int g = 0; g < G; g++) acc[g] = _mm256_setzero_ps();
    for (int j = 0; j < D; j += 8) {
        const __m256 r = _mm256_loadu_ps(row + j);
        for (int g = 0; g < G; g++) acc[g] = _mm256_fmadd_ps(r, _mm256_loadu_ps(qg[g] + j), acc[g]);
    }
    for (int g = 0; g < G; g++) out[g] = hsum256(acc[g]);
}

// Two rows x G heads: each q load feeds two FMAs (register blocking, 2G accumulators).
template <int G>
KVFLASH_AVX2 inline void dots2_avx2(const float * row0, const float * row1, const float * const * qg, int D,
                                    float * out0, float * out1) {
    __m256 a0[G], a1[G];
    for (int g = 0; g < G; g++) a0[g] = a1[g] = _mm256_setzero_ps();
    for (int j = 0; j < D; j += 8) {
        const __m256 r0 = _mm256_loadu_ps(row0 + j), r1 = _mm256_loadu_ps(row1 + j);
        for (int g = 0; g < G; g++) {
            const __m256 qv = _mm256_loadu_ps(qg[g] + j);
            a0[g] = _mm256_fmadd_ps(r0, qv, a0[g]);
            a1[g] = _mm256_fmadd_ps(r1, qv, a1[g]);
        }
    }
    for (int g = 0; g < G; g++) { out0[g] = hsum256(a0[g]); out1[g] = hsum256(a1[g]); }
}

// exp(x) for x <= 0 (clamped at -80), relative error ~2e-7.
KVFLASH_AVX2 inline __m256 exp256_nonpos(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-80.0f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504089f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

KVFLASH_AVX2 inline double lse_avx2(const float * x, int n) {
    float mx = -INFINITY;
    for (int t = 0; t < n; t++) mx = std::max(mx, x[t]);
    const __m256 vm = _mm256_set1_ps(mx);
    __m256 acc = _mm256_setzero_ps();
    int t = 0;
    for (; t + 8 <= n; t += 8) acc = _mm256_add_ps(acc, exp256_nonpos(_mm256_sub_ps(_mm256_loadu_ps(x + t), vm)));
    double s = hsum256(acc);
    for (; t < n; t++) s += std::exp((double)x[t] - mx);
    return mx + std::log(s);
}

inline bool & force_scalar() { static bool f = false; return f; }   // tests: exercise the portable path
inline bool have_avx2() {
    if (force_scalar()) return false;
    static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
    return ok;
}

// n (1..kMaxGroup) heads against row0, and against row1 too when non-null.
inline void dot_tile_avx2(const float * row0, const float * row1, const float * const * qg, int n, int D,
                          float * out0, float * out1) {
#define KVFLASH_CASE(N) case N: if (row1) dots2_avx2<N>(row0, row1, qg, D, out0, out1); else dots_avx2<N>(row0, qg, D, out0); break;
    switch (n) { KVFLASH_CASE(1) KVFLASH_CASE(2) KVFLASH_CASE(3) KVFLASH_CASE(4) KVFLASH_CASE(5) default: KVFLASH_CASE(6) }
#undef KVFLASH_CASE
}
#endif  // KVFLASH_FAST_X86

}  // namespace fastmass

// Per-chunk scores for one query; same contract as kvflash_exact_mass_scores.
inline bool kvflash_exact_mass_scores_fast(
    const KvFlashExactKeys & keys, const KvFlashQkDims & d, const float * q,
    std::vector<float> & out, int n_threads = 0, KvFlashExactTiming * tm = nullptr) {
    const int C = keys.n_chunks, ct = keys.chunk_tokens, D = d.head_dim;
    const int group = d.n_kv_heads > 0 ? d.n_q_heads / d.n_kv_heads : 0;
    out.assign((size_t)std::max(C, 0), 0.0f);
    if (C <= 0 || ct <= 0 || D <= 0 || group <= 0 || !keys.seg || !q) return false;
    const auto * tt = ggml_get_type_traits(keys.type);
    if (keys.type != GGML_TYPE_F32 && (!tt || !tt->to_float)) return false;
    const bool q4 = keys.type == GGML_TYPE_Q4_0 && D % fastmass::kQ4Block == 0;
#ifdef KVFLASH_FAST_X86
    const bool avx = fastmass::have_avx2() && D % 8 == 0;
#else
    const bool avx = false;
#endif
    const int LH = d.n_layers * d.n_q_heads;
    const float scale = 1.0f / std::sqrt((float)D);
    const double NEG = -INFINITY;
    std::vector<double> lse((size_t)C * LH, NEG);   // [c][l*Hq + hq]; absent chunks stay -inf

    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, C);
    std::atomic<int> next{0};
    auto worker = [&]() {
        std::vector<float> row((size_t)D), row2((size_t)D), logit((size_t)group * ct);
        int64_t t_deq = 0, t_dot = 0, t_exp = 0;
        using clk = std::chrono::steady_clock;
        auto ns = [](clk::time_point a, clk::time_point b) { return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count(); };
        for (int c; (c = next.fetch_add(1)) < C;) {
            bool absent = false;
            for (int l = 0; l < d.n_layers && !absent; l++) {
                for (int hk = 0; hk < d.n_kv_heads && !absent; hk++) {
                    const uint8_t * p = keys.seg(c, l, hk);
                    if (!p) { absent = true; break; }
                    const float * qg[64];
                    const int G = std::min(group, 64);
                    for (int g = 0; g < G; g++) qg[g] = q + ((size_t)l * d.n_q_heads + hk * group + g) * D;
                    const size_t row_bytes = q4 ? (size_t)(D / 32) * 18 : (keys.type == GGML_TYPE_F32 ? (size_t)D * 4 : (size_t)ggml_row_size(keys.type, D));
                    auto load_row = [&](int t, float * dst) -> const float * {
                        const uint8_t * src = p + (size_t)t * row_bytes;
                        if (keys.type == GGML_TYPE_F32) return (const float *)src;
                        if (q4) {
#ifdef KVFLASH_FAST_X86
                            if (avx) fastmass::dequant_row_q4_0_avx2(src, dst, D); else
#endif
                            fastmass::dequant_row_q4_0_scalar(src, dst, D);
                        } else tt->to_float(src, dst, D);
                        return dst;
                    };
                    for (int t = 0; t < ct;) {
                        const bool pair = avx && t + 1 < ct;
                        const auto a = tm ? clk::now() : clk::time_point();
                        const float * kr = load_row(t, row.data());
                        const float * kr1 = pair ? load_row(t + 1, row2.data()) : nullptr;
                        const auto b = tm ? clk::now() : clk::time_point();
                        for (int g0 = 0; g0 < G; g0 += fastmass::kMaxGroup) {
                            const int n = std::min(fastmass::kMaxGroup, G - g0);
                            float r[fastmass::kMaxGroup], r1[fastmass::kMaxGroup];
#ifdef KVFLASH_FAST_X86
                            if (avx) fastmass::dot_tile_avx2(kr, kr1, qg + g0, n, D, r, r1); else
#endif
                            fastmass::dots_scalar(kr, qg + g0, n, D, r);
                            for (int g = 0; g < n; g++) {
                                logit[(size_t)(g0 + g) * ct + t] = r[g] * scale;
                                if (pair) logit[(size_t)(g0 + g) * ct + t + 1] = r1[g] * scale;
                            }
                        }
                        if (tm) { const auto e = clk::now(); t_deq += ns(a, b); t_dot += ns(b, e); }
                        t += pair ? 2 : 1;
                    }
                    const auto e0 = tm ? clk::now() : clk::time_point();
                    for (int g = 0; g < G; g++) {
                        const float * x = &logit[(size_t)g * ct];
#ifdef KVFLASH_FAST_X86
                        lse[(size_t)c * LH + (size_t)l * d.n_q_heads + hk * group + g] = avx ? fastmass::lse_avx2(x, ct) : fastmass::lse_scalar(x, ct);
#else
                        lse[(size_t)c * LH + (size_t)l * d.n_q_heads + hk * group + g] = fastmass::lse_scalar(x, ct);
#endif
                    }
                    if (tm) t_exp += ns(e0, clk::now());
                }
            }
            if (absent) for (int x = 0; x < LH; x++) lse[(size_t)c * LH + x] = NEG;
        }
        if (tm) { tm->dequant_ns += t_deq; tm->dot_ns += t_dot; tm->exp_ns += t_exp; }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < n_threads; t++) th.emplace_back(worker);
    worker();
    for (auto & t : th) t.join();

    // Cross-chunk softmax per (layer, head), double precision, as the reference.
    for (int x = 0; x < LH; x++) {
        double mx = NEG;
        for (int c = 0; c < C; c++) mx = std::max(mx, lse[(size_t)c * LH + x]);
        if (mx == NEG) continue;
        double Z = 0.0;
        for (int c = 0; c < C; c++) Z += std::exp(lse[(size_t)c * LH + x] - mx);
        const double logZ = mx + std::log(Z);
        for (int c = 0; c < C; c++) out[(size_t)c] += (float)(std::exp(lse[(size_t)c * LH + x] - logZ) / LH);
    }
    return true;
}

// Sum of chunk scores over each block of `cpb` chunks; block b's score is at [b].
inline std::vector<double> kvflash_block_scores(const std::vector<float> & chunk_scores, int cpb) {
    std::vector<double> bs;
    if (cpb <= 0) return bs;
    for (size_t b = 0; (b + 1) * cpb <= chunk_scores.size(); b++) {
        double s = 0.0;
        for (size_t c = b * cpb; c < (b + 1) * cpb; c++) s += chunk_scores[c];
        bs.push_back(s);
    }
    return bs;
}

}  // namespace luce::common
