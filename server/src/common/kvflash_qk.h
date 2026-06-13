// KVFlash target-QK residency scoring (Phase-0-validated math, ported from
// optimizations/msa-sm86/msa_phase0_recall.py).
//
// Score of a sealed 64-token chunk = mean over the full-attention layers of
// the max over all (kv_head, group q-head) pairs of
//   cos( q_post_rope[layer, q_head], mean-pooled K_post_rope[chunk, layer, kv_head] )
// Pooled keys are L2-normalized at pool time; the query is normalized here.
//
// Basis note: when the K cache is FWHT-rotated (kv_k_rotated, default for
// Q8_0), both the pooled keys (read from the cache) and the captured query
// (taken AFTER the graph's turbo_wht) live in the rotated basis. The shared
// orthogonal transform preserves dot products and norms, so cosine here
// equals cosine in the unrotated basis — the Phase-0 domain.
//
// The pure scoring function is dependency-free (testable with synthetic
// data); KvFlashQkPool / KvFlashTargetQkScorer add the cache plumbing.

#pragma once

#include "kvflash_scorer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dflash::common {

struct KvFlashQkDims {
    int n_layers   = 0;   // full-attention layers pooled (16 on qwen35-27B)
    int n_q_heads  = 0;   // query heads (24)
    int n_kv_heads = 0;   // KV heads (4); group size = n_q_heads / n_kv_heads
    int head_dim   = 0;   // 256
};

// Pure scoring: pooled_keys[c] = nullptr for chunks without pooled keys
// (their out[] entry is left at `missing_score`). Non-null entries hold
// n_layers * n_kv_heads * head_dim floats, L2-normalized per (layer, head).
// query: n_layers * n_q_heads * head_dim floats, unnormalized.
inline void kvflash_qk_chunk_scores(
    const std::vector<const float *> & pooled_keys,
    const float * query,
    const KvFlashQkDims & d,
    std::vector<float> & out,
    float missing_score = 0.0f) {
    const int group = d.n_q_heads / d.n_kv_heads;
    const int n_chunks = (int)pooled_keys.size();
    out.assign((size_t)n_chunks, missing_score);

    // Normalize the query once: [n_layers, n_q_heads, head_dim].
    std::vector<float> qn((size_t)d.n_layers * d.n_q_heads * d.head_dim);
    for (int l = 0; l < d.n_layers; l++) {
        for (int h = 0; h < d.n_q_heads; h++) {
            const size_t off = ((size_t)l * d.n_q_heads + h) * d.head_dim;
            double ss = 0.0;
            for (int i = 0; i < d.head_dim; i++) ss += (double)query[off + i] * query[off + i];
            const float inv = 1.0f / ((float)std::sqrt(ss) + 1e-6f);
            for (int i = 0; i < d.head_dim; i++) qn[off + i] = query[off + i] * inv;
        }
    }

    const float inv_layers = 1.0f / (float)d.n_layers;
    for (int c = 0; c < n_chunks; c++) {
        const float * pk = pooled_keys[(size_t)c];
        if (!pk) continue;
        float acc = 0.0f;
        for (int l = 0; l < d.n_layers; l++) {
            float lmax = -2.0f;                       // cosine lower bound
            for (int hq = 0; hq < d.n_q_heads; hq++) {
                const int hk = hq / group;
                const float * q = qn.data() + ((size_t)l * d.n_q_heads + hq) * d.head_dim;
                const float * k = pk + ((size_t)l * d.n_kv_heads + hk) * d.head_dim;
                float dot = 0.0f;
                for (int i = 0; i < d.head_dim; i++) dot += q[i] * k[i];
                if (dot > lmax) lmax = dot;
            }
            acc += lmax;
        }
        out[(size_t)c] = acc * inv_layers;            // layer-MEAN (Phase-0 config)
    }
}

// Pure pin decision for chunk c: pinned if it falls in the header window
// (tool-defs / system prompt) OR a precomputed needle bitmask flags it.
// needle_forced may be empty (needle pinning disabled). Answer-blind:
// keys only on position and the externally-computed salience bitmask.
inline bool kvflash_chunk_should_pin(int c, int pin_header_chunks,
                                     bool pin_needle,
                                     const std::vector<uint8_t> & needle_forced) {
    if (pin_header_chunks > 0 && c < pin_header_chunks) return true;
    if (pin_needle && c >= 0 && c < (int)needle_forced.size() && needle_forced[(size_t)c]) return true;
    return false;
}

} // namespace dflash::common

// ── Cache plumbing (needs ggml) ─────────────────────────────────────────
#ifndef KVFLASH_QK_PURE_ONLY

#include "ggml.h"
#include "ggml-backend.h"

namespace dflash::common {

// Host-side store of pooled, L2-normalized post-RoPE keys per sealed chunk.
// Pool at SEAL time (the chunk is tail-protected, hence resident); entries
// are immutable afterwards (K rows never change once written).
class KvFlashQkPool {
public:
    void reset(const KvFlashQkDims & d) {
        dims_ = d;
        keys_.clear();
    }
    const KvFlashQkDims & dims() const { return dims_; }
    bool has(int c) const {
        return c >= 0 && c < (int)keys_.size() && !keys_[(size_t)c].empty();
    }
    const float * data(int c) const {
        return has(c) ? keys_[(size_t)c].data() : nullptr;
    }
    int n_chunks() const { return (int)keys_.size(); }

    // Pool chunk `c` whose rows sit in pool block `block` of each cache
    // tensor in `attn_k` ([head_dim, pool_tokens, n_head_kv], quantized).
    // Reads chunk_tokens rows per (layer, kv_head), dequantizes, mean-pools,
    // L2-normalizes.
    bool pool_chunk(const std::vector<ggml_tensor *> & attn_k,
                    int block, int chunk_tokens, int c) {
        if ((int)attn_k.size() != dims_.n_layers || block < 0) return false;
        if ((int)keys_.size() <= c) keys_.resize((size_t)c + 1);
        std::vector<float> & dst = keys_[(size_t)c];
        dst.assign((size_t)dims_.n_layers * dims_.n_kv_heads * dims_.head_dim, 0.0f);

        std::vector<uint8_t> raw;
        std::vector<float>   rows((size_t)chunk_tokens * dims_.head_dim);
        for (int l = 0; l < dims_.n_layers; l++) {
            ggml_tensor * t = attn_k[(size_t)l];
            const auto * tt = ggml_get_type_traits(t->type);
            const size_t seg = (size_t)chunk_tokens * t->nb[1];
            raw.resize(seg);
            for (int h = 0; h < dims_.n_kv_heads; h++) {
                const size_t off = (size_t)block * chunk_tokens * t->nb[1]
                                 + (size_t)h * t->nb[2];
                ggml_backend_tensor_get(t, raw.data(), off, seg);
                if (t->type == GGML_TYPE_F32) {
                    std::memcpy(rows.data(), raw.data(), seg);
                } else {
                    tt->to_float(raw.data(), rows.data(),
                                 (int64_t)chunk_tokens * dims_.head_dim);
                }
                float * pk = dst.data()
                           + ((size_t)l * dims_.n_kv_heads + h) * dims_.head_dim;
                for (int tok = 0; tok < chunk_tokens; tok++) {
                    const float * r = rows.data() + (size_t)tok * dims_.head_dim;
                    for (int i = 0; i < dims_.head_dim; i++) pk[i] += r[i];
                }
                double ss = 0.0;
                for (int i = 0; i < dims_.head_dim; i++) ss += (double)pk[i] * pk[i];
                const float inv = 1.0f / ((float)std::sqrt(ss) + 1e-6f);
                bool finite = true;
                for (int i = 0; i < dims_.head_dim; i++) {
                    pk[i] *= inv;
                    if (!std::isfinite(pk[i])) finite = false;
                }
                if (!finite) { dst.clear(); return false; }
            }
        }
        return true;
    }

    // ── Prefix-cache interop ────────────────────────────────────────────
    // Pooled keys are position-independent (mean-pooled, L2-normalized post-
    // RoPE K), so they travel with a prefix snapshot unchanged. Paged-out
    // chunks are not in the cache to re-pool from after a restore, so the
    // qk policy must persist them alongside the pager blob. Drafter / LRU
    // policies need none of this.
    void serialize(std::vector<uint8_t> & out) const {
        const int32_t n = (int32_t)keys_.size();
        const size_t row = (size_t)dims_.n_layers * dims_.n_kv_heads * dims_.head_dim;
        out.clear();
        auto put = [&](const void * p, size_t b) {
            const uint8_t * c = (const uint8_t *)p;
            out.insert(out.end(), c, c + b);
        };
        put(&dims_, sizeof(dims_));
        put(&n, sizeof(n));
        for (int c = 0; c < n; c++) {
            const uint8_t present = keys_[(size_t)c].size() == row ? 1 : 0;
            put(&present, 1);
            if (present) put(keys_[(size_t)c].data(), row * sizeof(float));
        }
    }

    // Restore pooled keys; validates the blob describes this model's dims.
    bool deserialize(const std::vector<uint8_t> & in) {
        const size_t row = (size_t)dims_.n_layers * dims_.n_kv_heads * dims_.head_dim;
        size_t p = 0;
        auto get = [&](void * d, size_t b) -> bool {
            if (p + b > in.size()) return false;
            std::memcpy(d, in.data() + p, b); p += b; return true;
        };
        KvFlashQkDims d{};
        int32_t n = 0;
        if (!get(&d, sizeof(d)) || !get(&n, sizeof(n)) || n < 0) return false;
        if (d.n_layers != dims_.n_layers || d.n_kv_heads != dims_.n_kv_heads ||
            d.n_q_heads != dims_.n_q_heads || d.head_dim != dims_.head_dim) return false;
        keys_.assign((size_t)n, {});
        for (int c = 0; c < n; c++) {
            uint8_t present = 0;
            if (!get(&present, 1)) return false;
            if (present) {
                keys_[(size_t)c].resize(row);
                if (!get(keys_[(size_t)c].data(), row * sizeof(float))) return false;
            }
        }
        return p == in.size();
    }

private:
    KvFlashQkDims dims_;
    std::vector<std::vector<float>> keys_;   // [chunk][L*Hkv*D], empty = missing
};

// KvFlashScorer adapter: scores from the QkPool + the latest captured query
// (post-RoPE, post-rotation, [n_layers, n_q_heads, head_dim]). `ids` is used
// only for the chunk count, keeping the token-only interface intact.
class KvFlashTargetQkScorer : public KvFlashScorer {
public:
    explicit KvFlashTargetQkScorer(const KvFlashQkPool * pool) : pool_(pool) {}

    void set_query(const float * q, size_t n) { query_.assign(q, q + n); }
    bool has_query() const { return !query_.empty(); }

    bool score_chunks(const std::vector<int32_t> & ids, int chunk_tokens,
                      std::vector<float> & out) override {
        out.clear();
        if (!pool_ || chunk_tokens <= 0) return false;
        const KvFlashQkDims & d = pool_->dims();
        if (d.n_layers <= 0) return false;
        if (query_.size() != (size_t)d.n_layers * d.n_q_heads * d.head_dim) return false;
        const int n_chunks = ((int)ids.size() + chunk_tokens - 1) / chunk_tokens;
        std::vector<const float *> pk((size_t)n_chunks, nullptr);
        for (int c = 0; c < n_chunks; c++) pk[(size_t)c] = pool_->data(c);
        kvflash_qk_chunk_scores(pk, query_.data(), d, out);
        return true;
    }

private:
    const KvFlashQkPool * pool_;
    std::vector<float> query_;
};

} // namespace dflash::common

#endif // KVFLASH_QK_PURE_ONLY
