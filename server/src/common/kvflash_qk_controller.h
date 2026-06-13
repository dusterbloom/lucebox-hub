// KvFlashQkController — model-agnostic glue that hoists the qwen35
// target-QK residency scoring control-flow into a shared class.
//
// BACKGROUND
// ----------
// The pure math (kvflash_qk.h: KvFlashQkPool, KvFlashTargetQkScorer,
// kvflash_qk_chunk_scores) is already model-agnostic.  What was qwen35-
// specific was the ~120 LOC of glue between the pool, the scorer, and the
// pager: pool_to(), maybe_reselect(), the q_cap read-back, the tau cadence,
// and the score_hook wiring.  That glue is identical for every backend;
// only the source of the captured query and the full-attn layer ordinals
// differ.
//
// This header provides KvFlashQkController which owns all the generic glue.
// Each backend satisfies a small INTEGRATION CONTRACT described below.
//
// BASIS INVARIANT
// ---------------
// Q and pooled K MUST live in the same orthogonal basis when
// kvflash_qk_chunk_scores() is called.  They need not be FWHT-rotated; the
// invariant is satisfied by any shared orthogonal transform (including the
// identity — raw post-RoPE — for non-rotating backends).
//
//   qwen35 (Q8_0 / TQ3_0): both K cache and q_cap are captured after
//     the graph's turbo_wht, so they share the FWHT basis.
//
//   generic-llama / non-rotating backends: no rotation applied; K rows
//     and Q are both raw post-RoPE.  The invariant holds trivially.
//
// GQA NOTE
// --------
// The current KvFlashQkDims assumes block-contiguous GQA layout:
//   hk = hq / (n_q_heads / n_kv_heads)
// See "GQA GUARD TODO" below for the interleaved-layout extension point.

#pragma once

#include "kvflash_qk.h"
#include "kvflash_pager.h"

#include <cassert>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

namespace dflash::common {

// ═══════════════════════════════════════════════════════════════════════════
// INTEGRATION CONTRACT — what every backend must provide
// ═══════════════════════════════════════════════════════════════════════════
//
// 1. DIMS at attach time
//    KvFlashQkDims populated from the model's attention geometry:
//      dims.n_layers   = number of full-attention layers (cache_.attn_k.size())
//      dims.n_q_heads  = query heads
//      dims.n_kv_heads = KV heads  (n_q_heads % n_kv_heads == 0 required)
//      dims.head_dim   = per-head dimension
//
// 2. attn_k TENSORS at seal time (on_committed)
//    std::vector<ggml_tensor*> of size n_layers, one tensor per full-attn
//    layer.  Each tensor has layout [head_dim, pool_tokens, n_kv_heads],
//    quantized (Q8_0 / TQ3_0 / F16 / F32).  The controller calls
//    KvFlashQkPool::pool_chunk() which dequantizes on the fly.
//    Freshness contract: the tensors' GPU rows for sealed chunks are
//    resident and stable (within the pool's tail window) at call time.
//
// 3. CAPTURED QUERY (post-RoPE, post-rotation) at reselect time
//    A float buffer of exactly n_layers * n_q_heads * head_dim floats,
//    layout [n_layers, n_q_heads, head_dim] (layer-major, head-minor).
//    Obtained by reading cache.q_cap after the last decode step.
//
//    BACKEND: the graph walk must emit the q_cap capture — see
//    emit_qk_query_capture() below.  For backends whose decode graph does
//    NOT naturally run the full-attn forward (e.g., qwen35moe pipelined
//    CUDA-graph path), run a minimal uncached q-capture forward on the
//    periodic reselect step instead of every decode step.
//
// 4. TOKEN HISTORY at reselect time
//    std::vector<int32_t> of all tokens committed so far (prompt + generated).
//    Used only by KvFlashScorer::score_chunks (the adapter passes it through
//    to n_chunks derivation; the QK scorer ignores token ids).
//
// 5. CHUNK_TOKENS
//    Must equal KvFlashPager::chunk_tokens() (consistent with the pager config).
//
// WHAT STAYS PER-BACKEND (NOT hoisted into this controller)
// ----------------------------------------------------------
//  A. Graph code: emit_qk_query_capture() per full-attn layer (see below).
//  B. Full-attn layer subset walk: the loop `if (is_attn) fa_idx++` and the
//     fa_idx-indexed attn_k / attn_v routing lives in the per-arch graph builder.
//  C. Pager attach / KV cache creation / mask upload / snap serialize.
//  D. Drafter scorer (KvFlashDrafterScorer) lifecycle — orthogonal to QK scorer.
//  E. Pin logic (kvflash_apply_pins) — optional, backend-level policy.

// ═══════════════════════════════════════════════════════════════════════════
// GQA GUARD TODO
// ═══════════════════════════════════════════════════════════════════════════
// KvFlashQkDims SHOULD gain:
//   int  n_q_heads_per_kv_head = n_q_heads / n_kv_heads;  // derived, not stored
//   enum GqaLayout { kBlock = 0, kInterleaved = 1 } gqa_layout = kBlock;
//
// Validity guard (add to controller reset()):
//   assert(dims.n_q_heads % dims.n_kv_heads == 0 &&
//          "n_q_heads must be divisible by n_kv_heads");
//
// In kvflash_qk_chunk_scores, the line:
//   const int hk = hq / group;
// holds for kBlock layout (heads 0..group-1 share kv_head 0, etc.).
// For kInterleaved (Falcon / some custom models):
//   const int hk = hq % n_kv_heads;
// Add a branch there, or a template parameter, when an interleaved-GQA
// backend is adopted.  All current targets (qwen35, qwen35moe, gemma4,
// generic-llama) use block-contiguous GQA so the guard is a no-op today.

// ═══════════════════════════════════════════════════════════════════════════
// Per-backend GRAPH HELPER (not in this class — lives in each arch's
// graph-builder, documented here for discoverability)
// ═══════════════════════════════════════════════════════════════════════════
//
// In the backend's graph-build inner loop, after building each full-attn
// block, emit the q_cap write-back:
//
//   // BACKEND: call once per full-attn layer, indexed by fa_ordinal.
//   // Q_rotated is the post-RoPE (and post-FWHT if applicable) query tensor
//   // from build_full_attn_block(), shape [head_dim, n_tokens, n_q_heads].
//   // cache.q_cap is [head_dim, n_q_heads, n_fa_layers] F32.
//   inline void emit_qk_query_capture(
//       ggml_context * ctx, ggml_cgraph * gf,
//       ggml_tensor * Q_rotated,      // [head_dim, n_tokens, n_q_heads]
//       ggml_tensor * q_cap,           // [head_dim, n_q_heads, n_fa_layers] cache buf
//       int fa_ordinal,                // 0-based index among full-attn layers
//       int n_tokens) {
//     // Extract last-token Q slice: [head_dim, 1, n_q_heads]
//     ggml_tensor * src = ggml_view_3d(ctx, Q_rotated,
//         Q_rotated->ne[0], 1, Q_rotated->ne[2],
//         Q_rotated->nb[1], Q_rotated->nb[2],
//         (size_t)(n_tokens - 1) * Q_rotated->nb[1]);
//     src = ggml_cont(ctx, src);
//     // Write into q_cap plane fa_ordinal
//     ggml_tensor * dst = ggml_view_3d(ctx, q_cap,
//         q_cap->ne[0], 1, q_cap->ne[1],
//         q_cap->nb[1], q_cap->nb[1],
//         (size_t)fa_ordinal * q_cap->nb[2]);
//     ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
//   }
//
// For qwen35moe pipelined-decode CUDA-graph (which omits q_capture):
//   Run a lightweight single-step forward with q_capture=true on the periodic
//   reselect step.  Approximately one decoder-step cost, amortized over tau
//   decode steps.

// ═══════════════════════════════════════════════════════════════════════════
// KvFlashQkController
// ═══════════════════════════════════════════════════════════════════════════

class KvFlashQkController {
public:
    // `pager` must outlive this controller.  The controller does NOT own the pager.
    explicit KvFlashQkController(KvFlashPager * pager)
        : pager_(pager) {}

    // ── Lifecycle ─────────────────────────────────────────────────────────

    // Called once when kvflash QK policy is activated (or on request reset
    // when the pool needs a hard clear).  Validates dims and resets state.
    // Asserts n_q_heads % n_kv_heads == 0 (GQA divisibility guard).
    void reset(const KvFlashQkDims & dims) {
        assert(dims.n_kv_heads > 0 && dims.n_q_heads % dims.n_kv_heads == 0 &&
               "GQA: n_q_heads must be divisible by n_kv_heads");
        dims_ = dims;
        pool_.reset(dims);
        // Re-create the scorer bound to the new pool.
        auto s = std::make_unique<KvFlashTargetQkScorer>(&pool_);
        scorer_raw_ = s.get();
        scorer_ = std::move(s);
        pooled_upto_ = 0;
    }

    // True after reset() has been called with valid dims.
    bool active() const { return dims_.n_layers > 0 && scorer_raw_ != nullptr; }

    // ── Seal-time pooling ─────────────────────────────────────────────────

    // Pool post-RoPE K for all chunks sealed before `committed` (i.e., fully
    // written chunks that will no longer change).  Call once after each
    // prefill step and after each committed verify step.
    //
    // BACKEND CONTRACT: attn_k must be the per-full-attn-layer cache tensors,
    // size == dims_.n_layers, layout [head_dim, pool_tokens, n_kv_heads].
    // At call time the chunks being pooled must be resident (they are within
    // the pager's protected tail window at seal time).
    void on_committed(int committed, const std::vector<ggml_tensor *> & attn_k) {
        if (!active() || !pager_) return;
        const int ct = pager_->chunk_tokens();
        if (ct <= 0) return;
        const int sealed = committed / ct;
        for (int c = pooled_upto_; c < sealed; c++) {
            const int blk = pager_->block_of(c);
            if (blk < 0 || !pool_.pool_chunk(attn_k, blk, ct, c)) {
                std::fprintf(stderr,
                    "[kvflash-qk] pool_chunk failed for chunk %d (block %d); "
                    "chunk scores as missing\n", c, blk);
                // BACKEND: if the backend implements pin-on-missing semantics,
                // a missed pool here means chunk c will score 0 (missing_score)
                // and may be evicted even if it holds relevant context.
                // Consider logging and pinning c if it is the current question.
            }
        }
        pooled_upto_ = std::max(pooled_upto_, sealed);
    }

    // ── Query injection ───────────────────────────────────────────────────

    // Supply the captured post-RoPE query before calling maybe_reselect().
    // `q` must hold dims_.n_layers * dims_.n_q_heads * dims_.head_dim floats,
    // layout [n_layers, n_q_heads, head_dim].
    // Obtained from cache.q_cap after the last decode step completes.
    //
    // Basis invariant: q and the pooled K keys must share the same orthogonal
    // basis.  For FWHT-rotating backends (qwen35 Q8_0/TQ3_0) both are already
    // in the rotated basis.  For non-rotating backends, both are raw post-RoPE.
    void set_query(const float * q, size_t n) {
        if (!scorer_raw_) return;
        // BACKEND: verify n == dims_.n_layers * dims_.n_q_heads * dims_.head_dim
        scorer_raw_->set_query(q, n);
    }

    // ── Periodic reselect ─────────────────────────────────────────────────

    // Tau-cadenced reselect: score every known chunk by QK cosine similarity
    // and rebuild pager residency to the top-pool set.  Wire the score_hook
    // on the pager before calling pager_.reselect().
    //
    // Returns the number of page events (0 = no change, >0 = residency moved).
    // Returns -1 if the scorer is not ready (no query set yet).
    //
    // `generated` is the count of tokens generated so far in this decode
    // session; `tau` is the configured reselect cadence; `history` is the
    // full token history used by the scorer's score_chunks interface.
    //
    // Caller pattern (in the decode loop):
    //   if (controller_.maybe_reselect(generated, tau, history) > 0)
    //       kvflash_upload_mask();  // epoch moved: refresh slot mask
    int maybe_reselect(int generated, int tau,
                       const std::vector<int32_t> & history) {
        if (tau <= 0 || generated % tau != 0) return 0;
        if (!active() || !scorer_raw_->has_query()) return -1;

        // Adaptive tau: cap scoring overhead at ~15% of decode time.
        // The configured tau is the floor; a longer history raises it
        // proportionally (0.6B-drafter tuning: ~history/45 heuristic).
        // BACKEND: backends without a 0.6B drafter may adjust this divisor.
        const int adaptive_tau = std::max<int>(tau, (int)(history.size() / 45));
        if (generated % adaptive_tau != 0) return 0;

        // Score every known chunk.
        if (!scorer_->score_chunks(history, pager_->chunk_tokens(), scores_)) {
            return -1;  // scorer failure — keep LRU this round
        }

        // Wire score_hook: the pager calls this for each candidate chunk
        // during reselect().  Missing chunks (no pooled key) stay at their
        // missing_score (0) which is below any real cosine similarity.
        pager_->score_hook = [this](int c) -> float {
            return c < (int)scores_.size() ? scores_[(size_t)c] : 1e30f;
        };

        const int events = pager_->reselect();
        if (events > 0) {
            std::fprintf(stderr,
                "[kvflash-qk] reselect @gen=%d: %d page events "
                "(resident %d blocks)\n",
                generated, events, pager_->resident_blocks());
        }
        return events;
    }

    // ── Prefix-cache interop ──────────────────────────────────────────────

    // Serialize pooled keys alongside the pager blob for snapshot save.
    // Call in snapshot_save() after pager_.serialize().
    void serialize(std::vector<uint8_t> & out) const {
        pool_.serialize(out);
    }

    // Restore pooled keys from a snapshot blob.  Returns false on
    // geometry mismatch (wrong model or pool dims).  Call in
    // restore_target_cache_from_snapshot() after pager_.deserialize().
    bool deserialize(const std::vector<uint8_t> & in) {
        bool ok = pool_.deserialize(in);
        if (ok) {
            // Advance pooled_upto_ to the restored pool's chunk count so
            // on_committed() does not redundantly re-pool chunks whose keys
            // were restored from the snapshot.
            pooled_upto_ = pool_.n_chunks();
        }
        return ok;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    const KvFlashQkPool &            pool()   const { return pool_; }
    const KvFlashQkDims &            dims()   const { return dims_; }
    KvFlashTargetQkScorer *          scorer() const { return scorer_raw_; }
    int                              pooled_upto() const { return pooled_upto_; }

    // Reset pooled_upto_ to a snapshot-restored position (called by
    // deserialize implicitly; exposed for backends that restore state
    // manually).
    void set_pooled_upto(int upto) { pooled_upto_ = upto; }

private:
    KvFlashPager *                   pager_       = nullptr;
    KvFlashQkDims                    dims_        = {};
    KvFlashQkPool                    pool_;
    std::unique_ptr<KvFlashTargetQkScorer> scorer_;
    KvFlashTargetQkScorer *          scorer_raw_  = nullptr;  // non-owning alias
    std::vector<float>               scores_;
    int                              pooled_upto_ = 0;
};

} // namespace dflash::common
