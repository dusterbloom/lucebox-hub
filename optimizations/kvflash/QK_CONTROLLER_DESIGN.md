# KvFlashQkController — Generalization Design

## 1. Problem

The target-QK residency scorer (Phase-0 validated: cosine similarity of
pooled post-RoPE keys vs. the decode query) lives entirely in
`kvflash_qk.h` / `kvflash_pager.h` as pure, model-agnostic math.
However, the ~120 LOC of control-flow glue that wires the pool, scorer,
and pager together is currently embedded in `Qwen35Backend`, blocking
every other backend from adopting QK-scored residency without copy-pasting
the same logic.

Affected LOC in `qwen35_backend.cpp`:
- `kvflash_qk_pool_to()` — seal-time pooling (15 LOC)
- `kvflash_maybe_reselect()` — tau cadence + score_hook wiring + reselect (35 LOC)
- `set_query()` read-back from `cache_.q_cap` inside reselect (8 LOC)
- Init, reset, serialize/deserialize passthroughs (25 LOC)
- State variables (`kvflash_qk_pool_`, `kvflash_qk_scorer_`, `kvflash_qk_pooled_upto_`) scattered across the class

The fix: hoist all of the above into a single header-only
`KvFlashQkController` in `server/src/common/`.

## 2. KvFlashQkController API

File: `server/src/common/kvflash_qk_controller.h`

```cpp
namespace dflash::common {

class KvFlashQkController {
public:
    explicit KvFlashQkController(KvFlashPager * pager);

    // Validate dims, reset pool, create scorer.
    void reset(const KvFlashQkDims & dims);

    bool active() const;

    // Seal-time: pool K for chunks sealed before `committed`.
    // attn_k: per-full-attn-layer cache tensors [head_dim, pool_tokens, n_kv_heads].
    void on_committed(int committed, const std::vector<ggml_tensor*> & attn_k);

    // Inject query before reselect.
    // q: n_layers * n_q_heads * head_dim floats, [n_layers, n_q_heads, head_dim].
    void set_query(const float * q, size_t n);

    // Tau-cadenced: score all chunks, wire score_hook, call pager_.reselect().
    // Returns page events, -1 if scorer not ready, 0 if not due yet.
    int maybe_reselect(int generated, int tau,
                       const std::vector<int32_t> & history);

    // Prefix-cache: passthrough to pool serialize/deserialize.
    void serialize(std::vector<uint8_t> & out) const;
    bool deserialize(const std::vector<uint8_t> & in);

    // Accessors
    const KvFlashQkPool &   pool()        const;
    const KvFlashQkDims &   dims()        const;
    KvFlashTargetQkScorer * scorer()      const;
    int                     pooled_upto() const;
    void                    set_pooled_upto(int upto);
};

} // namespace dflash::common
```

Key design decisions:
- Header-only (no new .cpp compilation unit); all methods have inline bodies.
- Does NOT own the pager (backends own the pager lifecycle).
- Does NOT own the drafter scorer (`KvFlashDrafterScorer`) — that stays
  per-backend because drafter load/unload (park/unpark, residency release)
  is backend-specific policy.
- `maybe_reselect` encapsulates the adaptive-tau heuristic (floor = tau,
  adaptive ceiling = history.size()/45) that was previously inline in
  `Qwen35Backend::kvflash_maybe_reselect`.

## 3. Per-Backend Integration Contract

Each backend wiring `KvFlashQkController` must provide:

| # | What | Where |
|---|------|-------|
| 1 | `KvFlashQkDims` from model geometry | `init()` / attach time |
| 2 | `attn_k` tensor vector per decode/prefill step | passed to `on_committed()` |
| 3 | Post-RoPE, post-rotation query float buffer from `cache.q_cap` | D2H read before `set_query()` |
| 4 | Token history `std::vector<int32_t>` | passed to `maybe_reselect()` |
| 5 | `chunk_tokens` consistent with pager config | used in `on_committed()` |
| 6 | Graph code: `emit_qk_query_capture()` per full-attn layer (see header) | graph builder |

**Basis invariant** (non-negotiable): Q and pooled K must share one
orthogonal basis at score time.  FWHT-rotating backends (qwen35 Q8_0 /
TQ3_0) satisfy this automatically; non-rotating backends satisfy it
trivially in the raw post-RoPE basis.  FWHT is NOT a prerequisite.

**GQA guard**: `n_q_heads % n_kv_heads == 0` is asserted in `reset()`.
The current scorer assumes block-contiguous GQA (`hk = hq / group`).
See the `GQA GUARD TODO` comment in the header for the interleaved-layout
extension point in `kvflash_qk_chunk_scores`.

## 4. Per-Model Adoption

### qwen35 (reference — rewire to controller)

Effort: S.  Replace the three qwen35-specific member variables
(`kvflash_qk_pool_`, `kvflash_qk_scorer_`, `kvflash_qk_pooled_upto_`)
with `KvFlashQkController qk_ctrl_{&kvflash_pager_}`.  Replace the
bodies of `kvflash_qk_pool_to()` and `kvflash_maybe_reselect()` with
single-line delegations.  Serialize/deserialize calls become
`qk_ctrl_.serialize(snap.pooled_qk)` / `qk_ctrl_.deserialize(snap.pooled_qk)`.

Nothing changes for the graph-builder (q_cap capture already present).

### qwen35moe (small — inherits same stack, one constraint)

Effort: S.  The paged prefill and AR-mode decode paths are near-identical
to qwen35.  Wire `on_committed` and `maybe_reselect` the same way.

Constraint: the pipelined-decode CUDA-graph path does not emit `q_capture`
on every step (the graph is frozen at compile time).  Solution: on the
step where `maybe_reselect` is due (generated % tau == 0), run a single
non-graph decode step with `q_capture=true` to produce `cache.q_cap`,
read it back, call `set_query()`, then call `maybe_reselect()` with
`generated` set so the cadence check passes.  Cost: one extra non-graph
forward, amortized over tau steps.

### generic-llama (dense full-attn, drop-in)

Effort: S.  Dense models have `fa_idx == il` (every layer is full-attn),
no rotation, uniform dims.  Adopt the controller directly with
`KvFlashQkDims{n_layers, n_q_heads, n_kv_heads, head_dim}`.  The basis
invariant is satisfied in the raw post-RoPE basis (identity transform).
No changes to `kvflash_qk_chunk_scores` required.

### gemma4 (lower priority — per-layer dim variation)

Effort: M.  Gemma4 uses alternating local/global attention and varies
head dim per layer (64 / 256 alternation).  Two options:
1. Use only the global (full-attn) layers in the pool, with uniform dims
   from the global-attention subset — same pattern as qwen35's
   full_attention_interval.
2. Extend `KvFlashQkDims` to a `std::vector` of per-layer dims, update
   `pool_chunk` and `kvflash_qk_chunk_scores` accordingly.

Option 1 is preferred for initial adoption (S effort); option 2 deferred
until a gemma4 QK-scored residency experiment shows a meaningful quality
delta over option 1.

The drafter scorer for gemma4 (KvFlashDrafterScorer) already covers the
residency case; QK scorer is additive.

## 5. Ordered Steps + Effort

| Step | Description | Effort |
|------|-------------|--------|
| 1 | Add `server/src/common/kvflash_qk_controller.h` (done) | S |
| 2 | Rewire `Qwen35Backend` to use `KvFlashQkController` (replace 3 member vars + 2 methods) | S |
| 3 | Extend `KvFlashQkDims` with the GQA divisibility guard (assert in reset()) | S |
| 4 | Wire qwen35moe: adopt controller in AR path; add periodic uncached q-capture in graph path | S |
| 5 | Wire generic-llama backend: zero-modification to controller, add `emit_qk_query_capture` to graph | S |
| 6 | Gemma4 (option 1: global-attn subset only) | M |
| 7 | GQA interleaved-layout extension (if/when a kInterleaved backend is adopted) | M |

**Basis invariant check** (part of step 2 validation): after rewiring,
run a 32K NIAH with `DFLASH_KVFLASH_POLICY=qk` and confirm recall
equivalence vs. the pre-refactor qwen35 baseline.  Acceptance criterion:
recall delta < 2 pp.

**FWHT clarification**: steps 4-6 do NOT require FWHT.  The invariant is
"Q and pooled-K share one orthogonal basis"; backends without a rotating
K cache use the identity basis (raw post-RoPE) and satisfy the invariant
unconditionally.
