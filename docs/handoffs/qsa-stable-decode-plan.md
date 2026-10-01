# Stable QSA decode graph: design and implementation plan

Status: exact partial reuse implemented on top of `a07d89fd` (fused decode IDs).
Full per-bucket QSA reuse remains deferred: no padded top-k equivalence is claimed.

## Implemented scope and exactness

`qwen4exp_forward()` keeps the 256-token K/V bucket and reuses a ratio-4 QSA
T=1 graph on the three steps **after** block completion. Completion and bootstrap
steps still run the original pooling/concat/norm/RoPE graph; the next step builds
a graph reading the completed F32 pooled prefix, and the following two steps
reuse it with input uploads only. This removes two of four graph builds in steady
decode. Gallocr backing storage survives rebuilds; every new graph is reserved.
`QWEN4EXP_QSA_STABLE=0` selects the original per-step QSA rebuild for comparison.

- The pooled prefix is immutable throughout the reuse interval. There is no
  redundant pooling and no change to the F32-to-F16 conversion or score shape.
- `ggml_ds4_indexer_score` uses `kv_start` only through `(kv_start+1)/4` for
  T=1 visibility, which is constant throughout the interval. Its kernels,
  parameters governing dispatch, head weights and logical width are unchanged.
- `ggml_top_k` still sees exactly `n_after=(pos+1)/4` columns. Its existing
  dispatch, tie behavior and fused downstream cell IDs are unchanged; the latter
  read the runtime position input.
- Only K/V views are bucketed. The selected-attention kernel uses their length
  solely to reject out-of-range IDs. Every nonnegative ID remains at most `pos`;
  bucket padding therefore introduces no extra reads. Strides, WMMA eligibility,
  selection-slot count, split count and reduction order are unchanged. Buckets
  are capped at the existing decode kernel's 262144-cell limit.
- K/V writes use the existing stable `set_rows` path, and QSA views depend on
  the returned write tensors. Raw indexer writes also use the runtime row.
- Replay requires the same model/backend/capacity, QSA mode, logical block count,
  budget and consecutive position. Model contents and dispatch environment are
  assumed immutable during inference. Prefill/reset discard the workspace;
  position discontinuities rebuild it. Cache ownership supplies cache identity.
- Native captures are retired using the existing synchronized
  `ggml_backend_cuda_graph_invalidate_range` before metadata reset/free, including
  reset, failures and cache teardown. No graph-replay default is changed.

The reasoning above preserves the original arithmetic and selection operations;
GPU bit equality and the throughput target still require the checks below.

## GPU acceptance commands

Use the same model, real-text token files and backend settings as the baseline.
`QSA_TOKENS` contains exactly 6000 IDs; `QSA_TOKENS_4K` and `QSA_TOKENS_16K`
contain exactly 4096 and 16384 IDs from the baseline generation inputs.

```bash
cmake -S server -B server/build
cmake --build server/build -j4 --target smoke_qwen4exp_forward test_qwen4exp_qsa_ids test_qwen4exp_indexer_score
server/build/test_qwen4exp_qsa_ids
server/build/test_qwen4exp_indexer_score
for split in 100:1 200:1 100:100 256:128; do
  QWEN4EXP_QSA=1 QWEN4EXP_QSA_STABLE=1 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS" QWEN4EXP_SMOKE_SPLIT="$split" server/build/smoke_qwen4exp_forward "$QSA_MODEL" 6000
done
# Expected KL, in order: 0.082874 / 0.118543 / 0.652715 / 0.734763.
# Every-step logits and persistent cache bits, rebuilt vs stable, across
# the dense/QSA transition, every residue, buckets and the 1024-block boundary:
QWEN4EXP_QSA=1 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS" QWEN4EXP_SMOKE_STABLE=3952 server/build/smoke_qwen4exp_forward "$QSA_MODEL" 6000
# Optional shorter differential: QWEN4EXP_SMOKE_STABLE=100.
QWEN4EXP_QSA=1 QWEN4EXP_QSA_STABLE=1 QWEN4EXP_SMOKE_TG=128 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS_4K" server/build/smoke_qwen4exp_forward "$QSA_MODEL" 4096
QWEN4EXP_QSA=1 QWEN4EXP_QSA_STABLE=1 QWEN4EXP_SMOKE_TG=128 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS_16K" server/build/smoke_qwen4exp_forward "$QSA_MODEL" 16384
```

Both tg128 results must exceed 21.5 tok/s; compare with the same commands using
`QWEN4EXP_QSA_STABLE=0`. Leave profiling off for acceptance timings; add
`QWEN4EXP_PROF=1` separately to see two input-only QSA replays per four steps.
The bit differential is intentionally expensive (reads all live cache rows each
step) and is not a benchmark. Repeat it under the baseline's capture-enabled and
capture-disabled settings; ggml graph reuse does not imply native capture replay
within a three-call window.

The remaining sections describe the deferred full-bucket design.

## Contract and current constraints

For T=1 and ratio 4, position `p` sees `n = floor((p+1)/4)` complete blocks.
Keep the current scores, the **same 512 selected block IDs**, their ascending
order, and the 2,051 output slots. Full blocks require `4*b+3 <= p`; tail slot
`2048+i` is `4*n+i` when that cell is at most `p`, otherwise -1. Never compact
invalid slots or change the attention split/reduction order.

`qwen4exp_forward()` at baseline `a07d89fd` reused a stable graph only in `QSA_DENSE` mode.
QSA rebuilds because `qsa_pooled_keys()` changes concatenations, views, RoPE
positions and copy offsets whenever the complete-block count changes.
`build_qsa_attn()` also embeds `kv_start` and the logical score width. The new
`ggml_qsa_decode_ids()` already accepts device position data, so its output shape
and parameters can remain constant during decode.

A padded score array is **not sufficient to preserve top-k ties**. Existing
`ggml_top_k` dispatch depends on logical width, backend, build options and
`GGML_DS4_TOPK_BLOCK_RADIX`. On the current HIP path, relevant boundaries include
1,024 columns, radix variants at 2,048/3,072/4,096/5,120, and hierarchical/tiled
routes beyond that. Padding can change tie membership even when padded scores
are strictly below visible scores. Do not replace this with a new index-based
tie rule: the requirement is the existing backend's behavior, not a different
stable ordering. This is the main prerequisite for full bucket reuse.

## Fixed storage and runtime inputs

Extend `Qwen4ExpDecodeWorkspace` in `qwen4exp_cache.h`, keeping ownership and
cleanup in `qwen4exp_cache.cpp`. Keep the existing 256-token K/V bucket policy
(initially allow the same forward slack), capped by `cache.max_ctx`. For bucket
capacity `B`, allocate `M = floor(B/4)` block rows. Rebuild on bucket, mode,
cache identity, model/layout, or relevant backend-dispatch changes.

| Item | Fixed shape/type | Per-token data |
| --- | --- | --- |
| positions / kv_start | existing I32[4] M-RoPE input | `[p,p,p,0]`; fused IDs view the first entry |
| K/V and raw-key write row | existing I32[1] | `p` |
| block visibility | F32[M,1], shared across full layers | 0 for `b<n`, -infinity otherwise |
| raw pooling rows | I32[4] | `4*(n-1)+[0,1,2,3]` |
| pooled destination row | I32[1] | `n-1` |
| pooled RoPE positions | I32[4] | `[4*(n-1),4*(n-1),4*(n-1),0]` |
| logical block count | I32[1], if selection needs it | `n` |
| pooled keys / scoring view | existing F32 cache, fixed F16[128,M] scoring input | update only completed pooled rows; mask hides future rows |
| score / selected / cell arrays | F32[M,1], I32[512,1], I32[2051,1] | graph outputs |

Use `ggml_ds4_indexer_score_masked(q, weights, comp16, visibility, 0, 4)`.
The runtime visibility mask replaces baked `kv_start` visibility; the op has no
runtime integer kv_start argument. The host derives both positions and the
mask from the same `p`. The kernel's invisible-score sentinel remains -1e30.
Initialize unused persistent rows once, and verify every scoring specialization
honors the mask without allowing stale/NaN padded data into visible scores.
Initially retain the current pooled F32-to-F16 conversion node, with fixed
bucket width. A persistent F16 cache is a separate optimization requiring its
own precision and lifecycle checks.

## Pool updates and dependency ordering

1. Before entering a stable QSA workspace, populate missing completed pooled
   keys using the existing `qsa_pooled_keys()` path. Dense processing already
   stores raw projected keys. Bootstrap after prefill/reset/restore as needed;
   do not assume `indexer_blocks` was maintained during dense decode.
2. Append K, V and `kraw` through `ggml_set_rows` at runtime row `p`. Use the
   returned tensors as subsequent sources, including K/V views and raw gathers,
   so writes precede reads through graph edges and buffer lifetimes.
3. Always gather the last complete block, `n-1`, in increasing token order.
   QSA starts beyond 512 complete blocks, so this index is valid. Recompute it
   every decode step initially: three of four steps redundantly rewrite an
   immutable block, but the graph topology stays constant.
4. Use `qwen4exp_pool_blocks()` unchanged: token-order additions followed by
   multiplication by 1/4, then existing RMS norm, scale and M-RoPE. Keep the F32
   raw and pooled caches. Store the result through `ggml_set_rows(indexer_k,
   fresh, pooled_row)` and feed that returned tensor's fixed prefix into scoring.
   Do not introduce a parallel reduction or pool in F16.
5. On successful execution, update `cache.indexer_blocks = n` on both the
   first-build path and stable early return, together with existing sequence
   state. On errors invalidate the workspace; do not advertise a completed
   cache transaction after only some layer writes succeeded.

Check F32 pooled-key bits against the current path before relying on redundant
recomputation: identical mathematical formulas alone do not establish identical
backend reduction behavior. Conditional updates once every fourth token can be
considered later; they are unnecessary for the first correct version.

## Selection strategy and staged delivery

1. Add runtime inputs and fixed bucket K/V, raw and pooled views, with a debug
   comparison against the rebuilding path. Keep stable QSA opt-in during
   development. Audit mode transitions, especially the dense-to-QSA transition
   at 2,052 tokens, independently of whether `kv_len` still fits the old bucket.
2. First preserve exact top-k by keeping its **logical-width input and current
   dispatch**. Rebuild a graph variant when `n` changes (once every four decode
   steps), while retaining bucket allocations. This is a useful intermediate
   optimization, but is not full per-bucket graph reuse. Do not claim that a
   device logical-count input makes the existing host dispatch capture-safe.
3. For full bucket reuse, audit each existing top-k route. Determine whether
   fixed-width masked execution gives exactly its unpadded tie membership and
   score bits throughout that route's range. Compare against the current op for
   every logical width in supported buckets with cutoff ties, all-equal/zero
   scores, and visibility-boundary ties. Test dispatch environment overrides.
   A proof must cover the algorithm's tie order, not just random samples.
4. Where padding cannot be proved equivalent, design a runtime-logical-width
   selection path that reproduces the current route and tie behavior, including
   original-width padding and candidate ordering. If dispatch requires different
   launches or temporary layouts, partition buckets further or retain logical-
   width graph variants. Never enable a changed tie rule by default to obtain a
   larger reuse interval. This work is a separate selection change, not part of
   the fused IDs op.
5. Only after the selection prerequisite, reuse one fully fixed graph throughout
   each qualified bucket and dispatch region. Keep a rebuilding fallback for
   other devices, shapes, or routes. Measure whether fixed-width scoring overhead
   is outweighed by CPU construction/allocation and replay savings.

## Workspace and graph lifecycle

Include QSA/dense mode, ratio, budget, K/V and block capacities, cache identity
and selection variant in workspace compatibility checks. Discard stale graphs
on reset, cache restore/rollback, context resize, prefill, or device/model change;
resume from authoritative cache state. Keep allocation planning on every new
graph: do not simply remove gallocr reserve calls.

Audit HIP/CUDA executable-graph ownership before freeing/reusing ggml metadata
or allocator addresses. `clear_qwen4exp_decode_workspace()` now retires native captures before freeing
the allocator and context. Pool trimming's legacy-pool graph clear is not a
universal per-workspace invalidation API. If existing graph identity checks do
not cover replacement, add explicit backend graph retirement with synchronization
before freeing captured buffers. Test allocator-address reuse as well as normal
bucket growth. Preserve the build's existing graph-replay defaults; enabling HIP
replay is a separate qualification, not a consequence of stable ggml topology.

## Validation and performance acceptance

- Compare baseline and candidate scores bitwise, selected blocks and all cell
  slots exactly, then logits and persistent caches for identical inputs. Cover
  all residues, 2,048..2,055 tokens, 256-token bucket crossings, every top-k route
  boundary, prefill-to-decode, reset, rollback, and workspace replacement.
- Run `test_qwen4exp_qsa_ids` and `test_qwen4exp_indexer_score`. Add a masked-score
  differential and stable/rebuild cache-state test when implementing this plan.
  Exercise capture/replay both enabled and disabled, including actual replay
  after warmup rather than just first execution.
- Reuse the exact real-text token IDs, model and backend settings of the baseline:
  `QWEN4EXP_QSA=1 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS" QWEN4EXP_SMOKE_SPLIT=100:1
  server/build/smoke_qwen4exp_forward "$QSA_MODEL" 6000` must print KL 0.082874;
  the same command with `200:1` must print 0.118543. The token file contains
  exactly 6,000 whitespace-separated integer IDs, not text or synthetic tokens.
- Use `QWEN4EXP_PROF=1` and a HIP trace at 4K/16K with identical generation input.
  Report build+alloc, uploads, score/selection/attention time, steady decode,
  capture warmup and bucket rebuild costs separately. No speed claim until
  measured; retain exact-output gating even if the candidate is faster.

Any unavoidable floating-point regrouping or changed selections is outside the
exact mode and must remain disabled by default and be reported explicitly.
