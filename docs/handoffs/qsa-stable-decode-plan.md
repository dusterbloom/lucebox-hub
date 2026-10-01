# Stable QSA decode graph (#3b)

Implemented on `0058b58d` in `qwen4exp-qsa-speed`. Ratio-4, top-512 T=1
QSA now keeps one ggml graph and gallocr plan per 256-token KV bucket,
including completion steps. `QWEN4EXP_QSA_STABLE` is removed. The dense stable
path and prefill computations retain their existing behavior.

## Runtime graph and pooled writes

Positions and the K/V/raw-key destination remain runtime inputs. Two new input
tensors, shared by all full layers, carry the F32 block visibility mask and
I32 values `[n, raw_rows[4], pooled_row, rope_positions[4]]`. Here
`n = floor((p+1)/4)`. Scores cover `floor(bucket/4)` blocks using
`ggml_ds4_indexer_score_masked(..., mask, 0, 4)`; no visibility parameter
captures `p`. The fused cell-ID op still reads the position tensor.

The raw-key `set_rows` result feeds `get_rows` for block `n-1`. Pooling uses
`qwen4exp_pool_blocks` unchanged: three left-associated F32 additions in token
order and multiplication by 1/4. Both paths share the original RMS norm,
learned scale and M-RoPE recipe. On completion (`(p+1)%4 == 0`), `set_rows`
stores the result at `n-1`; otherwise it stores at a dedicated last cache row,
`ceil(max_ctx/4)`. That row is outside every scoring view, including a truncated
final bucket, and outside the authoritative pooled prefix. No live pooled row
is rewritten on non-completion steps. Scoring consumes the write result, so
append, gather, pool, write, F16 conversion and score have explicit dependencies.

A cache created during dense attention may lack pooled keys. Once, on entering
QSA, the original pooling helper fills only the already-cached complete prefix
`[indexer_blocks, floor(p/4))` in a separate bootstrap graph. The stable graph
handles any block completed by the current token. Bootstrap uses the original
multi-block arithmetic; it is not repeated during normal bucket replay.

The pooled allocation gains one F32 scratch row and is initialized at creation.
The existing T=1 scorer's arithmetic for a column does not depend on score-row
width: width only controls its tile bounds/grid. Invisible columns are replaced
by -1e30. QSA head weights are exactly one, and its ReLU/head sum starts at +0,
so every visible score is nonnegative (+0 included). Thus every invisible score
is strictly smaller. The masked-score test also poisons invisible key columns
with NaNs and requires every visible score bit to match an exact-width call.

## Why top-k membership AND order are exact, including ties

`ggml_top_k_qsa` is a narrow top-512 variant with a device valid count and a
nonnegative-score/+0, -1e30-padding contract. It preserves the existing
backend's order; it does not introduce a new generic tie rule.

1. **513..1024 valid blocks:** the existing bitonic sort is not stable. The
   variant runs that same kernel with the device count in every validity
   comparison. Both exact and runtime calls have precisely 1024 lanes and the
   same compare/exchange network. Initial indices, valid key bits and every
   swap decision are identical. Returning only its first 512 indices therefore
   preserves tie membership and order. The no-CUB partial-bitonic route likewise
   reuses its original `KPAD=512` kernel with the runtime count.
2. **More than 1024 blocks:** the existing single-row HIP routes are stable
   radix sorts, including the flat block-radix, hierarchical and tiled routes.
   [rocPRIM block radix sort documents stability](https://rocm.docs.amd.com/projects/rocPRIM/en/docs-6.4.2/block_ops/ops_classes/sort.html#radix-sort),
   as does its [device radix sort](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/device_ops/sort.html).
   For QSA's nonnegative/+0 keys, the block kernels' integer key transform and
   the device sort's floating-point transform induce the same ordering. Hence
   each flat sort returns descending score, then original index on ties.
3. **Tile induction:** within each contiguous tile the retained 512 candidates
   have that same order. A discarded candidate already has 512 predecessors
   inside its own tile, so cannot belong to the global first 512. Merge inputs
   visit contiguous tiles/groups in source order. Equal scores within a tile
   keep index order; equal scores across tiles keep tile order. Each stable
   merge therefore returns the same ordered prefix as the flat sort. This
   argument repeats at every merge level and holds when tile counts or radix
   items-per-thread change.
4. **Padding:** appended invisible keys rank strictly below every visible key;
   with at least 513 visible keys they cannot enter the first 512 or affect
   their relative order. This proves equivalence across all radix boundaries,
   including `GGML_DS4_TOPK_BLOCK_RADIX=0/1`, rather than just within a dispatch
   region. For the bucket crossing 1024, the captured graph includes the radix
   route plus the original bitonic kernel: the latter overwrites the result
   only when the device valid count is <=1024. There is no host count readback
   or per-token launch-topology change.

Older CUDA CUB radix-sort builds use the same argument. CUDA builds selecting
`DeviceTopK` explicitly request nondeterministic, unsorted output; that route
has no equivalent tie guarantee and reports this variant unsupported, retaining
the original rebuilding graph. Builds without CUB support only <=1024 columns,
matching the original backend limit. CPU executes its original partial sort
with the runtime exact end iterator. The stable model path checks support.

## Lifecycle and checks

Replay requires the same model, backend, capacity, QSA mode/budget, consecutive
position and authoritative pooled-prefix count. Reset/prefill clear the workspace;
discontinuities rebuild. Native captures are retired before metadata/allocator
reuse or free, including bootstrap. Cache prefix metadata advances only after
successful compute. Backend dispatch settings/model contents remain immutable
during an inference session, as in the existing dense stable path.

`test_qwen4exp_qsa_ids` compares the ordered selections for every logical width
in buckets around every selection boundary, including all widths 513..1088,
all-zero/all-equal/cutoff/discrete ties and unique scores. Runtime inputs change
while the padded graph survives. `test_qwen4exp_indexer_score` adds bitwise
masked/padded-vs-exact T=1 comparisons. `QWEN4EXP_SMOKE_STABLE` uses a function
argument for the original rebuilding oracle, compares every logit and live
cache byte, and checks build/replay counters (pointer equality could miss a
rebuild into recycled metadata).

Local verification: the three supplied check scripts were copied to
`/tmp/qsa-3b-chk_{q4,unit,rope}.sh`, with W set to this worktree. Qwen sources,
smoke/tests and modified CUDA units compile. A separate CPU ggml build passed
9,095 runtime top-k comparisons plus all 89 existing QSA-ID cases. GPU model
parity, native replay and throughput remain to be measured on the GPU box.

## GPU acceptance commands

Run in this worktree with the existing GPU build configuration. `QSA_MODEL`
is the baseline UD model; the three token files must be exactly the baseline's
6000, 4096 and 16384 real-text token IDs, respectively. No profile logging for
timings. Keep the baseline's capture settings; repeat parity under both its
capture-enabled and capture-disabled configurations.

```bash
cmake --build server/build -j4 --target smoke_qwen4exp_forward test_qwen4exp_qsa_ids test_qwen4exp_indexer_score
server/build/test_qwen4exp_qsa_ids
GGML_DS4_TOPK_BLOCK_RADIX=0 server/build/test_qwen4exp_qsa_ids
GGML_DS4_TOPK_BLOCK_RADIX=1 server/build/test_qwen4exp_qsa_ids
server/build/test_qwen4exp_indexer_score
for split in 100:1 200:1 100:100 256:128; do
  QWEN4EXP_QSA=1 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS" QWEN4EXP_SMOKE_SPLIT="$split" server/build/smoke_qwen4exp_forward "$QSA_MODEL" 6000
done
# Exact KL in order: 0.082874 / 0.118543 / 0.652715 / 0.734763.
QWEN4EXP_QSA=1 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS" QWEN4EXP_SMOKE_STABLE=3952 server/build/smoke_qwen4exp_forward "$QSA_MODEL" 6000
QWEN4EXP_QSA=1 QWEN4EXP_SMOKE_TG=128 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS_4K" server/build/smoke_qwen4exp_forward "$QSA_MODEL" 4096
QWEN4EXP_QSA=1 QWEN4EXP_SMOKE_TG=128 QWEN4EXP_TOKEN_FILE="$QSA_TOKENS_16K" server/build/smoke_qwen4exp_forward "$QSA_MODEL" 16384
```

The stable comparison must be bitwise; its long run crosses dense-to-QSA,
completion residues, KV buckets and the 1024-block route boundary. The tg128
results must beat 22.1 tok/s at 4K and 22.3 at 16K; target >=23 at both. No GPU
quality or speed result is claimed from the local compile/CPU checks.
