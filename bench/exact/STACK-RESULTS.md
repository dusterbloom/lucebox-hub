# qwen4exp exact-stack: pipelining port result

Box: lucebox4 (gfx1151). Tree: `~/qwen4exp-exact-stack` (full independent cmake configure +
build, source copied from the exact 38.41 tree at commit 3f48adf6, built with the same
cmake options as `~/qwen4exp-private-exact/build1151`: Release, `GGML_HIP=ON`,
`GPU_TARGETS=gfx1151`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_MMQ_MFMA=ON`). Driver:
`bench/exact/driver_shared_epilogue.cpp` (`driver-stack`), same args/model as the reference
(`prompt.ids` 7235 tokens, 256 forced-follow decode steps, F16 KV, `MEASURE_FOLLOW`,
`LUCE_GPU_ARGMAX=1`), fresh process per rep via `gpu_exec.sh`.

## Baseline reproduction (unmodified default path, port installed but gated OFF)

`REPS=1 TAG=final_default_check bash run_stack.sh` (full baseline env, `LUCE_QWEN_SHARED_OVERLAP=1`,
`LUCE_QWEN_PIPELINE` unset): mode=1 ms_token = 38.498 / 38.479 / 38.485 / 38.489 (one 39.70 outlier
excluded, matches the documented cold/thermal noise band) — median **38.489 ms/token**, consistent
with the exact tree's documented 38.16–38.41 range. `[measure_tokens]` sequence bit-identical to
`~/qwen4exp-f16/private-exact/rep1.log`. **Gate (b) passes: the default path is unchanged.**

## Pipelining port (LUCE_QWEN_PIPELINE=1)

Ported from branch `qwen4exp-pipelined-decode` (32f05889, off `qwen4exp-fusion-wire3b`) onto the
exact tree's `forward_impl`/`Qwen4ExpDecodeWorkspace`: `qwen4exp_pipeline.h` (new),
`qwen4exp_cache.h` (+`pipelined`/`split`/`tok_in` fields), `qwen4exp_graph.cpp` (`forward_impl`
takes an optional `Qwen4ExpPipeline *`; splits the stable T=1 graph into two `ggml_graph_view`s at
the first PLE layer; device token feed via `get_rows(tok_embd_dev, tok_in)`; greedy pick fed back
with `ggml_cpy(argmax, tok_in)`; linear-layer state snapshot/rollback for the speculative
lookahead half-step). Driver: `bench/exact/driver_shared_epilogue.cpp` gained a pipelined decode
loop (`qwen4exp_pipeline_begin/step/wait/end`), used only when `LUCE_QWEN_PIPELINE=1`.

**Hard incompatibility found during the port**: `forward_impl`'s pipe path requires
`!shared_overlap_requested` (same exclusion as the upstream design: the split-graph-view
execution model does not fit `LUCE_QWEN_SHARED_OVERLAP`'s single-call async-overlap plan). The
exact tree's 38.41 baseline recipe always sets `LUCE_QWEN_SHARED_OVERLAP=1`. A fair A/B therefore
needs `LUCE_QWEN_SHARED_OVERLAP=0` on **both** arms (pipelining can never be measured against the
real 38.41 baseline directly — that comparison would be confounded by shared-overlap's own,
separate contribution). `run_stack_pipe.sh` runs both arms with `LUCE_QWEN_SHARED_OVERLAP=0`.

One session, same binary, 8 internal reps per process (schedule alternates mode 0/1, a separate
epilogue axis), interleaved by construction. `[measure_tokens]` identical across both arms and
identical to the baseline's decode sequence (`27775 383 279 1970 3766 11 279 2972 12610 4818 ...`).

| Arm | ms/token (8 internal reps) | median | vs OFF |
|---|---|---|---|
| OFF (serial, `SHARED_OVERLAP=0`) | 41.583 / 41.542 / 41.561 / 41.540 / 41.558 / 41.533 / 41.626 / 41.548 | **41.553** | — |
| ON (`LUCE_QWEN_PIPELINE=1`, `SHARED_OVERLAP=0`) | 41.739 / 41.726 / 42.270 / 42.251 / 41.734 / 41.734 / 41.731 / 42.247 | **41.737** | **+0.184 ms (regression, −0.4%)** |

Picks: bit-identical on both arms and identical to the `SHARED_OVERLAP=1` baseline's
`[measure_tokens]` sequence (256 forced steps). **Correctness gate passes.**

## Decision: gated OFF, not stacked

The measured delta is a **regression** (+0.184 ms/token), outside the ±0.15 ms noise band in the
wrong direction. The reference branch measured −1.08 ms on a plain dense-mask decode path with no
competing fusion; this exact tree's decode runs the stable-QSA/indexer path (`LUCE_QWEN_EXACT_ROUTER_SUFFIX`,
`qsa_params`/`qsa_visibility` upload per step) under heavier per-step host bookkeeping, and the
extra pinned-ring-buffer staging + two `ggml_backend_graph_compute_async` calls per token (instead
of one `ggml_backend_graph_compute`) cost more than the pre-PLE/PLE-gather overlap recovers on
this integrated-GPU backend. Per the stop rule, the port stays in the tree, fully gated behind
`LUCE_QWEN_PIPELINE=1` (default off, verified byte-unchanged default path), but is **not** enabled
in the final stacked build.

## K5 port (GGML_OP_MOE_ROUTE: router GEMV + top-k + shexp gate, `LUCE_QWEN_K5=1`)

Commit `0a5bc8d9`. Reference: branch `qwen4exp-fusion-wire3b`, commit `18c05971`.

**Overlap check first** (per coordinator instruction): read `EXACT_ROUTER_SUFFIX`'s fusion scope in
`ggml-cuda.cu` (`ggml_cuda_exact_router_suffix`, op-list
`{ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE}`) against `build_moe()`'s
pre-fusion kernel list per layer: router GEMV (`mm`), softmax, argsort_top_k, get_rows, sum_rows,
clamp, div — plus the **separate** shexp-gate GEMV (`mm(ffn_gate_inp_shexp, cur)`) and its sigmoid.
`EXACT_ROUTER_SUFFIX` only fuses the post-softmax suffix (argsort→view→get_rows→reshape→sum_rows→
clamp→div→reshape); it does **not** touch the router GEMV, the softmax, or the shexp-gate GEMV+
sigmoid. K5 is not already covered — ported.

**Port**: cherry-picked `18c05971` (`git cherry-pick -n`), pruned to only what this tree needs —
our tree has none of `GGML_OP_HC_BOUNDARY`/`GDN_TAIL`/`GDN_PREP` (K3/K4 not ported yet), so only
`GGML_OP_MOE_ROUTE` was added (`GGML_OP_COUNT` 113→114, not the reference's 117). `ggml.h`/`ggml.c`/
`ggml-cpu.c`/`ggml-cuda.cu` gained the op enum, builder (`ggml_moe_route` + `_sel`/`_wsel`/
`_sh_gate`/`_part_offset` views, packed I8 result, 256B-aligned parts), CPU reference
(`ggml_compute_forward_moe_route`, O(NE) exact top-k scan matching `ggml_argsort_top_k`'s
lowest-id tie-break), and the HIP kernel (`moe-route.cu`/`.cuh`, copied verbatim — self-contained,
only depends on `common.cuh` and the shared tensor/op_params ABI). `qwen4exp_graph.cpp`'s
`build_moe()` was NOT given the reference's unconditional `kMoeRouteFused=true`; instead gated
behind `LUCE_QWEN_K5=1` (default off) via `moe_route_env_on()`. Wired on the plain direct path only
(`fused_route = !parts && !overlap && n_tokens <= GGML_HC_BOUNDARY_MAX_T && moe_route_on(...)`): the
`parts`/fold path (`ggml_hc_combine_norm_moe`) and the `overlap` shared-overlap-scheduling path both
need the raw pre-sigmoid shexp logit as a distinct graph node, so they're excluded from fusion by
construction and keep the unfused `mm`+`soft_max`+`argsort_top_k` route — `EXACT_ROUTER_SUFFIX`
still applies there, unchanged. Dropped the reference's standalone `gate_moe_route` differential-test
executable (needs `test/bench/moe_route_kernels.cu`, a prototype file not carried over — out of
scope for the gated production port). CMake: `ggml-cuda/CMakeLists.txt` globs `*.cu`, so
`moe-route.cu` was picked up with no file-list change.

One fixup needed after the cherry-pick: `build_moe()`'s `overlap`-struct population and the `parts`
struct both referenced the old unconditional `shared_logit` name; repointed both to
`shared_gate_or_logit` (the unfused branch's raw pre-sigmoid logit — always correct here since
`parts`/`overlap` imply `fused_route == false`).

**Gates**: `REPS=1` fresh-process runs via `run_stack.sh`, full 38.41 env (`SHARED_OVERLAP=1`,
`EXACT_ROUTER_SUFFIX=1` and all other default fusions on), `LUCE_QWEN_K5=1` vs unset.
`[measure_tokens]` (mode=1, forced-follow, 256 steps) **bit-identical** between K5 ON and OFF —
confirmed by diff of the full 256-token sequence (`27775 383 279 1970 ...`). **Gate (a) passes.**
OFF path (K5 env unset) reproduces the known baseline sequence and ms/token band unchanged —
**gate (b) passes.**

**Timing**: two interleaved fresh-process reps per arm (8 internal mode=1 samples each, `run_stack.sh`'s
own mode 0/1 alternation), same session, same binary/libggml-hip.so, full 38.41 env on both arms.

| Arm | mode=1 ms_token samples | median |
|---|---|---|
| OFF (`k5_off`+`k5_off2`, 8 samples) | 38.440, 38.437, 38.452, 38.465, 38.511, 38.468, 38.498, 38.460 | **38.462** |
| ON (`k5_on`+`k5_on2`, 8 samples) | 38.548, 38.516, 38.528, 38.504, 40.376\*, 38.403, 38.424, 38.430 | **38.510** |

\*one cold-start outlier (first mode=1 sample right after process launch, consistent with the
documented cold/thermal noise band elsewhere in this file); included in the median above since the
OFF arm's own first-sample reps show no comparable exclusion-worthy spike, so the comparison stays
apples-to-apples (unfiltered both sides it would be 38.466 OFF median / 38.510 ON median — same
conclusion either way).

**Delta: ON − OFF = +0.048 ms/token**, inside the ±0.15 ms noise band.

**CORRECTION (caught in review, confirmed by instrumentation): the above timing comparison is
meaningless as a test of the fused kernel.** `fused_route = !parts && !overlap && T<=8 &&
moe_route_on(...)` (see code above), and the headline 38.41 config sets
`LUCE_QWEN_SHARED_OVERLAP=1`, which makes `build_moe`'s main T=1 decode call site
(`qwen4exp_graph.cpp:1745`, `build_shared_overlap && layer_T==1 && !fold`) pass a non-null
`overlap` for essentially every layer. `fused_route` is therefore `false` on the hot decode path
regardless of `LUCE_QWEN_K5`, and `GGML_OP_MOE_ROUTE` never executes in this config — both "ON" and
"OFF" ran the identical unfused code path. That is also why bit-identity "passed": there was nothing
different to diverge.

Added a debug counter (`LUCE_QWEN_K5_DEBUG=1`, `fired`/`skipped` atomics in `build_moe`, gated off
a static `getenv` check, zero cost when unset) and reran `LUCE_QWEN_K5=1 LUCE_QWEN_K5_DEBUG=1` under
the full env. Every logged checkpoint, including `overlap=1 parts=0 T=1` (the real decode call
site), reads `fired=0`:
```
[k5-debug] fired=0 skipped=1 (overlap=0 parts=0 T=4096)
[k5-debug] fired=0 skipped=513 (overlap=1 parts=0 T=1)
[k5-debug] fired=0 skipped=1025 (overlap=0 parts=1 T=3138)
[k5-debug] fired=0 skipped=1537 (overlap=0 parts=0 T=4096)
```
Confirmed: **0 fires** across the whole run. `moe_route_on(w, L, n_tokens)` and
`ggml_backend_cuda_moe_route_supported` were never even reached for the `overlap=1` case because the
`!overlap` guard short-circuits first.

**Decision: gated OFF (default), not stacked.** Record corrected from "measured-neutral" to
**"inactive under `SHARED_OVERLAP=1`, not executed — not a timing result at all"**. Per the
coordinator's instruction, deferred: K5 would need `fused_route` to also emit the pre-sigmoid shexp
logit as a separate output view (so the `overlap`-scheduling path can still split the graph at a
distinct node the way the unfused path does today), making it overlap-compatible. That redesign is
picked up only after K3/K4 are done, not this pass. `LUCE_QWEN_K5_DEBUG=1` stays in the tree
(zero-cost when unset) so any future re-attempt can re-verify fire counts before trusting a timing
number.

## K3 (GDN-prep: beta/gate projections + conv+silu + q/k l2-norm) — investigated, skipped

Reference: `3fa48a54`. K3 fuses three things into one launch for `T <= GGML_HC_BOUNDARY_MAX_T`:
(1) `beta = sigmoid(w_beta @ mixed)` + `gate = softplus(w_alpha @ mixed + dt_bias) * ssm_a`, (2) a
causal conv1d + SiLU over the concatenated qkv history, (3) q/k l2-norm over the conv output.

**Overlap check against this tree's actual `build_linear_attn`/`build_linear_attn_projected`** (not
just the reference's own pre-K3 baseline, which has drifted from ours — this tree has its own,
independently-evolved fast paths):

1. **Beta/gate projections (K3 part 1)**: already covered by `GDN_AB_EXACT`
   (`LUCE_QWEN_GDN_AB_EXACT=1`, on by default in `run_stack.sh`'s full env). Read
   `ggml_cuda_gdn_ab_exact_match` in `ggml-cuda.cu`: its op-list
   `{MUL_MAT, RESHAPE, ADD, UNARY(SOFTPLUS), MUL, RESHAPE, MUL_MAT, RESHAPE, UNARY(SIGMOID)}` is
   *exactly* K3's beta/gate math (alpha-GEMM→+dt_bias→softplus→*ssm_a, beta-GEMM→sigmoid), runtime
   subgraph-detected and fused on this tree's existing `GGML_OP_GATED_DELTA_NET` call site. Confirmed
   identical in both `build_linear_attn` (solo decode) and the batched-decode caller that feeds
   `build_linear_attn_projected`'s per-slot `beta`/`alpha` params.
2. **Conv+SiLU (K3 part 2), the dominant case**: this tree has its own fused kernel,
   `ggml_ssm_conv_step` (conv + causal-history update + SiLU in one launch), used unconditionally by
   `build_linear_attn_projected` (the batched decode path, every call) and by `build_linear_attn`'s
   `T==1 && f16 && !spec_conv` fast path (`qwen4exp_graph.cpp:434` — i.e. every ordinary T=1 decode
   step). This does not exist in the K3 reference's lineage at all; it is this tree's own, narrower,
   earlier fusion of the same conv+silu scope. K3's generic `ggml_concat(hist,qkv_t)` +
   `ggml_ssm_conv` + `ggml_silu` route only survives in `build_linear_attn`'s `else` branch — taken
   only for `T>1` (speculative/verify multi-token batches), not the hot single-token decode step that
   `STACK-RESULTS.md`'s ms/token measurements are based on.
3. **Q/K l2-norm (K3 part 3)**: genuinely not fused anywhere in this tree (separate
   `ggml_rms_norm`+`ggml_scale` per q/k) — the only part of K3 with no existing counterpart.

**Attempted a cherry-pick of `3fa48a54` to check feasibility of porting just the residual (l2-norm)
scope**: the ggml-level plumbing (`ggml.h`/`ggml.c`/`ggml-cpu.c`/`ggml-cuda.cu` + new
`gdn-prep.cu`/`.cuh`) is tree-agnostic and merged cleanly (same pattern as K5: prune
HC_BOUNDARY/GDN_TAIL, keep only GDN_PREP, `GGML_OP_COUNT` 114→115). The `qwen4exp_graph.cpp` hunks
did not: the reference's diff context assumes a `build_linear_attn` shape with `mm_q8`/`cur_q8`
(quantized producer) and no `ggml_ssm_conv_step` fast path, neither of which match this tree's
current, already-more-fused structure — the 3-way merge imported dead, non-native helpers
(`HcPending`, `hc_boundary()`, `mm_q8()`) that don't belong here and silently skipped reconciling
the real call sites. Correctly wiring K3 against this tree's actual code would mean hand-writing the
fusion into the `T>1` branch of `build_linear_attn` only (since `T==1` already goes through
`ggml_ssm_conv_step`, which does not produce the packed layout `GGML_OP_GDN_PREP` expects) — a
narrower, single-purpose port (just q/k l2-norm fused into the verify/speculative path) for a path
that is not on the per-token decode critical path this session is optimizing for.

**Decision: skip, not ported.** Two of K3's three fused components are already covered by existing
fusions on the hot decode path (`GDN_AB_EXACT` for beta/gate, `ggml_ssm_conv_step` for conv+silu);
the only uncovered piece (q/k l2-norm) only matters on the `T>1` verify path, and porting it there
would require a hand-built graph rewrite rather than a mechanical cherry-pick, with correctness risk
the original commit itself flags (solo/batched GDN-recurrence divergence from ULP-level fusion
differences) for a path this isn't measuring. Reverted the trial cherry-pick in full (`git checkout
HEAD --` on all six touched files, deleted the three new files) — working tree is clean, no K3
artifacts left in the commit.

## K4 (GDN tail: `rms_norm(x)*gamma*sigmoid(z)`)

Not yet attempted. Overlap check against "the gated-norm path" (per coordinator instruction) is the
next step, following the same investigate-first/gate/time/commit protocol used for K5.
