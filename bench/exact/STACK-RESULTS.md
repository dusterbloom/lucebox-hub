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

## K3 / K4 / K5 (GDN-prep fusion, GDN tail, MoE router+topk+shexp gate)

Not attempted this session. Scope check: each adds new `ggml` ops (`ggml.h`/`ggml.c`/`ggml-cpu.c`/
`ggml-cuda.cu` + new `.cu` kernel files) and new graph-routing call sites that must be reconciled
by hand against this tree's existing private fusions (`GDN_AB_EXACT` overlaps K3's GDN-prep scope;
`EXACT_ROUTER_SUFFIX` overlaps K5's router/top-k/shexp-gate scope) — materially more invasive than
the pipelining port (new CUDA kernels vs. host-side graph restructuring) and requires the same
build/measure/bit-identity cycle per fusion. Given the pipelining port's negative result and the
remaining session budget, K3/K4/K5 are left for a follow-up session with their own time budget.
