# LUCE_QWEN_HC_LO_FUSED timing: A3 (dense Q6_K), 6 fresh processes, interleaved OFF/ON/OFF/ON/OFF/ON

Post drop_caches + compact_memory. Correctness already verified separately (see
commit a152f4fd): Q8 and A3, gate ON vs OFF, byte-identical `measure_tokens`,
armed==matched==launched, no aborts. This file is timing only.

## Setup (parity contract)

- Model: `/home/duster/models/qwen4exp-dense-q6k-a3/Qwen3.8-Flash-Next-dense-Q6K-a3-00001-of-00004.gguf`
  (same shard, same quant, both arms).
- Build: `driver-stack` md5 `9a5e22fdbd4eb2fac5abe67136e107ee`, `libggml-hip.so.0.9.11`
  md5 `8e0211a7c9076bdd48073290e0146170` -- unchanged from the correctness-fix build,
  re-checked immediately before this run, identical on both arms.
- Env: full 38.41 env (`LUCE_QWEN_GRAPH_SH=1 LUCE_QWEN_HC_DOWN_INJECT=1
  LUCE_QWEN_SHARED_OVERLAP=1 LUCE_QWEN_QSA_CONT_ELISION=1
  LUCE_QWEN_EXACT_ROUTER_SUFFIX=1 LUCE_QWEN_EXPERT_ROW_WARPS=8
  LUCE_HOST_PREFILL_GUARDS=1 LUCE_QWEN_HC_SCALE_SILU=1 LUCE_QWEN_HC_LO_Q8=1
  LUCE_QWEN_PRODUCER_Q8=1 LUCE_QWEN_HC_UPMIX_ROW8=1 LUCE_QWEN_SHARED_EPILOGUE=1
  LUCE_QWEN_GDN_AB_EXACT=1 LUCE_GPU_ARGMAX=1 MEASURE_GPU_ARGMAX=1`), only
  `LUCE_QWEN_HC_LO_FUSED` toggled 0/1.
- Native mode, `prompt.ids 256 8` (256 decode tokens, 8 internal reps/process),
  `MEASURE_FOLLOW=follow.ids`.
- 6 fresh processes, each through `gpu_exec.sh` (single-tenant GPU lock), strict
  alternation OFF, ON, OFF, ON, OFF, ON. No other process touched the GPU for the
  duration (no builds, no other models).
- Same seed/sampling as the harness's own fixed-prompt/greedy-argmax decode
  (deterministic, no temperature convergence needed).
- Each process does 8 internal reps alternating `mode=0`/`mode=1` (an orthogonal
  sealed-graph mode flag, unrelated to the fuse gate); only `mode=1` reps are
  counted below per the coordinator's spec -- 4 mode=1 reps/process x 3
  processes/arm = 12 values/arm.

## Results (ms/token, decode)

### OFF (gate disabled) -- 12 mode=1 values

```
35.541326 35.659267 35.593858 35.625012   (process 1, median 35.609435)
37.624568 35.762486 35.759565 35.780862   (process 2, median 35.771674)
35.641053 36.321414 35.671296 35.736756   (process 3, median 35.704026)
```

- median = **35.704026**
- mean   = 35.893122
- best   = 35.541326
- max    = 37.624568

### ON (gate enabled) -- 12 mode=1 values

```
37.312738 37.309144 37.316420 35.506728   (process 1, median 37.310941)
35.604641 35.531472 35.606966 35.643103   (process 2, median 35.605804)
35.635419 35.528336 35.633904 35.630634   (process 3, median 35.632269)
```

- median = **35.632269**
- mean   = 36.021625
- best   = 35.506728
- max    = 37.316420

### ON - OFF median delta

**-0.072 ms/token** (ON marginally faster by median, but inside process-to-process
noise -- see below).

## Per-process medians

| process | arm | median ms/token |
|---|---|---|
| 1 | OFF | 35.609435 |
| 2 | OFF | 35.771674 |
| 3 | OFF | 35.704026 |
| 1 | ON  | 37.310941 |
| 2 | ON  | 35.605804 |
| 3 | ON  | 35.632269 |

Two processes (OFF process 2, ON process 1) each have one internal rep (or, for
ON process 1, three of four reps) elevated to ~37.3-37.6 ms/token while the
other reps in the same process sit at ~35.5-35.8. Both elevated clusters appear
as the *first* mode=1 readings a cold process produces after its own graph
capture, i.e. a per-process warm-up/recapture artifact, not something that
tracks the fuse gate -- it occurs on one OFF process and one ON process
symmetrically. Excluding those two anomalous processes, OFF medians cluster at
35.61-35.70 and ON medians at 35.61-35.63: no separable gap.

## measure_tokens identity

All 6 processes (3 OFF + 3 ON), `measure_tokens` across all 8 reps each,
stripped of the `rep=`/`mode=` prefix, hash to the same md5
(`0d42e23067d75880dc05887827d40f7a`) -- byte-identical decode output in every
process, both arms.

## OFF median vs. prior A3 figure

Prior A3 figure: **35.65** ms/token. This run's OFF median: **35.704026**
ms/token -- a 0.15% difference, within run-to-run noise, no regression.

## Verdict

No measurable launch-count-driven speedup survives process-to-process noise at
this sample size: median delta -0.072 ms/token is smaller than the ~0.3-1.9
ms/token spread already present between same-arm processes. This is consistent
with FUSION-CANDIDATES.md's own estimate for this candidate (0.17-0.40 ms/tok,
own-work only) sitting below this hardware's process-level noise floor.
Correctness (byte-identical output, zero aborts, counters matching) is the
deliverable this change banks; a decode-latency win from this specific fusion
is not established by this measurement.
