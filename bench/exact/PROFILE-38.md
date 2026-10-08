# Where the 38.4 ms/token goes — exact-stack decode, measured decomposition

Box: lucebox4 (gfx1151 Strix Halo), `~/qwen4exp-exact-stack/driver-stack`, exact-stack
gates per `run_stack.sh` (LUCE_QWEN_GRAPH_SH, HC_DOWN_INJECT, SHARED_OVERLAP,
QSA_CONT_ELISION, EXACT_ROUTER_SUFFIX, EXPERT_ROW_WARPS=8, HOST_PREFILL_GUARDS,
HC_SCALE_SILU, HC_LO_Q8, PRODUCER_Q8, HC_UPMIX_ROW8, SHARED_EPILOGUE, GDN_AB_EXACT,
GPU_ARGMAX/MEASURE_GPU_ARGMAX, MEASURE_FOLLOW=follow.ids). Model
`Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`, prompt.ids (7235 tokens) +
follow.ids from `/home/duster/pr823-release-20261007/artifacts`.

**Correction to the task's planned invocation**: `driver-stack <gguf> prompt.ids 256 2`
is not a valid arg combination — `driver_shared_epilogue.cpp:main()` requires
`(n=128,reps=2)` = "golden" (needs `MEASURE_LOGITS`/`MEASURE_STATE`, not set here) or
`(n=256,reps=8)` = "native" (the correct steady-state decode-census mode, 2 warm runs
+ 8 measured runs, mode alternating per schedule `{0,1,0,1,1,0,1,0}`) or `(n=64,reps=1)`
= "census". Anything else returns exit code 2 with zero output. Used `256 8` (native),
matching `run_stack.sh`'s own invocation.

## Reproduction: un-profiled wall, 8 measured reps

```
rep=0 mode=0  decode_ms=9867.73   ms_token=38.546
rep=1 mode=1  decode_ms=9851.37   ms_token=38.482   <- analyzed window below
rep=2 mode=0  decode_ms=9858.40   ms_token=38.509
rep=3 mode=1  decode_ms=9861.70   ms_token=38.522
rep=4 mode=1  decode_ms=9855.78   ms_token=38.499
rep=5 mode=0  decode_ms=9863.84   ms_token=38.531
rep=6 mode=1  decode_ms=9860.92   ms_token=38.519
rep=7 mode=0  decode_ms=9873.14   ms_token=38.567
```

Tight (38.48–38.57 ms/token, σ≈0.03 ms), confirming the reported 38.41 ms/token figure
and that this is deterministic replay (captured HIP graphs), not noise-bound.

## Profiled run (rocprofv3 --kernel-trace, csv, rocm 7.2.2 matching the driver's build)

```
rep=0 mode=0  ms_token=40.570
rep=1 mode=1  ms_token=40.736   <- analyzed window below (+2.25 ms/token profiler overhead, +5.9%)
rep=2 mode=0  ms_token=41.830
rep=3 mode=1  ms_token=41.674
rep=4 mode=1  ms_token=41.670
rep=5 mode=0  ms_token=40.488
rep=6 mode=1  ms_token=40.393
rep=7 mode=0  ms_token=40.486
```

Profiler overhead is modest (+5–9%) and fairly uniform; the un-profiled 38.4–38.6 ms/token
is the number to trust for "where it goes", the profiled run is used only to attribute
that time to kernels/gaps.

## Decode window analyzed

rep=1, mode=1 (4th forward() call: 2 warm runs, then rep0 mode0, then rep1 mode1),
tokens 16–255 of 256 (first 16 excluded as warm-up), 240 steady-state T=1 steps.
Token boundaries = consecutive `argmax_f32` kernel completions (exactly 1 per token;
2560 argmax launches total over 10 forward() calls × 256 tokens = exact match, confirms
one argmax/token and validates the boundary method). Kernels are attributed to the
token whose argmax they fall before (chronological row range between consecutive argmax
dispatches — single profiling queue stream per GPU engine, dispatch order == issue order).

## 1. Wall / busy / idle / launches per token (profiled, 240-token window)

| metric | value |
|---|---|
| wall | 40.609 ms/token |
| GPU busy (union of all kernel intervals, de-overlapped across the 4 concurrent streams SHARED_OVERLAP uses) | 33.047 ms/token (81.4%) |
| idle (wall − busy) | 7.562 ms/token (18.6%) |
| kernel launches | **1871.0/token** (exact, matches the task's stated figure) |
| Strata launches (window+commit+memcpy from the census log) | 1468 per spec-decode iteration (2 draft-verify tokens; not directly 1:1 with our strict-autoregressive 1 token — treat as a rough proxy, not an apples-to-apples count) |

## 2. Weight-streaming vs everything-else

Classification: `mul_mat_vec_q<*>`, `mul_mat_vec_f<*>` → weight-streaming (GEMV reads of
quantized weight rows). Everything else — **including the hc_\* hyper-connection fusion
kernels** (per task spec) — → "other".

| category | us/token (raw kernel-time sum, streams can overlap so this is GPU-seconds spent, not critical-path) | launches/token |
|---|---|---|
| weight_stream | 28.387 ms/token | 475.0 |
| other | 9.772 ms/token | 1396.0 |

Bandwidth floor: 6.33 GB/token streamed ÷ 242 GB/s measured LPDDR5X ≈ **26.2 ms/token**.

**weight-stream achieved 28.387 ms/token vs the 26.2 ms floor → 92.3% bandwidth
efficiency** (26.2/28.387). The GEMV kernels that actually move the weight bytes are
already close to the wall — this is *not* the big recoverable bucket.

Per-kernel bytes/GB-s were not computed (no gguf-py tensor-shape mapping was done in the
time available — the per-kernel table below gives launches/µs which is sufficient to see
where the time concentrates; a follow-up could join against `gguf-py` tensor shapes by
launch order to get true achieved-GB/s per GEMV call).

## 3. Top-15 kernels by µs/token (240-token steady-state window)

| launches/tok | µs/tok | mean µs/launch | category | kernel |
|---|---|---|---|---|
| 210.0 | 16712.9 | 79.59 | weight_stream | `mul_mat_vec_q<Q8_0,...>` (type 8) |
| 47.0 | 5594.9 | 119.04 | weight_stream | `mul_mat_vec_q<Q4_K,...>` (type 12) |
| 43.0 | 2649.4 | 61.61 | weight_stream | `mul_mat_vec_q<Q5_1,...>` (type 7) |
| 95.0 | 1995.9 | 21.01 | other | `hc_down_inject_mixed` |
| 96.0 | 1991.1 | 20.74 | other | `hc_upmix_row8_exact<false>` |
| 36.0 | 1254.4 | 34.84 | other | `gated_delta_net_cuda_grouped_cols` (GDN) |
| 48.0 | 1200.0 | 25.00 | weight_stream | `mul_mat_vec_q<Q8_0,...>` (type 8, 2nd variant) |
| 72.0 | 1077.2 | 14.96 | weight_stream | `mul_mat_vec_f<bf16,...>` |
| 95.0 | 952.8 | 10.03 | other | `hc_combine_norm_f32_b256` |
| 49.0 | 590.9 | 12.06 | weight_stream | `mul_mat_vec_f<float,...>` |
| 256.0 | 534.1 | 2.09 | other | `quantize_q8_1` |
| 5.0 | 422.1 | 84.42 | weight_stream | `mul_mat_vec_q<Q8_0,...>` (type 8, 3rd variant) |
| 12.0 | 348.9 | 29.08 | other | `qsa_decode_wmma_partial` (full-attention layers) |
| 48.0 | 302.6 | 6.31 | other | `k_argsort_f32_i32` |
| 36.0 | 247.1 | 6.86 | other | `gdn_ab_exact_f32` |

48 recurring launches/token on many of these = 48 decoder layers (36 GDN + 12 full-attention,
confirmed by `qsa_decode_*`=12/tok and `gdn_*`=36/tok). The Q8_0 GEMV (type 8) alone is
**16.7 ms/token across 210 launches**, i.e. it is the single largest line item and already
counted in the weight-streaming bucket.

## 4. Gap distribution (idle time between merged busy intervals, 240-token window)

| bucket | count/token | µs/token | % of total idle |
|---|---|---|---|
| <2 µs | 130.7 | 166.7 | 2.2% |
| 2–5 µs | 288.9 | 912.2 | 12.1% |
| 5–20 µs | 655.3 | 4916.9 | **65.0%** |
| >20 µs | 7.0 | 1566.6 | 20.7% |

The 5–20 µs bucket dominates idle and tracks directly with launch count (655 small gaps
per token against 1871 launches/token — roughly one small host-dispatch gap per 2.9
kernel launches). **This is launch-overhead idle, not memory-wait idle.**

**Caveat on the >20 µs bucket / "biggest single gaps"**: the top individual gaps in this
240-token window were 20.18 ms (token 191), 10.18 ms (token 190), 7.03 ms (token 173),
5.84 ms (token 174), 5.32 ms (token 138) — these are 100–500× the typical gap and occur
only a handful of times in 240 steps. The un-profiled run's per-token wall time has σ≈0.03
ms with **no** multi-ms stalls visible at the aggregate level, so these look like
rocprofv3 buffer-flush/ring-sync artifacts rather than real steady-state GPU idle. Average
magnitude of an ">20µs" event, excluding this handful of outliers, is far smaller
(375996 µs total / 1689 events ≈ 223 µs typical). Treat the >20µs bucket's *count* as
real (real host round-trips do exist) but do not take the extreme multi-ms tail at face
value without a longer/un-profiled timing cross-check per token (not done here — would
need a lower-overhead timestamp mechanism, e.g. HIP events inserted by the driver itself).

## 5. "Where the ms go" — summary

- **Wall (un-profiled, trustworthy number)**: 38.4–38.6 ms/token.
- **Floor**: 26.2 ms/token (6.33 GB/token ÷ 242 GB/s).
- **Gap to floor**: ~12.2–12.4 ms/token to explain.
- Profiled decomposition (same window, scale by ~0.95 to back out profiler overhead for
  absolute ms, proportions should hold): busy 81.4% / idle 18.6% of wall; weight-stream
  kernels themselves already run at 92.3% of floor bandwidth.

### Three largest recoverable buckets (ms/token), with evidence

1. **Launch-overhead idle: ~4.9 ms/token** — the 5–20 µs gap bucket (655 gaps/token,
   65% of all idle time). Directly tied to issuing **1871 launches/token** vs Strata's
   census of 1468/iteration; every extra kernel launch costs a few µs of host-dispatch
   gap even when it does real work. Evidence: §4 gap histogram, §1 launch count.
2. **Unfused hyper-connection (hc_\*) kernel family: ~5.1 ms/token** —
   `hc_down_inject_mixed` (95/tok, 2.00 ms) + `hc_upmix_row8_exact` (96/tok, 1.99 ms) +
   `hc_combine_norm_f32_b256` (95/tok, 0.95 ms) + `quantize_hc_lo_q8_1` (97/tok, 0.17 ms)
   = 383 launches/token, 5.11 ms/token, none of it weight-streaming by the task's own
   classification (hc_* → "other"). This is the "exact-stack" emulation of Strata's
   hyper-connection path and is the most concrete fusion/elimination target — 4 kernels ×
   ~95 calls/token (≈ once per decoder layer) that could plausibly collapse to 1–2 fused
   kernels. Evidence: §3 top-15 table.
3. **Weight-stream bandwidth gap: ~2.2 ms/token** — 28.387 ms/token actual vs 26.2 ms/token
   floor (92.3% efficiency). Smaller than the other two buckets but real: the Q8_0 GEMV
   (type 8, 210+48+5=263 launches/token, 18.3 ms/token combined) is the single biggest line
   item in the whole trace and is already fairly efficient, so gains here are capped at
   ~7.7% of weight-stream time. Evidence: §2, §3.

Sum of the three buckets ≈ 4.9 + 5.1 + 2.2 = **12.2 ms/token**, matching the 12.2–12.4 ms/token
gap between measured wall (38.4–38.6) and the bandwidth floor (26.2) almost exactly — i.e.
these three buckets plausibly account for essentially all of the non-floor time, with
launch-overhead idle and the unfused hc_* family roughly tied as the two largest
opportunities (each ~5 ms/token), and bandwidth-efficiency the smallest (~2 ms/token).

## Files

- `rep1_mode1.json` (this directory, copied from `~/qwen4exp-exact-stack/prof/rep1_mode1.json`
  on lucebox4) — full per-kernel table, gap histogram, biggest-gaps list, per-token wall
  series for the analyzed window.
- `analyze.py` (this directory) — the decomposition script, also live at
  `~/qwen4exp-exact-stack/prof/analyze.py` on lucebox4. The raw kernel-trace CSV it reads
  (`~/qwen4exp-exact-stack/prof/kt/run_kernel_trace.csv`, 1.7 GB) stays on the box only
  (not copied locally, not /tmp).

## Coordinator note on accounting (read before using the numbers)

Section 2's category times are raw kernel-time sums (28.39 + 9.77 = 38.16 ms). Their total is larger than the
de-overlapped busy time of 33.05 ms because SHARED_OVERLAP runs kernels on concurrent streams.

- **The 92% weight-stream efficiency is unreliable.** Overlapping kernels share bandwidth, so the raw duration of
  each streaming kernel is inflated.
- **The three buckets don't add up.** Their sum (4.9 + 5.1 + 2.2) is not a critical-path budget.
- **What holds:**
  - The 1871 launches per token.
  - The idle time. In the un-profiled run it is about 38.5 − 33.0 ≈ 5.5 ms/token, assuming the profiler leaves
    busy time unchanged. Most of it sits in 5–20 µs gaps between graph-replayed kernels.
  - Inside busy time, the hc_* family's launch count (383 per token) and its time share.
