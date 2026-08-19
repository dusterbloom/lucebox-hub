# P45a — asynchronous compact expert queue

## Verdict

**BYTE-EXACT MECHANISM GO; COMPARATIVE PERFORMANCE INCONCLUSIVE; DEFAULT-OFF
EXPERIMENTAL IMPLEMENTATION RETAINED.**

P45a replaced each P41 compact expert's five blocking tensor uploads and
blocking graph submission with pinned per-route staging, five asynchronous
uploads, an asynchronous graph submission and an immediate same-stream output
copy. The existing P42 ordered-join barrier remained the only layer-level
synchronization. A Release-active HIP sentinel exercised 16 mixed-depth jobs,
persistent-entry reuse, exact queued-versus-blocking outputs and abort recovery
on both Lucebox4 GPUs.

The mechanism works, but it does not earn a standalone promotion claim. The
P45-on arm reaches 1.473706970 true AR/s, above the official P42c 1.466218848/s
fact but below the deliberately ambitious 1.562500/s gate. Its two off controls
differ by 6.79216%, so the apparent uplift over their bracket is not a valid
comparative speed claim. The exact default-off queue is retained because the
23.22% expert-stage reduction is a useful composable boundary for persistent
routed preparation and grouped expert execution.

## HIP qualification

The measured runner is
`65f1a69c1b0567d842d2f39d3d5bd003ee0adfa196ccd02a8a25c9cf3d6abe64`.
Before the model fact, the P45 provider sentinel passes on gfx1201 and gfx1151;
ordered join passes on both; the common MoE tests pass 2/2 on both; and sparse-K
passes 68/68 on both. Local CUDA execution is not evidence: the WSL build is a
compile/link check and skips because no CUDA device is visible.

The qualified source boundary is frozen at `/tmp/p45a_hip_qualified`; its
four-file manifest SHA-256 is
`9ea5583854debe4213c5106119b7ca79b0206384e47c3c03f69d65d842291b90`.
This remains a temporary snapshot of the retained default-off feature.

## Matched short fact

The roots are
`/home/duster/kimi-k3-deploy/p45a-final-aba-20260819/{a1-off,b-on,a2-off}`.
Each arm produces the same Tokyo continuation, byte-identical full logits and
byte-identical logical traffic.

| Arm | Decode | True AR | Total | Routed prep | Experts | Join |
|---|---:|---:|---:|---:|---:|---:|
| A1 off | 6.072137912 s | 1.317493133/s | 758.773 ms | 272.948 ms | 451.149 ms | 20.904 ms |
| B async | 5.428487591 s | **1.473706970/s** | 678.278 ms | 275.829 ms | 366.499 ms | 22.078 ms |
| A2 off | 6.499066068 s | 1.230946095/s | 812.140 ms | 273.853 ms | 503.496 ms | 20.871 ms |

The off-seconds bracket is 6.285601990 seconds, or 1.272750011/s. B appears
15.79% faster than that bracket, but the 6.79% control spread exceeds any
credible small-fact attribution limit. The independent absolute gate blocks
standalone promotion: B is 5.68% below 1.5625/s and only 0.51% above the
official P42c 1.466218848/s result. The latter is directionally positive but
too small and non-adjacent to call a new record.

The useful measured signal is structural. B reduces the final-eight expert
stage 23.22%, from a 477.322-ms off bracket to 366.499 ms, while routed prep is
unchanged/slightly worse and join rises 1.19 ms. Queue counters are exact:
3,864 begins and flushes, 17,917 jobs and graph enqueues, 89,585 H2D calls,
56,234,092,400 H2D bytes, zero abort synchronizations and maximum inflight 16.
P41 completes 17,917/17,917 with zero fallback/invalid; P42 publishes 17,917
expert rows and performs 3,864 joins/output copies.

## Exactness and resource facts

- Logits SHA-256:
  `cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`.
- Traffic SHA-256:
  `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`.
- Every arm reports 55,976,374,272 payload H2D bytes,
  257,718,128 metadata H2D bytes and 24,106,500,096 physical direct-read
  bytes. P45 changes scheduling, not logical work or storage traffic.
- Peak process swap is zero in all three arms. Peak GPU1 memory is
  61,180.71 / 61,200.76 / 61,200.81 MiB for A1/B/A2.

## Size and retention boundary

P45 adds +291/−29 raw production lines, +71 existing-test lines, a 193-line
private sentinel and +16 CMake lines. Tokei reports +252 pure production,
+257 pure tests and +16 pure CMake: **+525 pure lines total**. This is larger
than the preferred standalone slice, so P45 remains default-off and must earn
deletions when composed into the persistent routed-preparation boundary.

The P42d fallback boundary remains identified by:

- provider:
  `fbc8f1fc7a149c2eefdeaae09157dfbb9b31b3f450c3de1644fb975b5140c3e1`;
- provider test:
  `b260587bed3f71664900064bf173027b93cf687bf9a03445b4e8874b116b9e3d`;
- CMake:
  `2f31094edbd005e5b90e465cc77b8b8586d49f787629e76a2953906f3763d184`;
- P42d runner:
  `1c9a203030ce5650034196cadb9b4c4d83413e20ab4048093a077600e5c5a2a2`.

The retained P45 source matches the frozen qualified snapshot. Local builds
pass; on Lucebox4 the P45 sentinel, ordered join and MoE pass on both HIP GPUs,
and sparse-K remains 68/68 exact on both.

## Decision

P45 proves that route-local synchronization is removable and that doing so
materially lowers the expert stage. It also proves this boundary alone is not
the ≥2/s solution. Retain it default-off as the submission primitive to compose
with stable device boundaries, persistent one-token routed-preparation graphs
and later grouped expert execution. Balanced P43b remains separate and must use
concurrent cost-aware ownership rather than repeating all-GPU0 P43a.
