# P60A — K3 control-plane census

## Verdict

**MEASURED HARD-STOP FOR THE LAYERPLAN / GPU-DESCRIPTOR LANE.**

P60A measured the host-side route publication, validation, behavioral-policy,
compact-map/job construction and P42 descriptor publication boundary across
the frozen Lucebox4 P55 broad-12 workload. The candidate measured service-time
bounds are only **2.795274–5.933539 ms per transition**. Both ends are
far below the registered **38.521937 ms/transition** hard floor, so building a
persistent `LayerPlan`, GPU-resident `CompactJob[]`, or descriptor kernel cannot
carry the next decode gate.

The dominant measured boundary is instead the separately reported GPU
dependency: **159.393603 ms/decode transition** and **160.043336 ms/prefill
position**. This includes necessary KDA/router/shared preparation and must not
be counted as candidate host control-plane service. The next bounded work is to
split and measure that dependency, alongside the expert critical path. The P58
exact multi-token verification comparison remains open.

## Registered interpretation

The diagnostic inserted synchronization points to separate otherwise
overlapped work. Those syncs contaminate recurrent-route and P42 publication
intervals. The suite therefore reports a bracket rather than treating all
instrumented time as candidate service:

```text
lower = MLA route publication + validation + descriptor control
upper = lower + recurrent route publication + P42 publication
dependency = required GPU completion wait, reported separately
```

Lower and upper are discriminator service-time bounds, not an observed or
guaranteed realizable speedup: replacement work remains, and the upper includes
inserted-sync contamination. At most, the upper is a zero-replacement-cost
optimistic ceiling—not a throughput projection.

The decode decision is made only after aggregating all 12 prompts and 129 true
autoregressive transitions:

- `lower >= 50.0 ms/position`: continue safely;
- `upper < 38.521937 ms/position`: hard-stop;
- otherwise: inconclusive.

Prefill is a separate question. Its materiality threshold is
**84.722396133 ms/position**, the exact saving required to improve the frozen
P55 prefill time by 1.16x. No decode label is applied to prefill.

## Exact census

| phase / component | aggregate ns | ms per position | included in lower | included in upper |
|---|---:|---:|:---:|:---:|
| prefill GPU dependency | 88,343,921,690 | **160.043336** | no | no |
| prefill recurrent route publication | 885,171,629 | 1.603572 | no | yes |
| prefill MLA route publication | 309,835,506 | 0.561296 | yes | yes |
| prefill validation | 5,341,790 | 0.009677 | yes | yes |
| prefill descriptor control | 978,212,569 | 1.772124 | yes | yes |
| prefill P42 publication | 900,232,295 | 1.630856 | no | yes |
| **prefill lower / upper** | **1,293,389,865 / 3,078,793,789** | **2.343098 / 5.577525** | — | — |
| decode GPU dependency | 20,561,774,814 | **159.393603** | no | no |
| decode recurrent route publication | 207,646,253 | 1.609661 | no | yes |
| decode MLA route publication | 72,407,116 | 0.561295 | yes | yes |
| decode validation | 1,112,721 | 0.008626 | yes | yes |
| decode descriptor control | 287,070,545 | 2.225353 | yes | yes |
| decode P42 publication | 197,189,839 | 1.528603 | no | yes |
| **decode lower / upper** | **360,590,382 / 765,426,474** | **2.795274 / 5.933539** | — | — |

The prefill upper bound is also far below the 84.722396133-ms materiality
threshold. P60A is therefore `below-material` for prefill and `hard-stop` for
decode.

## Frozen workload identities

The analyzer independently checked every raw record rather than trusting its
`counter-equations=pass` label. Twelve paired P56/P60 records were captured per
phase, for 24 phase records total.

| identity | prefill | decode |
|---|---:|---:|
| positions | 552 | 129 |
| recurrent-layer calls (`positions * 68`) | 37,536 | 8,772 |
| MLA-layer calls (`positions * 24`) | 13,248 | 3,096 |
| routed-layer calls (`positions * 92`) | 50,784 | 11,868 |
| compact jobs / attempted / completed / async jobs / graph enqueues / expert D2Ds | 223,579 | 51,429 |
| compact H2D calls | 907,564 | 208,812 |
| logical provider bytes | 724,784,406,528 | 167,799,668,736 |
| physical direct-read bytes | 291,245,891,584 | 44,258,549,760 |

Fallbacks, invalid compact jobs and queue aborts are all zero. Across both
phases the lifecycle totals are 275,008 P41 attempts/completions, 62,652 P45
begins/flushes and P42 joins/outputs, 1,116,376 compact H2D calls, 275,008 graph
enqueues and expert D2Ds, and 335,504,441,344 physical direct-read bytes.

## Qualification and identity limits

The final budget preflight required exactly 92 H23 entries, each at depth 24.
The capped Release build passed eight serialized model-free gate invocations:
provider, ordered join, MoE stream and sparse-K on each of `gfx1201` and
`gfx1151`. MoE passed 2/2 cases on both devices and sparse-K passed 68/68 on
both. The production suite then completed once, as preregistered.

All prompt tokens, output tokens and output text match the frozen P55 reference.
Logical traffic is byte-identical with SHA-256
`54b76f99439d90ecb35d033c12dd1d1f219e30007248aaf77f913fc69d76f383`.
Fresh logits and state hashes were deliberately **not** collected: the reviewed
P60 mode rejects the incompatible H16/draft/P20-I/O/route-stat/perf-logit/state
traces that would perturb or confound this census. This result therefore makes
no fresh logits-byte-identity or state-hash claim; it relies on unchanged
arithmetic plus token/text/traffic identity.

The added synchronization makes wall throughput diagnostic only. The measured
wall was 373.614163 s for prefill (1.477460 positions/s) and 78.709663 s for
decode (1.638935 true AR/s). It is not a replacement P55 performance baseline
and does not establish a regression.

## Evidence and resource envelope

The remote evidence root is
`/home/duster/kimi-k3-deploy/p60a-control-census-20260820`. The analysis,
aggregate, telemetry, stderr, traffic, suite-manifest and artifact-manifest
SHA-256 values are recorded in
`results/k3_p60a_control_plane_census.json`. The run consumed zero process swap;
peak RSS was 21,520,712 KiB, GPU1 peak allocation was 61,295.922 MiB, and sampled
GPU1 energy was 31,713.706 J.

The final measured P60 source snapshot is bound by the eight individual source
hashes in the result JSON. The tracked-diff snapshot SHA-256 is
`5c9f247b097427c5b86a474f916cd0de22150ca81fafee1b2135826d71418f73`.

## Source removal

P60 was a strict default-off census, not a production feature. Closure removes
all temporary instrumentation and restores every touched existing source/test
file exactly to HEAD. Removed scope:

- 311 added and 15 deleted raw production lines (the 15 HEAD lines restored);
- 74 added lines in the existing provider test; and
- 408 lines across the temporary preregistration, analyzer and analyzer test.

That is 793 experimental added lines removed, 15 HEAD lines restored, and no
P60 census symbol, parser, test or analyzer retained. Only this report, its
machine-readable result and roadmap revision 71 remain.

## Decision

Do not implement `LayerPlan`, host-visible GPU descriptor publication or a P60
descriptor kernel. The candidate measured service-time interval is bounded by
the contaminated 5.933539-ms upper, a zero-replacement-cost optimistic ceiling,
while 38.521937 ms is the hard floor for the next gate. No realizable speedup of
5.933539 ms is claimed.

Next, split the **159.393603-ms decode GPU dependency** into KDA, router and
shared-expert components using device-event timing that does not add per-layer
global synchronization. In parallel at the planning level, measure the expert
critical path against the P55/P60 identities. P58's exact recurrent-microtile /
MoE-macro seam remains available for a matched multi-token verification
comparison after those rooflines identify the larger prize.
