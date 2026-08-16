# H23 — adaptive authoritative-byte frontier

STATUS: MEASURED BYTE GEOMETRY / PROJECTED SLAB POLICIES / QUALITY OPEN

H23 asks how far the frozen progressive-slab representation can reduce real
authoritative expert traffic when bytes are allocated by measured behavioral
sensitivity rather than uniformly.  This document does **not** claim that a
projected policy preserves K3 quality.

## Inputs and scientific boundary

The planner uses:

- the frozen H22 92-layer atlas, where each layer has one **measured** isolated
  terminal-KL point at 96 slabs;
- the registered 2,048-token, whole-sequence-split captures and frozen slab
  residual statistics to extend the H22 curve to budgets
  `24,48,72,96,120,144,168,192`;
- the real 32-position P27 trace for per-layer mixed-qtype slab sizes and the
  exact fallback decisions actually encountered by the runtime.

For budgets other than 96, behavioral damage remains **PROJECTED** as

```text
KL_layer(96) * (omitted_proxy_layer(b) / omitted_proxy_layer(96))^2.
```

An exact dynamic program minimizes this projected additive cost under each
byte target.  Byte costs are measured from P27 and conservatively binned to one
MiB.  No end-to-end candidate result is used to fit the table.

The route-count axis is not silently mixed into this optimizer.  Existing K3
evidence contains only a uniform all-layer four-route pilot, not a per-layer
terminal-damage atlas at 4/6/8/12 routes.  Activation or local-cosine scores
cannot substitute for that missing behavioral evidence.

## What the trace says

| Quantity | P27 32-position trace |
|---|---:|
| Exact routed geometry | 9.0696 GiB / position |
| Exact-fallback-only floor | 1.2556 GiB / position |
| Fallback fraction of exact bytes | 13.84% |
| Minimum registered 24-slab policy | 2.3893 GiB / position |

The fallback floor is decisive.  With the current 2,048-token calibration,
the `1.2 GiB` moonshot is below the bytes consumed by exact fallbacks **before
one calibrated slab is retained**.  The `1.8 GiB` target is above that floor
but below the smallest registered 24-slab policy.  Neither is currently a
runtime policy.

Layer 12 provides a measured reason not to treat this floor as permanent.
Using the same `>=8` calibration-hit rule, its held-out fallback-route rate
fell from `302/6640 = 4.55%` with 2,048 tokens to
`119/28800 = 0.413%` with 10,000 tokens.  This is an 11x reduction on one
layer, not yet an all-layer projection.  Broader calibration coverage is the
cleanest path to lowering the fallback floor without weakening safety.

## Projected policies

The names below identify requested steering points, not quality verdicts.

| Policy target | Trace GiB / position | Exact fraction | Average slabs | Projected additive cost | Status |
|---|---:|---:|---:|---:|---|
| H22 average96 reference | 5.7599 | 63.5% | 96.0 | 0.2296 | one official-chat sanity pass; broad quality open |
| SAFE / 4.0 GiB | 3.9663 | 43.7% | 57.65 | 0.7608 | PROJECTED; requires end-to-end quality |
| MEDIUM / 2.5 GiB | 2.4533 | 27.0% | 25.30 | 2.4230 | PROJECTED; high risk |
| AGGRESSIVE / 1.8 GiB | — | — | — | — | INFEASIBLE with registered budgets/fallbacks |
| MOONSHOT / 1.2 GiB | — | — | — | — | INFEASIBLE with registered budgets/fallbacks |

The 4-GiB table spends 24/48/72/96/120 slabs on
24/31/18/14/5 layers respectively.  Its projected cost is 3.31x the H22
average96 cost, so `SAFE` must not be read as a quality claim.  The 2.5-GiB
table puts 87 of 92 layers at the 24-slab minimum and has 10.55x the H22
projected cost.  It is useful as a falsification point, not as the first
production candidate.

## Route axis: measured evidence and missing evidence

The all-layer H21 pilot measured:

- four complete routes: about 3.35 GiB/position, shared-prefix mean KL 0.128,
  coherent but token-divergent generation;
- four routes with six slabs each: about 2.13 GiB/position, mean KL 1.436 and a
  failed uniform gate.

This proves route sparsity and slab sparsity are not interchangeable.  It does
not price which individual layers tolerate 4, 6, 8, or 12 complete routes.
Until an isolated terminal-damage route atlas exists, a joint route/slab
knapsack would be false precision.

## Smallest valid next sequence

1. **Quality gate the 4-GiB slab-only table first.**  Run the frozen official
   chat template on a small suite of prompts that native K3 has already passed.
   Score retained native-task success, degeneration, generated decisions,
   aligned-history KL, exact fallback rate, and actual logical bytes.  Do not
   run the 2.5-GiB table unless 4 GiB leaves a useful quality margin.
2. **Raise calibration coverage before chasing 1.2 GiB.**  Repeating the
   all-layer export at 10K tokens is scientifically earned because exact
   fallback is already a measured 1.256-GiB floor.  Recompute the floor before
   changing any fallback semantics.
3. **Measure the missing route axis narrowly.**  Use isolated terminal KL—not
   cosine—for predeclared representative sensitive/tolerant layers at 4/8/12
   complete routes.  Expand to all layers only if route choices beat slab
   choices at equal measured bytes.  This is the minimum evidence needed for a
   legitimate joint optimizer.
4. **Keep systems and verification work parallel.**  P27 made delivery much
   cheaper, but 3.97 GiB remains too much for 4 token/s on one SN850X.  Oracle
   overlap and multi-token verification can improve the verifier independently
   while H23 discovers whether logical bytes can fall further.

## Quality substrate audit

The official-template 12-task fixture exists and covers factual, code,
arithmetic, reasoning, translation, writing, science, grammar, and data
structures.  The prior registered all-task adaptive run stopped after three
prompts and has no complete manifest.  The only completed official-template
all-layer H22 evidence is the Tokyo sanity prompt (`4/4` generated decisions,
identical output).  Therefore broad quality remains an explicit data gap; the
partial three-prompt files must not be promoted to a suite result.

## Reproduction

```bash
python3 scripts/plan_kimi_h23_adaptive_frontier.py \
  --atlas results/kimi_h22_layer_behavior_atlas.json \
  --capture-root /mnt/kimi-k3/captures/kimi-h18-all-layer-2048-chunk8 \
  --fit-root /mnt/kimi-k3/fit-state/kimi-h18-slab-calibration-2048 \
  --io-trace /mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816/io_trace.tsv \
  --traffic /mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816/traffic.tsv \
  --output-json results/h23_projected_byte_frontier.json \
  --output-csv results/h23_layer_options.csv \
  --policy-directory results/h23_policies
```

Machine-readable outputs record input hashes, measured byte costs, projected
behavioral costs, policy-table hashes, and every hard data gap.
