# H23 — adaptive authoritative-byte frontier

STATUS: MEASURED BYTE GEOMETRY / SMALL 4-GiB QUALITY GATE PASSED / BROAD QUALITY OPEN

H23 asks how far the frozen progressive-slab representation can reduce real
authoritative expert traffic when bytes are allocated by measured behavioral
sensitivity rather than uniformly.  A projected policy is not treated as
quality-preserving until it passes an end-to-end gate; only the 4-GiB policy has
passed the small gate documented below.

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
| SAFE / 4.0 GiB | 3.9663 | 43.7% | 57.65 | 0.7608 | MEASURED small-suite positive; broad quality open |
| MEDIUM / 2.5 GiB | 2.4533 | 27.0% | 25.30 | 2.4230 | PROJECTED; high risk |
| AGGRESSIVE / 1.8 GiB | — | — | — | — | INFEASIBLE with registered budgets/fallbacks |
| MOONSHOT / 1.2 GiB | — | — | — | — | INFEASIBLE with registered budgets/fallbacks |

The 4-GiB table spends 24/48/72/96/120 slabs on
24/31/18/14/5 layers respectively.  Its projected cost is 3.31x the H22
average96 cost, so `SAFE` must not be read as a quality claim.  The 2.5-GiB
table puts 87 of 92 layers at the 24-slab minimum and has 10.55x the H22
projected cost.  It is useful as a falsification point, not as the first
production candidate.

## Measured SAFE / 4-GiB quality gate

The first end-to-end H23 candidate used the frozen SAFE table, unchanged
calibrated mean-tail semantics, and unchanged exact-fallback rule through all
92 routed layers.  The official GGUF chat template ran with thinking disabled.
Native K3 first passed all six predeclared tasks; only then was the candidate
scored.

| Measured quantity | Native exact | SAFE / 4-GiB candidate |
|---|---:|---:|
| Native-success tasks retained | 6 / 6 | **6 / 6** |
| Generated sequences identical to native | — | **6 / 6** |
| Aligned terminal rows | — | 291 |
| KL, mean / median / p95 / max | — | 0.1674 / 0.0709 / 0.6150 / 3.9820 |
| Top-1 agreement on aligned rows | — | 247 / 291 (84.9%) |
| Logical routed bytes / position | 9.0696 GiB reference geometry | **3.8279 GiB** |
| Logical routed-byte fraction / saving | 100% / 0% | **42.21% / 57.79%** |
| Exact-fallback route fraction | — | 12.32% |
| Exact-fallback fraction of provider bytes | — | 29.23% |
| Process read / logical provider bytes | — | 0.9723x |
| Wall time, six prompts | 1023.97 s | 657.80 s |

This is a genuine positive gate: a layer-adaptive policy averaging 57.65 slabs
preserved every generated token and every task answer in this small suite while
requesting 57.79% fewer routed bytes than exact geometry.  It is **not** a broad
quality certification.  The nonzero KL and 44/291 top-1 changes show that the
distributions are not equivalent even though greedy output stayed identical.
The suite contains one factual, code, arithmetic/reasoning, grammar,
Italian-translation, and extraction task, each with only eight generated
tokens.  Longer generation and broader domains remain open.

The apparent 1.56x wall-time improvement is directional only.  A brief,
accidental S0 process overlapped the native reasoning prompt, so native timing
and its recorded peak VRAM are contaminated; quality artifacts are unaffected.
The candidate itself completed without GPU contention at a 2-GiB device cache
and 12 CPU threads.

### Rejected attempts retained as controls

- The first native launch requested the old 16-GiB device cache and reached
  24,254 MiB VRAM with only 71 MiB free.  It was stopped and archived as
  `kimi-h23-native-success-rejected-cache16g-20260816`; it is not evidence.
- The first candidate launch stopped at initialization because its stale runner
  binary rejected the new 24-slab table row.  It was archived as
  `kimi-h23-safe4gib-rejected-stale-binary-20260816`; no candidate inference was
  scored.  The runner now rebuilds the exact suite target with `-j4` and runs
  the progressive-provider unit test before creating a result directory.

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

1. **Broaden the 4-GiB quality gate.**  The six-task gate passed, but the
   distributional drift is material.  Add longer, multilingual, retrieval,
   tool-use, and harder native-success prompts before calling the policy safe.
   Do not run the 2.5-GiB table merely because six short greedy paths matched.
2. **Raise calibration coverage before chasing 1.2 GiB.**  Repeating the
   all-layer export at 10K tokens is scientifically earned because exact
   fallback consumed 29.23% of provider bytes in the measured candidate and is
   already a measured 1.256-GiB projected floor.  Recompute the floor before
   changing any fallback semantics.  The 10K export has not been started.
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

The new six-task fixture is deliberately a small native-success decision gate,
not a replacement for that broader fixture.  Future expensive suites should
run one prompt per immutable result directory and append a checksum-protected
index after each prompt.  A resume operation should skip only prompts whose
manifest, logits, telemetry, fixture hash, budget-table hash, calibration hash,
and sidecar hash all validate.  This avoids losing an hour-long monolithic run
to WSL interruption and makes progress inspectable without trusting partial
stdout.

## Reproduction identities

| Artifact | SHA-256 / identity |
|---|---|
| Checkpoint | `unsloth-Kimi-K3-GGUF/UD-IQ1_S`, 14-shard local set; first shard `Kimi-K3-UD-IQ1_S-00001-of-00014.gguf` |
| Fixture | `c8251c614b46c9fa9e7335625c5e55548d49cb0ae8b99872300cb96945f8f910` |
| SAFE budget table | `d92192c9cdff9ec61adc274fd3cf553b1481a46cadc93ebeba9b11da56daa57f` |
| Frozen calibration manifest | `ed321b400b99234522583d7ea279cca8ba2b053257daa8dd713137beb7546bc1` |
| Progressive sidecar manifest | `192efad90c790b8a8230e71bca56c260eee3baf6846108d71c9ec598f6762b7c` |
| Native suite manifest | `43cf3cf080541bf3491c3e6c8a781b9c84a44c2eb67cc98a91b16affc5ff82cc` |
| Candidate suite manifest | `3bfcb9c99046819da82e93419945ca4c3981971339cc34904aa8d8314b58227a` |
| Native executable commit recorded by runner | `0663e16575228b8b86c7ee9febe5ec9beb5af3a9` |
| Candidate executable commit recorded by runner | `a9665c2e846d51a9756964be6c5231bfbae047db` |

The runner did not content-hash all 594 GB of GGUF shards; the checkpoint row is
therefore an exact repository/path identity, not a claimed full-checkpoint
content hash.  The two suite manifests preserve the exact command,
configuration, prompt tokens, output tokens, and per-position logit paths.

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

Run the measured native-success gate with safe defaults:

```bash
scripts/gpu_lease.sh run H23 -- scripts/run_kimi_h23_quality.sh native
scripts/gpu_lease.sh run H23 -- scripts/run_kimi_h23_quality.sh candidate
```

The runner defaults to a 2-GiB device cache and 12 CPU threads, rebuilds only
the required targets with `-j4`, and refuses to overwrite an existing output.

## Proposed roadmap diff (not applied)

```text
H23 SAFE ~4 GiB:
  PROJECTED -> MEASURED SMALL-SUITE GO
  3.8279 GiB/position; 6/6 native-success tasks and 6/6 token sequences retained
  broad/long-generation quality remains OPEN because mean KL is 0.1674

H23 NEXT:
  1. run the already-earned 10K all-layer calibration to reduce exact fallback;
  2. recompute the byte floor and SAFE/MEDIUM policies with unchanged semantics;
  3. broaden the native-success suite using per-prompt resumable artifacts;
  4. only then quality-gate a lower-byte policy.

H23 HOLD:
  do not call 3.83 GiB sufficient for 4 tok/s;
  do not promote the 2.5-GiB projection or alter mean-tail/fallback semantics yet.
```
