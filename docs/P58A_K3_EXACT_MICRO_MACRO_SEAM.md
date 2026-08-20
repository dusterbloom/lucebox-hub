# P58A — Exact K3 micro/macro semantic seam

## Verdict

**SEMANTIC GO. THROUGHPUT NOT ACTIVATED. DEFAULT-OFF.**

P58A proves the first exact separation between K3's recurrent arithmetic width
and its routed-expert service width on Lucebox4. KDA, MLA, AttnRes, routing,
shared-expert work and state evolution retain the established one-row order
(`mu=1`). Eight exact routed rows are collected and presented to the calibrated
provider in one call per routed layer (`M=8`). The provider does not change a
row's expert arithmetic or the canonical row-wise join.

The bounded width-eight oracle is byte-exact in every registered semantic
boundary. Its committed diagnostic ratio is only **1.157015852x**. This was an
exactness qualification, not a matched prefill benchmark: no fact, broad,
long-context or throughput suite was run. P58A therefore activates the
micro/macro seam as research substrate only; it does not activate production
prefill or establish a performance claim.

## Exact execution boundary

For a complete eight-row macrochunk, the dedicated path:

1. evaluates embeddings and every dense/core row independently in native
   one-row order;
2. evolves KDA and MLA state row by row while retaining exact causal reduction
   widths;
3. gathers eight routed inputs, router IDs and router weights;
4. makes one `n_tokens=8` calibrated-provider call per routed layer;
5. joins each returned row independently in canonical order; and
6. restores the speculative recurrent snapshot, then commits the accepted
   eight rows with the qualified ReplaySSM path.

Prompt tails shorter than eight use the established one-row path. The seam is
available only when both of these explicit controls are set:

```text
DFLASH_KIMI_PREFILL_CHUNK=8
DFLASH_KIMI_P58_EXACT_MULTIROW=1
```

The default remains width one with P58 disabled. Startup fails closed for
device-output/P42 joins, asynchronous or persistent one-token paths,
dual-owner execution, drafting, hidden-layer diagnostics and trace modes. The
qualified provider is the sidecar-authoritative, all-layer calibrated96 host-
output provider on the single GPU1 owner.

## Lucebox4 qualification

The sole model run used prompt token `[18699]`, so the eight-token oracle span
started at nonzero base position 1. Its oracle tokens were
`[11, 374, 4936, 261, 814, 2742, 316, 374]`. The accelerator core and streamed
expert executor were both owned by GPU1. Fixed 96-slab policy was used for all
92 routed layers; H22 variable budgets, speculative decode and all incompatible
trace/one-token paths were disabled.

| semantic gate | sequential | P58A width eight | result |
|---|---:|---:|---|
| full logits | 8 x 163,840 values | 8 x 163,840 values | bit-equal |
| argmax rows | 8 | 8 | equal |
| aggregate recurrent hash | `3345846951683756339` | `3345846951683756339` | equal |
| aggregate MLA hash | `16569152724683205385` | `16569152724683205385` | equal |
| per-layer convolution hashes | 93 | 93 | all equal |
| per-layer SSM hashes | 93 | 93 | all equal |
| per-layer MLA hashes | 93 | 93 | all equal |
| logical provider bytes | 40,075,886,592 | 40,075,886,592 | equal |
| compact attempted/completed | 8,504 / 8,504 | 8,504 / 8,504 | equal, complete |
| compact fallback/invalid | 0 / 0 | 0 / 0 | pass |

The aggregate provider traffic file covers both prompt rebuilds and both
oracle arms: 89,915,965,440 logical bytes, including 340 authoritative policy
fallback routes. These are calibration-coverage decisions, not compact
executor failures. P41 completed 19,594/19,594 aggregate compact jobs with
zero execution fallback or invalid jobs.

## Diagnostic timing, not activation

| arm | time | diagnostic rate |
|---|---:|---:|
| eight sequential rows | 11.825277626 s | 0.676516886 rows/s |
| P58A verify | 9.635555403 s | — |
| ReplaySSM commit | 0.584942103 s | — |
| verify plus commit | 10.220497506 s | 0.782740761 rows/s |

The committed ratio is 1.157015852x and the wall reduction is 13.57%. Physical
read bytes were 21,386,264,576 for the sequential arm and 21,047,894,016 for
the P58A arm; that cache-sensitive difference is not an equality gate. The
whole process took 32.17 seconds including model initialization and prompt
rebuilds. Swap remained at 12 KiB before and after, and no GPU process remained.

This timing is not comparable to P55 broad prefill or the 16.28035098
positions/s cold-prefill target. It does not justify a broad P58 run or default
enablement. Its value is semantic: P59 can now target shared compact execution
behind an exact eight-row provider boundary without widening recurrent
arithmetic.

## Source and validation record

The qualification used base HEAD
`0c7d9c6d156d48d4bde98d8e2ce8c649a39069c6` plus the reviewed nine-file
working-tree diff
`53ae270e8fe01f1d4a50e014c8a1c79c2987e4c32569096ce187b96dd2d6525e`.
No P58 commit existed at qualification time. The raw implementation size is:

| scope | added | deleted |
|---|---:|---:|
| production | 842 | 41 |
| tests and oracle runner | 85 | 1 |
| total | 927 | 42 |

The source passed local CUDA compilation and the core-placement test. On
Lucebox4, the capped HIP build included the oracle runner, core/provider,
ordered join, MoE stream and sparse-K targets. Core/provider gates passed,
MoE stream passed 2/2 on each GPU, and sparse-K MMVQ passed 68/68 on both
`gfx1201` and `gfx1151`.

The frozen remote root is
`/home/duster/kimi-k3-deploy/p58a-mu1-m8-oracle-53ae270-20260820`.
Its `s0.json` SHA-256 is
`1bb1ede71e55ecdc4c28f28fcef06507007086a844ff9740c23a298424244be6`;
the qualified runner SHA-256 is
`591e8e8806ac55328cf2df899efc8e3fb3967502602114e3be6ee3ebd8ff241a`.
The complete machine-readable record is
`results/k3_p58a_exact_micro_macro_oracle.json`.

## Decision

Retain P58A default-off as an exact semantic discriminator. Do not claim a
prefill speedup, run broad P58, or infer that width-eight verification is a
competitive multi-token prediction policy. Use the seam only for the ordered
P58/P59 roofline work: measure core-only, all-resident expert and I/O-only
ceilings, then decide whether a descriptor-driven shared compact executor has
enough independent margin to justify its implementation.
