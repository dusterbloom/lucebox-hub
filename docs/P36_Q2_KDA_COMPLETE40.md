# P36 — Q2_K late-layer KDA and complete preparation

**VERDICT: EXPERIMENTAL SPEED GO; QUALITY CAUTION; NOT DEFAULT.**

P36 tests the byte-bounded continuation earned by P35. It changes only five
KDA matrix families in 40 predeclared late recurrent layers from Q4_K to Q2_K,
then keeps the complete pre-expert preparation for those layers on the RTX
3090. The calibrated 1.22-GiB progressive expert policy, sidecars, mean tails,
exact fallbacks, native full-width expert graph and deterministic expert
reduction are unchanged.

## Artifact integrity

The selected layers are the last 40 recurrent layers, model layers 38 through
90 with the native MLA layers omitted. The conversion changes 200 tensors and
saves 4,128,768,000 bytes (3.845 GiB) relative to the P32 Q4_K artifact.

The verifier scanned all 2,573 tensors. Every one of the 2,373 non-target
tensors passes a full byte hash, including all routed experts. The changed
tensors have the registered Q4_K-to-Q2_K type transition. This is a core-only
quantization experiment; it is not a new expert approximation.

## Capacity

Forty complete-preparation layers occupy 19.711 GiB of accelerator weights and
recurrent state. With the existing 2-GiB expert workspace, the frozen suite
peaks at 23,091 MiB on the 24-GiB RTX 3090. This is capacity-safe on this exact
machine but leaves little margin and remains opt-in.

## Frozen 12-prompt result

The reported rate counts autoregressive state transitions, not emitted output
tokens. The suite has 144 output tokens but only 132 timed transitions.

| metric | P32 Q4_K | P35 late-18 | P36 Q2_K late-40 |
|---|---:|---:|---:|
| true AR decode | 0.73425/s | 0.80063/s | **0.88976/s** |
| prefill | 0.73660/s | 0.79423/s | **0.88872/s** |
| wall time | 934.820 s | 863.674 s | **775.374 s** |
| native-success tasks | 12/12 | 12/12 | **11/12** |
| token-identical to native | 9/12 | 8/12 | 6/12 |
| peak VRAM | 16,063 MiB | 21,023 MiB | 23,091 MiB |
| energy | 111.787 kJ | 105.758 kJ | **98.326 kJ** |

P36 improves true decode by 1.2118x over P32 and 1.1113x over P35. Eleven
tasks remain correct, including factual, code, arithmetic, grammar,
translation and one decoy-rich extraction task. The other extraction prompt
requires `LIME-742`; P36 emits `LIME-7`. That failure prevents promotion to a
quality default even though it clears the original small-suite 90% retention
threshold.

Against native, aligned mean KL is 0.92088, median 0.54918, p95 3.38016 and
maximum 5.57712. Against P35 directly, eight sequences are token-identical and
aligned mean KL is 0.20289. These are substantial distributional changes, not
rounding noise.

## Current control room

The median transition is 1,116.44 ms:

| stage | median |
|---|---:|
| CPU routed preparation | 388.50 ms |
| accelerator preparation | 127.52 ms |
| expert provider | 516.06 ms |
| all other work | 84.36 ms |

The experiment proves that wider coherent placement is useful, but it also
shows the remaining joint limit. Even with both preparation stages free, the
measured median ceiling is about 1.67 transitions/s while the expert provider
remains. Even with experts free, preparation alone cannot reach 4/s. The next
speed result must combine a safer precision/placement point with a cheaper
provider and verified multi-token amortization.

## Decision

Keep P35 as the safer quality configuration. Keep P36 as a measured
experimental speed ceiling and as evidence that a Q2/Q4 mixed frontier is
worth locating. The next narrow quality experiment should reduce the number of
Q2_K layers without discarding the complete-preparation gain. Because the
checkpoint is split by depth, a shard-boundary hybrid may provide that test
without another full 555-GiB conversion; it must receive the same full tensor
integrity check before execution.

Do not call 0.8898/s four tokens/s. Do not infer official max-reasoning quality
from this thinking-disabled product suite.

## Reproduction artifacts

- `results/k3_p36_kda_q2k_late40_plan.json`
- `results/k3_p36_kda_q2k_late40_verification.json`
- `results/k3_p36_q2late40_complete40_broad12_quality.json`
- `results/k3_p36_complete40_vs_p35_broad12.json`
- `results/k3_p36_q2late40_complete40_broad12_stage.json`
- `results/k3_p36_q2late40_complete40_summary.json`
