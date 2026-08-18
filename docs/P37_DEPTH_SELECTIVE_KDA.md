# P37 — depth-selective Q2/Q4 KDA placement

**VERDICT: MEASURED QUALITY RECOVERY AND SPEED GO; NEW LEADING PRODUCT-MODE CANDIDATE.**

P37 follows the P36 quality failure to its depth boundary. P36 changed five KDA
matrix families in the last 40 recurrent layers from Q4_K to Q2_K and reached
0.8898 true autoregressive transitions/s, but truncated the registered
`LIME-742` extraction answer to `LIME-7`. P37 keeps Q4_K in the final model
shards, uses Q2_K only for the preregistered tensors physically stored in source
shard 12, and retains complete GPU preparation for 26 late recurrent layers.

The calibrated 1.22-GiB progressive expert policy, sidecars, mean tails, exact
fallback rules, native full-width expert graph and deterministic expert
reduction are unchanged.

## Causal controls

The same extraction prompt was run under three matched one-prompt controls:

| core / placement | result |
|---|---|
| P32 Q4_K + 28 complete-preparation layers | `LIME-742` |
| Q2_K only in source shard 13 + the same placement | `LIME-7` |
| Q2_K only in source shard 12 + the same placement | `LIME-742` |

This isolates the observed failure to Q2_K tensors in the later shard rather
than to complete-preparation widening itself. It does not prove every prompt is
insensitive to the earlier Q2_K tensors; the frozen broad suite is the next
control.

## Artifact integrity

The hybrid model is a no-copy split-file composition. All unmodified shards are
symlinks to the P32 Q4_K artifact, while source shard 12 is the corresponding
P36 Q2_K shard. Exactly 25 registered KDA tensors change from Q4_K to Q2_K in
recurrent layers 77, 78, 80, 81, 82 and 84. The file-level saving is
516,096,000 bytes (0.4807 GiB).

The structural verifier observes all 2,573 tensors, the exact 25-tensor target
set and matching types/shapes everywhere else. Sampled non-target bytes pass;
the exhaustive non-target hash result is recorded in
`results/k3_p37_hybrid_shard12_verification.json`.

## Frozen 12-prompt result

The clean run uses 26 complete-preparation layers:

`57,58,60,61,62,64,65,66,68,69,70,72,73,74,76,77,78,80,81,82,84,85,86,88,89,90`

It places 20,596,915,968 bytes of weights and 175,079,424 bytes of recurrent
state on the RTX 3090 (19.345 GiB total). Peak VRAM is 23,037 MiB.

| metric | P32 Q4_K | P35 late-18 | P36 Q2 late-40 | **P37 hybrid** |
|---|---:|---:|---:|---:|
| true AR decode | 0.73425/s | 0.80063/s | 0.88976/s | **0.86179/s** |
| prefill | 0.73660/s | 0.79423/s | 0.88872/s | **0.84642/s** |
| native-success tasks | 12/12 | 12/12 | 11/12 | **12/12** |
| token-identical to native | 9/12 | 8/12 | 6/12 | **9/12** |
| peak VRAM | 16,063 MiB | 21,023 MiB | 23,091 MiB | **23,037 MiB** |
| energy | 111.79 kJ | 105.76 kJ | 98.33 kJ | **100.87 kJ** |

P37 is 1.1737x P32 and 1.0764x P35. It retains all twelve registered native
successes, including exact `LIME-742` and `QUARTZ-918` extraction. Nine complete
generated sequences are token-identical to native. The three differences remain
task-correct: shorter carbon-dioxide wording and terminal punctuation changes in
Italian and Spanish.

Against native, aligned terminal KL is mean 0.68476, median 0.34578, p95
2.60610 and maximum 5.40591, with 429/660 top-1 agreement. These values remain
far from distributional identity. The suite disables thinking and is a small
product-mode gate, not an official K3 benchmark or broad quality certification.

In a direct P35-to-P37 comparison, 11/12 generated sequences are token-identical
and the only changed sequence is the still-correct, generation-capped Python
function. Aligned mean/median/p95/max KL is
0.04345/0.02347/0.13872/0.74244, with 600/659 top-1 agreement.

## Runtime and byte accounting

The run covers 552 prefill positions and 130 timed autoregressive transitions.
Decode takes 150.849 seconds and full elapsed time is 814.892 seconds. The
progressive provider requests 1.22113 logical GiB/model position. Process reads
are 1.0062x logical bytes; the physical sparse-delivery result remains intact.

Median transition time is 1,161.94 ms:

| stage | median |
|---|---:|
| CPU routed preparation | 487.98 ms |
| accelerator preparation | 100.82 ms |
| expert provider | 480.99 ms |
| all other work | 92.14 ms |

This is a better precision/placement point, not a route to four tokens/s by
itself. Preparation and expert delivery still each consume roughly 0.48 seconds
per transition and require multiplicative work.

## Crash recovery and suite durability

The first 28-layer broad run was interrupted by a WSL shutdown after six
complete prompts. Their logits survived, but the old monolithic runner wrote no
manifest until suite completion. P37 changes the harness to publish an atomic
`suite-manifest.partial.json` after every prompt. Resume is explicit and
validates model, suite, prompt tokens and the already-completed prefix before it
skips work. The clean reported run did not need resume; the mechanism was used
only to make future long evaluations crash-bounded.

Telemetry from a resumed segment cannot be combined into an honest full-run
performance number. A resumed run is suitable for quality recovery; performance
must still come from a clean uninterrupted run, as it does here.

## Decision

P37 supersedes P36 as the measured speed/quality balance and is the leading
product-mode candidate on this RTX 3090 setup. Preserve P35 as the conservative
all-Q4 fallback until thinking-enabled and broader benchmarks validate P37.

The next single systems experiment should target one of the two co-dominant
per-transition costs without changing this quality policy: either a real
selected-page cache/read reuse implementation or provider/preparation overlap.
Do not quantize the final KDA shard again merely to recover the last 3.1% of
P36 speed.

## Reproduction artifacts

- `results/k3_p37_hybrid_shard12_verification.json`
- `results/k3_p37_q4k_complete28_extract_stage.json`
- `results/k3_p37_hybrid_shard13_complete28_extract_stage.json`
- `results/k3_p37_hybrid_shard12_complete28_extract_stage.json`
- `results/k3_p37_hybrid_late28_broad12_quality.json`
- `results/k3_p37_hybrid_late28_broad12_stage.json`
- `results/k3_p37_hybrid_shard12_complete26_broad12_quality.json`
- `results/k3_p37_hybrid_shard12_complete26_broad12_stage.json`
- `results/k3_p37_vs_p35_broad12.json`
- `results/k3_p37_hybrid_shard12_complete26_summary.json`
