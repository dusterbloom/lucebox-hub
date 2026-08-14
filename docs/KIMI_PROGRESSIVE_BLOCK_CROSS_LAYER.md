# Kimi Progressive Expert Blocks: Layer-12 Control

**VERDICT: NO-GO for the current mean-tail provider at model layer 12.**

Here, a **slab** means one aligned 256-neuron routed-expert block: the matching
256 gate rows, value rows, and output columns. Each 3,072-neuron expert contains
twelve such blocks, and one token activates 192 blocks across sixteen routed
experts.

## Why this control was run

Layer one showed a small but genuine equal-byte advantage for adaptive blocks:

| Exact payload | Whole experts | Adaptive blocks |
| ---: | ---: | ---: |
| 50% | 0.975828 mean cosine | **0.976688** |
| 75% | 0.990747 | **0.991022** |

This experiment asked whether that advantage survives at the first layer of
the next twelve-layer Block AttnRes group, model layer 12. It reused the same
10,000-token, sequence-disjoint corpus and the native IQ1_S expert evaluator.
Every preceding routed layer was evaluated exactly through the production NVMe
streaming path.

## Exactness and coverage gates

The exact observer path remained bit-exact both when disabled versus enabled
and when reconstructing the routed aggregate. Splitting an expert into twelve
blocks reproduced one full dequantized expert at `1.000000004` mean cosine and
`9.86e-7` mean relative L2. The remaining all-block versus native-kernel gap was
`0.999824478` mean cosine, far too small to explain the partial-budget result.

The capture contained 8,200 calibration and 1,800 validation tokens. Forty of
896 experts were absent from calibration. Only two of 28,800 validation routes
used one of them, so those two routes remained exact. This added an average of
`0.01333` block-equivalents per token, or about `0.00694%` of the full routed
payload. No validation response was used to invent a mean or ordering.

## Matched-byte result

| Provider | Exact payload | Mean cosine | p05 | p01 |
| --- | ---: | ---: | ---: | ---: |
| 8 whole experts + mean tail | 50% | **0.866334** | 0.802177 | 0.768649 |
| 96 adaptive blocks + mean tail | 50% | 0.865533 | 0.801300 | 0.759115 |
| held-out residual oracle | 50% | 0.879895 | 0.824682 | 0.795117 |
| 12 whole experts + mean tail | 75% | **0.938669** | 0.909167 | 0.894036 |
| 144 adaptive blocks + mean tail | 75% | 0.937733 | 0.907423 | 0.888383 |
| held-out residual oracle | 75% | 0.947923 | 0.926069 | 0.915829 |

Unlike layer one, adaptive blocks do not beat complete experts at either
matched point. More importantly, even a non-deployable selector that sees the
actual held-out residual norm remains far below a plausible substitution gate.

## Falsification: was calibration simply too sparse?

The frontier was also measured on validation tokens for which every active
expert had substantial calibration support:

| Subset | Tokens | Adaptive 50% | Adaptive 75% |
| --- | ---: | ---: | ---: |
| all validation | 1,800 | 0.865533 | 0.937733 |
| at least 30 hits per active expert | 1,251 | 0.871618 | 0.940568 |
| at least 100 hits per active expert | 379 | 0.881866 | 0.945215 |

Coverage helps modestly but does not recover the layer-one regime. Sparse
calibration therefore cannot account for the negative result.

## Interpretation

This is evidence against treating the layer-one result as a network-wide
property. At layer 12, the omitted-block mean is the primary failure: better
selection helps only slightly, while both whole-expert and block policies lose
large amounts of direction when their tails are replaced by static means.

This result does **not** show that every later K3 layer behaves like layer 12.
It does show that a global progressive-bank deployment cannot be justified by
layer one alone. No layer-12 sidecar or final-logit intervention should be built
for this provider.

H16 behavioral selection remains justified at layer one, where the geometric
frontier is close and final-logit anisotropy is already measured. A Fisher
selector cannot plausibly rescue layer 12 while the oracle remains this low;
an aggregate omitted-tail correction would have to move the frontier first.

## Reproduction artifacts

The concise machine-readable record is
`results/kimi_layer12_neuron_slabs_summary.json`. Full artifacts remain on the
dedicated Kimi NVMe:

- capture: `/mnt/kimi-k3/captures/kimi_layer12_10000.bin`;
- exact teacher: `/mnt/kimi-k3/results/kimi_layer12_panel_10000.teacher.f32`;
- calibration: `/mnt/kimi-k3/fit-state/kimi_layer12_neuron_slabs_calibration.npz`;
- complete frontier: `/mnt/kimi-k3/results/kimi_layer12_neuron_slabs.json`;
- telemetry: `/mnt/kimi-k3/results/kimi_layer12_neuron_slabs.telemetry.json`.

The 10,000-token capture took 2,296.96 seconds, physically read 5.954 TB from
the NVMe at 4.691 GiB/s active throughput, peaked at 7,790,712 KiB resident
memory and 13,003 MiB graphics memory, and recorded zero NVMe errors.
