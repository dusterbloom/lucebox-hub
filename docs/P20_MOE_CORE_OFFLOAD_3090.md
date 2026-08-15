# P20 3090 MoE-core offload

**VERDICT: SYSTEMS GO; SEMANTIC GATE OPEN.**

This opt-in experiment moves the always-used routed-layer core to the RTX 3090
while leaving attention/recurrent execution on the CPU and routed-expert
delivery on the existing calibrated96 sparse-physical path. It is enabled only
with `DFLASH_KIMI_MOE_CORE_OFFLOAD=1`; the established default is unchanged.

## Placement

Across 92 routed layers, 736 tensors totaling 16,213,714,944 bytes (15.10 GiB)
are copied once to the 3090:

- router matrices and biases: about 2.20 GiB;
- routed latent down/up projections and normalization: about 3.61 GiB;
- full shared-expert gate/up/down matrices: about 9.29 GiB.

The CPU still runs layer 0, attention, KDA/MLA and AttnRes. Each routed layer
crosses to CUDA for router/latent/shared-expert preparation, uses the frozen P20
expert provider, and crosses to CUDA again for the native routed join. The
streamed-expert cache was reduced from 16 GiB to 2 GiB so the complete placement
fits with the P20 scratch path.

## Measured two-row control

Hardware and request are the same registered P20 `Hi` control: one prefill row
and one true autoregressive transition through all 92 routed layers. The
offload arm uses the same direct-read sparse provider and a 96-slab budget.

| quantity | registered P20 | core offload | change |
| --- | ---: | ---: | ---: |
| prefill | 26.842 s | **21.512 s** | **-19.9% latency** |
| decode transition | 26.976 s | **20.648 s** | **-23.5% latency** |
| transition rate | 0.0371/s | **0.0484/s** | **+30.7%** |
| whole-device reads | 106.34 GB | **84.48 GB** | **-20.6%** |
| GPU energy | 6416 J | **5979 J** | **-6.8%** |
| peak VRAM | 17,118 MiB | **18,250 MiB** | fits 24-GiB card |

The elapsed measurement includes the one-time 15.10-GiB copy at startup, so it
improves by only 8.7%. A serving process amortizes that copy; this short run does
not yet establish steady-state throughput.

## Semantic result

The generated IDs remain `11 374` (`Hi, I`) and top-1 agrees on both rows. The
two independent offload executions produced the same logit bytes, establishing
determinism for this control. They do not equal the CPU-core reference:

- mean `KL(reference || offload)`: 0.0037689;
- maximum row KL: 0.0062183;
- maximum absolute logit difference: 0.655904;
- top-1 agreement: 2/2.

This is execution-arithmetic sensitivity, not a slab-policy change: CUDA now
performs quantized shared-expert, router and latent-projection reductions that
the reference performs on the CPU. Therefore this experiment proves the
placement and read-pressure thesis, but does not pass the project's preferred
byte-identity gate or justify a broad quality claim.

## Next gate

The smallest useful refinement is a placement ablation, not another compression
idea. Preserve the native CPU router exactly, then measure shared-expert and
latent-projection offload separately. This determines whether most of the
15.10-GiB residency benefit can be retained with materially lower terminal KL.
Only the winning partition earns a longer frozen quality suite.

Machine-readable evidence is in `results/k3_p20_moe_core_offload.json`; raw
telemetry remains under `/tmp/kimi-p20-moe-core-offload-direct-20260815` because
the experiment NVMe is currently mounted read-only.
