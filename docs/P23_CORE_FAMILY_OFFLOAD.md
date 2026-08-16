# P23 K3 core-family offload

**VERDICT: SHORT-TRACE SYSTEMS GO; BROAD QUALITY OPEN.**

P23 makes the existing opt-in K3 routed-layer core placement selective. The
router, routed latent projections, and shared experts can be placed on the
accelerator independently while the calibrated96 sparse-physical expert policy
remains frozen. Default K3 execution is unchanged.

## Why this was necessary

With 48 GiB available to WSL and mapped-page release disabled, the CPU-resident
core improved a true decode transition from 26.976 seconds to about 15.155
seconds. Moving the original 15.10-GiB MoE core bundle to the 3090 reduced it
again, but changed CPU-versus-CUDA arithmetic. The preregistered refinement was
therefore to retain the native CPU router and isolate the other two families.

## Two-row family ablation

All arms use the same model, prompt, calibrated96 provider, 2-GiB expert cache,
native 3,072-wide expert reduction, and one true autoregressive transition.

| GPU families | weights | decode transition | mean KL vs CPU | max KL | top choice |
| --- | ---: | ---: | ---: | ---: | ---: |
| router + latent + shared | 15.10 GiB | 10.922 s | 0.003769 | 0.006218 | 2/2 |
| shared | 9.29 GiB | 11.525 s | 0.003713 | 0.005886 | 2/2 |
| latent | 3.61 GiB | 12.151 s | **0.002491** | **0.003954** | 2/2 |
| latent + shared; CPU router | 12.90 GiB | **11.035 s** | 0.003489 | 0.005318 | 2/2 |

The tiny two-row sample cannot rank quality reliably. It does establish that
router offload is not required for the systems gain. Latent-only gives the
lowest local divergence; latent+shared gives the best speed/quality balance to
take to a longer control.

## Eight-row leading-arm control

The CPU-router/GPU-latent+shared arm generated the same eight IDs and text as
the 48-GiB CPU-arithmetic reference:

```text
11 374 4936 261 814 2742 316 374
Hi, I’m a new user and I
```

Across all eight scored rows:

- top-choice and generated-token agreement: **8/8**;
- mean / median / maximum KL: **0.006498 / 0.003011 / 0.031144**;
- decode: **65.191 seconds for seven real transitions**;
- decode time: **9.313 seconds/transition** or **0.1074 transitions/s**;
- speedup: **2.90x** versus the registered 26.976-second cold path and
  **1.63x** versus the 48-GiB CPU-resident path;
- peak RAM / swap / VRAM: 45.85 GiB / 0 / 15,958 MiB;
- explicit provider direct-I/O time: 6.476 seconds across eight rows.

This is a real end-to-end all-92-layer result, but not a broad quality verdict.
CPU-versus-CUDA reductions are deterministic yet not byte-identical.

## Crash recovery

The subsequent router-only control was interrupted during prefill by a WSL
stop. It produced no logits and is excluded. All completed arms, their logits,
telemetry, traces and checksums survived on the SN850X. No OOM event was found
in the previous boot journal; repeated WSL GPU-bridge errors were present. A
broken background user service was stopped before the clean eight-row run.

## Next gate

Run the CPU-router/GPU-latent+shared partition on the registered frozen quality
prompts. If it preserves behavior, profile the remaining approximately nine
seconds per transition by attention/recurrent CPU time, host boundaries,
shared/latent CUDA time, expert CUDA time and I/O wait. The storage path is no
longer the dominant term on this trace.

Machine-readable evidence is in
`results/k3_p23_core_family_ablation.json`.
