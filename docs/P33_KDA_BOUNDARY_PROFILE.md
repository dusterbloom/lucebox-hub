# P33 — KDA core-boundary profile

## Verdict

**MEASURED: twelve CPU threads remain the local optimum, and recurrent KDA
graph compute is the dominant core cost.** Further host packing or thread
oversubscription cannot remove this boundary.

P33 keeps the P32 Q4_K core, 10K-calibrated 1.22-GiB slab policy, mean tail,
fallbacks, native expert arithmetic, and prompt fixed. It changes no model
semantics. All three thread controls generated the same 24 output token IDs.

## CPU-thread control

One frozen code prompt supplies 23 steady decode transitions:

| CPU threads | median transition | routed preparation | expert provider | rate |
|---:|---:|---:|---:|---:|
| 8 | 1,453.667 ms | 711.742 ms | 590.775 ms | 0.6879/s |
| **12** | **1,328.166 ms** | **634.375 ms** | **561.455 ms** | **0.7529/s** |
| 18 | 2,068.043 ms | 1,127.245 ms | 773.364 ms | 0.4835/s |

Eight threads lose 8.63% against twelve; eighteen lose 35.78%. The larger
count increases shared-memory and I/O contention on this WSL topology. Thread
tuning is closed unless the hardware or placement changes.

## Boundary decomposition

`DFLASH_KIMI_BOUNDARY_PROFILE=1` separates graph expansion, allocation, input,
compute and output. A 53-token prompt plus four generated tokens produced 56
complete positions, each with exactly 92 routed-layer records. Three decode
positions form the steady diagnostic subset.

| routed family | layers | compute / position | median / layer |
|---|---:|---:|---:|
| recurrent KDA | 68 | **572.676 ms** | 8.052 ms |
| MLA | 24 | 159.139 ms | 6.358 ms |

Graph expansion, allocation, input and output are typically tens of
microseconds per routed layer; compute is several milliseconds. KDA therefore
accounts for 78.25% of measured routed attention-preparation compute. This is
not the same as the separately offloaded latent/shared MoE preparation stage.

## Decision

1. Keep twelve CPU threads as the current default for this path.
2. Do not spend another cycle on host packing inside routed preparation.
3. The next bounded ceiling test is opt-in accelerator placement for a subset
   of complete KDA layers, using the existing P32 Q4_K weights and preserving
   CPU state as the reference.
4. Measure one layer and one prompt before allocating a large resident bank.
5. Any CUDA arithmetic change requires a practical-quality gate; it is not an
   exact-mode claim.

Artifacts:

- `results/k3_p33_core_boundary_profile.json`
- `results/k3_p33_thread8_stage.json`
- `results/k3_p32_kda_q4k_stage_profile.json`
- `results/k3_p33_thread18_stage.json`
- `/mnt/kimi-k3/results/kimi-k3-p33-boundary-profile-20260818`
