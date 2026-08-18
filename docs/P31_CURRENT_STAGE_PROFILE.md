# P31 current revision-40 stage profile

## Verdict

**MEASURED: both the remaining routed preparation/core and the expert provider
must be accelerated. Neither axis alone can reach four tokens per second.**

The profile used the same revision-40 1.22-GiB adaptive policy and one
official-template 53-token code prompt. It recorded 23 steady decode
transitions and produced the same tokens as the frozen autoregressive output.

| Stage | Median per transition | Share |
|---|---:|---:|
| total | 1,446.729 ms | 100% |
| remaining CPU routed preparation | **802.549 ms** | **55.47%** |
| expert provider | **493.997 ms** | **34.15%** |
| accelerator latent/shared preparation | 57.615 ms | 3.98% |
| join | 37.913 ms | 2.62% |
| dense | 22.958 ms | 1.59% |
| output head | 21.461 ms | 1.48% |

The measured median rate is 0.6912 transition/s. If the complete expert
provider were free, the hard ceiling would be only 1.0496/s. If the complete
remaining routed preparation were free, the ceiling would be 1.5524/s. Making
both free would expose a 6.66/s residual ceiling, which is why adaptive bytes,
core acceleration, and multi-token verification must multiply rather than
replace one another.

This also explains P30: selected direct storage reads are nested inside a
493.997-ms expert-provider stage, while the 802.549-ms CPU core boundary is
unchanged. A large read cache can validate a byte-hit model without moving the
integrated transition materially.

Machine-readable evidence is in
`results/k3_p31_current_stage_profile.json`.
