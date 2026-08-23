# K3 production P40 layer epoch

## Result

The production ponytail's wide P40 path was thrashing its own 8-GiB device
cache across layers. A default-off layer-epoch reset raises the
M1024 exact-path prefill result from **1.552159 to 2.134617 positions/s**
(`1.3753x`) on physical GPU1/gfx1151. It halves expert wall time, removes 92.8%
of payload H2D, and leaves the authoritative physical byte count unchanged.

This is an engineering **GO** for the qualified wide profile. It is not an
automatic all-width default and it is not evidence of 10 positions/s.

## Frozen closure

- branch: `perf/k3-production-ponytail`
- base: `origin/main` at `21506614`
- token replay support: `44f8426a`
- candidate implementation: `10e79c805e695cc4581fa78be1cfc3b7fee9ae50`
- candidate smoke binary SHA-256:
  `8250f281307034e2d99715c5584608ad40e41e9339f79875a8d5b130de82b0dc`
- machine: Lucebox4, HIP/ROCm, physical GPU1 `gfx1151`, logical device 0,
  `performance` platform profile
- topology: single owner; 18 CPU workers; fixed96/calibrated96; P30 16 GiB;
  P40 8 GiB; direct-pread; hipBLASLt disabled

The existing `smoke_kimi_k3_forward` token-replay path ran under the existing
GPU lease. M64 was a same-binary A/B whose only variable was
`DFLASH_KIMI_P40_LAYER_EPOCH=0|1`; it used `DFLASH_KIMI_PREFILL_CHUNK=64`, max
context 65, and one output token. The longer M1024 control was measured at
`44f8426a` immediately before the default-off implementation, while the
candidate was measured at `10e79c80`; it used chunk/context 1024 and no output
token. The disabled implementation is behavior-preserving and M64 proves the
same-binary effect, but the M1024 pair must not be described as same-binary.
No new harness was introduced.

## A/B results

| Width | Metric | Control | Layer epoch | Change |
| ---: | --- | ---: | ---: | ---: |
| 64 | prefill | 0.947836 pos/s | 1.029940 pos/s | `1.0866x` |
| 64 | expert wall | 50.283 s | 44.906 s | `-10.7%` |
| 64 | payload H2D | 196.084 GB | 110.045 GB | `-43.9%` |
| 64 | cold fills / evictions | 42,627 / 41,532 | 22,543 / 0 | — |
| 64 | physical bytes | 110,658,715,648 | 110,658,715,648 | identical |
| 1024 | prefill | 1.552159 pos/s | **2.134617 pos/s** | **`1.3753x`** |
| 1024 | expert wall | 378.169 s | **198.543 s** | **`-47.5%`** |
| 1024 | payload H2D | 2,340.370 GB | **168.213 GB** | **`-92.8%`** |
| 1024 | cold fills / evictions | **576,248 / 575,153** | **34,014 / 0** | — |
| 1024 | physical bytes | 169,143,877,632 | 169,143,877,632 | identical |
| 1024 | direct I/O wall | 133.943 s | 44.224 s | `-67.0%` |
| 1024 | completed / fallback / error | 1,244,278 / 0 / 0 | 1,244,278 / 0 / 0 | identical |

M64 retained terminal argmax token 5801 and identical traffic SHA-256
`57a76723bff549e8284777a4daa75d7b828c126112dde14f9ea91b33b3eb54f9`.
M1024 retained identical traffic SHA-256
`c0dfd31205484facf903d1e55cd75ccf05338359583b1f9d76fbaa032b6cea15`,
physical bytes, and provider call counts.

Correctness wording matters: the underlying P40 M1024 path had already passed
full-logit and state exact qualification. This production M64 A/B checked the
terminal argmax and traffic. This production M1024 smoke generated no output
token and therefore did **not** produce a fresh full-logit or state hash. It is
wrong to describe this particular A/B as a new full-logit/state qualification.

## Root cause

P40 has 1,095 external device slots and uses frequency-first LFRU replacement.
Its keys are layer-specific. At the end of a wide layer, every resident key is
therefore dead for the next layer, but its accumulated frequency survives.
During the next layer those stale, high-frequency keys outrank newly filled
current-layer keys. The cache consequently evicts the useful new entries and
fills them again within the same layer.

The M1024 control's **576,248 cold fills and 575,153 evictions** are the direct
signature. In fact, `576,248 - 575,153 = 1,095`: after the initial fill of all
1,095 slots, every further cold fill evicted something. Trace accounting
establishes at least **493,816 redundant cold fills**; the raw
control-minus-candidate difference is 542,234. The candidate starts a cold P40
residency epoch at each qualified wide-layer boundary. It synchronizes, fails
if a lease remains active, clears external slot identity and replacement
history, and reuses the existing allocation. Evictions fall to zero.

Physical sidecar bytes do not change because P30 already deduplicates the
backing reads. The damage was repeated host-cache-to-device upload/scatter and
the resulting loss of orderly request cadence, not a different semantic read
plan. That is why M1024 payload H2D falls by 2.172 TB while physical bytes and
traffic remain identical.

## Post-win critical path

| M1024 candidate term | Seconds | Share |
| --- | ---: | ---: |
| causal one-row core | 266.028 | 55.5% |
| experts | 198.543 | 41.4% |
| join | 10.364 | 2.2% |
| output | 4.510 | 0.9% |
| other | 0.241 | 0.1% |
| total | 479.687 | 100% |

The measured result is **2.134617 positions/s**. An impossible zero-cost expert
stage would reach only about **3.642 positions/s**; an impossible zero-cost
causal core would reach only about **4.792 positions/s**. Ten positions/s
requires a 102.4-second M1024 prefill, or 377.311 seconds removed from this
run—81.2% of the combined core-plus-expert time. The honest next conclusion is
that causal core and exact expert execution are co-binding. The P40 cache fix
does not by itself put 10 positions/s within reach.

## Artifacts and verdict

The machine-readable record is
`results/k3_production_p40_layer_epoch.json`. It includes the exact source and
binary closure, environment, raw counters, stage ledger, fixture hashes, and
log hashes. Raw logs remain on the measurement host and are not copied into
the production branch.

Verdict: **ENGINEERING GO, DEFAULT OFF** for the qualified P40 wide-macro
profile. Width-one behavior is deliberately unchanged. Promotion beyond that
profile requires its own matched correctness and throughput gate.
