# K3 production P46 exact-macro graph reuse

## Result

Commit `014aaf4a5ac4b5c11af2461c69e61b810391a517` extends the
existing default-off P46 persistent routed-preparation graph to the exact
multirow path. On the production M1024 profile it reduces the causal-core
interval from **266.028 to 236.624 seconds** and raises end-to-end prefill from
**2.134617 to 2.263908 positions/s** (`1.0606x`). The measured total falls from
479.711 to 452.315 seconds.

This is an engineering **GO, DEFAULT OFF**. It is a real production-path win,
not the 10-pos/s solution: causal core and experts remain co-binding.

## Implementation and closure

The exact macro previously rebuilt the recurrent routed-preparation graph for
each row. P46 now builds a second immutable graph per recurrent layer for macro
replay, reuses one allocator workspace, and publishes the normalized KDA input
into the correct replay row on the same backend stream. The ordinary width-one
graph and disabled path are unchanged. Unsupported shapes fail closed.

- branch: `perf/k3-production-ponytail`
- base: `origin/main` at `21506614`
- parent winner: P40 layer epoch at `10e79c80`
- P46 commit: `014aaf4a5ac4b5c11af2461c69e61b810391a517`
- `kimi_k3_graph.cpp` SHA-256 at the commit:
  `c19948ab752d71043736e1ec7ea378cba22d4aa8b7f8cee999165e0077ac115d`
- clean production smoke SHA-256:
  `b52be8a2c12f7264709b9b8ed188ded38c8fc4e823e877461d98b623664c7436`
- production server SHA-256:
  `988fb99305ebe075dd37893e22582e9d85d50533ebee2d37e6a36c8a66324848`
- machine: Lucebox4, HIP/ROCm, physical GPU1 `gfx1151`, logical device 0,
  performance platform profile, single-owner topology, 18 CPU workers
- policy: calibrated96/fixed96, P30 16 GiB, P40 8 GiB with layer epochs,
  P41/P42/P45 enabled, direct-pread, hipBLASLt disabled
- switch: `DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP=0|1`

The M8 and M64 component checks are same-binary A/Bs. The M1024 baseline is
the immediately preceding P40 winner at an adjacent source commit; it is not a
same-binary A/B. That distinction is important because M1024 establishes the
production magnitude while M64 establishes causal attribution.

## Component A/B

| Width | Arm | Total | Core | Experts | Direct I/O | Result |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 8 | P46 off | 11.315 s | 2.162 s | 9.006 s | 7.336 s | 0.706561 pos/s |
| 8 | P46 on | 13.140 s | 1.887 s | 11.105 s | 9.845 s | 0.608114 pos/s |
| 64 | P46 off, reversal | 56.140 s | 16.256 s | 38.898 s | 25.695 s | 1.139837 pos/s |
| 64 | P46 on | 55.874 s | 14.373 s | 40.515 s | 25.605 s | 1.145091 pos/s |

M8 saved 275.151 ms of core, or 34.394 ms/position. Its end-to-end result is
**INVALID for performance attribution** because storage varied by 2.508
seconds in the opposite direction. It still supplies a consistent component
signal. At M64, matched I/O shows a **1.883-second core saving**
(`29.424 ms/position`, `11.6%`) and a 266-ms end-to-end saving. A repeat P46-on
arm reproduced core at 14.370 seconds, although its slower storage makes its
total unsuitable for the matched comparison.

Both M8 arms used identical 31,580,471,296 physical bytes and output IDs
`48430,10867`. Both M64 arms used identical 115,133,669,376 physical bytes,
identical traffic SHA-256
`898a397272075995ba469d4db79526c7d099b7c6ed148516f9cf6d154dc1a9fd`,
and output IDs `5801,114820`. The M64 same-binary SHA-256 was
`47187c31941dedda76f9e9dd5e3bda3328bb2e4bea7f4a53444f5f359f7723c3`.

## Exact partial-commit gate

A temporary probe compared ordinary sequential execution with exact-macro
replay followed by `commit_n=3`. P46 off and on both passed:

- all three committed rows of full vocabulary logits byte-identical;
- per-layer convolution, SSM, and MLA state hashes identical after commit 3;
- the next ordinary step's full logits byte-identical;
- per-layer convolution, SSM, and MLA state hashes still identical after that
  next step;
- next-logits hash `7899592251675440270` in both arms;
- traffic SHA-256
  `d0436d265e1b04585481f46e33b45c84064ee27d7a65e42d16b4e1856f9b68f1`
  in both arms.

The probe reported `conv3/ssm3/mla3/conv4/ssm4/mla4=-1`; `-1` means no
mismatching layer. Its binary SHA-256 was
`3e9a6ad7ee7eb950b28be4be1f78f8097c8519a984d7d78caf7811d8b0147b00`.
The probe patch was temporary and is not production code.

## M1024 production result

| Term | Adjacent P40 baseline | P46 winner | Change |
| --- | ---: | ---: | ---: |
| prefill wall | 479.711 s | **452.315 s** | **-27.396 s** |
| positions/s | 2.134617 | **2.263908** | **`1.0606x`** |
| stage total | 479.686969 s | **452.274679 s** | **-27.412290 s** |
| causal core | 266.028214 s | **236.624296 s** | **-29.403918 s (`-11.1%`)** |
| experts | 198.543269 s | 200.357841 s | +1.814572 s |
| join | 10.363974 s | 10.543054 s | +0.179080 s |
| output | 4.510118 s | 4.507782 s | -0.002336 s |
| direct I/O | 44.223623 s | 47.471886 s | +3.248263 s |
| physical bytes | 169,143,877,632 | 169,143,877,632 | identical |
| payload H2D | 168,212,803,584 | 168,212,803,584 | identical |
| P40 cold / evictions | 34,014 / 0 | 34,014 / 0 | identical |
| completed / fallback | 1,244,278 / 0 | 1,244,278 / 0 | identical |

The winner executed 69,632 P46 graphs, all through the replay variants, with
zero invalidated native graphs. Expert sub-counters were: 47.471886 seconds
direct I/O, 0.306308 seconds packing, 6.311063 seconds scatter, 95.680121
seconds expert graph, and 14.696659 seconds readback. Traffic SHA-256 remained
`c0dfd31205484facf903d1e55cd75ccf05338359583b1f9d76fbaa032b6cea15`.

Correctness wording remains bounded. The separate `commit_n=3` probe is a
fresh byte-exact full-logit and per-layer state gate for P46. The M1024
production smoke itself generated no output token and therefore did not emit a
new full-logit/state hash. Its identical traffic, physical bytes, H2D,
provider counts, and zero-fallback execution support the performance result,
but do not replace the partial-commit exactness gate.

## Updated critical path

| M1024 P46 term | Seconds | Share |
| --- | ---: | ---: |
| causal one-row core | 236.624 | 52.3% |
| experts | 200.358 | 44.3% |
| join | 10.543 | 2.3% |
| output | 4.508 | 1.0% |
| other | 0.242 | 0.1% |
| total | 452.275 | 100% |

The measured result is **2.263908 positions/s**. An impossible zero-cost
expert stage reaches only about **4.065 positions/s**; an impossible zero-cost
core reaches only about **4.748 positions/s**. Ten positions/s requires a
102.4-second M1024 prefill, or another 349.875 seconds removed—80.1% of the
combined core-plus-expert wall. The next material work must therefore improve
both phase-specific causal execution and exact expert service; storage alone
cannot close the gap.

## Verdict

**ENGINEERING GO, DEFAULT OFF** for the qualified exact-macro profile. Keep the
ordinary exact fallback. Promotion to a production default requires a matched
long-fixture full-logit/state gate and broad operational qualification; it does
not require reopening the already-passed P46 mathematical seam.
