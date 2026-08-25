# K3 X8 Lane A: gfx1201 Complete Prompt Provider at M64

Status: **NO-GO**

Date: 2026-08-25

Issue: <https://github.com/dusterbloom/lucebox-hub/issues/14>
Machine-readable result: `results/k3_x8_lane_a_m64_result.json`

## Result

The complete gfx1201 provider is exact in traffic and final logits, but it is about three times slower than the gfx1151 single-owner control. Both candidate repeats fail both preregistered performance gates by a very large margin. M1024 is therefore not earned.

| Arm | Topology | Prefill pos/s | Stage (s) | Experts (s) | Queued upload + graph host wall (s) | NVMe (s) | Return (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | gfx1151 single-owner | 2.082007 | 30.714619 | 25.347881 | 4.516238 | 19.497310 | 0 |
| B1 | gfx1201 complete provider | 0.705859 | 90.645443 | 85.183013 | 63.203737 | 20.070081 | 0.575396 |
| B2 | gfx1201 complete provider | 0.714564 | 89.541805 | 84.070169 | 63.219079 | 18.941683 | 0.574512 |
| A2 | gfx1151 single-owner | 2.093165 | 30.550557 | 25.196708 | 4.517438 | 19.354355 | 0 |

The mean candidate throughput is 0.710212 positions/s versus 2.087586 for the controls: **0.3402x**, or a **65.98% regression**. Candidate expert wall grows by 59.35 seconds on average. The paired stage savings are negative: -59.93 seconds for B1 versus A1 and -58.99 seconds for B2 versus A2.

The decisive term is the complete queued upload-plus-graph dependency service on gfx1201: 63.20 seconds in both candidate repeats versus 4.52 seconds on gfx1151. The `expert-graph-ns` counter is host wall inside `ggml_backend_graph_compute`; it includes preceding queued H2D dependencies. No device activity trace separated H2D from kernel execution, so this result does **not** attribute the whole 58.7-second delta to IQ1_S arithmetic. NVMe time remains in the same 18.94–20.07-second band and result return is only about 0.575 seconds. Even free result return cannot rescue this topology.

## Correctness and frozen plan

All four final arms used one binary (`sha256:21da0db792850d114c9a768fe9f13151a04a3e160d3c8b615d67c34f8c3b6bdd`) and the preregistered A1-B1-B2-A2 order.

- Physical bytes: `110,658,715,648` in every arm.
- Physical-plan/traffic SHA-256: `57a76723bff549e8284777a4daa75d7b828c126112dde14f9ea91b33b3eb54f9` in every arm.
- Raw-logit SHA-256: `20d9dabc54e824816ccfb10b7e207166211871c8f6fd36ceaef8db6d04fbdaca` in every arm.
- Submission order: FNV64 `cbf29ce484222325`; 21,694 groups, 189,685 records and 66,073 output rows in every arm.
- Provider fallbacks: zero.
- Retained partial-replay state probe: **PASS**. It reports `exact=1`, exact prefix and next logits, and matching convolution/SSM/MLA state evolution. Probe log SHA-256: `383b3c7eb436470a40a5dc941a0397721add5eeb142947058c269d7cfbde2a80`.

The candidate transferred 104,425,194,496 weight bytes and 1,079,332,208 metadata bytes in 171,690 async submissions. Average host submission cost was 1.18 microseconds in both candidate repeats. The measured gfx1201 link was PCIe 5.0 x16 (`32.0 GT/s`, width 16); gfx1151 reported `16.0 GT/s`, width 16.

## Timing interpretation

The run exposes aggregate direct-I/O, compact scatter, queued upload-plus-graph host wall, host submission, event/dependency wait, result-return and first/last delivery timestamps. The candidate delivery envelopes span 89.56 and 88.47 seconds; dividing payload by those envelopes gives 1.178 and 1.193 GB/s, respectively. These are **whole-delivery-envelope rates, not exclusive PCIe H2D bandwidth**: no device-side activity trace isolated H2D completion time from storage and graph execution.

Dependency waits were negligible (35–38 microseconds aggregate). Scatter was stable at 0.234 seconds. The evidence is already decisive without pretending that additive terms are serial or calling the envelope an H2D-only interval.

The preregistered 31.419-second estimate was an additive dependency estimate, not a demonstrated physical floor. The corrected accounting is 32.015 seconds before overlap and about 22.39 seconds when applying the conservative 9.623-second overlap already measured in X7. Neither estimate predicted the actual answer: the A/B measured it, and gfx1201 queued delivery-plus-compute service dominated at 63.2 seconds.

## Invalid diagnostics retained

The failed development attempts are not promoted as arms, but they close three useful root causes:

1. The first candidate incorrectly failed closed when the calibrated plan contained legitimate exact-route fallbacks. That condition was repaired without changing the frozen plan.
2. Mixed calibrated/fallback tokens initially selected source-device rows for the gfx1151 join instead of returned mirror rows. The final four-arm sequence restarted after the fix.
3. Raw direct peer return then produced repeatable wrong layer/logit hashes. A host-stage discriminator kept gfx1201 arithmetic unchanged and restored byte identity: layer-1 SHA-256 `6008b002...` and final-logit SHA-256 `20d9dabc...` both matched the gfx1151 control. Thus the current gfx1201 IQ1_S arithmetic is byte-exact for this fixture; the raw peer-return mechanism caused the corruption.

## Decision boundary

> A valid Lane A NO-GO closes only the complete gfx1201 prompt-provider topology at M64 with the current IQ1_S kernels and all macro-union payload crossing PCIe. It does not close concurrent dual-owner expert execution, persistent hot-subset residency, a wider-M crossover, or a different gfx1201 kernel/representation.

Do not run M1024 and do not construct another complete provider. The next permitted discriminator is a model-free concurrent dual-owner graph census over the recorded expert jobs at widths 64/128/256/512/1024. Only a census that predicts a material concurrent partition earns hardware measurement.
