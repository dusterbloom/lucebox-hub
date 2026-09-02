# K3 experiment scratchpad — 2026-09-01

## Decisions to retain

- The route12/Budget16 V16 single-call oracle is closed **NO-GO** before
  timing: the persistent Core8 implementation still accepts exactly eight
  hidden rows and rejected the width-16 vector at layer 0. No V16 target graph
  executed. Do not relax another guard or tune this fixture. A future attempt
  must preregister composition of two state-chained Core8 calls.
- Freeze the candidate IDs from a scalar route12/Budget16 run. Do not compare
  the V16 target against a native trajectory after the approximate policy has
  diverged.
- Require all 16 target argmax rows, the next scalar continuation, terminal
  full-vocabulary logits, and the complete committed recurrent/MLA state to
  match the scalar control.
- V16 did not earn continuation. Resume the already-evidenced B16→B20 lane
  with the smallest incremental-hydration discriminator. Do not implement a
  broad progressive subsystem first.
- Separately screen the existing generic chunked delta-net helper at M64. Its
  prior Qwen state drift is a preregistered exactness risk. A failure closes
  only this helper inside the current Core8-grouped executor, not a true
  prompt-wide DPLR/FlashKDA implementation.
- Keep `perf/k3-production-ponytail` at
  `fac048c090c74e5f8f989bffcda3aadc0bc8c266` untouched. A narrowly passing
  primitive still needs production requalification before transplant.
- Treat Hy4 as a process prototype, not a universal tensor recipe. Standardize
  source/inventory/calibration/quality/package/performance records and allow
  every architecture to compile a different allocation.
- Close cascaded IQ1_S-to-STQ as the leading K3 complement: the frozen
  equal-byte holdout lost 12/12 rows. Native/highest-authority-source STQ is
  still untested and is the only STQ arm worth retaining.
- Use the final self-contained DeepSeek4 ROCmFPX artifact as the resident-model
  process control and K3 as the provider/BWS control. Learned codebooks and
  modes are authoritative serving data, not optional sidecars.
- Qualify ROCm 10 side by side against 7.2.2. Do not upgrade production based
  on release notes, and do not infer gfx1151 GPU-direct storage from hipFile's
  Instinct-only support listing.
- Preserve the staged executor rule: reference dequant+dense first; earn a
  compressed GEMV/MMQ only from an end-to-end bottleneck and equal-output gate.
- Make GSQ refinement of an existing standard GGUF format a mandatory
  equal-byte codec arm. It is attractive precisely because it does not require
  a new decoder or kernel.
- Do not implement full RCO for K3 yet. Its non-decomposable terminal objective
  needs a differentiable evaluator or a held-out validated terminal surrogate;
  until then use the smallest discrete allocator over measured utilities.
- Treat an RCO allocation as compiled evidence, not a universal per-role
  recipe. Preserve the candidate database, exact budget, constraints, emitted
  assignment, and objective traces.

## Current provenance

- Experiment branch: `experiment/k3-terminal-kl-bws-v2`
- Committed experiment head before the Hy4/STQ patch:
  `87f9836f0b5c66c6796aa5c573e641b969996ae1`
- Production branch/head remains:
  `perf/k3-production-ponytail` at
  `fac048c090c74e5f8f989bffcda3aadc0bc8c266`.
- Hy4 repository revision:
  `779242edccdedc2109a0b36b164263a88f015bfa`.
- Hy4 STQ patch SHA-256:
  `b6deb5d1eda8cc241c417c28725c426dfe36d280ed744dc40ac9aa4472748ec1`.
- Hy4 tensor recipe SHA-256:
  `6ccb89ba093ece88becac2c920ca74e18cb332a27059243db1ca57407a6244d0`.
- K3 STQ held-out preregistration SHA-256:
  `21181d2ec117a71b9acbdc632b77b645ed066cf68b7e54d18652e6ad059a08d1`.
- K3 STQ held-out raw result SHA-256:
  `6d03d03c2219f6bbc632a7798355669ffd4bf8f4dbc815d56c3f1da25bb4aa0a`.
- K3 STQ held-out decision SHA-256 before commit:
  `acb1f221fab6c3ff4bcde16adaad8b09a274c540803cab31d9d7022dc7a20907`.
- Qwen3.8 GSQ-RCO artifact revision:
  `888cc868537099e09a9c4f41a2b9a421b346f88b`.
- Qwen3.8 GSQ-RCO README SHA-256:
  `37978f1c37de9d71a8d1612bf8f7e57e1792f018d1832775ee76cfe903cdfdd0`.
- GSQ code revision: `03fc16484c369e3127225615d5e03e8d3a6043e3`.
- RCO code revision: `9a1e09c07d468109cbe60a1b87d5036034a79d10`.
- GSQ-RCO immutable source-review SHA-256 before commit:
  `bfce83e90e2e816bc6ee23fdcab579e52b0b5c4e0b9ca4c183857a3a285ba875`.
- V16 preregistration SHA-256:
  `af210eaa8365af576b74743118943d96616cd5ed4df8c4479f1641ceeb8b0dd6`
- Before any V16 model run, the frozen prompt hash in that preregistration was
  found to be a transcription error. Preserve the original file unchanged and
  apply `results/k3_v16_route12_budget16_oracle_prereg_amendment_20260902.json`:
  the retained 53-ID q7 prompt hashes to
  `fd835f624d053f0f2da04114215461906430685463e99fb5e94cdf17115acbb0`.
  The target IDs, Budget16 policy, protocol, and gates are unchanged.
  Amendment SHA-256 before commit:
  `ee6044bf80ee37fb7548be5f401166e7a99a0b79e3ea314e6c20b06dcaadfedc`.
- V16 single-call NO-GO result SHA-256 before commit:
  `1f9642db69446b582af74f0719a02e634ea95bb3d673841ef3d84245ea4b52ba`.
- Chunked-KDA preregistration SHA-256:
  `d89d71b99cee4f46b1a82ae1c4d32aaedd4eb1b034ca0785186e3e8d4264df06`

## Verification status

- `git diff --cached --check`: PASS.
- Modified backend, graph, and calibrated-provider translation units: syntax
  compile PASS using the available local CUDA headers with only the existing
  HIP-only `cudaDeviceProp::gcnArchName` field mapped to CUDA's `name` field.
- Qualified HIP build: PENDING on Lucebox4.
- Lucebox4 measurement: NOT STARTED.

## External blocker

The local Tailscale daemon disappeared during the turn and later restarted.
The local client is now healthy enough to reach another direct peer, but
Lucebox4 still has no handshake or return traffic: Tailscale ping, Tailscale
userspace `nc`, and direct SSH all time out. No GPU work started. Resume after
Lucebox4 itself is online/reachable, then first verify whether the staged patch reached
`/home/duster/k3-terminal-kl-bws-v2-guarded` before applying it again.
## 2026-09-02 — incremental B16 to B20 discriminator

- Decision: test `B20 = B16 + sum(exact added slab - stored slab mean)` on frozen exact trajectories before implementing rescue state or caching.
- Do not re-litigate: B16 and B20 use one stable ranking; the only open question in this gate is finite-precision composition at terminal logits.
- Stop condition: any selected-set mismatch, terminal top-one change, or registered KL threshold failure closes the current composition order.
- Provenance correction: V1 mixed a p32-core exact-logit hash with the native-IQ1S model. Layer 1 is development-only; the immutable correction and held-out V2 preregistration preserve this explicitly.
- Held-out result: layers 46 and 92 produced byte-identical control/incremental terminal logits, KL `0`, and max logit delta `0`; earn only the all-layer on-policy equivalence gate.
- Do not re-litigate: these runs prove composition, not sparse delta reads, risk prediction, traffic savings, or throughput.
- Next registered gate: one ordinary eight-token sequence and the known tool first-token boundary, all layers on-policy, B20 control versus B16-plus-four composition. Exact output/logit equality is required; host-reference timings are non-evidence.
- On-policy result: strict NO-GO. Both outputs and the tool boundary survived, but logits were not identical, control/candidate KL reached `0.02794` on the tool row, and route drift changed fallback bytes.
- Positive retained evidence: candidate native-teacher KL improved on both rows (`4.12e-5` to `3.69e-5`; `0.65615` to `0.61775`) and the tool margin improved to `+0.34857`.
- Do not re-litigate: this reads all 20 records and saves no bytes. Fixed last-row/last-eight prompt rescue was already closed earlier; do not tune another static tail on the discovery fixture.
