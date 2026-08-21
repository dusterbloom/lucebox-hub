# P58B-G0 K3 stage attribution

**Date:** 2026-08-21

**Status:** `INVALID_TELEMETRY_STRONG_NEGATIVE`

**Official timing decision:** null.

**Roadmap action:** no intervention is earned. Close the retained all96 mu=1 / registered perfect-mu=2 stage-attribution lane as a strong negative, without extending that result to P55, production verification, accepted-token throughput or K3 generally.

## Question and boundary

P58B-G0 asked whether the exact P58A micro/macro seam still had a credible path to ten oracle/verifier rows per second under its current research-only all96 provider. It reused the frozen P58A prompt `[18699]`, oracle `[11,374,4936,261,814,2742,316,374]`, base position 1 and width 8. The run compared eight sequential one-row transitions against one exact P58 width-eight verification plus committed state replay.

This was a stage-attribution harness, not a replacement implementation. It did not measure a mu=2 kernel, snapshot elimination, a new provider, P55/H23 density, generation, a proposer or production device-chain compatibility. Its two zero-provider cycles are optimistic arithmetic counterfactuals with zero replacement cost. They are not measured achievable cycles or speedups.

The run used the retained all96 sidecar policy at 4.665447235 logical GiB per position. P55/H23 uses 1.220680909 GiB per position; all96 is 3.822003933 times denser. No P55 or production inference is allowed.

## Qualification history

A fresh detached Lucebox4 worktree at `3932febd3f3de00fac88d7959728ec1f3fbb3164` built the reviewed four-file snapshot as HIP Release for `gfx1201;gfx1151`, with mixed CUDA/HIP off, CUDA off and HIP graphs off. Five local and five remote analyzer tests passed. Progressive-provider, ordered-join, MoE-stream and sparse-K tests passed serially on physical GPU0 and GPU1: eight of eight hardware gates.

Exactly one wrapper/model invocation then ran on physical GPU1 (`gfx1151`). It completed normally in 73.404557 seconds and wrote every registered artifact. Exactly one analyzer invocation followed. It exited 1 before producing an analysis JSON:

```text
P58B invalid: sampled target swap
```

There was no repeat.

Two independent telemetry gates are invalid:

1. `telemetry.json` reports target `peak_swap_kib=0`, and 128 numeric CSV samples report zero swap, but the telemetry runner appends a 129th post-exit row whose process fields, including `swap_kib`, are blank. The preregistered analyzer correctly requires every sampled value to be present and numeric, so the terminal race invalidates the timing evidence.
2. The wrapper's host-wide bracket records `pswpin` 223,552 to 223,553 and `pswpout` unchanged at 1,141,965: one 4-KiB page-in versus the exact zero-delta gate.

Cgroup `oom` and `oom_kill` remain zero. The target process itself reports zero peak swap. Those facts do not repair either failed gate. The official timing decision is therefore **null**, `intervention_earned=false`, and no 10-rows/s or realizable-speedup claim is made.

## Exact semantic result

The semantic run is complete even though timing qualification is not:

- logits and argmax rows are bit-identical;
- per-layer hidden capture was disabled by the exact P58 mode, so no hidden-row values or hidden-row equality claim are reported;
- recurrent state, all 93 convolution hashes, all 93 SSM hashes, all 93 MLA hashes and terminal MLA state match directly;
- recurrent hash is `3345846951683756339` in both arms;
- MLA hash is `16569152724683205385` in both arms;
- logical provider traffic is equal at 40,075,886,592 bytes per arm;
- sequential and verification arms each complete 8,504 compact jobs with zero fallback or invalid jobs;
- aggregate P41 completes 19,594 of 19,594 jobs with zero fallback or invalid jobs;
- the frozen traffic TSV has 92 layers, 1,656 layer-tokens, 158,976 requested/selected slab records, 26,156 calibrated routes, 340 exact-fallback routes and 89,915,965,440 provider bytes;
- the full traffic TSV SHA-256 is the frozen `db5ce71d8ecd1b78149ef9689a93926585a070fdb6d1eea6edd737fcf4b1ae77`.

The S0 timers report 18.964205608 seconds sequential, 12.932867354 seconds verification, 0.580892855 seconds commit and 13.513760209 seconds verification plus commit. This is `1.466357389x` verify-only and `1.403325597x` committed. These live all96 numbers are descriptive only.

## Descriptive counterfactual math

The stage log is printed to 0.001 ms. Its eight measured sequential rows and one P58 row give:

| Quantity | Raw value |
|---|---:|
| Sequential stage sum `S` | 18,962.901 ms |
| Sequential expert sum `S_E` | 16,108.232 ms |
| P58 total `B_T` | 12,932.818 ms |
| P58 expert `B_E` | 10,595.054 ms |
| State replay `B_R` | 580.892855 ms |
| Verify-plus-commit cycle `B_cycle` | 13,513.760209 ms |

The preregistered conservative current mu=1 zero-provider counterfactual is:

```text
C_mu1_zero = max(B_cycle, B_T + B_R) - B_E
           = 2,918.706209 ms
           = 2.740940481 rows/s
```

The registered perfect-mu=2 eligible interval is one-row core plus join plus output plus replay:

```text
E_mu2 = 2,192.821 + 102.984 + 36.429 + 580.892855
      = 2,913.126855 ms

C_mu2_zero = C_mu1_zero - 0.5 * E_mu2
           = 1,462.142782 ms
           = 5.471421876 rows/s
```

Both optimistic zero-provider cycles remain above the 800-ms ceiling required for eight oracle/verifier rows at 10 rows/s. The counterfactual mu=2 ratio is 1.996184125, and the sequential-to-batched provider ratio is 1.520353931. Those materiality signals do not earn an intervention because the timing run is invalid and neither replacement execution was measured.

For context only, the live all96 provider rate is 0.755069299 rows/s, the live verify-plus-commit cycle is 0.591989193 rows/s and the truncated stage sum divided by the cycle is 1.403229057. None is a production accepted-token rate.

## Provenance

Raw evidence remains on Lucebox4 at:

```text
/home/duster/kimi-k3-deploy/p58b-g0-stage-e9e9ced2-20260821
/home/duster/kimi-k3-deploy/p58b-g0-input-e9e9ced2-20260821
```

| Artifact | SHA-256 |
|---|---|
| Reviewed four-file snapshot | `e9e9ced20621b10d205fa77fc55e3892384b74cdde3f21895452b00c9c8b0273` |
| HIP oracle binary | `f43ecdf90b9c2acd6698d2802d3167bc92fee5fead6082a8432a025f3f07af00` |
| CMake cache | `9fc7457011bd1e2ab577de0eeee04827c3a21e10e747ea0e2d752025b9ee4fc6` |
| Checkpoint lock | `682d8ce6895311a90997146e821fbbf8c85750895412264c3f8c3a59db09545a` |
| Auxiliary manifest | `1189c85743a1e5f41901f09b090ea3a4a567c2020bce53400e9ae5f039409f56` |
| Sidecar manifest | `86216b66256f61bdcbe8d796dc2e9316e3dccdeba801ea296b2e779cf86394e9` |
| Wrapper run manifest | `ebf06732617d82e6f52d999f3bcf2de80f36ef0bcae9a381326c2f4b2d26d270` |
| S0 JSON | `2d0f05af5953d2888fbbbcef86ba5728fa3c2b289b089892a20f5fd864d50328` |
| Traffic TSV | `db5ce71d8ecd1b78149ef9689a93926585a070fdb6d1eea6edd737fcf4b1ae77` |
| Telemetry JSON | `9855beb732ff647f0d6497c7b70fdaa10882989e8215f7d5c95fcc6b90c6e34a` |
| Telemetry CSV | `5ba84a182510cc6bbfb8f30bc8282bca7a226527e8975f45101694c5af858ffa` |
| Child stderr | `8a361715665e78dd6a9580620c7838847324f27431503774ab053da233c6b15a` |
| Empty child stdout | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Analyzer rejection stdout | `1df4ca9ee45cd0832ad0a2fb0bd4a6e5716a3b4598e34d31e34400a38c6cb97e` |

Individual reviewed source hashes and the complete checkpoint/source/build identity are retained in the machine-readable result.

## Closure and next work

The entire temporary four-file, 970-line harness is deleted. No production C++, CMake behavior, retained P58A seam or model artifact changes.

P58B-G0 is `INVALID_TELEMETRY_STRONG_NEGATIVE`. It closes only this retained all96 mu=1/perfect-mu=2 stage-attribution lane as unearned. It does not close exact multi-token verification globally and does not authorize a proposer.

The next bounded decode experiment is a recorded-input, model-free `gfx1151` KDA component roofline tied to broad exact decode. The next cold-prefill work is offline P56 density, residency and two-drive simulation. Only a measured component with enough critical-path value should reopen implementation work.
