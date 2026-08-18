# P35 — complete routed-layer preparation on the accelerator

**VERDICT: PRACTICAL GO, OPT-IN.**

P34 proved that one KDA layer is 4.27–5.04x faster on the RTX 3090, but
offloading KDA alone improved end-to-end decode by only 2.98%. P35 tests the
narrowest coherent boundary that can remove that penalty. For 18 late
recurrent layers it executes, in one accelerator graph:

1. both AttnRes mixtures and their normalization;
2. KDA with accelerator-resident recurrent state;
3. the post-attention prefix update and routed normalization;
4. the native router and routed latent projection;
5. the complete shared expert.

The graph returns only the prefix, routed latent, route IDs/weights, and shared
output required by the already-frozen streamed-expert boundary. The calibrated
1.22-GiB slab policy, mean tails, exact fallback, native 3,072-wide expert
graph, deterministic expert reduction, and routed join are unchanged.

The path is opt-in through `DFLASH_KIMI_COMPLETE_PREP_LAYERS`. It currently
supports ordinary one-token execution only. Replay/speculative batches and
divergence tracing fail closed instead of silently mixing CPU and accelerator
recurrent state.

## Isolated ceiling

Each real P32 layer benchmark includes its actual AttnRes checkpoint count,
all input uploads, graph construction/allocation, and every output readback
needed by the expert streamer. Each copied layer has 408,804,736 bytes of
weights in this measurement.

| model layer | checkpoints | CPU median | RTX 3090 median | speedup | router IDs |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 10.773 ms | 2.226 ms | 4.839x | 16/16 |
| 40 | 4 | 11.262 ms | 3.024 ms | 3.724x | 16/16 |
| 73 | 7 | 11.909 ms | 3.045 ms | 3.911x | 16/16 |
| 88 | 8 | 11.755 ms | 3.278 ms | 3.586x | 13/16 |

The layer-88 synthetic router change is the reason P35 requires a behavioral
gate. CPU and CUDA reductions are deterministic but not numerically identical.

## Capacity-safe integrated policy

The selected model layers are:

```text
68,69,70,72,73,74,76,77,78,80,81,82,84,85,86,88,89,90
```

They add 4.68 GiB of weights and 116 MiB of KDA state to the existing
latent+shared placement. Peak measured VRAM is 21,023 MiB on the 24-GiB RTX
3090. Accelerator state is explicitly cleared whenever the ordinary K3 cache
is reset; the 12-prompt run exercises eleven such prompt boundaries.

### Immediate same-binary bracket

The identical binary, prompt, model, slab policy, storage path, and CPU thread
count were run first without and then with the selected complete layers.

| metric | P32 bracket | P35 late-18 |
|---|---:|---:|
| median transition | 1,334.989 ms | **1,257.945 ms** |
| transition rate | 0.7491/s | **0.7949/s** |
| routed preparation | 690.778 ms | **553.875 ms** |
| accelerator preparation | 62.963 ms | 85.604 ms |
| expert provider | **478.884 ms** | 517.032 ms |
| join | 42.531 ms | **28.860 ms** |

The paired transition gain is **1.06125x**. The expert stage was noisier in the
candidate, so the gain is not a cache artifact in its favor. All 24 generated
tokens and 76/76 top choices match the bracket. Direct P32-to-P35 mean KL is
0.006543, with maximum 0.11147.

## Frozen 12-prompt gate

| metric | P32 | P35 late-18 | change |
|---|---:|---:|---:|
| native-success tasks retained | 12/12 | **12/12** | equal |
| token-identical to native | 9/12 | 8/12 | -1 |
| elapsed wall time | 934.820 s | **863.674 s** | **1.08238x** |
| true AR decode rate | 0.73425/s | **0.80063/s** | **1.09041x** |
| median staged transition | — | 1,242.652 ms | 0.80473/s |
| mean KL vs native | 0.69034 | 0.70482 | +0.01447 |
| peak VRAM | 16,063 MiB | 21,023 MiB | +4,960 MiB |
| energy | 111.787 kJ | 105.758 kJ | -5.39% |

The direct comparison against P32 is much tighter than either arm against the
native teacher: 11/12 generated sequences are token-identical, 655/675 aligned
top choices agree, and P32-to-P35 KL is mean 0.008731, median 0.002479, p95
0.038213, maximum 0.430667.

P35 therefore passes as a practical opt-in systems result. It is not
distributional equivalence and it is not yet a default. The frozen task suite
uses the registered thinking-off product template; official max-reasoning
quality remains open.

## What this changes

P34's negative result was about a split layer, not accelerator execution. P35
proves that keeping the coherent pre-expert half together converts the isolated
ceiling into a real, repeatable 6–9% gain. It also shows the remaining local
limit clearly: the measured P35 median still contains roughly 648 ms of routed
plus accelerator preparation and 501 ms in expert delivery. Extending the same
Q4_K placement to all 69 recurrent layers would exceed 24-GiB capacity, and
even a free routed-core stage would not by itself reach four tokens/second.

The next earned experiment is a **byte-bounded lower-precision KDA placement
ceiling**, tested first for held-out behavior and capacity. It should proceed
only if the complete all-recurrent placement plus the existing expert-provider
and width-four verification measurements form a credible path to the target.

## Reproduction

Isolated layer:

```bash
server/build-k3-p20-cuda126b/bench_kimi_k3_routed_preparation \
  /mnt/kimi-k3/models/kimi-k3-kda-q4k-p32/Kimi-K3-KDA-Q4_K-00001-of-00014.gguf \
  88 7 0 /mnt/kimi-k3/results/kimi-p35-routed-prep-layer88.json 12
```

Integrated frozen gate:

```bash
KIMI_P35_FIXTURE=server/test/fixtures/kimi_k3_h23_broad_native_success_v1.jsonl \
KIMI_P35_NATIVE_ROOT=/mnt/kimi-k3/results/kimi-h23-broad-native-v1-20260817 \
KIMI_P35_OUTPUT=/mnt/kimi-k3/results/kimi-k3-p35-complete-prep-broad12 \
scripts/run_kimi_k3_p35_complete_prep.sh
```

Machine-readable summaries:

- `results/k3_p35_complete_routed_preparation.json`
- `results/k3_p35_complete_prep_late18_broad12_quality.json`
- `results/k3_p35_complete_prep_vs_p32_broad12.json`
- `results/k3_p35_complete_prep_late18_broad12_stage.json`
