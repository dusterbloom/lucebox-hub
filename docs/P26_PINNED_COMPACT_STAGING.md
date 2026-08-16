# P26 — Persistent pinned compact staging

VERDICT: GO

## Question

Does replacing P25's per-expert pageable compact allocation with a persistent
page-locked host buffer improve the real all-92 calibrated96 path without
changing one model output bit?

## Change

P26 is opt-in through:

```text
DFLASH_KIMI_P23_PERSISTENT_SCRATCH=1
DFLASH_KIMI_P25_COMPACT_UPLOAD=1
DFLASH_KIMI_P26_PINNED_COMPACT=1
```

Each persistent sparse-expert geometry retains a reusable CUDA host allocation
beside its device compact staging allocation. Selected records are packed in
the same order into that page-locked buffer, uploaded once, scattered to the
same native positions, and evaluated by the unchanged native-width graph.
P25's pageable path remains available and the default remains disabled.

Four cumulative timers were also added for compact host packing, compact
upload/scatter, native expert graph execution, and output readback.

## Semantic gate — MEASURED PASS

Two-row, eight-row, and 32-row pinned outputs are byte-identical to their P25
pageable references. The 32-row trace hash is:

```text
SHA-256 c1330897e6819c1b0ae289155e76a2359fca1c55a95b44985cb91e179d3f2e44
```

All 32 generated IDs are identical.

## Eight-row substage attribution — MEASURED

| cumulative substage | pageable | pinned | change |
|---|---:|---:|---:|
| host compact pack | 4.513 s | 3.206 s | -29.0% |
| compact upload + scatter | 3.164 s | 2.597 s | -17.9% |
| native expert graph | 0.476 s | 0.478 s | unchanged |
| expert result readback | 0.181 s | 0.186 s | unchanged |
| complete expert-provider stage, 7 decode rows | 21.250 s | 20.136 s | -5.2% |

The adjacent eight-row wall time was inconclusive because routed preparation
varied oppositely: pinned decoded in 31.806 seconds and pageable in 31.173.
This is why the longer gate below determines the verdict.

## 32-row decision gate — MEASURED GO

| window | pageable P25 | pinned P26 | throughput change |
|---|---:|---:|---:|
| all 31 transitions | 0.2121/s | **0.2240/s** | **+5.6%** |
| final 16 transitions | 0.2070/s | **0.2226/s** | **+7.6%** |
| final 8 transitions | 0.2091/s | **0.2228/s** | **+6.6%** |

Total decode time fell from 146.182 to 138.368 seconds. Board-sampled process
energy fell from 19.053 to 18.008 kJ (-5.5%). Peak VRAM remained 16,035 MiB.
Peak shared RSS rose by about 47 MiB, consistent with the pinned buffers;
anonymous RSS was lower and total file-backed RSS varied with page-cache state.

Selected direct-I/O time was essentially unchanged (24.860 versus 24.891
seconds), as expected. Authoritative H2D bytes, logical provider bytes, slab
decisions, fallbacks, routes, native graph arithmetic, logits, and generated
tokens are unchanged.

## Interpretation

P26 earns a merge as a small, reproducible integrated gain and as the required
substrate for asynchronous delivery. It also confirms that the remaining
calibrated native expert graph math is cheap: across all 32 rows it consumed
1.851 seconds, versus 12.770 seconds packing, 10.232 seconds in upload/scatter,
and 24.891 seconds in selected direct reads.

Next: read/coalesce selected sidecar records directly into reusable pinned
route buffers so the host repack disappears, then overlap route N+1 delivery
with route N execution while retaining deterministic final accumulation.
