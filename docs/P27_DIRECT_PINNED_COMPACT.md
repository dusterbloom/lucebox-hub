# P27 — Direct-read pinned compact routes

VERDICT: MAJOR GO

## Question

Can the layer-wide direct-I/O workers construct P25's exact compact expert wire
image directly in reusable pinned route buffers, eliminating the intermediate
per-slab vectors and second full host copy?

## Change

P27 is opt-in through:

```text
DFLASH_KIMI_P23_PERSISTENT_SCRATCH=1
DFLASH_KIMI_P25_COMPACT_UPLOAD=1
DFLASH_KIMI_P26_PINNED_COMPACT=1
DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT=1
```

For each calibrated route, one persistent pinned buffer holds the same 32-byte
natural-position header and compact gate/up/down records consumed by the P25
scatter kernel. The direct-read workers copy each aligned sidecar record into
its final compact offset. The evaluator uploads that buffer without allocating
three vectors per slab or repacking them into another vector.

Sidecar bytes, alignment, selected slab decisions, exact fallbacks, activation
mask, native graph, mean tail, expert order and output accumulation are
unchanged.

## Stop gate and correction

The first plumbing run correctly triggered all-layer exact fallback because the
scatter launcher still received the empty legacy vector's slab count. No speed
or quality result from that run was accepted. The launcher was changed to use
the prepacked payload's registered count, after which the two-row identity gate
passed before the 32-row run was allowed.

## Semantic gate — MEASURED PASS

P27 and P26 produced byte-identical two-row and 32-row logits. The complete
32-row hash and all generated IDs match:

```text
SHA-256 c1330897e6819c1b0ae289155e76a2359fca1c55a95b44985cb91e179d3f2e44
```

No runtime layer fell back because of P27.

## 32-row decision gate — MEASURED MAJOR GO

| window | P26 pinned repack | P27 direct pinned | throughput change |
|---|---:|---:|---:|
| all 31 transitions | 0.2240/s | **0.3330/s** | **+48.6%** |
| final 16 transitions | 0.2226/s | **0.3522/s** | **+58.2%** |
| final 8 transitions | 0.2228/s | **0.3507/s** | **+57.4%** |

Total decode time fell from 138.368 to 93.107 seconds (-32.7%). Sampled board
energy fell from 18.008 to 12.123 kJ (-32.7%). Peak VRAM remained 16,035 MiB.
Peak shared RSS rose by about 87 MiB for the 16 reusable route buffers;
anonymous RSS was lower and file-backed RSS remained page-cache-sensitive.

## Attribution — MEASURED

Across all 32 rows:

| cumulative stage | P26 | P27 | result |
|---|---:|---:|---:|
| selected direct read | 24.891 s | 26.993 s | +8.4% |
| second host repack | 12.770 s | **0.005 s** | eliminated |
| upload + scatter | 10.232 s | 10.359 s | unchanged |
| native calibrated graph | 1.851 s | 1.867 s | unchanged |
| result readback | 0.709 s | 0.842 s | small/noisy |

The direct read becomes slightly slower because its single copy now targets
pinned memory. That cost is overwhelmed by eliminating the second full copy and
the allocation/destruction of hundreds of thousands of gate/up/down slab
vectors. The late expert stage falls from 3.114 to 1.623 seconds/transition.

## Interpretation

The selected K3 expert math was not the dominant calibrated96 cost. Object
lifecycle and redundant host movement were. P27 raises the working all-92 path
from the original P20 0.0371 transition/s to 0.3330/s overall and 0.3522/s in
the final window—about a ninefold measured gain—without changing the frozen
model outputs.

Next: reuse aligned O_DIRECT worker scratch instead of allocating it once per
slab, then overlap layer-wide reads with per-route upload/execution while
preserving deterministic final accumulation.
