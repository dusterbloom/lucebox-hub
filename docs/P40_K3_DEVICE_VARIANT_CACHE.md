# P40 — calibrated device-variant cache trace gate

## Verdict

**DEVICE OVERLAY BYTE-EXACT GO; THROUGHPUT MARGINAL; P30 RETIREMENT NO-GO;
P41 COMPACT EXECUTOR PROMOTED.**

The Lucebox4 P39 I/O trace can reconstruct every calibrated request's exact
natural-slab mask. It shows that caching expanded expert weights on the GPU is
useful for avoiding upload/scatter, but too coarse to replace the 8 GiB P30
host slab cache.

At 8 GiB, a global LRU device cache avoids 6,569 of 17,917 sidecar scatter
calls and covers 46.94% of requested slabs. After the existing P30 cache,
however, those hits overlap heavily with host hits and remove only an
indicative 1.807 GiB, or 7.12%, of the remaining physical sidecar reads.

The runtime A/B confirms the overlay is safe but not transformative. Across
the 12-prompt Lucebox4 suite, all P40-on/off logits are byte-identical. P40
improves the canonical true decode rate only 2.84%, from 1.3896 to 1.4290
transitions/s, despite cutting aggregate scatter time 44.04%. The much larger
43.3% short-fact result did not reproduce once the host was warm and is
classified as cold-start/order confounded.

The implementation order is therefore:

1. keep P30 as the fine-grained host/NVMe tier;
2. add common device-resident variants as an overlay for scatter/H2D removal;
3. measure the overlay before changing its admission policy;
4. prioritize a compact fused expert executor that preserves slab granularity;
5. delete P30 only if a later compact device tier matches its byte coverage.

## Trace and reconstruction

The source is the uncached P39 GPU1 repeat:

```text
/home/duster/kimi-k3-deploy/
  p39-gpu1-core-dualarch-repeat-20260818/io_trace.tsv
```

- SHA-256: `70e26818cdfc67d32fc219fc983d526d06f5c4a6fa6e7cb9cc1663b1915497a7`
- rows: 365,687 requests plus one header
- bytes: 76,016,268
- model positions: 42

For each sidecar `gate` row, the natural slab is recovered from the validated
sidecar header:

```text
expert_base = payload_offset + expert_id * record_bytes
natural_slab = (file_offset - expert_base) / slab_bytes
```

Grouping by prompt, base position, token index, model layer, expert and exact
fallback yields 17,917 nonempty sidecar route events. Every event contains
exactly `prefix_depth` distinct natural slabs; there are no ambiguous groups.
`traffic.tsv` alone is not sufficient because it has no per-route offsets.

The cache key is the immutable sidecar artifact/layout domain, local routed
layer, expert and complete MoE numerical specification. One key owns a mutable
12-bit resident-slab union: subset requests are hits and supersets extend the
same entry. Exact fallback requests all twelve bits while retaining its
existing execution and accumulation semantics.

## Geometry

| Layer qtype layout | Expanded expert bytes |
|---|---:|
| IQ1 / IQ1 / IQ1 | 6,451,200 |
| IQ2_XXS / IQ2_XXS / IQ1 | 7,827,456 |
| IQ1 / IQ1 / IQ2_XXS | 7,139,328 |

The sizes include the observed 128-byte backend alignment. The simulation
uses each layer's actual layout rather than dividing capacity by one average.

The denominator is 17,917 events, 101,532 requested slab records and 6,889
distinct layer/expert keys. It includes 17,184 calibrated events and 733
sidecar-authoritative exact fallbacks.

## Global LRU result

| Capacity | Full hits | Partial extensions | Cold/reloaded | Slab coverage | Evictions |
|---|---:|---:|---:|---:|---:|
| 2 GiB | 163 (0.91%) | 43 events / 170 slabs | 17,711 | 1.35% | 17,388 |
| 8 GiB | 6,569 (36.66%) | 1,241 / 5,862 | 10,107 | 46.94% | 8,806 |
| 16 GiB | 8,534 (47.63%) | 1,342 / 6,077 | 8,041 | 57.94% | 5,436 |

A full hit avoids one destination clear, H2D/scatter submission and
synchronization. A partial extension uploads and scatters only missing natural
slabs without clearing resident data. The 8 GiB run holds 1,301 variants at
the end and loads 27.709 GiB of logical sidecar payload across the trace:
24.711 GiB on cold/reloaded entries and 2.998 GiB on extensions.

A trace-ranked pinned-hot policy improves 8 GiB to 7,859 full hits (43.86%)
and 54.81% slab coverage, but this is an upper bound unless an independent
calibration profile reproduces the ranking. It is not the first runtime
admission policy.

## Interaction with P30

| Path | Aligned physical sidecar traffic |
|---|---:|
| P39 uncached | 52.423 GiB |
| P39 P30 8 GiB | 25.389 GiB |
| P30 reduction | 51.569% |

The 8 GiB device LRU overlaps with P30's hits and suppresses only an
indicative 1.807 GiB (7.12%) of the recorded P30 physical sidecar traffic.
This is not a coupled-cache forecast because the two traces were simulated
independently.

The stronger benefit is removing 36.66% of scatter calls or 46.94% of slab
work. Applied only as a bound to P39's measured 5.840529321-second aggregate
scatter counter, that represents roughly 2.14–2.74 seconds of removable
aggregate work. It is not yet a wall-clock or decode-throughput claim.

## Measured runtime gate

The opt-in common cache uses the same stream-engine LFRU pool and requires an
explicit 8 or 16 GiB allocation. Its identity contains the immutable source
domain/generation, local routed layer, expert and complete numerical spec. A
12-bit resident union supports full hits and missing-bit extensions; pending,
pinned and executing slots cannot be evicted.

The broad adjacent pair used the same HIP dual-architecture runner, GPU1 core,
8 GiB P30 cache, 8 GiB device pool, 1.22068 logical GiB/model-position policy,
552 prompt positions and 129 true decode transitions. Only P40 differed.

| Broad 12-prompt result | P40 off | P40 on | Change |
|---|---:|---:|---:|
| True AR transitions/s | 1.389576 | 1.429024 | **+2.84%** |
| Prompt positions/s | 1.313629 | 1.384192 | **+5.37%** |
| End-to-end wall | 534.175 s | 506.881 s | **−5.11%** |
| Sampled GPU energy | 37,060 J | 34,993 J | **−5.58%** |
| Direct physical bytes | 407.292 GB | 337.256 GB | **−17.20%** |
| Aggregate scatter time | 78.408 s | 43.875 s | **−44.04%** |
| Aggregate expert-graph time | 26.883 s | 28.694 s | **+6.74%** |

P40 served 275,008 route events with 70,013 full hits, 672 extensions,
204,323 cold fills, zero unavailable entries, zero aborts and zero fallbacks.
The full-hit rate was 25.46% and 29.56% of requested slabs were already
resident. All twelve P40-on/off logits traces, token sequences and texts are
byte-identical. Both arms retain 11/12 registered tasks: `code-function` ends
at `x %` under the 24-token cap before emitting the required `2`. Since the
failure and its logits are identical with P40 disabled, it is a HIP/GPU1
baseline difference rather than a cache regression. `LIME-742` and
`QUARTZ-918` remain exact.

The ordered broad pair is enough to reject a large throughput claim, not to
establish a statistically precise 2.84% gain. The earlier short pair measured
1.4268/s on versus 0.9959/s off, but the later warm P40-off broad rows return
to roughly 1.4/s. That short result is retained only as cold-start evidence.

The overlay therefore passes identity, lifecycle and useful-work gates, but
does not materially close the 10/s target. P30 remains the lower tier and P41
must remove the expanded graph rather than merely cache it. The smallest
future trace-schema improvement is still to persist `route_index` and
`natural_slab` directly in `io_trace.tsv`.

## Compact-executor contract

The promoted executor must not perform an ordinary `S*256`-wide down
projection or sum twelve independent 256-wide projections. Both change the
FP32 association that H17 localized as the first source of route divergence.
The smallest credible same-device exact path is:

1. retain P30's natural sidecar records and aligned reads;
2. form a component-major compact image: a 32-byte natural-ID header, then all
   selected gate slabs, all selected up slabs and all selected down slabs;
3. reuse ordinary GGML MMVQ for compact gate/up and the existing unfused SiTU
   graph;
4. add one generic sparse-K down MMVQ that presents virtual `K=3072`, maps each
   natural 256-wide block to its compact slot, and preserves the native block,
   lane, cross-wave and warp reduction order;
5. keep down scaling, mean-tail correction, router weighting and global expert
   accumulation unchanged.

Component-major layout is required because mixed IQ1/IQ2 record strides are
not necessarily divisible by the gate/up quant block size. IQ1_S and IQ2_XXS
both have `QK=256`, so one natural slab is exactly one quant block per down
row. Missing natural blocks contribute positive zero in their native schedule;
selected blocks never shift lane ownership.

At the observed 5.91 slabs per calibrated event, this retains about half of
the 6.45--7.83 MiB expanded expert footprint and removes three full-weight
clears, the full-width scatter and its activation-mask upload. It does not
change authoritative selected bytes.

The initial generic op/kernel plus Kimi integration is projected at 400--580
new production code lines. Promotion then makes roughly 430--600 lines of the
scatter/evaluator/repacking machinery deletable, for an expected net of about
`-150..+100` pure production lines. That reduction is earned only after
same-device byte parity on CUDA, `gfx1201` and `gfx1151`, the frozen fact and
12-prompt gates, and counters showing zero full-weight clears/scatters.

The executor boundary must be `eval_into(device_output)`. That allows the
later PR600-style R9700/Strix split to place every expert result in global
route order and perform one destination-device ordered join. Owner-local
partial sums are forbidden when bit identity is requested because they change
association; mean-tail corrections enter the same global ordering.

Machine-readable simulation and runtime results are in
`results/k3_p40_device_variant_cache_sim.json` and
`results/k3_p40_device_variant_cache_runtime.json`.
