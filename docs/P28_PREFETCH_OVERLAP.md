# P28 — Oracle delivery-overlap ceiling

VERDICT: TRACE STRONG GO; INTEGRATED ORACLE REPLAY REQUIRED

## Question

How much P27 latency could disappear if selected expert-prefix reads were
overlapped with useful work, without changing the recorded calibrated96
requests or deterministic expert-ID accumulation?

This result is a CPU-only analysis of the archived P27 32-row trace. It did not
run a predictor, modify K3, or occupy the GPU. Every speed result below is
**PROJECTED FROM MEASURED P27 TIMERS**, not an integrated speed measurement.

## Reproduction

```bash
python3 scripts/analyze_kimi_p28_overlap.py \
  --trace /mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816/io_trace.tsv \
  --traffic /mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816/traffic.tsv \
  --stderr /mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816/stderr.log \
  --p27-results results/k3_p27_direct_pinned_compact.json \
  --output results/p28_oracle_overlap_ceiling.json
```

The analyzer verifies all 32 rows × 92 routed layers and 40,594 calibrated
routes. It uses the provider traffic ledger for authoritative/H2D bytes, the
I/O trace for route and layer geometry, and the P27 footer for actual physical
bytes. Exact-fallback work stays in an opaque serial remainder because P27 does
not time native fallback fetches per route.

## Measured input

| item | P27 measurement |
|---|---:|
| 31-transition decode | 93.107 s / 0.3330 transition/s |
| selected direct reads | 26.993 s |
| upload + CUDA scatter | 10.359 s |
| selected native expert graph | 1.867 s |
| result readback | 0.842 s |
| opaque serial remainder | 53.045 s |
| selected authoritative bytes | 4.529 GiB/row |
| exact-fallback logical bytes | 1.256 GiB/row |
| mean/max selected layer payload | 50.41 / 59.67 MiB |

The I/O trace and P27 footer both record 156.479 GB of aligned direct sidecar
submissions; authoritative H2D is 155.611 GB. Buffered mean-card reads are
excluded from this direct-read stage instead of being misclassified as compact
expert payload.

## Oracle ceilings

| schedule | hidden read | lower-bound decode | projected rate | gain | extra pinned memory |
|---|---:|---:|---:|---:|---:|
| current P27 | 0 | 93.107 s | 0.3330/s | — | — |
| within-layer route pipeline | 11.615 s | 81.491 s | 0.3804/s | **14.3%** | ≤59.72 MiB |
| one-layer oracle read lookahead | 26.984 s | 66.122 s | 0.4688/s | **40.8%** | ≤59.72 MiB |
| two-layer oracle read lookahead | same | 66.122 s | 0.4688/s | 0% beyond one | ≤119.33 MiB |

The within-layer trace replay preserves a barrier after each routed layer and
pays route-pipeline startup/drain 2,944 times. It misses the 15% GO gate narrowly
(14.3%). Perfect one-layer route knowledge meets STRONG GO. Two-layer lookahead
has no further steady-state value because one layer already contains enough
useful work to cover the average selected read.

Byte-attributed selected-read exposure is 9.17 ms/layer on average and 10.85 ms
at both p95 and maximum. The measured non-read remainder averages 22.46
ms/layer. This supports the one-layer ceiling, but per-layer non-read timing was
not captured, so only an integrated oracle replay can prove that the overlap is
real rather than resource-contended.

All oracle cases request exactly the archived bytes, so speculative overfetch is
zero. A future learned predictor would have to report its own wasted bytes.

## H2D qualification

P27 exposes upload and CUDA scatter as one 10.359-second timer. The trace cannot
separate DMA from the scatter kernel, and the scatter may contend with native
GPU work. The conservative ceilings therefore hide **zero H2D time**.

If both read and the entire upload/scatter timer were free, the mathematical
upper bound would be 55.767 seconds or 0.5559 transition/s (+67.0%). This is an
upper bound, not a forecast. An integrated double-buffer experiment must split
DMA and scatter timing before claiming any of it.

## Control-room ceilings

The stage-free ceilings show why overlap matters but cannot deliver 4 tok/s by
itself:

| stage made free | maximum speedup |
|---|---:|
| selected read | 1.408× |
| upload + scatter | 1.125× |
| selected graph | 1.020× |
| readback | 1.009× |
| read + upload/scatter | 1.670× |

Even the optimistic delivery-free rate is only 0.556/s under the current
roughly 5.8-GiB/row provider policy. Adaptive-byte reduction and verification
amortization remain necessary.

## Decision

1. Do not ship a within-layer-only optimization from this projection: it misses
   its preregistered GO gate.
2. Build the smallest trace-driven one-layer oracle replay with one extra layer
   buffer. Reads may complete out of order, but expert outputs must accumulate
   in frozen expert-ID order.
3. Instrument H2D and scatter separately and test byte identity before timing.
4. Do not build a predictor until integrated one-layer oracle gain reaches 25%.
5. Do not pursue two-layer lookahead: it doubles buffering and has no ceiling
   advantage on this trace.

P28 is worthwhile as a bounded systems optimization. It is not a substitute
for H23's target of materially fewer authoritative bytes.

## Integrated oracle implementation — BUILD-READY, NOT YET MEASURED

The minimum one-layer replay is now implemented behind:

```text
DFLASH_KIMI_P28_ORACLE_TRACE=<archived-P27-io_trace.tsv>
```

It is accepted only with the complete P27 opt-in stack. At startup it reduces
the archived trace to `(position, layer, expert, natural slabs)` addresses. For
each live layer it:

1. waits for the one-layer-ahead read, if present;
2. verifies every unique live expert and selected natural slab against the
   frozen oracle;
3. consumes the pinned compact payload only on an exact address match;
4. otherwise counts the read as wasted and executes unchanged synchronous P27;
5. launches exactly one future layer into the other buffer;
6. leaves mean-tail arithmetic, native graph execution and final expert-ID
   accumulation unchanged.

Duplicate native route slots are handled as one unique expert address without
changing their live router arithmetic. The additional buffer is one contiguous
pinned arena which grows only to the maximum future selected-layer payload; the
runtime reports its actual high-water capacity.

The implementation and CPU controls compile and pass. The integrated GPU gate
has deliberately not run while H23/S0 own the shared GPU. Reproduce it with:

```bash
scripts/gpu_lease.sh run P28 -- \
  scripts/run_kimi_p28_oracle.sh \
  /mnt/kimi-k3/results/kimi-p28-oracle-32-row-YYYYMMDD 32
```

`scripts/analyze_kimi_p28_integrated.py` requires byte-identical frozen logits
and at least 25% measured throughput gain before it labels predictor research
earned. Until that command passes, the implementation status remains OPEN and
the 40.8% value above remains only a ceiling.
