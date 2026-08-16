# P29 — K3 CPU boundary profile and thread control

VERDICT: THREAD CONTROL GO / GRAPH-LIFECYCLE OPTIMIZATION NO-GO

## Question

After P27 reduced the steady transition to roughly 2.84 seconds, is the
remaining routed preparation dominated by graph lifecycle and host boundaries,
or by the actual CPU graph computation?  Can the existing runtime improve by
using fewer CPU workers on this WSL host?

## Opt-in profiler

`DFLASH_KIMI_BOUNDARY_PROFILE=1` splits every existing host-boundary graph into
graph expansion, allocation, input transfer, graph compute, and output
transfer.  It changes no tensor, execution order, or default behavior.

The eight-row profiler run reproduced the frozen eight-row P27 logits exactly:

```text
SHA-256 8daa924c13dd94489541f5d259eb2b72873b9cd49a074aee348aecd5dae90ca7
```

On stable rows, all 92 routed-preparation boundaries together spent about
`2.7 ms` allocating, `1.0 ms` copying inputs, `0.45 ms` reading outputs, and
`1.02–1.13 s` in `ggml_backend_graph_compute`.  Actual graph computation was
more than 99.5% of the measured boundary.  Slow refault-sensitive rows showed
the same classification: the time grew inside graph compute, not allocation or
host copies.

Therefore persistent graph metadata or allocator work cannot materially move
the whole-transition rate.  The next compute investigation must profile the
KDA/MLA/AttnRes/router work by layer and include mapped-weight residency.

## Recovered per-layer atlas

The frozen eight-row boundary log already contains one routed-preparation
record per layer in deterministic model order.  The checkpoint metadata gives
the exact per-layer KDA/MLA map.  `analyze_kimi_p29_layer_profile.py` joins
those sources without another model run; it skips the first row and reports
seven rows.

| family | routed layers | CPU preparation weights | sum of per-layer median compute | robust time share |
|---|---:|---:|---:|---:|
| KDA | 68 | 24.882 GiB | 767.5 ms | 83.0% |
| MLA | 24 | 4.868 GiB | 157.0 ms | 17.0% |

The complete preparation working set is approximately 29.75 GiB after
excluding the expert bank and the latent/shared/join families already placed
on CUDA.  Median KDA boundaries are about 11.17 ms and MLA boundaries about
6.36 ms.  Large row-to-row means come from mapped-weight refaults; medians are
the useful placement statistic.

An optimistic greedy ranking by median milliseconds per resident GiB gives
the following hard upper bounds before CUDA execution, state transfer,
synchronization and arithmetic differences:

| additional GPU capacity | layers fitting | perfect CPU time removed |
|---:|---:|---:|
| 4 GiB | 15 | 133.8 ms |
| 6 GiB | 22 | 199.6 ms |
| 8 GiB | 28 | 263.0 ms |

This creates a hard capacity boundary.  Moving every KDA layer would require
about 24.9 GiB, beyond the free RTX 3090 capacity.  A hypothetical perfect
8-GiB placement can remove only roughly 0.25 seconds of the current transition
before CUDA execution and synchronization are counted.  Selective attention
placement is consequently a secondary single-digit/low-double-digit-percent
opportunity, not the route to 4 token/s.  It also needs a behavioral gate
because changing CPU/CUDA arithmetic may change logits.

The attribution remains graph-level: each boundary includes AttnRes mixing,
KDA or MLA, normalization and the CPU router.  It does not pretend to assign
time to individual operators inside that graph.

## Adjacent CPU-thread control

The machine exposes 18 logical processors as nine cores to WSL.  Three
adjacent 16-row P27 runs used 9, 12, and 18 CPU workers.  Every logits file was
byte-identical:

```text
SHA-256 6a35b1183b0d0a476f10ced598974059e0b827247a253e10cb0d6bac76f307a9
```

| CPU workers | 15-transition decode | final-eight rate | routed preparation | expert provider |
|---:|---:|---:|---:|---:|
| 9 | 41.027 s | 0.3622/s | 0.888 s | 1.720 s |
| **12** | **40.910 s** | **0.3696/s** | **0.877 s** | **1.678 s** |
| 18 | 45.455 s | 0.3391/s | 1.023 s | 1.773 s |

Twelve workers improve the adjacent final-eight rate by 9.0% over 18 with no
semantic change.  Nine and twelve are close; twelve wins this trace and is the
current machine-specific setting to take to a longer control.  This does not
establish a portable default for other CPUs.

## Control-room ceilings

Using the stable P27 final-16 transition (`2.8397 s`):

| stage made free | lower-bound transition | ceiling rate | maximum speedup |
|---|---:|---:|---:|
| selected direct reads | 1.996 s | 0.501/s | 1.42x |
| upload + scatter | 2.516 s | 0.397/s | 1.13x |
| native selected-expert graph | 2.781 s | 0.360/s | 1.02x |
| complete expert provider | 1.216 s | 0.822/s | 2.33x |
| routed preparation | 1.780 s | 0.562/s | 1.60x |
| provider and routed preparation | 0.157 s | 6.37/s | 18.1x |

No single current stage can yield four tokens per second.  The target requires
both fewer authoritative bytes and amortization/acceleration of the core
forward.  These are mathematical ceilings, not projected integrated rates.

## Next gate

1. Keep twelve CPU workers for the next same-machine control.
2. Use the recovered KDA/MLA layer atlas and add operator splits only if a
   candidate placement needs them.
3. Rank any selective accelerator placement by measured milliseconds saved per
   resident GiB, with an official-template quality gate because CPU/CUDA
   arithmetic can differ.
4. Do not write a fused expert kernel or persistent CPU graph: both have
   already lost their stage ceilings.
