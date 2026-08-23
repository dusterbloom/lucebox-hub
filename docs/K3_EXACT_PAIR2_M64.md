# K3 exact pair2 at M=64

## Verdict

**EXACT, PHYSICAL-BYTE-CLEAN, PERFORMANCE NO-GO. DO NOT COMMIT OR ENABLE
STANDALONE PAIR2.**

The default-off candidate reused one authoritative compact payload across two
rows with the same complete source/spec/expert/natural-slab key. Gate/up ran as
one width-two MMVQ graph; down remained two independent one-row sparse-K nodes;
each result rejoined at its original canonical route position. This changed no
quantization, route, state or storage layout.

The post-repair M=64 result is fully exact and removes substantial internal
work, but it misses the registered materiality gate:

| M=64 verify arm | matched control | exact pair2 | change |
|---|---:|---:|---:|
| expert stage | 46.600748 s | 43.073502 s | -3.527246 s (-7.57%) |
| verify + exact commit | 68.891173 s | 65.370010 s | -3.521163 s (-5.11%) |
| committed rate | 0.929001 pos/s | 0.979042 pos/s | +5.39% |
| aggregate graph | 23.449742 s | 18.609678 s | -4.840064 s |
| authoritative H2D | 673.159 GB | 568.783 GB | -104.377 GB |
| direct physical bytes | 286,156,455,936 | 286,154,293,248 | -2,162,688 |

The keep gate required at least **5.0 seconds** removed from the expert stage.
The qualified saving is **3.527 seconds**, short by **1.473 seconds**. The code
therefore earns neither a production commit nor a broader throughput claim.
The earlier 39.998-second candidate expert result is not promoted: it incurred
one extra 540,672-byte P30 miss and failed the registered physical-byte gate.
There was no favorable rerun after the post-repair performance miss.

## Exactness and execution closure

The corrected dual-architecture Release binary has SHA-256
`ca91e02c28f408ef8fa5a14cc3a3c8a90fc02fc742cc61d79998daf20b66e72b`.
It ran on Lucebox4 physical GPU1 (`gfx1151`) with the core and routed expert
path on the same device, `platform_profile=performance`, the retained ROCm
7.2.4 HIP/HSA/COMGR overlay, `ROCBLAS_USE_HIPBLASLT=0`, a 16-GiB P30 host
cache and the authoritative direct-pinned compact path. GPU0 did not own any
expert work.

Both M=8 and M=64 passed:

- bit-identical logits and argmax;
- zero max-absolute and relative-L2 logit delta;
- identical recurrent and MLA hashes, including per-layer state gates;
- identical logical provider traffic;
- every compact job completed; and
- zero compact fallback or invalid jobs.

The M=64 candidate grouped 39,596 of 68,877 compact routes into 19,798 pairs.
It avoided 104,376,616,960 payload H2D bytes. The P30 repair recorded 189,170
no-copy logical touches covering 104,957,747,200 aligned bytes while preserving
the ordinary read on any cache miss. Swap stayed at 12 KiB; `pswpin`,
`pswpout` and `oom_kill` did not move.

## Why this stops

Pair2's 4.840-second graph reduction becomes only a 3.527-second expert-stage
reduction. Even eliminating all observed non-graph loss would yield only 4.840
seconds, still below the registered 5-second gate. Planner cleanup alone cannot
promote this candidate, and 741 added versus 29 removed lines is too much
production surface for an unstable five-percent end-to-end diagnostic gain.

Do not sweep more pair ratios or rerun M=64. The retained P65 2,048-position
causal trace supplies a favorable H23 distributional check without another
model run. Across 32 M=64 blocks and 92 layers, 80.663% of pair groups can
collapse into quads, below the approximately 84% plausibility bound needed to
recover pair2's measured 1.473-second gate shortfall. Width four is therefore
also stopped. This proxy covers exactly the nonzero-mask compact candidates;
it is not relabelled as the actual fixed-96 M=64 distribution.

The complete pair path is removed. Its failed diff is recoverable for audit at
`/tmp/k3_exact_pair2_failed_031e7a3.patch`, but it is not committed to the
product branch. Next work moves to the already-existing expert-major
multi-row executor and a genuinely wide layer-major prompt regime where rows
per expert are materially larger. The complete machine-readable record is
`results/k3_exact_pair2_m64.json`.
