# P56R — Same-layer direct-read coalescing discriminator

## Verdict

**BYTE-EXACT STORAGE REPLAY, PERFORMANCE NO-GO. NO MODEL RUN. THE TEMPORARY
PRODUCTION COALESCER WAS REMOVED.**

P56 showed a large gap between physical slab-read events and adjacent aligned
ranges. P56R tested whether collapsing that request count inside the existing
one-token, one-layer dependency boundary could recover part of K3's expert
service time without changing selected bytes, cross-layer scheduling, or
arithmetic.

The planner exactly reconstructs the frozen P56 prefill trace. It reduces
524,863 physical requests to 187,654 spans, a 64.25% request reduction, but
reduces bytes by only 0.39%: 291,245,891,584 to 290,115,694,592 bytes. A
100-layer-group verification replay reconstructs every original aligned read
from the merged spans and produces the same SHA-256,
`234b33708124548218f8f7d70ce6be853d8ab68e9af42757de1df009a5f178bd`.

## Lucebox4 O_DIRECT A/B/A

The full replay uses the real sidecar files on Lucebox4, `O_DIRECT`, aligned
anonymous buffers, `preadv()` and queue depth 16. Groups are never merged
across prompt positions, model layers or files.

| arm | requests | bytes | seconds | submitted GiB/s |
|---|---:|---:|---:|---:|
| A1 current | 524,863 | 291,245,891,584 | 134.024655 | 2.02384 |
| B coalesced | 187,654 | 290,115,694,592 | 159.263155 | 1.69651 |
| A2 current | 524,863 | 291,245,891,584 | 149.569180 | 1.81350 |

The controls differ by 11.60%, but the conclusion is not a marginal
comparison: the coalesced arm is slower than both controls and 12.32% slower
than their mean. The larger reads reduce useful device/queue parallelism on
this workload. Request count is not the limiting resource when almost all
bytes remain.

## Decision

Stop before a model fact. Remove the temporary production/header/test slice
and retain only the reusable replay tool plus this evidence. Do not use simple
same-layer adjacency merging as the storage half of P59/P60. A future storage
redesign must either remove substantially more physical bytes or preserve deep
parallelism while coalescing submission overhead.

The machine-readable record is
`results/k3_p56r_layer_read_coalesce_replay.json`.
