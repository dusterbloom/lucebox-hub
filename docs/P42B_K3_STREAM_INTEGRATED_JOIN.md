# P42b — stream-integrated ordered device join

## Verdict

**BYTE-EXACT GO; THROUGHPUT NO-GO; ONE FINAL MEAN-RESIDENCY FACT GATE IS
JUSTIFIED.**

P42b removes P42a's 17,917 per-expert default-stream synchronizations. Expert
outputs are copied into one backend-owned arena on the same GGML stream, one
backend barrier drains the route copies immediately before the ordered join,
and the completed output is published asynchronously into the graph-owned
input. The frozen contribution order and separately rounded FP32 arithmetic
are unchanged.

The adjacent off/on/off fact is exact, but P42b remains slower. The P42b arm
reaches **1.221483 true transitions/s**. The bracketing P42-off controls reach
1.343542/s and 1.351042/s, a 1.347292/s mean. P42b is 9.34% below that mean
and below both the 1.275600 P41 reference and the slower 1.343543/s off arm.
No broad run, default enablement, deletion, or dual-owner work is earned.

## Stream and lifetime contract

The arena is one GGML backend allocation containing a contiguous F32 row
tensor, 208 persistent row views, device descriptor tensors, and one output
tensor. A P41 compact result is copied asynchronously to its row view before
that compact entry can be reused. All subsequent compact graph work uses the
same backend stream, so reuse is ordered after the copy without a host wait.

Host mean batches retain their P42a lifetime and cost: one synchronous,
contiguous tensor upload per batch directly from the existing scratch. There
is no extra host repack. Descriptor vectors remain alive through an async set
and the single backend synchronization before the ordered join. The raw join
is synchronized once before its output is enqueued on the backend stream;
the following graph compute consumes the published tensor in stream order.

Abort handling tracks queued backend work independently from a completed,
pending output. If evaluation fails after an expert copy but before the join,
discard synchronizes the backend before any arena row can be reused. Source
and destination tensors must be contiguous F32 `[3584,1,1,1]` on the exact
same backend. Invalid layouts fail before the generic async-copy API.

## Qualification

The capped HIP build used `CCACHE_DISABLE=1` and `-j4`. The ordered-join test
passes on both gfx1201 and gfx1151, the provider test passes, and the unchanged
sparse-K gate passes 68/68 cases on each GPU. Final hashes are:

| Artifact | SHA-256 |
|---|---|
| Runner | `e869d4da55d3885947604bb65ac293eff5a6b2babd428368503a4182db0bfed2` |
| Ordered join test | `d481fe86afb80109f7f701367080689ae1f0f6d1a021de3ac32c2ddd0f654d2d` |
| Provider test | `9e7ee7c1a156a66e114ed6b8fae3f6459747337b666cb0553881a8f58ce5e4dc` |
| Sparse-K test | `62e14f893d7f4b286bfb8f91da110aa99b4e890f560dc3ec790f8c75e2310278` |
| Ordered-join HIP object | `df3e248bca60a5217f5222e0ec827c2cfd0eb4e80891fab08d1381d3c59e192a` |
| GGML HIP library | `773a13a8f2274c75e8e28e3b76a6419635ebd1fdd570187569c28500a42075c4` |

The known unit gap is narrow: the private arena abort/publish lifecycle is not
directly callable from the model-free test. Static review covers the queued
work state machine, and the integrated P42b arm exercises the publish path.

## Adjacent fact gate

All three arms use the same GPU1 provider, calibrated layer table, P27 compact
delivery, 8-GiB P30 cache, P41 compact executor, authoritative sidecars, and
fact fixture. Only `DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN` changes.

| Result | A: off | B: P42b on | C: off |
|---|---:|---:|---:|
| True AR rate, 8 transitions | 1.343542/s | **1.221483/s** | 1.351042/s |
| Decode time | 5.954410 s | **6.549417 s** | 5.921353 s |
| Prefill time | 27.243261 s | 27.729889 s | 27.374391 s |
| Mean decode total | 744.059 ms | **818.418 ms** | 739.918 ms |
| Mean routed prep | 275.126 ms | 274.627 ms | 275.961 ms |
| Mean expert stage | 429.744 ms | **508.241 ms** | 424.197 ms |
| Mean join stage | 24.907 ms | **21.498 ms** | 25.347 ms |

The join stage improves by 3.629 ms against the bracketing-control mean, but
the expert stage is 81.270 ms slower and total transition time is 76.430 ms
slower. P42b still moves 733,092 mean rows / 10,509,606,912 bytes. It removes
all expert readback, completes all 17,917 P41 compact evaluations with zero
fallback/invalid events, and preserves byte-identical logits and traffic:

- logits SHA-256: `cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`
- traffic SHA-256: `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`

Artifact roots are:

- `/home/duster/kimi-k3-deploy/p42b-ab-a-off-20260819`
- `/home/duster/kimi-k3-deploy/p42b-ab-b-on-20260819`
- `/home/duster/kimi-k3-deploy/p42b-ab-c-off-20260819`

## Code and deletion boundary

Against the captured qualified P42a source, raw numstat is production
120 additions / 105 deletions and tests 25/15. Tokei's comment/blank-excluding
delta is **+9 production code lines** and +10 test code lines; comments add 5
and blanks add 1. P42b therefore meets the bounded consolidation target, but
its exactness does not earn deletion because performance still fails.

## One final P42c decision

Full device mean residency is 13.207 GiB. Added to P42b's measured 46.569-GiB
peak, the projected peak is 59.776 GiB on the 96-GiB GPU1, leaving about
36.224 GiB. Capacity is not the blocker.

Promotion requires no regression versus the slower off arm: decode at most
5.954410 seconds, or at least 1.343543/s. P42c must recover 74.376 ms per
transition by decode accounting (74.359 ms by summed stage rows), 97.29% of
P42b's 76.430-ms stage gap versus the bracketing controls and 91.50% of its
81.270-ms expert-stage gap. Eliminating 10.51 GB of short-run mean transfers
and their synchronous calls is large enough to justify exactly one terminal
P42c fact gate, but the earlier 1.30/s value is diagnostic only and does not
promote the path.

This is not a route to 10 AR/s by itself. Even deleting the entire measured
508.241-ms expert stage leaves roughly 310.177 ms/transition, an absolute
stage-accounting ceiling of 3.224/s. If full mean residency misses the P42c
threshold, remove the P42a/P42b production integration while retaining its
evidence and generic ordered-join test seam.

Machine-readable evidence is in
`results/k3_p42b_stream_integrated_join_runtime.json`.
