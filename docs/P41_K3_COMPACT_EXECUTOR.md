# P41 — component-major compact expert executor

## Verdict

**HIP BYTE-EXACT GO; INTEGRATED THROUGHPUT NO-GO; BROAD RUN AND LEGACY
DELETION STOPPED; ORDERED PR600 DEVICE-OUTPUT JOIN PROMOTED.**

P41 proves that selected Kimi slabs can execute without reconstructing a full
expanded expert while preserving the native down-projection arithmetic. The
generic sparse-K MMVQ passes all 68 registered comparisons on both Lucebox4
architectures, and the integrated 42-position fact run produces the same
logits bytes, tokens and text as the P39/P40 same-placement controls.

The result is not fast enough to promote. Eight true decode transitions take
6.271558505 seconds, or 1.275600 transitions/s. That is 8.41% below the warm
P39 8-GiB host-cache fact control and 10.59% below the P40-on fact run. The
compact graph removes full-weight clears and scatter, but its expert graph is
only 6.74% cheaper than the P40-on fact graph. Component-major repacking,
additional selected-weight upload and colder physical reads more than consume
that saving.

The preregistered performance gate therefore stops before the 12-prompt run.
P30 and P40 remain available, P41 remains opt-in research code, and no
expanded evaluator, scatter path or cache tier is deleted from this result.
The next systems slice is the PR600-derived ordered device-output boundary:
remove the owner-result host bounce before attempting heterogeneous expert
ownership.

## Exact execution contract

P41 keeps the authoritative P27 natural-slab records but transforms each
route into one component-major image:

1. a 32-byte natural-slab ID header;
2. all selected gate blocks;
3. all selected up blocks;
4. all selected down blocks.

Gate and up use ordinary compact GGML MMVQ. The existing unfused SiTU
activation is preserved. Down uses the generic
`ggml_mul_mat_sparse_k_blocks` operation with a virtual K of 3072. Its kernel
is a specialization of the native one-column MMVQ body: natural blocks keep
the native lane ownership and reduction order, while a 12-entry map resolves
each present block to its compact resident slot. Missing blocks do not move
selected work to different lanes.

The opt-in switch is `DFLASH_KIMI_P41_COMPACT_EXECUTOR=1`. Qualification
requires the persistent P27 path. P40 and P41 are mutually exclusive for this
gate, malformed layouts fail closed, and the expanded evaluator remains the
fallback. Runtime counters distinguish attempts, layouts, uploads, gate, up,
SiTU, sparse-down, completion, fallback and invalid events.

## HIP micro-parity gate

The same dual-architecture binary was run once on the discrete R9700
`gfx1201` and once on the Strix `gfx1151`. Each device passes 68/68 exact
comparisons across IQ1_S and IQ2_XXS:

- every singleton natural block;
- three-slab, first-four, six-slab and all-twelve masks;
- two- and three-term same-lane accumulation;
- reversed compact residency and resident supersets;
- resident counts one through twelve;
- signed-zero/NaN input;
- production-width 3,584-row output.

Reference and sparse output floats are compared byte-for-byte. The test
binary SHA-256 is
`62e14f893d7f4b286bfb8f91da110aa99b4e890f560dc3ec790f8c75e2310278`.
The HIP library SHA-256 is
`773a13a8f2274c75e8e28e3b76a6419635ebd1fdd570187569c28500a42075c4`
and `mmvq.cu.o` is
`a077cd3c4c79339e2cd42b98251a95041f27a600bad8a1ac6056cfa6af925993`.
The dense MMVQ ISA sizes and VGPR counts are unchanged from the pre-P41
binary, so this qualification does not hide a dense-kernel resource change.

This is same-device HIP parity, not CUDA-to-HIP identity. The local CUDA
microtest was registered but skipped because no compatible CUDA device was
available in that environment.

## Integrated fact gate

The measured command keeps the P37 policy, GPU1 core/provider placement,
8-GiB P30 host cache, direct pinned P27 delivery and sidecar-authoritative
fallback. P40 is disabled and only P41 is enabled. The fixture contains 34
prompt positions and emits nine tokens, of which eight are true
autoregressive transitions.

| Result | P41 compact | Warm comparison |
|---|---:|---:|
| True AR rate | **1.275600/s** | P39 host cache 1.392766/s (**−8.41%**) |
| True AR rate | **1.275600/s** | P40-on fact 1.426755/s (**−10.59%**) |
| Prompt rate | 1.135587/s | P40-on fact 1.184556/s (−4.13%) |
| End-to-end wall | 58.328807 s | — |
| Sampled GPU energy | 2,482.728 J | — |
| Peak GPU memory | 47,690.301 MiB | — |

All 17,917 routed expert events complete the compact layout, upload, gate,
up, SiTU and sparse-down stages. Fallbacks and invalid layouts are both zero.
The generated continuation remains the exact nine-token Tokyo response. The
42-row logit payload is byte-identical to the P39 cached and P40 on/off fact
controls, with SHA-256
`cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`.
The provider traffic is also byte-identical, with SHA-256
`e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`.

Across the eight decode positions, mean/median stage latency is:

| Stage | Mean | Median | Change vs P40-on mean |
|---|---:|---:|---:|
| Total | 783.692 ms | 787.901 ms | +11.85% |
| Routed preparation | 275.247 ms | 274.875 ms | −0.03% |
| Experts | 469.140 ms | 472.483 ms | +21.35% |
| Join | 25.069 ms | 25.302 ms | +0.15% |

Aggregate P41 provider counters across all 42 positions are:

| Counter | Value |
|---|---:|
| Authoritative H2D | 55,976,374,272 bytes |
| Explicit provider reads | 29,496,033,280 bytes |
| Direct physical reads | 25,424,379,904 bytes |
| Direct I/O | 11.625074 s |
| Component-major pack | 1.554670 s |
| Full-weight zero/scatter | **0 bytes / 0 s** |
| Expert graph | 1.770864 s |
| Readback | 0.347291 s |

Relative to the P40-on fact, expert-graph time falls 6.74% and scatter is
eliminated, but component-major packing rises from 0.106206 to 1.554670
seconds, H2D rises 62.90%, physical reads rise 11.69% and direct-I/O time rises
41.94%. This is one short ordered comparison, so individual storage deltas are
order/cache sensitive. The end-to-end regression against two warm controls
is nevertheless sufficient to reject the compact executor as the next
throughput route. A broad run cannot rescue a failed fact gate.

## Next boundary

P41 establishes two reusable facts even though the implementation is a speed
NO-GO: native-schedule sparse-K MMVQ can be bit exact on both Lucebox4 GPUs,
and the evaluator can terminate at a device output. The next slice should use
that second boundary without changing accumulation order:

1. expose caller-owned per-route device outputs from the existing evaluator;
2. keep one globally ordered route/expert destination list;
3. join on the destination device with peer copies and events;
4. compare the single-owner device-output path with the current CPU-result
   bounce before assigning any routes to a second owner;
5. only then test stable R9700/Strix expert ownership and overlap.

PR600's stable owner planning, concurrent owner streams and device join are
the relevant reuse. Its DS4 qtype policy and per-batch dynamic route balancing
remain out of scope for Kimi. Owner-local partial sums remain forbidden for
the exact path because they change FP32 association.

Machine-readable evidence is in
`results/k3_p41_compact_executor_runtime.json`.
