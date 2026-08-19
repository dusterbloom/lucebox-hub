# P43a — single-secondary-owner peer publication

## Verdict

**BYTE-EXACT; THROUGHPUT NO-GO; IMPLEMENTATION REMOVED.**

P43a kept P42c's 13.207-GiB calibrated-mean table, canonical descriptors,
ordered join and core graph on GPU1, while moving every P41 compact expert to
GPU0. Each 3,584-F32 expert result was preserved on GPU0, published once to its
canonical GPU1 arena row, and consumed by the unchanged ordered join. The one
and only matched fact retained exact tokens, text, logits and traffic, but took
12.485984906 seconds for eight true decode transitions: **0.640718378/s**.

That misses every preregistered speed boundary: at most 600 ms/transition
(at least 1.667/s), at most 291 ms in experts, and at most 5 ms of peer wait per
model position. There was no repeat, broad run or concurrency experiment. All
P43-specific production, backend and test wiring was removed after evidence
was frozen; the qualified P41/P42c source boundary and generic GGML peer-copy
support remain.

## Bounded implementation measured

- GPU0 owned the P41 compact evaluator; GPU1 owned resident means, arena,
  descriptors, ordered join and destination graph.
- A 16-row GPU0 preservation arena protected graph-owned output from immediate
  compact-entry reuse. The layer's publications were then issued consecutively
  through `ggml_backend_tensor_copy_async`.
- `GGML_BATCH_PEER_COPIES=1` produced one source event/destination wait at the
  GPU1 layer barrier. The legacy knob and unified memory were rejected.
- Startup was default-off and fail-closed outside HIP, unless source/destination
  were distinct same-runtime GPU0/GPU1 backends with bidirectional peer access,
  P41+P42c, authoritative calibrated96, route-prefix depth zero and one token.
- The global descriptor sequence and separate fallback subtotal were unchanged;
  no owner-local partial sum, worker concurrency or second mean table existed.

The measured build still initialized the prior dual-engine scaffolding, leaving
an unused GPU1 expert engine/cache. No hotness profile was active, so that does
not invalidate the bounded result, but it makes the run explicitly
**pre-consolidation** and is another reason not to promote the discarded code.

## Model-free qualification

The capped HIP build and gates passed after the ambient-device fix:

```text
CCACHE_DISABLE=1 cmake --build server/build-k3-hip-dual -j4 \
  --target run_kimi_k3_h16_suite test_kimi_k3_ordered_join \
           test_kimi_k3_progressive_provider test_sparse_k_mmvq_cuda
DFLASH_TEST_GPU=0 ./server/build-k3-hip-dual/test_kimi_k3_ordered_join
DFLASH_TEST_GPU=1 ./server/build-k3-hip-dual/test_kimi_k3_ordered_join
DFLASH_TEST_GPU=0 ./server/build-k3-hip-dual/test_sparse_k_mmvq_cuda
DFLASH_TEST_GPU=1 ./server/build-k3-hip-dual/test_sparse_k_mmvq_cuda
./server/build-k3-hip-dual/test_kimi_k3_progressive_provider
```

The two-device ordered test preserved 16 immediately reused GPU0 outputs,
published all 16 to GPU1, fed those rows into the real ordered-join kernel and
matched the separate-multiply/add teacher exactly on repeated reuse. The HIP
trace showed one `copies=16 bytes=229376 reason=backend-synchronize` batch per
iteration. Sparse-K remained 68/68 exact on gfx1201 and gfx1151.

| Artifact | SHA-256 |
|---|---|
| Runner | `0a08ae3e17a8b3bbeec243cc95c27815d7db9fa5d52245969c17a07f069d55d5` |
| Ordered join test | `e0c4592272f137914087ca5fcaace051cb35f24ce5bc131c17c32c7c1c23c349` |
| Provider test | `4077c6cc0deefcd5b70ffde948d9d950ce7b73ccc2a2b843b5c179fa06c0c7ee` |
| Sparse-K test | `62e14f893d7f4b286bfb8f91da110aa99b4e890f560dc3ec790f8c75e2310278` |

## Official fact

The evidence root is
`/home/duster/kimi-k3-deploy/p43a-peer-fact-20260819`. The run completed once
after model-free gates; no result from the pre-fix interrupted attempt is used.

| Result | P42c | P43a | Change / gate |
|---|---:|---:|---:|
| True AR rate, 8 transitions | 1.466219/s | **0.640718/s** | −56.30%; requires ≥1.667/s |
| Decode time | 5.456211 s | **12.485985 s** | +128.84% |
| Mean decode total | 681.786 ms | **1,560.507 ms** | requires ≤600 ms |
| Mean routed prep | 274.137 ms | 305.189 ms | +11.33% |
| Mean expert stage | 372.748 ms | **1,204.725 ms** | requires ≤291 ms |
| Mean join stage | 20.952 ms | 36.514 ms | +74.28% |
| Peer barrier wait | — | **9.682 ms/position** | requires ≤5 ms |

Exactness and traffic invariants all pass:

- the nine-token Tokyo continuation and text match P42c;
- logits SHA-256 is
  `cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`;
- traffic SHA-256 is
  `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`;
- P41 completed 17,917/17,917 events with zero fallback or invalid event;
- 17,917 results produced 17,917 peer copies and 256,858,112 peer bytes;
- 3,864 layer barriers recorded 406,645,165 ns total peer wait;
- expert readback, hot mean reads and hot mean H2D are all zero.

The source side still performed 24,106,500,096 physical read bytes,
55,976,374,272 payload-H2D bytes and 257,718,128 metadata-H2D bytes. Its
aggregate direct I/O, compact packing and graph-plus-save-drain counters were
13.791950, 1.569578 and 1.676195 seconds. The GPU0 expert path—not the 256.9-MB
peer payload—dominates the regression. Peak GPU1 memory was 61,028.508 MiB,
peak sampled utilization 23%, peak power 39.016 W and sampled energy
3,001.578 J over 104.524 seconds.

## Code and deletion boundary

The retained measured P43 snapshot is `/tmp/p43a_pre_removal_snapshot`; its
five source hashes are recorded in the machine-readable result. Direct
P42c-to-P43 diffing gives:

| Isolated P43a | Added | Deleted | Net / pure code |
|---|---:|---:|---:|
| Production raw | 331 | 48 | +283 |
| Tests raw | 167 | 0 | +167 |
| Total raw | 498 | 48 | +450 |
| Production pure code | 324 | 48 | +276 |
| Test pure code | 159 | 0 | +159 |
| Net comments / blanks | — | — | +9 / +6 |

Whole-file Tokei totals independently cross-check the net as +435 code,
+9 comments and +6 blanks. After the NO-GO, all 450 net P43 raw lines were
removed and the five touched files were restored to their exact qualified
P42c hashes. A device-output capability guard for speculative decoding that
review exposed is intentionally deferred to a separate P42 hardening slice;
it is not bundled into this exact restoration.

The post-removal `CCACHE_DISABLE=1`, `-j4` HIP rebuild passes. Provider and
ordered-join tests pass, ordered join on both GPU0 and GPU1, and sparse-K again
passes 68/68 on each device. The rebuilt runner, ordered, provider and sparse-K
test hashes are respectively `e09e351a…38a`, `f00fed0d…a55`,
`80c232e6…f26` and `62e14f89…278`: the exact qualified P42c binaries.

## Decision

P43a rejects only this unbalanced ownership experiment: it assigns 100% of
compact expert execution to GPU0 while GPU1 retains the core, resident means
and ordered join. The peer transport is cheap (256,858,112 B total and
9.682 ms/position); the 1,204.725-ms GPU0 expert stage is the failure.

A balanced, cost-aware, concurrently scheduled dual-owner P43b remains
**UNTESTED/OPEN**. It must preserve the canonical GPU1 ordered join and assign
work by measured cost, not a 50/50 route split. It is not earned as the first
step: single-GPU compact submission and persistent routed-preparation work are
the preceding exact gates. The retained source snapshot is only under
`/tmp/p43a_pre_removal_snapshot`; durable archival is a process action before
that evidence can be relied on as the sole source copy.
