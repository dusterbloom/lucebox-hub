# P62A K3 layer I/O/device-overlap census

**Date:** 2026-08-21

**Status:** `INVALID_TIMING / DISQUALIFIED_STRONG_SIGNAL`

**Decision:** no scheduling or speedup claim; proceed only to a bounded P62B-G0 mixed-cell replay.

## Question and boundary

P62A was a default-off, measurement-only census of the existing calibrated P41/P42/P45 path. It paired, for each routed layer occurrence, blocking direct-storage/cache delivery with the subsequent mixed post-submit device window. It did not overlap work, change request/cache/job/join order, or add a production scheduling path.

The single full broad-12 process completed all 12 requests, 552 prefill positions, 129 true decode transitions and 62,652 routed-layer records. Runtime identities were complete: 50,784 prefill records, 11,868 decode records, 275,008 jobs, 1,116,376 H2D calls, 46,308 input D2Ds and matching job/enqueue/expert-D2D counts, with zero P41 fallback/invalid and zero P45 abort. The exact logical traffic totals remained 724,784,406,528 prefill bytes and 167,799,668,736 decode bytes; the complete traffic TSV retained the frozen P55 SHA-256 `54b76f99439d90ecb35d033c12dd1d1f219e30007248aaf77f913fc69d76f383`.

This proves the diagnostic process completed the frozen workload and preserved traffic/counter identities. Logits were intentionally disabled, so it is not a fresh logits or recurrent-state qualification.

## Why the official timing result is invalid

The dedicated evidence wrapper failed after the child exited successfully with:

```text
RuntimeError: live child has no VmSwap field
```

The sampler used `poll -> /proc/PID/status -> poll`. Linux can release a task's memory descriptor before `waitpid(WNOHANG)` observes exit, so a naturally exiting child can temporarily lack `VmSwap` while `Popen.poll()` still reports it live. The evidence manifest was atomically finalized and the child exit was zero, but the wrapper correctly returned failure. The analyzer was invoked exactly once and rejected the evidence schema; no official `analysis.json`, timing ceiling or verdict exists.

The same wrapper bracket also failed the registered system-swap gate:

| Evidence | Observed | Gate |
|---|---:|---:|
| Target sampled `VmSwap`, initial / maximum / final | 0 / 49,944 / 9,840 KiB | all zero |
| Host `pswpin` delta | 11,279 pages | descriptive component |
| Host `pswpout` delta | 13,716 pages | descriptive component |
| Combined host swap delta | 24,995 pages | at most 1,024 pages |
| Cgroup `oom` / `oom_kill` delta | 0 / 0 | 0 / 0 |
| Maximum sample gap | 285,579,938 ns | at most 1,000,000,000 ns |

Therefore the qualification is **INVALID_TIMING**, irrespective of the raw timing signal. The failed wrapper must not be selectively widened or rerun to rescue a favorable result.

## Exact attribution mismatch

Total physical traffic remained exactly 335,504,441,344 bytes, but the phase boundary moved one 540,672-byte aligned request:

| Phase | Frozen P55/P56 | P62A observed | Delta |
|---|---:|---:|---:|
| Prefill physical bytes | 291,245,891,584 | 291,245,350,912 | -540,672 |
| Decode physical bytes | 44,258,549,760 | 44,259,090,432 | +540,672 |
| Total | 335,504,441,344 | 335,504,441,344 | 0 |

Prefill physical requests were 524,862 rather than the frozen 524,863. Because the preregistration required exact per-phase attribution, this independently disqualifies the official analyzer result even though complete traffic is unchanged.

## Descriptive diagnostic only

After the official analyzer failed, a separate read-only diagnostic explicitly bypassed only the failed wrapper/process gate, sampled-swap gate, 1,024-page limit, per-phase physical-byte split and prefill-request count. Its JSON SHA-256 is `a180c6c050aa2a8628c8dbe94e4d291b5e347ae32810a5a36e9f4584745360c4`.

These numbers are unqualified, zero-replacement-cost service-time upper bounds. They are **not observed overlap, device availability, realizable savings or a speedup projection**.

| Phase | Narrow pre-submit bound | Full conservative bound | Swap-equivalent debit | Descriptive adjusted full bound |
|---|---:|---:|---:|---:|
| Prefill | 112.115141 ms/position | 142.404727 ms/position | 0.101064 ms/position | 142.303663 ms/position |
| Decode | 96.042250 ms/transition | 116.457305 ms/transition | 0.432461 ms/transition | 116.024844 ms/transition |

The same-run physical/direct-I/O service rate was 1,835,171,837.525245 B/s, making 24,995 swapped 4-KiB pages equivalent to 55.787430 ms of service. The raw full bounds clear the 84.722396-ms prefill materiality threshold and 50-ms decode continuation threshold only as a **disqualified strong signal**.

Raw component sums:

| Phase | Provider wall | Pre-submit reads | Late exact reads inside device window | Mixed post-submit device window | Outside residual | Narrow bound | Full bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 220.716880 s | 143.542608 s | 11.440267 s | 67.657580 s | 9.516692 s | 61.887558 s | 78.607409 s |
| Decode | 41.462826 s | 25.804554 s | 2.031659 s | 13.582128 s | 2.076143 s | 12.389450 s | 15.022992 s |

Exact-route diagnostics remained internally coherent: prefill/decode exact-policy routes were 7,954/1,620, affected records 6,452/1,364, late read calls 7,954/1,620 and late bytes 23,255,416,832/3,728,277,504. Zero-physical-read records were 9,326/3,234. These facts localize a promising overlap question; they do not qualify its magnitude.

## Provenance

The successful-runtime artifact root was:

```text
/home/duster/kimi-k3-deploy/p62a-v2-broad12-final-r2-20260821
```

Key evidence SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Reviewed tree HEAD | `20b2979e4fd5cceed8df88aa6d6060729278c59c` |
| Suite binary | `6e2a5fe00ca005400d1038b64d9a9b7b9c06b078f4a679f6270c9807cbce34ca` |
| Fixture | `6d1c0583df52738820559bef66f6a96839bcde44c0bae7bdc4bb7bbe7332d4cc` |
| Frozen reference manifest | `124b6b119ce28418967d0f541b485f88f63077b01ae78dccbcf88a922ed211bb` |
| P62 records | `18e95291d5483d27f0a4398f704dfb743c89ded0153c37ddde176cac5ecdb4cb` |
| Traffic TSV | `54b76f99439d90ecb35d033c12dd1d1f219e30007248aaf77f913fc69d76f383` |
| Suite manifest | `427fbbd8c4e404037ff3311893d988f9525a07c49e610e0fecfc3c5b0b3ca29d` |
| Wrapper evidence manifest | `c914ba40a0ab36ee1975d75d7c14c7bbc85a0146a0da29d39104d50ad19f6838` |
| Wrapper stderr | `2162027c3b47b280dc39f6e58b31443c551d8b19a0f509e6e034a253b75bb791` |
| Analyzer stderr | `4eb4eca0ecab01e27f67f09bafa87bf5dd06d0235ef3ee5fe7a9247573062374` |

Reviewed temporary source hashes are recorded in the machine-readable result. The temporary production delta was 416 additions and 13 replaced HEAD lines across five tracked runtime/runner files. Closure removes those 416 additions, restores all 13 HEAD lines, and deletes the four untracked preregistration/analyzer/wrapper/test files. No P62 runtime symbol remains.

Two earlier stopped launches are retained as failure provenance only:

1. `p62a-overlap-qualified-20260821/broad12` stopped at the first prompt/layer 2 with a layer-accounting mismatch after 15 jobs; it is not measurement evidence.
2. `p62a-v2-broad12-final-20260821` completed only `fact-capital`, then stopped on a reconstruction error that expected 12 rather than the frozen reference's eight `fact-science` transitions; it is not broad evidence.

## Decision

P62A is closed as **INVALID_TIMING / DISQUALIFIED_STRONG_SIGNAL**. There is no P62A speedup claim and no authorization to change the live scheduler.

The next bounded step is **P62B-G0**, a model-free mixed-cell replay that reproduces the calibrated-plus-late-exact ordering and tests whether overlap can exist with bounded buffers and byte-exact outputs. Only if G0 passes its exactness, memory and timing gates may **P62B-G1** integrate a default-off single-layer experiment. P58's exact multi-token verification comparison remains open in parallel.
