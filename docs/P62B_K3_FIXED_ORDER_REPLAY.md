# P62B-G0 K3 fixed-order mixed-cell replay

**Date:** 2026-08-21

**Status:** `VALID EXACT NO_GO`

**Decision:** close same-layer storage/device overlap; P62B-G1 is not earned. Return to the P58 exact multi-token verification and roofline comparison.

## Question and scope

P62B-G0 asked one bounded question left open by P62A: can the current 16-worker storage pool deliver a real mixed calibrated-plus-late-exact layer cell in fixed logical order while compact expert work executes, with enough model-free benefit to justify a default-off single-layer integration?

The frozen cell is `fact-capital`, base position 0, routed model layer 2. It contains six exact 3,584-wide compact jobs. Its calibrated submit order is experts 22, 388, 441, 491 and 496; expert 71 is the late exact depth-12 job. The production read plan remains distinct from the fixed compute/output plan. The replay uses the real hash-pinned layer-2 natural sidecar and deterministic input, but stops at six staged expert-output rows: it does not execute the P42 mean table, ordered join, model, prefill or decode.

Every timed arm preserves the registered traffic contract:

| Counter | Exact value |
|---|---:|
| Pre-submit reads / bytes | 24 / 12,976,128 |
| Late reads / bytes | 12 / 6,488,064 |
| Total physical requests / bytes | 36 / 19,464,192 |
| Payload H2D calls / bytes | 24 / 19,353,888 |
| Input D2D copies / bytes | 1 / 14,336 |
| Jobs / graph enqueues | 6 / 6 |
| Output D2Ds / bytes | 6 / 86,016 |
| Begins / flushes / synchronizations | 1 / 1 / 1 |
| Aborts / fallbacks / invalid jobs | 0 / 0 / 0 |

## Qualification

The reviewed 11-file snapshot was built fresh as HIP Release for `gfx1201;gfx1151` in a detached Lucebox4 worktree. The P62B contract self-test and seven analyzer tests passed. Progressive-provider, MoE-stream, sparse-K and ordered-join tests then passed serially on both physical GPUs: eight of eight GPU gates.

Exactly one measured wrapper run executed on physical device 1, HIP `gfx1151`. It completed normally with the registered performance `exit 2`; the analyzer was invoked exactly once and returned a schema-valid `NO_GO`. There was no model run and no repeat.

Correctness and environment gates passed:

- all synchronous P41 teacher and replay outputs share SHA-256 `0706855c262a9353bc51aa9e3dba8169479bffec884f7119a7877d6c2f519509`;
- payload, natural-map, input and logical read/compute/output plan fingerprints match the frozen fixture;
- the persistent worker pool has 16 workers and observed maximum activity 16;
- all per-arm and storage/device/empty control counters are exact;
- backend/device/architecture is exactly HIP / 1 / `gfx1151`;
- 70 target samples report zero `VmSwap`;
- system `pswpin` and `pswpout` have zero same-run delta;
- cgroup `oom` and `oom_kill` have zero delta;
- the maximum sample gap is 250,465,698 ns, below 1 second.

## Measured result

Central whole-cell medians were:

| Arm or control | Median |
|---|---:|
| Production-order A | 4.433866 ms |
| Fixed-order A | 4.410983 ms |
| Overlapped B | 3.777766 ms |
| Fixed A device window | 1.673769 ms |
| B device window | 2.446248 ms |
| Resident device-only control | 1.556910 ms |
| Storage-only control | 3.910685 ms |
| Empty synchronization | 0.000140 ms |
| Hot scheduler A / B | 0.788810 / 0.784031 ms |

The preregistration deliberately used six conservative A-B-B-A blocks. For each block, `A=min(A1,A2)` and `B=max(B1,B2)`; B had to win, save at least 2 ms and reach at least 1.20x. All six blocks failed:

| Block | A kind | Conservative saving | Conservative speedup | Pass |
|---:|---|---:|---:|---|
| 0 | production | 0.539292 ms | 1.141768x | no |
| 1 | fixed | 0.682090 ms | 1.184459x | no |
| 2 | production | 0.585217 ms | 1.152058x | no |
| 3 | fixed | 0.406964 ms | 1.103698x | no |
| 4 | production | 0.681037 ms | 1.182228x | no |
| 5 | fixed | 0.502083 ms | 1.129670x | no |

The minimum saving is **0.406964 ms**, versus the 2-ms gate. The minimum conservative speedup is **1.103698x**, versus 1.20x. The overlapped B device window is **57.121992%** slower than the resident device-only control, versus the maximum 5% inflation gate.

The following secondary gates pass: whole-pipeline scheduling efficiency 1.085374 against 0.35, production-A spread 3.4685%, fixed-A spread 5.9670%, B spread 6.1938%, empty fraction 0.0090% and hot-path regression -0.6058%. The efficiency metric is the preregistered whole-pipeline scheduling formula; it is not a literal device-availability fraction or a production speedup.

The three binding failures are therefore:

1. all six conservative blocks miss the 2-ms saving floor;
2. all six miss the 1.20x speedup floor;
3. the B device window inflates 57.12%, showing that readiness coordination moves substantial work onto the device critical path.

## Provenance

Evidence is retained on Lucebox4 at:

```text
/home/duster/kimi-k3-deploy/p62b-g0-qualification-20260821
```

| Artifact | SHA-256 |
|---|---|
| Local/remote HEAD | `df63d18fa3a8d5e7a384504dda15ccc266a3cc4d` |
| Reviewed 11-file combined source | `b1abf3b718f37f6d8a940b3b3974ffd680a983d21868d2784b10889a7e6d5aae` |
| Canonical tracked diff (`--abbrev=7`) | `5612ab4ad6a797d7657a4aab6dca081961b6370ad389d1c91471bdf71b089303` |
| Fresh HIP binary | `ecea248fc9c21552a06e8113f1c11cb0caf2087fc27fec824c035de7de3bd213` |
| Fixture | `39457414b6f7c351a8bfcd9a54327a57c329aec05ca115475644bb270618da3a` |
| Natural sidecar | `270efec50b41b8fcc49faeb0260b74d7819792deca73e9f65871bea06b08e6b9` |
| Result | `b60e626235950525feff65263543131f05daaba5a66e0b84c7c9f322157e942d` |
| Telemetry | `c6749286b27f26bb7c5dbf01149ee076ad756e28bab9b325604f58e5f2b994a9` |
| Analysis | `08f34e37638b19667176f7d1b1ee22cb7994f1273e641eb2068be061672be8ca` |
| Child stderr | `5c31acb273c6d0f86e42e2099a058f7c391aa0c5a8c6814e154a88970f016619` |
| Empty child stdout | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Individual reviewed source hashes are recorded in the machine-readable closure result.

## Closure and next decision

The complete temporary G0 scope is removed: four tracked files are restored exactly to HEAD and seven untracked experiment files are deleted. That removes the 1,488-line reviewed raw/net scope, including the default-off CMake target, provider instrumentation, runner, analyzer, fixture and tests. No P62B implementation or test symbol remains.

P62B-G0 is a **valid exact NO_GO**. It does not earn P62B-G1, a scheduler change, a model run, or any decode/prefill speedup claim. Same-layer delivery/device overlap is closed at this fixed-order boundary. The roadmap returns to the retained P58 exact recurrent-microtile/MoE-macro verification seam and its matched compute/storage roofline comparison. The broad exact decode target of at least 2.0 tok/s, 10x cold-prefill target, 4–5 tok/s pre-speculation base and later multi-token prediction remain open.
