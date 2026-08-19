# P42d — ordered-join consolidation

## Verdict

**BYTE-EXACT CONSOLIDATION GO; 95 PURE LINES REMOVED; NO INTRINSIC SPEEDUP
CLAIM.**

P42d reduces the qualified P42c implementation without changing its ordered
arithmetic or sparse-K kernel. C1 removes the production-only ordered-join
reference implementation, folds resident means, 16 transient rows,
descriptors, weights and output into one destination-backend allocation, and
adds the one-token device-output capability guard. C2 gives the compact host
and device paths one typed attempt outcome and counter-update path, and parses
the P42 flag once at the provider factory boundary.

Against the retained pre-consolidation snapshot
`/tmp/p42_consolidation_before`, production loses 77 pure lines and tests lose
18. The complete P42 stack falls from 1,279 to 1,184 pure lines. Exact model
outputs, traffic and execution counters remain unchanged.

## Allocation and lifecycle boundary

The one destination GGML context requires exactly **14,181,187,328 bytes**:

| Tensor group | Bytes |
|---|---:|
| Resident calibrated means | 14,180,941,824 |
| 16 transient 3,584-F32 rows | 229,376 |
| Padded I32 row descriptors | 896 |
| Padded F32 weights | 896 |
| 3,584-F32 output | 14,336 |

The provider asks the backend allocator for the exact context size, checked-adds
the 8-GiB reserve, verifies the allocated buffer is at least that large and
checks the mean extent before preload. Both consolidated model runs initialize
the provider and preload all 92 mean cards successfully. The hot path still
records zero calibrated-mean reads and zero calibrated-mean H2D.

No move-only routed-result or broader RAII lifetime refactor is included. That
possible C3 cleanup is deferred: it is estimated to remove only 15–30 more
production lines and needs separate failure, pending-output, graph-publication
and provider-destruction lifecycle qualification.

## Model-free qualification

All builds were capped at four jobs:

```text
CCACHE_DISABLE=1 cmake --build server/build-k3-p20-cuda126b -j4 \
  --target test_kimi_k3_progressive_provider test_kimi_k3_ordered_join
CCACHE_DISABLE=1 cmake --build server/build-k3-hip-dual -j4 \
  --target test_kimi_k3_progressive_provider test_kimi_k3_ordered_join
```

- Local provider test passes; ordered join correctly skips without a visible
  GPU.
- HIP provider passes.
- HIP ordered join passes on gfx1201 and gfx1151.
- Sparse-K was untouched and retains its qualified 68/68 exact result on both
  GPUs.
- `git diff --check` passes after each slice.

| Consolidated artifact | SHA-256 |
|---|---|
| Runner | `1c9a203030ce5650034196cadb9b4c4d83413e20ab4048093a077600e5c5a2a2` |
| Provider test | `c8e0858c840bdcef8eb6f5d08aab64bf019ab61142b72a894f06e4f1305c9782` |
| Ordered-join test | `c8cc17cbd3cd2c41f667234a57f9c289c96b4cd834a92dc8183354dbd69de6b3` |
| Sparse-K test | `62e14f893d7f4b286bfb8f91da110aa99b4e890f560dc3ec790f8c75e2310278` |
| Provider source | `fbc8f1fc7a149c2eefdeaae09157dfbb9b31b3f450c3de1644fb975b5140c3e1` |

## Model evidence and contaminated first sample

The first consolidated fact at
`/home/duster/kimi-k3-deploy/p42c-c1c2-terminal-fact-20260819` is exact but
reaches only 8/6.329393191 = **1.263944229/s**. It is not accepted as a code
regression measurement. Compared with the official P42c run, aggregate direct
I/O rises 44.17% from 9.086873 to 13.100132 seconds, major faults rise from 216
to 3,554, and sampled process swap rises from zero to 181,624 KiB. Its expert
stage rises from 372.748 to 484.106 ms while routed preparation and join do not
regress materially.

The sample still proves the consolidated allocation/preload and exactness:

- logits: `cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`;
- traffic: `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`;
- the same nine-token Tokyo continuation and text;
- P41 completes 17,917/17,917 attempts with zero fallback or invalid;
- expert readback and hot calibrated-mean I/O/H2D remain zero;
- 17,917 expert copies and 3,864 join/output publications complete.

## Exact adjacent binary control

The retained source snapshot was rebuilt in the same source/build path. A
forced rebuild reproduced the qualified revision-58 runner exactly, after
which the consolidated sources and runner were restored and also reproduced
exactly. The canonical remote tree was left at C1+C2.

| Arm | Runner SHA-256 | Decode | True AR | Mean expert | Direct I/O |
|---|---|---:|---:|---:|---:|
| rev58 old | `e09e351af4795f7a2576647a5b4e0327696799c499183c654e402ee90daf038a` | 6.257874753 s | 1.278389280/s | 474.416 ms | 11.831 s |
| C1+C2 new | `1c9a203030ce5650034196cadb9b4c4d83413e20ab4048093a077600e5c5a2a2` | 5.503774952 s | **1.453547805/s** | 379.246 ms | 9.932 s |

The roots are respectively
`/home/duster/kimi-k3-deploy/p42c-c1c2-adjacent-control-20260819/old-rev58`
and
`/home/duster/kimi-k3-deploy/p42c-c1c2-adjacent-control-20260819/new-c1c2`.
Both produce the exact logits and traffic hashes above, identical tokens/text,
17,917 successful compact attempts and zero fallback, invalid, readback or hot
mean I/O/H2D.

The new arm is 13.7015% faster than the adjacent old arm and only 0.8642%
below the official 1.466218848/s P42c fact. This rules out the first sample's
apparent 13.8% consolidation regression. It does **not** establish an intrinsic
C1+C2 speedup: direct-I/O time still differs 16.05% between adjacent arms,
process read bytes differ 13.33%, and both arms observe some process swap.

## Code size

| P42d versus pre-consolidation | Added | Deleted | Net |
|---|---:|---:|---:|
| Production raw | 103 | 188 | −85 |
| Tests raw | 5 | 24 | −19 |
| Total raw | 108 | 212 | **−104** |

Tokei whole-file totals change as follows:

- production pure code: 9,295 → 9,218, or −77;
- test pure code: 526 → 508, or −18;
- total pure code: 9,821 → 9,726, or **−95**;
- comments: −6; blanks: −3.

The complete P42 stack is now 1,184 pure lines: 800 production, 348 tests and
36 CMake. This is a real deletion result, not a namespace move.

## Decision

Retain P42d as the exact, smaller P42c implementation. Broad and steady-state
qualification remain open, and the path stays opt-in. The next performance
priority is single-owner GPU1 compact slab residency, followed by routed-
preparation reduction. The measured K3 path remains far below 10 true AR/s.
