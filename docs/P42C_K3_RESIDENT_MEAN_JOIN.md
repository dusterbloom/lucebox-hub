# P42c — resident-mean ordered device join

## Verdict

**HIP BYTE-EXACT GO; STRICT SHORT FACT THROUGHPUT GO; BROAD, STEADY-STATE,
DEFAULT ENABLEMENT AND DUAL-OWNER REMAIN OPEN.**

P42c keeps all calibrated mean cards resident on GPU1 and removes their reads
and H2D copies from the routed hot path. The official matched fact run produces
byte-identical logits and traffic and completes eight true decode transitions
in 5.456211406 seconds, or **1.466218848 transitions/s**. That is 0.498199
seconds inside the preregistered 5.954410-second ceiling and 9.13% above its
1.343543/s floor.

This closes the terminal short-fact gate without invoking the planned P42
removal. It does not establish broad or steady-state throughput. P42 remains
off by default, and no heterogeneous ownership work begins before those gates.

## Bounded implementation

One fixed backend allocation holds 92 layers × 896 experts × 12 calibration
ranks × 3,584 FP32 values: 14,180,941,824 bytes, or 13.20703125 GiB. It is
loaded once at provider initialization. Allocation is accepted only on GPU1
with at least an 8-GiB free-memory reserve. Because GGML enables managed memory
on environment-variable presence, P42c fails closed whenever
`GGML_CUDA_ENABLE_UNIFIED_MEMORY` is present, including the string `0`.

Ordered-join descriptors are independent of transient row count:

- nonnegative values address one of 16 transient expert-output rows;
- `-1 - resident_row` addresses the fixed mean table;
- the device kernel guards both ranges and `INT32_MIN` before indexing;
- resident-only schedules are valid with zero transient rows;
- the canonical calibrated ordering, separate fallback subtotal and
  non-contracted multiply/add arithmetic are unchanged from P42a/P42b.

The asserted path is deliberately narrow: all-layers-calibrated96,
`route_prefix_depth=0`, authoritative sidecars, P41 compact execution, one
token and identical expert/destination GPU1 backends. Other configurations
fail closed or use the existing default-off host path.

## Model-free qualification

The final capped build was:

```text
CCACHE_DISABLE=1 cmake --build server/build-k3-hip-dual -j4 \
  --target run_kimi_k3_h16_suite test_kimi_k3_ordered_join \
           test_kimi_k3_progressive_provider
```

The ordered join passes on gfx1201 and gfx1151. It covers exact resident rows
at the first, middle and final boundaries; a resident-only zero-transient
schedule; invalid `-(resident_count+1)` and `INT32_MIN` sentinels; the actual
208-operation ceiling with 16 transient and 192 resident descriptors; the
fallback subtotal; duplicates; signed zero; a subnormal; NaN; an FMA-separating
triple; and output reuse. The provider test passes. The unchanged sparse-K
binary remains 68/68 exact on each GPU.

| Artifact | SHA-256 |
|---|---|
| Runner | `e09e351af4795f7a2576647a5b4e0327696799c499183c654e402ee90daf038a` |
| Ordered join test | `f00fed0d7ee962b18b78c4b296300c744eab6c32c0e43ae7c974c7288ee85a55` |
| Provider test | `80c232e62e1dffe65c7a9822b38a7aff833d5132154168b675f5a0745418bf26` |
| Sparse-K test | `62e14f893d7f4b286bfb8f91da110aa99b4e890f560dc3ec790f8c75e2310278` |
| Ordered-join HIP object | `fa910e562bf7a9a52b9d716dea7e8f933604edd7bb3887014c710d7792a894d5` |
| GGML HIP library | `773a13a8f2274c75e8e28e3b76a6419635ebd1fdd570187569c28500a42075c4` |

## Official matched fact gate

The official root is
`/home/duster/kimi-k3-deploy/p42c-resident-fact-final-20260819`. Its command is
the P42b-on command array with only artifact paths/prompt label changed, the
new binary, and unified memory explicitly unset. Stage profiling is enabled.

| Result | P42b | P42c | Change |
|---|---:|---:|---:|
| True AR rate, 8 transitions | 1.221483/s | **1.466219/s** | **+20.04%** |
| Decode time | 6.549417 s | **5.456211 s** | **−1.093206 s** |
| Prefill time | 27.729889 s | 27.528810 s | −0.201079 s |
| Mean decode total | 818.418 ms | **681.786 ms** | −136.632 ms |
| Mean routed prep | 274.627 ms | 274.137 ms | −0.490 ms |
| Mean expert stage | 508.241 ms | **372.748 ms** | −135.493 ms |
| Mean join stage | 21.498 ms | 20.952 ms | −0.546 ms |

The exactness invariants are unchanged:

- logits SHA-256:
  `cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`;
- traffic SHA-256:
  `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`;
- same nine-token Tokyo continuation and text;
- P41 completes 17,917/17,917 events with zero fallback and zero invalid;
- P42c reports zero hot mean reads and zero hot mean H2D bytes;
- 17,917 expert D2D copies, 3,864 join launches and 3,864 output copies.

Peak GPU1 memory is 61,208.992 MiB. The short run uses 2,547.105 J over
55.946 seconds. Capacity therefore passes on this fixture, but longer-run
allocator behavior and stable throughput are still unmeasured.

## Code size

The qualified P42b tree was not retained. The isolated P42c figures are
reconstruction-derived from current P42c versus the retained P42a snapshot,
minus the frozen published P42b delta. The raw net exactly cross-checks against
Tokei code plus comment plus blank deltas:

| P42c versus P42b | Added | Deleted | Net / pure code |
|---|---:|---:|---:|
| Production raw | 193 | 64 | +129 |
| Tests raw | 82 | 5 | +77 |
| Total raw | 275 | 69 | +206 |
| Production pure code | — | — | +127 |
| Test pure code | — | — | +74 |
| Comments / blanks | — | — | +2 / +3 |

The complete P42a+b+c stack is 1,596 additions / 231 deletions, net 1,365
raw lines: 1,279 pure code lines, 29 comments and 57 blanks. Pure code splits
into 877 production, 366 test and 36 CMake lines. The residency gate earns
retention for broad qualification, not immediate deletion of the host control.

## Next gate

Run broad and longer steady-state qualification against adjacent P42-off
controls. Require byte-exact logits/traffic, zero hot mean reads/H2D, stable
memory below the GPU1 reserve and a repeatable throughput win. Only then may
P42 become a default or support the heterogeneous expert-owner split. No
additional optimization or broad run is part of this result.
