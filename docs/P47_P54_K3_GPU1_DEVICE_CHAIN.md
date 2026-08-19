# P47–P54 — GPU1 device chain and host-cache boundary

## Verdict

**THE COMPLETE K3 CORE ON GPU1 IS THE QUALIFIED TOPOLOGY. P47–P53 ARE
BYTE-EXACT; THE BEST MEASURED NO-SPECULATION SHORT FACT IS 1.991574475 TRUE
AR/S. P54'S 24-GIB HOST CACHE IS AN EXACT CONFIGURATION NO-GO AND WAS
REVERTED.**

The 37.56-GiB core, recurrent KDA/MLA state, 13.207-GiB resident calibrated
means and canonical ordered join remain on Lucebox4 GPU1 (`gfx1151`). P47–P53
remove redundant transfers and transient graph boundaries without moving that
state. GPU0 (`gfx1201`) remains available for a later bounded, concurrently
overlapped expert share; it is not part of the qualified fast path.

This distinction matters. P43a measured an all-GPU0 expert implementation, not
a balanced PR600 split. Its 0.640718378/s result rejected that ownership policy;
it did not reject the GPU1 core placement or a future cost-aware concurrent
expert helper.

## Qualified device chain

P47–P51 progressively keep the recurrent preparation inputs and outputs on
GPU1. A shared compact input tensor replaces route-local input uploads; prefix,
routed and shared tensors feed the expert/join boundary directly; checkpoint
storage remains device-resident. The P51 repeat records 2,856 routed-input D2D
copies and 72,676 H2D calls, down from 89,585 at the original P45 boundary.

P52 gives all 92 routed layers immutable join graphs sharing one 100,352-byte
workspace. P53 keeps the joined hidden state on GPU1 between layers and
materializes it only at AttnRes checkpoint/capture boundaries. These changes
preserve the established expert order, omitted-mean rank order and separate
authoritative fallback subtotal.

| Fact | Decode | True AR | Routed prep | Experts | Join | Direct I/O |
|---|---:|---:|---:|---:|---:|---:|
| P51 first | 5.116544910 s | 1.563555122/s | — | — | — | 11.269153311 s |
| P51 adjacent repeat | **4.016922339 s** | **1.991574475/s** | 223.022375 ms | 246.644500 ms | 20.678250 ms | 9.123191054 s |
| P52 repeat | 4.144776488 s | 1.930140268/s | 224.464000 ms | 265.534250 ms | 15.602125 ms | 9.629947740 s |
| P53 hidden chain | 4.879480802 s | 1.639518696/s | 224.382375 ms | 359.258250 ms | **13.895875 ms** | 11.191448728 s |

P52 and P53 reduce the deterministic join stage by 5.076 and 6.782 ms per
transition relative to P51. Their headline samples are slower because expert
and direct-I/O windows are slower, not because the persistent join or hidden
chain regresses. Accordingly, 1.991574475/s remains the measured ceiling; a
derived normalized value above 2/s is not reported as measured throughput.

## Exactness and hardware gates

- Every fact produces the same Tokyo continuation and output token IDs.
- Full-logit SHA-256 is
  `cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`.
- Logical-traffic SHA-256 is
  `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`.
- P41 completes 17,917/17,917 compact evaluations with zero fallback, invalid
  or expert readback; P42 performs 3,864 canonical joins.
- The capped HIP build and focused provider and ordered-join gates pass on both
  `gfx1201` and `gfx1151`; unchanged sparse-K remains 68/68 exact on both.
- After restoring the qualified 16-GiB policy, the remote runner SHA-256 is
  `eb7a819777321d3e49851384898020e91f9dfe90def1820c5a492dca91ac1155`.

These gates qualify the arithmetic and lifecycle on both GPUs. The measured
fast topology nevertheless keeps the stateful core on GPU1 because that avoids
per-layer peer/state movement and leaves GPU1 as the single canonical ordering
domain.

## P54 24-GiB host-cache result

The temporary 24-GiB P30 configuration is exact and memory-safe, but it is not
useful. It retains 20,857,053,184 bytes and avoids only 100,696,064 additional
physical bytes versus the qualified 16-GiB fact because almost all extra
records are not reused. It reaches 1.849742412/s with 12.617750256 seconds of
direct-I/O time. The temporary parser ceiling was reverted and the 16-GiB
configuration restored.

## Next boundary

The immediate goal is a robust measured 2.0/s, followed by 4–5/s before
speculative decode. The next work should reduce compact expert work/variance
or qualify a genuinely concurrent, cost-aware GPU0 expert share. It should not
move KDA/MLA state, the resident means or canonical join away from GPU1.
