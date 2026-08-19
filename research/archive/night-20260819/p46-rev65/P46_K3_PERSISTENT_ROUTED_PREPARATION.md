# P46 — persistent routed preparation on GPU1

## Verdict

**BYTE-EXACT SHORT-FACT GO; NEW MEASURED SHORT CEILING 1.678584379 TRUE
AR/S; DEFAULT-OFF PENDING CONSOLIDATION AND BROAD QUALIFICATION.**

P46 keeps the entire K3 core on Lucebox4 GPU1 and changes only the lifecycle of
the 68 recurrent routed-layer preparation graphs.  Their tensor topology,
weights, KDA state updates, router, routed-down projection and shared expert are
unchanged.  Immutable per-layer graph metadata shares one pre-reserved compute
workspace; MLA, dense, replay, tracing, panel capture and offloaded-core paths
remain on their established implementations.  Explicit P46 use outside the
qualified one-token P45 path fails closed.

This is a lifecycle result, not a KDA arithmetic rewrite.  It confirms that
placing the full core on GPU1 was already correct: the measured saving comes
from stable graph identity and metadata/workspace reuse, not from moving
weights between CPU and GPU.

## Qualification ladder

The production integration followed a production-shaped HIP replay gate on
recurrent layers 1, 40 and 88.  Persistent and transient GPU1 outputs were
byte-identical for prefix, routed latent, shared output, route weights and
selected IDs.  Persistent medians were 1.59–1.66x faster than transient graph
lifecycle on those layers.

After integration, the capped local CUDA build linked the runner and benchmark.
On Lucebox4, the capped HIP build linked the runner and all existing focused
targets.  The retained P45 provider sentinel and ordered join passed on gfx1201
and gfx1151; common MoE tests passed 2/2 on both; sparse-K passed 68/68 on both.
The measured runner SHA-256 is
`528d722d294697140713f7466d62ac1279f87e9f4961215056d9d0551abd833b`.

## Adjacent A/B/A fact

The immutable roots are
`/home/duster/kimi-k3-deploy/p46-persistent-routed-aba-20260819/{a1-off,b-on,a2-off}`.
All arms use the same runner, GPU1 core, P42 ordered join, P45 asynchronous
compact queue, prompt and storage policy.  Only
`DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP` changes.

| Arm | Decode | True AR | Total | Routed prep | Experts | Join |
|---|---:|---:|---:|---:|---:|---:|
| A1 P46 off | 5.148113908 s | 1.553967170/s | 643.278 ms | 274.716 ms | 333.207 ms | 21.509 ms |
| B P46 on | **4.765920678 s** | **1.678584379/s** | **595.498 ms** | **229.100 ms** | 332.835 ms | 22.189 ms |
| A2 P46 off | 5.973579903 s | 1.339230433/s | 746.448 ms | 275.276 ms | 435.315 ms | 22.025 ms |

The controls have noisy expert/storage time, so their bracket is not used to
claim a precise average uplift.  The attribution is nevertheless unusually
clean:

- routed preparation is stable within 0.560 ms across the two controls;
- P46 removes 45.896 ms/position, or 16.68%, from that stage;
- B's expert stage differs from the fair A1 control by only -0.372 ms;
- B direct-I/O time, 9.852 seconds, lies between A1's 9.438 seconds and A2's
  12.025 seconds;
- B is 8.02% faster than A1 and faster than both controls.

P46 initializes 68 graphs with a 6,873,856-byte shared compute workspace and
93,649,952 bytes of host graph metadata.  It completes exactly 2,856 persistent
graph executions: 68 recurrent routed layers across 42 model positions.  Peak
GPU1 memory is 61,200.86 MiB versus 61,180.74 MiB in A1.

## Exactness and common work

- Every arm produces the same Tokyo continuation and output token IDs.
- Full-logit SHA-256 is
  `cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`.
- Logical-traffic SHA-256 is
  `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`.
- Each arm completes P41 17,917/17,917 with zero fallback, invalid or readback;
  P42 publishes 17,917 expert rows and performs 3,864 joins; P45 submits 17,917
  graphs with zero abort synchronization.
- Logical payload/H2D and physical direct-read bytes are unchanged.  P46 alters
  graph lifecycle only.
- All arms report zero process swap.  Major faults are 46 / 0 / 4.

## Size and next boundary

Against the exact pre-integration graph/cache sources, the production slice is
+422/-29 raw lines and **+375 pure production lines** (+2 comment, +16 blank).
The implementation remains default-off until a consolidation pass removes
duplicated benchmark/production graph construction and broad qualification is
earned.

P46 raises the exact short ceiling from P45's 1.473706970/s to 1.678584379/s,
but it does not reach 2/s.  The B transition still spends 332.835 ms in compact
experts and 229.100 ms in routed preparation.  The next bounded path is genuine
cost-aware concurrent P43b ownership across GPU1 and GPU0 while retaining the
canonical GPU1 ordered join.  It must overlap owner work; P43a's serialized
all-GPU0 execution is not the design.
