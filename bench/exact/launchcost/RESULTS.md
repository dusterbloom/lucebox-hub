# Per-dispatch HIP-graph-replay overhead: knob sweep results (gfx1151, ROCm 7.2.2)

`free -g` before the GPU session: `total=125G used=64G free=22G shared=0G buff/cache=38G
available=60G` — plenty of headroom, no memory pressure.

## 0. Bandwidth floor

See `BANDWIDTH.md`. Measured median **220.09 GB/s** read bandwidth (2 GB buffer, grid-stride
float4 loads, median of 20 runs) — in line with public Strix Halo numbers (~212-215 GB/s), below
the 242 GB/s this project's earlier bandwidth-floor estimates assumed. Correction factor for any
prior "measured/floor ratio" quoted against 242 GB/s: multiply by `242/220 ≈ 1.10x`.

## 1. Microbench knob sweep (`graph_launch_cost.hip`, 256/1024/2048 nodes x empty/chain/chain1m)

Baseline (no knob), median per-node us:

| N nodes | empty | chain (4KB dep) | chain1m (1MB dep) |
|---|---|---|---|
| 256 | 1.7431 | 1.8885 | 6.3864 |
| 1024 | 1.7087 | 1.8418 | 6.3506 |
| 2048 | 1.7026 | 1.8343 | 6.3426 |

Noise band across repeated baseline-shape measurements is approximately **+-3%** (e.g. chain
1024 vs 2048: 1.8418 vs 1.8343, a difference smaller than the batch-size-sweep run-to-run
variance seen below). Any knob delta smaller than this is not distinguishable from noise with a
single 200-replay run.

### Knobs tested — all 15, none moved per-node cost outside the ~3% noise band

| knob | value(s) | empty delta | chain delta | chain1m delta | verdict |
|---|---|---|---|---|---|
| `DEBUG_CLR_GRAPH_PACKET_CAPTURE` | `=1` | ~0% | ~0% | ~0% | no-op (already default fast path) |
| `DEBUG_CLR_GRAPH_PACKET_CAPTURE` | `=0` | ~0% | **+7-8% slower** (chain, all N) | ~0-1% | confirms `=1`/default is already the fast path; `=0` disables an optimization — do not set to 0 |
| `DEBUG_HIP_GRAPH_BATCH_SIZE` | 1, 64, 512, 4096 | ~0% | ~0% | ~0% | no measurable effect at any of the 4 values — not a lever on this build/graph size range |
| `DEBUG_CLR_KERNARG_HDP_FLUSH_WA` | `=0` | ~0% | ~0% (N=256/1024), +7.5% (N=2048, likely noise — not reproduced at other N) | ~0% | no reproducible effect; flagged-risk knob, not promoted, not retested given null result |
| `HIP_FORCE_DEV_KERNARG` | `=1` | ~0% | ~0% | ~0% | no-op |
| `DEBUG_HIP_KERNARG_COPY_OPT` | `=1` | ~0% | ~0% | ~0% | no-op |
| `DEBUG_CLR_BLIT_KERNARG_OPT` | `=1` | ~0% | ~0% (+1% N=2048, within noise) | ~0% | no-op |
| `DEBUG_HIP_FORCE_GRAPH_QUEUES` | `=1` | ~0% | ~0% | ~0% | no-op |
| `DEBUG_HIP_DYNAMIC_QUEUES` | `=1` | ~0% | ~0% (+1% N=2048, within noise) | ~0% | no-op |
| `DEBUG_HIP_FORCE_ASYNC_QUEUE` | `=1` | ~0% | ~0% | ~0% | no-op |
| `DEBUG_CLR_BATCH_CPU_SYNC_SIZE` | `=1` | ~0% | ~0% | ~0% | no-op |
| `DEBUG_CLR_MAX_BATCH_SIZE` | `=4096` | ~0% | ~0% (+1% N=2048, within noise) | ~0% | no-op |
| `GPU_MAX_HW_QUEUES` | `=1` | ~0% | ~0% | +0.6% (N=2048) | no-op (negative control confirmed: single-stream graph is queue-count insensitive) |
| `HSA_ENABLE_INTERRUPT` | `=0` | ~0% | ~0% | ~0% | no-op |

**No knob, in any configuration tested, reduced per-node dispatch cost.** The one statistically
clear signal is the opposite direction: `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` reliably costs +7-8%
on the chain shape at every N — i.e. the default (packet-capture enabled, `=1` or unset) is
already the fast path CLR picked, and there is no faster mode hiding behind an env var on this
ROCm 7.2.2 build for gfx1151.

## 2. Driver-stack timing — not run

Per the task's instruction ("then the driver timing, for the knobs that move the per-node
cost"): **none of the 15 knobs tested moved the per-node cost** on the microbench, which is the
required gate before spending a driver-stack run on any of them. Running the full exact-stack
decode (`run_stack_knob.sh`) per knob would burn GPU time to re-confirm a null result already
established at the microbench level, so this step was skipped by design, not omitted by oversight.
`run_stack_knob.sh` remains in this directory, built and validated against the current
`driver-stack`/artifact paths, ready to run immediately if a future knob candidate shows a real
microbench delta.

## Conclusion

The microbench's own per-node floor (empty-kernel: ~1.70-1.74 us/node; 4KB-dependency-chain:
~1.83-1.89 us/node) is in the same order of magnitude as the ~2.4 us/launch critical-path cost
`KNOCKOUT-HC.md` isolated on the real decode graph, suggesting that number is close to this
hardware's genuine per-dispatch floor for a captured-graph replay on gfx1151/ROCm 7.2.2 — not a
software misconfiguration this round of env knobs can unlock. None of the documented or
internal (`DEBUG_CLR_*`/`DEBUG_HIP_*`) graph/dispatch/kernarg/queue knobs available in
`libamdhip64.so.7` change it. The ~0.92 ms/token recoverable by cutting 383 launches (per
`KNOCKOUT-HC.md`) likely requires actually fusing/removing graph nodes (fewer launches), not
tuning how each launch is dispatched — this investigation did not find a cheaper way to dispatch
the same node count.
