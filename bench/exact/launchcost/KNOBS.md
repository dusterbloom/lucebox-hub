# Candidate runtime knobs for HIP-graph dispatch/launch overhead (gfx1151, ROCm 7.2.2)

Context: `KNOCKOUT-HC.md` arm A-vs-B isolates **-0.92 ms/token for 383 launches**, i.e. ~2.4 us
of critical-path cost per graph-node dispatch, on top of the kernels' own compute. The captured
graph for the exact-stack decode has ~1871 kernel nodes/token, so a 2.4 us/node tax (if uniform)
is ~4.5 ms/token of the 38.4 ms budget. Below are the env knobs most likely to affect per-node
HIP-graph-replay dispatch latency on an APU (gfx1151, unified memory, ROCm 7.2.2 / CLR userspace
dispatch). None of these knobs touch floating-point math or kernel code paths — they only change
how/when the GPU command processor and ROCr fetch kernargs, queue signals, flush caches, or batch
dispatch packets. If any run produces different `measure_tokens` content, that knob is
disqualified regardless of timing.

**Correction (coordinator, confirmed via `strings /opt/rocm/lib/libamdhip64.so.7` on lucebox4):**
the earlier version of this doc said `/opt/rocm` ships no symbol table for these — wrong tool,
checked only `/opt/rocm/include` headers (public API surface), not the shared object's string
table where internal `DEBUG_CLR_*`/`DEBUG_HIP_*` env-var names actually live. They are present and
real in this exact build: `libamdhip64.so.7` strings-matched for `^(DEBUG_CLR|DEBUG_HIP|HIP_FORCE|HIP_GRAPH|GPU_MAX|HSA_ENABLE)` gives:

```
DEBUG_CLR_BATCH_CPU_SYNC_SIZE
DEBUG_CLR_BLIT_KERNARG_OPT
DEBUG_CLR_GRAPH_PACKET_CAPTURE
DEBUG_CLR_KERNARG_HDP_FLUSH_WA
DEBUG_CLR_LIMIT_BLIT_WG
DEBUG_CLR_MAX_BATCH_SIZE
DEBUG_CLR_SYSMEM_POOL
DEBUG_HIP_BLOCK_SYNC
DEBUG_HIP_DYNAMIC_QUEUES
DEBUG_HIP_FORCE_ASYNC_QUEUE
DEBUG_HIP_FORCE_GRAPH_QUEUES
DEBUG_HIP_GRAPH_BATCH_SIZE
DEBUG_HIP_GRAPH_DOT_PRINT
DEBUG_HIP_KERNARG_COPY_OPT
DEBUG_HIP_MEM_POOL_VMHEAP
GPU_MAX_COMMAND_BUFFERS
GPU_MAX_HEAP_SIZE
GPU_MAX_HW_QUEUES
GPU_MAX_REMOTE_MEM_SIZE
GPU_MAX_SUBALLOC_SIZE
GPU_MAX_USWC_ALLOC_SIZE
GPU_MAX_WORKGROUP_SIZE
HIP_FORCE_DEV_KERNARG
HIP_FORCE_SPIRV_CODEOBJECT
```

These are internal CLR/HIP-runtime debug env vars (no public docs; names and plausible semantics
inferred from naming convention + ROCm/HIP source knowledge of the CLR graph/dispatch path).
Every knob below is a hypothesis to test empirically, not a documented guarantee. The microbench
and driver-stack runs are the actual evidence; this doc only ranks what to try first.

## Highest-confidence candidates for per-dispatch cost

### 1. `DEBUG_CLR_GRAPH_PACKET_CAPTURE=1` — TOP CANDIDATE (promoted per coordinator)
- **What**: gates how CLR constructs AQL dispatch packets when capturing a HIP graph. The
  `_CAPTURE` naming and co-location with the other `DEBUG_HIP_GRAPH_*`/`DEBUG_HIP_FORCE_GRAPH_QUEUES`
  symbols in the same string block strongly suggests this controls whether packet construction at
  *capture* time takes a fast path (e.g. pre-building final AQL packets once, so replay just
  re-submits fixed bytes) vs a slower path that re-derives packet fields at each `hipGraphLaunch`.
  If the default path re-validates/rebuilds packet state on every *replay* rather than freezing it
  at capture, this is exactly the per-node replay tax the knockout isolated.
- **Why it matters here**: direct AQL-packet-path knob, co-located in the binary with the other
  graph-dispatch internals (`DEBUG_HIP_GRAPH_BATCH_SIZE`, `DEBUG_HIP_FORCE_GRAPH_QUEUES`) — same
  subsystem as the ~1871-node captured decode graph.
- **Numeric risk**: packet-construction mechanics only, not kernel arguments or math. Must still
  be verified bit-identical — some internal debug flags gate correctness checks that, if skipped,
  could silently let a malformed packet through. Treat a "faster but still bit-identical
  measure_tokens" result as required proof, not an assumption.
- **Test**: both as `=1` and (if `=1` does nothing) `=0`, since the direction of this flag's
  boolean sense is inferred from naming, not confirmed.

### 2. `DEBUG_HIP_GRAPH_BATCH_SIZE=<N>` — sweep, second-highest confidence
- **What**: most plausibly controls how many graph nodes get batched into a single command-buffer
  submission to the hardware queue during replay, rather than one submission per node. If replay
  currently submits in small batches (or one at a time), raising this could directly amortize the
  per-dispatch tax the knockout measured across more nodes per submission — this is the most
  mechanistically direct explanation for a flat ~2.4 us/node cost we've seen (fixed per-submission
  overhead divided by a small/default batch size).
- **Sweep values**: `1` (worst case / isolate per-submission cost, expect this to be *slower* than
  default and confirms the mechanism), `64`, `512`, `4096` (likely larger than our ~1871-node
  graph, i.e. "whole graph in one batch" — if this is fastest, it's strong evidence the entire
  dispatch tax is submission-batching overhead, not per-node hardware-queue processing).
- **Numeric risk**: none — submission grouping, not computation or ordering of results (the
  dependency graph itself, not just the batching of its submission, determines execution order).
- **Test plan**: run this as a 4-way sweep in the microbench (`chain`/`chain1m` shapes, since batching
  interacts with dependency depth) before touching the driver.

### 3. `DEBUG_CLR_KERNARG_HDP_FLUSH_WA=0` — alongside `HIP_FORCE_DEV_KERNARG=1`
- **What**: `HDP_FLUSH_WA` = "Host Data Path flush workaround". APUs with a unified memory
  controller sometimes need an explicit HDP (or equivalent write-combining buffer) flush after the
  CPU/GPU writes a kernarg buffer, to guarantee the GPU's command processor sees the up-to-date
  bytes before it fetches them for a dispatch — this is a cache/write-buffer coherency workaround,
  not a decorative knob. Disabling it (`=0`) skips that flush, which is a latency win **only if the
  workaround is unnecessary for this specific path** (e.g. kernarg already written once at graph
  capture time and never touched again — all our node args are static across replays once the
  graph is captured).
- **Coherence risk — flagged explicitly, per coordinator's ask**: this is the one knob on this list
  with a real risk of **silent data corruption that doesn't crash**, specifically garbage-but-
  plausible-looking kernarg values (wrong pointer/wrong scalar) rather than a trap. The WA exists
  presumably because gfx11 APU HDP semantics require it in the general case. **This MUST be tested
  with the full bit-identical `measure_tokens` gate before being trusted — do not accept "ran
  without crashing" as proof; silent kernarg corruption under a graph of ~1871 fixed-argument nodes
  can easily reproduce the *same* token sequence by luck on one run and diverge on the next.** Run
  it 3x not 1x, and diff `measure_tokens` byte-for-byte across all 3 reps, not just vs baseline.
- **Why test anyway**: it is specifically a kernarg-path-adjacent flush, same subsystem as
  `HIP_FORCE_DEV_KERNARG` (#4), and the two may compose (device-resident kernarg memory could make
  the flush workaround moot, since there's no host-side write-combining buffer to flush from).
- **Test order**: test `HIP_FORCE_DEV_KERNARG=1` alone first (safe), then
  `DEBUG_CLR_KERNARG_HDP_FLUSH_WA=0` alone (flagged risk, extra identity reps), then the
  combination, only if both individually pass the identity gate.

### 4. `HIP_FORCE_DEV_KERNARG=1`
- **What**: forces the kernel-argument buffer for each launch to be allocated in device memory
  instead of host/pinned system memory, changing where the GPU command processor fetches kernarg
  bytes from before each dispatch.
- **Why it matters here**: ~1871 kernarg fetches/token on the critical path; if kernarg memory
  type affects fetch latency, this directly attacks the measured 2.4 us/node number.
- **Numeric risk**: none — kernarg *location*, not kernarg *content*, changes. Bit-identical output
  expected.
- **Caveat**: graph capture bakes kernarg addresses into the graph at capture time; must be set
  before `driver-stack` builds its graph (whole-process env export satisfies this).

### 5. `DEBUG_HIP_KERNARG_COPY_OPT` / `DEBUG_CLR_BLIT_KERNARG_OPT`
- **What**: two more kernarg-path optimizer toggles in the same string block as #3/#4 —
  `KERNARG_COPY_OPT` plausibly skips a redundant copy when staging kernarg bytes; `BLIT_KERNARG_OPT`
  plausibly uses the GPU's blit (copy) engine instead of a CPU memcpy to stage kernarg buffers.
  Same subsystem, same hypothesis class as #3/#4: all four are permutations of "how do kernarg
  bytes get from where they're written to where the command processor reads them."
- **Numeric risk**: same class of risk as #3 (kernarg *path* optimization can skip a step that was
  there for coherency, not just for show) — treat with the same "3x identity gate" discipline as #3.
- **Test**: lower priority than #1-#4; only run on the microbench first (chain shapes reveal
  kernarg-dependent timing since each node reads argument pointers to the *previous* node's output).

### 6. `DEBUG_HIP_FORCE_GRAPH_QUEUES` / `DEBUG_HIP_DYNAMIC_QUEUES` / `DEBUG_HIP_FORCE_ASYNC_QUEUE`
- **What**: three queue-management toggles in the same block as `DEBUG_HIP_GRAPH_BATCH_SIZE`.
  `FORCE_GRAPH_QUEUES` plausibly pins graph replay to a dedicated queue type (vs sharing with
  regular stream ops); `DYNAMIC_QUEUES` plausibly lets the runtime create/destroy queues
  on demand instead of a fixed pool; `FORCE_ASYNC_QUEUE` plausibly forces async-compute queue use.
  Our decode graph is single-stream, so queue *selection* shouldn't matter per earlier reasoning
  for `GPU_MAX_HW_QUEUES` (#8) — included for completeness since they're graph-labeled, but lower
  priority than #1/#2.
- **Numeric risk**: none expected (queue assignment, not computation).

### 7. `DEBUG_CLR_BATCH_CPU_SYNC_SIZE` / `DEBUG_CLR_MAX_BATCH_SIZE`
- **What**: CLR-side (as opposed to HIP-graph-side) batching/sync-threshold knobs — plausibly
  control command-buffer batch size for the general dispatch path and the point at which the CPU
  blocks to sync with the GPU for non-graph dispatch. May or may not apply to graph replay
  specifically (vs regular stream dispatch) — test on the microbench to find out empirically,
  don't assume from the name.
- **Numeric risk**: none expected.

## Secondary candidates (public API, included from original research pass)

### 8. `GPU_MAX_HW_QUEUES` (default 4 on most ROCm builds) — negative control
- Single-stream graph replay shouldn't be queue-count sensitive. If this changes anything, the
  decode isn't single-stream the way we assume — itself a finding worth flagging.
- **Numeric risk**: none (scheduling only).

### 9. `HSA_ENABLE_INTERRUPT=0`
- Switches ROCr's signal-completion wait from interrupt-driven to busy-polling — affects host-side
  wait-for-signal latency after a dispatch/graph completes, not intra-graph GPU-side
  dispatch-to-dispatch latency. Cheap, zero numeric risk, low expected impact (host waits once per
  token, not once per node).

### 10. `AMD_SERIALIZE_KERNEL` / `AMD_LOG_LEVEL` — confirm OFF (default), do not enable
- Debug knobs that only ever add per-dispatch overhead. Confirm-unset sanity checks for every arm
  (including baseline), never swept as "on".

### 11. `HSA_ENABLE_SDMA=0` / `GPU_MAX_COMMAND_BUFFERS` (public alias class of `GPU_MAX_HW_QUEUES`)
- Copy-engine selection / command-buffer-count caps. Our hot loop is compute-dispatch-heavy, not
  copy-heavy; low expected impact, included for completeness.

## Not swept at all
- `DEBUG_HIP_GRAPH_DOT_PRINT` — graph visualization/debug dump, strictly adds I/O overhead, never a
  speedup; `DEBUG_HIP_MEM_POOL_VMHEAP`, `DEBUG_HIP_BLOCK_SYNC`, `GPU_MAX_HEAP_SIZE`,
  `GPU_MAX_REMOTE_MEM_SIZE`, `GPU_MAX_SUBALLOC_SIZE`, `GPU_MAX_USWC_ALLOC_SIZE`,
  `GPU_MAX_WORKGROUP_SIZE`, `DEBUG_CLR_SYSMEM_POOL`, `DEBUG_CLR_LIMIT_BLIT_WG`,
  `HIP_FORCE_SPIRV_CODEOBJECT` — memory-pool sizing, workgroup-size caps, or codeobject-format
  knobs unrelated to per-node dispatch latency on an already-compiled, already-captured graph.
- `AMD_SERIALIZE_KERNEL` set non-zero, `AMD_LOG_LEVEL` set non-zero — only ever add overhead.
- Any `GGML_HIP_GRAPHS`-adjacent toggle — task keeps HIP graphs ON throughout; this is
  dispatch-overhead-*within*-a-captured-graph, not graph-vs-no-graph.
- ROCt pre-Vega / discrete-GPU-only knobs (`ROC_ENABLE_PRE_VEGA`, `HSA_FORCE_FINE_GRAIN_PCIE`) —
  not applicable to an APU with unified memory.

## Test plan mapping
Microbench (`graph_launch_cost.hip`) sweeps #1-#9 on synthetic N-node graphs (256/1024/2048 nodes,
three node-cost shapes: empty/chain/chain1m) vs no-knob baseline, 200 replays each, report us/node.
`DEBUG_HIP_GRAPH_BATCH_SIZE` gets its own 4-value sweep (1/64/512/4096) per shape/N. Only knobs
that show a real, reproducible us/node delta on the microbench graduate to a driver-stack run
(`run_stack_knob.sh`: full exact-stack env + one knob at a time, baseline vs knob, interleaved
fresh processes x3, `measure_tokens` md5 identical required across ALL reps). Flagged-risk knobs
(#3 `DEBUG_CLR_KERNARG_HDP_FLUSH_WA=0`, #5's two kernarg-copy-opt variants) get 3 identity reps
minimum before any timing number from them is trusted — "didn't crash" is not proof, "byte-identical
measure_tokens across 3 independent reps" is. #10 is a baseline-sanity check run once, not swept.
