# MTP target-verify GPU cost vs T=1 (exact stack, gfx1151)

Box: duster@100.115.193.112. Lib actually run: libggml-hip.so.0.9.11
sha256 `8be9b29a4d03e9c7d52fad40c6a895c22dfa1f92fbf8019819188aefefa9a7fa`
(built 2026-10-09 12:13 UTC, after the hc_lo-fusion rebuild; libluce_common.a
unchanged since 2026-10-08 17:50, newer than qwen4exp_graph.cpp so no stale
object). No build was in flight when these runs executed.

Harness: no existing GPU-timing tool exposed verify=true at n_tokens=k
(bench_qwen4exp_decode.cpp only drives n_tokens=1; test_qwen4exp_mtp*.cpp are
CPU-only unit tests). Wrote `bench/exact/mtp/bench_verify.cpp` (57 lines),
which calls the existing `qwen4exp_forward(..., verify)` / `create_qwen4exp_cache`
API directly: prefill ~8000 tokens, then chrono-time `reps+3` consecutive
forward calls per width k (first 3 discarded as warmup), k=1 with
`verify=false` (plain T=1 decode), k>1 with `verify=true`. No production
source was edited. Compiled against the box's existing build1151 libs with
the same manual link line as `build_driver_stack.sh`. 20 measured reps per k,
8K-ish context (pos drifts 8032->8423 across widths within one run; attention
cost at that scale is flat over the drift). Host chrono wall time around the
(synchronous, logits-readback) forward call — this is GPU wall time, no
separate host-only component isolated.

Models: Q8 = Qwen3.8-Flash-Next-UD-Q4_K_XL (MoE), A3 = dense Q6_K
Qwen3.8-Flash-Next-dense-Q6K-a3. Same MTP sidecar for both
(mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf; A3's own repo has no MTP/ dir, so
the sidecar path was passed explicitly instead of relying on discovery).

| k | Q8 ms | A3 ms | ms/row (Q8) | ms/row (A3) | ratio vs T=1 (Q8) | ratio vs T=1 (A3) |
|---|------:|------:|------------:|------------:|------------------:|------------------:|
| 1 | 43.19 | 40.27 | 43.19 | 40.27 | 1.00 | 1.00 |
| 2 | 56.65 | 58.00 | 28.33 | 29.00 | 1.31 | 1.44 |
| 3 | 68.08 | 72.80 | 22.69 | 24.27 | 1.58 | 1.81 |
| 4 | 79.36 | 87.96 | 19.84 | 21.99 | 1.84 | 2.18 |
| 8 | 131.13 | 157.73 | 16.39 | 19.72 | 3.04 | 3.92 |

Per-row cost falls with width on both models (Q8_0 MMVQ weight reuse across
invariant columns helps more than Q6_K's single-column passes: Q8 ms/row
drops 43.19->16.39 (-62%) vs A3's 40.27->19.72 (-51%) from k=1 to k=8), so a
verify forward is cheaper per candidate row than four separate T=1 forwards
at every width tested, and the type gap (Q8_0 vs Q6_K) widens with k.

Kernel count per verify forward: not measured. rocprofv3/rocprof is not
installed on this box (/opt/rocm-7.2.2 only ships rocprofiler-register, no
CLI binary), and no existing driver hook reports ggml_cgraph node count to
an external caller — exposing one would require a production-source edit,
which was out of scope per the task's minimal-edit constraint. Flagging this
as the one unmet sub-goal rather than fabricating a number.

## Graph-replay headroom (measurement only, no source edits)

Lib: same build1151 libggml-hip.so.0.9.11, sha256
`96ac2100d2794842849d16a5788dbb0b0a1ea6f668f09fdd196a4dc045851772`
(no rebuild happened between runs below).

Knob found: `GGML_CUDA_DISABLE_GRAPHS_DEVICES=<device-index>`
(ggml-cuda.cu:416-434, consulted at ggml-cuda.cu:7754). Device index inside
the process after `HIP_VISIBLE_DEVICES=1` is `0` (confirmed from the
`Device 0: Radeon 8060S` log line). This knob disables HIP graph
capture/replay only; the qwen4exp stable decode workspace (`decode_ws`,
same `ggml_cgraph`, same device buffers) is untouched either way — it is
arm B as specified (replay off, persistent workspace kept). No second knob
exists to force a full per-call eager rebuild of the T=1 stable graph
without a source edit (`use_stable_graph` has no env override in
qwen4exp_graph.cpp), and routing k=1 through `reference=true` would also
swap in the unfused/differential-check kernels, not just the graph
strategy, so that is not a clean arm C — arm C is skipped rather than
faked. Confirmed with `GGML_CUDA_GRAPH_STATS=1` (ggml-cuda.cu:7837-7855,
`GGML_CUDA_GRAPH_STATS_EVERY=5`) that the knob does what it claims: arm A
shows `replay=18/20` (warmup then steady-state replay) at `n_nodes=5227`;
arm B on the same build shows `eager=20/20, replay=0` at the same
`n_nodes=5227`. No `LUCE_QWEN_*` fusion envs were set in either arm
(matches verify's default-off fusions).

20 reps after 3-rep warmup, Q8 model, 8K-ish context (pos 8021-8423,
identical range to the table above), fresh process per arm, interleaved:

| arm | k | median ms | delta vs A |
|---|---|------:|------:|
| A (graphs on, default) | 1 | 43.11 / 43.09 (two runs) | — |
| B (`GGML_CUDA_DISABLE_GRAPHS_DEVICES=0`) | 1 | 43.07 / 42.96 (two runs) | **-0.1 ms (-0.2%)** |
| A (graphs on, default) | 3 (verify) | 67.52 / 67.40 (two runs) | — |
| B (`GGML_CUDA_DISABLE_GRAPHS_DEVICES=0`) | 3 (verify) | 67.08 / 66.70 (two runs) | -0.3..-0.8 ms, within the arm's own run-to-run spread |

The A-B spread (±0.1-0.2 ms) is the same size as each arm's own
run-to-run noise. **The graph-replay prize for a single T=1 forward is
~0, not a few ms** — HIP graph replay only removes kernel-launch overhead,
and at `n_nodes=5227` over a ~43 ms forward that overhead is a rounding
error against the actual GEMV/MoE compute time.

Node counts (from the same `GGML_CUDA_GRAPH_STATS=1` capture,
`GGML_CUDA_GRAPH_STATS_EVERY=1`): k=1 stable graph = **5227 nodes**, fixed
across calls (that's what makes replay possible). k=3 verify = **6499-6847
nodes**, a different count on almost every call (6847, 6499, 6847 across 3
consecutive forwards) — the verify graph is never structurally identical
twice, so `ggml_cuda_graph_update_required` would see a "properties
changed" graph every time even if capture were attempted.

Verify k=3 with HIP graphs enabled vs disabled: **identical — eager both
ways**, `total=1` per distinct call with `replay=0, capture=0, eager=1` in
every sample whether `GGML_CUDA_DISABLE_GRAPHS_DEVICES` is set or not.
Generic CUDA-graph capture does nothing for verify today: it never even
attempts to capture (node count isn't stable enough to pass
`ggml_cuda_graph_update_required` twice in a row), so the "enabled" flag
in the knob is moot for k>1.

**Bottom line for anyone about to spend days on this**: at k=1 the
replay path already exists, is active 90% of the time (18/20 after
warmup), and still buys ~0 ms. Building graph replay for verify(k) would
have to first get the k>1 node graph to stop changing shape call-to-call
(mask/indexer-driven branching), and even then the ceiling demonstrated
by the k=1 A-B delta is ~0.1-0.2 ms out of 43-68 ms. Not worth building
on this box/model/context regime.
