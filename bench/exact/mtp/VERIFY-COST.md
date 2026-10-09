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
