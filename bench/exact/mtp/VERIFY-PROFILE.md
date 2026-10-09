# Verify k=1 vs k=4 kernel profile (2026-10-09, lucebox4, Q8 GGUF, rocprofv3 --kernel-trace)

Traces: box `/home/duster/qwen4exp-f16/scratch/vprof/`. Runner: `run_vprof.sh`.

- **Busy time:** GPU busy time grows +6.75 ms per extra row (default) and +6.15 ms (LUCE_MMID_GROUPED=1, TYPES=5). The grouped kernel is within noise.
- **Idle time:** at k=4 unprofiled (79.36 ms), ~20 ms is GPU idle. In the trace, 73 gaps of >150 us (median ~225 us) add up to 17.9 ms per call. They sit after the router `k_argsort` / before the expert `quantize_q8_1` -> `mul_mat_vec_q_moe`. There is no `sync_fallback` dispatch. Host-side cause under investigation.
- **bench_verify caveat:** `bench_verify.cpp` feeds the same token on all k rows (`toks(k, token)`), so every row routes to the same experts. Expert-byte cost per row is NOT represented. VERIFY-COST.md per-row figures understate real verify for the MoE part.
- **Grouped-expert kernel coverage:** it cannot cover the trunk's Q5_1 down_exps (41 of 48 layers; no Q5_1 case in `mmid_grouped_type_ok`).

## Root cause of the idle gaps (rocprofv3 --hip-trace, k=4)
Traces: `/home/duster/qwen4exp-f16/scratch/vprof/hiptrace_k4/`.

- **Measured:** 68 gaps >150 us, 15.6 ms per call. 83.5% of the gap time is inside `hipLaunchKernel` (833 of 3692 launches take >50 us). `hipMalloc` is 2.8% and `hipMemcpyAsync` 1.4%.
- **Why every launch is eager:** verify never replays a HIP graph. `qwen4exp_graph.cpp:1690` calls `ggml_backend_cuda_graph_invalidate_range` over the verify workspace arena on every call before rebuilding the graph. That syncs the device and erases verify's cached graph entry, so the 2-call warmup never completes.
- **Fix:** upstream main (#827, f2f54154) retains one graph per verify width (`cache.verify_workspace[n_tokens]`, `use_stable_graph = (reuse_ws || verify) && ...`). Port that rather than patch here.
- **Expected:** <= ~16-18 ms per k=4 call.
