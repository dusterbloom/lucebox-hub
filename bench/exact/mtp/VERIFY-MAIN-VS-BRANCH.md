# Verify cost: upstream main 49c6b015 (#827 retained verify graphs) vs our branch (2026-10-13, lucebox3)

Hardware and setup:
- Strix Halo iGPU (gfx1151, HIP_VISIBLE_DEVICES=1). The device name was confirmed in the logs.
- The R9700 on GPU0 was busy with another user's job the whole time.
- Model: UD-Q4_K_XL trunk with the shared-Q8_0 MTP sidecar. KV f16.
- Input: real consecutive tokens from `verify_ids_8k.txt` (code text), prefill ~9K tokens.
- Each width gets 3 warmup calls, then 20 timed reps; the median is reported.
- The branch was run with the default env, without the exact-stack LUCE_QWEN_* fusions. Both arms run the same engine features apart from #827.
- Caches were not dropped before the runs, so treat these numbers as indicative.
- max_ctx differs slightly: 13510 on main, 12000 on the branch.
- Benches: `bench_verify_main` (main API) and `bench_verify_realtok` (branch API, commit dbcc98ae).

| k | main ms | branch ms | main saves | main graph builds/replays |
|---|---|---|---|---|
| 1 (decode) | 46.62 | 46.00 | -0.6 | 1/22 |
| 2 | 57.80 | 62.66 | 4.9 | 1/22 |
| 3 | 70.12 | 76.49 | 6.4 | 1/22 |
| 4 | 77.44 | 89.66 | **12.2 (-13.6%)** | 2/21 |
| 8 | 126.99 | 142.13 | 15.1 | 1/22 |

- **Replay works on main.** Each width builds its verify graph once and replays it every call after that. That is the fix for the per-call invalidate found in VERIFY-PROFILE.md.
- **The gain is real but partial.** The profile measured ~18 ms of idle per k=4 call; replay recovers 12.2 ms of it.
- **Marginal cost per extra row on main is still ~10 ms** (k4->k8: 12.4 ms per row). That is above the +6.75 ms/row of GPU busy time measured before, so the remaining verify cost is in the kernels, not the launches.
- **lucebox3 decode is slower than lucebox4.** k=1 decode here is ~46 ms against 35-38 ms on lucebox4. Compare main and branch only within this table.

Decision: port #827 (retained verify graphs) onto the fresh-from-main branch. Then profile main's k=4 call to see what is left in the per-row cost: Q5_1 down experts falling outside the grouped mmid kernel is the lead.
