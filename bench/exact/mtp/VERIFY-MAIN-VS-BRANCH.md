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

## Q5_1 grouped MMID (Codex idea #2)

Added Q5_1 to the grouped-mmid admission (`mmid_grouped_type_ok`, new mask bit 128) and to
`mul_mat_vec_q_grouped_dispatch` in `mmvq.cu`, gated by the existing `LUCE_MMID_GROUPED_TYPES`
env mask. Default policy (unset env) is unchanged -- Q5_1 is still ungrouped unless bit 128 is set.

**Test first.** Extended `test_deepseek4_mmid_grouped_cuda.cpp`'s dispatch-matrix test: added
`GGML_TYPE_Q5_1` to `k_test_types` and bit 128 to the "grouped" child's `LUCE_MMID_GROUPED_TYPES`
mask (15 -> 143). Built and ran on lucebox3's iGPU (gfx1151, HIP_VISIBLE_DEVICES=1, fresh cmake
configure + `make -j12` of `test_deepseek4_mmid_grouped_cuda` under `scratch/q51grp/`, LUCE_GPU_BACKEND=hip):
bit-exact parity between the grouped and legacy (ungrouped) Q5_1 outputs at every tested width
(2,4,8,9,16), `compared_cases=99`, `grouped_grouped=120`, `parity=PASS`. No output mismatches.
Combine-vec4 parity also PASS (nmse=2.6e-15).

**A/B/C on `bench_verify_realtok`** (same binary, manually linked against a fresh `scratch/q51grp`
build matching `build1151`'s CMakeCache flags; model UD-Q4_K_XL, shared-Q8_0 MTP sidecar, real
tokens from `verify_ids_8k.txt`, 20 reps + 3 warmup, max_ctx=12000). A = grouped on, old type mask
(Q5_1 excluded, `LUCE_MMID_GROUPED=1`); B = same + Q5_1 bit (`LUCE_MMID_GROUPED_TYPES=199`); C =
grouping off (`LUCE_MMID_GROUPED=0`). Order A,B,C,B,A,C, 2 reps per arm. Argmax-sum per row (added
a cheap `LUCE_PRINT_ARGMAX=1` check to the bench) was **identical across all 6 runs** at every k
(k=1:430, k=2:296, k=3:4887, k=4:79218, k=8:10984) -- the grouped Q5_1 kernel is numerically a
no-op on real model logits, as expected from the unit-test parity.

Median ms per arm (median of the 2 runs = mean of 2):

| k | A (no Q5_1 bit) | B (+Q5_1 bit) | C (grouping off) | B-C | B-A |
|---|---|---|---|---|---|
| 1 | 47.94 | 44.44 | 45.82 | -1.38 | -3.50 |
| 2 | 66.73 | 62.80 | 64.31 | -1.51 | -3.93 |
| 3 | 83.70 | 77.19 | 78.94 | -1.75 | -6.51 |
| 4 | 94.08 | 91.27 | 93.99 | **-2.72** | -2.81 |
| 8 | 149.52 | 145.58 | 151.70 | -6.12 | -3.94 |

**Verdict: reject per Codex's 1 ms bar is nominally cleared (k=4 gain 2.7-2.8 ms), but the
evidence doesn't survive the bench-engine-parity "reproduced above noise" check.** The two reps of
each arm disagree in sign: A1 vs B1 at k=4 show B 12.2 ms faster, but A2 vs B2 show B 6.5 ms
*slower*. Run-to-run noise (~10-18 ms, uncached iGPU, no drop_caches) is larger than the claimed
~2.7 ms effect. Do not ship on this data; would need dropped caches and more reps (as done for
`VERIFY-PROFILE.md`'s rocprofv3 traces) to separate a real few-ms win from noise. The code change
itself is safe (default-off, bit-exact parity proven) and can stay; the performance claim is not
proven.

Diff: `bench/exact/mtp/bench_verify_realtok.cpp` (+15/-0, argmax-sum debug print),
`server/deps/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` (+14/-3, bit 128 + Q5_1 dispatch case),
`server/test/unit/test_deepseek4_mmid_grouped_cuda.cpp` (+12/-6, Q5_1 in test matrix + updated
expected counts). 3 files, 32 insertions(+), 9 deletions(-).
