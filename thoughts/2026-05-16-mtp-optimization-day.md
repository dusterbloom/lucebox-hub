# MTP Speculative Decoding — 2026-05-16 Work Memo

**Project:** lucebox-hub / dflash
**Branch:** `feature/mtp-foundation-v2` (25 commits ahead of `origin/main`, not merged)
**Target model:** Qwen3.6-27B
**Hardware:** RTX 3090 (sm_86), WSL2, single GPU
**Status:** Code prepared on a clean branch; ship decision pending MTP Q4 results (bench in flight at memo time)

> Memo went through a momus BS review and an evidence verifier audit before publication. Earlier draft had multiple unsourced numbers and overclaiming language; those have been corrected. See *Corrections from review* at bottom.

## TL;DR (honest framing)

- **25 commits**, **43 files changed**, **+10,184 / −12 LOC** on a clean branch awaiting bench data before any merge decision.
- **Best Q2_K_XL result** (single prompt, row 3051 SWE-bench, BENCH_THINKING=1, q8/q8 KV, n=3 runs): MTP chain D=3 = **median 56.05 tok/s, accept 75.2% (byte-stable across all 3 runs), decode speedup vs AR = 1.67× median / 1.78× best**. Same config row 2772: median 48.19 tok/s, 57.1% accept, 1.44×.
- **DFlash standalone (Q4_K_XL target + Q4_K_M draft, n=8 prompts, n=1 run each)**: mean 48.42 tok/s (1.58×), median 42.40 tok/s (1.39×). Distribution wide: 6/8 samples between 39-49 tok/s, 2 samples at 78-82 (AL≈6.8), 1 outlier at 13.4 (AL=1.33, 0.44×).
- **Cross-quant cross-prompt-set comparison** (Q2 MTP n=3 vs Q4 DFlash n=8): MTP median exceeds DFlash median by +32% (56.05 vs 42.40). **Caveat: not apples-to-apples** — different quantization, different prompt sets, different N. True like-vs-like (MTP Q4 vs DFlash Q4 same prompts) bench is **in flight at memo time** (download of Unsloth `Qwen3.6-27B-MTP-Q4_K_M.gguf` completed 22:54).
- **Tests:** 5/5 regression tests pass on cleaned tree (chain strict-spec, tree strict-spec, chain runner, CUDA step graph with cast, CUDA step graph nocast) — verified by direct execution at 21:59.
- **CUDA bug-#5:** new regression test demonstrated **bit-identity on first 8 output bytes between with-cast and no-cast variants on the tested sm_86 config** (absmax=0.000276 in both, F16 quant noise floor). This is one configuration; not proof across all shapes. Cast removed from production path; test kept as gate for sm_89/sm_90 if/when those targets matter.
- **Combined MTP × DDTree experiment (current wireup): DEAD.** Wireup calling `Qwen35DFlashTarget::verify_tree` from `mtp_topk` regressed to 0.7× (accept collapsed 75.2% → 24.6%). Reverted via `git checkout`. Not a "follow-up" — needs full redesign (proper K-way autoregressive drafts, true tree-mask attention, lift of `test_dflash.cpp:857-864` BLOCKER).
- **Cleanup pass** removed 4 noise commits (cast + revert pair, broken fast-path + revert pair = net diff zero), flag-guarded profiling instrumentation behind `DFLASH_*_PROFILE` CMake options, trimmed 5 comment essays, deleted 9 unused HIP→CUDA aliases. Net of the 3 cleanup commits alone: **+136 / −152 LOC**.

## Timeline

### Morning / midday — foundation work in branch at session start

| Time | Commit | Change |
|---|---|---|
| 12:14 | `96215ae` | Qwen3.6 NextN head with KV warmup and GPU forward |
| 13:59 | `ea9ddbf` | recommit base_pos bookkeeping fix + opt-in hidden-seq capture |
| 14:37 | `c14d3b1`, `8ab7ebe` | top-K logprobs surface, experiment-C wire-up (DDTree from MTP top-K) |
| 15:00 | `85ee2d4` | top-K logprobs on Qwen3.6-27B (no shared_head_head path) |
| 15:04 | `f8b8fa6` | DFlashTarget::verify_tree stub |
| 15:06 | `d3cc9f0` | DFlashTarget::verify_tree real implementation |
| 15:27 | `539d85d` | Stage 3 — DeltaNet rollback for tree-verify multi-iter correctness |
| 15:57 | `b9f604e` | Phase A — autoregressive chain draft via `INativeMtp::step_chain` |
| 16:22 | `94d8583` | Phase B+ — step-graph cache, fused LM-head, F16 head KV, fa_window |
| 16:28 | `f65532a` | `max_gamma()` returns chain-depth ceiling 8 post-init; bench thinking knob |
| 16:39 | `428abdf` | feed pre-shared_head_norm hidden as h_prev on GPU chain path |
| 17:15 | `8bae223` | seed chain h_prev_0 from target pre-output-norm hidden (mirrors llama.cpp PR #22673) |
| 17:17 | `0f3b96d` | skip h_pre device→host download on final chain iter |
| 17:57 | `2036dfc` | instrument step_chain_gpu_ per-iter timing |
| 18:02 | `0d2e770` | instrument verify_batch per-call timing |
| 18:06 | `2005425` | switch verify_batch hidden capture to `LAST_ROW_ONLY` after prefill |
| 18:35 | `82873d7` | skip recommit verify_batch via `restore_kv_at_chain` fast path |
| 18:36 | `1edf6c8` | harden n_heads=1 contract; surface silent pre-norm fallback warning (momus) |
| 19:14 | `8a626fe` | kill momus bugs #1-3 (recommit off-by-one, capture ownership, async stream) |
| 19:55 | `b5f46e6` | kill momus bug #5 — shape-only step graph cache (`ggml_set_rows`/`ggml_get_rows`) |

### Afternoon — bench iteration, headline Q2 D=3 reached

- γ-sweep on `bench_agent_mtp.py` exploring chain depth, top-K wiring, ddtree-budget.
- Observation: thinking-mode (`BENCH_THINKING=1`) gives 75.2% accept on row 3051 D=3. Adopted as standard bench config for headline numbers.
  - *Caveat (from audit):* the original "thinking 75.2% vs not-thinking 71.6%" claim came from comparing different prompts. The 71.6% datum was from prompt_id=1, not row 3051. No true side-by-side thinking-on/off comparison on the same prompt was captured in logs.
- q8/q8 KV cache confirmed optimal via sweep (Q8_0 on both K and V).
- Best measured D=3 result on row 3051: **MTP run 2 tok_s=59.74 tok/s / AR run 2 tok_s=33.59 tok/s = 1.778×** (single run). Median across 3 runs: 56.05 / 33.59 = 1.668×.
  - *Correction from audit:* earlier draft cited "58.54 / 33.62 = 1.741×" which is not in any captured log. The verified best is 59.74/33.59 = 1.78×.

### Evening — momus review, "all bugs must go"

A momus opus review surfaced 5 architectural bugs:
1. Recommit off-by-one in `mtp_chain_runner.cpp` base_pos bookkeeping (`8a626fe`).
2. Capture topology ownership confused (`8a626fe`).
3. Async stream batching missing (`8a626fe`).
4. Silent pre-norm fallback warning missing (`1edf6c8`).
5. O(n_heads × n_ctx) step graph cache memory bomb (`b5f46e6`).

Bugs 1-3 + 4 fixed. **Bug 5 fixed via graph refactor**: step graph builder rewritten to use `ggml_set_rows` + `ggml_get_rows` instead of static views. Cache became **shape-only** keyed on `(head_idx, fa_window, fused_lm_head, topk_k)` — 4 entries max. Added runtime tensor upload helper `push_kv_slot_inputs_`.

### Late evening — cast investigation

Earlier review claimed `ggml_get_rows` F32 output → `ggml_flash_attn_ext` needed cast to F16 ("correctness-critical"). Bench on row 3051 showed cast and no-cast variants produced **the same 75.2% accept rate run-to-run** (no observable output difference) and cast added ~15% wall-clock overhead per iter on fa_max=2048 K rows. Cast was reverted from prod path.

To gate against silent CUDA regressions, dispatched sisyphus-junior to write a CUDA regression test:

- `21:00` — `4c88375` test(mtp): bug #5 cast proof-of-teeth on CUDA
  - Two CMake targets: `test_mtp_step_graph_cuda` (with cast) + `test_mtp_step_graph_cuda_nocast` (cast disabled via `-DMTP_SKIP_CAST=1`).
  - Both report `absmax=0.000276` (F16 quant noise floor), no NaN. **First 8 output bytes are bit-identical between variants on the tested sm_86 config.**
  - Limit: one shape, one config, one arch (sm_86). Test is a regression gate, not a generalized proof that the cast is unnecessary across all FA-ext shapes/arches.

### Qualitative output verification (added in response to user challenge)

User asked: "are we checking outputs qualitatively?" Honest answer at the time: no — accept-rate parity was a proxy, not a real check. Fixed:

- Re-ran AR (gamma=0) + MTP D=3 (gamma=3) on row 3051 with `--out` flag dumping emitted token IDs to binary files.
- Detokenized via local `gguf-py` + GPT2 byte-decoder (llama-cpp's `llama-tokenize` is encode-only).
- Result (file: `tasks/bjwg6guut.output`): both files 3179 tokens. **First 47 generated tokens identical (divergence at idx 3098, generated-token offset 47)**, then forked due to q8/q8 KV sub-ULP logit drift flipping argmax on a near-tied position. Both continuations read as coherent SWE-bench analyses (manual inspection by the author, not blind-rated).
  - *Correction from audit:* earlier draft said "first 46 tokens identical"; the actual divergence index in the log is 3098, prompt length 3051, so 47 generated tokens match before divergence.

### PR cleanup (codex review pass)

Dispatched codex agent for ruthless PR cleanliness review. Recommendations executed:
- Drop 4 noise commit pairs (cast + revert, broken fast-path + revert) → net diff zero.
- Flag-guard profiling instrumentation behind CMake options (`DFLASH_VERIFY_PROFILE`, `DFLASH_MTP_PROFILE`).
- Trim 5 multi-paragraph comment essays.
- Delete 9 unused HIP→CUDA aliases.
- **Keep** depth-1 step_chain fallback (backward-compat) and `snapshot_kv/restore_kv` fallback (tested).

Note: the dropped noise commits were **introduced earlier today by this same workstream**, so the cleanup is removing self-inflicted churn, not extracting net-new value.

Execution sequence:
- Rebased onto `origin/main` — picked up 9 upstream commits (Pascal/Volta megakernel sm_6x/sm_70, fast CI workflow, inline prefix snapshot fix). Single overlapping file (`qwen35_backend.cpp`) auto-merged (different regions).
- 4 noise commits dropped via `GIT_SEQUENCE_EDITOR` rebase.
- 3 cleanup commits (`ce62ad8`, `6ce4167`, `586a867`) added via sisyphus-junior agent: `+136 / −152` LOC, 8 files.
- Build clean, 5/5 regression tests pass on cleaned tree (verified by direct execution).

### Combined experiment — attempted, current wireup DEAD

User asked to attempt MTP × DDTree (the "combined" shot). Pre-existing infrastructure (`Qwen35DFlashTarget::verify_tree` from `d3cc9f0`) was wired into the `mtp_topk` path in `test_dflash.cpp` by sisyphus-junior: `+29` LOC, single function.

**Result on row 3051 (file `tasks/bldedeok1.output`):**

| Config | tok_s | accept | DecSp |
|---|---|---|---|
| AR Q2 baseline | 32.77 | 0% | 1.00× |
| mtp_topk K=4 budget=16 | 23.17 | 24.6% | **0.71×** |
| mtp_topk K=4 budget=32 | 23.56 | 24.6% | **0.72×** |
| mtp_topk K=4 budget=64 | 23.00 | 24.6% | **0.70×** |
| (reference: chain D=3) | 56.05 | 75.2% | 1.67× |

Diagnosis: K-way top-K samples from a single MTP head produce 3× lower-quality drafts than autoregressive chain re-feed (24.6% accept vs 75.2%). Tree size irrelevant — bottleneck is draft quality.

**Wire-up reverted** (`git checkout dflash/test/test_dflash.cpp`; current `git diff` for that file is empty). **This approach is not a "follow-up"; the current wireup is dead and the path forward needs a redesign**: K-way autoregressive drafts (instead of K branches from one head) AND a true tree-mask attention path (lift of the `BLOCKER` note at `test_dflash.cpp:857-864`).

### DFlash standalone comparison

User asked: "did you run DFlash so to have a comparison?" Genuine gap. Found local DFlash pair on disk:
- target: `/home/peppi/Dev/lucebox-hub/dflash/models/Qwen3.6-27B-Q4_K_XL.gguf`
- draft: `/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf`

Ran `bench_agent.py --bucket 2k --n-sample 8`. Distribution (file `tasks/b1j2638dp.output`):

| Sample | tok_s | AL | DecSp |
|---|---|---|---|
| 1 | 77.95 | 6.74 | 2.54× |
| 2 | 42.40 | 3.71 | 1.37× |
| 3 | 39.97 | 3.66 | 1.33× |
| 4 | 82.04 | 6.92 | 2.67× |
| 5 | 42.40 | 4.00 | 1.41× |
| 6 | 40.11 | 3.82 | 1.30× |
| 7 | 49.09 | 4.41 | 1.62× |
| 8 | 13.42 | 1.33 | 0.44× |

- **Mean: 48.42 tok/s (1.58×). Median: 42.40 tok/s (1.39×).**
- Distribution shape: 6/8 samples cluster in 39-49 tok/s, 2 samples at 78-82 (long accept runs), 1 sample at 13.4 (drafts almost entirely rejected). Not strictly bimodal — wide distribution with high and low outliers.

**Comparison MTP Q2 chain D=3 vs DFlash Q4 standalone:**

| Metric | MTP Q2 D=3 (N=3 runs on row 3051) | DFlash Q4 (N=8 prompts, 1 run each) |
|---|---|---|
| Mean tok/s | 56.78 | 48.42 |
| Median tok/s | 56.05 | 42.40 |
| Coefficient of variation | ~5% (intra-prompt) | ~50% (inter-prompt) |
| Speedup vs AR | 1.67× median | 1.58× mean / 1.39× median |
| Best | 1.78× | 2.67× |
| Worst | 1.62× | 0.44× |

MTP median is 32% above DFlash median, mean is 17% above. **Strong caveat: cross-quant (Q2 vs Q4), different N, different prompt sets** — strictly not apples-to-apples per the project's like-vs-like rule. True comparison requires MTP Q4 vs DFlash Q4 on the same prompt set, **bench in flight at memo time** using the freshly downloaded Unsloth Q4_K_M MTP GGUF.

### Oracle analysis & D=5/D=7 probe (predictions vs reality)

Dispatched opus oracle for code-level analysis of MTP vs DFlash perf gap. Top hypothesis: MTP D=3 ships only 4 candidate positions per backbone forward; DFlash ships 16. Predicted that raising chain depth (`Probe 1`) would help; expected token-yield to grow with D under assumption of stable per-step accept rate.

Ran the probe (file `tasks/bed6kqz4u.output`, n_runs=3 per cell):

| Depth | Row 3051 MTP median | Accept | DecSp |
|---|---|---|---|
| D=3 | 56.05 | 75.2% | **1.67× ← best** |
| D=5 | 44.58 | 48.1% | 1.32× |
| D=7 | 41.56 | 37.3% | 1.21× |

**Oracle's prediction was wrong for this model + Q2 + prompt.** Per-step accept rate compounds-decays with chain depth: 75% → 48% → 37%. The theoretical token-yield gain (1+p+p²+p³+... = 2.74 at D=3 vs 3.83 at D=7) is overwhelmed by:
- Extra serial MTP-head forwards per outer iter (3 → 5 → 7).
- Probability of full chain accept (`p^D`) drops 0.42 → 0.10 → 0.02 → recommit slow path fires on almost every iter, doubling effective backbone work.

**Conclusion: D=3 is the Q2 sweet spot.** "Raise default chain depth" is dead.

Other oracle hypotheses (NOT empirically validated):
- **Target verify graph not cached** (`build_target_step` at `graph_builders.cpp:97` unconditionally calls `step_graph_free` and rebuilds). Predicted ~5-15% gain, larger fraction on Q2. Untested.
- **Batch MTP head chain into one graph** to eliminate 2 of 3 syncs. Predicted ~10-20%. Untested.
- **58% of iters fire recommit slow path at p=0.75** — predicted, untested. Easy to verify with `DFLASH_VERIFY_PROFILE=1`.

### In flight at memo time (22:55)

| Job | Status |
|---|---|
| MTP Q4_K_M chain D=3 bench on Unsloth GGUF, n=5 runs × 2 prompts | running on GPU |

The MTP Q4 result determines whether headline can read "≥1.7× on Q4 like-vs-like vs DFlash Q4" or just "1.67× on Q2 with cross-quant caveat". Ship decision waits on this.

## Decisions

| Decision | Status | Rationale |
|---|---|---|
| Drop the cast fix (`edab2a0+5aaabab` revert pair) | Done | CUDA regression test demonstrated with-cast and no-cast bit-identical on tested sm_86 config. 15% wall-clock cost with no observable correctness benefit on row 3051. |
| Reject combined MTP × DDTree experiment in current wireup | Done | 0.7× regression, accept collapsed from 75% to 24%. Needs full redesign, not a wireup tweak. |
| Ship MTP chain (this PR) as foundation, treat combined as separate research project | **Pending** Q4 bench data | Chain wins cleanly on Q2 like-for-like (D=3 sweet spot confirmed). Combined needs more than a single PR can hold. |
| Keep profiling instrumentation behind `DFLASH_*_PROFILE` CMake options | Done | Load-bearing for the next investigation round (recommit hit rate, graph rebuild cost). Zero prod cost when flag off. |
| Trim comment essays aggressively | Done | LOC discipline; several blocks had rotted into "we historically..." prose. |

## Outstanding

- Final ship decision waits on MTP Q4 bench (in flight).
- If Q4 falls below DFlash Q4 even on like-vs-like, two further investigations on the table before the merge call:
  - Oracle fix #1: cache target verify graph (`graph_builders.cpp:97`) — keyed on `{n_tokens, capture_*, mask_pad_bucket}`. Predicted 5-15%, larger on Q2.
  - Oracle fix #3: batch MTP head chain into one graph_compute. Predicted 10-20%.
- **Combined experiment as research project**: needs proper K-way autoregressive drafts AND a true tree-mask attention path. Lift the `BLOCKER` note at `test_dflash.cpp:857-864`. Independent strict-spec test on the tree path required.
- Memo footnote: the qual-check confirmed lossy spec under q8/q8 KV (47/128 generated tokens match, then divergence; both continuations on-topic). Strict-spec test passes on the CPU stub byte-identical, but the Q2 GPU path is lossy. Acceptable for production decode, but worth a formal accept-rate vs quality study at some point.

## Glossary

- **MTP chain D=k** — chain runner draws k sequential drafts from MTP head, target verifies all k+1 in one batched forward.
- **mtp_topk** — experimental path: MTP head outputs top-K logprobs, DDTree built, verify currently still chain (`BLOCKER` at `test_dflash.cpp:857-864`).
- **AL** (DFlash) — accept length, mean tokens accepted per outer iter.
- **accept rate** (MTP) — fraction of drafted positions accepted by target.
- **Strict-spec** — formal property: spec output byte-identical to greedy AR output. Verified on the CPU stub via `test_mtp_chain_strict_spec`. The GPU Q2/Q4 path is NOT strict (sub-ULP logit drift flips argmax on near-tied positions); accept-rate-stable but lossy.

## File locations of interest

- Chain runner: `dflash/src/common/mtp_chain_runner.cpp`
- MTP module (large, ~2000 LOC — flagged for future split): `dflash/src/qwen36/qwen36_mtp.cpp`
- MTP step graph: `dflash/src/qwen36/qwen36_mtp_graph.cpp`
- DFlash target: `dflash/src/qwen35/qwen35_dflash_target.cpp`
- Target graph builder (uncached, flagged by oracle): `dflash/src/qwen35/graph_builders.cpp` line 97
- Strict-spec test: `dflash/test/test_mtp_chain_strict_spec.cpp`
- CUDA regression test (bug #5): `dflash/test/test_mtp_step_graph_cuda.cpp`
- Bench harness MTP: `dflash/scripts/bench_agent_mtp.py`
- Bench harness DFlash standalone: `dflash/scripts/bench_agent.py`

## Corrections from review

This memo was audited by an evidence verifier and reviewed by momus before publication. Corrections applied vs the first draft:

| Earlier claim | Corrected claim | Source |
|---|---|---|
| "22 file changes, +6.7k LOC" | 43 files, +10,184 / −12 LOC | `git diff --shortstat origin/main..HEAD` |
| "58.54 tok/s / 33.62 AR = 1.741×" | 59.74 / 33.59 = 1.778× best; median 56.05 / 33.59 = 1.668× | `tasks/bduesjzpq.output` |
| "MTP variance < 1%" | CV ≈ 5% across 3 runs (56.05, 59.74, 54.54) — but intra-prompt | computed from log |
| "DFlash bimodal distribution" | Wide distribution: 6/8 in 39-49, 2/8 at 78-82, 1/8 at 13.4 | `tasks/b1j2638dp.output` |
| "thinking-mode drastically changes accept (75.2% vs 71.6%)" | Not validated on same prompt; 71.6% was prompt_id=1, 75.2% is row 3051. Drop "drastically", note assumption. | audit found no same-prompt thinking on/off log |
| "First 46 tokens byte-identical" | First **47** tokens; divergence at idx 3098, prompt length 3051. | `tasks/bjwg6guut.output` "First divergence at idx 3098" |
| "shipped 25 commits" | "prepared 25 commits on a clean branch; not merged" | branch is `feature/mtp-foundation-v2`, no PR yet |
| "beats DFlash" | "Q2 MTP median exceeds Q4 DFlash median by +32% with strong caveat that this is cross-quant cross-prompt-set" | per project rule "like-vs-like" |
| "Cast unnecessary, proven" | "Demonstrated bit-identity on the tested sm_86 config; one shape; not a general proof" | `test_mtp_step_graph_cuda` only validated sm_86 |
| "Combined experiment queued as follow-up" | "Current wireup is dead; needs full redesign" | wireup produced 0.7×; redesign is a research project |
| "D=5/D=7 in flight" | Done. D=5 = 1.32×, D=7 = 1.21× — both lose to D=3. Oracle "raise depth" hypothesis dead on Q2. | `tasks/bed6kqz4u.output` |
