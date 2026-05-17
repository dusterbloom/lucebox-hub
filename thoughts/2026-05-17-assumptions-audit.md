# Assumptions Audit — 2026-05-17

**Author:** momus (opus) | **Reviewed branch:** `feature/mtp-foundation-v2` @ `9052aae`

## TL;DR

Of ~45 explicit and implicit assumptions today's work rests on, **6 are CRITICAL and unverified**, **11 are MAJOR with at most partial verification**, and the rest are minor or verified. The single biggest risk is **AR-baseline comparability** between MTP and DFlash benches: they use different binaries (`test_dflash --gamma 0` vs `test_generate`) and different per-cell warmup behavior (15s sleep vs none), which silently inflates or deflates the MTP-vs-DFlash decode-speedup ratio that the entire "MTP wins" narrative rests on. Second-biggest: the L6 plan and Architecture A both depend on the `mtp_topk` BLOCKER having been lifted, but the code at `dflash/test/test_dflash.cpp:859-863` still says the verify path is chain, not tree — the codex "BLOCKER is STALE" claim is itself stale.

---

## CRITICAL — must verify before any further claim is made

### C1. AR baselines are NOT measured by the same binary across MTP and DFlash benches
- **Assumption:** "AR baseline" tok/s is comparable between the two harnesses.
- **Where load-bearing:** `thoughts/2026-05-16-mtp-optimization-day.md` lines 14-15, 146-156 (MTP-vs-DFlash decSp table); `thoughts/2026-05-17-harness-runbook.md` section 6 reference table; ship-plan PR3 framing.
- **Status:** LIKELY WRONG. `dflash/scripts/bench_agent_mtp.py:223` runs AR via `test_dflash --gamma 0`. `dflash/scripts/bench_agent.py:190-198` runs AR via `test_generate`. Different binaries with different prefill/decode paths. The MTP bench's `decode_speedup = mtp_med["tok_s"] / ar_med["tok_s"]` divides MTP-binary tok/s by DFlash-binary-zero-gamma tok/s, while bench_agent.py computes against `test_generate`. The "MTP 1.574× vs DFlash 1.39×" comparison is between two different baselines.
- **Verify (<10 min):** Run both AR binaries on the same prompt with identical max_ctx and KV settings; compare decode tok/s. If they differ by >5%, every speedup ratio needs recomputation against a single AR.
- **Risk if wrong:** The headline "MTP beats DFlash" may invert. The ship plan PR3 framing collapses.

### C2. The 15-second `time.sleep(15)` in MTP bench has no effect on measurement
- **Assumption:** the WSL2 CUDA teardown mitigation is wall-clock-irrelevant.
- **Where load-bearing:** `dflash/scripts/bench_agent_mtp.py:111`.
- **Status:** PARTIALLY VERIFIED — sleep is outside `subprocess.run` so it doesn't enter `tok_s`. HOWEVER `bench_agent.py` does NOT sleep between cells. The MTP cell therefore always starts with cold GPU caches/cooler GPU; the DFlash cell always starts hot. Systematic bias.
- **Verify:** Diff `nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm` polled during both bench runs at cell start.
- **Risk if wrong:** Cross-bench speedup deltas of ±5% are noise from thermal/clock state, not from algorithm.

### C3. The `mtp_topk` BLOCKER comment is "STALE" (codex claim) — Architecture A is buildable from existing infra
- **Assumption:** codex_next_steps.md line 70 — "The BLOCKER comment at `test_dflash.cpp:857-864` is STALE — `verify_tree` already exists."
- **Where load-bearing:** Architecture A's "2-3 days, ALL infra reused" estimate; `/tmp/momus_l6_tdd_spike.md` Phase D plan.
- **Status:** PARTIALLY WRONG. `Qwen35DFlashTarget::verify_tree` DOES exist as a method. BUT the BLOCKER comment at `dflash/test/test_dflash.cpp:859-863` reads "a true tree-mask verify against the target would need DFlashTarget to grow a tree-verify entry point (or to share test_dflash's spec-decode loop). Today we surface the DDTree build + report mean_tree_size to confirm the composition is invokable; **the verify path is still chain**." So the harness wiring to invoke `verify_tree` from `mtp_topk` does NOT exist — codex confused "function on the class" with "wired into the test harness."
- **Verify (<5 min):** `grep -n "verify_tree(" dflash/test/test_dflash.cpp` — confirm zero matches in that file.
- **Risk if wrong:** Architecture A's "ALL infra reused, 2-3 days" estimate becomes "2-3 days PLUS lift of spec-decode loop out of qwen35 graph builder" — easily +1 week and a refactor.

### C4. The `DFLASH_GPU_TESTS=ON` CMake flag exists
- **Assumption:** runbook can build/skip GPU tests via that option.
- **Where load-bearing:** `thoughts/2026-05-17-harness-runbook.md` section 0; ship-plan PR1; `/tmp/momus_l6_tdd_spike.md` dependencies §2.
- **Status:** LIKELY WRONG. `grep DFLASH_GPU_TESTS dflash/` returns nothing. CMake only defines `DFLASH_VERIFY_PROFILE` and `DFLASH_MTP_PROFILE` (CMakeLists.txt:87-93). GPU tests are gated on `DFLASH27B_GPU_BACKEND STREQUAL "cuda"` (line 692). Passing `-DDFLASH_GPU_TESTS=ON` to cmake is a silent no-op today.
- **Verify (already done):** `grep -n DFLASH_GPU_TESTS dflash/CMakeLists.txt` — no hits.
- **Risk if wrong:** Runbook lies; ship-plan PR1 has a real undone TODO.

### C5. "First 47 generated tokens byte-identical" generalizes to "MTP output quality ≈ AR"
- **Assumption:** memo TL;DR + qualitative-verification section.
- **Where load-bearing:** memo §"Qualitative output verification"; runbook §3 expected output; implicit "ship is safe" decision.
- **Status:** UNVERIFIED at population scale. The check was N=1 prompt, N=128 generated tokens, author-graded coherence. No HumanEval+, no MT-Bench, no blind rater. The 47/128 figure is "first divergence at 47"; nothing said about token-level edit distance or semantic divergence over the remaining 81 tokens.
- **Verify (~30 min):** Run qual diff on 4-5 more prompts, compute longest common prefix + BLEU between continuations.
- **Risk if wrong:** Ship-plan PR3 carries a quality regression hidden inside `decode_speedup` numbers.

### C6. The "Q4 like-vs-like" bench was actually run (the headline that drives ship/no-ship)
- **Assumption:** memo line 188 — "MTP Q4_K_M chain D=3 bench on Unsloth GGUF, n=5 runs × 2 prompts | running on GPU." Runbook §6 quotes "MTP chain D=3 (Q4) | 49.51 tok/s | 1.574× | 67.0%."
- **Status:** UNVERIFIED in agent-accessible scope. Memo cites `tasks/bduesjzpq.output`, `tasks/b1j2638dp.output`, `tasks/bldedeok1.output`, `tasks/bjwg6guut.output`, `tasks/bed6kqz4u.output` — none findable by glob (cleanup between sessions). The /tmp bench logs are Q2 runs.
- **Verify (~15 min):** Re-run runbook §2a command and tee to `/tmp/harness_mtp_q4.log`.
- **Risk if wrong:** Ship decision was made on memo numbers that aren't backed by a re-findable artifact.

---

## MAJOR — should verify before sisyphus dispatches more work

### M1. The drafter `dflash-draft-3.6-q4_k_m.gguf` was trained for the Q4_K_XL target distribution
- **Assumption:** runbook §0 pairing of `Qwen3.6-27B-Q4_K_XL.gguf` (target) + `dflash-draft-3.6-q4_k_m.gguf` (draft).
- **Status:** UNVERIFIED, LIKELY WRONG. `/home/peppi/models/qwen3.6-27b-dflash/README.md` line 22 states: "must be used in conjunction with the target model `Qwen/Qwen3.6-27B`" (the unquantized HF model). Drafter was almost certainly trained against BF16/FP16 target distribution; using it against a Q4_K_XL dynamic quant may explain the 0.44× catastrophic-tail.
- **Verify:** Check z-lab paper for training-target quant.
- **Risk:** DFlash baseline is sandbagged — MTP comparison is unfair UP-ward (MTP looks better than it is in a fair fight).

### M2. The 8 SWE-bench prompts are the same set across MTP and DFlash benches AFTER the C1 fix
- **Assumption:** memo and ship-plan claim alignment after fixing `bench_agent_mtp.py` C1.
- **Status:** PARTIALLY VERIFIED. Both scripts call `select_rows_for_bucket(df, ..., n_sample=8, seed=42)`. Alignment relies on the underlying `df` ordering being deterministic (parquet row order). If parquet file is re-sorted between runs, seed=42 picks different rows.
- **Verify (~3 min):** `python -c "from bench_agent import select_rows_for_bucket, _load_swe_rows; print([r['instance_id'] for r in select_rows_for_bucket(_load_swe_rows(), None, 8, 42)])"` and compare across runs.
- **Risk:** Cross-run drift means "the 8 prompts" silently changes.

### M3. The Qwen3.5 tokenizer is correct for prompts going to Qwen3.6 models
- **Assumption:** `bench_agent.py:49` defaults `DFLASH_TOKENIZER="Qwen/Qwen3.5-27B"`; MTP bench inherits this; runbook does not warn.
- **Status:** UNVERIFIED, POTENTIALLY WRONG. `run.py:107` defaults to `Qwen/Qwen3.6-27B`. The two tokenizers may share BPE merges/vocab but have different special-token IDs / chat templates. Chat-template choice affects `enable_thinking=` interpretation.
- **Verify (~5 min):** `python -c "from transformers import AutoTokenizer; t5=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-27B'); t6=AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B'); print(t5.vocab_size, t6.vocab_size, t5.eos_token_id == t6.eos_token_id)"`.
- **Risk:** Every bench number could shift; chat template-induced "thinking mode" may be ineffective.

### M4. `BENCH_THINKING=1` is the right default and the "75.2% vs 71.6%" comparison was apples-to-apples
- **Assumption:** memo §Afternoon implicitly assumes thinking-mode is the production config.
- **Status:** PARTIALLY VERIFIED. Memo self-corrects: "the original '75.2% vs 71.6% thinking' claim came from comparing different prompts." Choice of `BENCH_THINKING=1` as default is faith-based.
- **Verify (~12 min):** Run §2a with `BENCH_THINKING=0` on same 8 prompts; compare median accept.
- **Risk:** If thinking-mode only marginally helps accept while adding prefill cost, "1.574×" headline degrades.

### M5. q8/q8 KV cache is "optimal" (memo line 54)
- **Assumption:** swept and confirmed; ship-plan and runbook commit to it.
- **Status:** PARTIALLY VERIFIED via `/tmp/bench_kv_*` logs. q8_0/q8_0 d=3 → accept 70.7%. f16/q8_0 d=3 → accept 64.1% (WORSE, not better as oracle question #2 hoped). q8/f16 and f16/f16 logs exist but matrix not fully compared.
- **Verify (~2 min):** `grep -h accept /tmp/bench_kv_*_d3.log` to settle the matrix.
- **Risk:** Oracle's "KV precision swap, Architecture B wins" hypothesis is half-disproven; codex didn't pick it up.

### M6. n=8 prompts × 3 runs gives a defensible "median" claim
- **Status:** UNVERIFIED statistically. With 8 prompts and DFlash CV ≈ 50%, the median's 95% CI is roughly ±30% relative — wider than the +14% gap memo claims between MTP (49.51) and DFlash (42.40). The MTP comparison may be in the noise.
- **Verify (~5 min):** Bootstrap CI on the 8 DFlash samples.
- **Risk:** "MTP median > DFlash median" may not be statistically detectable at n=8.

### M7. CUDA bug-#5 cast removal safe on sm_86 only — but already removed from prod
- **Status:** VERIFIED for one shape on one arch. NOT verified on sm_89/sm_90. Production binary ships with cast removed; no safety net for newer GPUs.
- **Risk:** Silent numerical drift on Ada/Hopper users.

### M8. The 4-PR ship plan is achievable as described — `96215ae` split via `rebase -i edit`
- **Status:** UNVERIFIED. Recipe stages files by directory; no guarantee the staging boundary creates two commits that each build standalone.
- **Verify (~30 min):** Dry-run the rebase on a throwaway branch and check each split commit builds.
- **Risk:** PR1 ships interface + tests that won't link without PR2's impl; CI breaks per-commit.

### M9. Squashing bug-fix commits into parents won't break the build at any intermediate commit
- **Status:** UNVERIFIED. `--autosquash` reorders fixups before the parent; if `8a626fe` fixes a state assumed broken at `82873d7`, squashing may produce an intermediate state where neither bug fires.
- **Verify (~45 min):** `git rebase --exec 'cmake --build build -j4 --target test_dflash'` to test buildability of every commit.
- **Risk:** Reviewer-visible bisect history is broken.

### M10. Median-of-medians aggregation is statistically defensible
- **Status:** UNVERIFIED. Median-of-medians is biased on small N with heavy-tailed distributions. Memo reports both mean and median for DFlash, but MTP reports only median.
- **Verify (~10 min):** Recompute with mean-of-medians + bootstrap CI.
- **Risk:** Headline is gameable by aggregation choice.

### M11. The L6 hybrid kill criterion ("accept ≤ 35% on first prompt") generalizes
- **Status:** PARTIALLY VERIFIED only as a heuristic. One-prompt accept is a noisy signal. Killing at <35% on prompt 1 may kill prematurely if prompt 1 is adversarial.
- **Verify:** Add a second-prompt confirm before kill.
- **Risk:** False-negative kill loses a viable architecture.

---

## MINOR — track as follow-ups

| ID | Assumption | Status |
|---|---|---|
| m1 | `gguf_mmap.h` duplicates upstream `ggml_mmap` | UNVERIFIED |
| m2 | MTP-Q4 GGUF not corrupted since clean download | no checksum recorded |
| m3 | WSL2 vs bare-metal Linux comparable | not verified |
| m4 | 9 unused HIP→CUDA aliases truly unused | UNVERIFIED across all build configs |
| m5 | Reviewers accept "infra ships before consumer" pattern | UNVERIFIED; team-norm dependent |
| m6 | `qwen36_mtp.cpp` at ~2092 LOC won't block as "too large" | UNVERIFIED |
| m7 | Strict-spec test on CPU stub validates GPU contract | UNVERIFIED; CPU stub doesn't test lossy GPU path |
| m8 | Oracle's Architecture A 1.90-2.10× prediction | model output, no measurement |
| m9 | Drafter SWA window 2048 is right value | not verified against drafter config |

---

## VERIFIED — for the record

- **V1.** `bench_agent_mtp.py:111` sleeps 15s — confirmed; outside timed subprocess.
- **V2.** `bench_agent.py` doesn't sleep between cells — confirmed (no `time.sleep`).
- **V3.** `Qwen35DFlashTarget::verify_tree` exists — confirmed via grep.
- **V4.** `mtp_topk` harness path still routes through chain verify — confirmed by Read of `test_dflash.cpp:859-863`.
- **V5.** `CMakeLists.txt:87-93` defines ONLY `DFLASH_VERIFY_PROFILE` and `DFLASH_MTP_PROFILE` — confirmed.
- **V6.** MTP-Q4 GGUF and DFlash drafter exist at runbook-claimed paths — confirmed.
- **V7.** /tmp KV-sweep logs exist for q8/q8, f16/q8, q8/f16, f16/f16 at d=1 and d=3. Accept at d=3: q8/q8=70.7%, f16/q8=64.1% (counter-intuitive).
- **V8.** D=5/D=7 negative result is structurally consistent with kv-sweep accept-rate decay.
- **V9.** Both bench scripts call `select_rows_for_bucket(..., n_sample, seed=42)` — confirmed by grep.
- **V10.** Tokenizer default is `Qwen/Qwen3.5-27B` in `bench_agent.py:49`, inherited by MTP bench — confirmed.

---

## Recommended next 60 minutes — retire CRITICAL risks first

1. **(10 min) Verify AR baselines match (C1).** Run both `test_dflash --gamma 0` and `test_generate` on same prompt with identical max_ctx + KV. If >5% delta, freeze every speedup claim until recomputed.
2. **(15 min) Find or re-create the Q4 like-vs-like log (C6).** Re-run runbook §2a; confirm 49.51 tok/s median is reproducible.
3. **(5 min) Confirm `verify_tree` is NOT wired into `mtp_topk` (C3).** Grep result will determine Architecture A LOC estimate and L6 spike Phase D plan.
4. **(5 min) Add `DFLASH_GPU_TESTS` CMake option for real (C4).** Dispatch sisyphus to add the option and wrap cuda-gated test blocks.
5. **(15 min) Verify tokenizer correctness (M3) and drafter target-distribution (M1).** If drafter trained on FP16, DFlash baseline is sandbagged.

---

## Reference files

- `dflash/scripts/bench_agent_mtp.py` (lines 111, 197-198, 223, 277, 287)
- `dflash/scripts/bench_agent.py` (lines 49, 190-198, 312-316)
- `dflash/test/test_dflash.cpp` (lines 859-863, 1256-1262)
- `dflash/CMakeLists.txt` (lines 87-93, 534-547, 692-710)
- `dflash/src/qwen35/qwen35_dflash_target.cpp` (and `.h`)
- `/home/peppi/models/qwen3.6-27b-dflash/README.md` (line 22)
- `/tmp/bench_kv_q8_0_q8_0_d3.log` and `/tmp/bench_kv_f16_q8_0_d3.log`
- `thoughts/2026-05-16-mtp-optimization-day.md`
- `thoughts/2026-05-16-ship-plan-handoff.md`
- `thoughts/2026-05-17-harness-runbook.md`
- `/tmp/momus_l6_tdd_spike.md`
- `/tmp/codex_next_steps.md`
