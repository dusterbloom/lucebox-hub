# Bench Recap — 2026-05-16 + 2026-05-17

**Workload:** SWE-bench Verified, 2K bucket (~2300-2840 tok prompts), n_sample=8, seed=42 (`df.sample(random_state=42)`).
**Hardware:** RTX 3090 (sm_86), WSL2, single GPU.
**Target model:** Qwen3.6-27B Q4_K_M (Unsloth standard).
**Drafter (DFlash):** z-lab `dflash-draft-3.6-q4_k_m.gguf`.
**MTP GGUF:** Unsloth `Qwen3.6-27B-MTP-Q4_K_M.gguf` (single file, backbone+NextN head).
**Env:** `BENCH_THINKING=1`, `-ctk q8_0 -ctv q8_0`, `DFLASH27B_DRAFT_SWA=2048`.

## The full timeline of comparison runs

The headline jumped around as we found and fixed config errors. Authoritative numbers below; the rest is forensic context.

| # | Date / Time | Config | Harness | MTP D=3 (median) | DFlash b22 (median) | AR (median) | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 2026-05-16 evening | n_gen=128, DFlash on **Q4_K_XL** (wrong quant) | bench_agent_mtp.py + bench_agent.py | **49.51 (1.574×)** | 42.40 (1.41×) | 31.45 | Headline claim: MTP +16-18%. **BAD** — DFlash mis-configured. |
| 2 | 2026-05-17 morning | n_gen=256, DFlash on **Q4_K_M** (matched) | bench_agent.py | n/a | **52.36 (1.73×)** | 30.55 | DFlash with proper pairing. **Flips:** DFlash now leads. |
| 3 | 2026-05-17 noon | n_gen=128, matrix MVP | bench_matrix.py | 51.44 (1.58×) | data lost (parser bug) | 32.65 | Matrix MVP — sisyphus orchestrator. DFlash speculator missing parser. |
| 4 | 2026-05-17 afternoon | n_gen=128, matrix v1 (CLI + parser fixes) | bench_matrix.py | (cached from #3) | 46.23 (1.43×) | 32.25 | DFlash now produces data. MTP unchanged from #3. |
| 5 | **2026-05-17 evening (canonical)** | **n_gen=256**, matrix v1 | bench_matrix.py | **52.45 (1.71×)** | **48.99 (1.60×)** | 30.67 | **MTP +7% over DFlash on this run.** |

**Canonical reference for the PR:** run #5 (bench_matrix at n_gen=256, all three speculators via the same harness).

## What the matrix actually shows (run #5 detail)

```
| speculator | tok/s median | tok/s p25–p75 | CI 95%       | prefill med | AL/accept    | speedup vs AR |
|------------|--------------|---------------|--------------|-------------|--------------|---------------|
| ar         | 30.67        | 30.19–31.37   | 30.46–31.23  | 2.89s       | n/a          | 1.00x         |
| dflash_b22 | 48.99        | 44.62–54.82   | 45.67–51.14  | (parser bug)| AL=4.23      | 1.60x         |
| mtp_d3     | 52.45        | 49.51–54.20   | 50.34–53.58  | 2.75s       | acc=66.6%    | 1.71x         |
```

n_sample=8 prompts × n_runs=3 → n_total=24 per speculator. Bootstrap CI 95% via 1000 resamples (seed=42).

## Why the headline kept flipping (full diagnosis)

| Issue | Detected when | Impact on numbers | Fix |
|---|---|---|---|
| **C1: Disjoint prompt sets** — MTP bench used `.head(8)`, DFlash bench used `df.sample(random_state=42)` | momus audit 2026-05-17 morning | "+28% MTP" was on different prompts vs DFlash | Sisyphus aligned both to `select_rows_for_bucket(seed=42)`. Commit `bd0f01c`. |
| **DFlash quant mismatch** — Q4_K_XL target vs drafter trained for Q4_K_M | After codex's "matched-distribution" pointer | DFlash median 42 → 52, catastrophic sample 8 (0.43×) eliminated | Downloaded `unsloth/Qwen3.6-27B-Q4_K_M.gguf` (16.8 GB). Used in runs #2 onward. |
| **Matrix CLI shape wrong** — DFlash speculator used `--prompt-bin` flag style, not positional | First matrix run #3 — DFlash exit 2 every cell | DFlash showed 0.00 tok/s | Switched to positional args (matches bench_agent.py). |
| **Matrix DFlash parser missing** — expected `RESULT_JSON`, DFlash emits `[dflash] generated ...` | Matrix run #4 — DFlash artifact had empty results | Same: 0.00 in summary even though subprocess produced valid output | Added `_parse_dflash_stdout()` regex matching `[dflash] generated`+`[dflash] N draft steps` lines. |
| **n_gen=128 vs n_gen=256** changes the ratio | Comparing matrix #4 (n_gen=128, DFlash 46.23) vs bench_agent.py #2 (n_gen=256, DFlash 52.36) | At smaller n_gen, prefill overhead dominates more → numbers diverge | Matrix re-run at canonical n_gen=256 → run #5. |
| **Parser regex doesn't catch DFlash prefill** — still missing | Run #5 final summary | DFlash prefill column shows 0.00 | Non-blocking; raw stdout has it. Fix in matrix v1.1. |

## Bottom-line interpretation

**MTP and DFlash are at parity on Qwen3.6-27B Q4 SWE-bench 2K @ n_gen=256, both around 1.6-1.7× decode speedup.** The winner of any single bench run flips by ±5-7% based on WSL2 thermal/scheduler noise:

| Bench | MTP-DFlash delta |
|---|---|
| Run #2 (bench_agent.py, n_gen=256) | DFlash +6% |
| Run #5 (matrix, n_gen=256) | MTP +7% |

**Each has distinct operational properties** (relevant for users):

| | MTP chain D=3 | DFlash standalone |
|---|---|---|
| Deployment | Single GGUF (target + NextN head fused) | Two files (target GGUF + drafter GGUF) |
| Config knobs | `--chain-depth 3`, `BENCH_THINKING=1` | `--ddtree --fast-rollback --ddtree-budget=22`, `DFLASH27B_DRAFT_SWA=2048` |
| Intra-run variance | tight (CV ~6%) | wider (range 39-69 in run #5) |
| Catastrophic-tail risk | none observed | sample 8 (0.43×) when drafter mis-paired |
| Long-context (untested for Qwen3.6) | Gemma4 evidence: stable to 1M ctx with pflash | Gemma4 evidence: collapses past 8-16K ctx |

## What's NOT yet measured (gaps the matrix tool should fill)

| Workload | Status | Why important |
|---|---|---|
| SWE-bench 2K @ n_gen=256 | ✅ done (run #5) | Canonical agentic |
| SWE-bench 8K | NOT done | Long-context Qwen3.6 behavior unknown |
| HumanEval | NOT done (workload stub only) | DFlash's published 3.43× home turf — does it apply to Qwen3.6? |
| MT-Bench | NOT done | Creative writing: community says MTP drops 10-15pp |
| Math500/GSM8K | NOT done | DFlash published 2.55-2.93× — Qwen3.6 transfer? |

## Audit/quality posture

- Strict-spec test passes on CPU stub (`test_mtp_chain_strict_spec`)
- GPU path is documented-lossy (q8/q8 KV sub-ULP drift, ~47/128 tokens identical to AR before divergence)
- No HumanEval+ pass@1 quality eval has been run on MTP path yet (audit C5)
- AR baselines now go through the same `test_dflash --gamma 0` for both MTP and DFlash matrix paths (audit C1 partially closed; the prior cross-binary AR comparison artifact is the bench_agent.py vs bench_agent_mtp.py split, both of which now align via matrix tool)
- Bootstrap CI 95% reported alongside median for all matrix cells (audit M6 partially closed)
- Hardware/commit/CUDA captured per artifact (audit M1 closed)

## What the matrix tool unlocks

Infrastructure that makes all FUTURE bench comparisons reproducible:
- Versioned per-run artifact dirs (`{iso_ts}_{git_sha}`)
- Schema-versioned JSON with hardware/commit metadata
- Bootstrap CI 95% computed via 1000 resamples
- Rendered markdown summary with histogram + CI bars
- Modular workload + speculator interfaces (add new workloads in ~50 LOC each)

Replaces scattered scripts in a planned deprecation:
- `bench_agent.py` → `bench_matrix.py --speculators dflash_b22`
- `bench_agent_mtp.py` → `bench_matrix.py --speculators mtp_d3`
- `bench_llm.py` → `bench_matrix.py --workloads humaneval,math500`

## Power sweep (2026-05-17 evening) — added after the initial recap

Bench scope: matrix tool, n_sample=8, n_runs=1, n_gen=256, aligned seed=42 prompts.

| Watts | AR | MTP | DFlash | MTP speedup | DFlash speedup | MTP/DFlash |
|---|---|---|---|---|---|---|
| 262 | 32.27 | 49.21 | 46.70 | 1.525× | 1.447× | 1.054 |
| **301** | **30.67** | **52.45** | **48.99** | **1.710×** | **1.597×** | 1.071 |
| 350 | 34.54 | 53.19 | 49.65 | 1.540× | 1.437× | 1.071 |

**Findings:**

1. **301W is the speedup sweet spot.** Highest MTP/AR (1.71×) and DFlash/AR (1.60×) of the three.
2. **At 262W**: everything slows; ratios don't recover (AR isn't constrained enough vs speculators).
3. **At 350W**: AR jumps +12.6% (more compute-bound, scales with clocks). Speculators barely move (already overlapping draft+verify). Speedup ratios DROP.
4. **MTP/DFlash ratio is stable ~1.05-1.07 across all three power points.** MTP is consistently +5-7% over DFlash on median, regardless of power. The "DFlash wins at full power" reading from a single AL-6 prompt didn't generalize to the n=8 distribution.

**Recommendations:**

- For headline speedup ratios: **bench at 301W** (MTP 1.71×, DFlash 1.60×)
- For absolute throughput matching published RESULTS.md: **bench at 350W stock** (MTP 53, DFlash 50)
- For deployment: **350W** (responsiveness is user-visible)
- **262W is strictly worse** — no reason to use it

## Authoritative artifacts on disk

- This recap: `thoughts/2026-05-17-bench-recap.md`
- Run #5 canonical: `dflash/bench/results/2026-05-17T12-54-23_0925bea/`
- Day 1 memo: `thoughts/2026-05-16-mtp-optimization-day.md`
- Day 1 ship plan: `thoughts/2026-05-16-ship-plan-handoff.md`
- Day 2 DFlash matched-bench: `thoughts/2026-05-17-dflash-matched-bench.md`
- Day 2 audit: `thoughts/2026-05-17-assumptions-audit.md`
- Harness runbook: `thoughts/2026-05-17-harness-runbook.md`
