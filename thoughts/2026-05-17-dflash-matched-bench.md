# DFlash Matched-Distribution Bench — 2026-05-17

**Status:** Correction to the 2026-05-16 memo's "MTP +16-18% over DFlash" claim.
**Hardware:** RTX 3090 (sm_86), WSL2, single GPU. **Branch:** `spike/l6-hybrid`.
**Bench harness:** `dflash/scripts/bench_agent.py --bucket 2k --n-sample 8 --budget 22`.

## TL;DR

**MTP +16-18% over DFlash was an artifact of DFlash mis-configuration.** When DFlash is run with its canonical setup (drafter matched to target distribution per z-lab README), DFlash median jumps from 42.10 → **52.36 tok/s** and edges MTP by ~+6%. The two paths are effectively at parity on Qwen3.6-27B Q4 SWE-bench (both around 1.7× decode), each with distinct operational properties.

## What changed in the setup

| | Prior bench (2026-05-16) | This bench (2026-05-17) |
|---|---|---|
| Target | `Qwen3.6-27B-UD-Q4_K_XL.gguf` (Unsloth dynamic Q4) | `Qwen3.6-27B-Q4_K_M.gguf` (Unsloth standard Q4_K_M) |
| Drafter | `dflash-draft-3.6-q4_k_m.gguf` | `dflash-draft-3.6-q4_k_m.gguf` (same) |
| `DFLASH27B_DRAFT_SWA` | unset → 2048 (sweep showed no diff) | 2048 |
| `--ddtree --fast-rollback` | yes (hardcoded in bench_agent.py) | yes |
| `--ddtree-budget` | 64 (then 22 — both same numbers) | 22 |

**The single load-bearing fix: target quant family.** z-lab README explicitly says the drafter must be paired with `Qwen/Qwen3.6-27B`. Unsloth's "Q4_K_M" is the canonical Q4 conversion of that base; "Q4_K_XL" is a different (dynamic) scheme. Distribution mismatch crushed accept length (AL) on multiple prompts and produced the catastrophic 0.43× tail sample.

## Per-sample results (n=8, aligned via `seed=42`)

| Sample | n_prompt | DFlash old (Q4_K_XL) tok/s | AL old | **DFlash new (Q4_K_M) tok/s** | **AL new** | Δ tok/s |
|---|---|---|---|---|---|---|
| 1 | 2556 | 76.64 | 6.74 | 61.29 | 5.54 | −15.35 |
| 2 | 2564 | 41.26 | 3.71 | **55.20** | **5.02** | **+13.94** |
| 3 | 2705 | 40.19 | 3.66 | 40.42 | 3.71 | +0.23 |
| 4 | 2330 | 78.97 | 6.92 | 68.51 | 5.82 | −10.46 |
| 5 | 2452 | 41.50 | 4.00 | 31.37 | 3.00 | −10.13 |
| 6 | 2549 | 42.70 | 3.82 | 49.51 | 4.27 | +6.81 |
| 7 | 2840 | 48.37 | 4.41 | 43.43 | 4.06 | −4.94 |
| 8 | 2457 | **12.97** (catastrophic 0.43×) | **1.33** | **61.15** (2.05×) | **5.29** | **+48.18** |
| **Mean** | 2557 | 47.83 | 4.32 | **51.36** | **4.59** | **+3.53** |
| **Median** | | 42.10 | — | **52.36** | — | **+10.26** |
| **Median decSp** | | **1.41×** | | **1.73×** | | |

**Sample 8 is the smoking gun**: 12.97 → 61.15 tok/s with the same drafter, same code, only the target quant family changed. The mismatch had been collapsing accept rate on adversarial prompts; matching the quant family restored it.

## Mixed signal — the matched-quant gains are not uniform

Half the samples got SLOWER (1, 4, 5, 7), half got FASTER (2, 6, 8) with the matched quant. AL pattern is symmetric: easy prompts (AL > 6) dropped, hard prompts (AL < 4) gained.

Hypothesis: `Q4_K_XL` (Unsloth dynamic) preserves "common" tokens more precisely (importance-weighted), helping DFlash on repetitive code patterns where AL was already high. `Q4_K_M` (uniform) is more balanced, helping on irregular prompts where the drafter previously diverged. **Median + mean both prefer Q4_K_M.**

## Updated MTP vs DFlash comparison

| Config | Median tok/s | Median decSp | Mean tok/s | Mean decSp | Variance |
|---|---|---|---|---|---|
| AR baseline (Q4_K_M) | 30.24 | 1.00× | 30.24 | 1.00× | tight |
| MTP Q4 chain D=3 (memo headline) | 49.51 | **1.574×** | 49.02 | 1.560× | CV 7.5% |
| ~~DFlash Q4 mis-matched (memo headline)~~ | ~~42.10~~ | ~~1.41×~~ | ~~47.83~~ | ~~1.58×~~ | catastrophic 0.43× tail |
| **DFlash Q4 properly matched** | **52.36** | **1.73×** | **51.36** | **1.70×** | tighter, no catastrophic tail |

**Net: DFlash properly matched edges MTP by ~+6% median, ~+5% mean, with no catastrophic outlier.** The two paths are within noise on this workload (~1.7× decode each).

## Updated ship narrative

**Before:** "MTP chain D=3 beats DFlash by +18% on like-for-like."
**After:** "MTP chain D=3 and DFlash standalone (properly configured) are at parity on Qwen3.6-27B Q4 SWE-bench, both around 1.7× decode speedup. Each has distinct properties:
- **MTP**: single GGUF (target + NextN head fused), no separate drafter, no SWA env tuning required, slightly tighter intra-prompt variance
- **DFlash**: separate drafter GGUF (requires Q4_K_M target match + `DFLASH27B_DRAFT_SWA=2048`), slightly faster on this workload, wider prompt-to-prompt spread but no catastrophic outliers when properly configured"

## Implications for the PR

The MTP foundation work is still valuable — it's a brand-new speculative path that hits 1.574× decode on real SWE-bench workloads with rock-stable variance. **The perf-comparison angle changes from "MTP wins" to "MTP achieves parity with the best-tuned DFlash setup, with simpler deployment (one GGUF)."**

PR3's body (verify-side optimizations) must drop the "+18% over DFlash" framing and substitute the parity claim above.

## What this validates from earlier audits

- **Momus audit M1** ("drafter trained for BF16 target; Q4_K_XL pairing is a distribution mismatch") — **VERIFIED**. Was the load-bearing failure mode. The catastrophic sample 8 collapse was the smoking gun.
- **Codex pointer "use one target family for fair testing"** — **VERIFIED**. I refused this earlier on theoretical grounds. Was wrong to refuse.

## Files touched today (uncommitted to feature branch; spike commits on spike/l6-hybrid)

- New: `thoughts/2026-05-17-dflash-matched-bench.md` (this file)
- Edit: `thoughts/2026-05-16-mtp-optimization-day.md` — correction notice at top + amended cross-quant claim

## Next steps

1. Commit memo amendment + this doc on `feature/mtp-foundation-v2`
2. Update ship-plan handoff (`2026-05-16-ship-plan-handoff.md`) to reflect parity framing
3. PR3 body must say "parity with DFlash properly configured", not "+18%"
4. Optional next experiment: lift `Qwen3.6-27B-Q4_K_M.gguf` into `dflash/models/` symlink and update the runbook to use it as the canonical DFlash target
