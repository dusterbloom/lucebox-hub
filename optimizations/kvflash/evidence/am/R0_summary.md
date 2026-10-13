# R0 / R0s summary: Attention Matching on Qwen3-4B (paper's own repo, unmodified)

Code: github.com/adamzweiger/compaction at 2020f6d37940fe013e7e7f1ebf409e5d107c5670. Model Qwen/Qwen3-4B rev 1cfa9a7208912126459214e8b04321603b3df60c. H100 80GB on Modal. Everything pre-registered in `PREREG.md` (6f49fa0f) with Amendments 1 (b966c2a9) and 2 (39acd6b8). Provenance for all 143 runs: `manifests_summary.json` (full per-run `manifest.json` on volume `kvflash-am-results`).

| Question | Result | Verdict | File |
|---|---|---|---|
| Does reasoning survive repeated mid-trajectory compaction (Table 9)? | c<=2: 7.5/30 vs baseline-4096 8.25; c<=6: 12.0/30 vs baseline-8192 12.0; paper's 5 numbers all inside our 4-seed CI; paired -0.75 [-1.50, 0.00] and 0.00 [-1.50, +1.50] | **Reproduced** | `R0_aime_table9.md` |
| Does AM beat random keys with the same fit? (115 articles, 2086 q, 0.10) | 65.0% vs 59.6%, +5.37 points, CI [+3.26, +7.60] | **AM better, >= 3 points** | `R0_quality_vsrandom.md` |
| Which online config? (selection, 20 articles) | S1: HighestAttnKeys + repeat queries + per-head budgets + on-policy = 72.3% = full cache; ties S4 (OMP-fast) at 1/24 of the cost | S1 selected | `R0_quality_selection.md` |
| Per-head budgets vs uniform (S1 vs S2) | 72.3% vs 62.2% on the same 357 q | large effect (+10 points) | `R0_quality_selection.md` |
| Stale queries (S5 vs S1, selection split) | 0.00 points, CI [-2.06, +2.23] | within the +-3 margin on selection; held-out not run (cost) | `R0_quality_selection.md` |
| Online S1 vs full cache, held-out (30 articles, 537 q) | -1.68 points, CI [-4.49, +1.12] | within ~3 points; CI does not exclude -4.5 | `R0_quality_heldout.md` |
| **Primary:** online S1 vs ceiling S6, held-out (10 articles, 177 q) | -5.08 points, CI [-7.95, -2.27]; selection split: -3.08 [-6.11, -0.27] | **Not within 3 points**: ceiling better; ceiling costs 29x more compaction time | `R0_quality_heldout.md` |

## What this means for the engine plan
- The paper's headline claims we depend on reproduce with its own code: AIME with repeated compaction and QuALITY near full-cache accuracy at 10% cache with per-head budgets.
- The cheap online configuration (repeat-prefill queries, HighestAttnKeys, per-head budgets, ~29 s per article on H100 for ~6k tokens) matches the full cache within ~2 points; the stale-query test says queries available at fit time are enough. Self-study and OMP buy ~3-5 points more at 30x the fit cost; whether they are worth it is an engine-time budget question, not a quality necessity.
- Per-head budgets matter more (10 points) than query source (2 points, noise) or key selection (OMP-fast = HighestAttnKeys on S4 vs S1).

## Not run (cost gate, Amendments 1 and 2)
Target 0.05; plain OMP in every cell; non-on-policy variants; OMP-fast with context-prefill; uniform budget except S2; 66 of the 72 matrix cells; the truncate control at scale (first pass: 9-19 points worse than AM at n=181); the leave-one-out query ablation; the held-out stale-query arm; held-out beyond 30 (win) and 10 (ceiling) articles; AIME seeds for the first-pass target-0.2 config.

## Deviations (all cost-driven, logged in PREREG)
Ceiling is OMP-fast not OMP; held-out sizes reduced; held-out jobs were submitted minutes before the Amendment 2 commit (no held-out result had been read); target 0.10 only.

## Total cost of this launch (Modal H100 list rate ~$3.95/h, from manifests)
AIME Table 9 + extra baselines $56.4; timing pilot $28.6 (ran 2 h+, over-long); vs-random $33.7; selection $53.7; held-out $16.3 (win $5.7, ceiling $10.6). Total **$188.7** of the $190 cap. Earlier first pass (previous task): ~$43 (target 0.2 AIME, first QuALITY rows, smoke). Grand total on the volume: $231.4 over 143 runs.
