# R1(e): segment-local vs global F queries, uniform budgets (lucebox6 iGPU)

Preregistered in `PREREG.md`, section "R1(e)", before any run (commit 2c1b66d8). Run 2026-10-14 on lucebox6 GPU1 (Radeon 8060S, gfx1151), torch 2.9.1+rocm7.13 nightly, transformers 5.19.0, Qwen/Qwen3.8-27B bf16 at revision 1d4bf0f2, pinned adamzweiger/compaction @ 2020f6d3 + patches 0001-0003 + overlay. Driver `lucebox/r1e/r1e_run.py`, report `lucebox/r1e/r1e_report.py`. No Modal spend.

Arms, all at total budget 0.10 of the article tokens and the same algorithm (cell FUNI, every head the same slots):
- ORIG: no compaction.
- F-UNI-GLOBAL: the whole article compacted once from whole-article queries (R1b `Session.compact("FUNI")`).
- F-UNI-LOCAL: each sealed 640-token block compacted to 64 slots from its own prefill queries, behind the already-compacted prefix (`r2_core.gen_fblk`, k=0); the tail under 640 tokens stays exact.

## Held-out (decides): 30 articles, 537 questions

| Arm | Correct | Accuracy | Drop vs ORIG, points [95% CI] |
|---|---|---|---|
| ORIG | 461 | 85.8% |  |
| F-UNI-GLOBAL | 434 | 80.8% | +5.03 [+2.60, +7.69] |
| F-UNI-LOCAL | 438 | 81.6% | +4.28 [+2.21, +6.37] |

LOCAL - GLOBAL: **+0.74 points [-1.87, +3.36]**, McNemar only-LOCAL 23 / only-GLOBAL 19, p = 0.644.

**Verdict: the lower bound -1.87 is above -3, so M8 ships segment-local F.** Global queries or a query reservoir are not needed.

## Selection (sanity only): 20 articles, 357 questions

ORIG 319, F-UNI-GLOBAL 308, F-UNI-LOCAL 309; LOCAL - GLOBAL +0.28 [-1.65, +2.31].

## Notes

- Statistics: unit question, cluster article, paired cluster bootstrap with 10,000 resamples, percentile 95% (`r1b_stats.py`).
- Cross-check with R1b (Modal, CUDA torch 2.8): F-UNI held-out at 0.10 was 435/537; GLOBAL here is 434/537 on ROCm kernels.
- Deviation, known before the held-out run: LOCAL compacts every token of a sealed block, including system and chat-template tokens, while GLOBAL keeps the non-article tokens exact. This follows `gen_fblk` as preregistered; the total budget differs by a few dozen slots.- Article 0 of the selection split was the smoke article and its three accuracies were seen before the selection run.
- Speed (informational): about 100-240 s per article for all three arms together.
- Rows: `R1e/sel/*.jsonl`, `R1e/held/*.jsonl` (one row per question, field `correct`).
