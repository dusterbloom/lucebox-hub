# A3 stage 2: held-out run, early look (lucebox6, 2026-10-16)

This follows PREREG "A3 routed exact fetch", stage 2. W = 2 was taken from the stage-1 report. The router is a_last.
- Run: `a1_launch_a3.sh`, attempt `20261016T011948280315090`, launched at 03:19 CEST on lucebox6 with `HIP_VISIBLE_DEVICES=1`. The run is still going.
- The early look came after the first 20 held-out trajectories (trajectories 7 to 26 in file order). It was taken at 99.9% and logged before trajectory 21 finished at 06:54 CEST (`early_look.log`).

| Probes | Retention | 99.9% interval | Bar | Verdict |
|---|---|---|---|---|
| 304 | 86.5% | [75.47, 93.60] | 75.6% | CONTINUE |

The lower bound misses an early PASS by 0.13 points. As registered, all 54 held-out trajectories run, and section 6 decides at 95%. The rows are committed with the final report.

## Final (all 54 held-out trajectories, finished 16:35 CEST)

Report: `PYTHONPATH=~/r1e/code/am_qwen35 ~/r1e/venv/bin/python a1_a3_report.py proper ~/r1e/out/a3 ~/r1e/out/a1 ~/r1e/data/a1/B0_3090` (a1_a3_report.py 6f4bcf66, a1_a3.py da09ad08, both identical to the repo copies) writes `a3_report.json`. The size requirement (50 trajectories, 600 retained probes) is met.

| Probes | Retention | 95% interval | Bar | Verdict |
|---|---|---|---|---|
| 877 | 83.5% (732) | [80.34, 86.26] | 75.6% | PASS |

Same trajectories, B0 POOL16K: 75.1% (631/840). On the 782 probes both have, A3 676 vs POOL16K 600; A3 alone right on 142, POOL16K alone on 66, exact McNemar p = 1.5e-7.

| Distance back | A3 | POOL16K (same probes) |
|---|---|---|
| 0-4K | 236/243 | 235/243 |
| 4-8K | 98/112 | 106/112 |
| 8-16K | 209/255 (82%) | 249/255 (98%) |
| over 16K | 133/172 (77%) | 10/172 (6%) |

A3 wins beyond 16K and loses 8-16K: with an exact window of only 2 blocks, recent facts depend on the router (a_last recall@8 86%, over 16K 78%; median rank 3 among 27.5 candidates). Mean rows 14,875 of the 16,384 budget; k used 9 or 11 blocks on 619 of 877 probes.
