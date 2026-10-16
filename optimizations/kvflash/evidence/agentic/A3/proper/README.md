# A3 stage 2: held-out run, early look (lucebox6, 2026-10-16)

This follows PREREG "A3 routed exact fetch", stage 2. W = 2 was taken from the stage-1 report. The router is a_last.
- Run: `a1_launch_a3.sh`, attempt `20261016T011948280315090`, launched at 03:19 CEST on lucebox6 with `HIP_VISIBLE_DEVICES=1`. The run is still going.
- The early look came after the first 20 held-out trajectories (trajectories 7 to 26 in file order). It was taken at 99.9% and logged before trajectory 21 finished at 06:54 CEST (`early_look.log`).

| Probes | Retention | 99.9% interval | Bar | Verdict |
|---|---|---|---|---|
| 304 | 86.5% | [75.47, 93.60] | 75.6% | CONTINUE |

The lower bound misses an early PASS by 0.13 points. As registered, all 54 held-out trajectories run, and section 6 decides at 95%. The rows are committed with the final report.
