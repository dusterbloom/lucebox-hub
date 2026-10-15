# E5: shipping arm at ratio 0.25 (lucebox6, 2026-10-15)

This follows PREREG "E5: shipping arm at ratio 0.25". It decides nothing and changes no bar.
- Run: `a1_launch_e5.sh`, from 22:15 to 22:48 CEST, on lucebox6 with `HIP_VISIBLE_DEVICES=1`. The watchdog did not trigger: host memory peaked at 99.8 GiB and swap grew from 468 to 857 MiB. The GPU reserved at most 72.1 GiB.
- Setup: the A1 shipping arm with `--ratio 0.25`, so each 640-token block keeps 160 slots instead of 64.
- Data: the same 6 pilot trajectories and 118 reference-correct probes. The reference is the pilot's FULL rows, copied unedited (the md5 matches).
- Code: provenance.txt records the md5s. The report is `../E3E2/e_report.json` (`a1_e_report.py ~/r1e/out/e3e2 ~/r1e/out/e5 ~/r1e/out/a1_pilot`).

## Retention on the 118 probes

The intervals are 95% bootstrap, clustered by trajectory.

| Arm | Slots per block | Correct | Retention [95% interval] |
|---|---|---|---|
| Pilot shipping arm | 64 | 10/118 | 8.5% [4.4, 12.5] |
| **E5** | 160 | **32/118** | **27.1% [19.4, 35.7]** |
| E3: the fact's block exact, the rest at 64 | 64, and 640 for about 1.04 blocks | 99/118 | 83.9% [77.5, 91.1] |

By distance from the fact to the question:

| Distance (tokens) | Pilot | E5 | E3 |
|---|---|---|---|
| 0-4K | 2/41 | 7/41 | 36/41 |
| 4-8K | 3/26 | 10/26 | 21/26 |
| 8-16K | 3/33 | 8/33 | 27/33 |
| over 16K | 2/18 | 7/18 | 15/18 |

By probe type:

| Type | Pilot | E5 | E3 |
|---|---|---|---|
| line | 1/61 | 5/61 | 50/61 |
| defname | 2/24 | 13/24 | 17/24 |
| cmdline_path | 2/14 | 6/14 | 13/14 |
| cmdline_text | 2/15 | 4/15 | 15/15 |
| traceback | 0/1 | 1/1 | 1/1 |
| failed_test | 3/3 | 3/3 | 3/3 |

## Reading

- 2.5 times the slots in every block recovers 22 of the 108 probes the pilot missed. One exact block, the right one, recovers 89 of them.
- Exact line content almost never survives compaction: line probes go from 1/61 to only 5/61 at 160 slots.
- So the loss comes from approximating the fact's own block. Spending more slots uniformly is a poor fix; fetching that block exact (A3) is the targeted one.
