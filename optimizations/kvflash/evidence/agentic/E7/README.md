# E7: system and tool prefix exact (lucebox6, 2026-10-15)

This follows PREREG "E7: system and tool prefix exact". It decides nothing and changes no bar.
- Run: `a1_launch_e7.sh`, from 23:55 to 00:50 CEST, on lucebox6 with `HIP_VISIBLE_DEVICES=1`.
- Watchdog: it did not trigger, but came close. Swap grew from 470 to a peak of 1,452 MiB (+982 against the 1,024 MiB limit). Host memory peaked at 94.4 GiB, and the GPU reserved at most 78.5 GiB.
- Change: the blocks that overlap the rendered system message stay exact, with beta 0 and no fit. The system message includes the tool schema. Every other sealed block is compacted at 0.10, as in the shipping arm.
  - In all 6 trajectories the system message ends at token 5920, so 10 blocks (6,400 tokens) stay exact.
- Data: the same 6 pilot trajectories and 118 reference-correct probes. The reference is the pilot's FULL rows, copied unedited (the md5 matches).
- Code: provenance.txt records the md5s. The report is `../E3E2/e_report.json` (`a1_e_report.py ~/r1e/out/e3e2 ~/r1e/out/e5 ~/r1e/out/a1_pilot ~/r1e/out/e6 ~/r1e/out/e7`).

## Retention on the 118 probes

The intervals are 95% bootstrap, clustered by trajectory.

| Arm | Correct | Retention [95% interval] |
|---|---|---|
| Pilot shipping arm | 10/118 | 8.5% [4.4, 12.5] |
| **E7** (system and tool prefix exact) | **9/118** | **7.6% [3.6, 12.5]** |
| E6 (repeat-prefill queries) | 9/118 | 7.6% [4.2, 11.0] |
| E5 (160 slots per block) | 32/118 | 27.1% [19.4, 35.7] |
| E3 (the fact's block exact) | 99/118 | 83.9% [77.5, 91.1] |

By distance, E7 against the pilot: 0-4K 2/41 (2/41), 4-8K 2/26 (3/26), 8-16K 3/33 (3/33), over 16K 2/18 (2/18).

By type, E7 against the pilot: line 1/61 (1/61), defname 2/24 (2/24), cmdline_path 2/14 (2/14), cmdline_text 1/15 (2/15), traceback 0/1 (0/1), failed_test 3/3 (3/3).

## Reading

- Keeping the system and tool prefix exact changes nothing: 9 against 10 of 118.
- The control tokens of the prefix are not what breaks recall, so the matched-capacity control that PREREG E7 held back for a gain is not run.
- With E5 and E6, this leaves one cause: the compacted copy of the fact's own block. Only making that block exact (E3) recovers recall. That is what A3 tests.
