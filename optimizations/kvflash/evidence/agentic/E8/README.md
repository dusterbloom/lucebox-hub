# E8: selection-only twin of the shipping arm (lucebox6, 2026-10-16)

This follows PREREG "Diagnostic E8: selection-only twin of the shipping arm". It decides nothing and changes no bar.
- Run: `a1_launch_e8.sh`, from 02:46 to 03:18 CEST, on lucebox6 with `HIP_VISIBLE_DEVICES=1`.
- Watchdog: it did not trigger. Swap grew from 472 to a peak of 969 MiB (+497). Host memory peaked at 105.0 GiB, and the GPU reserved at most 60.6 GiB.
- Change: each sealed block keeps the 64 original key and value rows that the shipping fit selects, copied verbatim. The bias is 0 and no value is fitted (`a1_core.select_only`, `a1_run.py --select-only`). Everything else is the shipping arm: 640-token blocks, ratio 0.10, open tail exact, DeltaNet state carried.
- Data: the same 6 pilot trajectories and 118 reference-correct probes. The reference is the pilot's FULL rows, copied unedited (the md5 matches).
- Code: commit 0c1b5374; provenance.txt records the md5s. The report is the `E8_select_only` section of `../E3E2/e_report.json` (`a1_e_report.py ~/r1e/out/e3e2 ~/r1e/out/e5 ~/r1e/out/a1_pilot ~/r1e/out/e6 ~/r1e/out/e7 ~/r1e/out/e8 ~/r1e/out/e9`).

## Retention on the 118 probes

The intervals are 95% bootstrap, clustered by trajectory.

| Arm | Correct | Retention [95% interval] |
|---|---|---|
| Pilot shipping arm (fit, beta) | 10/118 | 8.5% [4.4, 12.5] |
| **E8** (selection only, no fit) | **11/118** | **9.3% [5.6, 15.2]** |
| E5 (160 slots per block) | 32/118 | 27.1% [19.4, 35.7] |
| E3 (the fact's block exact) | 99/118 | 83.9% [77.5, 91.1] |

By distance, E8 against the pilot: 0-4K 2/41 (2/41), 4-8K 2/26 (3/26), 8-16K 3/33 (3/33), over 16K 4/18 (2/18).

By type: line 0/61, defname 2/24, cmdline_path 6/14, cmdline_text 0/15, traceback 0/1, failed_test 3/3.

## Reading

- Under the preregistered reading, the fit earns nothing measurable over plain key selection on these traces: 11 against 10 of 118, with overlapping intervals. Selection does not help either.
- On this evidence alone, M5 is neither removed nor kept. Together with E9, where exact keys rank blocks at least as well as the fitted slots, nothing on the agentic traces shows the beta and value fit paying for itself.
- E5, E6, E7 and E8 all stay between 7.6% and 27.1%. Every compacted form of the fact's own block tried so far loses most facts: 64 fitted rows, 160 fitted rows, and 64 selected rows. Only the exact block (E3) recovers it, and that is what A3 stage 2 (launched 03:19 CEST) tests on held-out trajectories.
