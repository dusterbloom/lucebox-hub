# E6: repeat-prefill queries at ratio 0.10 (lucebox6, 2026-10-15)

This follows PREREG "E6: repeat-prefill queries at ratio 0.10". It decides nothing and changes no bar.
- Run: `a1_launch_e6.sh`, from 22:51 to 23:37 CEST, on lucebox6 with `HIP_VISIBLE_DEVICES=1`.
- Watchdog: it did not trigger, but came close. Swap grew from 470 to a peak of 1,426 MiB (+956 against the 1,024 MiB limit). Host memory peaked at 95.3 GiB, and the GPU reserved at most 63.3 GiB.
- Change: each block is fitted on the queries of a repeat pass on a clone of the cache, instead of its own prefill queries.
  - The repeat prompt (19 tokens, from `run.log`) is `<|im_start|>user\nRepeat the previous context verbatim.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, followed by the block's 640 tokens.
  - Everything else is the shipping arm: 64 slots per block, the same selection, beta and value fit.
- Data: the same 6 pilot trajectories and 118 reference-correct probes. The reference is the pilot's FULL rows, copied unedited (the md5 matches).
- Code: provenance.txt records the md5s. The report is `../E3E2/e_report.json` (`a1_e_report.py ~/r1e/out/e3e2 ~/r1e/out/e5 ~/r1e/out/a1_pilot ~/r1e/out/e6`).

## Retention on the 118 probes

The intervals are 95% bootstrap, clustered by trajectory.

| Arm | Correct | Retention [95% interval] |
|---|---|---|
| Pilot shipping arm (prefill queries) | 10/118 | 8.5% [4.4, 12.5] |
| **E6** (repeat-prefill queries) | **9/118** | **7.6% [4.2, 11.0]** |
| E5 (160 slots per block) | 32/118 | 27.1% [19.4, 35.7] |
| E3 (the fact's block exact) | 99/118 | 83.9% [77.5, 91.1] |

By distance, E6 against the pilot: 0-4K 2/41 (2/41), 4-8K 3/26 (3/26), 8-16K 2/33 (3/33), over 16K 2/18 (2/18).

By type, E6 against the pilot: line 1/61 (1/61), defname 3/24 (2/24), cmdline_path 2/14 (2/14), cmdline_text 0/15 (2/15), traceback 0/1 (0/1), failed_test 3/3 (3/3).

## Reading

- Fitting on the paper's repeat-prefill queries changes nothing: 9 against 10 of 118.
- So the mismatch between prefill queries and later questions is not what loses the facts. The approximation of the fact's own block is (E3, E5).
- PREREG E6 said a recovery would make a repeat-prefill fit a candidate. There is none, so it is not one.
- By the same reading, a delayed fit on later turns' queries (arXiv 2608.00902) is not pursued.
