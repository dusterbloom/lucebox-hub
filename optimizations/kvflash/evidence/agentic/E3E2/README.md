# E3 and E2: oracle exact source block, and router recall (lucebox6, 2026-10-15)

These follow PREREG "Diagnostics E2, E3, E5 before A2". They decide nothing and change no bar.
- Run: `a1_launch_e3e2.sh`, from 21:33 to 22:15 CEST, on lucebox6 with `HIP_VISIBLE_DEVICES=1`. The watchdog did not trigger.
- Data: the 6 pilot trajectories and the 118 reference-correct probes. The reference is the pilot's FULL-CHK rows.
- Code: provenance.txt records the md5s. The A1 real-model suite passed beforehand, after the E3 test fix in commit 53bf2c54.
- Report: `a1_e_report.py` writes `e_report.json`. The same-probe comparison with POOL16K comes from `lucebox/a1/e3_vs_pool.py`.

## E3: the block holding the fact swapped back to exact

Each probe is answered from the shipping state (`gen_fblk_window`, window 0, ratio 0.10). The only change is that the blocks holding the fact have their compacted entry **replaced** by the exact keys and values.

Retention on the 118 probes, with 95% bootstrap intervals clustered by trajectory:

| Arm | Correct | Retention [95% interval] | Blocks swapped per probe |
|---|---|---|---|
| Pilot shipping arm (no swap) | 10/118 | 8.5% [4.4, 12.5] | 0 |
| **E3**: blocks where the gold string first occurs, located for 118/118 | **99/118** | **83.9% [77.5, 91.1]** | 1.04 |
| E3SPEC: blocks at the start of the source message | 39/118 | 33.1% [26.9, 40.5] | 1.03 |

E3 by distance from the fact to the question:

| Distance (tokens) | Pilot shipping | E3 |
|---|---|---|
| 0-4K | 2/41 | 36/41 |
| 4-8K | 3/26 | 21/26 |
| 8-16K | 3/33 | 27/33 |
| over 16K | 2/18 | 15/18 |

**Same probes as B0 POOL16K** (3090 B0 run, exact pool of 16K rows). 105 of the 118 probes have a POOL16K row:

| Distance | n | E3 | E3SPEC | POOL16K |
|---|---|---|---|---|
| 0-4K | 34 | 29 | 6 | 32 |
| 4-8K | 25 | 21 | 11 | 23 |
| 8-16K | 29 | 26 | 9 | 28 |
| over 16K | 17 | 15 | 10 | 1 |
| all | 105 | 91 (86.7%) | 36 (34.3%) | 84 (80.0%) |

Reading:
- One exact block, the right one, brings recall from 8.5% back to 84%. So the loss lives in the compacted copy of the fact's block, not in the rest of the compacted history.
- Within 16K tokens, POOL16K (83/88) beats E3 (76/88), because E3 still compacts the recent blocks.
- Beyond 16K, only E3 recalls: 15/17 against 1/17.
- E3SPEC shows the block has to be the one holding the fact, not just the start of its message.

## E2: can the question find the block?

The question's queries are captured under the shipping cache. Each sealed block is then ranked; there are 38.5 blocks per cut on average. Recall@k is the share of probes whose source block is in the top k.

| Router | All, recall@1/2/4/8/16 | Over 16K (n=18), recall@1/2/4/8/16 | Median rank (all) |
|---|---|---|---|
| a_last: compacted-slot attention mass, last question token | 18.6 / 36.4 / 60.2 / **85.6** / **95.8**% | 22.2 / 38.9 / 72.2 / **100** / 100% | 4 |
| a_max: the same, max over question tokens | 11.9 / 30.5 / 52.5 / 77.1 / 94.9% | 5.6 / 27.8 / 66.7 / 94.4 / 100% | 4 |
| b_last: pooled-key cosine (the engine's 64-token chunk index, as E1), last token | 0 / 3.4 / 14.4 / 40.7 / 66.1% | 0 / 5.6 / 27.8 / 61.1 / 66.7% | 10 |
| b_max: the same, max over tokens | 1.7 / 7.6 / 22.9 / 39.0 / 73.7% | 0 / 0 / 11.1 / 27.8 / 72.2% | 10 |

Reading:
- The compacted blocks work as an index. The attention mass their 64 slots receive from the question ranks the source block in the top 8 for 86% of probes, and for all 18 facts older than 16K.
- The pooled-key cosine index is far weaker, consistent with E1.

## Ship check (PREREG)

The shipping arm was re-answered from the unswapped state for trajectory 1: 7 of 11 answers have token ids equal to the pilot's SHIP rows. All 4 mismatches are in trajectory 1, and they are longer answers (for example 19 tokens). That points to numerical divergence on re-run, not a different state. E3 and E2 are read with this stated. The gap between E3 and the shipping arm (99 against 10 of 118) is far larger than 4 answers.

## Files

- `E3.jsonl`, `E3SPEC.jsonl`, `e2.jsonl`: the rows.
- `progress.jsonl`, `run.log`, `stdout.log`, `mem.log`, `watchdog.out`, `provenance.txt`
- `ship_check.json` and `ship_check.jsonl`
- `e_report.json` and `e_report.stdout`
