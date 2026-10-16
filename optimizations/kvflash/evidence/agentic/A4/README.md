# A4: exact fetch and compact-on-evict on the 6 pilot trajectories (result)

Run on lucebox6, attempt 20261016T144054400376960: started 14:40 UTC, ended 19:22 UTC, stopped none, watchdog empty. Qwen3.8-27B bf16 with PyTorch SDPA. Rules: `../../am/PREREG.md`, the A4 section and its amendment 1 (compact-on-evict arms).

Report: `python3 a1_a4_report.py ~/r1e/out/a4 ~/r1e/out/a1_pilot ~/r1e/data/a1/B0_3090 ~/r1e/out/a3sel ~/r1e/out/e3e2`, rc 0. It wrote `a4_report.json` (md5 bfa44b6d) with an empty stderr. The reference-md5, universe (118 probes) and no-eviction digest checks passed.

## Verdict: compact-on-evict is NOT_POSITIVE. Exact fetch (A4_K4) is the best arm.

Primary rule (A4_CT against A4_K0):
- Beyond the pool, the difference is 0.0 [0.0, 0.0]: 2/18 vs 2/18.
- The within-pool guard fails: A4_K0-only 12 against A4_CT-only 1, a net loss of 11 against a limit of 2.
- Within the pool, the difference is -11.0 points [-21.7, -1.9] (McNemar p = 0.0034).

| Arm | What leaves the 16K window | Fetch | Overall /118 | 0-4K /41 | 4-8K /26 | 8-16K /33 | >=16K /18 |
|---|---|---|---|---|---|---|---|
| A4_K0 | dropped (pure LRU) | 0 | 96 | 40 | 23 | 31 | 2 |
| **A4_K4** | dropped, exact copy archived | 4 exact blocks | **106** | 41 | 23 | 29 | **13** |
| A4_K4P | as K4, plus 10 pinned system and tool blocks | 4 | 86 | 40 | 24 | 12 | 10 |
| A4_CT | compacted to 0.2 (selection, delayed queries) | 0 | 85 | 40 | 24 | 19 | 2 |
| A4_CA | compacted to 0.2 (AM fit, delayed queries) | 0 | 85 | 41 | 26 | 17 | 1 |
| A4_CT0 | compacted to 0.2 (selection, own queries) | 0 | 90 | 41 | 24 | 19 | 6 |
| A4_CTK4 | as CT, plus 4 compacted blocks swapped for exact | 4 | 93 | 41 | 24 | 21 | 7 |

Paired comparisons:
- A4_K4 vs A4_K0: 14 vs 4 discordant, p = 0.031.
- A4_K4P vs A4_K4: 3 vs 23, p = 9e-5. Pinning the prefix costs the 8-16K window.
- A4_CTK4 vs A4_CT: 10 vs 2. A4_CA vs A4_CT: 3 vs 3. A4_CT0 vs A4_CT: 6 vs 1.

Next to the same trajectories:
- B0 POOL16K: 88/111.
- A3_W2 (pilot): 95/118.
- E3 (oracle exact source block): 99/118.

## Why compaction loses here

- **Compaction takes rows away from the exact window.** A compacted block costs 128 rows, so the CT build carries about 20 compacted blocks per cut (max 41). That leaves a mean cut window of 18.4 blocks, against A4_K0's 24.
- **The loss sits in 8-16K.** A4_CT gets 19/33 there against 31/33 for K0, because the facts that fell out of the smaller window are only held in compacted form.
- **The compacted copies answer almost nothing beyond the pool.** CT 2/18, CA 1/18, CT0 6/18. Delayed queries (the paper's recipe) did not beat own queries (CT 2, CT0 6), and the AM fit did not beat selection (CA 1).
- **Compaction plus fetch (A4_CTK4, 93) is worse than fetch alone (A4_K4, 106).** The router over compacted keys finds fewer sources (7/18 beyond the pool, against 13/18), and the window is smaller.

## Caveats

- This is a descriptive pilot: 6 trajectories and 118 probes. The beyond-pool bin has 18 probes in 6 clusters (1, 7, 3, 3, 1, 3).
- POSITIVE would only have justified a held-out run. NOT_POSITIVE means no held-out run, per the rule.
- These numbers come from the bf16 reference implementation on lucebox6, not the engine. The engine's exact fetch is measured separately (`../rlru4k/`).

## Files

- **Rows:**
  - `A4_*.jsonl`, plus `_route` for the fetch arms;
  - per-cut records `A4_cuts_*.jsonl`;
  - done markers `done_*.jsonl`;
  - `FULL.jsonl`, the copied reference, md5 93a29de5.
- **Run:** `manifest.json`, `progress.jsonl`, `summary.json`, `provenance.*.txt`, `run.*.log`, `watchdog.*.out` (empty), `mem.log`, and `stdout.*.log.xz`.
- **Report:** `a4_report.json` and `a4_report.stderr` (empty).
