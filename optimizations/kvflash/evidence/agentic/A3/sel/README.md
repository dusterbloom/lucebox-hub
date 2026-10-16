# A3 stage 1: routed exact fetch, window selection (lucebox6, 2026-10-16)

This follows PREREG "A3 routed exact fetch", stage 1, with Amendments 1 and 2. It chooses W for stage 2 and decides nothing else.
- Run: `a1_launch_a3sel.sh` with attempt `20261015T225213302738916`, from 00:52 to 02:20 CEST, on lucebox6 with `HIP_VISIBLE_DEVICES=1`. There was one attempt and no stop.
- Watchdog: it did not trigger. Swap grew from 474 to a peak of 1,426 MiB (+952 against the 1,024 MiB limit), and the GPU reserved at most 88.0 GiB.
- Before launch:
  - Code: commit 0c1b5374, synced to lucebox6. The A3 files are unchanged since the lucebox6 test gate (94 passed) and Codex's final GO.
  - Prediction: the peer session's offline prediction was committed before any row (`../../A3_sim/`, db4dd28a and 442de080).
- Report: `a1_a3_report.py sel ~/r1e/out/a3sel ~/r1e/out/a1_pilot ~/r1e/data/a1/B0_3090` writes `a3_report.json`, with `complete: true` over the fixed universe of 118 probes and reference md5 93a29de5...4047.
- Per-miss listing: `/mnt/kimi-k3/build/a3_diag/a3_miss.py`.

## Selection

| W | Correct | Retention [95% interval] | Mean rows (mean peak) | Peer's prediction |
|---|---|---|---|---|
| **2 (selected)** | **95/118** | **80.5% [67.9, 91.0]** | 14,711 (14,867) | 99-104 |
| 6 | 94/118 | 79.7% [67.0, 90.0] | 14,711 (14,867) | 93-104 |

The rule is the highest retention, so stage 2 runs with W = 2. W=2 lands 4 probes below its predicted range.

## Against B0 POOL16K

POOL16K is the exact 16K pool on the 3090 B0 run. On the same trajectories it gets 88/111 = 79.3% [74.4, 83.5].

On the 105 probes both have, by distance:

| Distance (tokens) | n | A3 W=2 | POOL16K |
|---|---|---|---|
| 0-4K | 34 | 31 | 32 |
| 4-8K | 25 | 25 | 23 |
| 8-16K | 29 | **17** | **28** |
| over 16K | 17 | **15** | **1** |
| all | 105 | 88 | 84 |

The discordant pairs are 17 against 13 (McNemar exact p = 0.58).

## Where W=2 loses its 23 probes

| Source block at answer time | n | Correct |
|---|---|---|
| Exact resident (in P, W or the tail) | 12 | 11 |
| Candidate, all source blocks fetched | 90 | 81 |
| Candidate, some fetched | 2 | 2 |
| Candidate, none fetched (router miss) | 14 | 1 |

- **Router misses, 13 losses.** All 13 are at the longest cuts (43-45 candidates, k = 9), where the source ranked 10th to 29th by a_last. This is why the 8-16K band drops: POOL16K holds those facts exactly in its recent 16K, while A3 has to route to them.
- **Fetched but wrong, 9 losses.** The source block was exact at answer time and the answer was still wrong, 5 of them line probes, several answered with the neighbouring line.
- **Router recall over the candidates** (a_last, W=2, n=106): recall@8 84.0%, recall@16 94.3%, median rank 3. Over 16K (n=18): recall@8 88.9%, recall@16 100%.

By k used (W=2): k = 9 gets 18/36 correct; k = 5, 6, 7, 10 and 11 together get 77/82.

## Cost (HF bf16 driver, not the engine)

- Median routing pass: 0.91 s.
- Median probe wall: 9.8 s, including the swap of about 0.46 GB of exact K/V at k = 11.

## Reading

- With W=2, A3 beats POOL16K beyond 16K (15/17 against 1/17) and loses to it at 8-16K (17/29 against 28/29). Overall that is a wash at equal rows: 88 against 84, p = 0.58.
- The loss at 8-16K is a routing loss at the longest cuts. E9, which runs next, measures whether exact-key attention mass ranks those blocks better.
- Stage 2, the held-out test against the 75.6% bar, waits for E9 and E8 as preregistered.
