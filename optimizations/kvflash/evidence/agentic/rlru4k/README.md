# R-LRU mode 4 with more recall blocks (k = 6, 10): result

Run on 2026-10-16 on the 3090, under PREREG.md (602f3c87).

**Setup:**
- Binary: prs7 (`/mnt/kimi-k3/build/prs7-cuda/luce_server`, md5 7024e7cdc72ba92095c7c8c20a664fb0) for every phase, logged per phase in `provenance_arms.txt`.
- Lock windows:
  - smokes 13:35:02–13:41:43;
  - OFF 13:47:35–14:38:04;
  - gate (d) 14:49:06–14:49:07;
  - ON2 14:57:52–16:01:20;
  - ON6 16:12:00–17:14:58;
  - ON10 17:21:26–18:24:29.
- Every arm exited with rc 0 and 0 errored rows.

## Verdict: ship k = 2. Neither k = 6 nor k = 10 passes.

| | OFF | ON2 | ON6 | ON10 |
|---|---|---|---|---|
| overall (951) | 713 | **767** | 814 | 831 |
| within 16K (744) | 700 | 700 | 688 | 683 |
| beyond 16K (207) | 13 | 67 | 126 | 148 |

Trajectory-clustered 95% CI, in points:

| | overall | within 16K | beyond 16K |
|---|---|---|---|
| ON2 − OFF | +5.7 [+3.8, +7.5] | +0.0 [−1.5, +1.3] | +26.1 [+19.4, +33.5] |
| ON6 − OFF | +10.6 [+7.6, +13.6] | **−1.6 [−3.8, +0.5]** | +54.6 [+46.8, +62.5] |
| ON10 − OFF | +12.4 [+9.3, +15.4] | **−2.3 [−4.6, +0.0]** | +65.2 [+57.6, +72.7] |
| ON6 − ON2 | +4.9 [+2.2, +7.8] | | |
| ON10 − ON6 | +1.8 [−0.1, +3.7] | | |

- **Rule 1 (primary, ON6 − ON2 > 0):** lower bound +2.2. Pass.
- **Rule 2 (ON10 − ON6 > 0):** lower bound −0.1. Fail.
- **Rule 3 (guard, within-16K lower bound ≥ −2.0):**
  - ON2 −1.5, pass;
  - ON6 −3.8, **fail**;
  - ON10 −4.6, **fail**.
- **Rule 4 (bar 719 with the guard):** ON2 passes. ON6 and ON10 fail through the guard.
- **Ship rule:** k = 10 needs rules 1–4 and k = 6 needs rules 1, 3 and 4. Both fail rule 3, so **k = 2 stays**. That is the rlru4 design, which this ON2 reproduces at 767 on the fixed binary.
- **Rule 5 (predictions):**
  - ON2 near 766: actual 767.
  - ON6 804: actual 814.
  - ON10 820: actual 831.
  - Totals ran 10–11 above prediction, mostly from beyond 16K: 120 predicted vs 126 measured at k = 6, and 134 vs 148 at k = 10.
  - Within 16K landed near the calibrated estimate: 684 vs 688 at k = 6, 686 vs 683 at k = 10.
  - The guard fails anyway, because the calibration placed the within-16K drop against OFF at 4 and 2 probes, while it measured 12 and 17.
- **Rule 6 (speed flag, wall minus scorer > 1.25× ON2's):** not raised (ON6 1.21 s, ON10 1.22 s vs ON2 1.24 s).
- **Rule 7 (caching parity):**
  - median pt 349 on every arm;
  - median cached prefix 24,064 on every arm;
  - disk_hit false on every row (asserted);
  - restore path (`cached_prefix_tokens`, effective prompt) identical to OFF on all 1,106 requests of ON2, ON6 and ON10.

## Smokes (all on prs7, before any arm)

- **(a) noop:** 6 re-prefills, logits max|diff| 0.0 on all 6; the fitting probe skips and equals OFF.
- **(b) k = 2, 6 and 10:** every reselect logs its scorer line and a re-prefill ok. Page-ins are 1.125 MiB each. k = 2 pages in 5–23 chunks per reselect, k = 10 pages in 13–73 (up to 82 MiB).
- **(c) drift ids, OFF vs ON2:** 5 fitting probes, restore paths equal, answers equal.
- **(d) reduced vs full history, OFF:** 5 fitting probes, restore paths equal, answers equal. On prs6 this comparison was red (`../rlru4/drift/cmp_drift.out`). The layout fix (4bd85c49) turns it green.

## Speeds (medians over probe rows; clock caveat below)

| | OFF | ON2 | ON6 | ON10 |
|---|---|---|---|---|
| prefill | 416.7 ms | 1,041.3 ms | 1,046.4 ms | 1,060.8 ms |
| decode | 42.4 tok/s | 42.1 tok/s | 42.8 tok/s | 43.6 tok/s |
| client wall | 0.972 s | 1.662 s | 1.617 s | 1.624 s |
| exact scorer (CPU) | | 441 ms | 431 ms | 431 ms |
| wall minus scorer | | 1.244 s | 1.211 s | 1.221 s |
| page events / page-ins per reselect | | 22 / 11 | 40 / 20 | 58 / 29 |
| H2D per reselect | | 12.4 MiB | 22.5 MiB | 32.6 MiB |

- 827 of 1,106 requests reselect in every ON arm.
- More recall blocks cost about 10 MiB of H2D per 4 blocks and no measurable wall.
- The scorer dominates ON's extra prefill. The fast exact scorer (964be767: 540 → 38 ms with identical picks) is not in prs7.

**OFF before and after (prs6 → prs7, `off_prs6_vs_prs7.out`):**
- prefill 425.7 → 416.7 ms, decode 41.8 → 42.4 tok/s, wall 0.988 → 0.972 s;
- correct 700 → 713;
- 167 answers changed: 30 gains (28 within 16K), 17 losses (16 within).
- These rows cannot separate the two fixes in prs7 (truncate layout, page-out race fence).

## Why k = 6 and k = 10 lose within 16K (post-hoc diagnosis)

`missdump_on{2,6,10}.out`: out-of-window probes, engine picks vs the dump's top k (OFF state, `mass_end`, same block grid and sum).

| within 16K | engine hit, dump hit | engine miss, dump hit | engine miss, dump miss | engine hit, dump miss |
|---|---|---|---|---|
| ON2 | 7 (OFF 7, ON 7) | 7 (OFF 7, ON 3) | 7 (OFF 4, ON 2) | 0 |
| ON6 | 71 (OFF 65, ON 64) | **22 (OFF 21, ON 1)** | 5 (OFF 5, ON 4) | 0 |
| ON10 | 177 (OFF 164, ON 161) | **23 (OFF 21, ON 1)** | 13 (OFF 13, ON 10) | 0 |

- The within-16K drop is in the "engine miss, dump hit" cell. These are probes whose source fell out of the arm's smaller window, which the OFF-state query would have recalled.
- The engine never hits a within-16K source that the dump misses.
- Beyond 16K, the same cell holds 18, 16 and 14 probes.

**Cause** (`../../rlru4k/SPEC_recent_pool.md`):
- `KvFlashPager::truncate(C)` with a prefix that does not fit keeps every current resident and refills only freed blocks.
- The refill is ranked by the previous request's score ledger.
- So a mode-4 request's recalled blocks stay resident through the next request's scoring pass, and that pool lacks the newest history OFF holds.
- OFF never reselects (its log has no reselect line) and stays plain LRU.

**Excluding the 27** (ON6's in-sample diagnosis set, `diag27.json`, from `analyze_v2.py`'s dry run on these rows):
- ON6 − OFF within 16K is +1.3 [−0.3, +2.8];
- ON10 − OFF within 16K is −1.0 [−3.1, +1.1].
- ON6's within-16K loss is all in those 27. ON10 also loses elsewhere: recall classes `diag4_on10.out`, window 9,408 tokens.

**Recall classes** (`diag4_on{2,6,10}.out`):
- A recalled source answers beyond 16K at these rates (whole message / part):
  - 85.7% / 91.2% at k = 2;
  - 92.5% / 92.6% at k = 6;
  - 92.2% / 92.8% at k = 10.
- Beyond-16K misses fall from 146 (k = 2) to 72 (k = 6) to 48 (k = 10).

**Next design:** `v2/PREREG_v2.md`, mode 4 with the recency scoring pool (`LUCE_KVFLASH_SCORE_POOL_RECENT=1`) on prs8.

## Caveats

1. Other agents compiled at nice 19 -j4 during the arms and took the 3090 lock between phases. The lock was held per phase, and timed phases ran with the GPU otherwise idle (`nvidia-smi --query-compute-apps` polled before each arm).
2. This box's monotonic clock (tsc) runs about 7% slow. Absolute tok/s read about 8% high, and the client wall clock is stepped by WSL about 2.3 s every ~32 s. Per-request walls are reported as medians only. Same-box ratios hold. Do not compare these speeds with other machines uncorrected.
3. The arm counts for OFF and ON2 moved against prs6 (700 → 713, 766 → 767). prs7 changes the physical layout (rounding) and fences the page-out race, and these rows cannot attribute the change to either.
4. `missdump.py`, `diag27.py` and the cross-tab are post-hoc and not preregistered. The 27 are in-sample for any design they motivate, which is why PREREG_v2 reads its guard with and without them.

Files:
- **Rules:** `PREREG.md`.
- **Runner:** `run_arms4k.sh`, its log `run_arms4k.log` and `provenance_arms.txt`.
- **Analysis:**
  - `analyze4k.py` (14916453), with `analyze4k.out` (616cc3d8) and `report.json` (e39ad42b);
  - `missdump.py` (025cae42), with `missdump_on{2,6,10}.out`;
  - `diag4_on{2,6,10}.out` (`../rlru4/diag4.py`);
  - `diag27.py` (90d0af38), with `diag27.json` (340f8658);
  - `peek6.py` (ebf2fe40, the interim ON6 bootstrap);
  - `off_prs6_vs_prs7.py`/`.out`.
- **Smokes:** `gate_cd.py`, and `smoke_d/` (the earlier smoke-(d) attempts).
- **Next:** `SPEC_recent_pool.md`, and `v2/` (the next prereg).
