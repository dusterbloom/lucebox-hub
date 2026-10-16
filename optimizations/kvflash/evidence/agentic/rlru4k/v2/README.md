# rlru4k-v2: mode 4 with the recency scoring pool: result

Run on 2026-10-16 on the 3090, under `PREREG_v2.md`:
- frozen at 0370fbb6;
- binary amendment 584a9bc2, file md5 6cb6bb2f.

**Binary and branch:** prs8 (`/mnt/kimi-k3/build/prs8-cuda/luce_server`, md5 a116ca38bcdb6326e2a486aceabda64b) for every phase, on engine `kvflash/2.0r-prefill-reselect` db4becd1. It is logged per phase in `provenance_v2.txt`.

**Lock windows:**
- shadow 18:45:00–18:48:36;
- smokes 18:48:36–18:54:52;
- OFF 18:54:52–19:44:26;
- gate (d) 19:44:26;
- ON2 19:44:26–20:40:55;
- ON6 20:40:55–21:39:22;
- ON10 21:39:22–22:39:52.

Every phase exited with rc 0 and 0 errored rows. kvflash-am-a0 held all builds on this machine from 18:45 until the run ended. The frozen scripts' md5s are unchanged (`run_v2.sh` 8a895447, `resident_cmp.py` a4b7be71, `analyze_v2.py` b328d231).

## Verdict: ship k = 6 with `LUCE_KVFLASH_SCORE_POOL_RECENT=1`: 847/951 (89.1%)

The shipped stack is `QK16KND` plus:
- `LUCE_KVFLASH_PREFILL_RESELECT=4`;
- `LUCE_KVFLASH_RESELECT_SCORER=exact`;
- `LUCE_KVFLASH_RECALL_BLOCKS=6`;
- `LUCE_KVFLASH_SCORE_POOL_RECENT=1`.

It replaces the rlru4 design (k = 2, flag off, 766/767) and clears the B0 POOL16K bar of 719/951 (75.6%) by 128 probes.

| | OFF | ON2 | ON6 | ON10 |
|---|---|---|---|---|
| overall (951) | 713 | 793 | **847** | 853 |
| within 16K (744) | 700 | 709 | 709 | 702 |
| within 16K, excluding the 27 (717) | 674 | 686 | 685 | 676 |
| beyond 16K (207) | 13 | 84 | 138 | 151 |

Trajectory-clustered 95% CI, in points (`a1_core.boot_ci`, 10,000 resamples, seed 0):

| | overall | within 16K | within, excluding the 27 | beyond 16K |
|---|---|---|---|---|
| ON2 − OFF | +8.4 [+6.5, +10.4] | +1.2 [−0.3, +2.7] | +1.7 [+0.4, +3.0] | +34.3 [+27.8, +41.2] |
| ON6 − OFF | +14.1 [+11.6, +16.6] | +1.2 [−0.4, +2.8] | +1.5 [+0.0, +3.1] | +60.4 [+52.1, +68.9] |
| ON10 − OFF | +14.7 [+11.8, +17.5] | +0.3 [−1.7, +2.1] | +0.3 [−1.7, +2.1] | +66.7 [+58.5, +74.6] |
| ON6 − ON2 | +5.7 [+3.9, +7.5] | | | |
| ON10 − ON6 | +0.6 [−1.1, +2.3] | | | |

- **Structural check (gates validity):** each arm's resident set equals OFF's on all 1,106 requests, with 0 mismatches, for ON2, ON6 and ON10. Every arm is valid.
- **Rule 1 (ON6 − ON2 > 0):** lower bound +3.9. Pass.
- **Rule 2 (ON10 − ON6 > 0):** lower bound −1.1. Fail.
- **Rule 3 (guard, read twice, both lower bounds ≥ −2.0):**
  - ON2 −0.3 / +0.4, pass;
  - ON6 −0.4 / +0.0, pass;
  - ON10 −1.7 / −1.7, pass.
- **Rule 4 (bar 719 with the guard):** all three arms pass.
- **Ship rule:**
  - k = 10 needs rule 2, and fails it.
  - k = 6 needs rules 1, 3 and 4, and holds all three. **k = 6 ships.**
  - `analyze_v2.py` prints `"ship": "k=6 (flag on)"`.
- **Rule 5 (predictions, `predict_v2.out`, from prs7 rows only), predicted against actual (within / beyond / total):**

  | Arm | Predicted | Actual |
  |---|---|---|
  | ON2 | 704 / 73 / 777 | 709 / 84 / **793** |
  | ON6 | 707 / 136 / 843 | 709 / 138 / **847** |
  | ON10 | 703 / 155 / 858 | 702 / 151 / **853** |

  - ON6 and ON10 land within 5 of the prediction in every cell.
  - ON2 beats its prediction by 16, 11 of them beyond 16K. Its engine hits 11 beyond-16K sources the dump misses (4 on prs7), and the method had moved that cell toward the miss rate (mechanism table below).
- **Rule 6 (speed flag, wall minus scorer > 1.25× ON2's 1.218 s):** not raised. ON6 is 1.338 s and ON10 is 1.416 s.
- **Rule 7 (caching parity):**
  - median pt 349 on every arm;
  - median cached prefix 24,064 on every arm;
  - `disk_hit` false on every row (asserted);
  - restore paths covered by the structural check.

## What changed against prs7 (mechanism)

`analyze_v2.py` checks whether the out-of-window sources are hit by the engine's picks, and by the dump's top k (OFF state, `mass_end`, same block grid and sum).

| | within: engine miss, dump hit | beyond: engine miss, dump hit | beyond: engine hit, dump miss |
|---|---|---|---|
| ON2 prs7 → v2 | 7 → 1 | 18 → 3 | 4 → 11 |
| ON6 prs7 → v2 | **22 → 1** | 16 → 1 | 5 → 5 |
| ON10 prs7 → v2 | **23 → 0** | 14 → 5 | 6 → 3 |

- As predicted, the within-16K "engine miss, dump hit" cell falls to near 0. With the scoring pool reset to plain LRU, the router's picks agree with the OFF-state query, and the engine never misses a within-16K source the dump holds.
- **The stratum the window gives up** (post hoc, `stratum_v2.out`, same definition as `../analyze4k.py`): within-16K probes whose source lies outside the arm's own window.

  | Arm | n | OFF | prs7 arm | v2 arm |
  |---|---|---|---|---|
  | ON2 | 21 | 18 | 12 | 15 |
  | ON6 | 98 | 91 | 69 | **92** |
  | ON10 | 213 | 198 | 172 | 195 |

  At k = 6 the arm now answers that stratum as well as OFF does. At k = 10 it trails OFF by 3.
- **Recall classes** (`diag4_on{2,6,10}.out`, post hoc):
  - A recalled source answers beyond 16K at these rates (whole message / part):
    - 92.6% / 86.0% at k = 2;
    - 96.2% / 87.5% at k = 6;
    - 92.4% / 89.0% at k = 10.
  - Beyond-16K misses fall from 124 (k = 2) to 57 (k = 6) to 42 (k = 10).
  - Against OFF within 16K, ON6 gains +8 in window and +4 on wholly recalled sources. ON10 gains +5 and −1. Its window is shorter: on the 33K smoke request the logged tail is 151 chunks, against ON6's 191.

## Smokes and shadow (all on prs8, before any arm)

- **Shadow, flag off, prs8 against prs7:**
  - k = 2, 6 and 10 on `smoke_ids4`: 8 requests and 6 reselects each.
  - k = 2 on the drift ids: 13 requests and 7 reselects.
  - 0 pick mismatches and 0 answer diffs on every run. So rlru4k's flag-off results carry over to prs8.
- **(a) noop:** 6 re-prefills, logits max|diff| 0.0 on all 6; the fitting probe skips and equals OFF. The flip rate against OFF (batch boundary at R4) is 1/5 probes, reported, not gated.
- **(b) k = 2, 6 and 10:** each reselect logs its scorer line and a re-prefill ok, and every pooled request logs one recency line. On the 33K smoke request, the recency step pages in 21–22 chunks at k = 2 and 90 at k = 10 (101 MiB).
- **(c) drift ids, OFF vs ON2:** 5 fitting probes, restore paths equal, answers equal.
- **(e) structural:** resident sets equal OFF's for noop, k = 2, 6 and 10, and the drift pair, with 0 mismatches.
- **(d) reduced vs full history, OFF:** 5 fitting probes, restore paths equal, answers equal.

**OFF, prs7 → prs8:**
- 713 → 713, with **0 of 951 answers changed**;
- prefill 416.7 → 398.9 ms, decode 42.4 → 44.1 tok/s, wall 0.972 → 0.918 s.

## Speeds (medians over probe rows; clock caveat below)

| | OFF | ON2 | ON6 | ON10 |
|---|---|---|---|---|
| prefill, prs7 → prs8 | 416.7 → 398.9 ms | 1,041.3 → 684.1 ms | 1,046.4 → 819.3 ms | 1,060.8 → 913.6 ms |
| decode, prs7 → prs8 | 42.4 → 44.1 tok/s | 42.1 → 44.0 tok/s | 42.8 → 44.2 tok/s | 43.6 → 43.6 tok/s |
| client wall, prs7 → prs8 | 0.972 → 0.918 s | 1.662 → 1.250 s | 1.617 → 1.370 s | 1.624 → 1.448 s |
| exact scorer (CPU), prs7 → prs8 | | 441 → 33.9 ms | 431 → 34.7 ms | 431 → 32.5 ms |
| wall minus scorer, prs7 → prs8 | | 1.244 → 1.218 s | 1.211 → 1.338 s | 1.221 → 1.416 s |
| page events / page-ins per reselect, prs8 | | 46 / 23 | 108 / 54 | 154 / 77 |
| H2D per reselect, prs7 → prs8 | | 12.4 → 25.9 MiB | 22.5 → 60.8 MiB | 32.6 → 86.6 MiB |
| recency step H2D, median / max (1,106 steps) | | 23.6 / 24.8 MiB | 46.1 / 69.8 MiB | 59.6 / 114.8 MiB |

- 827 of 1,106 requests reselect in every ON arm.
- The prs7 and prs8 ON arms differ in two ways: the binary (the fast scorer) and the flag.
- The fast scorer takes about 400 ms off every reselect.
- The flag roughly doubles to triples H2D per reselect, and adds the recency step on every pooled request:
  - The recency step pages out the previous request's recalled blocks, and the mode-4 step pages its own picks back in.
  - At k = 6 that is about 107 MiB of H2D per request (46 MiB + 61 MiB), against 22.5 MiB on prs7.
  - Wall minus scorer rises 0.12 s (10%) from ON2 to ON6.
- **The shipped k = 6 costs, against OFF on prs8:**
  - +420 ms prefill (398.9 → 819.3 ms);
  - decode unchanged (44.1 → 44.2 tok/s);
  - +0.45 s median wall (0.918 → 1.370 s).
  - In exchange: +134 probes (713 → 847), +125 of them beyond 16K.

## Caveats

1. This box's monotonic clock (tsc) runs about 7% slow, so absolute tok/s read about 8% high. The client wall clock is stepped by WSL about 2.3 s every ~32 s, so per-request walls are reported as medians only. Same-box ratios hold. Do not compare these speeds with other machines uncorrected.
2. **In-sample design.**
   - The recency pool fix and the k grid were chosen on these same 60 trajectories. The 27 diagnosis probes are excluded in the second guard read, which holds (ON6 +1.5 [+0.0, +3.1]).
   - Out-of-sample confirmation needs fresh agentic trajectories.
3. The recency step pages out blocks that mode 4 may re-pick in the same request. A pager that skips that round trip would cut the 46 MiB median at k = 6. It is not built and not measured.
4. The engine branch `kvflash/2.0r-prefill-reselect` carries Co-Authored-By trailers on 7 commits (910e070d … a0b503b7). No push or PR is allowed until kvflash-am-a0's rewrite lands. Its hash map will be committed next to this evidence, and it maps the db4becd1 cited here.
5. `stratum_v2.py`, `diag4_on*.out` and `peek_on2.py` are post hoc and not preregistered. PREREG_v2 lists the stratum and the recall classes as reported, not gated.

## Files

- **Rules:** `PREREG_v2.md`.
- **Runner:** `run_v2.sh`, its log `run_v2.log` (md5 0815840d) and `provenance_v2.txt` (6e9ff366).
- **Checks:** `resident_cmp.py`, `shadow_cmp.py`.
- **Analysis:**
  - `analyze_v2.py`, with `analyze_v2.out` (9e975532) and `report_v2.json` (9424c58b);
  - `predict_v2.py`, with `predict_v2.out`.
- **Post hoc:**
  - `stratum_v2.py` (acb8ab9f), with `stratum_v2.out` (1d963d2b);
  - `diag4_on2.out` (1315bc77), `diag4_on6.out` (998fcda7) and `diag4_on10.out` (6fc2b2da), all from `../../rlru4/diag4.py <worktree> <on> off`;
  - `peek_on2.py` (c072ad9a), the interim ON2 and ON6 read.
- **Rows:** `off/`, `on2/`, `on6/`, `on10/`, `shadow/`, `smoke/`.

**Next:** after the trailer rewrite, make the k = 6 stack the KVFlash 2.0b default on the engine branch. Then confirm it on fresh agentic trajectories.
