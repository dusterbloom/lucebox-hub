# PREREG: R-LRU mode 4 (rewind the turn), end to end on the 3090

Written 2026-10-16T10:35:09+02:00 (from `date`). At that time:
- no mode-4 binary exists;
- `rank4.py` has not been run;
- no mode-4 row exists.

The previous design (`../rlru/`, mode 3) failed. Its post-hoc diagnostic (`../rlru/README.md`) found:
- routing held;
- a recalled block barely helped: old facts 33%, even with the whole source message resident;
- the question had been prefilled before the reselect.

This preregistration tests that explanation.

## Design under test

Same pool and router as `../rlru/` (16,384 rows; exact recent window plus k exact 640-token blocks; exact-mass router with the final token's query). The change is when the question sees the pool. Per request, `LUCE_KVFLASH_PREFILL_RESELECT=4`:
1. Prefill from R4 (the start of the last user turn, clamped to this request's prefill start) to the end.
2. Score with the final token's query.
3. Restore the state at R4 and reselect: window, k blocks, and a reserve for the tail.
4. Re-prefill [R4, end) over the new pool, with the same batch plan.

Mode 4 runs only when the pool cannot hold the prompt: chunks for [0, end) + reserve > 256. Otherwise it is a strict no-op, with no snapshot, scoring, rewind or re-prefill, and the same batch plan as OFF. The engine logs `skip, fits` for these requests. Recall is active only when the pool is full.

The engine commit and binary md5 are appended before any row; `prs6-cuda` is being built.

## Step 1: choose k from the existing rank dump (rule fixed now, before running `rank4.py`)

`rank4.py` reads `../rlru/dump` (the OFF state, `mass_end` = mode 4's query). For each k, it computes:
- the window: 256 − 1 − 10k − reserve − 1 chunks, with reserve = ceil(pt/64) + 1;
- each probe's estimate:
  - the fact is in the window: OFF's answer;
  - the whole source message is in the window plus the top-k blocks: 0.95 within 16K, 0.81 beyond;
  - the fact's block is recalled but the message is cut: 0.53;
  - otherwise: 0.

The rates are engine-measured with the source resident while the question was prefilled (`../prefill_reselect_diag`).

**Rule:**
- Among k ∈ {2, 4, 6} whose estimated within-16K drop vs OFF is at most 15 probes (the guard), take the highest estimated total.
- If the top two are within 5 probes, take the smaller k.
- If no k meets the guard, take the k with the smallest estimated drop and record that the guard is predicted to fail.
- The choice and its estimate (total, within, beyond) are appended with a `date` timestamp before any answer row. That estimate is the prediction.

Known bias, stated now: routing on the ON state found the source less often than the dump on the OFF state (134 vs 145 old facts at k = 6). So the estimate leans high.

## Step 2: arms

- **Arms:** identical to `../rlru/PREREG.md` step 2, except the ON arm uses `LUCE_KVFLASH_PREFILL_RESELECT=4`:
  - OFF = `QK16KND`;
  - ON = OFF + `LUCE_KVFLASH_PREFILL_RESELECT=4`, `LUCE_KVFLASH_RESELECT_SCORER=exact`, `LUCE_KVFLASH_RECALL_BLOCKS=<k>`.
- **Probes:** the 951 retained (`retained.json`, md5 c77c8901…).
- **Binary:** one binary for every smoke and both arms. OFF is re-run on it.

**Smokes on `smoke_ids.json`. Rows are not counted; if one fails, no arm runs.**
- **(a)** OFF, and mode 4 with `LUCE_KVFLASH_REWIND_NOOP=1`: rewind and re-prefill with no reselect.
  - Every request logs a re-prefill with logits max|diff| exactly 0 against the first pass. There are no row errors.
  - Report, not gate: the number of probe answers that differ from OFF, which is the effect of the batch boundary at R4.
- **(b)** ON on the smoke ids. Every request logs `mode=4 scorer=exact k=<k>` and a re-prefill ok line, with no reselect failure and no row errors.

## Read rules

Identical to `../rlru/PREREG.md`:
1. Primary: ON − OFF lower bound > 0.
2. Beyond 16K: lower bound > 0.
3. Within-16K guard: a drop of at most 2.0 points (15 probes).
4. Bar: ON ≥ 719/951, with rules 1 and 3 holding.
5. Prediction: the step-1 estimate.
6. Speeds: medians of prefill, decode and wall, plus the scorer ms; flag if ON's prefill is above 1.5× OFF's; no speed gate.
7. Caching parity: pt and disk_hit.

The trajectory-clustered bootstrap is `a1_core.boot_ci` (10,000 resamples, seed 0). `analyze.py` is copied from `../rlru` and must be adapted only in its log regex, for `mode=4`, before any row.

Reported separately, not gated: `../rlru/diag.py`'s recall classes applied to the mode-4 ON log. In particular, "within 16K, whole source message recalled" (39 probes under mode 3, where it scored 20 against OFF's 36) and "beyond 16K, whole source message recalled" (54 probes, 18 under mode 3). Mode 4's hypothesis predicts both near the 0.95 and 0.81 rates. A3's offline mode-4 protocol reached 86.5% [75.5, 93.6] on 304 held-out probes in its stage-2 early look (a274b04f).

The scorer's median ms is its own line in the speed report. Wall time with and without the scorer is reported separately, because the scorer runs on the CPU and a GPU scorer would remove it.

## Stop conditions

These are the same as `../rlru/PREREG.md`:
- A server crash, or more than 5 errored probes in an arm: stop, report, no verdict.
- Past 2.5 hours for both arms together: release the lock between arms and tell kvflash-am-a0.
