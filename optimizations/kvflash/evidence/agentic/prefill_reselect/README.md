# Prefill reselect on the 3090 (peer session, 2026-10-16)

The peer session ran this from branch `kvflash/2.0r-prefill-reselect` at 0ecda5d4 under `PREREG.md`, which was written before any row. This session copied the files from `/mnt/kimi-k3/split-work/prefill_reselect_run/` unedited, except for the `agentic_eval/` code copy, whose md5s are in `provenance.txt`.
- Arms: OFF is QK16KND (the exact 16K pool). ON is the same arm with `LUCE_KVFLASH_PREFILL_RESELECT=1`, which rescores and reselects the whole pool at prefill.
- Parity:
  - One binary, md5 749138e9..., on both arms, with the ggml .so md5s in provenance.
  - 951 probes per arm, 0 errors.
  - The median pt is 349 on both arms, cache_hit is 100% on both, and disk_hit is 0.
- Report: `report.json` (`analyze.py`). Intervals are 95% bootstrap, clustered by trajectory.

## Result: FAIL

| Subset | OFF | ON | ON−OFF [95% interval] |
|---|---|---|---|
| overall | 700/951 | 648/951 | −5.5 pts [−9.1, −2.1], rule 1 fails |
| within 16K | 688/744 | 553/744 | −18.1 pts [−22.2, −14.3], guard of 2.0 fails |
| beyond 16K | 12/207 | 95/207 | +40.1 pts [+32.9, +47.2] |
| beyond 16K, the 17 shared with E3 | 1 | 13 | (E3: 15) |

ON lands 648/951 against the 719/951 bar.

Speed medians, OFF then ON: prefill 410.6 then 482.6 ms (×1.18, under the 1.5 flag); decode 44.0 then 43.1 tok/s; wall 0.933 then 1.051 s.

## Post hoc (`diag.py`, `calib.py`; not preregistered)

- Within-16K losses grow with distance. By distance bin:

  | Distance | 1-2K | 2-4K | 4-8K | 8-12K | 12-16K |
  |---|---|---|---|---|---|
  | Losses | 1 | 21 | 38 | 54 | 45 |

- The wrong answers are neighbouring lines or "I can't see line N", so context is missing; nothing is corrupted.
- The engine reselected at the last turn cut, with the query at the end of the user message. The simulation that predicted 736/744 within-16K residency used the last prompt token's query instead, so this run did not test the simulated policy.

## Reading

- Query-scored whole-pool reselect recovers old facts: +40 points beyond 16K.
- It pays for that by evicting recent context the answers need: −18 points within 16K.
- A policy that keeps a recent exact window and reselects only the older chunks has the same structure as A3.
