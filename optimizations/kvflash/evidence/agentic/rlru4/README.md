# R-LRU mode 4 (rewind the turn), k = 2: result

Run on 2026-10-16 on the 3090, under PREREG.md (b0b834fb) and its Amendments 1 and 2.

**Setup:**
- Binary: prs6 (`/mnt/kimi-k3/build/prs6-cuda/luce_server`, md5 4f47802f…) for every phase. Each phase logged the md5 in `provenance_arms.txt`.
- Lock windows:
  - smokes: 10:51:39 to 10:55:18;
  - OFF: 10:55:34 to 11:46:16;
  - ON: 11:46:39 to 12:51:36.
- Both arms exited with code 0 and logged 0 row errors.

## Verdict: passes every preregistered rule

| | OFF | ON | ON − OFF (trajectory-clustered 95% CI) |
|---|---|---|---|
| overall (951) | 700 (73.6%) | **766 (80.5%)** [77.3, 83.7] | **+6.9 pts [+4.8, +9.1]** |
| within 16K (744) | 688 | 697 | +1.2 [−0.8, +3.3] |
| beyond 16K (207) | 12 | 69 | +27.5 [+20.6, +35.1] |

- **Rule 1 (primary):** lower bound +4.8 > 0. Pass.
- **Rule 2 (beyond 16K):** lower bound +20.6 > 0. Pass.
- **Rule 3 (within-16K guard):** no drop; ON is 9 probes above OFF. Pass.
- **Rule 4 (bar):** 766 ≥ 719 (B0 POOL16K, 75.6%), with rules 1 and 3 holding. **Pass, 47 probes above the bar.**
- **Rule 5 (prediction):** predicted 726 (679 within, 46 beyond); actual 766 (697, 69). The estimate leaned **low**, not high as stated in step 1. It scored a recalled block that cut its source message at 0.53, and those probes measured 0.94 (see the classes below).
- **Rule 7 (caching parity):** median pt is 349 on both arms, cache hit 100% on both, disk_hit 0 on both. Speeds are valid.

## Speeds (medians over probe rows)

| | OFF | ON |
|---|---|---|
| prefill | 426 ms | 1,071 ms |
| decode | 41.8 tok/s | 41.6 tok/s |
| client wall | 0.99 s | 1.68 s |
| wall minus scorer | | 1.26 s |
| exact scorer (CPU) | | 447 ms |

- **Rule 6 flag:** ON's prefill is 2.5× OFF's. There is no speed gate.
- Where the extra prefill goes: the CPU scorer (447 ms) and the re-prefill of the last user turn.
- Rewinds: 827 of 1,106 ON requests rewound, including 503/744 probes within 16K and 207/207 beyond.
- Decode is unchanged.

## Recall classes (post-hoc, `diag4.py` on the ON log; reported, not gated)

| band | class | n | OFF | ON | ON rate |
|---|---|---|---|---|---|
| within | fits (mode 4 skipped) | 241 | 221 | 228 | 94.6% |
| within | in window | 482 | 449 | 458 | 95.0% |
| within | whole message recalled | 2 | 2 | 2 | 100% |
| within | part recalled | 5 | 4 | 5 | 100% |
| within | missed | 14 | 12 | 4 | 28.6% |
| beyond | in window | 6 | 3 | 2 | 33.3% |
| beyond | whole message recalled | 19 | 0 | 18 | 94.7% |
| beyond | part recalled | 35 | 3 | 33 | 94.3% |
| beyond | missed | 147 | 6 | 16 | 10.9% |

**The mode-3 diagnosis is confirmed.** Under mode 3 the question was prefilled before the reselect, and a recalled source answered 33% of probes beyond 16K (43 of 134). Under mode 4 the question is prefilled after the reselect, and a recalled source answers 94% (51 of 54). That holds even when only part of the source message came back.

**The limit is now routing and capacity, not recall quality.**
- Of the 201 probes beyond 16K whose source is outside the window, the engine's two blocks held the source for 54.
- The rank dump's top 2, computed on the OFF state, holds it for 69.
- 147 are missed.

## Caveats

1. **Probes that mode 4 skipped still drift.**
   - Of the 241 probes skipped as fitting the pool, 21 answer differently from OFF: 228 correct on ON against 221 on OFF.
   - Prefilled and cached token counts match OFF exactly on all 21, and every request just before them also skipped.
   - OFF is bit-deterministic. Three OFF runs on three binaries (prs-cuda, prs5, prs6) agree on all 951 answers (`fits_drift.py`, `fits_drift.out`).
   - So state left by earlier rewound requests in the same server process reaches later no-op requests. This is an engine issue under investigation.
   - It does not change the verdict: with those probes forced to OFF's answers, ON is 759 (79.8%), still 40 above the bar.
2. **The runner ignores arm failures.** `arm()` logs the exit code but returns echo's status, so a failed OFF or ON arm would not stop the next phase (recorded in b0b834fb). Both arms logged rc=0 and 0 errored rows, and OFF was checked before ON was read.
3. **Lock released between arms (Amendment 2).** No other job ran during an arm, since each arm held the lock. OFF and ON rows are not interleaved in time.

## Files

| File | What it holds |
|---|---|
| `off/`, `on/`, `smoke/{off,noop,on}/` | rows and server logs |
| `run_arms4.log` | runner log |
| `provenance_arms.txt` | binary md5 per phase |
| `analyze.out`, `report.json` | preregistered read |
| `diag4.py`, `diag4.out` | recall classes |
| `fits_drift.py`, `fits_drift.out` | caveat 1 |
