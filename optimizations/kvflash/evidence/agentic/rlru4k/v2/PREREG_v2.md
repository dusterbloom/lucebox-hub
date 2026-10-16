# PREREG v2: mode 4 with the recency scoring pool (rlru4k-v2)

Draft written 2026-10-16T17:43:37+02:00 (from `date`). At that time:
- no v2 row exists;
- the rlru4k ON10 arm (prs7) is still running;
- the prs8 binary is not built.

Before any v2 row, three things are appended to this file: the ON10 prediction row (computed by `predict_v2.py` once ON10 ends), the prs8 commits and md5, and the script md5s. kvflash-am-a0 commits the file, and its sha is frozen.

## Why

rlru4k (`../PREREG.md`, frozen rules) does not ship k = 6. ON6 scores 814/951 and beats ON2 by +4.9 points [+2.2, +7.8]. But its within-16K ON − OFF is −1.6 [−3.8, +0.5], and that lower bound breaks the −2.0 guard.

The loss is one class:
- 27 within-16K probes whose source fell out of ON6's window and was not recalled. OFF answers 26 of them, ON6 answers 5.
- For 22 of the 27, the OFF-state query (`../../rlru/dump`, `mass_end`, same block grid and sum) has the source in its top 6. The engine never hits a within-16K source that the dump misses (`../missdump.py`).

Cause, in `KvFlashPager::truncate`:
- When the retained prefix does not fit the pool, truncate keeps every current resident and only refills freed blocks.
- The refill is ranked by the previous request's score ledger.
- So a mode-4 request's recalled blocks stay resident through the next request's scoring pass. That pool lacks the newest history OFF holds, and the scoring query is formed without it.
- OFF never reselects and stays plain LRU.

Fix: `../SPEC_recent_pool.md` (md5 25211f67). With `LUCE_KVFLASH_SCORE_POOL_RECENT=1`, every pooled request first reselects the pool to the sink plus the newest chunks that fit, then drops the hook (plain LRU). This happens after every restore path and before the mode-4 decision. Mode 4 is otherwise unchanged.

The 27 are in-sample: they motivated the fix. `../diag27.json` (md5 340f8658, written by `../diag27.py` from prs7 ON6) freezes them, and the guard is read with and without them.

## Arms (all on prs8; the 951 retained probes; lock taken per phase)

| Arm | Settings |
|---|---|
| OFF | `QK16KND` + `LUCE_KVFLASH_LOG_RESIDENT=1` (log only) |
| ON2 | OFF + `LUCE_KVFLASH_PREFILL_RESELECT=4`, `LUCE_KVFLASH_RESELECT_SCORER=exact`, `LUCE_KVFLASH_RECALL_BLOCKS=2`, `LUCE_KVFLASH_SCORE_POOL_RECENT=1` |
| ON6 | the same with `RECALL_BLOCKS=6` |
| ON10 | the same with `RECALL_BLOCKS=10` |

OFF is re-run on prs8, not reused from prs7. Runner: `run_v2.sh <prs8 luce_server>`. Its phases are shadow, smoke, off, gate_d, on2, on6, on10. Any failed arm, smoke or check stops the run. The env checks (inherited `KVFLASH` env refused, each arm's env logged) are as in `../run_arms4k.sh`.

**Shadow (before any smoke).** prs8 with the flag off must pick exactly what prs7 picked: every `picked=[...]`, with k, tail and @R4, request by request (`shadow_cmp.py`). It runs on `../../rlru4/smoke_ids4.json` at k = 2, 6 and 10 (against `../smoke/on{2,6,10}`), and on `../../rlru4/drift/ids.json` at k = 2 (against `../smoke/drift_on2`). Answers are reported. Together with the fast scorer's identical picks (964be767), this carries rlru4k's flag-off results over to prs8.

**Smokes**, flag on:
- (a) The noop rewind is bit-exact, and the fitting probe equals OFF (`../../rlru4/cmp_smoke4.py`).
- (b) At k = 2, 6 and 10, each reselect logs its scorer line and a re-prefill ok. Every pooled request logs one recency line (counted against the resident lines).
- (c) On the drift ids, OFF and ON2 answer identically on every request mode 4 skipped, with equal restore paths (`../gate_cd.py c`).
- (e) Structural: the resident set equals OFF's on every request (`resident_cmp.py`), for noop, k = 2, 6 and 10, and the drift pair.
- (d) After OFF: the reduced-history OFF equals the full OFF on fitting probes (`../gate_cd.py d`).

## Structural check (gates validity, before any answer number)

After each ON arm, `resident_cmp.py off on<k>`:
- On every request, the `[kvflash] resident @<kv_offset>: n=<N> chunks=<ranges>` line logged immediately before the first prefill batch must equal OFF's.
- 0 mismatches is required. Any mismatch stops the run, and no answer number from that arm counts.
- On prs7 the restore path (`cached_prefix_tokens`, effective prompt) already matches OFF on all 1,106 requests of ON2, ON6 and ON10. So kv_offset is expected to match too, and the check compares sets at the same point.

## Read rules (trajectory-clustered bootstrap, `a1_core.boot_ci`, 10,000 resamples, seed 0)

1. **Primary:** ON6 − ON2 lower bound > 0.
2. **Secondary:** ON10 − ON6 lower bound > 0. Read only if rule 1 holds.
3. **Guard, per ON arm, read twice:** the 95% lower bound of within-16K ON − OFF is at least −2.0 points, (i) on all within-16K probes and (ii) excluding the 27 in `../diag27.json`. Both must hold.
4. **Bar, per ON arm:** at least 719/951, with the guard holding.

**Ship rule** (fixed now):
- ON10 if rules 1–4 hold for it;
- else ON6 if rules 1, 3 and 4 hold for it;
- else ON2 with the flag if rules 3 and 4 hold for it;
- else stay at the rlru4 design (k = 2, flag off), which passed at 766.

A failed structural check voids the arm, whatever its counts.

5. **Predictions** (`predict_v2.py`, computed from prs7 rows only):
   - The method assumes the v2 engine picks the dump's top k on every out-of-window probe. Engine-miss/dump-hit probes rise to the arm's measured hit rate, and engine-hit/dump-miss probes fall to its measured miss rate.
   - The assumption is optimistic (the router query is close to OFF's, not identical: batch plans differ at R4), so treat these as upper-leaning.

| Arm | prs7 (flag off) within / beyond / total | predicted within / beyond / total |
|---|---|---|
| ON2 | 700 / 67 / 767 | 704 / 73 / **777** |
| ON6 | 688 / 126 / 814 | 707 / 136 / **843** |
| ON10 | 683 / 148 / 831 | 703 / 155 / **858** |

The ON10 row was appended 2026-10-16T18:27 after rlru4k ON10 ended (rc 0, 0 errored rows), from `predict_v2.out`, before any v2 row.
- At k = 10 the predicted within-16K gain over OFF is only +3. The guard there may fail even if the mechanism works.
- ON10's within-16K cells on prs7:
  - engine-miss/dump-hit 23 (OFF 21, ON 1);
  - engine-miss/dump-miss 13 (OFF 13, ON 10);
  - recalled 119 (OFF 111, ON 110).
  - Recall itself costs about 1 point within 16K, whatever the router does.

6. **Speeds:**
   - Report per arm the medians of prefill, decode and wall; scorer ms; wall minus scorer; page events and H2D per reselect; and the recency step's page events and H2D (median, max).
   - Report prs7 vs prs8 for OFF and for each ON arm (the prs7 ON arms have the flag off, so that pair is binary plus flag).
   - Flag ON6 or ON10 if its median wall minus scorer is more than 25% above ON2's. No speed gate.
   - Clock: this box's tsc runs about 7% slow, so absolute tok/s read about 8% high. Same-box ratios hold. Per-request walls are medians only.
7. **Caching parity:** pt on every arm, `disk_hit` false on every row (asserted), and restore paths covered by the structural check.

Reported, not gated:
- the mechanism table per ON arm: engine vs dump hits on out-of-window sources, prs7 vs prs8. Prediction: within-16K engine-miss/dump-hit falls from 7 (k = 2) and 22 (k = 6) to near 0;
- `../../rlru4/diag4.py` recall classes;
- the stratum.

`analyze_v2.py <worktree>` computes everything above and writes `report_v2.json`.

## Stop conditions

- A server crash, more than 5 errored probes in an arm, or any failed shadow, smoke or structural check: stop, report, no verdict.
- Each arm's rc and errored rows are checked by the runner (`arm()` fails on either).

## Known conditions

- Other agents may compile at nice 19 -j4 during arms and take the 3090 lock between phases.
- The prs8 build waits for kvflash-am-a0's race-fix bench (`/mnt/kimi-k3/build/racefix-bench.done`).

## Binary (appended 2026-10-16T18:39:37+02:00 from `date`, before any v2 row; base file md5 08ddfd54)

`/mnt/kimi-k3/build/prs8-cuda/luce_server`, md5 a116ca38bcdb6326e2a486aceabda64b.
- Linked 18:38:08, after `racefix-bench.done` appeared (18:31:37), at nice 19 -j4.
- Engine branch `kvflash/2.0r-prefill-reselect` at db4becd1, tree 74799e54ad6b03806638f4f45b8cba9a64585576.
- Every shadow, smoke and arm phase runs on it.

Commits on top of prs7's ef4a2748:
- `508a9762`: the three tests against stubs. Built and run red.
  - `recent_pool_after_truncate_equals_fresh_lru_pager` fails only its flag-on assertion (`test_kvflash_pager_serde.cpp:789`). The flag-off half passes: the stale pick stays resident.
  - `format_ranges_collapses_runs` fails 3 of 4 assertions.
  - `reselect_recent_on_a_fitting_prefix_is_a_noop` passes on the stub, as expected.
- `a0b503b7`: `reselect_recent()`, `kvflash_format_ranges`, and the backend wiring.
  - `LUCE_KVFLASH_SCORE_POOL_RECENT`: on every pooled request, before the mode-4 decision.
  - `LUCE_KVFLASH_LOG_RESIDENT`: immediately before the first prefill batch.
  - `ctest -R "Kvflash|kvflash" -E CudaRace`: 103/103.
- `db4becd1`: merge of `kvflash/2.0r-fast-scorer` (002cb9b9, 43735d13), with no conflicts.
  - The same set: 109/109.
  - The CUDA race test (`reused_slot_write_is_ordered_after_queued_page_out`) passes under the 3090 lock. All 110 pass.

Script md5s at this append:

| File | md5 |
|---|---|
| `run_v2.sh` | 8a89544794bd1d3624a08b92069cba1d |
| `resident_cmp.py` | a4b7be71b154712f4740430f86a73023 |
| `shadow_cmp.py` | 3f711f5111b606b94fe92f1cdd3e2b9a |
| `analyze_v2.py` | b328d23188babf796c334d927700e60a |
| `predict_v2.py` | 91a6fd51ee8f45a91fb64dfc34b8a56d |
| `predict_v2.out` | cd02a2459dc701f2e6a8a5c4443d1436 |
| `../gate_cd.py` | 2df88c7209af1dcef936694e0c40a00a |
| `../diag27.json` | 340f865891f6b53fac1107d85432ff12 |
| `../../rlru4/cmp_smoke4.py` | 74906832a60f8e53a0870b1ea0b6924e |
| `../../rlru4/agentic_eval/run_arm.py` | c016b19cb8e2684581a0a8350158d526 |

`run_v2.sh` also logs these md5s into `provenance_v2.txt` at the shadow phase.

One gap, recorded before any row: "the stratum" (listed above as reported) is not computed by `analyze_v2.py`. It will be computed separately and labelled post hoc, as will `../../rlru4/diag4.py`'s recall classes per ON arm.
