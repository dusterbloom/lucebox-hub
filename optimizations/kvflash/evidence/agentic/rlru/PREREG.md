# PREREG: R-LRU (exact recent window + k routed exact 640-token blocks), end to end on the 3090

Written 2026-10-16 05:55 CEST. At that time no row of the rank dump (`dump/`) or of any R-LRU answer arm exists.

## Design under test

Today's KVFlash pool (16,384 rows, LRU, 64-token sink) is changed at prefill end as follows:
- the newest chunks stay as an exact contiguous window;
- k whole 640-token blocks (10 chunks each) are paged back in exact from host RAM;
- the rest of the old history is absent;
- nothing is compacted.

Blocks are ranked by the exact attention mass of one query over every sealed chunk's exact keys. The scorer runs on the CPU over host and resident chunks (`LUCE_KVFLASH_RESELECT_SCORER=exact`).

Engine branch `kvflash/2.0r-prefill-reselect` 347f32be. Binary `/mnt/kimi-k3/build/prs3-cuda/luce_server`, md5 751d34260db3e9fb5afc10bb28016db1, for both answer arms.

## Step 1: choose k and the query, from the rank dump only (rule fixed now)

`dump/` is a probes-only pass over the 951 with no reselect. It writes, per request, the exact-mass chunk scores for two queries:
- `mass_r`: token R − 1, where R is the last turn cut; this is the end of the user message, and mode 1 uses it;
- `mass_end`: the last prompt token; mode 3 (rewind and re-run) uses it.

`rank_dump.py` turns these into an estimate per (k, query):
- the OFF answer if the fact chunk is inside the window;
- 0.81 if a source block ranks within the top k of the eligible blocks;
- 0 otherwise.

The 0.81 is the engine-measured old-fact rate with the source resident, from the diagnostic.

**Rule:**
- Take the (k, query) with the highest estimated correct, over k ∈ {2, 4, 6} and query ∈ {mass_r → mode 1, mass_end → mode 3}.
- If the top two are within 5 probes, take mode 1 over mode 3, since it has no rewind, and then the smaller k.
- The chosen (k, mode) and its estimate are appended to this file, with a timestamp, before any answer row exists. That estimate is the prediction.

## Step 2: answer arms

- **OFF:** arm `QK16KND` of the harness snapshot in `agentic_eval/`. It uses the B0 POOL16K flags:
  - `--kvflash 16384`, `KVFLASH_STATE_LADDER=1`, `KVFLASH_LIVE_RESTORE=1`;
  - `--max-ctx 52224`, prefix cache 12288 MiB, 4 slots;
  - temperature 0, seed 1;
  - B0's GGUF and template;
  - plus `LUCE_KVFLASH_POLICY=qk`.
- **ON:** OFF plus:
  - `LUCE_KVFLASH_PREFILL_RESELECT=<mode>`;
  - `LUCE_KVFLASH_RESELECT_SCORER=exact`;
  - `LUCE_KVFLASH_RECALL_BLOCKS=<k>`.
- Both run `--probes-only` over the 951 retained B0 probes (`retained.json`, md5 c77c89018f683acaf262beafde5908c1), in run_arm.py's order.
- One server per arm, both arms inside one hold of `flock /mnt/kimi-k3/build/3090.lock`, GPU idle before each arm.

**Smoke tests, before the arms. Rows are not counted; if one fails, no arm runs.**
- **(a)** Only if mode 3 is chosen. Run mode 3 with `LUCE_KVFLASH_RESELECT_TAIL_CHUNKS=255` and no `RECALL_BLOCKS`, so the reselect keeps the LRU pool, on the 5 smoke probes plus the OFF arm on the same 5 probes. Every answer must match OFF character for character. This proves that rewind and re-run is exact.
- **(b)** ON on the 5 smoke probes. Every request logs a reselect with `picked=[...]` non-empty or a window-only pool, the scorer's ms, and no row errors.

## Metric and read rules

These are the same as the prefill-reselect PREREG (`../prefill_reselect_run/PREREG.md`).
- Retention is measured over the 951, with `probes.score_exact`.
- Intervals use the trajectory-clustered bootstrap (`a1_core.boot_ci`, 10,000 resamples, seed 0, 95%), paired ON − OFF.

1. **Primary.** ON − OFF overall. ON wins if the lower bound is > 0.
2. **Beyond 16K** (207 probes). ON − OFF lower bound > 0.
3. **Within-16K guard** (744 probes). ON's within-16K drop vs OFF is at most 2.0 points (15 probes), point estimate. Otherwise ON is not a win.
4. **Bar.** ON ≥ 719/951 (B0 POOL16K, 75.6%), with rules 1 and 3 holding. OFF is the calibration and is never re-based: the earlier OFF was 700/951.
5. **Prediction.** The step-1 estimate, appended before the arms.
6. **Speeds.** Medians of prefill ms, decode tok/s and client wall, both arms; also the median scorer ms from the log.
   - Flag if ON's prefill median is above 1.5× OFF's. This is expected: the scorer is CPU-side, about 0.2–0.5 s. The flag is reported as the cost to remove with a GPU scorer. There is no speed gate.
7. **Caching parity.** pt and cached prefix medians on both arms, and disk_hit. A pt difference above 10% voids the speed comparison only.

## Stop conditions

- A server crash, or more than 5 errored probes in an arm: stop, report, no verdict.
- Past 2.5 hours for both arms together: release the lock between arms and tell kvflash-am-a0.

## Amendment 1 (2026-10-16 06:05 CEST, before any dump or answer row)

Smoke (a) uses `LUCE_KVFLASH_RESELECT_TAIL_CHUNKS=254`, not 255. `reselect()` protects the sink chunk and every chunk `c > cur - 1 - tail`, so 255 protects 257 chunks for 256 slots and `std::sort` drops one arbitrarily. 254 protects exactly the LRU set. The gate is `cmp_smoke.py`: at least one rewind, a `rerun ok` for every rewind, 0 page events on every rewind, every answer equal to OFF, no row errors. The runner is `run_arms.sh <binary> <mode> <k>`.

## Step 1 result (appended 2026-10-16T06:57:36+02:00, after the rank dump, before any answer-arm row)

Dump: 951 probes, 1,106 requests, 1,106 score lines, 0 errors, binary prs-cuda md5 356f79c7f93468534ffd91a34b7b9415. Its own answers score 700/951 (688/744 within 16K, 12/207 beyond), identical to the earlier OFF.

`rank_dump.py` estimates, over the rule set k in {2, 4, 6}:

| k | query | in window | recalled | missed | est. correct | est. beyond 16K | recall@k out of window | random |
|---|---|---|---|---|---|---|---|---|
| 2 | mass_r | 736 | 49 | 166 | 718 | 34 | 22.8% | 6.4% |
| 2 | mass_end | 736 | 83 | 132 | 745 | 59 | 38.6% | 6.4% |
| 4 | mass_r | 707 | 98 | 146 | 734 | 51 | 40.2% | 12.8% |
| 4 | mass_end | 707 | 163 | 81 | 787 | 102 | 66.8% | 12.8% |
| 6 | mass_r | 653 | 184 | 114 | 752 | 78 | 61.7% | 19.3% |
| 6 | mass_end | 653 | 237 | 61 | **795** | 117 | 79.5% | 19.3% |

The top two are 795 (k 6, mass_end) and 787 (k 4, mass_end), 8 apart, so the 5-probe tie rule does not apply.

**Choice: mode 3 (`LUCE_KVFLASH_PREFILL_RESELECT=3`), k = 6.** Smoke (a) applies.

**Prediction (rule 5):** ON 795/951 = 83.6%. That is 678/744 within 16K (a 10-probe drop, inside the 15-probe guard) and 117/207 beyond. ON minus OFF = +95 probes against OFF 700.

## Amendment 2 (2026-10-16T08:02:38+02:00 from `date`; before any answer-arm row; smoke rows only so far)

### What happened

**Smoke (a) failed at 07:31:23 on prs3, so no arm ran.** Two of 6 requests broke it:
- **p36:7:** the answer differs from OFF. Rewind had 0 page events. Answers: "This project is…" against "This library is…".
- **p36:4:** 2 page events with tail 254. Its answer matches OFF.

**Control at 07:36–07:37** (`run_split_ctl.sh`, prs3, smoke rows only, `smoke_split/`): mode 2 with tail 254. It makes the same split (prefill to end − 1, then a 1-token batch) with no rewind.
- **p36:7:** the split control equals the mode-3 no-op ("project"). The 1-token final batch flips this near-tie; the rewind does not.
- **p36:4:** the 2 page events came from an engine bug. `kvflash_prefill_apply` left its exact-mass ledger installed as `score_hook` into the next request.
  - p36:11 decoded 31 tokens. p36:4's live restore to 33280 then freed blocks.
  - `refill_after_truncate_` recalled host chunks ranked by p36:11's query, so the pool was no longer sink + newest 255.
  - The two modes' ledgers differ (query end − 2 vs end − 1), which is also why mode 2 and mode 3 answered p36:4 differently.

**Correction to a claim made in passing, not in this file:** qk-policy eviction is not involved.
- OFF installs no hook on these probes. The only other install is the decode-time `kvflash_maybe_reselect`.
- It fires only when generated % τ = 0, with τ ≥ history/45 ≥ 266, and answers are capped at 64 tokens.
- Zero `reselect @gen` lines in `dump/` or `../prefill_reselect_run/off/`. OFF is pure LRU.

### Engine changes

Branch `kvflash/2.0r-prefill-reselect`, on top of 347f32be, three commits:

| Commit | Change |
|---|---|
| 19ac63ae | `reselect()` breaks score ties by keeping resident chunks first. Protecting every chunk (tail 100000) then moves 0 pages, whatever the pool state. |
| a67d724f | Mode 3 logs the final token's first-pass logits against the rerun's: `rerun ok, logits max\|diff\| X vs first pass (argmax a / b)`. |
| dfc5a3db | `ReselectRequestScope` (RAII `ScopedScoreHook` + clearing `kvflash_scores_`) is held per request in `generate_impl` and `restore_and_generate_impl`. A reselect's hook lasts through its own request's decode and is restored on every exit path. |

- Unit tests are red-then-green for each; ctest 91/91.
- OFF installs no hook here (see above), so this leaves OFF's behaviour unchanged.

**Binary for every smoke and both arms:** `/mnt/kimi-k3/build/prs5-cuda/luce_server`, md5 6884c4fd682b6b3b8de2e40a90188488. This replaces prs3 (751d3426…) in "Design under test".

### Smoke (a), redefined

Run OFF and the mode-3 no-op (`LUCE_KVFLASH_RESELECT_TAIL_CHUNKS=100000`) on `smoke_ids.json`, into `smoke2/`. Gate (`cmp_smoke.py`, md5 39230dfb…):
- at least one rewind;
- 0 page events on every rewind;
- rerun logits max|diff| exactly 0 on every rewind;
- no row errors.

This tests the rewind inside one request, so nothing from earlier requests can confound it.

**Flip report:** the number of no-op probe answers that differ from OFF. With the hook request-scoped, it measures only the 1-token final batch.
- It is printed in `run_arms.log` before the OFF arm starts.
- If more than 1 of 5 flip, the report states it as part of ON − OFF that is not recall.
- It is not a gate.

Smoke (b) is unchanged. The runner is `run_arms.sh` (md5 510457c9…): `bash run_arms.sh /mnt/kimi-k3/build/prs5-cuda/luce_server 3 6`.

**Unchanged:** the step-1 choice (mode 3, k = 6), the prediction (795/951; 678 within 16K; 117 beyond), the 951 probes, the read rules and the 719 bar.
