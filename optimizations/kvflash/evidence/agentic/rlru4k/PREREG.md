# PREREG: R-LRU mode 4 with more recall blocks (k = 6 and k = 10)

Draft written 2026-10-16T13:03:56+02:00 (from `date`). Revised at 2026-10-16T13:11:47+02:00 with kvflash-am-a0's six review edits: noninferiority guard, ship rule, speed flag against ON2 with H2D bytes, the stratum the window gives up, smoke (d), env and disk-cache parity. At both times:
- no row of this design exists;
- the binary is not fixed. It will be the build carrying the layout fix for `../rlru4/README.md` caveat 1, and its commit and md5 are appended before any row.

## Why

Mode 4 at k = 2 passed (`../rlru4/README.md`, b48e4f40): 766/951 against OFF's 700 and the 719 bar.
- A recalled source beyond 16K answers 94% of the time, whether the whole message or only part of it came back.
- What limits it now is routing capacity: 147 of 201 out-of-window sources beyond 16K were missed with two blocks.

`../rlru4/rank4_cal.py` is rank4 refitted with the rates mode 4 measured (whole 0.95, part 0.94, missed 0.29 within and 0.11 beyond). It reproduces k = 2 at 770 against the measured 766, and its beyond-16K estimate is high by 14/83 (17%), because the dump's routing on the OFF state finds more sources than the engine does. Its estimates:

| k | window (tokens) | est. within | est. drop | est. beyond | total |
|---|---|---|---|---|---|
| 2 | 14,528 | 688 | 0 | 83 | 770 |
| 4 | 13,248 | 683 | 5 | 127 | 810 |
| 6 | 11,968 | 684 | 4 | 144 | 828 |
| 8 | 10,688 | 681 | 7 | 156 | 836 |
| 10 | 9,408 | 686 | 2 | 162 | 848 |

That table was computed before this draft. So the two k values are not picked by a hidden rule:
- k = 6 takes most of the estimated gain;
- k = 10 is the largest k computed and tests whether the gain keeps going.

The within-16K estimates at large k rest on only 7 measured within-16K recalls, so they are the weakest number here.

## Arms (all on the one fixed binary; the 951 retained probes; lock taken per arm)

| Arm | Settings |
|---|---|
| OFF | `QK16KND` |
| ON2 | OFF + `LUCE_KVFLASH_PREFILL_RESELECT=4`, `LUCE_KVFLASH_RESELECT_SCORER=exact`, `LUCE_KVFLASH_RECALL_BLOCKS=2` (the k = 2 comparator, re-run on the fixed binary) |
| ON6 | the same with `RECALL_BLOCKS=6` |
| ON10 | the same with `RECALL_BLOCKS=10` |

**Precondition for every smoke:** `ctest -R kvflash` is fully green on the fixed build, including the two tests that are red at 4eb3ba60.

**Smokes** before any arm:
- (a) On `../rlru4/smoke_ids4.json`: noop, as in `../rlru4`. Every re-prefill is bit-exact, and the fitting probe equals OFF.
- (b) On `../rlru4/smoke_ids4.json`: k = 2, 6 and 10 each log their scorer line and a re-prefill ok for every reselect.
- (c) The layout fix under mode 4: `../rlru4/drift/ids.json` run as OFF and as ON2. The five f9b0 p12 probes, which mode 4 skips, must be identical between the two.
- (d) The layout fix under plain KVFlash. OFF on `../rlru4/drift/ids.json` (a reduced history) must answer every probe that fits the pool identically to the full-history OFF arm. This is gated after the OFF arm and before any ON arm.
  - On prs6 this comparison is red. f9b0 p12:1 and p12:7 differ (`../rlru4/drift/cmp_drift.out`), with the same restore path (cached 5,632 and 11,776 tokens).
  - Revised 13:20:59, before any row. The first version (two orders of the same probes) has no power:
    - "Reverse order" cannot be done through `--retained`, because `run_arm.py` loads it as a set and runs in task-file order.
    - Two order pairs built with task files both passed on prs6 (`smoke_d/`): 08db-p36 vs 31effae-p36 before f9b0, and 08db-p36 vs 08db-all before f9b0.
    - A third pair (a fitting trajectory before f9b0) voids itself. A fitting history leaves a dense live session, which restores at the exact shared length (5,922 tokens). An overflowed history restores from the 5,632 rung. So the batch plans differ.
    - `smoke_d/cmp_d.py` checks cache-path equality for this reason.

If any smoke fails, no arm runs. If (c) or (d) fails on the fixed binary, nothing is patched blind. First log the physical maps, the logical KV hashes at the shared 5,632 cut, and the logits at the first divergence.

**Binary:** carries both fixes, the truncate layout fix (`4eb3ba60` green) and the page-out/slot-reuse ordering (an event wait). Report the decode speed before and after on the smokes.

**Env and cache parity.**
- The runner refuses to start if any `LUCE_KVFLASH_*` or `KVFLASH_*` variable is set in its inherited environment. Each arm then sets only the variables in the table above, through explicit prefixes.
- The runner logs each arm's `LUCE_KVFLASH_*` and `KVFLASH_*` environment as the server sees it. `run_arm.py` clears only `KVFLASH_*`, then sets `KVFLASH_STATE_LADDER`, `KVFLASH_LIVE_RESTORE` and `LUCE_KVFLASH_POLICY` itself.
- This binary has no `--no-kv-cache` flag. Its disk KV cache turns on only with `--kv-cache-dir`, which the harness never passes. `analyze.py` asserts `disk_hit` is false on every row of every arm.

## Read rules

The trajectory-clustered bootstrap is `a1_core.boot_ci` (10,000 resamples, seed 0).

1. **Primary:** ON6 − ON2 lower bound > 0. More blocks beat two.
2. **Secondary:** ON10 − ON6 lower bound > 0. The gain keeps going past 6. Read only if rule 1 holds.
3. **Guard, per ON arm (noninferiority):** the trajectory-clustered 95% lower bound of within-16K ON − OFF is at least −2.0 points. This is the same 2-point margin the engine gate uses. An arm that breaks the guard is not shipped, whatever its total.
4. **Bar, per ON arm:** at least 719/951, with the guard holding.

**Ship rule** (fixed now, before any row): ship the largest k whose rules all pass.
- ON10 if rules 1, 2, 3 and 4 hold for it;
- else ON6 if rules 1, 3 and 4 hold for it;
- else stay at k = 2.
5. **Predictions:** the calibrated estimate with beyond-16K cut by the measured 17% overestimate.
   - ON2: near its measured 766. The layout fix changes rounding only, so expect a few answers to flip either way;
   - ON6: 684 + 120 = **804**;
   - ON10: 686 + 134 = **820**.
6. **Speeds:**
   - Report medians of prefill, decode and wall, plus scorer ms and wall minus scorer.
   - Report page events and H2D bytes per reselect. The fixed build logs page-ins and their MiB on the mode-4 reselect line.
   - Flag ON6 or ON10 if its median wall minus scorer is more than 25% above ON2's. ON2 is already 2.5× OFF's prefill, so a flag against OFF would fire on every arm.
   - No speed gate.
7. **Caching parity:** pt on every arm, and `disk_hit` false on every row (asserted).

Reported, not gated:
- `../rlru4/diag4.py` recall classes per ON arm, and the engine's picks vs the dump's top k.
- **The stratum the window gives up.** Per arm, within-16K probes whose source lies between window(k) and 16K back, with OFF, ON2 and that arm's answers. Window(k) is the arm's own logged tail. Any within-16K drop at k = 10 (window 9,408 tokens) would come from here, and the calibration rests on only 7 within-16K recalls.

## Stop conditions

- A server crash, or more than 5 errored probes in an arm: stop, report, no verdict.
- Because the runner does not stop on a failed arm (`../rlru4`, b0b834fb), check each arm's rc line and its errored rows before reading the next.

## Binary (appended 2026-10-16T13:33:56+02:00 from `date`, before any row)

`/mnt/kimi-k3/build/prs7-cuda/luce_server`, md5 7024e7cdc72ba92095c7c8c20a664fb0, built 13:30:16, on engine branch `kvflash/2.0r-prefill-reselect`. Every smoke and every arm runs on it. `ctest -R "Kvflash|kvflash"` passes 101/101, including the two tests that were red at 4eb3ba60.

Commits on top of 4eb3ba60:
- `910e070d`: the mode-4 reselect line logs `; N in (X MiB H2D)` after `page events`; the rest of the line is unchanged.
- `5f95f149`: Codex's case as a test (8 blocks; chunks 0–5, then 6–13; truncate to 6), plus a swap-cycle case. It asserts the map, the next appends, retained bytes and zeroed vacated blocks.
- `4bd85c49`: the layout fix. When the retained prefix fits the pool, `truncate` moves every retained chunk to block c (residents through a host scratch buffer, host-only chunks from their host copy, cycles parked first). It rebuilds the free stack to the fresh order, zeroes the vacated blocks, renumbers `last_use` and advances the epoch. The path for a prefix that does not fit is unchanged.
- `ec022d31`: a red CUDA stress test for the page-out / slot-reuse race. On the unfixed pager it corrupted 28 of 40 rounds (260 chunks).
- `ef4a2748`: the race fix. `slot_for()` records an event on the page stream after queued page work, and the compute stream waits on it on the device, with no host sync. The test then shows 0 corrupted chunks.

Effect on the comparators: OFF and ON2 run on this binary, so the comparisons stand. Their counts can move against prs6 (700 and 766) for two reasons: rounding from the new layout, and any corruption the race caused on prs6. These rows cannot separate the two. Decode tok/s before and after (prs6 vs prs7) is reported from the smoke OFF runs, which use the same ids as `../rlru4/smoke`.
