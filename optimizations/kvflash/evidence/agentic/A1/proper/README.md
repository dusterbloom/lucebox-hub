# A1 proper run, stage 1 (probes), 20 trajectories: early look FAIL, lucebox6 integrated GPU

The proper run of PREREG A1 (Amendment 1 early look, Amendment 4 launch): FULL-CHK reference and the shipping arm (F-BLK-UNI, ratio 0.10) on the first 20 trajectories in file order, then the block-512 control on the first 8. Trajectories 1 to 6 are the pilot's rows, copied unedited (md5 in `provenance.txt`). Code: commit 9c0f1958, files checked against the launch md5 list on every resume (`md5_resume_try2_swapkill.txt`). The tables come from `lucebox/a1/a1_proper_report.py` (`proper_report.json`); scoring is the harness `probes.score_exact`, intervals are `boot_ci` (trajectory as the cluster, 10,000 resamples, seed 0; at 95% it equals `r1b_stats.boot_mean`).

## Verdict

**FAIL.** At the preregistered early look (20 trajectories, 337 retained probes) retention is **38 of 337, 11.3%, 99.9% interval [6.1, 16.7]**, against the bar of 75.6%. The upper bound is below the bar, so stage 1 stopped with `early-FAIL` (`summary.json`, `checks.json`). The 95% interval is [8.2, 14.4]. Gate 3 (first 5 trajectories) passed: reference yield 97 of 168, 57.7%, inside 35-70%. Over all 20 trajectories the reference answered 337 of 642 probes correctly (52.5%).

| Item | Value |
|---|---|
| Retention, shipping arm | **38 / 337, 11.3%**; 95% [8.2, 14.4]; 99.9% [6.1, 16.7]; 20 clusters |
| Trajectories 1-6 (pilot rows) / 7-20 | 10 / 118 (8.5%) / 28 / 219 (12.8%) |
| B0 POOL16K, same 20 trajectories (another machine and precision) | 256 / 330, 77.6%, 95% [74.6, 80.8] |
| Same probes retained by B0 full KV and the A1 reference | n = 309: shipping arm 37, POOL16K 243; discordant pairs 7 vs 213, McNemar exact p = 5.5e-54 |

Retention by distance of the fact from the cut, shipping arm against POOL16K on the same trajectories:

| Distance (tokens) | Shipping arm | POOL16K |
|---|---|---|
| 0-4096 | 15 / 116 | 104 / 108 |
| 4096-8192 | 8 / 56 | 49 / 55 |
| 8192-16384 | 8 / 104 | 101 / 106 |
| over 16384 | 7 / 61 | 2 / 61 |

By probe type, shipping arm (correct / retained): line 9 / 160, defname 9 / 58, cmdline_path 5 / 45, cmdline_text 6 / 63, traceback 4 / 6, failed_test 5 / 5. Every retained probe has a shipping-arm row (`retained_without_ship_row` is empty).

## Block-512 control (FULLB)

The prereg's definition (PREREG A1 section 4, "Noise floor", item 2): *"The FULL-CHK arm is re-run with 512-token blocks on the first 8 trajectories of the file ... Reported as agreement of block-512 with block-640 on the same probes and decisions. No gate; if more than 5 points of the retained probes flip, the verdict text says the reference is soft by that much."* Section 5 adds: *"Control: agreement of block-512 with block-640 FULL-CHK; no interval beyond counts."*

Result: 8 trajectories, 147 retained probes, all with a FULLB row. **1 of 147 flips** (0.7%, `chatcmpl-2a396fbe...:p12:10`), correctness agrees on 146 of 147, and the answer token ids are identical on 146 of 147. That is far below 5 points, so the reference is not soft. The control does not bear on the verdict.

## How the run went: stops and resumes

Times are lucebox6 host time (UTC; CEST is UTC+2). The watchdog (`a1_watchdog.py`) kills the run when swap grows by more than 1,024 MiB over its level at watchdog start; that level was 465 MiB both times (`mem.log.*`).

| Event | Time (UTC / CEST) | Host used | Swap | What happened |
|---|---|---|---|---|
| Launch 1 | 12:21:34 / 14:21:34 | 66.7 GiB | 465 MiB | Cold weight load (`stdout.log.try1`, `mem.log.try1_swapkill`). |
| **Stop 1** | 12:22:35 / 14:22:35 | 97.1 GiB | 1,602 MiB (+1,137) | Killed during the weight load, 60 s in. No row was written (`run.log.try1` empty). |
| Launch 2 | 12:32:25 / 14:32:25 | 65.8 GiB | 465 MiB | Fresh launch from the launch script, same memory gate (`provenance.txt`, pilot rows copied unedited). |
| **Stop 2** | 15:57:49 / 17:57:49 | 101.8 GiB | 1,502 MiB (+1,037) | After 18 trajectories were finished (FULL and SHIP rows, 3 h 25 min into the attempt). The partly finished unit was rerun from the start on resume (rows resume per trajectory and arm). |
| Resume (`~/r1e/a1_resume_main.sh try2_swapkill`) | 16:10:31 / 18:10:31 | 66.6 GiB | 466 MiB | Code md5s identical to the launch list: yes (`md5_resume_try2_swapkill.txt`); rows at resume 582 FULL, 310 SHIP, 36 progress. |
| Finish | 17:54 / 19:54 | 107.2 GiB | 1,237 MiB | Trajectories 19 and 20 done, early look gave FAIL (the look is taken at 20), stage 1 stopped, then the control ran on 8 trajectories and finished. No stop after the resume. |

Peak host memory used: 110.9 GiB (resume attempt), 107.6 GiB (attempt 2); peak swap 1,502 MiB. Both stops were the swap-growth rule, not the 120 GiB used rule. Nothing was edited by hand.

## Seconds and memory

Summed over `progress.jsonl`. These include the pilot's rows for trajectories 1-6 (pilot: FULL-CHK wall 4,657 s, shipping arm 1,802 s). A unit cut by a stop is not in `progress.jsonl`; it appears once, from its completed rerun. The elapsed wall from launch 2 to the finish was 5 h 22 min including the 13 min between stop 2 and the resume.

| Arm | Build s | Answer s | Wall s | Peak GPU allocated / reserved |
|---|---|---|---|---|
| FULL-CHK (20 trajectories) | 5,453 | 9,607 | 15,075 | 64.8 / 99.9 GiB |
| Shipping arm (20 trajectories) | 2,844 | 2,662 | 5,523 | 52.4 / 61.0 GiB |
| FULLB, block 512 (8 trajectories) | 2,201 | 1,799 | 4,005 | 63.6 / 100.0 GiB |

## Decisions the prereg attaches to this verdict

- **A2 runs after an A1 FAIL.** PREREG A2: "A2 starts after A1's stage-1 verdict, because it reuses A1's FULL-CHK rows. If A1's stage 1 does not stop with FAIL at the early look, A2 is not run." The condition holds. A2 reuses these 20 FULL-CHK trajectories as its reference.
- **Diagnostics E2, E3, E5, E6, E7 run first**, each preregistered ("Diagnostics E2, E3, E5 before A2" and "Diagnostics E6, E7"), on the 6 pilot trajectories, one after the other on lucebox6. They decide nothing and change no A1 or A2 rule.
- Stage 2 (decisions) is not started by this verdict (Amendment 4).

## Files

`FULL.jsonl`, `SHIP.jsonl`, `FULLB.jsonl`, `progress.jsonl`, `checks.json`, `summary.json`, `provenance.txt`, `md5_resume_try2_swapkill.txt`, `proper_report.json`, and the logs `run.log*`, `stdout.log*`, `mem.log*`, `watchdog.out*` (suffix `.try1` and `.try2_swapkill` are the stopped attempts). Copied with scp from `~/r1e/out/a1/` and md5-checked against the host after the copy. The `pid` files are not kept.
