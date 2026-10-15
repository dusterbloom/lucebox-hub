# A1 pilot: 6 trajectories, stage-1 probes, lucebox6 integrated GPU

The pilot defined in PREREG A1 Amendment 2, run under Amendment 3. It started 2026-10-15 09:53:12 UTC and finished at 11:42 UTC (PID 1350697). `summary.json` says `stopped: null`, and the watchdog stopped nothing. Code: commits 7725ffc2 and 53cfda9a. The files on lucebox6 have the committed md5s. Rows were copied from `~/r1e/out/a1_pilot/`. `pilot_report.json` is from `lucebox/a1/a1_pilot_report.py` (scoring with the harness `probes.score_exact`, intervals from `boot_ci` = `r1b_stats.boot_mean` at 95%).

**The pilot decides nothing (Amendment 2).** It says whether the proper run goes ahead as written.

| Item | Value |
|---|---|
| Trajectories | the first 6 in file order |
| Reference (FULL-CHK) correct probes, gate 3 | 118 of 204, 57.8%, inside 35-70% |
| Retention, shipping arm (F-BLK-UNI) | **10 of 118, 8.5%, [4.4, 12.5]** (6 clusters, indicative only) |
| B0 POOL16K retention on the same 6 trajectories (another machine and precision) | 88 of 111, 79.3%, [74.4, 83.5] |
| Same probes retained by both B0 full KV and the A1 reference (105) | shipping arm 10, POOL16K 84 |
| Determinism (trajectory 1, 3 repeats) | 6 of 6 rows equal |
| Seconds, FULL-CHK | build 1628, answer 3025, wall 4657 |
| Seconds, shipping arm | build 848, answer 949, wall 1802 |
| Peak GPU memory, FULL-CHK | 64.8 GiB allocated, 99.7 reserved |
| Peak GPU memory, shipping arm | 52.4 GiB allocated, 60.6 reserved |
| Peak host memory | 110.4 GiB used |

Retention by distance of the fact from the cut, shipping arm vs B0 POOL16K on the same trajectories:

| Distance (tokens) | Shipping arm | POOL16K |
|---|---|---|
| 0-4096 | 2 / 41 | 34 / 36 |
| 4096-8192 | 3 / 26 | 23 / 25 |
| 8192-16384 | 3 / 33 | 30 / 31 |
| over 16384 | 2 / 18 | 1 / 19 |

By probe type, shipping arm:

| Probe type | Correct / retained |
|---|---|
| line | 1 / 61 |
| defname | 2 / 24 |
| cmdline_path | 2 / 14 |
| cmdline_text | 2 / 15 |
| failed_test | 3 / 3 |
| traceback | 0 / 1 |

The shipping arm's wrong answers are fluent and on topic, but not the fact asked for. They give a name, path or line from the same repository and the same file, but not the one in question. For example, gold `./src/plopp/graphics/graphicalview.py` became `./src/plopp/backends/plotly/canvas.py`, and gold `panzoom` became `Canvas`. They are not gibberish and they do not run to the length limit (1 of the 20 rows sampled hit 64 tokens).

**The 64-token answer cap explains none of the losses** (`cap_check.py`, run 2026-10-15). The cap is the harness's `PROBE_MAX_TOKENS`, the same in B0 (asserted in `a1_core.Harness`). Scoring is exact match after whitespace, period and quote normalization. So a capped answer can only lose a correct gold when the gold itself does not fit.
- The longest gold of the 204 probes is 118 characters, which the reference answered correctly in 30 tokens. Every retained probe's gold fit, because the reference answered it within the cap.
- 10 of the 118 shipping-arm rows hit the cap, against 2 of 204 for the reference. None was cut partway through its gold. All 10 are non-answers: the model calls a tool (`sed -n '259p' ...`), reasons aloud ("The user is asking me to recall the exact text of line 5 ..."), or emits a `finish` call. Exact match scores them wrong at any cap.
- Counting all 10 as correct would still give 20/118, 16.9%.
- No row in A1 or B0 has reasoning or think tokens.
