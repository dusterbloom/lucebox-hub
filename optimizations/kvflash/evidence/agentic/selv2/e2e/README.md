# Selector v2 end to end on the 3090 (2026-10-17): WIN

Preregistered in `PREREG_e2e.md` (commit 4121bd29, before any v2 row; one gate amendment at 22:25, before any accuracy
number). Binary selv2-cuda 497cbeca (22b59a62) on both arms, same session; harness QK16KND, 951 probes, 16K pool.
v2 = ON6 env + `LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2`; on6ref = ON6 env (flags off).

Validity: shadow (flags off = prs8 ON6 picks and answers on the smoke ids, 0 mismatches), smoke and full v2 gate (827/827
reselects ran selector v2; 64 queries on 822, 33-63 on 5 short prefills), structural check (resident sets = prs8 OFF on
1106/1106 requests, both arms), 951/951 probes, 0 errors, 0 disk_hit, median pt 349 on both. on6ref answers = prs8 ON6 on 951/951.

| | on6ref | v2 | v2 - on6ref (95% cluster bootstrap) |
|---|---|---|---|
| beyond (207) | 138 | **176** | +18.4 [+10.9, +25.6] pts; discordant 44 vs 6, sign p 3e-8 |
| within (744) | 709 | 712 | +0.4 [-0.7, +1.5] |
| within excl. 27 (717) | 685 | 686 | +0.1 [-0.9, +1.2] |
| overall (951) | 847 | **888** | +4.3 [+2.5, +6.3]; discordant 58 vs 17 |
| beyond facts fully held (206 out of window) | 147 | 186 | missed 57 -> 14 |
| prefill ms (median) | 848 | 2190 | the scorer |
| decode tok/s (median) | 42.9 | 43.8 | noise |
| client wall s (median) | 1.40 | 2.81 | |
| scorer ms per reselect (median / max) | 31 / 65 | 1720 / 3094 | |

Predictions: accuracy, mechanism and decode inside the predicted ranges; scorer cost above them (predicted 800-1400 ms).
Cost line: not shippable as the default (median scorer 1720 ms > 200 ms). The WIN means: build the GPU scorer.

Files: `run_e2e.sh` (phases), `gate_e2e.py` (checks 1-2), `analyze_e2e.py` (read rules), `report_e2e.json` (final),
`report_e2e.provisional.json` (against prs8 ON6, same numbers), `run_e2e.log`, `provenance_e2e.txt`. Arm dirs stay on
/mnt/kimi-k3/split-work/selv2/e2e (v2/, on6ref/, shadow/, smoke/).
