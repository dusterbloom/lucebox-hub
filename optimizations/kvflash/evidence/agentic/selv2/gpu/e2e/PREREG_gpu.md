# Selector v2 with the GPU scorer, end to end: preregistration

Written 2026-10-18, before any row of the arms below exists, while the GPU scorer is being implemented. Design and bench in
`optimizations/kvflash/GPU_SCORER.md` (commit 0320b7a3); the CPU-scorer arm is `../e2e` (PREREG_e2e.md, WIN, cost line failed:
scorer median 1720 ms per reselect).

## Question

Does selector v2 with the GPU scorer keep v2's accuracy win over ON6 at k=6 and bring the scorer under the cost line
(median <= 200 ms per reselect), so v2 can ship as the default?

## Binary and arms (3090, harness `rlru4/agentic_eval/run_arm.py` QK16KND, probes only, retained list `rlru4/retained.json`,
16K pool, max-ctx 52224, temp 0, seed 1, one server at a time under `/mnt/kimi-k3/build/3090.lock`)

One binary for every arm: `/mnt/kimi-k3/build/selv2gpu-cuda/luce_server` (md5 logged in `provenance_gpu.txt` at the first phase).

- **gpu**: ON6 env (`LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=6
  LUCE_KVFLASH_SCORE_POOL_RECENT=1 LUCE_KVFLASH_LOG_RESIDENT=1`) + `LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2
  LUCE_KVFLASH_SCORER_DEVICE=gpu`.
- **on6ref**: ON6 env only, same binary, after gpu in the same session. The reference for every number.
- Read only: `../e2e/v2` (v2, CPU scorer, 497cbeca) for the answer-agreement line; prs8 OFF (`rlru4k/v2/off`) for the
  structural check.

## Checks (a failed one stops the run)

1. Shadow (smoke ids `rlru4/smoke_ids4.json`, gpu env + `LUCE_KVFLASH_SCORER_SHADOW=1`): every reselect logs a `gpu scorer:`
   line and the shadow pick comparison; no `gpu scorer unavailable`; picks identical to the CPU scorer's on >= 90% of
   reselects and >= 26 of 30 units in common on every reselect.
2. gpu arm log gate (`gate_gpu.py`): PREREG_e2e check 2 as amended (selector v2 at 64 queries, units of 2, 30 distinct picks
   >= unit 5, one re-prefill ok, no `prefill reselect...failed` line, recency line per pooled request, no disk_hit) plus one
   `gpu scorer:` line per reselect and no `gpu scorer unavailable` line.
3. Flags-off identity: on6ref answers equal `../e2e/on6ref` (497cbeca) on >= 99% of the 951 probes (expected 951/951: the
   env-unset path is unchanged code).
4. Structural: resident sets at the first prefill batch equal prs8 OFF's on every request (`resident_cmp.py`), both arms.
5. 951 of 951 probes on both arms, 0 errored rows, no disk_hit, median pt equal.

## Metric and decision (`analyze_e2e.py` copied unchanged into this directory: arm dirs `v2` = gpu, `on6ref`)

Probe correct = `probes.score_exact`. Bands: beyond (207), within (744), within excluding diag27 (717). Paired gpu - on6ref
differences, trajectory-clustered 95% bootstrap (`a1_core.boot_ci`, seed 0), discordant counts, two-sided sign test.

- **WIN**: beyond lower bound > 0, both within lower bounds >= -2.0 pts, overall mean > 0.
- **HARM**: either within lower bound < -2.0 pts.
- **NULL**: otherwise.
- **Cost**: shippable as the default on WIN with the median `scorer` ms of the mode-4 lines <= 200 ms.
- Reported, not gating: answers equal to `../e2e/v2` (CPU scorer); prefill ms, decode tok/s, client wall (medians), pt, scorer
  ms median and max, the `gpu scorer:` stage medians, mechanism table.

## Predictions (written now)

- Scorer median 90-140 ms per reselect; max <= 450 ms (the first call per server process pays gallocr/cuBLAS setup).
- Prefill median 880-1050 ms (on6ref ~848); client wall median 1.42-1.60 s (on6ref ~1.40); decode tok/s within 3% of on6ref.
- Accuracy as the CPU-scorer arm within its noise: beyond 168-184 of 207, within 707-716 of 744, overall 878-895 of 951.
  Verdict WIN, cost line passes.
- Answers equal to the CPU-scorer v2 arm on >= 95% of probes (near-tie picks may flip a unit).
