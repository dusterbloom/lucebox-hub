# Selector v2 end to end: preregistration

Written 2026-10-17 before any row of this run exists, after the offline derisk (`../PREREG_selv2.md`, GO: held-out
beyond fact coverage 66 -> 88 of 95, net +22, sign test p 3e-6; within 36 -> 39 of 39).

## Question

At the shipped budget (k=6: 60 chunks of 64 tokens beside the exact window), does selector v2 (score with the last 64
prompt queries of the first pass, max over queries of the mean of the 8 largest (layer, head) planes, picked in units of
2 chunks) answer more beyond-pool probes than the shipped ON6 selector (last query, mean over 384 planes, 10-chunk
blocks), without costing within-pool answers?

Out of scope here: "rewind only the last chunk on long turns" (doc B s4) and the GPU scorer.

## Arms (all on the 3090, harness `rlru4/agentic_eval/run_arm.py` QK16KND, probes only, retained list `rlru4/retained.json`,
16K pool, max-ctx 52224, temp 0, seed 1, one server at a time under `/mnt/kimi-k3/build/3090.lock`)

- **v2**: `/mnt/kimi-k3/build/selv2-cuda/luce_server` (commit 22b59a62, md5 497cbeca), env
  `LUCE_KVFLASH_PREFILL_RESELECT=4 LUCE_KVFLASH_RESELECT_SCORER=exact LUCE_KVFLASH_RECALL_BLOCKS=6
  LUCE_KVFLASH_SCORE_POOL_RECENT=1 LUCE_KVFLASH_LOG_RESIDENT=1 LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2`.
- **on6ref**: the same binary, the same env without the two SELECT flags (= shipped ON6), run after v2 in the same session.
  It is the reference for every number (same binary md5, same session). Until it finishes, `analyze_e2e.py` reads the
  shipped prs8 ON6 arm (`rlru4k/v2/on6`, a116ca38, 2026-10-16) instead and labels the verdict provisional.
- Read only: prs8 OFF (`rlru4k/v2/off`) for the structural check and the OFF column.

## Checks (a failed one stops the run; `run_e2e.sh`)

1. Shadow: flags off, the selv2 binary on `rlru4/smoke_ids4.json` picks exactly what prs8 ON6 picked on the same ids
   (`rlru4k/v2/smoke/on6`, every mode-4 reselect line), answers identically, and logs no selector v2 line.
2. Smoke and full v2 arm (`gate_e2e.py v2`): every mode-4 reselect logs `selector v2: N of 64` with N > 0, `picked units
   of 2 chunks`, 30 distinct picked units, all >= 5 (chunk 10), and one `re-prefill ... ok`; no failed reselect; a recency
   line per pooled request; no disk_hit.
3. Structural: on every request the resident set at the first prefill batch equals prs8 OFF's (`resident_cmp.py`, 0 mismatches),
   for the smoke (against `rlru4k/v2/smoke/off`), v2 and on6ref.
4. 951 of 951 probes on v2 and the reference, 0 errored rows, no disk_hit; median pt equal on v2 and the reference.

## Metric and decision (`analyze_e2e.py`, fixed with this file)

Probe correct = `probes.score_exact` on the answer. Bands as in PREREG_v2: beyond = source distance > 16384 tokens (207),
within (744), within excluding the 27 diag27 probes (717). Paired v2 - reference differences with the trajectory-clustered
95% bootstrap (`a1_core.boot_ci`, seed 0), plus discordant counts and a two-sided sign test.

- **WIN**: beyond lower bound > 0, within and within-excl-27 lower bounds >= -2.0 pts, overall mean > 0.
- **HARM**: either within lower bound < -2.0 pts. v2 as configured is dropped whatever beyond does.
- **NULL**: otherwise (coverage did not turn into answers; read the mechanism table).
- **Cost** (separate line): v2 is shippable as the default only on WIN with a median scorer time per reselect <= 200 ms.
  The CPU scorer is expected to fail this; a WIN then means "build the GPU scorer", not "ship".

Reported for both arms: prefill ms, decode tok/s and client wall (medians over probes), pt, scorer ms per reselect, H2D MiB
per reselect, and the mechanism table (out-of-window probes: fact fully held by the picks, partly, missed; answers per class).

## Predictions (written now)

- Reference (on6ref) reproduces prs8 ON6: >= 99% equal answers, 138 +- 3 beyond, 847 +- 5 overall.
- v2 beyond 170-185 of 207 (reference 138); within 709 +- 5 of 744; overall 878-895 of 951. Verdict WIN.
- Mechanism: v2 holds the whole fact on >= 175 of the 206 out-of-window beyond probes (offline, all beyond: 187 vs v1's 143).
- Scorer median 800-1400 ms per reselect for v2 against tens of ms for the reference; decode tok/s within 3%; prefill ms
  higher by about the scorer delta. Cost line fails.

**Amendment (2026-10-17 22:25 CEST, after the v2 arm, before any accuracy number was read).** Check 2 as coded matched
"failed" anywhere in a request block that also held "prefill reselect"; it flagged 48 requests whose only such line is the
server's `tool_call parse failed; suppressing buffered tool text` (primers emitting `<tool_call>`; the shipped ON6 log has
the same line at the same rate). Neither arm logs a `prefill reselect...failed` line. The check now matches
`prefill reselect.*failed` on one line, as run_v2.sh's gate_b does; the v2 arm passes it (827/827 reselects v2, 64 queries
on 822, 33-63 on 5 short prefills) and the structural check (0 mismatches). The run resumes at on6ref (`PHASES=on6ref`).
