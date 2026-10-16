# PREREG: prefill-end reselect, end to end on the 3090

Written 2026-10-16 00:10 CEST, before any row of this run exists.

## Arms

Same binary for both: `kvflash/2.0r-prefill-reselect`, built at `/mnt/kimi-k3/build/prs-cuda`. The md5s of the binary and of every ggml .so go into `provenance.txt`.

- **OFF**: arm `QK16KND` of the harness snapshot in `agentic_eval/`, with the B0 POOL16K flags:
  - `--kvflash 16384`, `KVFLASH_STATE_LADDER=1`, `KVFLASH_LIVE_RESTORE=1`;
  - `--max-ctx 52224`;
  - prefix cache 12288 MiB, 4 slots;
  - temperature 0, seed 1, the B0 GGUF and template;
  - plus `LUCE_KVFLASH_POLICY=qk`, no dump.
- **ON**: OFF plus `LUCE_KVFLASH_PREFILL_RESELECT=1`.

Both run `--probes-only` over the 951 retained B0 probes (`retained.json`, md5 c77c89018f683acaf262beafde5908c1), with the harness primers. The order is run_arm.py's. One server per arm, under `flock /mnt/kimi-k3/build/3090.lock`, GPU idle before each arm.

## Metric

**Retention** is the share of the 951 probes answered correctly. Every retained probe is one full KV answered correctly, so retention is measured against full KV. Scoring uses the harness `probes.score_exact`.

Intervals use the trajectory-clustered bootstrap: `a1_core.boot_ci`, 10,000 resamples, seed 0, 95%, clusters = trajectory id. Paired differences are bootstrapped as per-probe ON - OFF values with the same function.

## Read rules

1. **Primary.** ON - OFF overall retention, paired. ON wins if the 95% lower bound is above 0.
2. **Beyond 16K** (distance_tok > 16384, 207 probes). ON - OFF, paired. A win needs a 95% lower bound above 0.
   - Also reported: the 17 old-fact probes shared with E3 and POOL16K (POOL16K 1/17, E3 15/17).
3. **Within-16K guard** (744 probes). ON's within-16K retention may be at most 2.0 points below OFF's (point estimate, at most 15 probes). If the drop is larger, ON does not count as a win, whatever rules 1 and 2 say.
4. **External bar.** B0 POOL16K's 719/951 = 75.6%. ON clears the bar if its point estimate is at least 75.6% and rules 1 and 3 hold.
   - OFF is the calibration. The probes-only replay differs from B0's full sequence: E1 measured within 16K 688/744 against B0's 707/744.
   - If OFF lands below 719, the result is stated as both the measured ON and ON - OFF. It is never re-based.
5. **Prediction** (from `../prefill_reselect_sim/`): old facts 12 → 60 to 140 of 207; within 16K at least 690/744; overall above 719/951.
6. **Speeds.** Medians of prefill ms, decode tokens per second and client wall, for both arms. There is no speed gate. Flag it if ON's prefill median is more than 1.5 times OFF's.
7. **Caching parity.** On both arms, report the medians of `cached_tokens`/`prefilled_tokens` (pt) and the prefix length. Report `disk_hit` if the rows carry it. If pt differs between arms by more than 10% at the median, the speed comparison is void and is reported as void. The retention comparison stands.

## Stop conditions

- Preflight (added 00:35, still before any row): 5 ON probes (`smoke_ids.json`, from `pick_smoke.py`) go to `smoke/` and are not counted. If the log shows no reselect with page events, or any row has an error, the arms do not run.

- A server crash, or more than 5 errored probes in an arm: stop, report, no verdict.
- If the two arms together run past 2.5 hours, release the lock between arms and tell kvflash-am-a0.
