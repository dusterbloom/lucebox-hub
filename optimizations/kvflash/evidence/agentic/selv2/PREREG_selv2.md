# Selector v2 derisk: preregistration

Written 2026-10-17 before any multi-query dump row exists. Exploratory work so far (offline_gran.out, rank_miss.out) used
the 1-query rank dump of 2026-10-16 (`../rlru/dump`); it is the reason for this run, not part of its result.

## Question

At v1's budget (60 chunks of 64 tokens = 6 blocks of 640), does scoring with more of the prompt's own queries, and/or
aggregating the (layer, head) planes by their largest values instead of the mean, hold the probe's source fact more often
than the shipped selector (last token, mean over 384 planes, 640-token blocks)?

Why: in the shipped ON6 run (rlru4k/v2 diag4_on6.out), 57 of 206 beyond-pool probes had no source block picked, and
ON6 answered 2 of them; with the source picked it answered 88-96%. Under the last token's query the missed facts rank
61-500 of the eligible chunks (rank_miss.out), and finer units alone recover at most 13 of the 61 dump misses (offline_gran.out).

## Data

- Binary: `/mnt/kimi-k3/build/selv2-cuda/luce_server`, branch `kvflash/2.0c-selector-v2` (md5 and commit in provenance_mq.txt).
  The change only adds `LUCE_KVFLASH_EXACT_DUMP_MQ=N`: the queries of the last N prompt tokens (gathered from every
  prefill batch holding any of them; R sits ~4 tokens before the end, so the final batch alone has ~4) are scored by
  the same exact scorer and written to `exact_dump_mq.jsonl` (per query: chunk mass averaged over the planes, over the 8
  largest planes, and the largest plane). No selection changes; answers must equal OFF's.
- Arm: `run_mq_dump.sh`, the same harness, task file, trajectories and retained list as `../rlru/run_dump.sh` (QK16KND,
  probes only, 16K pool, max-ctx 52224), with `LUCE_KVFLASH_EXACT_DUMP` and `LUCE_KVFLASH_EXACT_DUMP_MQ=64`.
- Windows: the first window chunk per probe comes from the shipped ON6 run (`rlru4k/v2/on6` server log), as in diag4.

## Checks before any variant is read

1. Row count of exact_dump_mq.jsonl equals the requests; no request errors.
2. The new `mass_end` equals the 2026-10-16 dump's `mass_end` per chunk within 1e-3 relative on every request (same
   binary semantics for the 1-query path), and the last row of the new `mean` equals the new `mass_end` within 1e-3.
3. Answers: the new arm's answers equal the 2026-10-16 dump arm's on at least 99% of probes (temp 0; the dump arm ran OFF).
If 1 or 2 fails the run is void.

**Amendment (2026-10-17 19:45 CEST, after 50 rows, before any variant was computed).** Checks 2 and 3 named the wrong
reference. The 2026-10-16 dump ran `prs-cuda`, an older engine whose OFF state is not prs8's: its answers equal the shipped
prs8 OFF arm (`rlru4k/v2/off`, a116ca38) on only 783/951 probes, and under the 16K pool KV depends on that history. The new
arm equals prs8 OFF on 57/57 shared probes. Checks now read:
2'. The last row of the new `mean` equals the new `mass_end` within 1e-3 relative on every request.
3'. Answers equal prs8 OFF (`rlru4k/v2/off`) on at least 99% of probes.
The old-dump comparison is printed for information only. v1's B count is recomputed from the new dump (it is a column of
the same table); "66 of 95" above is the old dump's. The ON6 windows come from prs8, so the new dump matches them better.

## Variants (fixed grid)

queries N in {1, 4, 16, 64} (last N prompt tokens; fewer only when the request prefilled fewer), query aggregation in {mean, max}
(N = 1: one), plane aggregation in {mean, top8, max}, unit in {10, 2, 1} chunks with k = 60 / unit, ranked by summed score,
eligible chunks [10, first window chunk). v1 = (1, mean, mean, 10).

## Metric and decision

Per out-of-window probe: fact fully held (all fact chunks in the selected set). Split by trajectory: A/B by sha1 of the
trajectory id mod 2 (A: 111 beyond, 59 within; B: 95 beyond, 39 within; v1 holds 66 of B's 95 beyond on the old dump).
The variant with the most beyond hits on A is chosen (ties: fewer queries, mean before top8 before max, larger unit).
On B, against v1, paired:

- **GO** (build the selector, then the end-to-end arm): beyond net gain >= +9 (about 30% of B's misses) with a two-sided
  sign test p < 0.05 on discordant probes, and within net >= -1.
- **KILL** (multi-query and plane aggregation do not fix selection; report and stop v2 as designed): beyond net <= +3.
- Otherwise **WEAK**: report; the end-to-end arm runs only if the chosen variant costs under 200 ms of scorer time.

The cost readout (scorer math ms at N = 1, 16, 64, from the `exact-scorer fast split` log lines) is reported alongside.
Fact coverage is a proxy: a GO here is a reason to build, and the claim comes from the end-to-end arm and S1.
