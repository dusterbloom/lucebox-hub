# Selector v2 derisk: multi-query exact dump (3090, 2026-10-17)

Question, data, checks and decision rule: `PREREG_selv2.md` (one amendment, made before any variant was computed).
Binary: `kvflash/2.0c-selector-v2` 7c9180a9, luce_server md5 0c7c598b (`provenance_mq.txt`).

## Result: GO

Run valid (`prereg_checks.out`): 1106/1106 requests with a dump row and an mq row, last mq row = mass_end exactly,
answers = prs8 OFF on 950/951 probes, 64 queries on 1085/1106 requests.

Fact fully held at v1's budget (60 chunks), `offline_mq.out`. Chosen on split A: 64 queries, max over queries, mean of
the 8 largest planes, 2-chunk (128-token) units. Held-out split B against v1:

| band | n | v1 | v2 | net | gains / losses | sign test p |
|---|---|---|---|---|---|---|
| beyond pool | 95 | 66 | 88 | +22 | 23 / 1 | 3.0e-6 |
| within band | 39 | 36 | 39 | +3 | 3 / 0 | 0.25 |

All 206 beyond-pool probes: v1 143, v2 187. Averaging over queries hurts at every N; the gain needs max over queries and
appears at N = 64 (N = 4 and 16 do not beat v1), so the signal sits in the last user turn's content tokens, not the
template tail.

Cost (`cost_readout.py` on the server logs, i7-12700K): 0.66 ms per query per 1K sealed tokens batched (0.985 for one
query). At the median 24K sealed tokens: N = 1 24 ms, N = 16 ~260 ms, N = 64 ~1.0 s (1.4 s at 33K). Shipping v2 needs
the scorer on the GPU; the end-to-end arm measures quality first on the CPU scorer.

Fact coverage is a proxy. The claim comes from the end-to-end arm (v2 selection in mode 4 against ON6).

## Files

- `offline_gran.py`, `rank_miss.py` (+ `.out`): exploratory, on the 2026-10-16 dump (`prs-cuda`, an older OFF state).
- `run_mq_dump.sh`, `run_mq_dump.log`, `provenance_mq.txt`: the arm.
- `prereg_checks.py`, `offline_mq.py`, `cmp_answers.py`, `cost_readout.py` (+ `.out`): checks and analysis.
- `dump/QK16KND.{jsonl,meta.json,server.log}`: answers and server log. The raw dumps (746 MB) stay at
  `/mnt/kimi-k3/split-work/selv2/dump/exact_dump/`, md5 in `raw_dumps.md5`.
