# R-LRU, end to end on the 3090 (PREREG.md, amendments 1 and 2): FAILED

Arms ran 08:36–10:31 on 2026-10-16. Binary prs5-cuda, md5 6884c4fd682b6b3b8de2e40a90188488 (engine dfc5a3db). Mode 3, k = 6, exact-mass router.

    python3 -P analyze.py <worktree>   # preregistered read rules -> report.json
    python3 -I diag.py <worktree>      # post-hoc, not preregistered: per-probe recall class from the ON log's own picks

## Smokes

- **(a) Rewind check:** 6 rewinds, all with 0 page events and rerun logits max|diff| = 0. Flip rate vs OFF: 1/5 (p36:7, the 1-token final batch).
- **(b) ON run:** each request recalled 6 blocks, with 24–48 page events and a scorer time of about 0.5 s.

## Result (preregistered)

| Subset | OFF | ON | ON − OFF, 95% clustered CI | Predicted ON |
|---|---|---|---|---|
| Overall (951) | 700 | 692 | −0.8 pts [−3.4, +1.6] | 795 |
| Within 16K (744) | 688 | 649 | −5.2 pts [−7.9, −2.7] | 678 |
| Beyond 16K (207) | 12 | 43 | +15.0 pts [+9.6, +20.6] | 117 |

Verdict by rule:
- rule 1 (primary): false;
- rule 2 (beyond 16K): true;
- rule 3 (within-16K guard): false, a 5.2-point drop against a 2.0 limit;
- rule 4 (bar): false, 692 < 719.

Speed medians, OFF vs ON:

| | OFF | ON |
|---|---|---|
| Prefill | 451 ms | 1,008 ms |
| Decode | 41.1 tok/s | 41.1 tok/s |
| Client wall | 1.01 s | 1.63 s |
| Scorer (CPU) | — | 446 ms |

- Rule 6 flag: true. That is the CPU scorer.
- Caching parity: pt 349 on both arms, disk_hit 0.

## Why: post-hoc (`diag.py`)

| Band | Class | n | OFF | ON |
|---|---|---|---|---|
| within | in window | 647 | 598 | 610 (94%) |
| within | whole source message recalled | 39 | 36 | 20 (51%) |
| within | part of message recalled | 33 | 29 | 13 (39%) |
| within | missed | 25 | 25 | 6 |
| beyond | whole source message recalled | 54 | 0 | 18 (33%) |
| beyond | part of message recalled | 80 | 11 | 25 (31%) |
| beyond | missed | 72 | 1 | 0 |

1. **Routing is not the main loss.** On ON's own state, the engine's picks hold the source block for:
   - 134 of 206 old facts (the dump ranked 145 in its top 6 on OFF's state);
   - 72 of 97 out-of-window recent facts (dump: 91).
2. **A recalled block barely helps, even with the whole source message resident.**
   - Old facts score 33%, against the 81% the estimate assumed.
   - Recent facts recalled whole score 51%, where OFF's contiguous window gets 92%.
3. **The likely cause is timing.** Mode 3 reselects after the question's ~350 tokens are prefilled, so their K/V were computed without the fact. Only the final token and the decode attend over the recalled blocks.
   - The 81% came from the pool diagnostic, where the source was resident while the question was prefilled.
4. **Inside the window, ON beats OFF**: 610 vs 598 of 647.

## Next design

Mode 4 rewinds the whole turn:
1. prefill the question;
2. score with its last token;
3. restore the state at the turn cut;
4. reselect: window + k blocks;
5. re-prefill the question over the new pool.

The cost is one more tail prefill, about 0.35 s. The first pass and the re-prefill use the same batch plan, so the no-op rewind check is bit-exact, with no 1-token split.
