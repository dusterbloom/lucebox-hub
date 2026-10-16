# Prefill-reselect pool diagnostic (2026-10-16 05:01–05:22, 3090, not preregistered, no verdict)

Binary `/mnt/kimi-k3/build/prs-cuda/luce_server`, md5 0b5920a0544572f39608402ed4824db0, branch kvflash/2.0r-prefill-reselect c4ab7bf8.

The run covers 67 probes (`ids.json`, from `pick.py`): the 9 (trajectory, cut) groups with the most within-16K losses in the failed run, plus the top group for old-fact gains.

Four arms, all with `KVFLASH_DEBUG_RESIDENT=1`, so the log holds the engine's actual pool when the answer starts:

| Arm | Reselect | Query token | Protected recent chunks |
|---|---|---|---|
| off | none | — | — |
| m1 | at the last turn cut, as in the failed run | end − 5 | — |
| m2 | at end − 1 | end − 2 | — |
| m2t128 | as m2 | end − 2 | last 129 chunks (8K) |

    python3 -I compare.py <worktree>      # actual pool vs the replay (E1 scores_last), per arm
    python3 -I rlru_est.py <worktree>     # estimate for recent window + k routed blocks, from E9 ranks

## Results

| Arm | Within 16K correct (54) | Fact chunk resident (actual / replay) | Whole message resident (actual / replay) | Old correct (13) |
|---|---|---|---|---|
| off | 52 | 54 / 54 | 54 / 54 | 0 |
| m1 | 21 | 39 / 52 | 10 / 15 | 10 |
| m2 | 20 | 25 / 52 | 5 / 15 | 11 |
| m2t128 | 31 | 33 / 50 | 24 / 32 | 6 |

The engine's actual pool when OFF matches the replay exactly (Jaccard 1.00).

### Answer rate by what is resident at answer time

Engine-measured, pooled over the 4 arms:

| Resident at answer time | Within 16K | Beyond 16K |
|---|---|---|
| Whole source message | 88/93 = 95% | 17/21 = 81% |
| Fact chunk ±2 neighbours | 20/28 = 71% | — |
| Fact chunk only | 16/30 = 53% | 10/12 = 83% |
| Nothing from the source | 0/65 | 0/19 |

## Reading

1. **The cosine router's choice depends on which single query token it uses.**
   - With the end − 5 query (m1) the engine kept the fact chunk for 39 of 54 recent facts; with end − 2 (m2), only 25.
   - The replay, with the last token's query, keeps 52.
   - Moving the reselect to end − 1 did not help; it made things worse.
2. **Answers need the source message, not just its chunk.** Recent facts score 95% with the whole message resident and 53% with only the fact chunk. Scattering 64-token chunks across the history, which is what chunk-level reselect does, loses this.
3. **Protecting 8K of recent context helps recent facts (20 → 31 of 54) but not enough, and it halves old facts.** Old facts fall because ~127 recall slots are too few for this router.

## The next design, estimated

The design keeps a recent exact window and recalls k whole 640-token blocks routed by exact attention mass, with no compaction.

The estimate comes from E9's per-probe ranks (`rlru_est.py`; ranks over all blocks, which is pessimistic). On the 105 E9 probes in this run's 951, OFF scores 86:

| k | Window | a_last | x_last | x_max |
|---|---|---|---|---|
| 0 | 15,984 | 83.0 | 83.0 | 83.0 |
| 2 | 14,704 | 88.7 | 92.7 | 93.5 |
| 4 | 13,424 | 91.3 | **95.4** (old 13.8/17) | 94.6 |
| 8 | 10,864 | 90.7 | 90.7 | 90.7 |

Caveat: E9 ranked on E2's all-compacted state. The router query in this design comes from a different state, so the ranks may shift, as they did from E2 to A3.
