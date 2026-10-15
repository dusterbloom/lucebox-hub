# A3 policy simulation (offline, 2026-10-15, before any A3 row)

This is an estimate, not a measurement. No GPU and no new model calls. It recombines rows that already exist:
- E2 router ranks (`a_last`);
- E3 oracle answers;
- the pilot shipping answers;
- B0 POOL16K.

It is written down before A3 stage 1 runs, so A3 can check it.

    python3 -I a3_sim.py ../E3E2 ../A1/pilot ../B0_3090/POOL16K.jsonl /mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl

## Model

For each of the 118 pilot probes, the cut's state follows the A3 row budget (PREREG "A3 routed exact fetch"):
- P: the system/tool prefix, 10 blocks;
- W: the recent window;
- every other sealed block is compacted to 64 slots;
- k: fetched blocks, which fill the 16,384 rows after the question and answer reserve.

The predicted answer for each probe:
- **Every block exact** (k reaches all candidates): correct, because the state equals the reference.
- **Source block in P, in W or in the open tail, or in the router's top k**: E3's answer (the source exact, everything else compacted).
- **Otherwise**: the shipping arm's answer.

E2 ranked all sealed blocks, while A3 ranks only the candidates, so the true candidate rank is unknown. It lies between the E2 rank ("lo", pessimistic) and the E2 rank minus the exact blocks ("hi").

## Estimate

| Policy | Est. correct of 118, lo-hi | 0-4K | 4-8K | 8-16K | >16K | mean k |
|---|---|---|---|---|---|---|
| Shipping, all compacted (measured) | 10 (8.5%) | 2/41 | 3/26 | 3/33 | 2/18 | - |
| E3 oracle (measured) | 99 (83.9%) | 36/41 | 21/26 | 27/33 | 15/18 | - |
| **A3 W=2** | **99-104 (83.9-88.1%)** | 41/41 | 20/26 | 23/33 | 15/18 | 8.8 |
| A3 W=6 | 93-104 (78.8-88.1%) | 41/41 | 17/26 | 21/33 | 14/18 | 4.8 |
| hybrid, k=2, window fills the rest | 80-104 (67.8-88.1%) | 41/41 | 19/26 | 11/33 | 9/18 | 1.4 |
| hybrid, k=4, window fills the rest | 89-104 (75.4-88.1%) | 41/41 | 18/26 | 18/33 | 12/18 | 2.8 |
| hybrid, k=8, window fills the rest | 99-104 (83.9-88.1%) | 41/41 | 20/26 | 23/33 | 15/18 | 5.7 |
| no GPU index, W=2 (an exact-key router, ranked here by the `a_last` ranks as a stand-in) | 99-104 (83.9-88.1%) | 41/41 | 20/26 | 23/33 | 15/18 | 10.6 |
| no GPU index, W=6 (same stand-in) | 99-104 (83.9-88.1%) | 41/41 | 20/26 | 23/33 | 15/18 | 6.6 |

The distance columns use "lo". POOL16K measured on the 105 of these probes it has rows for: 84 (80.0%), with 1 of 17 beyond 16K.

## Reading

- **Prediction for A3 stage 1:** W=2 retains 84-88% and W=6 retains 79-88%. Both are above POOL16K's 80% on the same probes.
- **Fetch beats window.** Spending rows on fetched blocks beats spending them on a recent window. At equal rows, fewer fetches lose probes 8-16K and beyond 16K.
- **Where W=2's 19 misses come from** (pessimistic ranks, `chk_breakdown.py`):
  - 34 probes sit at cuts where every block is exact: 34 correct.
  - 6 have the source in P, W or the tail: 6 correct.
  - 67 have the source fetched: 59 correct (88%), so 8 misses with the source exact.
  - 11 are routing misses: 0 correct.
  
  Dropping the GPU index frees about 2 more fetches, which turns 2 routing misses into fetches that E3 also got wrong. A better router could win back up to the 11 routing misses, about 92% at E3's 88% conditional rate. The rest needs more than the source block.
- **Where the estimate is weakest:**
  - it assumes the fetched source answers as in E3;
  - it scores the 0-4K bin 41/41 because the 5 near probes E3 missed sit at 11-12K cuts, where A3's whole state is exact (checked);
  - it covers only 6 trajectories.
