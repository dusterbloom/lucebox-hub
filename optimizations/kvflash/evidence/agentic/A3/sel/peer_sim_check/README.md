# A3 stage 1 against the offline A3 simulator, probe by probe (2026-10-16 02:40)

    python3 -I a3_vs_sim.py <worktree> 2   > W2.out
    python3 -I a3_vs_sim.py <worktree> 6   > W6.out

The script reads the committed A3/sel rows (b907debf) and the simulator in `A3_sim/a3_sim.py`, then puts each probe in a class twice: once as the simulator predicted it (all / ex / fetch / miss), once as measured.

## Where W=2's 99 predicted became 95 measured

| Cause | Probes | Net |
|---|---|---|
| Fully exact states (k = candidates, the 34 shortest-cut probes): predicted all correct, measured 31 | 34 | −3 |
| Simulator said fetched (E2 a_last rank 2–7), A3's router ranked the source 10–21 → miss, all wrong | 7 (5 predicted correct) | −5 |
| Simulator said miss (E2 rank 11–22), A3 ranked it 4–11 → fetched | 4 | +2 |
| Miss in both, answered anyway | 1 | +1 |
| Source exact in both, answer differs from E3's | 63 | +1 (4 gained, 3 lost) |
| **Total** | 118 | **−4** |

## Two findings

1. **E2's ranks do not carry over to A3's state.** These are the same probes and the same query rule.
   - On 7 probes the source moved from E2 rank 2–7 to A3 rank 10–21. On 4 other probes it moved the other way.
   - A candidate-only rank can't be worse than the all-blocks rank under the same query. So the query or the scored keys differ between E2's all-compacted state and A3's state, where P and W are exact.
   - The router is state-sensitive. E2's 85.6% recall at top 8 is not A3's. E9's exact-mass ranks must be computed on A3's state to be predictive.
2. **Fully exact states are not reproducible across W.** On all 34 probes at cuts 12–18, k equals the candidate count, so every block is exact under both W=2 and W=6.
   - W=2 scores 31/34 and W=6 scores 33/34.
   - Two probes flip between the two runs, both at cut 12:
     - `…85fa42f3bef6b39d:p12:7`: W=2 answers a different sentence;
     - `…07b5072b93b0bbdc:p12:3`: W=2 has "for creating" against "to create".
   - One probe is wrong in both: `…e135f99fb02cfcfc:p12:10`, a badge URL.
   - So either the instrument is nondeterministic, or the "exact" state depends on W. The second would be the case if blocks compacted during prefill change the KV and recurrent state of later tokens before the exact copy is fetched back.
   - Either way, about ±2–3 of 118 is noise at the instrument level. W=2 against W=6 (95 against 94) is a tie.
