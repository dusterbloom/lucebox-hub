# E9: the exact-mass router (lucebox6, 2026-10-16)

This follows PREREG "Diagnostic E9: the exact-mass router". It decides nothing and changes no A1, A2 or A3 rule or bar.
- Run: `a1_launch_e9.sh`, from 02:26 to 02:45 CEST, on lucebox6 with `HIP_VISIBLE_DEVICES=1`. That is 19 minutes against the estimated hour.
- Try 1 (`try1_swapkill/`) was stopped by the watchdog during the model load: swap went from 472 to 1,509 MiB in under 3 minutes, before any row. The retry loaded the weights from the page cache, and swap stayed between 471 and 474 MiB. The GPU stayed within E3/E2's memory.
- Change: one more router, `x`. It is the attention mass the question puts on each block's exact 640 keys, with zero beta. `a` (the compacted slots with beta, E2) and `b` (kvflash_qk cosine, E2) are ranked on the identical state in the same pass. No answers are generated.
- Data: the same 6 pilot trajectories, 118 reference-correct probes and cuts. The state is E2's shipping state (window 0). The reference is the pilot's FULL rows, copied unedited (the md5 matches).
- Code: commit 0c1b5374; provenance.txt records the md5s. The report is the `E9_exact_mass_router` section of `../E3E2/e_report.json` (`a1_e_report.py ~/r1e/out/e3e2 ~/r1e/out/e5 ~/r1e/out/a1_pilot ~/r1e/out/e6 ~/r1e/out/e7 ~/r1e/out/e8 ~/r1e/out/e9`).

## Ranks over every sealed block (E2's ranking)

| Router | recall@1 | @2 | @4 | @8 | @16 | Median rank | Over 16K (n=18): @1, @4, @8 |
|---|---|---|---|---|---|---|---|
| a_last (E2) | 18.6% | 36.4% | 60.2% | 85.6% | 95.8% | 4 | 22.2%, 72.2%, 100% |
| a_max | 11.9% | 30.5% | 52.5% | 77.1% | 94.9% | 4 | 5.6%, 66.7%, 94.4% |
| **x_last** | **36.4%** | **56.8%** | **74.6%** | **87.3%** | **96.6%** | **2** | **61.1%, 94.4%, 100%** |
| x_max | 21.2% | 56.8% | 71.2% | 86.4% | 97.5% | 2 | 0%, 94.4%, 94.4% |
| b_last | 0% | 3.4% | 14.4% | 40.7% | 66.1% | 10 | 0%, 27.8%, 61.1% |
| b_max | 1.7% | 7.6% | 22.9% | 39.0% | 73.7% | 10 | 0%, 11.1%, 27.8% |

`a_last` reproduces E2's numbers on the same probes.

Paired per probe, `x_last` against `a_last`: better on 76, equal on 26, worse on 16. `x_max` against `a_max`: 78, 29, 11.

## Inside A3's fetch budget (candidates only, A3's k)

| W | Source is a candidate | Mean k | a_last | x_last | x_max | b_last |
|---|---|---|---|---|---|---|
| 2 | 106 | 8.81 | 98 | 99 | 100 | 68 |
| 6 | 66 | 4.81 | 55 | 58 | 57 | 26 |

## Reading

- By the preregistered reading, exact mass matches or beats compacted mass everywhere. It doubles recall@1 (36.4% against 18.6%) and nearly triples it beyond 16K (61.1% against 22.2%). So routing does not need the compacted index: no beta kernel and no value fit are needed for routing.
- Inside A3's budget the two are almost equal (99 against 98 of 106 at W=2), because k is already about 9.
- This ranks on E2's state, not A3's. On E2's state `a_last` would fetch 98 of 106 at W=2. In A3's own state it fetched 92 (A3 stage 1: 13 router misses). That loss is the router rank shift the peer's check found (`../A3/sel/peer_sim_check/`). E9 does not say whether `x_last` holds up in A3's state.
- Stage 2 of A3 runs with `a_last` as preregistered. An arm with an exact-mass router on A3's own state is the number that would decide whether the engine needs a compacted index at all.
