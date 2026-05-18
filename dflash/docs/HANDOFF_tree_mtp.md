# Tree-MTP — Current State and Path Forward

## Where we are

`feat/mtp-tree-arena` (this branch) contains:

- Tree-runner skeleton with chain-equivalent B=1 path (byte-identical to chain runner).
- GPU arena allocation, per-path K/V routing, mirror-to-chain for canonical sibling, F32-cast composed FA read path.
- `verify_tree` now captures pre-norm hidden states (matching `verify_batch`).
- 8/8 orchestrator + 4/4 prefix-cache regression gates green.

## Bench numbers (standard harness, n_sample=5)

| Suite | B=1 chain | B=2 tree | ratio |
|-------|-----------|----------|-------|
| he    | 62.7      | 38.2     | 0.61  |
| gsm   | 56.0      | 35.8     | 0.64  |
| math  | 55.5      | 35.9     | 0.65  |
| agent | 54.2      | 44.0     | 0.81  |

Accept rate: chain 0.73, tree 0.33-0.48.

## Why tree underperforms chain on these suites

Tree at B=2 γ=2 proposes 7 tokens per verify (vs chain γ=3 proposing 4). At accept_rate ~0.40 per path tree nets ~1.4 tokens/iter; chain at 0.73 × 3 nets ~2.2. Tree overhead erases its diversity benefit on workloads where chain already accepts well — exactly the regime the adaptive shape selector is designed for.

## Default behavior

`DFLASH27B_MTP_TREE_B` defaults to 1 → chain runner → byte-identical to baseline. Tree mode is opt-in via env, intended as substrate for the adaptive shape selector to dispatch into on high-entropy iters.

## Open follow-ups (not blocking ship)

- Adaptive shape selector wiring (separate branch `feat/mtp-adaptive`): orchestrator dispatch site at `dflash/src/common/mtp_orchestrator.cpp:211` consumes selector decision per-iter.
- Tree mode tuning: explore γ=1 + B≥3 trade-off; current γ=2 B=2 may be suboptimal.
- Verify whether `step_batch_arena` per-path graph cache scales beyond B=2 (currently sized at sg_cache=8).
