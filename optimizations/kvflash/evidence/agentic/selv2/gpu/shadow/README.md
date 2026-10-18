# Selector v2 GPU scorer, shadow smoke (3090)

Binary: `/mnt/kimi-k3/build/selv2gpu-cuda/luce_server` (md5 20e33b31, commit 2f1247c0). Harness `run_arm.py QK16KND`, smoke ids `smoke_ids4.json`,
`--probes-only`, env: RESELECT=4, scorer exact, RECALL_BLOCKS=6, SCORE_POOL_RECENT=1, SELECT_QUERIES=64, SELECT_UNIT=2,
`LUCE_KVFLASH_SCORER_DEVICE=gpu`, `LUCE_KVFLASH_SCORER_SHADOW=1` (the CPU fast scorer also runs; `scorer shadow: ref=` is the CPU pick, `fast=` the GPU pick).
`summary.out` is `summarize_shadow.py` over `QK16KND.server.log`.

- reselects: 6; gpu scorer lines: 6; "unavailable" lines: 0
- gpu scorer total ms: median 82.3, max 153.9 (the first two calls, 147 and 154, carry the one-time pinned alloc / cuBLAS / kernel load: gather 57-63 ms); steady 77-84 ms
- picks (30 units each): identical in order 4 of 6; identical as a set 6 of 6 (worst overlap 30/30); the two ordered differences swap adjacent units inside the picked set (gap between 30th and 31st score block is 0.8-5.9%)
- select-score max relative error 1.0-1.75% (max abs 0.0009-0.0017) on real keys, which is above the 1e-2 of the synthetic test: the q8_1 activation quantization of the MMQ path moves the smallest scores most
- CPU scorer in the same runs: 1.67-1.98 s per reselect
