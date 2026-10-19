# Packed pager (strided 2D page copies): 951-probe identity and cost, 3090

Binary `pager-cuda` md5 `f619149bbdcaf8c432acf99f758b4af9` (kvflash/2.0d-packed-pager `fead179b`), one binary for both arms; the legacy per-head copies selected with `LUCE_KVFLASH_PAGER_LEGACY=1`. ON6 flags (mode 4, k = 6, exact scorer, recency pool), QK16KND harness, 951 retained probes plus primers (1106 rows per arm), `run_full.sh`. Packed ran 2026-10-17 23:55-00:51, legacy 00:52-01:50, 0 errored rows each.

## Identity
- Answers: packed and legacy equal on 951/951 probes and on every primer, tool-call ids aside. `analyze_full.py` reports 950/951 against the rlru4k-v2 ON6 reference; the one row (`d005386703…:p12:0`, a 12K cut inside the pool) differs only in the random tool-call id, in both arms and against each other.
- Resident sets: 1106/1106 requests pooled in both arms, 0 mismatches against the OFF reference set log (`*.resident_cmp.txt`).
- Unit tests: 782/0 on CUDA at `fead179b`; on HIP (R9700, gfx1201, lucebox6) 765/765 plus the 5 race-test cases once the race test builds on HIP (`3b84a7a7`), so the 2D copies and zeroing run on AMD too.

## Cost (probe medians, `analyze_full.out`)
| | packed | legacy | paired packed - legacy |
|---|---|---|---|
| prefill ms | 658.3 | 812.8 | -134.4 |
| client wall ms | 1224 | 1345 | -116.0 |
| decode tok/s (all probes) | 42.2 | 44.2 | -1.9 |
| decode tok/s (probes with >= 32 tokens, n = 58) | 39.7 | 41.6 | -1.8 |
| reselect total ms | 753.5 | 915.2 | |
| copy calls per reselect (d2h / h2d / memset) | 3744 / 3328 / 3808 | 14976 / 13312 / 15232 | |

Prefill and wall gains come from the 4x fewer copy and memset calls (reset 105 -> 32 ms, page_submit 111 -> 30 ms, apply 119 -> 38 ms).

## Open: the decode gap
Packed decode is 1.9 tok/s (4.5%) lower, flat across both halves of the arm, and legacy matches the older ON6 reference (41.7 on the same 58 probes). But packed was the first arm, started two minutes after the nvcc build, without `drop_caches` / `compact_memory`, the same ~5% decode drop seen before on uncompacted boxes. Not attributed until the ABBA rerun (`run_ab.sh`: 303 probes from the 47 cut groups holding the 58 long probes, A1 packed, B1 legacy, B2 legacy, A2 packed, compaction before each block). The legacy switch stays until that result.
