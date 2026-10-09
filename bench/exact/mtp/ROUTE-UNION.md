# Routed-expert reuse across consecutive decode tokens (2026-10-09, lucebox4, Q8 GGUF)

300 greedy tokens per prompt after an 800-token prefill, 48 layers, top-10 of 512 experts.
r_w = mean |union of top-10 over a w-token window| / (10*w). Lower means more reuse.
Scratch build with env-gated `LUCE_QWEN_ROUTE_DUMP`; not in product code.

| w | code consecutive | code shuffled | prose consecutive | prose shuffled |
|---|---|---|---|---|
| 2 | 0.836 | 0.940 | 0.850 | 0.955 |
| 3 | 0.747 | 0.880 | 0.779 | 0.909 |
| 4 | 0.690 | 0.849 | 0.727 | 0.868 |
| 8 | 0.571 | 0.716 | 0.613 | 0.751 |

Per-layer r_4 (code): min 0.52, median 0.70, max 0.91. QSA layers reuse more (median 0.66) than GDN layers (0.71).

Read: the reuse is temporal (consecutive minus shuffled is 15-19 points at every w).
- Routed experts are ~7.3 ms per verify row.
- A union-grouped verify at w=4 reads ~0.69-0.73 x 4 rows' experts, which saves ~8-9 ms per window out of the ~24 ms verify gap to Strata.
- At w=8 it saves ~40%.
