# AM reference fixtures v2: what the reference fit sees, and the score floor the engine gates must allow

Fixtures are from `lucebox/a1/am_fixture_dump.py` at 80d2d0a5, run on lucebox6 on 2026-10-15 (2281 s). They are kept at `~/r1e/out/am_fixtures_v2` on lucebox6, with a copy in `/mnt/kimi-k3/build/am_fixtures_v2`. There are 4 cases: traj1_first, traj1_12k, traj1_12k_exactq and article_r1e.

## The fit sees 3839 of the 3840 grouped queries per key-value head

The first dump reported `q_group_check=False` on every case. The v2 dump also stores the queries the fit actually received (`fitq`), and its check reports `fit_queries_per_kv_head: [3839]` on every case.

A 640-token block with 6 query heads per key-value head gives 3840 grouped queries. The reference's query generator then:
1. computes its target count as `int(n_train * fraction * min_scaling)`, which truncates to 3839;
2. drops one query at random (`randperm`, sorted).

The dropped index is shared by all layers and heads. This comes from `compaction/query_generation/query_generator.py` in the reference, read and not run.

The layout itself is correct: the head-major groups that `_attention_to_kv` builds match.

Consequence: parity checks of the scores and the fit (M4, M5) must use `fitq`. M3, which checks query capture, compares against `q` (all 640 tokens by 24 heads).

## The reference's scores come from bf16 logits

The reference computes `(queries @ K.T)` on bf16 inputs, so the logits are the bf16 matmul output. Softmax and the RMS aggregation then run in fp32.

`lucebox/a1/fixture_qcheck.py` replays the scores from `fitq` and `k`. It has two modes: float64 logits (exact math, the default) and emulated bf16 logits (`bf16` argument).

Normalized error is max |replayed − reference| / max |reference| per head. Overlap is the replayed top-64 against the reference `idx`. Each case has 64 heads (16 layers by 4 key-value heads).

| Case | Logits | Normalized error median | p90 | max | Overlap min per head | Overlap mean | Heads fully equal |
|---|---|---|---|---|---|---|---|
| traj1_first | float64 | 4.94e-3 | 7.19e-3 | 8.64e-3 | 0.969 | 0.9951 | 45/64 |
| traj1_first | bf16 emulated | 6.38e-5 | 1.29e-3 | 5.62e-3 | 1.000 | 1.0000 | 64/64 |
| traj1_12k | float64 | 3.78e-3 | 6.31e-3 | 9.37e-3 | 0.984 | 0.9954 | 45/64 |
| article_r1e | float64 | 5.09e-3 | 7.51e-3 | 1.06e-2 | 0.969 | 0.9949 | 45/64 |

So with exact math the scores sit about 1e-2 (normalized) from the reference, and the top-64 differs by up to 2 keys per head. That gap is the reference's bf16 logits, not an error in the math.

An engine that computes in F32 cannot match the reference's scores to 1e-3. Its gates therefore bind on two things:
- agreement with exact math on the same inputs, at 1e-3;
- agreement with the reference's selection within this floor: overlap of at least 99% per case and at least 62/64 per head, with each swapped key within 2e-2 times the head max of the 64th score.
