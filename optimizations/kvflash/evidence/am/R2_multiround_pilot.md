# R2 multi-round pilot (2 probe articles, k in {1, 8})

Pilot only. No pre-registered question (Q1, Q2) is answered here; N = 33 questions over 2 articles. Rows: `R2_multiround_pilot/*.jsonl`. Table produced by `modal/am_qwen35/r2_report.py`.

## Setup
- Probe articles 0 (quality_52845, 14 q) and 1 (quality_30029, 19 q). Distractors fixed by seed (art 0: 51,76,72,61,78,67,58,55; art 1: 71,58,66,77,72,79,56,75). One pass per (article, arm), nested checkpoints give k=1 and k=8.
- GPU: NVIDIA H200 for every reported row. A100-80GB was tried first (smoke at k=1 only, $0.17) and ran out of memory in F-RE (`compaction/algorithms/base.py:168`); H100 is also 80 GB, so all arms were moved to H200. F-RE peaks at 89.2 GB at k=8.
- Slots: "tensor" includes per-head padding to the layer maximum (zero K/V, beta -inf); "effective" is the real per-head slot count. Tensor is about 1.6x effective for F-SEG/F-BLK/F-RE.

## Results (means over the 2 articles for time and memory; counts summed)
| arm | k | correct / N | ctx tokens | slots tensor / effective | prefill s | compact s | answer s | peak GB |
|---|---|---|---|---|---|---|---|---|
| FULL | 1 | 29/33 | 13634 | 13634 / 13634 | 33.5 | 0 | 20.8 | 53.8 |
| FULL | 8 | 29/33 | 46247 | 46247 / 46247 | 12.8 | 0 | 4.1 | 61.9 |
| F-SEG | 1 | 28/33 | 13634 | 2292 / 1389 | 8.4 | 17.0 | 2.5 | 62.0 |
| F-SEG | 8 | 29/33 | 46247 | 7742 / 4677 | 31.5 | 33.5 | 2.7 | 71.6 |
| F-BLK | 1 | 29/33 | 13634 | 2693 / 1819 | 2.1 | 4.1 | 3.4 | 51.8 |
| F-BLK | 8 | 30/33 | 46247 | 8084 / 5037 | 7.7 | 14.4 | 3.6 | 53.9 |
| F-RE | 1 | 10/33 | 13634 | 1796 / 1079 | 0.9 | 15.5 | 76.0 | 77.5 |
| F-RE | 8 | 0/33 | 46247 | 1918 / 1152 | 5.0 | 166.4 | 250.6 | 89.2 |
| F-RE-UNI | 1 | 23/33 | 13634 | 791 / 791 | 10.2 | 25.1 | 7.2 | 65.2 |
| F-RE-UNI | 8 | 0/33 | 46247 | 561 / 561 | 25.8 | 144.3 | 284.3 | 81.2 |

Prefill/compact/answer seconds are the per-step values logged at each k (not cumulative across arms). Long F-RE and F-RE-UNI answer times at k=8 come from degenerate (gibberish) generations running to the token limit. F-RE outputs at k=1 and k=8 were gibberish on both articles; F-RE-UNI (uniform budget, Amendment 1) was coherent at k=1 on article 1 (15/19) and collapsed at k=8.

## How the repo's online loop treats previously compacted tokens
- `per_layer_head_on_policy.py:316-317, 476, 487-492, 512-513, 572, 590`; `run_reasoning_evaluation.py:191, 367, 538-539, 546, 552, 566, 924` (`--max-compactions`); `per_layer_head.py:141, 192, 496, 501, 505, 525, 575`; `registry.py:146, 159-160, 168`.
- The loop recompacts the whole current cache (earlier compacted tokens included) on each trigger. The repo's own method (`PerLayerHeadCompaction`) carries the previous beta; setup F's on-policy wrapper drops it on re-compaction (this is what F-RE inherits, no code patched). F-RE-beta is not available without code changes and was not run. Details and padding behaviour: PREREG "R2 Amendment 1".

## Cost (measured, H200 at $4.54/h, model load about 31 s per job)
- Pilot jobs: probe 0, 4 arms, 622 s, $0.78; probe 1, 4 arms, 821 s, $1.04; F-RE-UNI probes 0,1, 995 s, $1.25. Total $3.07, plus an A100 smoke and OOM attempts (smoke $0.17); the pilot stayed well under the $10 cap.
- F-RE (collapsed) is about 75% of the 4-arm job time: roughly 600 s of 790 s for probe 1; FULL+F-SEG+F-BLK together about 180 s.
- Projection for the full design (ks 1,2,4,8, arms FULL, F-SEG, F-BLK, F-RE; F-RE-UNI is a diagnostic, excluded):
  - FULL+F-SEG+F-BLK: about 230 s per article (the extra ks add answering only).
  - F-RE: about 860 s per article while it still generates degenerate text (compaction about 190 s per article plus about 670 s answering at k=1,2,4,8); about 250 s if answers stop running to the token limit.
  - 20 articles: about $28 with F-RE as measured ($6 without F-RE). 10 articles: about $14 ($3 without F-RE). Each job adds about $0.04 of load time.
