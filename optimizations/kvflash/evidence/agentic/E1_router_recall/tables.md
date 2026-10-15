
### Hit = fact span (primary); query (a) last prefill token's query

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 8/207 (3.9%) | 19/207 (9.2%) | 49/207 (23.7%) | 94/207 (45.4%) | 135/207 (65.2%) |
| within 16K | 744 | 77/744 (10.3%) | 168/744 (22.6%) | 379/744 (50.9%) | 545/744 (73.3%) | 689/744 (92.6%) |
| all 951 | 951 | 85/951 (8.9%) | 187/951 (19.7%) | 428/951 (45.0%) | 639/951 (67.2%) | 824/951 (86.6%) |
| chance (all 951) | | 2.4% | 4.9% | 9.7% | 19.4% | 38.9% |
| chance (older than 16K) | | 1.4% | 2.9% | 5.7% | 11.4% | 22.8% |

### Hit = fact span (primary); query (b) max over the question's prefill tokens

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 0/207 (0.0%) | 2/207 (1.0%) | 14/207 (6.8%) | 39/207 (18.8%) | 84/207 (40.6%) |
| within 16K | 744 | 41/744 (5.5%) | 104/744 (14.0%) | 221/744 (29.7%) | 409/744 (55.0%) | 612/744 (82.3%) |
| all 951 | 951 | 41/951 (4.3%) | 106/951 (11.1%) | 235/951 (24.7%) | 448/951 (47.1%) | 696/951 (73.2%) |
| chance (all 951) | | 2.4% | 4.9% | 9.7% | 19.4% | 38.9% |
| chance (older than 16K) | | 1.4% | 2.9% | 5.7% | 11.4% | 22.8% |

### Hit = literal PREREG rule: [src_tok, src_tok + gold length]; query (a) last prefill token's query

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 10/207 (4.8%) | 26/207 (12.6%) | 64/207 (30.9%) | 128/207 (61.8%) | 183/207 (88.4%) |
| within 16K | 744 | 51/744 (6.9%) | 155/744 (20.8%) | 420/744 (56.5%) | 609/744 (81.9%) | 701/744 (94.2%) |
| all 951 | 951 | 61/951 (6.4%) | 181/951 (19.0%) | 484/951 (50.9%) | 737/951 (77.5%) | 884/951 (93.0%) |
| chance (all 951) | | 2.4% | 4.9% | 9.7% | 19.4% | 38.9% |
| chance (older than 16K) | | 1.4% | 2.9% | 5.7% | 11.4% | 22.8% |

### Hit = literal PREREG rule: [src_tok, src_tok + gold length]; query (b) max over the question's prefill tokens

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 1/207 (0.5%) | 1/207 (0.5%) | 9/207 (4.3%) | 44/207 (21.3%) | 108/207 (52.2%) |
| within 16K | 744 | 44/744 (5.9%) | 147/744 (19.8%) | 321/744 (43.1%) | 490/744 (65.9%) | 645/744 (86.7%) |
| all 951 | 951 | 45/951 (4.7%) | 148/951 (15.6%) | 330/951 (34.7%) | 534/951 (56.2%) | 753/951 (79.2%) |
| chance (all 951) | | 2.4% | 4.9% | 9.7% | 19.4% | 38.9% |
| chance (older than 16K) | | 1.4% | 2.9% | 5.7% | 11.4% | 22.8% |

### Hit = whole source message; query (a) last prefill token's query

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 44/207 (21.3%) | 101/207 (48.8%) | 158/207 (76.3%) | 196/207 (94.7%) | 206/207 (99.5%) |
| within 16K | 744 | 264/744 (35.5%) | 509/744 (68.4%) | 672/744 (90.3%) | 727/744 (97.7%) | 741/744 (99.6%) |
| all 951 | 951 | 308/951 (32.4%) | 610/951 (64.1%) | 830/951 (87.3%) | 923/951 (97.1%) | 947/951 (99.6%) |
| chance (all 951) | | 2.4% | 4.9% | 9.7% | 19.4% | 38.9% |
| chance (older than 16K) | | 1.4% | 2.9% | 5.7% | 11.4% | 22.8% |

### Hit = whole source message; query (b) max over the question's prefill tokens

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 7/207 (3.4%) | 10/207 (4.8%) | 52/207 (25.1%) | 152/207 (73.4%) | 196/207 (94.7%) |
| within 16K | 744 | 218/744 (29.3%) | 393/744 (52.8%) | 603/744 (81.0%) | 719/744 (96.6%) | 744/744 (100.0%) |
| all 951 | 951 | 225/951 (23.7%) | 403/951 (42.4%) | 655/951 (68.9%) | 871/951 (91.6%) | 940/951 (98.8%) |
| chance (all 951) | | 2.4% | 4.9% | 9.7% | 19.4% | 38.9% |
| chance (older than 16K) | | 1.4% | 2.9% | 5.7% | 11.4% | 22.8% |

### Rank distribution, fact span, variant last

| subset | n ranked | 1-1 | 2-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65-128 | 129+ | none | median | q25 | q75 | median rank / n_chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 0 | 2 | 6 | 11 | 30 | 45 | 41 | 72 | 0 | 74 | 34 | 172 | 0.132 |
| within 16K | 744 | 1 | 33 | 43 | 91 | 211 | 166 | 144 | 55 | 0 | 32 | 18 | 66 | 0.110 |
| all 951 | 951 | 1 | 35 | 49 | 102 | 241 | 211 | 185 | 127 | 0 | 36 | 19 | 81 | 0.112 |

### Resident at prefill end and recall over the non-resident probes, fact span, variant last

| subset | n | resident at end (or in tail) | non-resident n | nonres recall@8 | nonres recall@16 | nonres recall@32 | nonres recall@64 | nonres recall@128 |
|---|---|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 12 | 195 | 8/195 (4.1%) | 19/195 (9.7%) | 46/195 (23.6%) | 86/195 (44.1%) | 126/195 (64.6%) |
| within 16K | 744 | 744 | 0 | - | - | - | - | - |
| all 951 | 951 | 756 | 195 | 8/195 (4.1%) | 19/195 (9.7%) | 46/195 (23.6%) | 86/195 (44.1%) | 126/195 (64.6%) |

### Rank distribution, fact span, variant max

| subset | n ranked | 1-1 | 2-4 | 5-8 | 9-16 | 17-32 | 33-64 | 65-128 | 129+ | none | median | q25 | q75 | median rank / n_chunks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 0 | 0 | 0 | 2 | 12 | 25 | 45 | 123 | 0 | 173 | 75 | 296 | 0.305 |
| within 16K | 744 | 2 | 13 | 26 | 63 | 117 | 188 | 203 | 132 | 0 | 57 | 27 | 103 | 0.188 |
| all 951 | 951 | 2 | 13 | 26 | 65 | 129 | 213 | 248 | 255 | 0 | 69 | 33 | 135 | 0.204 |

### Resident at prefill end and recall over the non-resident probes, fact span, variant max

| subset | n | resident at end (or in tail) | non-resident n | nonres recall@8 | nonres recall@16 | nonres recall@32 | nonres recall@64 | nonres recall@128 |
|---|---|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 12 | 195 | 0/195 (0.0%) | 2/195 (1.0%) | 13/195 (6.7%) | 38/195 (19.5%) | 80/195 (41.0%) |
| within 16K | 744 | 744 | 0 | - | - | - | - | - |
| all 951 | 951 | 756 | 195 | 0/195 (0.0%) | 2/195 (1.0%) | 13/195 (6.7%) | 38/195 (19.5%) | 80/195 (41.0%) |
