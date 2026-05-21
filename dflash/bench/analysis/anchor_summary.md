# Anchor Coverage Summary

Total cases analyzed: **115**

## Overall Zero-Hit Rates

| N-gram | Zero-hit cases | % of total |
|--------|---------------|------------|
| 2-gram | 0 | 0.0% |
| 4-gram | 13 | 11.3% |
| 6-gram | 25 | 21.7% |

## 4-Gram Zero-Hit Rate by Task x Context

| Task | Ctx | Total | Zero-hit | Zero% |
|------|-----|-------|----------|-------|
| fwe | 8192 | 5 | 0 | 0.0% |
| niah_single | 1024 | 40 | 0 | 0.0% |
| niah_single | 2048 | 20 | 0 | 0.0% |
| niah_single | 4096 | 5 | 0 | 0.0% |
| niah_single | 8192 | 5 | 0 | 0.0% |
| niah_single | 16384 | 5 | 0 | 0.0% |
| niah_single | 32768 | 5 | 0 | 0.0% |
| niah_single | 65536 | 5 | 0 | 0.0% |
| vt | 4096 | 5 | 3 | 60.0% |
| vt | 8192 | 5 | 2 | 40.0% |
| vt | 16384 | 5 | 4 | 80.0% |
| vt | 32768 | 5 | 2 | 40.0% |
| vt | 65536 | 5 | 2 | 40.0% |

**Hottest row (highest anchor-zero rate):** `| vt | 16384 | 5 | 4 | 80.0% |`

## Hit Count Distribution

### 2-gram hit distribution

| Hits | Cases |
|------|-------|
| 5 | 3 |
| 6 | 40 |
| 7 | 38 |
| 8 | 7 |
| 9 | 1 |
| 10 | 1 |
| 12 | 2 |
| 13 | 1 |
| 14 | 1 |
| 15 | 2 |
| 16 | 19 |

### 4-gram hit distribution

| Hits | Cases |
|------|-------|
| 0 | 13 |
| 1 | 7 |
| 2 | 3 |
| 3 | 4 |
| 4 | 40 |
| 5 | 37 |
| 6 | 6 |
| 16 | 5 |

### 6-gram hit distribution

| Hits | Cases |
|------|-------|
| 0 | 25 |
| 1 | 2 |
| 2 | 40 |
| 3 | 37 |
| 4 | 6 |
| 16 | 5 |

## Multi-Resolution Rescue Analysis

Cases where 4-gram=0 but another n-gram finds hits:

| Rescue source | Cases rescued | % of 4-gram-zero |
|---------------|--------------|-----------------|
| 2-gram | 13 | 100.0% |
| 6-gram | 0 | 0.0% |

## TF-IDF Weighted Top-3 Anchor 4-Grams (10 Random Cases)

- case 2 (niah_single/1024): `['Below is a long']`
- case 4 (niah_single/1024): `['Below is a long']`
- case 2 (niah_single/8192): `['Below is a long']`
- case 1 (niah_single/2048): `['Below is a long']`
- case 4 (niah_single/65536): `['Below is a long']`
- case 0 (niah_single/65536): `['Below is a long']`
- case 4 (fwe/8192): `['The grass is green']`
- case 1 (vt/65536): `['The grass is green']`
- case 2 (niah_single/4096): `['Below is a long']`
- case 4 (niah_single/16384): `['Below is a long']`

