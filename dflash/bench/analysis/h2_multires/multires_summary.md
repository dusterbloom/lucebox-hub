# H2 Multi-Resolution Anchor — Results Summary

Total cases: **250** (real_captured=30, synthetic=220)

## Cross-tab: Corpus x Resolution

| Corpus | Resolution | Cases | Mean hits | Anchor-zero rate | High-prec chunk frac | Forced chunks (mean) |
|--------|------------|-------|-----------|-----------------|---------------------|----------------------|
| real_captured | 2gram | 30 | 15.8 | 0.0% | 0.36 | 6.5 |
| real_captured | 4gram | 30 | 13.8 | 3.3% | 0.41 | 5.3 |
| real_captured | 6gram | 30 | 12.4 | 13.3% | 0.36 | 4.5 |
| real_captured | union | 30 | 6.7 | 0.0% | n/a | 6.7 |
| synthetic_fwe | 2gram | 5 | 16.0 | 0.0% | 0.17 | 14.0 |
| synthetic_fwe | 4gram | 5 | 16.0 | 0.0% | 0.51 | 14.0 |
| synthetic_fwe | 6gram | 5 | 16.0 | 0.0% | 0.30 | 14.8 |
| synthetic_fwe | union | 5 | 16.0 | 0.0% | n/a | 16.0 |
| synthetic_niah_single | 2gram | 190 | 6.6 | 0.0% | 1.00 | 4.2 |
| synthetic_niah_single | 4gram | 190 | 4.6 | 0.0% | 1.00 | 4.2 |
| synthetic_niah_single | 6gram | 190 | 2.6 | 0.0% | 1.00 | 4.2 |
| synthetic_niah_single | union | 190 | 4.2 | 0.0% | n/a | 4.2 |
| synthetic_vt | 2gram | 25 | 13.8 | 0.0% | 0.95 | 3.0 |
| synthetic_vt | 4gram | 25 | 0.8 | 52.0% | 0.48 | 1.4 |
| synthetic_vt | 6gram | 25 | 0.0 | 100.0% | 0.00 | 0.0 |
| synthetic_vt | union | 25 | 3.0 | 0.0% | n/a | 3.0 |

## Headline: Anchor-Zero Rescue Rate (real_captured only)

Real anchor-zero cases (4-gram hits=0): **1** / 30

| Rescue source | Rescued | % of anchor-zero |
|---------------|---------|-----------------|
| 2-gram | 1 | 100.0% |
| 6-gram | 0 | 0.0% |
| union (2g+6g) | 1 | 100.0% |

## Precision Proxy (high-prec chunk fraction, real_captured)

| Resolution | Mean high-prec frac | Cases with hits |
|------------|---------------------|-----------------|
| 4gram (baseline) | 0.427 | 29 |
| 2gram | 0.365 | 30 |
| 6gram | 0.418 | 26 |

*(high-prec = forced chunk anchored by n-gram with body-local freq <= 3)*

## Needle Recall (synthetic_niah_single only)

NIAH cases with located needle: **190** / 190

| Scheme | Needle recall |
|--------|--------------|
| 4gram (baseline) | 190/190 = 100.0% |
| union (multi) | 190/190 = 100.0% |

## Top-3 Rescued Anchor-Zero Cases

1. `real_-home-peppi-Dev-luce` (real_captured) — rescued by **2g** with 13 hits (forced 5 chunks)

## Momus Go/No-Go Verdict

- Anchor-zero rescue by union on real_captured: **100.0%** (threshold: >=40% PROCEED, <15% KILL)

- High-precision fraction drop (4g → 2g): **6.2 pp** (threshold: >15pp KILL)


**VERDICT: PROCEED union rescues 100% of real anchor-zero cases with 6.2pp precision drop**

