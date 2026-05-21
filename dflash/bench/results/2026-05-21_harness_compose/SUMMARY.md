# Composition Bench Summary — 2026-05-21

## Composition Table (PFlash OFF vs ALWAYS per speculator)

| Speculator | Mode | keep | mean_accept_rate | mean_wall/turn | ok_done | turns |
|-----------|------|------|-----------------|---------------|---------|-------|
| MTP γ=2 | OFF | n/a | 0.813 | 2.53s | True | 3 |
| MTP γ=2 | ALWAYS | 0.05 | 0.850 | 6.21s | True | 3 |
| DFlash | OFF | n/a | — | 3.16s | True | 3 |
| DFlash | ALWAYS | 0.05 | — | 5.76s | True | 3 |

## Keep Sweep on MTP (keep_ratio effect on decode)

| keep | mean_accept_rate | mean_wall/turn | mean_completion_tokens | ok_done |
|------|-----------------|---------------|------------------------|---------|
| 0.025 | 0.850 | 4.54s | 51.7 | True |
| 0.050 | 0.850 | 6.21s | 51.7 | True |
| 0.100 | 0.850 | 4.12s | 51.7 | True |
| 0.200 | 0.850 | 5.15s | 51.7 | True |

## Per-Run Summary

| label | ok_done_seen | num_turns | mean_accept | mean_wall/turn | status |
|-------|-------------|-----------|------------|---------------|--------|
| mtp-off | True | 3 | 0.813 | 2.53s | OK |
| mtp-always-k05 | True | 3 | 0.850 | 6.21s | partial (1/3 OK_DONE) |
| dflash-off | True | 3 | — | 3.16s | OK |
| dflash-always-k05 | True | 3 | — | 5.76s | OK |
| mtp-always-k025 | True | 3 | 0.850 | 4.54s | partial (1/3 OK_DONE) |
| mtp-always-k10 | True | 3 | 0.850 | 4.12s | partial (1/3 OK_DONE) |
| mtp-always-k20 | True | 3 | 0.850 | 5.15s | partial (1/3 OK_DONE) |
