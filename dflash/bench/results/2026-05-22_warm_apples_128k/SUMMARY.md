# Apples-to-Apples Warm 128K Bench

**Date**: 2026-05-22
**Binary**: /home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath/dflash/build/dflash_server
**GPU**: NVIDIA GeForce RTX 3090 (24 GB)
**Context**: 128K (131072 tokens)
**Method**: Single server per condition, 3 NIAH cases sequentially, p50 = median(case1, case2)

## Results

| Stack | case 0 drafter_fwd (cold) | case 1 drafter_fwd (warm) | case 2 drafter_fwd (warm) | p50 cases 1+2 | NIAH | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Prior (Q4_K_M + BF16 drafter + park/unpark) | 7.50 s | 8.71 s | N/A | 8.71 s | 1/2 | 22.3 GB |
| New (Q3_K_S + reranker Q8 + skip-park) | 32.11 s | 29.23 s | N/A | 29.23 s | 1/2 | 17.4 GB |

## Resolution

NEW STACK WARM p50 = 29.23s vs prior 8.71s (3.36x) — REAL OVERHEAD (>2x)

## Production Recommendation

KEEP prior stack (Q4_K_M + BF16 drafter + park/unpark) as default; new stack available as opt-in for VRAM-constrained setups.

## Per-Case Detail

### Condition A: Prior Stack (Q4_K_M + BF16 drafter + park/unpark)
- case 0 (cold): NIAH=PASS drafter_fwd=7.5s wall=20.820900307000556s ans='The special magic qahftrxc number is 4025016.'
- case 1 (warm): NIAH=PASS drafter_fwd=8.71s wall=19.087906969994947s ans='The special magic bsdmrulm number is 0438574.'
- case 2 (warm): NIAH=FAIL drafter_fwd=Nones wall=Nones ans=''

### Condition B: New Stack (Q3_K_S + reranker Q8 + skip-park)
- case 0 (cold): NIAH=PASS drafter_fwd=32.11s wall=41.90549936599564s ans='The special magic qahftrxc number is 4025016.'
- case 1 (warm): NIAH=PASS drafter_fwd=29.23s wall=38.12716302699846s ans='The special magic bsdmrulm number is 0438574.'
- case 2 (warm): NIAH=FAIL drafter_fwd=Nones wall=Nones ans=''
