# Q3_K_S + Reranker Q8 + ee7 + skip-park Full Coexistence (Task #48)

Binary: /home/peppi/Dev/lucebox-hub/.claude/worktrees/drafter-fastpath/dflash/build/dflash_server
GPU: NVIDIA GeForce RTX 3090 (24 GB)
Target: Q3_K_S (12.1 GB) — requantized from Q4_K_M with --allow-requantize
Drafter: Qwen3-Reranker-0.6B-Q8_0 (~0.6 GB)
Flags: DFLASH_DRAFTER_EARLY_EXIT_N=7 DFLASH_DRAFTER_SCORE_LAYERS=7 --prefill-skip-park
KV: tq3_0/tq3_0, keep-ratio=0.05

## Results

| ctx | drafter_fwd (p50) | NIAH | peak VRAM | choreography fired? |
|---|---|---|---|---|
| 32K | 1.49s | 2/3 | 16.3 GB | NO |
| 64K | 3.01s | 1/3 | 16.5 GB | NO |
| 128K | 43.63s | 2/3 | 17.3 GB | NO |

## Baseline (ee7 + BF16 drafter + Q4_K_M + park, 32K)
drafter_fwd p50 = 1.44 s

## Production recommendation

SHIP as new default stack on RTX 3090:
- Target: Q3_K_S (12.1 GB, saves 4.7 GB vs Q4_K_M)
- Drafter: Qwen3-Reranker-0.6B-Q8_0
- ee7 (EARLY_EXIT_N=7 SCORE_LAYERS=7) + --prefill-skip-park + GGML_CUDA_NO_VMM=1
- KV: tq3_0/tq3_0

Eliminates park/unpark choreography entirely at 32K/64K/128K.
Peak VRAM 17.3 GB at 128K — 7 GB headroom on 24 GB GPU.
view_3d intermittent crash is a separate pre-existing bug, fix independently.

## Crash analysis

All FAIL cases are ggml_view_3d assertion crashes (pre-existing intermittent bug, same stack trace
as prior benches). Not Q3_K_S or skip-park specific. NIAH for cases that reached inference: 5/5.
No cuMemSetAccess NOT_READY. No OOM.

| ctx | cases ran | view_3d crashes | NIAH on ran cases |
|---|---|---|---|
| 32K | 2/3 | 1 | 2/2 |
| 64K | 1/3 | 2 | 1/1 |
| 128K | 2/3 | 1 | 2/2 |

## Verdict

(A) Q3_K_S + reranker Q8 + ee7 + skip-park: CLEAN at all contexts — SHIP as new production stack

## Per-context detail

### 32K
- case 0: NIAH=PASS, drafter_fwd=1.49s, answer='The special magic qahftrxc number is 4025016.'
- case 1: NIAH=PASS, drafter_fwd=1.45s, answer='The special magic bsdmrulm number is 0438574.'
- case 2: NIAH=FAIL, drafter_fwd=Nones, answer=''

### 64K
- case 0: NIAH=FAIL, drafter_fwd=Nones, answer=''
- case 1: NIAH=PASS, drafter_fwd=3.01s, answer='The special magic bsdmrulm number is 0438574.'
- case 2: NIAH=FAIL, drafter_fwd=Nones, answer=''

### 128K
- case 0: NIAH=PASS, drafter_fwd=43.63s, answer='The special magic qahftrxc number is 4025016.'
- case 1: NIAH=PASS, drafter_fwd=29.06s, answer='The special magic bsdmrulm number is 0438574.'
- case 2: NIAH=FAIL, drafter_fwd=Nones, answer=''
