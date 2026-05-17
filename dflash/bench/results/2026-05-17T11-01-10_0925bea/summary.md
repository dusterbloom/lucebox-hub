# Bench matrix — 2026-05-17  commit 0925bea

GPU: NVIDIA GeForce RTX 3090, 595.79  |  Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39  |  Build cuda_12.6.r12.6/compiler.35059454_0

## swe_bench_2k  (n_sample=8, n_runs=24)

| speculator | tok/s median | tok/s p25–p75 | CI 95% | accept_rate | speedup vs AR | distribution |
| ---------- | ------------ | ------------- | ------ | ----------- | ------------- | ------------ |
| ar | 32.65 | 32.14–33.25 | 32.25–33.06 | n/a | 1.00x | [██████|██████|██████████|█████|██] 31–35 |
| dflash_b22 | 0.00 | 0.00–0.00 | 0.00–0.00 | n/a | 0.00x | (no data) |
| mtp_d3 | 51.44 | 47.21–54.43 | 49.55–53.19 | 65.5% | 1.58x | [███████|█████|██████████|██████████|████████] 43–58 |
