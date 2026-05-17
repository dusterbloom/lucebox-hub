# Bench matrix — 2026-05-17  commit 0925bea

GPU: NVIDIA GeForce RTX 3090, 595.79  |  Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39  |  Build cuda_12.6.r12.6/compiler.35059454_0

## swe_bench_2k  (n_sample=8, n_runs=8)

| speculator | tok/s median | tok/s p25–p75 | CI 95% | prefill s med | AL/accept | speedup vs AR | distribution |
| ---------- | ------------ | ------------- | ------ | ------------- | --------- | ------------- | ------------ |
| ar | 32.27 | 32.03–32.66 | 32.02–32.66 | 2.88 | n/a | 1.00x | [███| |██████████|███|██████████] 32–33 |
| dflash_b22 | 46.70 | 42.83–51.46 | 42.68–51.96 | 0.00 | AL=4.23 | 1.45x | [██████████|███████|███████| |███] 38–68 |
| mtp_d3 | 49.21 | 45.97–49.96 | 45.59–49.86 | 3.01 | acc=66.6% | 1.53x | [███|███|███|███████|██████████] 43–51 |
