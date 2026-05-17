# Bench matrix — 2026-05-17  commit 0925bea

GPU: NVIDIA GeForce RTX 3090, 595.79  |  Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39  |  Build cuda_12.6.r12.6/compiler.35059454_0

## swe_bench_2k  (n_sample=8, n_runs=8)

| speculator | tok/s median | tok/s p25–p75 | CI 95% | prefill s med | AL/accept | speedup vs AR | distribution |
| ---------- | ------------ | ------------- | ------ | ------------- | --------- | ------------- | ------------ |
| ar | 34.54 | 34.48–34.61 | 34.47–34.62 | 2.54 | n/a | 1.00x | [█| | | |██████████] 31–35 |
| dflash_b22 | 49.65 | 43.90–52.74 | 43.53–53.52 | 0.00 | AL=4.23 | 1.44x | [██████████|██████████|███| |███] 39–72 |
| mtp_d3 | 53.19 | 50.57–56.01 | 50.79–55.62 | 2.68 | acc=66.6% | 1.54x | [██████████|███|███████|███|███] 50–59 |
