# Pass A: NIAH 1K-16K — ee14 vs baseline

| ctx | condition | drafter_fwd_p50 | ttft_p50 | NIAH | speedup |
|---|---|---|---|---|---|
| 1024 | baseline | 0.300s | 5.05s | 1/3 | 1.00x |
| 4096 | baseline | 0.810s | 2.64s | 1/3 | 1.00x |
| 8192 | baseline | 1.355s | 5.05s | 2/3 | 1.00x |
| 16384 | baseline | 2.585s | 6.72s | 2/3 | 1.00x |
| 1024 | ee14 | 0.210s | 4.97s | 1/3 | 1.43x |
| 4096 | ee14 | 0.470s | 1.86s | 1/3 | 1.72x |
| 8192 | ee14 | 0.765s | 4.34s | 2/3 | 1.77x |
| 16384 | ee14 | 1.380s | 5.42s | 2/3 | 1.87x |
