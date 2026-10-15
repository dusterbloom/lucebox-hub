# 2.0b M2 with no bias set: main 6f2b1795 vs M2 fd1a7b96, RTX 3090

Run 2026-10-15 on the local RTX 3090. M2 adds an optional key bias input to the flash-attention op (branch `kvflash/2.0b-m2-kv-bias`). The question: when no bias is set, is anything slower than main, prefill in particular?

An earlier check with two runs per arm (`/mnt/kimi-k3/kvflash-gate-m2null/`, M2 at 535de216) gave a pooled prefill median of 716 tok/s for main against 655 for M2. It was not settled, so this check reruns with five runs per arm. It also scores the cold first turn on its own, because it is the only prefill long enough to time.

Workload: the Gate A chat, 10 turns, with an 11,606-token first turn, temperature 0, seed 1, 80 max tokens. No `--kvflash`. Order: main, M2, alternating, five runs each. Scripts are in `scripts/`: `ab_all.sh` drives the ten runs, `ab_run.sh` runs one, and `ab_cmp.py` builds `cmp.out`. `gate4_live.py` and `kvlib.py` are the same as in `../kvoff_parity_3090_6f2b1795/`.

| Axis | Value |
|---|---|
| Model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, the same file on both |
| Binaries | main 6f2b1795: `luce_server` md5 117e87665bc2d60cb715a35d7f2c4076, `libggml-cuda.so` md5 cb05afca10c6b69faa0ff72a5b48f608, `libggml-base.so` md5 56e714da50d4679c9117cb76af1b032f. M2 fd1a7b96: `luce_server` md5 7690af4619341f6571531d2920926a62, `libggml-cuda.so` md5 a956d65370b10167663ece59307adc81, `libggml-base.so` md5 40f4e050c763851987d3de28044643cd |
| Server flags | default KV type, `--max-ctx 131072`, `--no-kv-cache`, otherwise the same on both |
| Caching | RAM prefix cache on both. `prefix_len` matches turn for turn on all ten runs (0, 11600, 11952, 12398, 12607, 12892, 13259, 13705, 13915, 14199). `disk_hit=false` on all 100 requests |
| GPU | all ten runs ran in one `flock /mnt/kimi-k3/build/3090.lock` call, with one server at a time and no compute apps on the GPU before each start. See `provenance.txt`, which the locked script wrote |

The `luce_server` binaries were copied to fixed directories, but they load the ggml libraries from their build directories (`/mnt/kimi-k3/build/main-6f2b-cuda`, `/mnt/kimi-k3/build/m2-cuda`). The library md5 values above were taken after the run. The `luce_server` md5 is the same as for M2 535de216, because between 535de216 and fd1a7b96 only `libggml-cuda` (a support check that no op in this workload reaches) and the test changed. The M2 `libggml-cuda.so` was built at 11:46:17 from the tree carrying 239f8d63, which was committed at 11:47:08.

## Result

- Answers are byte-identical across all ten runs (10/10 turns).
- Medians, all values tok/s except wall:

| Metric | main | M2 |
|---|---|---|
| Turn-1 prefill, 11.6K tokens (median of 5 runs) | 1398.3 | 1398.3 |
| Turn-1 prefill, min to max | 1365 to 1415 | 1382 to 1415 |
| Turns 2-10 prefill (pooled) | 663.3 | 716.0 |
| Decode (all turns pooled) | 43.80 | 43.70 |
| Wall, s (all turns pooled) | 1.30 | 1.30 |

Turn-1 prefill seconds, in run order: main 8.4, 8.3, 8.5, 8.2, 8.3; M2 8.4, 8.3, 8.2, 8.3, 8.4. The spread in each arm is one 0.1 s tick in the server log, with no drift.

**Verdict.** With no bias set, M2 is not slower than main: the cold 11.6K-token prefill is the same at a median of 1398 tok/s, decode is within 0.1 tok/s, and wall is the same. Turns 2-10 each prefill about 350 tokens in about 0.5 s, timed to 0.1 s, so their tok/s values land on a few discrete steps (663, 716). Their pooled median flips between steps from run to run. In the earlier two-run check it went 716 against 655 in main's favour; here it is 663 against 716 in M2's favour. That gap is the timer, not the kernel. No separate template switch for the bias is needed.

Files: `cmp.out` (parity checks and medians per arm), `<arm>_<N>/driver.out`, `provenance.txt`. Full logs and answers are in `/mnt/kimi-k3/kvflash-gate-m2prefill/`.
