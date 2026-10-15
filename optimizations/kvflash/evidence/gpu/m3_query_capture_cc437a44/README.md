# KVFlash 2.0b M3, query capture during prefill: evidence

**Superseded by `../m3_query_capture_a4d37de1/` (review fix: capture no longer changes the chunk plan).** This directory is the "before" record: capture-on prefill -4.5%.

Commit under test: `cc437a44` (branch `kvflash/2.0b-m3-query-capture`, parent `fd1a7b96` = M2 tip).
GPU: local RTX 3090. Every GPU run was under `flock /mnt/kimi-k3/build/3090.lock`, with
`nvidia-smi --query-compute-apps` empty before each start (see `provenance.txt`), except two
`test_server_unit` full-suite runs that were launched without the lock on an idle GPU (unit tests, not the A/B).

## What M3 adds

- `TargetCache::q_blk`: F16 `[head_dim, block_tokens, n_head, n_fa]` for the block being sealed
  (640 x 24 x 256 x 2 B = 7,864,320 B per full-attention layer, 125.8 MB = 120.0 MiB for the 16 layers
  of Qwen3.8-27B). One plane per full-attention layer is `[head_dim, T, n_head]`, the Q operand of
  `ggml_mul_mat(K[d, n, h_kv], Q[d, T, h])`.
- Not part of `base_buf`. Allocated on demand (`alloc_q_block_capture`), freed at the seal
  (`kvflash_am_seal_block`) or when the prefill ends. Capture off never allocates.
- Graph: in `build_qwen35_graph`, for every full-attention layer, the window
  `QwenGraphInputs::q_blk_{first,count,dst}` of `q_fa` (the tensor flash attention consumes: post-norm,
  post-RoPE, post-rotation) is `ggml_cpy`'d (F32 to F16) into the layer's plane at the block row.
- Backend: `LUCE_KVFLASH_AM_CAPTURE=1` (default off). With it on, each prefill chunk is cut at the 640-token
  block boundary; a block that began before the prefill (restored prefix) is skipped.

## Correctness (unit tests, `server/test/unit/test_qwen35_q_block_capture.cpp`)

Synthetic 2-layer model, all layers full attention, real attention shape (head_dim 256, 12 heads, 2 KV heads,
n_rot 64, sections 11/11/10/0), F16 KV. The reference is a scalar double-precision implementation in the test
(rms norm, Q/gate split, Q norm, interleaved M-RoPE, then all of layer 0 for layer 1's input). It does not read
any capture tensor.

| Backend | layer 0 worst max abs diff / max abs Q | layer 1 | thresholds |
|---|---|---|---|
| CPU | 2.93e-04 | 6.92e-04 | 1e-3 / 2e-3 |
| RTX 3090 | 4.95e-04 | 1.23e-03 | 1e-3 / 2e-3 |

The error is the F16 rounding of Q (up to 4.9e-4 relative) plus, for layer 1, layer 0's F16 KV cache.
Bitwise checks: the F16 capture of the last token equals `fp16(q_cap)` of the same graph tensor, exactly, on
both backends. Rows outside the window keep their sentinel. Windows tested: two ubatches with the second at a
non-zero `kv_start` straddling the block end; a window starting at `q_blk_first = 8` inside one ubatch.
Capture off: no buffer, node count and the hidden output bit-identical (request without buffer, buffer with
count 0, and capture on).

Red: `unit/red_stub_alloc.log` (stub `alloc_q_block_capture` returning false: 4 failed, 2 planner tests pass).
Green: `unit/green_cpu.log`, `unit/green_gpu_3090.log`. Full `test_server_unit`: 1022 passed, 0 failed, 7 skipped.
`test_fattn_kv_bias` (M2) on the 3090: unchanged, 0 failures. `ctest -L cpu`: all pass except
`server_unit_entrypoint`, which fails identically on the M2 tip build (host-dependent shell test).
No new compiler warnings in the touched files.

## Cost A/B, 3090

Parity contract: same GGUF `/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf`, default KV type,
`--max-ctx 131072`, `--no-kv-cache` (`disk_hit=false` on all 150 chat lines), no `--kvflash`, greedy chat client,
workload `gate4_live.py A` (warm-up of 1.3K tokens, then 10 turns; turn 1 is the 11,606-token cold prefill),
5 interleaved runs per arm, `prefix_len` identical turn for turn on all 15 runs, answers byte-identical
between all M2 and M3 (capture off) runs and also for the capture-on runs.
Prefill tok/s = (effective_in - prefix_len) / prefill seconds, where the server log rounds seconds to 0.1.

| md5 | file |
|---|---|
| 7690af4619341f6571531d2920926a62 | M2 `luce_server` (`/mnt/kimi-k3/build/m2-fd1a7b96-ab/luce_server`) |
| bed2fd3aef709243cf5e82c729cafcd4 | M3 `luce_server` (`/mnt/kimi-k3/build/m3-cuda/luce_server`) |
| 40f4e050c763851987d3de28044643cd / 71647ba71595c5bafaffbfbdbda458cb / 78ecf2898f47319df4b19d83121565ca / a956d65370b10167663ece59307adc81 | M2 libggml-base / -cpu / libggml / libggml-cuda (`m2-cuda/deps/llama.cpp/ggml/src`) |
| 7a6a683b45db6751291b8ad88d1176bd / 86678a23e684d77a2eaf99df5be69fa9 / 135c796c50fc7f435ad03ca2956329ea / 17043d5adbf6f570649d2e3f0f94eb80 | M3 libggml-base / -cpu / libggml / libggml-cuda (`m3-cuda/deps/llama.cpp/ggml/src`) |

The ggml sources are identical between `fd1a7b96` and `cc437a44` (`git diff` over `server/deps` is empty); the .so
md5s differ because the two build dirs differ.

Medians of 5 runs (turn 1) and of all turns pooled (decode, wall):

| arm | turn-1 prefill tok/s | turn-1 wall s | decode tok/s (50 turns) | wall s (50 turns) |
|---|---|---|---|---|
| M2 tip | 1381.67 | 8.80 | 44.60 | 1.30 |
| M3, capture off | 1398.31 | 8.70 | 44.50 | 1.30 |
| M3, capture on (`LUCE_KVFLASH_AM_CAPTURE=1`) | 1318.86 | 9.20 | 45.05 | 1.30 |

Capture off vs M2: decode -0.2% (limit -2%), turn-1 prefill +1.2% (inside the 0.1 s quantisation: 8.0 to 8.7 s
on both arms), wall equal. Capture on: turn-1 prefill -4.5% vs M2 (-5.7% vs M3 off), decode unchanged. Full
tables with mean, min, max and per-run values in `cmp.out`.
