# Qwen3.8-Flash-Next (`qwen4exp`) backend

Hand-written ggml graph and CUDA/HIP kernels for `Qwen/Qwen3.8-Flash-Next`, the
experimental hybrid-attention MoE. Runs on the existing `dflash_server`
(`--target-device hip:0`), alongside the DFlash/PFlash backends.

## Model

- 48 layers, hidden 2560; layout `12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))`.
- MoE: 512 experts, top-10 routed + 1 shared; expert intermediate 640.
- QSA (full-attention layers): 24 Q / 2 KV heads, head_dim 256, RoPE dim 64;
  indexer = MQA with 4 Q heads / 1 KV head, head_dim 128, budget 512 blocks (2048 tokens).
- Hyper-connections: 4 streams, bottleneck rank 320. Per-layer n-gram embeddings (PLE).
- 125B params / 6B active, plus a 51B n-gram embedding table and a 4B MTP head.
  Context 262144 native.

## Layout

- `server/src/qwen4exp/` — loader (sharded GGUF, lazy PLE reader), backend/daemon
  wiring, KV + recurrent + indexer cache, CPU embedding, and the forward graph.
- `server/deps/llama.cpp/ggml/src/ggml-cuda/` — QSA selected-attention kernels
  (`qsa*.cu[h]`), the fused DS4 indexer, the bf16-dequant GEMM family (`mmb*`),
  fused hyper-connection / MoE / GDN / PLE-conv kernels, and packed `get_rows`.

## Measured (gfx1151, Radeon 8060S, 3-shard IQ4_NL, planted-correct)

Prefill is reported as **min-of-N prompt-processing-only** (single-run
end-to-end timings include decode + warmup and hide 1-2% changes).

| prompt | prefill tok/s | notes |
|---|---|---|
| 13,664 | ~960 | single chunk |
| 16,366 | **988** | vs 738 for the pwilkin reference on this box |
| 32,792 | ~893 | chunked QSA |
| 65,528 | ~926 | `--chunk 16384`, chunked QSA |

Decode ~29 tok/s (autoregressive; MTP is not wired yet).

## Enablement / env

Best-known config:
`QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5 DFLASH_MMB_SHADOW=1 LLAMA_MMB_HC16=2`.

- `QWEN4EXP_QSA` — QSA selected attention (chunked prefill via the indexer-K cache).
- `QWEN4EXP_MMB_CUBLAS` — route quantized GEMM shapes to cuBLAS/hipBLASLt via a
  bf16 weight shadow (1/3/5 add ssm_out and HC down/up).
- `DFLASH_MMB_SHADOW` — bf16 weight shadow mode (1 = IQ4_NL/Q5_K, 2 = Q6_K).
- `LLAMA_MMB_HC16` — keep the hyper-connection normalized stream bf16-only.
- `QWEN4EXP_CHUNK` — prefill chunk size (16384 recommended; smaller sizes still
  populate the indexer cache).

## Benchmarking and correctness

- **Correctness gate:** `client_test_runner.py bench --suite recall` (planted-fact
  long-context recall at 13.6k/24k); the FA launch counter
  (`QWEN4EXP_FA_TELEMETRY=1`) asserts QSA engaged.
- **Quality:** `client_test_runner.py bench --suite he,gsm,math` (HumanEval scored by
  executing gold tests).
- **Kernel profiling:** `rocprofv3 --kernel-trace`; `GGML_CUDA_OP_PROF=1` for
  per-shape `mul_mat` timings.
- **Kernel differentials:** `server/test/` (mmb-vs-cuBLAS, DS4 mmid regression,
  forward smoke).

## Not included

Speculative decoding (the model's MTP head) and QSA decode attention are planned
follow-ups; decode is currently dense autoregressive.
