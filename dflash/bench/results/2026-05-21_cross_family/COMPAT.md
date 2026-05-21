# Cross-Family Drafter Loader Compat — 2026-05-21

## Summary

Both SmolLM2 models downloaded and converted to BF16 GGUF successfully.
The dflash drafter loader **rejects both** — it cannot load llama-arch GGUFs.
The loader is hardwired to read `qwen3.*` GGUF metadata keys and expects
`blk.<i>.attn_q_norm.weight` / `blk.<i>.attn_k_norm.weight` (QK-norm tensors
present in Qwen3 but absent from Llama-family models).

## Candidates

| Candidate | Downloaded | Size on disk | GGUF converted | GGUF size | Loader accepts? | Gap if rejected |
|---|---|---|---|---|---|---|
| SmolLM2-360M | yes | 691 MB safetensors | yes | 692 MB BF16 GGUF | **NO** | arch mismatch + missing QK-norm tensors |
| SmolLM2-135M | yes | 257 MB safetensors | yes | 259 MB BF16 GGUF | **NO** | arch mismatch + missing QK-norm tensors |

## Loader Analysis

### Architecture detection (`qwen3_loader.cpp` lines 90–97)

The loader reads metadata with hardcoded `qwen3.*` keys:

```cpp
out.n_embd    = (int)get_u32(gctx, "qwen3.embedding_length", 1024);
out.n_ff      = (int)get_u32(gctx, "qwen3.feed_forward_length", 3072);
out.n_head    = (int)get_u32(gctx, "qwen3.attention.head_count", 16);
out.n_head_kv = (int)get_u32(gctx, "qwen3.attention.head_count_kv", 8);
out.n_layer   = (int)get_u32(gctx, "qwen3.block_count", 28);
out.n_ctx_max = (int)get_u32(gctx, "qwen3.context_length", 40960);
out.head_dim  = (int)get_u32(gctx, "qwen3.attention.key_length", 128);
out.rope_theta = get_f32(gctx, "qwen3.rope.freq_base", 1000000.0f);
```

SmolLM2 GGUFs use `llama.*` keys (e.g. `llama.embedding_length`). All
`get_u32` calls return their hardcoded Qwen3-0.6B defaults — the model would
be silently misconfigured even if it did not fail on tensors.

### Missing QK-norm tensors (`qwen3_loader.cpp` lines 240–241)

Per layer the loader unconditionally demands:

```
blk.<i>.attn_q_norm.weight
blk.<i>.attn_k_norm.weight
```

SmolLM2 is a standard Llama-architecture model with no per-head QK-norm.
Neither GGUF contains these tensors. `copy_tensor_from_file` returns false for
each missing tensor (prints `[qwen3-0.6b] missing tensor: blk.0.attn_q_norm.weight`
32x2=64 times for 360M), accumulates `ok = false`, and the loader returns
failure with `"one or more Qwen3-0.6B tensors failed to load"`.

### Actual GGUF shape of SmolLM2-360M

```
general.architecture = llama
llama.block_count           = 32
llama.embedding_length      = 960
llama.feed_forward_length   = 2560
llama.attention.head_count  = 15
llama.attention.head_count_kv = 5
llama.attention.key_length  = 64     # head_dim = 64 (NOT 128)
llama.rope.freq_base        = 100000
vocab_size                  = 49152  # SmolLM BPE (not Qwen3 151936)
tie_word_embeddings         = true   # no output.weight tensor
blk.0 tensors: attn_norm, attn_q, attn_k, attn_v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down
```

### Actual GGUF shape of SmolLM2-135M

```
general.architecture = llama
llama.block_count           = 30
llama.embedding_length      = 576
llama.feed_forward_length   = 1536
llama.attention.head_count  = 9
llama.attention.head_count_kv = 3
llama.attention.key_length  = 64
llama.rope.freq_base        = 100000
vocab_size                  = 49152
tie_word_embeddings         = true
```

## Cross-Family Loader Adapter — Task Description

File: `dflash/src/qwen3/qwen3_loader.cpp` + `dflash/src/qwen3/qwen3_drafter_model.h`

Three changes needed:

1. **Arch-aware key prefix** (`qwen3_loader.cpp` ~line 84-97, ~15 LOC).
   Read `general.architecture` from the GGUF. If `"llama"` substitute
   `"llama."` for `"qwen3."` when calling `get_u32`/`get_f32`. Default
   values must update for llama-arch (head_dim=64, n_vocab=49152, rope_theta=100000).

2. **Optional QK-norm load** (`qwen3_loader.cpp` ~line 240-241, ~10 LOC).
   Detect whether `blk.0.attn_q_norm.weight` exists in the GGUF.
   If absent, set `L.q_norm = L.k_norm = nullptr` and skip copy.
   The forward pass in `qwen3_graph.cpp` must null-check before applying RMSNorm.

3. **Forward graph QK-norm guard** (`qwen3_graph.cpp`, ~5 LOC).
   Wrap the per-layer `ggml_rms_norm(q_norm)` / `ggml_rms_norm(k_norm)`
   calls behind `if (L.q_norm != nullptr)`.

Estimated total: ~30 LOC across two files. The arch-detection refactor is the
riskiest part — the hardcoded defaults in `get_u32` for all the Qwen3-specific
fields must become correct llama-arch defaults, otherwise dimension mismatches
produce silent garbage without any assertion.

## Recommendation

Queue the cross-family loader-adapter task. Both SmolLM2 GGUFs are on disk at
`/home/peppi/models/SmolLM2-{135M,360M}-BF16.gguf`. The adapter is ~30 LOC;
once it lands, queue SmolLM2-{135M,360M} for bench sweep after the
Qwen2.5-0.5B + ee14 stack bench completes.

Smoke test not run — `dflash_server` requires CUDA; CPU-only environment.
Loader rejection confirmed by static code analysis against the GGUF tensor
inventory extracted via `gguf.GGUFReader`.
