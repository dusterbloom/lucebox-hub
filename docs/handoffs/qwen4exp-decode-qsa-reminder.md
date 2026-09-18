# Reminder: enable QSA on the decode path (deferred until prefill work is done)

Status as of the QSA-padding commit (`ffbd074f`): decodes still run **dense**
flash attention. The per-graph FA launch counter (`QWEN4EXP_FA_TELEMETRY=1`)
shows `dense=12, qsa=0` for every decode graph, `qsa=12` for the prefill.

## Why it does not engage

`ggml_cuda_flash_attn_ext_qsa_decode_supported` (`qsa-decode.cuh:80`) requires
`dst->src[5]` (ids = selected cells) and `src[6]/src[7]` null. Our server only
builds a QSA attn when `qsa_ok` (`qwen4exp_graph.cpp` `build_full_attn`:
`T >= 128 && pos0 == 0 && kv_len == T`); for `T < 128` it takes the dense
branch, which sets no ids, so the decode check fails on `!ids`.

Building ids for a decode needs the **pooled indexer keys of the whole
context**; the prefill computes `pooled` transiently and discards it. We have
no indexer-K cache.

## What to do (reference is `strix-ref`)

1. Indexer-K cache in `qwen4exp_cache` (per full-attn layer, `[idim, max_ctx/r]`).
2. Port `build_qsa_store_k` (`strix-ref src/models/qwen4exp.cpp:963`) and
   `build_qsa_top_k` (`:1014`, ~220 lines; trim to single-stream).
3. Absolute-position visibility (`pos0 + row`) — the same change chunked-prefill
   QSA needs; see the sub-quadratic indexer work.
4. Decode branch in `build_full_attn`: emit `ids`, `src[6]/src[7] = nullptr`,
   plus the mask (`build_attn_qsa`, `:1248`).

## Validation

Planted gate over a 5-rep run (prefill + decode), with the FA launch counter
asserting `qsa`/`qsa_decode` engaged on the decode graphs too.
