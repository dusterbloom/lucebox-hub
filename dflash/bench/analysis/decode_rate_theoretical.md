# Theoretical decode tok/s — PFlash KV compression, RTX 3090

## Model summary

Architecture: **Qwen3-27B** (n_layer=64, n_head=64, n_kv_head=8/GQA, head_dim=128)
KV quant: **tq3_0** — 14 bytes per 32 elements = **3.5 bits/value = 0.4375 bytes/value**
Hardware: **RTX 3090**, 936 GB/s memory bandwidth

The claim being tested: at long context, decode is memory-bandwidth-bound; a smaller KV
cache proportionally reduces bandwidth per step, so decode tok/s scales inversely with
context length. PFlash's compressed KV (keep_ratio=0.05 retains 5 % of tokens) reduces
effective context by 20× — the predicted decode speedup equals the compression ratio.

---

## Step 1 — bytes_per_element for tq3_0

From `ggml-common.h`:
```c
#define QK_TQ3_0 32
struct block_tq3_0 {
    uint8_t qs[QK_TQ3_0 / 4];    // 8 bytes (2-bit low indices)
    uint8_t signs[QK_TQ3_0 / 8]; // 4 bytes (1-bit sign per element)
    // NOTE: implicit 2-byte scale per block (sizeof(block_tq3_0) == 14)
};
// 14 bytes / 32 elements = 0.4375 bytes/elem = 3.5 bits/elem
```

Corrected vs. task prompt: the prompt cited 3.0625 bits (256-element block assumption).
The actual block size is 32 elements, giving 3.5 bpv, not 3.0625.

---

## Step 2 — KV cache size table (MiB)

Formula: `2 × ctx_eff × n_kv_head × head_dim × bytes_per_elem × n_layer`
= `2 × L × 8 × 128 × 0.4375 × 64 = L × 3584 bytes = L × 3.5 KiB/token`

| ctx_full | keep=1.00 (MiB) | keep=0.20 (MiB) | keep=0.10 (MiB) | keep=0.05 (MiB) | keep=0.025 (MiB) |
|----------|-----------------|-----------------|-----------------|-----------------|------------------|
|       4K |           224.0 |            44.8 |            22.4 |            11.2 |              5.6 |
|       8K |           448.0 |            89.6 |            44.8 |            22.4 |             11.2 |
|      16K |           896.0 |           179.2 |            89.6 |            44.8 |             22.4 |
|      32K |         1,792.0 |           358.4 |           179.2 |            89.6 |             44.8 |
|      64K |         3,584.0 |           716.8 |           358.4 |           179.2 |             89.6 |
|     128K |         7,168.0 |         1,433.6 |           716.8 |           358.4 |            179.2 |

RTX 3090 has 24 GB VRAM. At 128K uncompressed, the KV alone is 7 GiB; at 32K it is 1.75 GiB.
With the full Q4_K_M model (~16 GiB), **64K uncompressed does not fit** (3.5 + 16 = 19.5 GiB, tight);
**128K uncompressed does not fit** (7.2 + 16 = 23.2 GiB, possible but no headroom).
At keep=0.05 those same contexts consume 180–360 MiB for the KV, leaving ample room.

---

## Step 3 — Arithmetic intensity check

At 32K context:
- Attention FLOPs per step ≈ `2 × n_head × ctx × head_dim × n_layer`
  = `2 × 64 × 32768 × 128 × 64 = 34.4 GFLOPs`
- Bandwidth per step = `2 × ctx × n_kv_head × head_dim × bytes_per_elem × n_layer`
  = `2 × 32768 × 8 × 128 × 0.4375 × 64 = 1,879 MB`
- Arithmetic intensity = 34,400 / 1,879 ≈ **18.3 FLOP/byte**
- RTX 3090 roofline ridge ≈ 35,600 GF / 936 GB/s ≈ **38 FLOP/byte**

18.3 < 38 → **decode is bandwidth-bound** at all context lengths in this table.
The BW-bound model is valid.

---

## Step 4 — Decode tok/s table

`tok/s = BW_GB_s × 10⁹ / bytes_per_step`

| ctx_full | keep=1.00 | keep=0.20 | keep=0.10 | keep=0.05 | keep=0.025 |
|----------|----------:|----------:|----------:|----------:|-----------:|
|       4K |    3,985  |   19,930  |   39,811  |   79,622  |   160,025  |
|       8K |    1,993  |    9,965  |   19,930  |   39,811  |    79,622  |
|      16K |      996  |    4,981  |    9,965  |   19,930  |    39,811  |
|      32K |      498  |    2,491  |    4,981  |    9,965  |    19,930  |
|      64K |      249  |    1,245  |    2,491  |    4,981  |     9,965  |
|     128K |      125  |      623  |    1,245  |    2,491  |     4,981  |

---

## Step 5 — Predicted speedup table (vs keep=1.0)

| ctx_full | keep=1.00 | keep=0.20 | keep=0.10 | keep=0.05 | keep=0.025 |
|----------|----------:|----------:|----------:|----------:|-----------:|
|       4K |     1.00× |     5.00× |     9.99× |    19.98× |     40.16× |
|       8K |     1.00× |     5.00× |    10.00× |    19.98× |     39.96× |
|      16K |     1.00× |     5.00× |    10.00× |    20.00× |     39.96× |
|      32K |     1.00× |     5.00× |    10.00× |    20.00× |     40.01× |
|      64K |     1.00× |     5.00× |    10.00× |    20.00× |     40.01× |
|     128K |     1.00× |     5.00× |    10.00× |    20.00× |     40.00× |

**The speedup equals the compression ratio exactly** (the model is purely BW-bound and linear).
There is no "crossover" context below which compression stops helping decode — the relationship
is linear in context length.  Even at 4K context the theoretical gain is 20× at keep=0.05;
however at short context other bottlenecks (GEMM weight loading, dispatch overhead) dominate in
practice, so the empirical speedup at short context will be much smaller.

---

## Step 6 — ASCII chart

```
Decode tok/s (theoretical) — Qwen3-27B tq3_0, RTX 3090 936 GB/s
──────────────────────────────────────────────────────────────────

     80K ┤                  keep=0.05 ●
         │
     60K ┤
         │
     40K ┤                             ●
     20K ┤          ●
         │                                      ●
     10K ┤                    ●
      5K ┤          ●                                       ●
         │  ●
      1K ┤   keep=1.0 starts here, drops sharply            ●
         │
         └──────┬──────┬──────┬──────┬──────┬──────
               4K     8K    16K    32K    64K   128K
                         Context length (tokens)

● keep=1.0:  3985 → 1993 → 996 → 498 → 249 → 125 tok/s
● keep=0.05: 79K  → 40K  → 20K → 10K → 5K  → 2.5K tok/s
```

The SVG chart with two rendered curves is at `decode_rate_chart.svg` (same directory).

---

## Step 7 — Caveats and model limitations

1. **tq3_0 dequant cost**: At decode time the GPU must dequantize 3.5-bit blocks to FP16
   before the dot-product. This adds compute that isn't in the pure BW model. The dequant
   FLOP/byte ratio for tq3_0 is low (simple 2-bit lookup + sign), so it likely stays
   BW-bound, but the effective throughput may be lower than the peak 936 GB/s figure.

2. **GQA broadcast vs re-read**: The BW formula assumes each KV head's data is read once
   per layer per step and broadcast across the (n_head / n_kv_head = 8) Q heads sharing it.
   If the CUDA kernel re-reads rather than broadcasts, true BW cost is 8× higher — though
   this would hurt uncompressed and compressed equally, leaving the speedup ratio intact.

3. **BSA vs full-attention decode path**: dflash's Block Sparse Attention forward path for
   decode may access KV blocks non-contiguously (based on the anchor-coverage mask). Random
   access to DRAM has lower effective bandwidth than sequential; the 936 GB/s figure assumes
   sequential streaming. Non-sequential access could reduce the practical ceiling.

4. **VRAM fit constraint**: At 128K context, keep=1.0 occupies 7.2 GiB KV alone; with a
   16 GiB Q4_K_M model the 24 GB card is near the limit. The theoretical model does not
   enforce this — uncompressed 128K may simply OOM.

5. **MTP (Multi-Token Prediction, γ=2) interaction**: Each verify step checks 2 draft tokens,
   requiring 2 Q vectors to attend over the full KV cache. This doubles the attention BW cost
   per accepted token on average but keeps the BW-bound regime and the compression speedup
   ratio identical.

6. **dflash hidden-state capture for MTP**: dflash captures residual hidden states for the
   MTP secondary head. This adds ~n_layer × hidden_dim × sizeof(fp16) bytes per token to
   VRAM but is not part of the decode attention BW path — it doesn't change this model.

7. **Non-attention layers**: FFN, layer norm, embeddings, and lm_head together account for
   most of the decode BW at short context. The model isolates only the KV attention BW.
   At 4K context the attention term is small relative to weight loading (~16 GiB), so
   empirical decode speedup at short context will be a fraction of the 20× theoretical figure.

---

## Compared to empirical (TODO)

The TTFT investigation agent (`a81d7cf23c7581452`) is expected to produce empirical data at
`dflash/bench/results/2026-05-21_ttft_investigation/`. Once available, compare:

- Empirical decode rate at each context = `(wall_total − TTFT) / output_tokens`
- Plot empirical points over the theoretical curves in `decode_rate_chart.svg`
- Check ratio: does empirical speedup track the theoretical 20× at 32K, or is it lower?
  If lower, estimate the fraction attributable to non-attention BW vs. dequant overhead.

**Expected finding**: empirical decode speedup at 32K will be 5–15× (not the full 20×)
because non-attention layers are not compressed and dominate at shorter contexts.

---

## Reproducibility

```bash
python3 dflash/bench/analysis/decode_rate_model.py
```

Outputs: tables to stdout, `dflash/bench/analysis/decode_rate_chart.svg`
