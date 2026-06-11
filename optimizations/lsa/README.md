# Qwen3.5 LSA Encoder

This directory contains the offline half of the minimum viable Qwen3.5
Lookahead Sparse Attention implementation. It deliberately does not modify the
production Qwen graph yet.

## Frozen Contract

- Query input: Qwen3.5 post-layer-46 hidden state, BF16, 5120 values. Lucebox
  already captures this layer in the DFlash feature ring.
- Block key: layer 47 K after K RMS normalization and before M-RoPE.
- Key geometry: four KV heads by 256 values, mean pooled over a sealed
  64-token block and L2-normalized per head.
- Retrieval cadence and future-label horizon: 64 committed tokens.
- Oracle: exact causal attention mass over future queries from all 16
  full-attention layers. Forced sink and last-8K blocks are excluded from loss.
- Encoder: `Linear(5120, 256)`, SiLU, `Linear(256, 1024)`, then per-head
  normalization. The two matrices contain 1,572,864 parameters, about 3 MiB
  in BF16 or 1.5 MiB in INT8.

Do not derive frozen keys from the current KV cache. Qwen3.5 stores K after
M-RoPE and the backend can additionally FWHT-transform it; either operation
breaks the position-independent block-key contract.

The opt-in graph capture exposes post-M-RoPE Q/K for teacher labels and
pre-M-RoPE K for index-key pooling. `oracle.py` retains the full causal
softmax denominator, bins only cold-history mass, abstains when total cold mass
is below `0.02`, applies per-layer top-p selection, and requires three Qwen
full-attention layers to vote for a positive.

## Runtime Integration Boundary

LSA applies only to the 16 full-attention layers. Qwen3.5's DeltaNet SSM and
convolution state are separate from attention K/V and should remain exact;
there is no reason to retrieve, approximate, or replay old SSM states during
normal decode.

The first production oracle should keep dense K/V as an archival source and
build a fixed-capacity packed active K/V view with original token positions.
That intentionally saves no memory, but it lets an all-chunks selection prove
logit and token parity before CPU offload is introduced. Packed state is
derived: rebuild it after prefix restore instead of extending snapshot formats.
Freeze one selection across DFlash verification and accepted-token replay, and
advance the 64-token cadence only from committed tokens. If FlowKV has already
rewritten the effective prompt, construct the LSA catalog from that resulting
token sequence.

## NPZ Shard V1

One shard contains one source sequence:

| Array | dtype and shape | Meaning |
|---|---|---|
| `metadata_json` | `uint8 [bytes]` | Versioned geometry and tap contract |
| `block_keys` | `float16 [blocks, 4, 256]` | Frozen pre-RoPE block keys |
| `query_hidden_bf16` | `uint16 [examples, 5120]` | Raw BF16 query bits |
| `boundary_pos` | `int32 [examples]` | Committed-token boundary |
| `visible_blocks` | `int32 [examples]` | Candidate blocks at the boundary |
| `label_offsets` | `int64 [examples + 1]` | CSR offsets into `label_mass` |
| `label_mass` | `float16 [sum(visible)]` | Oracle mass in `[0, 1]` |

NPZ is the training interchange format. The future extractor should write
sequence-sharded raw arrays plus checksums and convert them to NPZ, so a crash
does not require rebuilding a monolithic archive.

The raw writer now uses `luce.lsa.qwen35.raw.v1`. It appends pre-RoPE K from
layer 47 and post-RoPE K from every oracle layer for all tokens, but writes
post-RoPE Q and the layer-46 hidden only for explicitly selected, block-aligned
boundaries. This is important: archiving Q for every token would dominate the
dataset size. Each finalized directory contains a geometry manifest, file
sizes, and FNV-1a checksums.

Training emits a `luce.lsa.qwen35.encoder.v1` directory containing
`encoder.json` and `encoder.f16.bin`. The F16 file is the single source of
truth for both Python evaluation and the C++ runtime loader; the manifest
records tensor shapes, offsets, and an FNV-1a checksum. This avoids evaluating
an FP32 checkpoint that differs from the deployed compact encoder.

Extraction and conversion are separate so GPU capture can be reprocessed
without another target-model forward:

```bash
flock /tmp/dflash_gpu.lock build/lsa_extract_qwen35 \
  model.gguf prompt_tokens.i32 /tmp/lsa-sequence boundaries.txt
python3 raw_dataset.py /tmp/lsa-sequence \
  --output /tmp/lsa-sequence.npz --device cuda
```

`boundaries.txt` contains one committed-token position per line. Positions
must be 64-token aligned and leave a complete 64-token future window. The
extractor always captures historical K, but archives future Q only at those
positions. Omitting the file chooses up to eight distributed boundaries.

## Smoke Test

```bash
cd optimizations/lsa
python3 -m unittest -v test_lsa.py
python3 make_synthetic.py /tmp/lsa-synthetic.npz \
  --hidden-size 64 --head-dim 16
python3 train.py /tmp/lsa-synthetic.npz \
  --output /tmp/lsa-encoder --device cpu --rank 16 --max-steps 4
python3 evaluate.py /tmp/lsa-synthetic.npz \
  --model /tmp/lsa-encoder --device cpu
```

## Minimum Useful Dataset

Start with a 64-document pilot: 24 at 16K, 20 at 32K, 16 at 64K, and 4 at
128K. This is 2.62M source tokens and 512 stratified boundaries. Split 48/8/8
by source, train three seeds, and budget roughly 2-3 RTX 3090 GPU-hours. The
teacher extraction dominates; compact-encoder training should take minutes.

If the pilot passes, scale to 384 documents and 16.78M source tokens:
128 at 16K, 128 at 32K, 96 at 64K, and 32 at 128K. Split 288/48/48 by source.
The working budget is 15-18 RTX 3090 GPU-hours including extraction and three
training seeds, with substantial variance from the future oracle kernel.

Suggested source mixture:

- 40% long agent and coding traces with repeated files and tool output.
- 30% synthetic needle, multi-hop, and scattered evidence retrieval.
- 20% long technical documents and repositories.
- 10% local-only or no-context controls to measure false retrieval.

The first gate is mass-recall at a fixed block budget against random,
recency-only, and direct hidden-to-key projection baselines. Do not integrate
the learned selector into production unless it wins that offline gate and an
all-chunks packed-KV oracle reproduces dense Qwen logits.

PR274/FlowKV prefix snapshots remain useful infrastructure, but they restore a
complete compressed-prefix state, including exact DeltaNet SSM and convolution
state. They are not an arbitrary sparse-block KV store. LSA should reuse their
validated tensor strip-copy and snapshot ownership patterns while keeping a
separate archival full-attention KV source and fixed-capacity active view.

Reject the pilot if it misses 90% oracle-mass recall at 20% cold-history keep,
80% recall at 10% keep, or retains more than 5% of cold chunks on local-only
examples. Also require a 30-point advantage over random and distance-only
baselines at the same keep ratio.
