# Kimi K3 first-layer panel probe

## Verdict

VERDICT: BLOCKED

The implementation and fixture tests are ready. The real verdict remains
blocked only until all fourteen pinned model shards finish downloading and
pass their expected hashes. No model-quality conclusion has been drawn.

## Scientific question

For zero-indexed layer 1, can each exact routed expert be replaced by a fixed
vector plus a coordinate-wise gain on the existing 3,584-value latent input,
while preserving the direction of the aggregate routed result on held-out
sequences?

The layer computes exact latent input `z`, exact expert identities, and exact
router weights. Only the routed expert output producer may be replaced. The
router, latent down-projection, routed normalization, routed up-projection,
shared expert, attention, and residual mixing remain native.

## Frozen source

- Repository branch base: `codex/kimi-k3-dspark`
- Base revision: `e9aa6bd13702dabc042292341b46fe9f06e46734`
- Teacher repository: `unsloth/Kimi-K3-GGUF`
- Teacher revision: `a0836360ce58dfec088d966a97f2ddc8a606279b`
- Quantization: `UD-IQ1_S`, fourteen shards
- Expected total bytes: `594040923616`
- Expected shard hashes: `scripts/kimi_k3_ud_iq1s.sha256`

## Capacity-safe capture

The production loader normally places about 57.93 GiB of non-routed tensors
on one primary device. That does not fit a 24 GiB RTX 3090. Capture mode uses a
selective allocation policy instead:

1. retain the token embedding;
2. retain the complete exact dense layer 0;
3. retain layer 1 attention, residual scores, router, routed down-projection,
   routed normalization, and routed up-projection;
4. omit layer 1 shared-expert tensors and routed expert weights;
5. omit every later layer and every final-output tensor.

The forward graph stops after producing exact `z`, sixteen expert identities,
and sixteen normalized router weights. It does not request a streamed expert.
Input rows are split by sequence before tokenization. Calibration and
validation tokens from one sequence can never cross splits.

## Capture artifact version 1

The binary begins with an 80-byte little-endian header:

- magic `K3PNL001`;
- format version;
- model layer;
- latent dimension;
- routed top count;
- sequence count;
- token count;
- latent storage code (`1` means bfloat16);
- router-weight storage code (`0` means float32);
- four reserved 64-bit words.

Each record stores sequence identifier, split, token count, token identities,
token-major latent values, token-major expert identities, and token-major
router weights. A JSON sidecar records the same shape and sequence split in a
human-readable form. The binary is written to a process-specific temporary
path, synchronized, and renamed only after completion.

## Graduated run

1. 2,048 total tokens: implementation and shape smoke test.
2. 10,000 total tokens: complete small fit and held-out evaluation.
3. 50,000 to 100,000 tokens: only if the small run is correct and the
   diagonal result remains plausibly close to its registered gate.

The first smoke corpus interleaves conversational and code sequences and
assigns every fifth sequence to validation. It is an engineering smoke corpus,
not the final representativeness claim.

## Fitting variants

For each of 896 experts, load its exact quantized gate, up, and down tensors
once, evaluate all captured inputs routed to it in bounded batches, and fit:

1. fixed vector only;
2. unweighted diagonal affine;
3. squared-router-weighted diagonal affine.

Sufficient statistics use 64-bit floating point and are checkpointed after
every expert. Full individual expert outputs are not retained.

## Registered primary gate

Evaluate only held-out sequences and report the mean direction agreement
between exact and reconstructed routed aggregates.

- at least `0.9998`: `GREEN`;
- from `0.99` through less than `0.9998`: `YELLOW`;
- below `0.99`: `RED`.

Also report median and lower-tail direction agreement, relative error after
routed normalization and routed up-projection, results by router confidence,
and the worst sequences and experts. The threshold is preliminary and does
not by itself authorize converting another layer.

## Exactness and systems gates

- Observer disabled: the optimized exact evaluator remains unchanged.
- Observer enabled: the exact aggregate must be byte-identical to observer
  disabled on the numerical fixture.
- Selective capture does not claim a complete-model output.
- Before a complete exact-versus-panel comparison, exact mode must produce
  identical logits and greedy tokens before and after panel support.
- Complete runs record prompt rate, generation rate, solid-state-drive bytes,
  host memory, graphics memory, graphics power, and elapsed time. Energy is
  integrated from sampled power when a direct cumulative counter is absent.

`scripts/run_kimi_exact_baseline.sh` performs two serial exact runs, writes a
full-vocabulary float32 trace for every forward step, and requires the two
traces and their generated behavior to be byte-identical. The accompanying
comparison reports teacher-to-candidate probability divergence and top-choice
agreement; the same format will compare exact and panel modes later.

## Current tests

- `test_moe_stream_compute`: exact individual expert observation and
  byte-identical aggregate with observation enabled.
- `kimi_k3_selective_load`: pure policy test proving the intended tensor
  inclusion and exclusion boundary.

The first real capture is intentionally not runnable until the model download
is complete and the graphics card and model drive are uncontended.
