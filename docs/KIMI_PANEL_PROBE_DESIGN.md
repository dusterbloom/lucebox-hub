VERDICT: RED

# Kimi K3 first-layer panel probe

The registered 10,000-token layer-one test is complete on the pinned fourteen-
shard Unsloth checkpoint. The best registered diagonal arm reaches only
`0.624748935` mean direction agreement on held-out sequences, far below the
preliminary `0.99` yellow gate. This rules out the proposed codeword-plus-
diagonal panel as a drop-in replacement for the first routed expert bank.

The red result is numerical, not operational. Every expert was covered by the
calibration set, all model reads completed without errors or timeouts, and the
real exactness audits below pass. Exploratory follow-ups are reported
separately and do not overwrite the registered verdict.

## Executed registered result

- Corpus: 109 sequence-preserving conversational and code streams, 10,000
  tokens total.
- Calibration: 8,200 tokens; validation: 1,800 tokens. Splitting is by whole
  sequence.
- Exact routed weights read for the fit: 5,780,275,200 payload bytes, one
  complete first-layer expert bank.
- Fixed-vector mean held-out cosine: `0.618449843`.
- Unweighted diagonal mean held-out cosine: `0.624748935`.
- squared-router-weighted diagonal mean held-out cosine: `0.604720300`.
- The unweighted diagonal is best, contrary to the original expectation that
  squared route weighting would help.
- Router confidence is inversely associated with panel fidelity. The four
  confidence quartiles score `0.77261`, `0.68723`, `0.61500`, and `0.42417`
  respectively.

## Exactness lock

- Two independent real captures are byte-identical.
- Two independent complete expert passes produce byte-identical fitted panels
  and byte-identical 10,000-token exact teacher aggregates.
- The real Q6_K token-embedding host fallback is byte-identical to processor
  `GET_ROWS` on selected real rows.
- Real quantized expert evaluation is byte-identical with the observer disabled
  and enabled.
- Reconstructing the real expert output from observer callbacks is byte-
  identical to the native evaluator aggregate.
- Timing counters are intentionally not expected to be identical. The two
  repeated registered fits differed only in measured storage busy-time.

The complete capture SHA-256 is
`5dc24f94da22a854eb9d67888174abdb157fd66d6d627e53ebe5168be72d0d9d`;
the fitted panel SHA-256 is
`905624b74dce9add6251ed2eb290f3de958355c667d07e2eb6f904f3ed930b2c`;
the exact teacher aggregate SHA-256 is
`c3dbc469663ae31da0d483f0d19d31f66e6c532545097e5090355eb9b1aa2f78`.
The full paths and telemetry remain in the mounted result artifacts.

## Systems accounting

The selective 10,000-token capture took `7.19` seconds, peaked at `4,174 MiB`
of graphics memory, reported a `2.47 GiB` process high-water mark, and consumed
about `1.21 kJ` of sampled graphics-card energy. The complete expert fit took
`13.81` seconds, peaked at `3,516 MiB` of graphics memory, reported a `2.51 GiB`
process high-water mark, and consumed about `1.45 kJ`. Its exact storage engine
read `5.383 GiB` of expert payload with no errors or timeouts.

The three native-width D0-D3 replications took `36.95` seconds and about
`9.78 kJ` in aggregate. Their maximum observed graphics memory was `2,192 MiB`.
They did not read the model drive because they operate on the exported exact
teacher artifact.

Prefill rate, decode rate, and final-token probability divergence are marked
unavailable for these bounded layer-boundary experiments: they do not execute
the complete model. Reporting a token rate here would conflate capture batching
with generation and would be misleading.

## Exploratory low-memory follow-ups

These tests use an internal sequence-disjoint development subset for model
selection and preserve the original validation sequences for final reporting.
They are scientific follow-ups, not revisions to the registered primary gate.

### Direct directional refit

Training the same diagonal representation directly for aggregate direction,
rather than individual-expert squared error, improves held-out mean cosine from
`0.624748945` to `0.645053566` at a conservative learning rate. A larger
learning rate overfits immediately. The improvement is real but remains red.

### Exact fallback ladder

Keeping the highest-ranked exact experts and approximating the rest gives:

| exact experts per token | exact traffic | mean held-out cosine |
| ---: | ---: | ---: |
| 0 | 0% | 0.624749 |
| 1 | 6.25% | 0.849229 |
| 2 | 12.5% | 0.904104 |
| 4 | 25% | 0.943329 |
| 8 | 50% | 0.971404 |
| 12 | 75% | 0.987245 |
| 13 | 81.25% | 0.990640 |
| 15 | 93.75% | 0.996918 |
| 16 | 100% | 1.000000 |

The first exact expert has large value for a speculative or tiered system, but
the yellow gate requires retaining thirteen of sixteen exact expert
evaluations. No partly approximate point clears `0.9998`. Whole-token fallback
is most effective on the highest-confidence routes, matching the quartile
diagnostic, but also requires too much exact work to rescue the compression
claim.

A later exact-response export tested a stronger, still trivial selector:
router weight times the expert's calibration mean output norm, with a fixed
calibration mean for every omitted expert. This nearly saturates the held-out
oracle and improves the useful middle of the ladder:

| exact experts per token | exact traffic | mean cosine | p05 cosine | greedy-oracle mean |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 25% | 0.946460 | 0.873143 | 0.950789 |
| 8 | 50% | 0.975828 | 0.936845 | 0.978166 |
| 12 | 75% | 0.990747 | 0.973657 | 0.991812 |
| 15 | 93.75% | 0.998154 | 0.994543 | 0.998372 |

The small oracle gap means expert-subset selection is not the main remaining
problem. Eight- and twelve-exact-route points are now the first candidates for
one-layer complete-model probability-divergence tests; neither is declared
safe from this layer-boundary measurement.

### Storage-heavy and exact systems follow-ups

The optional exact-response exporter retained all 160,000 individual routed
answers from the same 10,000-token capture without changing the exact teacher
checksum. It enabled lookup, response-subspace, sparse-route, locality,
lossless-compression, and internal-channel controls. The full measured map and
next-road decision are in `docs/KIMI_EXPERT_TERRITORY.md`.

The central negative controls are:

- every calibration answer plus latent-cosine interpolation reaches only
  `0.752808` mean and `0.250058` p05 cosine;
- even a perfect output-cosine address over those stored answers reaches only
  `0.781160` mean and `0.364061` p05, so address learning is not the dominant
  missing piece at this scale;
- an optimistic rank-64 response subspace fitted separately per expert reaches
  `0.810673` mean and `0.439854` p05 before any address error;
- Zstandard makes sampled real IQ1_S expert components `0.0028%` larger;
- a 64-expert-per-layer least-recently-used cache hits `24.03%` and projects to
  `35.37 GiB` across all layers;
- oracle internal-channel pruning can save at most one third of expert bytes,
  because gate and up must be read before active down channels are known.

These results reject plain “store the answer” and generic exact compression as
the missing solution. They support an exact-subset cascade followed by one
learned aggregate omitted-tail model and adaptive exact fallback.

### Real Kimi shared nonlinear D0-D3

The Smol-Kimi architecture was transferred directly: one shared native-width
SiTU-GLU core, then router-mixed shift and scale cards. Three native-width
seeds give:

- shared core D0 mean: `0.636372`;
- full scale-and-shift D3 mean: `0.643665`;
- D3 range: `0.641327` to `0.644986`;
- permuted-card control mean: `0.629960`;
- uniform-route control mean: `0.641052`.

The expert cards carry real route-specific information because permutation
hurts, but their contribution is small. Shift provides almost all of the gain;
scale alone is negligible.

A width bracket also flattens quickly:

| shared hidden width | D3 held-out cosine | BF16 bytes per layer |
| ---: | ---: | ---: |
| 768 | 0.638574 | 29,360,128 |
| 3,072 | 0.641327 | 78,905,344 |
| 6,144 | 0.647432 | 144,965,632 |

Doubling beyond native expert width buys only about `0.0061` cosine while
nearly doubling storage. Blindly increasing shared-core capacity is therefore
not a credible route to the target.

## What is still blocked by current WSL memory

This layer-boundary capture cannot measure final-token probability divergence,
top-choice agreement, generation quality, or autoregressive stability because
it stops before the remaining ninety-one model layers. Those are deliberately
recorded as unavailable, not zero. The next valid gate is full exact end-to-end
execution after WSL has enough memory to place the roughly 58 GiB non-routed
core across host and graphics memory.

The current evidence changes the next full-model experiment: do not implement
an all-layer diagonal panel. First validate exact end-to-end execution and its
bit-exact repeat, then test at most one layer with the strongest D3 artifact or
a one-exact-expert hybrid while recording final-token probability divergence.

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
every expert. Registered runs do not retain full individual expert outputs. An
explicit optional research directory can export them as atomic per-expert
files; leaving that argument absent preserves registered behavior and cost.

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
- Real fit startup: host-decoded embeddings versus processor `GET_ROWS`, and
  observer-disabled versus observer-enabled real quantized experts.
- `train_kimi_panel_directional.py`: sequence-disjoint aggregate-direction
  refit with early stopping.
- `evaluate_kimi_panel_fallback.py`: exact-rank and confidence fallback
  ladders.
- `train_kimi_d0_d3.py`: shared-core D0-D3 screen, causal controls, capacity
  accounting, and deterministic seeds.

## Progressive slab follow-up

The matched-byte campaign changes the preferred mixed provider. Each
3,072-neuron expert was split into twelve byte-exact 256-neuron slabs.
Calibration-only importance followed by global router-weighted water filling
selects 96 of 192 active slabs at 50% bytes. It reaches `0.976688` mean and
`0.939117` p05 held-out cosine, improving on eight complete experts at
`0.975828` / `0.936845`. At 75%, 144 adaptive slabs reach `0.991022` /
`0.974435`, also improving on twelve complete experts.

A 5,780,303,872-byte layer-one sidecar reordered the unchanged IQ1_S bytes into
per-expert importance prefixes. Direct NVMe reads sustained `5.632` and `5.280
GiB/s`, versus `5.388` and `5.201 GiB/s` for matched-byte whole experts. The
finer allocation therefore survives the first physical layout test. The next
complete-model gate should test 96 and 144 slab prefixes, retaining 8 and 12
whole experts as equal-byte controls.
