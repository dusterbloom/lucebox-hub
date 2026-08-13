# Kimi K3 routed-expert territory after the first real geometry campaign

## Outcome

The real layer-one data rules out a broad class of tiny bank-free substitutes,
but reveals one credible performance path: allocate exact bytes adaptively over
important internal expert-neuron slabs and approximate their aggregate remainder.

The distinction matters. A full replacement tries to infer

\[
u=\sum_{i=1}^{16}p_i E_i(z)
\]

without reading the expert bank. The measured hybrid instead computes a set
\(S\) exactly and estimates only

\[
r_S=\sum_{i\notin S}p_i E_i(z).
\]

On held-out real Kimi layer-one streams, one calibration scalar per expert and
slab is enough to improve on whole-expert allocation at the same bytes. This
changes the shortest road from “find a miraculous tiny expert model” to “find
the smallest progressive exact-byte fraction whose complete-model probability
divergence is safe.”

These are exploratory layer-boundary measurements. They do not revise the
registered diagonal-panel verdict, and they cannot report final-token
probability divergence until complete-model execution is available.

## Measured map

| mechanism | best measured point | held-out directional result | physical meaning | disposition |
| --- | ---: | ---: | --- | --- |
| fixed answer per expert | 0.55 GiB across 92 layers | mean `0.61845`, p05 `0.23310` | about 10 MiB response traffic per token | rejected as replacement |
| diagonal affine panel | 1.18 GB across 92 layers | mean `0.62475` | original panel proposal | registered **RED** |
| native-width D3 | about 6.8 GiB across 92 layers | mean `0.64366` | shared nonlinear core plus expert cards | rejected as replacement |
| shared-input linear chart | rank 16, 9.37 GiB | mean `0.68190`, p05 `0.24586` | one shared input basis, expert-specific output charts | rejected as replacement |
| stored exact-answer atlas | all 131,200 calibration routes | mean `0.75281`, p05 `0.25006` | latent-cosine address, 16-answer interpolation | rejected at 10K scale |
| perfect-address atlas control | same stored answers | mean `0.78116`, p05 `0.36406` | oracle addresses by the unknown exact answer | coverage, not address, is the main limit |
| one shared response basis | rank 256 | mean `0.71713`, p05 `0.43271` | only 37.1% of centered response energy retained | no tiny universal answer dictionary |
| separate response basis per expert | rank 64, about 34.84 GiB | mean `0.81067`, p05 `0.43985` | optimistic projection using the exact answer | insufficient even before address error |
| exact subset plus mean tail | 8 of 16 experts | mean `0.97583`, p05 `0.93685` | 50% exact expert traffic | first serious full-model candidate |
| exact subset plus mean tail | 12 of 16 experts | mean `0.99075`, p05 `0.97366` | 75% exact expert traffic | conservative full-model candidate |
| adaptive 256-neuron slabs plus slab-mean tail | 96 of 192 active slabs | mean `0.97669`, p05 `0.93912` | 50% exact weight bytes; 6.60 GiB all-layer BF16 slab means | best measured equal-byte point |
| adaptive 256-neuron slabs plus slab-mean tail | 144 of 192 active slabs | mean `0.99102`, p05 `0.97444` | 75% exact weight bytes | best measured conservative point |
| published uniform half-width bank | 16 half-width experts | mean `0.90902`, p05 `0.82271` | 50% exact weight bytes | rejected at equal bytes |
| published half-width + 8 full refinements | 8 half + 8 full experts | mean `0.98855`, p05 `0.96750` | 75% exact weight bytes | dominated by adaptive slabs and 12 full experts |
| progressive slab sidecar, direct NVMe | 96 slabs, 128-token trace | 5.63 / 5.28 GiB/s repeats | 1,399 reads versus 1,024 whole-expert reads | physical layout gate passed |
| oracle active internal channels | strongest 25% | individual mean `0.97648` | ideal byte fraction 75%, at most 1.33 times faster | secondary only |
| oracle active internal channels | strongest 50% | individual mean `0.99684` | ideal byte fraction 83.3%, at most 1.20 times faster | secondary only |
| generic lossless compression | Zstandard levels -5, 1, 3 | compressed fraction `1.000028` | sampled IQ1_S bytes become larger | conclusively rejected |
| previous-token route reuse | one preceding token | recall `12.93%` | exact route set never repeats in the sample | weak |
| transition prefetch | 32 candidates | recall `33.53%` | may hide some latency, removes no bytes | auxiliary |
| exact expert cache | 64 experts per layer | hit rate `24.03%` | projects to 35.37 GiB across 92 equal caches | useful but not transformative |

The underlying result files are:

- `results/kimi_layer01_geometry_campaign.json` (headline and telemetry index);
- `results/kimi_layer01_response_atlas.json`;
- `results/kimi_layer01_shared_input_charts.json`;
- `results/kimi_layer01_response_basis.json`;
- `results/kimi_layer01_per_expert_response_basis.json`;
- `results/kimi_layer01_sparse_exact_routes.json`;
- `results/kimi_layer01_route_locality.json`;
- `results/kimi_layer01_lossless.json`;
- `results/kimi_layer01_expert_channels.json`.
- `results/kimi_layer01_neuron_slabs.json`;
- `results/kimi_layer01_halfwidth_frontier.json`;
- `results/kimi_layer01_halfwidth_source.json`;
- `results/kimi_layer01_slab_io.json`.

## The progressive-slab result

Each full Kimi routed expert is exactly additive over 3,072 internal neurons.
IQ1_S makes 256 neurons a natural byte-aligned unit, giving twelve slabs per
expert and 192 active slabs per token. Twelve independently dequantized slabs
reconstruct one unsplit dequantized expert at mean cosine effectively `1.0`
and mean relative error `7.01e-7`.

At exactly 50% routed weight bytes:

| allocation | mean cosine | p05 cosine | post-up mean cosine |
| --- | ---: | ---: | ---: |
| eight complete experts + expert-mean tail | `0.975828` | `0.936845` | `0.971578` |
| uniform six slabs from every expert | `0.892040` | `0.760674` | `0.883697` |
| adaptive 96 slabs + slab-mean tail | **`0.976688`** | **`0.939117`** | **`0.972841`** |
| held-out residual-norm diagnostic | `0.981070` | `0.949815` | `0.978457` |

The deployable selector is only router weight times each slab's calibration
mean residual norm. Because slab importance has a fixed order inside an expert,
global selection produces a prefix length per active expert. A repacked runtime
therefore needs at most one progressive range per active expert rather than 96
unrelated reads.

The real layer-one sidecar contains 5,780,303,872 bytes and preserves every
IQ1_S weight byte, merely reordering it. Over two direct-I/O passes on 128 held-
out tokens, adaptive prefixes sustained `5.632` and `5.280 GiB/s`, versus
`5.388` and `5.201 GiB/s` for eight whole experts. Alignment overhead was only
1,183,744 bytes over 6.606 GB. Thus the finer allocation did not sacrifice
storage bandwidth in the first physical test.

The BF16 slab-mean table costs 6.60 GiB across all 92 layers. That is material
but modest beside the 495 GiB routed bank, and it can be replaced later by a
smaller aggregate-tail model if end-to-end probability divergence passes.

## Published half-width model control

The pinned `vcruz305/Kimi-K3-GGUF` layer-one range confirms a 1,536-neuron
expert width with all sixteen routes retained. On the same full-teacher latent
states and routes, it reaches only `0.90902` mean and `0.82271` p05 cosine at
the same 50% routed bytes. Replacing its four most discrepant experts with full
experts reaches `0.97742` at 62.5% bytes; replacing eight reaches `0.98855` at
75%. Both are below the adaptive-slab curve near the same budgets.

This does not judge the published model on its own co-adapted hidden-state
distribution. It answers the narrower question relevant to Lucebox: uniform
half-width experts are not the best drop-in provider for the full-width teacher.

## What the geometry says

### 1. The answer manifold is not tiny

The diagonal panel, shared chart, shared response basis, and per-expert response
basis form a consistent ladder. More degrees of freedom improve the mean, but
the lower tail remains far from the proposed gate. The routed functions vary in
many directions on real inputs. This is not primarily a bad optimizer or an
incorrect magnitude loss.

The per-expert basis outperforms the universal basis, so experts do have
distinct local geometry. But even an optimistic rank-64 projection—which is
allowed to see the exact validation answer before projecting it—reaches only
`0.81067` aggregate cosine. An implementable address-to-coordinate model can
only be worse at the same rank.

### 2. A better lookup address is not the missing breakthrough

With every calibration response stored, latent-cosine interpolation reaches
`0.75281`. A perfect but impossible output-cosine address reaches `0.78116`.
The oracle gain is real but small compared with the gap to `0.99`. At this data
scale, the stored answers do not cover held-out responses densely enough.

The full atlas also projects to `161.16 GiB` across 92 layers when both bfloat16
addresses and answers are counted. A naive exhaustive address scan would read
about `1.44 GiB` of addresses per generated token before retrieving answers.
Larger calibration would improve coverage only while increasing both costs.

Learned or indexed addressing remains technically underexplored, but the oracle
control moves it below the exact-subset hybrid in priority.

### 3. Routed contributions and internal slabs are unequal even when expert load is balanced

The router emits sixteen experts, but their actual contribution to the final
sum is not equal. Selecting by

\[
p_i\,\mathbb{E}_{\mathrm{cal}}\|E_i(z)\|
\]

beats native router order and nearly saturates the exact-contribution oracle.
At eight exact routes the cheap selector reaches `0.97583` versus the greedy
oracle's `0.97817`; at twelve it reaches `0.99075` versus `0.99181`.

The slab experiment sharpens this: whole experts are not the optimal atomic
unit. At every matched whole-expert point—25%, 50%, and 75%—adaptive slabs
improve both mean and lower-tail direction. The gains are modest but consistent,
and a residual-norm diagnostic shows additional selection headroom.

This means selection itself does not need a large model. The open problem is
whether the remaining directional error survives routed normalization, the
following 91 layers, and final token probabilities.

### 4. Exact byte shortcuts are scarce

IQ1_S expert components are already entropy-dense. Independent lossless frames
over 16 representative experts saved no space at any tested Zstandard level.

Internal channel sparsity is genuine—half the channels retain over 99% of
activation energy—but gate and up weights must be read before the exact active
channels are known. Only the down projection can then be pruned. Since the
three components occupy equal bytes, keeping a fraction \(f\) of down channels
has the ideal lower bound

\[
B(f)=\frac{2+f}{3}B_{\mathrm{expert}}.
\]

Even an oracle therefore gives at most `1.33x` at 25% channels and `1.20x` at
50%, before quantization-block rounding and scattered-read overhead.

### 5. Small exact caches fight the model's balancing mechanism

Layer-one routes have little temporal locality: the previous token recalls
`12.93%` of the next token's selected experts, and no consecutive pair in the
sample repeats the complete 16-expert set. A 64-entry least-recently-used cache
hits `24.03%`. Caching and prefetching remain worthwhile systems work, but they
cannot be the compression mechanism.

## Territory by objective

| objective | viable now | underexplored | measured dead end |
| --- | --- | --- | --- |
| preserve exact arithmetic | asynchronous same-token reads, larger device/host cache, multi-device striping, cross-layer route prefetch | block-aligned down-channel paging; direct-input access and storage layout | generic lossless compression; small hot cache as sole answer |
| halve routed traffic | eight exact experts chosen by expected contribution, mean or learned aggregate tail | direct training of the aggregate tail residual; token-adaptive 8/12/16 budget | top-eight plus current diagonal as a quality assumption without full-model test |
| delete most expert storage | none validated | expert-axis weight dictionaries; block vector quantization; nonlinear aggregate residual trained across layers | diagonal panel, present D3, plain response atlas, small response bases |
| avoid latency but not bytes | demand prefetch as soon as same-layer router is known; overlap transfer and compute | predict future-layer routes from earlier hidden states; multi-token speculative fetch | previous-token identity alone |
| exploit repeated workloads | exact prefix and state cache; prompt-specific answer memoization | persistent application-domain response cache with exact miss fallback | general response lookup from 10K calibration answers |
| co-adapt a smaller model | sequential, multi-token distillation of an aggregate tail or reduced expert budget | recurrent context-conditioned residual, then all-layer training | assuming local reconstruction automatically composes |

## The shortest next road

1. Increase WSL memory and lock a repeated, byte-identical complete exact run.
   Record prompt and generation rates, solid-state-drive and memory traffic,
   graphics memory, energy, output identities, and the full-vocabulary logits.

2. Add only one layer-one progressive provider. Test two frozen points:

   - 96 adaptive exact slabs plus a slab-mean tail;
   - 144 adaptive exact slabs plus a slab-mean tail.

   Retain eight and twelve whole experts as equal-byte controls.

   Measure final-token probability divergence and top-choice agreement against
   exact execution. Do not convert a second layer yet.

3. If eight or twelve passes the complete-model gate, train a compact model for
   the **aggregate omitted tail**, not sixteen individual expert surrogates.
   The target is \(r_S\) after the exact selected contributions are removed.
   This is a smaller and better-conditioned problem than reproducing the whole
   bank. Keep exact token identities during short multi-token unrolls and let
   student state compose.

4. Only after a one-layer final-logit pass, measure sequential depth. Choose
   per-layer exact budgets from observed sensitivity rather than forcing the
   same fraction across all 92 layers.

5. In parallel, benchmark exact systems improvements that do not change model
   behavior: route-to-read overlap, cross-layer prefetch, host cache sizing, and
   storage striping. Treat their benefit as latency reduction, not compression.

## How the outlook changes

The approximately 58 GB bank-free deployment is not supported by current real
Kimi evidence. All tested compact response models remain too inaccurate at the
first routed layer, including storage-heavy answer lookup and optimistic
low-rank response projections.

A faster 594 GB deployment is more plausible. Adaptive slabs cut the nominal
`8.844 GiB` routed payload per token in half while retaining `0.97669`
layer-one direction, or retain `0.99102` at 75% traffic. Whether either point improves
useful generation speed without unacceptable probability divergence is now an
end-to-end question, not a geometry question.

The strongest research bet is therefore a mixed-rate cascade:

\[
\boxed{\text{cheap slab importance ranking}
\rightarrow \text{progressive exact prefixes}
\rightarrow \text{one learned aggregate tail}
\rightarrow \text{adaptive exact fallback}}
\]

It is less spectacular than deleting every expert, but it is now the first path
whose quality, byte allocation, repacked storage layout, and direct-NVMe
bandwidth are all supported by measurements on the real Kimi checkpoint.
