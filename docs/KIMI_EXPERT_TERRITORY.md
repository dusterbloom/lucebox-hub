# Kimi K3 routed-expert territory after the first real geometry campaign

## Outcome

The real layer-one data rules out a broad class of tiny bank-free substitutes,
but reveals one credible performance path: compute a deliberately chosen subset
of routed experts exactly and approximate their aggregate remainder.

The distinction matters. A full replacement tries to infer

\[
u=\sum_{i=1}^{16}p_i E_i(z)
\]

without reading the expert bank. The measured hybrid instead computes a set
\(S\) exactly and estimates only

\[
r_S=\sum_{i\notin S}p_i E_i(z).
\]

On held-out real Kimi layer-one streams, one calibration scalar per expert is
already enough to choose \(S\) almost as well as an oracle. This changes the
shortest road from “find a miraculous tiny expert model” to “find the smallest
exact traffic fraction whose complete-model probability divergence is safe.”

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

### 3. Routed contributions are unequal even when expert load is balanced

The router emits sixteen experts, but their actual contribution to the final
sum is not equal. Selecting by

\[
p_i\,\mathbb{E}_{\mathrm{cal}}\|E_i(z)\|
\]

beats native router order and nearly saturates the exact-contribution oracle.
At eight exact routes the cheap selector reaches `0.97583` versus the greedy
oracle's `0.97817`; at twelve it reaches `0.99075` versus `0.99181`.

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

2. Add only one layer-one mixed-rate provider. Test two frozen points:

   - eight exact experts selected by expected contribution plus a mean tail;
   - twelve exact experts selected the same way plus a mean tail.

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

A faster 594 GB deployment is more plausible. Eight exact routes cut the
nominal `8.844 GiB` routed payload per token in half while retaining `0.97583`
layer-one direction with a negligible mean-tail table. Twelve routes retain
`0.99075` while cutting traffic by one quarter. Whether either point improves
useful generation speed without unacceptable probability divergence is now an
end-to-end question, not a geometry question.

The strongest research bet is therefore a mixed-rate cascade:

\[
\boxed{\text{cheap contribution ranking}
\rightarrow \text{selected exact reads}
\rightarrow \text{one learned aggregate tail}
\rightarrow \text{adaptive exact fallback}}
\]

It is less spectacular than deleting every expert, but it is the first path
whose quality and physical savings are both supported by measurements on the
real Kimi checkpoint.
