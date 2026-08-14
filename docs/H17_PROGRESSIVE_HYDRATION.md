# H17: Whole-model progressive hydration

**STATUS: ARITHMETIC IDENTITY RECOVERED — the original split all-192 path
failed, but sidecar recomposition through the native full-width kernel is now
byte-identical through all 92 routed layers and at terminal logits. Partial
96/144 budgets remain open.**

H17 asks whether progressive 256-neuron routed-expert blocks survive on-policy
composition through all 92 routed Kimi K3 layers. It uses the production Kimi
forward path and the immutable exact NVMe expert evaluator.

## Locked exact baseline

Two passes over 12 whole-sequence prompts produced 126 full-vocabulary logit
rows. Every row is byte-identical across the two passes: maximum logit
difference and teacher-to-reference KL are zero. The passes are stored under
`/mnt/kimi-k3/results/kimi-h16-registered/exact-a` and `exact-b`.

## Gate order

1. All 192 active slabs at every routed layer.
2. Static calibrated prefixes at budget 96.
3. Greedy prefix oracle at budget 96.
4. Greedy prefix oracle at budget 144.

The all-192 control uses natural physical neuron order. This does not change
selection because every slab is read, and it avoids fabricating calibration
statistics for layers that have not yet been calibrated. Before its execution,
the terminal numerical gate was locked to:

- mean teacher-to-candidate KL at most `1e-6`;
- maximum KL at most `1e-5`;
- exact top-choice agreement on every scored row.

Failure stops H17 before partial-budget calibration. Passing permits creation
of real all-layer means and fixed importance orderings for the 96/144 runs.
The shadow control reads both the native expert and every slab, so it has no
expected speedup.

## All-192 result

The control executed on-policy through all 92 routed layers at commit
`8ec862e38c9853987ae450c2ffee83c859347ddb`. It retained every one of the 192
active 256-neuron blocks at every routed layer. The prompt was `According to all
known laws`; four teacher-forced continuation tokens produced eight aligned
full-vocabulary logit rows.

| measurement | result | gate |
| --- | ---: | ---: |
| mean teacher-to-candidate KL | `0.0397783268` | `<= 1e-6` |
| maximum KL | `0.2749676223` | `<= 1e-5` |
| maximum absolute logit difference | `2.55742836` | diagnostic |
| top-choice agreement | `8/8` | `8/8` |

This is a measured **STOP**. Static-96, prefix-oracle-96, and
prefix-oracle-144 were not run, exactly as required by the locked protocol.
The generated continuation under aligned teacher tokens remained
`\nof aviation,\n`, but this is not a free-generation quality result.

The control also produced an important falsification result. At each layer,
the progressive evaluator's all-slab routed output was compared with the native
exact routed output on the candidate's current state and native routes:

| local measurement across 92 routed layers | result |
| --- | ---: |
| mean of per-layer mean relative L2 | `2.4571771e-7` |
| worst token relative L2 | `3.37597e-7` (layer 28) |
| printed mean cosine | `1.0` at every layer |

Therefore the stored weights, mixed quantization handling, and additive slab
decomposition agree locally to floating-point roundoff. Nevertheless, the
different accumulation path is not behaviorally identical after recurrent
composition through depth and token state.

The control took 237.01 seconds, read 520.09 GB from the drive, peaked at
17,066 MiB of graphics memory, used about 0.88 GB anonymous host memory plus
25.4 GB file-backed mappings, and consumed 26.78 kJ of measured graphics-board
energy. It is a deliberately doubled shadow path, not a speed measurement.

Machine-readable summary: `results/kimi_h17_all_slabs_control.json`. Raw
artifacts and their checksums are under
`/mnt/kimi-k3/results/kimi-h17-all-slabs-control`.

## Divergence localization

A paired native-versus-all-192 trace then captured 736 routed-layer records for
the same frozen prompt. The traced native run remained byte-identical to the
locked exact logits, so the instrumentation itself did not perturb the teacher.

| event | first layer | measurement at that point |
| --- | ---: | --- |
| numerical difference | 1 | routed latent relL2 `2.88317e-7`, maxabs `4.09782e-8` |
| large conditional amplification | 4 | routed latent relL2 `1.98385e-4`; route set still identical |
| router ordering change | 6 | set overlap `16/16`; pre-router hidden relL2 `0.00445606` |
| router membership change | 7 | set overlap `15/16`; pre-router hidden relL2 `0.00409431` |

The error does not grow smoothly. Layers 1--3 remain near rounding error. At
layer 4 the routed result jumps by roughly three orders of magnitude without a
route change, layer 5 grows further, layer 6 swaps two route ranks, and layer 7
changes one expert and produces a routed-output relL2 of `0.181744`. The
post-MoE state and the next layer input were byte-identical within each run, so
there is no hidden copy or insertion discrepancy between those boundaries.

The likely cause is now narrow. The native path performs one 3,072-neuron down
reduction for each expert, applies its router weight once, and accumulates the
16 experts. The original slab path performs twelve independent 256-neuron down
reductions, applies the router weight to each partial result, and accumulates
192 partial results. These are algebraically equal but not floating-point
equivalent.

One surgical control grouped the twelve slab outputs per expert, applied the
router weight once, and accumulated experts in native expert-ID order. It
improved but did not solve parity:

| measurement | direct 192 | grouped 192 |
| --- | ---: | ---: |
| layer-1 routed relL2 | `2.88317e-7` | `1.01414e-7` |
| first route membership change | layer 7 | layer 8 |
| terminal mean KL | `0.0397783` | `0.0162708` |
| terminal maximum KL | `0.274968` | `0.0992338` |

This falsifies router-weight placement as the entire explanation while showing
that it materially contributes. The remaining mismatch is consistent with the
twelve split down reductions themselves. Exact identity likely requires the
native full-width reduction semantics, not another selector or tail model.

Machine-readable localization summary:
`results/kimi_h17_divergence_localization.json`. Reproducible raw traces,
logits, telemetry, and checksums are under
`/mnt/kimi-k3/results/kimi-h17-divergence-trace` and
`/mnt/kimi-k3/results/kimi-h17-grouped-parity`.

## Pre-down isolation and arithmetic identity recovery

An environment-gated probe at layer 1, token 0 captured the complete
3,072-value post-gate/up activation for every one of the 16 active experts. It
also concatenated the twelve corresponding 256-neuron slab activations and
ran both vectors through the same native full-width down-projection kernel.

| measurement | result |
| --- | ---: |
| native versus concatenated-slab activation | bit-identical for `16/16` experts |
| native activation versus slab activation through native full down | bit-identical for `16/16` experts |
| standalone native aggregate versus production routed aggregate | bit-identical |
| twelve split down projections, aggregate relL2 | `9.3729836e-8` |
| twelve split down projections, aggregate maxabs | `2.2351742e-8` |

This is decisive. Gate/up tiling, quantized byte slicing, SiTU activation, and
the slab ordering are exact. The mismatch begins only when one 3,072-neuron
down reduction is replaced with twelve separately rounded 256-neuron
reductions. It is execution-arithmetic sensitivity, not semantic slab error.

The next control rebuilt complete gate/up/down tensors from the natural slab
sidecars and executed them through the unchanged full-width expert arithmetic.
It intentionally reads all slab bytes and has no speed claim. On both a fresh
one-token control and the original frozen eight-row trajectory:

| identity measurement | result |
| --- | ---: |
| routed/local trace across all 92 layers | byte-identical |
| terminal logits | byte-identical |
| maximum absolute logit difference | `0` |
| teacher-to-candidate KL, mean/median/maximum | `0 / 0 / 0` |
| top-choice agreement | `8/8` |

The native and recomposed frozen logits share SHA-256
`66ab2dea90fd84d991170fcde70255a36ae9f93c2362f61551286af369bea3b4`.
The complete traces share SHA-256
`1117017d54bb70295b7692176b18a10204b70f0a57027675142f8d7f6cd14db8`.
Raw artifacts are under `/mnt/kimi-k3/results/kimi-h17-predown-probe`,
`/mnt/kimi-k3/results/kimi-h17-recomposed-control`, and
`/mnt/kimi-k3/results/kimi-h17-recomposed-frozen`. The machine-readable
summary is `results/kimi_h17_arithmetic_identity.json`.

The frozen recomposition run took 261.75 seconds, peaked at 17,072 MiB of
graphics memory, consumed 30.51 kJ of measured board energy, and caused about
509.18 GB of drive reads because the research provider deliberately evaluates
the exact teacher and rebuilds every active expert. These are control costs,
not projected serving costs.

## Exploratory natural-prefix screen

At the user's request, a practical free-generation screen also ran a natural
six-of-twelve prefix per expert with a zero omitted tail. It is **not** the
calibrated all-layer selector and used the old split arithmetic, so it remains
**EXPLORATORY — confounded by all-192 arithmetic divergence**.

It matched `0/12` native token sequences, triggered the simple degeneration
flag on `2/12`, and produced first-position KL values from about `0.167` to
`0.428`. A small answer checker scored `4/12`, equal to native and preserving
all four native successes, but that weak task score does not outweigh the
large distributional divergence. This specific zero-tail natural-prefix mode
is not viable.

## Interpretation and next gate

The arithmetic confound is now removable. A partial provider should place
selected slab bytes in their native neuron positions, represent omitted
positions without changing the 3,072-neuron kernel shape, and perform one
full-width down reduction per expert. This retains native accumulation order
while permitting physical reads to remain progressive. The first such control
should use budget 96 and no learned component; its purpose is to separate true
omission error from the eliminated split-reduction error.

The 96/144 quality hypothesis remains neither validated nor falsified. The
next run must first prove that its 192-budget form stays byte-identical and then
compare terminal KL at the matched partial budget. Fisher, Observer, and tail
training remain deferred until this arithmetic-stable partial provider is
measured.

## Deferred Observer constraint

Observer work is explicitly outside H17. If progressive hydration survives:

1. private Observer state predicts which progressive prefixes remain live;
2. the runtime reads and evaluates those exact slabs;
3. only if needed, an aggregate omitted-tail correction is predicted while
   conditioning on the exact live result;
4. private state never enters K3's residual or latent stream directly—only the
   expert-shaped correction may enter the existing MoE output slot.

This ordering prevents a later Observer from replacing the exact computation
that H17 is designed to preserve.
