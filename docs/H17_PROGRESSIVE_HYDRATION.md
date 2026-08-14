# H17: Whole-model progressive hydration

**STATUS: STOP — exact baseline locked; all-192 control failed the predeclared
whole-model numerical gate.**

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

## Interpretation and next gate

The 96/144 progressive-hydration hypothesis is neither validated nor
falsified by this run. The all-192 evaluator failed as a numerical identity
control, so a smaller-budget result would mix deliberate omission error with a
known change in arithmetic trajectory. The registered identity gate therefore
remains stopped. A separately requested free-generation screen is allowed only
as **exploratory practical quality**, and every 96/144 result must be labelled
**EXPLORATORY — confounded by all-192 arithmetic divergence**. No selector,
tail corrector, Fisher fit, or Observer training follows from that screen.

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
