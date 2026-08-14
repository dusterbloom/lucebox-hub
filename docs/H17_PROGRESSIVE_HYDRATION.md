# H17: Whole-model progressive hydration

**STATUS: RUNNING — exact baseline locked; all-192 control in preparation.**

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
