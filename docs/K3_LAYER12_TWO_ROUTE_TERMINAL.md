# K3 layer-12 two-route terminal intervention

**VERDICT: TERMINAL ATTENUATION IS REAL; TWO ROUTES ARE NOT A SAFE SPECULATIVE DRAFT.**

## Question

The layer-12 two-route approximation has poor local geometry: held-out
post-routed-up cosine is about `0.7106`. Does the rest of K3 suppress that
error enough that terminal probabilities and greedy tokens remain useful?

## Frozen intervention

All routed layers except zero-indexed layer 12 remain native exact. At layer
12, the runtime keeps two complete experts selected by router weight times the
calibrated mean native-response norm. The remaining fourteen routes are
replaced by their frozen native expert means. This is the same causal policy
used by the registered 100x local oracle; no held-out answer enters selection.

The existing shadow provider evaluates the native layer before the
intervention, so this run measures behavior and not speed.

## Results

| prompt | local post-up cosine | terminal KL mean / max | top-1 | generation |
| --- | ---: | ---: | ---: | --- |
| `According to all known laws` | 0.7106 atlas reference | 0.00984 / 0.03277 | 7/8 | first 3/4 tokens match |
| held-out `Compute two to the tenth power` | 0.7106 atlas reference | 0.02833 / 0.08486 | 8/10 | first answer token differs |

For the held-out math prompt, native chooses token `4415`, the known exact
suite answer token for `1024`. The two-route intervention chooses token
`50339` instead. It therefore fails the task and would be rejected at the
first speculative token.

## Interpretation

The result confirms a major loophole in the local analysis. AttnRes, shared
experts, residual mixing, and the downstream suffix discard or attenuate much
of a very large Euclidean layer-local error. Local cosine alone is therefore
not a valid ship gate.

It does **not** rescue the two-route 100x proposal. Long speculative batches
require near-perfect greedy agreement, whereas the two measured prompts give
7/8 and 8/10 scored-position agreement and the hard prompt rejects
immediately. Multi-position verification cannot amortize one full-bank read
when the draft fails on its first answer token.

The shortest next experiment is a tiny route-budget curve at this same layer:
2, 4, and 8 complete routes plus frozen means, scored by terminal KL, top-1
margin, and accepted greedy prefix. That identifies whether a moderate draft
can provide useful speculative batches before any all-layer runtime work.

## Reproduction identity

- code: `276907fe3a585425507d54f76c24bed806220d49`
- layer-12 calibration aux SHA-256:
  `26737b38967a41579b678d60ff08ab22ec9aca45ccdb23d384dcfbd9c8182d6f`
- machine-readable result: `results/kimi_layer12_two_route_terminal.json`
- raw runs: `/tmp/kimi-k3-100x-terminal-layer12-whole2*`

