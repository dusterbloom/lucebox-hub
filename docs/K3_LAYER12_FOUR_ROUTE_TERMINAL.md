# K3 layer-12 four-route terminal knee

**VERDICT: POSITIVE EXACT-TEACHER SPECULATIVE SIGNAL; TASK AND ALL-LAYER GATES REMAIN OPEN.**

The frozen layer-12 intervention was repeated with four complete selected
experts instead of two. The remaining twelve route outputs use frozen native
expert means. Every other routed layer remains exact.

| control | scored rows | mean / max terminal KL | top-1 | generated tokens |
| --- | ---: | ---: | ---: | ---: |
| raw math, one token | 10 | 0.01694 / 0.03306 | 10/10 | 1/1 identical |
| aviation continuation | 8 | 0.06070 / 0.34534 | 8/8 | 4/4 identical |
| raw math, eight tokens | 17 | 0.02218 / 0.04504 | 17/17 | 8/8 identical |

Across all three controls, four routes preserved all 35 scored greedy choices.
The aviation maximum KL of `0.34534` did not flip the top choice. This confirms
that neither local cosine nor KL alone determines speculative acceptance; the
logit decision margin matters.

The eight-token raw math control is not a task-quality result. Native K3 itself
generates ` No explanation. No punctuation. No words` because this legacy
fixture does not use the checkpoint's official chat template. It remains a
valid exact-teacher acceptance trace.

This result does not yet establish an all-layer draft or a speedup. The legacy
shadow provider computes the exact layer as well as the intervention. It does,
however, identify a sharp measured knee between two routes (first-token
rejection on one control) and four routes (35/35 top-choice agreement).

The next behavioral gate is a correctly templated prompt where native K3 is
independently verified to succeed. The next byte gate is four routes with six
of twelve calibrated slabs on an ordinary, non-block-start layer; four complete
routes at every layer would still consume 25% of the routed expert bytes.
