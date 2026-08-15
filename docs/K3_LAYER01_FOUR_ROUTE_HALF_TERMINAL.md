# K3 layer-one four-route, half-slab terminal control

VERDICT: POSITIVE SINGLE-LAYER SIGNAL; SMALL ALL-LAYER COMPOSITION SMOKE EARNED.

This control retained the four active routes with the largest calibrated
`abs(router weight) * native response norm`, evaluated the first six calibrated
slab ranks of each selected route, and replaced everything else with frozen
calibration means.  It therefore evaluated 24 of 192 active slabs (12.5%) at
routed layer 1.  All other routed layers stayed exact.  Selected bytes were
recomposed into the native 3,072-neuron expert layout and used one native-width
down projection; the rejected split-down arithmetic was not used.

| Control | terminal KL mean / max | top choice | greedy continuation |
|---|---:|---:|---:|
| Aviation, four generated tokens | 0.04514 / 0.27906 | 8/8 | 4/4 native IDs |
| Raw math, eight generated tokens | 0.07494 / 0.25234 | 14/17 | 8/8 native IDs |

The raw math prompt is deliberately reported only as exact-teacher acceptance.
It lacks the official K3 chat template, and the native model itself does not
answer `1024`.  Matching that continuation is not task success.

The important observation is narrower: at a robust early layer, retaining only
one eighth of the active neuron slabs preserved both frozen greedy continuations
despite nontrivial terminal distribution drift.  That makes an all-92-layer
composition smoke scientifically warranted.  It does not establish that the
same policy survives composition, is useful quality, or is faster; the current
shadow implementation evaluates extra work.

Machine-readable results are in
`results/kimi_layer01_four_route_half_terminal.json`.
