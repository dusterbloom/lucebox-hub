# H21 — four-route composition through all 92 K3 routed layers

VERDICT: THE MODEL IS ALIVE AT FOUR COMPLETE ROUTES; THE HALF-SLAB COMBINATION FAILS.

This is the first on-policy full-model composition test of the aggressive
four-route proposal.  Every routed layer used its own 2,048-token calibration
artifact.  Under-sampled experts remained exact.  The physical sparse path read
only selected sidecar slabs plus those exact fallbacks and used the unchanged
native-width expert reduction.

| All-layer policy | logical routed GiB / position | shared-prefix KL mean / max | first generated token | practical result |
|---|---:|---:|---|---|
| Four routes × 6/12 slabs | 2.13 | 1.436 / 3.312 | differs | fails the uniform 12.5%-slab gate |
| Four routes × 12/12 slabs | 3.35 | 0.128 / 0.361 | differs | coherent, but not native-equivalent |

The four-complete-route arm was extended to four generated tokens.  Native
continued with a line-broken `of aviation,`; the candidate generated
` of aviation, there`.  It is clearly coherent and non-repetitive.  This is
important evidence that local cosine pessimism did not imply immediate model
death.  It is not evidence of general task quality or low divergence.

The half-slab comparison localizes most of the additional damage: preserving
all neurons in the four selected routes reduced the shared-context mean KL by
more than 11×.  Still, omitting twelve routes at every routed layer changes the
terminal choice after 92 compositions.  A uniform all-layer 100× policy is
therefore not earned.

The next experiment should change *placement*, not invent a new tail.  Hold
four complete routes at approximated layers, keep other layers exact, and test
pre-registered AttnRes phases (block ends versus starts and middles).  If block
ends preserve behavior, grow backward from those firebreaks and measure the
quality/byte frontier.  If even block-end-only placement fails, the four-route
branch should close.

Performance remains dominated by roughly 47 GiB of process reads per model
position from the mapped non-routed core under the 27 GiB WSL allocation.  The
measured three-transition decode rate was 0.043 transitions/s; no 100× speed
claim is made.

Machine-readable results are in `results/kimi_h21_all_layer_four_route.json`.
