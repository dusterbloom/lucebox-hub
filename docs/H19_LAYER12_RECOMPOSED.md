# H19: Layer-12 full-width recomposed slabs

## Status

**MEASURED — one frozen prompt; the 12-prompt identity control is in progress.**

H19 removes the numerical confound from the earlier progressive provider. It
reads selected 256-neuron records from the calibration-ordered sidecar, maps
them back to their native positions in the full 3,072-neuron quantized tensors,
and executes one full-width native down projection. The selected activation
rows are then live; omitted rows are masked and receive the existing slab-mean
tail. This is a quality-control implementation. It is not yet a speed path
because the current tracing wrapper also evaluates the native teacher output.

## Arithmetic gate

At model layer 12, with all 192 active slabs retained, the recomposed provider
is byte-identical to native K3 on the frozen prompt and four teacher-forced
positions:

| metric | result |
| --- | ---: |
| routed-output relative L2 | 0 |
| terminal logit KL | 0 |
| logit bytes | identical |
| top-1 | 8 / 8 |
| generated tokens | identical |

This is the required difference from the older split-down `slabs` provider,
whose all-192 arithmetic was locally near-identical but accumulated observable
terminal KL.

## One-prompt partial controls

The same prompt (`According to all known laws`) and 8 scored rows were run with
only layer 12 altered; all other routed layers remain native exact.

| retained slabs | local cosine | local relL2 | terminal KL mean | terminal KL max | top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 192 | 1.000000 | 0 | 0 | 0 | 8 / 8 |
| 96 | 0.860304 | 0.471477 | 0.018209 | 0.111415 | 8 / 8 |
| 144 | 0.934343 | 0.329900 | 0.012412 | 0.043101 | 7 / 8 |

Both partial modes produced the same four-token continuation as native in this
small screen. The 144 case nevertheless changes one scored top choice, so it
is not a safety result. Neither partial mode has a measured serving-speed claim:
although selected sidecar records are only 50% / 75% of the per-layer slab
payload, the current evaluator still runs native exact output for trace
comparison.

## Reproduction

```bash
KIMI_H16_OUTPUT_DIR=/mnt/kimi-k3/results/kimi-h19-layer12-recomposed-identity \
KIMI_H16_LAYER=12 \
KIMI_H16_SLAB_AUX=/mnt/kimi-k3/artifacts/kimi_layer12_slab_runtime.k3aux \
KIMI_H16_SLAB_SIDECAR=/mnt/kimi-k3/artifacts/kimi_layer12_progressive_slabs.k3slab \
KIMI_H16_MODES='recomposed192' \
bash scripts/run_kimi_h16_frozen.sh
```

Then replace the mode with `recomposed96 recomposed144` for the partial
controls. The provider implementation is commit `1000289`; the trace analyzer
recognition fix is commit `3beed06`.

## Next gate

Complete the already-running 12-prompt / 126-row recomposed-192 control. Only
if it remains byte-identical should the matched recomposed-96 and recomposed-144
suite runs be interpreted as broader layer-12 terminal-KL evidence.
