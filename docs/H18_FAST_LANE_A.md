# H18-FAST Lane A

**VERDICT: BLOCKED — no calibrated 92-layer slab provider exists.**

The arithmetic-stable full-width sidecar path is ready, but the only all-layer
sidecars are natural-order identity controls whose manifest registers budget
192 as their sole valid budget. They contain no slab response means, residual
importance, or calibrated per-expert ordering. Ten-thousand-token calibration
states exist only for model layers 1 and 12.

Copying layer-1 calibration to the other 90 layers would not be the requested
experiment. Expert functions and response distributions are layer-specific,
and the measured layer-12 control already found the current mean-tail provider
far weaker than layer 1 (`0.865533` mean cosine at budget 96 and `0.937733` at
144). A natural-prefix, zero-tail fallback was started only to validate the
narrow full-width partial mechanism, then stopped before its first prompt
completed when the fallback was superseded. It produced no quality result.

## Shortest valid substrate path

The existing per-layer capture interface is:

```text
capture_kimi_k3_panel MODEL CORPUS CAPTURE 0 LAYER 10000 512 8 cpu
```

The existing expert-batched calibration interface is:

```text
python3 scripts/probe_kimi_neuron_slabs.py \
  SHARD CAPTURE TEACHER RESPONSES RESULT \
  --fit-state FIT_STATE --calibration-only \
  --exact-fallback-uncalibrated --layer LAYER --device cuda
```

Running the capture separately for all missing layers would repeat exact prefix
work and is projected at roughly 243 hours. The shortest valid preparation is
a bounded multi-layer capture mode that records every layer during one exact
full-model pass, followed by the unchanged per-layer fitter. Based on the
measured layer-12 run, planning estimates are approximately 5.28 hours for a
10K-token full-depth capture plus 5.04 hours to fit the remaining 90 layers.
The expected storage is about 6.7 GiB of captures, 15 GiB of fit states, and
15 GiB of runtime mean/order data; the experiment drive has ample capacity.

These are **PROJECTED** preparation costs, not measured completion times. Lane
A terminal divergence, generation, routing, and byte-frontier metrics remain
unmeasured until this calibration substrate exists.
