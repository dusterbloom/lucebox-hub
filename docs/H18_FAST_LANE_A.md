# H18-FAST Lane A

**STATUS: CAPTURE SUBSTRATE PASSED — 10K all-layer capture is the next gate.**

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

The capture runner now accepts `layer=all`. It executes one ordinary exact
full-model pass and records the pre-expert latent, native top-16 identifiers,
router weights, token identifiers, split, and whole-sequence identifier for
every routed layer. Each layer is written in the unchanged v1 capture format,
so the existing expert-batched fitter reads it without conversion.

An eight-token smoke produced all 92 layer artifacts in `45.856` seconds. It
peaked at 17,087 MiB of graphics memory and wrote 5.14 MiB of capture payload.
The exact streamed evaluator reported 42.776 GiB of physical expert reads with
zero errors. The output vectors are bounded to one chunk at a time; the files
are streamed per whole sequence.

The semantic controls passed bit-for-bit:

| comparison | BF16 latent | expert IDs | router weights |
| --- | --- | --- | --- |
| all-layer layer 1 vs standalone layer 1, same 8-token shape | exact | exact | exact |
| all-layer layer 12 vs registered standalone layer 12 prefix | exact | exact | exact |

Layer 1 was intentionally compared with a new same-shape standalone capture.
The older layer-1 10K artifact used a 34-token first batch, whose different
attention graph shape changes numerical results before the expert boundary.
The same-shape standalone and all-layer layer-1 files share SHA-256
`ba2c4413e92a0685abf4e7ebfac93640be3a247fcb0f2fd5ee607afe12fd52e1`.
Layer 12 matches the established capture prefix exactly.

Smoke reproduction:

```bash
bash scripts/run_kimi_h18_all_layer_capture.sh
python3 scripts/compare_kimi_h18_multilayer_capture.py \
  /mnt/kimi-k3/captures/kimi-h18-all-layer-8 \
  results/kimi_h18_multilayer_capture_smoke.json \
  --layer1-reference /mnt/kimi-k3/captures/kimi-h18-single-layer01-8.bin
```

The long capture command is the same runner with:

```bash
KIMI_H18_CAPTURE_TOKENS=10000 \
KIMI_H18_CAPTURE_ROOT=/mnt/kimi-k3/captures/kimi-h18-all-layer-10000 \
bash scripts/run_kimi_h18_all_layer_capture.sh
```

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
work and is projected at roughly 243 hours. The implemented one-pass mode avoids
that repetition. Based on the measured layer-12 run, the pre-smoke planning
estimate was 5.28 hours for a 10K-token full-depth capture plus 5.04 hours to
fit the remaining 90 layers. The expected storage is about 6.7 GiB of captures,
15 GiB of fit states, and 15 GiB of runtime mean/order data; the experiment
drive has ample capacity.

The 10K runtime and fitting costs remain **PROJECTED**, not measured completion
times. Lane A terminal divergence, generation, routing, and byte-frontier
metrics remain blocked until the 10K capture and per-layer fit complete.
