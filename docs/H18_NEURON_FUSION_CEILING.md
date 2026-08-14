# H18 Lane B — 16× native-neuron fusion ceiling

**VERDICT: NO-GO at 16×.**

## Protocol

Expert 64 in zero-indexed model layer 1 was predeclared before inspecting any
compression result.  It was selected solely because it has the largest route
count in the registered 10K capture: 887 routes, divided by whole sequence into
719 training, 26 development, and 142 untouched test samples.

For every route, the experiment evaluates the real IQ1_S gate/up weights in
FP32 with K3's SiTU-GLU formula to obtain the native 3,072 intermediate
activations.  Training-only column-pivoted QR chooses the retained native
neurons.  A dense FP32 ridge output map is then fit from those activations to
the captured native expert output.  Ridge is chosen only on development
sequences.  Registered validation sequences are opened once for the reported
test.

This is an optimistic output least-squares ceiling, not a deployable fusion.
It is strictly more flexible than the proposed hard many-to-one sparse fusion.

## Offline teacher-fidelity control

Before interpreting compression, the complete reconstructed 3,072-dimensional
activation was passed through the dequantized native down matrix and compared
with the captured native IQ1_S expert output:

| metric | result |
|---|---:|
| cosine mean | 0.9998475 |
| cosine median | 0.9998847 |
| cosine p05 | 0.9996148 |
| relative L2 mean | 0.0166782 |
| norm-ratio median | 0.9997667 |

This passes the registered offline-fidelity diagnostic.  The remaining small
difference is the expected Python-dequantized versus native quantized-kernel
execution difference; it is tiny compared with the compression error below.

## Untouched-test results

| Native activations | Compression | cosine mean | cosine median | cosine p05 | relative L2 mean | norm-ratio median |
|---:|---:|---:|---:|---:|---:|---:|
| 192 | 16× | 0.829454 | 0.908481 | 0.339181 | 0.479942 | 0.965362 |
| 384 | 8× | 0.848563 | 0.929641 | 0.400623 | 0.450531 | 0.990302 |
| 768 | 4× | 0.885695 | 0.953658 | 0.512625 | 0.378889 | 0.983471 |

The hard 16× gate was:

```text
NO-GO if median cosine < 0.95 OR p05 cosine < 0.80
```

Both conditions fail decisively at 192 activations.  The prescribed 384 and
768 controls locate a slow compression knee; even 768 activations leave a very
poor p05 tail.  Therefore the experiment stops before expert replication,
sparse fusion, optimal transport, or native runtime intervention.

## Reproduction

```bash
PYTHONPATH=scripts OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
python3 scripts/probe_kimi_h18_neuron_fusion_ceiling.py \
  /mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00002-of-00014.gguf \
  /mnt/kimi-k3/captures/kimi_layer01_10000.bin \
  /mnt/kimi-k3/results/kimi_layer01_panel_10000_teacher.teacher.f32 \
  /mnt/kimi-k3/responses/kimi_layer01_10000 \
  results/kimi_h18_lane_b_predeclaration.json \
  results/kimi_h18_neuron_fusion_ceiling.json \
  --threads 6
```

Artifact hashes:

```text
0812fcdd3bfd32996add25c5643d514b1169d4ff7c31d01aeec0114b15b833a7  predeclaration
29b757a1ad769b4f5206a1d4a3ac83ae7226032d17116bb573a669ca457bed2c  result
```

## Decision

The 192-neuron output-LS ceiling fails, so a sparse 192-neuron fusion cannot be
expected to rescue it.  Do not spend implementation time on optimal transport,
sparse fusion, or a one-expert native intervention for this 16× hypothesis.
The evidence instead supports progressive exact slab hydration, which preserves
many more native activations while reducing authoritative weight reads.
