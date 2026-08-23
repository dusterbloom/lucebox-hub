# K3 IQ multi-row exactness gate

## Verdict

The ordinary width-four/eight MMQ path is **not exact**.  MMVQ width four and
eight was byte-identical on this discriminator's first eight deterministic
inputs, but later 245-row continuations falsified that as a general exactness
envelope.  A separate width-two union continuation was also non-exact.
Generic compact multi-row execution is therefore **not an exact production
path**.  The specific M64 pair2 fixture remains exact evidence, not a universal
arithmetic guarantee.

This is a model-free arithmetic gate on physical GPU1 `gfx1151`, not a
performance or live-prefill result.

## Existing-test discriminator

`test_moe_stream_compute` now has a default-off case selected by
`DFLASH_TEST_KIMI_IQ_MULTIROW_EXACT=1`. It creates one complete K3-shaped
expert (`3584 -> 3072 -> 3584`) with IQ1_S gate/up, IQ2_XXS down and SiTU.
Deterministic full-shape F32 values are quantized in small chunks with a unit
importance matrix. Eight separate width-one evaluations are the teacher.

The test then runs both policies using fresh stream engines:

- current production dispatch, with MMVQ capped at three columns;
- an exact fallback, with MMVQ capped at eight columns.

Existing backend launch counters prove the dispatched kernel family. The test
reports byte equality, mismatch count, maximum absolute difference and
relative L2. It requires the MMVQ rows to be exact but records rather than
mislabels the MMQ numerical path.

## GPU1 result

The first preflight exposed a topology problem in the pre-existing F32 test:
making both heterogeneous GPUs visible and selecting index one crashes inside
the HIP runtime. Isolating physical GPU1 as the only visible device fixes the
unchanged control. No numerical row from the invalid topology is retained.

The qualified invocation uses `HIP_VISIBLE_DEVICES=1`, visible test device
zero, `ROCBLAS_USE_HIPBLASLT=0`, and the retained ROCm 7.2.4 HIP/HSA overlay.
The system ROCm 7.2.2 closure independently produced the same values.

| policy | width | observed path | exact | mismatches | max abs | relative L2 |
|---|---:|---|---|---:|---:|---:|
| production ceiling 3 | 2 | 3 MMVQ, 0 MMQ | yes | 0 | 0 | 0 |
| production ceiling 3 | 4 | 0 MMVQ, 3 MMQ | no | 14,336 | 0.000123003 | 0.00832551 |
| production ceiling 3 | 8 | 0 MMVQ, 3 MMQ | no | 28,672 | 0.000123003 | 0.00809432 |
| exact ceiling 8 | 4 | 3 MMVQ, 0 MMQ | **yes** | 0 | 0 | 0 |
| exact ceiling 8 | 8 | 3 MMVQ, 0 MMQ | **yes** | 0 | 0 | 0 |

Both original eight-input test runs pass.  This table is retained as the
fixture-level observation it actually established, not as a universal MMVQ
claim.  The MMQ result remains closed for the exact lane.

## Superseding long-row evidence

The follow-up used the same physical GPU and complete K3 shape, one reordered
full-12 compact payload, independent row masks, and 245 deterministic inputs.
The width-eight candidate dispatched 62 MMVQ launches and no MMQ.  It was
nominally `3.561165x` faster than the current rowwise compact teacher
(`58.478407 ms -> 16.421144 ms`) and reduced authoritative weight H2D from
722,856,960 to 7,139,328 bytes, but it was **not byte-identical**.  The
correctness gate aborted before the padded-tail arm.

A separate six-slab identical-mask width-eight arm was also fast
(`4.289587x`) and non-exact.  Thus the failure is not repaired by avoiding
differing masks or by using a full-12 resident tensor.  Results and frozen
hashes are in `results/k3_compact_schedule_discriminator.json` and
`results/k3_full12_union_long.json`.

The final full-12 width-two continuation was nominally `2.876200x`
(`58.684213 ms -> 20.403382 ms`) and reduced weight H2D from 722,856,960 to
7,139,328 bytes, but it too failed byte identity over the 245 rows.  Its
result is `results/k3_full12_union_width2.json`.

## Engineering decision

Do not integrate compact multi-row MMVQ into the generic exact runtime.  Its
speed prize is real, but its arithmetic differs for later inputs even at width
two.  The test-only union executor was removed after preserving the valid
NO-GOs.  Exact production work must keep one-row arithmetic while improving
layer-major storage/order.  Any multi-row continuation must report
numerical/state/logit deltas and remain behind the exact fallback.

Machine-readable results and binary/runtime hashes are in
`results/k3_iq_multirow_exact_gate.json`.
