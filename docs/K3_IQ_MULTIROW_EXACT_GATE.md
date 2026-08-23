# K3 IQ multi-row exactness gate

## Verdict

The ordinary width-four/eight MMQ path is **not exact**. The existing MMVQ
path remains byte-identical through width eight and therefore supplies a
bounded exact implementation path without a new kernel.

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

Both complete test runs pass. The MMQ result may be studied later as an
explicit quality-gated approximation, but it is closed for the exact lane.

## Engineering decision

The exact wide service should not execute an expert at its entire empirical
`M_e` through MMQ. It should stage each selected expert payload once, keep it
resident, and run its same-payload rows in MMVQ chunks of at most eight. The
existing `ggml_mul_mat_sparse_k_blocks` down operation retains the natural-K
mapping and canonical one-row reduction used by P41. The existing outer
per-route join order remains unchanged.

Before implementing that generalization, the retained P65 trace must show
that M=2048 creates enough identical expert/slab-mask groups for width-eight
chunks to remove materially more uploads than the rejected M64 pair2 path.
That offline grouping calculation is the next bound; it requires no model or
GPU run.

Machine-readable results and binary/runtime hashes are in
`results/k3_iq_multirow_exact_gate.json`.
