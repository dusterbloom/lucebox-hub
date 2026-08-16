# H23 — crash-bounded 10K calibration runbook

STATUS: ONE-CHUNK GATE PASSED / RESUME TESTED / LONG RUN STOPPED

The earned H23 next step is to increase calibration coverage from 2,048 to
10,000 tokens without changing slab ordering, the calibrated mean tail, the
minimum-hit fallback rule, or any runtime selector semantics.  Better coverage
can lower the exact-fallback floor; it is not itself a quality claim.

## Audit finding

The old all-layer capture executable is numerically appropriate but is not a
safe 10K job by itself.  It opens 92 temporary files, runs the entire corpus in
one process, and publishes all files only at the end.  The original 10K attempt
failed after the first 34-token sequence and left 92 partial files totaling
22,609,920 bytes in:

```text
/mnt/kimi-k3/captures/kimi-h18-all-layer-10000
```

Those files have zero-count provisional headers and no complete manifest.  They
remain intact for forensics but are rejected as scientific input.

The downstream machinery is already substantially safer:

- `fit_kimi_k3_panel` checkpoints aggregate state every 16 experts and writes
  each expert response atomically;
- slab calibration publishes one atomic NumPy state per layer and validates a
  completed state before skipping it;
- the runtime export is atomic per layer.  It now accepts an explicit capture
  token count rather than assuming `2048` in the filename.

The missing firebreak was the expensive all-layer capture.

## Minimal change

`kimi_h23_capture_chunks.py` keeps the existing exact capture executable and
divides only the corpus schedule:

1. copy eight complete JSONL rows into an immutable chunk corpus;
2. run the existing one-pass, all-92-layer exact capture on that chunk;
3. validate every binary header, record boundary, sequence identity, split,
   token total, layer index, sidecar index, and exact end-of-file;
4. treat only a complete 92-layer manifest as a checkpoint;
5. on restart, skip validated chunks and quarantine only the incomplete active
   directory by moving it under `rejected/`;
6. concatenate record payloads into the original capture format and publish one
   atomic merged file plus receipt per layer.

No tensor arithmetic is changed.  The fitter consumes the merged capture
directly because its header and record layout are the existing
`K3PNL001` format.  A synthetic test creates two 92-layer chunks, simulates a
crashed second chunk, resumes it, merges all layers, removes the global merge
manifest, and proves the per-layer receipts resume without rewriting outputs.

## Chunk count and sequence split

The frozen corpus has 200 rows.  At eight rows per chunk the deterministic plan
contains at most 25 chunks.  The prior 2,048-token capture reached its limit in
26 rows, or 78.8 tokens per row.  If that yield holds, an eight-row chunk carries
about 630 tokens and 10K finishes in roughly 16 chunks.  This is a projection;
the first-chunk timing gate will measure the actual yield.

Rows are copied in source order with their original `calibration` or
`validation` label.  Chunking never redistributes tokens between splits.  The
capture executable receives the remaining global token budget, so only the
final source row can be truncated when the exact 10,000-token cap is reached;
its whole-sequence split label remains unchanged.  The merge is ordered record
concatenation and cannot interleave calibration and validation tokens.

## Startup overhead and the timing gate

Each chunk deliberately pays one model/stream-engine startup to bound crash
loss.  The surviving measurements are:

| Capture | Wall time | Qualification |
|---|---:|---|
| all-layer, 8 tokens, chunk 8 | 45.86 s | startup-dominated |
| all-layer, 34 tokens, chunk 128 | 75.23 s | best small batching anchor |
| all-layer, 2,048 tokens, chunk 8 | 10,227.37 s | complete but inefficient |

The first two runs suggest roughly 30–45 seconds of fixed startup.  Sixteen
typical chunks would therefore add about 8–12 minutes; all 25 planned chunks
would add about 13–19 minutes.  This is acceptable relative to a projected
5–7-hour chunk-128 capture, but the range is not yet measured at an eight-row
chunk size.  The old chunk-8 result gives a conservative capture upper bound of
about 15 hours.

Before any long run, execute exactly one useful, resumable chunk and stop:

```bash
KIMI_H23_ALLOW_BENCHMARK=1 \
scripts/gpu_lease.sh run H23-10K -- \
scripts/run_kimi_h23_10k_capture.sh benchmark
```

The benchmark became chunk zero of the eventual run; it was not discarded.  It
completed and the runner stopped before chunk one:

| One-chunk result | Measured |
|---|---:|
| rows / tokens | 8 / 640 |
| split | 7 calibration rows / 1 validation row |
| wall time | 691.78 s |
| capture throughput | 0.925 tokens/s |
| validated layer files | 92 / 92 |
| disk reads | 1,563,403,902,976 bytes |
| output footprint | 411 MiB (`du`) |
| peak RAM | 47,802,920 KiB (45.6 GiB) |
| anonymous / file-backed peak | 1.14 / 44.5 GiB |
| peak swap | 198 MiB |
| peak VRAM | 2,700 MiB |
| GPU energy | 80.31 kJ |

The exact chunk manifest SHA-256 is
`8a47803d4ff3cb3f066407b33de0754cb84918121c5d1f6849021cf05b7391cb`.
The telemetry and validation hashes are respectively
`02418b1982e62f584807042105dfe259ab9e1b933a7ee4a1a05433bc7f0f51f3`
and `643a146ebd02dd93147d6cb8b93355f4d3fe046eb878f3d5d545ead873d13752`.

At this measured rate, 10K capture projects to 10,809 seconds, or 3.00 hours.
Allowing corpus-mix variation gives a conservative 3–4-hour range.  The VRAM
margin is excellent.  Host RSS is the operational caution: almost all of the
45.6-GiB peak is file-backed mapping rather than anonymous allocation, and the
chunk completed safely, but it leaves little headline room inside a 47-GiB WSL
limit.  The full loop remains blocked behind a separate
`KIMI_H23_ALLOW_LONG_RUN=1` gate pending that review.

## Rejected accelerator-core capture gate

The production backend can place the latent and shared MoE-core tensors on the
GPU, but the capture executable did not wire that optional object into its
forward options.  A minimal capture-only plumbing patch was built and tested on
the first registered 34-token sequence with `latent,shared` offload.  It moved
12.90 GiB to the RTX 3090 and completed in 48.37 seconds, compared with the
75.23-second prior 34-token CPU anchor.  Peak VRAM was a safe 15,891 MiB.

The speed result is not usable for calibration because the captured trajectory
changed materially:

| Capture boundary | latent relL2 | max abs | cosine | exact route rows |
|---|---:|---:|---:|---:|
| layer 1 | 0.0882384 | 0.0407715 | 0.9960996 | 34 / 34 |
| layer 12 | 0.0994477 | 0.0302734 | 0.9951455 | 0 / 34 |

At layer 12 only 259/544 individual top-16 route IDs matched and router-weight
relL2 was 0.0416694.  Peak sampled RSS fell only 3.75% versus chunk zero, while
the process high-water mark remained effectively unchanged (47,788,516 versus
47,848,408 KiB) because copying the offloaded weights first touches their mmap
source pages.  Therefore this gate is **PARITY FAIL / RSS NO-GO**.  The plumbing
patch was reverted, the rejected smoke remains available for forensics, and the
run script explicitly sets `DFLASH_KIMI_MOE_CORE_OFFLOAD=0`.  The measured CPU
chunk-zero capture remains authoritative.

## Storage ledger

The 350-GiB free-space gate covers the complete pipeline, not just capture:

| Artifact | GiB | Basis |
|---|---:|---|
| immutable capture chunks | 6.251 | exact record geometry |
| merged captures | 6.251 | retained beside chunks for provenance |
| individual expert responses | 196.753 | 10K × 16 routes × 3,584 float32 outputs × 92 |
| fitter expert statistics/checkpoints | 29.108 | measured 2K footprint; token-count independent shapes |
| exact teacher aggregates | 2.457 | projected at the frozen 20% validation split |
| per-rank validation aggregates | 39.307 | 16 × teacher aggregate |
| fitted panel outputs | 5.5 | measured fixed shape; diagnostic but emitted by fitter |
| slab calibration states | 14.8 | measured fixed shape |
| calibrated runtime export | 14.0 | measured current all-layer export |
| **projected working total** | **314.43** | before small logs/receipts/temporary files |
| **enforced free-space minimum** | **350.0** | 35.6-GiB operational margin |

Preflight measured approximately 1,827 GiB free, so capacity is not the blocker.
The existing 508-GiB natural-order sidecars are reused and are not duplicated in
this ledger.

## Time estimate after capture

Measured real 10K layer-12 anchors are 11.26 seconds for exact response fitting
and 201.72 seconds for slab calibration.  Across 92 layers those suggest about
0.3 hours and 5.2 hours respectively.  With the measured 3–4-hour capture, the
full path to a new runtime export is approximately 8.5–9.5 hours.  Only chunk
zero is MEASURED; the complete-run figure remains PROJECTED and no unattended
long run has started.

## Commands and gates

CPU-only preflight:

```bash
scripts/run_kimi_h23_10k_capture.sh preflight
python3 -m unittest server.tests.test_kimi_h23_capture_chunks -v
```

After the one-chunk timing review, resume the same immutable plan:

```bash
KIMI_H23_ALLOW_LONG_RUN=1 \
scripts/gpu_lease.sh run H23-10K -- \
scripts/run_kimi_h23_10k_capture.sh capture
```

The runner uses a 2-GiB device cache, 12 CPU threads, a 128-token forward chunk,
and caps its only build at `-j4`.  It enforces 350 GiB free space.  It never
touches the rejected old 10K root.

After merged capture, exact-response fitting and slab calibration remain
per-expert/per-layer restartable.  Export with the existing sidecars and the new
token-count argument:

```bash
python3 scripts/export_kimi_all_layer_calibrated_runtime.py \
  /mnt/kimi-k3/fit-state/kimi-h23-slab-calibration-10000 \
  /mnt/kimi-k3/captures/kimi-h23-all-layer-10000-chunked-v1/merged \
  /mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars \
  /mnt/kimi-k3/artifacts/kimi-h23-calibrated96-runtime-10000 \
  --capture-tokens 10000 --minimum-expert-hits 8 --preflight-only
```

Only after all 92 layers pass that preflight should the real export, fallback
floor recomputation, H23 policy optimization, and broader quality rerun proceed.

Machine-readable estimates and source hashes are in
`results/h23_10k_preflight.json`.
