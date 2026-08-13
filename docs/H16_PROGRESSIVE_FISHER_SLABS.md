# H16 Progressive Fisher Slabs

**VERDICT: BLOCKED — NO-GO beyond the exact-baseline gate.**

H16 did not start fitting a behavioral metric. The mandatory complete-model
exact baseline failed during model loading, before logits or generated tokens
existed. Continuing to fit or select Fisher slabs would violate the registered
protocol because terminal probability divergence is the primary target.

## Claim labels

- **MEASURED:** produced by the command executed for this H16 attempt.
- **VERIFIED:** an existing immutable artifact whose checksum and recorded
  values were checked in this attempt.
- **OPEN:** cannot be answered before the exact baseline runs twice.

## Frozen prior state

The worktree began clean at commit
`0a18d2db3c3c8d70a48c02ebf8aaa24ce9032ec6` on
`experiment/kimi-k3-panel-probe`, based on
`e9aa6bd13702dabc042292341b46fe9f06e46734`.

The existing sequence-disjoint first-layer artifacts were **VERIFIED** without
changing the teacher, split, evaluator, sidecar, or selector:

| Provider | Exact payload | Mean cosine | p05 cosine | Status |
| --- | ---: | ---: | ---: | --- |
| adaptive 96/192 slabs + slab-mean tail | 50% | 0.976688038 | 0.939117187 | VERIFIED |
| 8 whole experts + mean tail | 50% | 0.975828305 | 0.936845320 | VERIFIED |
| adaptive 144/192 slabs + slab-mean tail | 75% | 0.991022351 | 0.974435379 | VERIFIED |
| 12 whole experts + mean tail | 75% | 0.990747139 | 0.973657450 | VERIFIED |
| external uniform half-width, all routes | 50% | 0.909017167 | — | VERIFIED |

The immutable artifact checksums also match their registered values:

| Artifact | SHA-256 | Status |
| --- | --- | --- |
| 10,000-token capture | `5dc24f94da22a854eb9d67888174abdb157fd66d6d627e53ebe5168be72d0d9d` | VERIFIED |
| native exact teacher aggregate | `c3dbc469663ae31da0d483f0d19d31f66e6c532545097e5090355eb9b1aa2f78` | VERIFIED |
| slab calibration state | `f8e73b19760061d7e7052b0083708810449ab95bdeca2f3fc8ffb8281ecf6776` | VERIFIED |
| progressive slab sidecar | `e6d45975cb506073733f1b9af069786eab2afb60e7ed2ced0ac02ee12938113e` | VERIFIED |

The sidecar remains 5,780,303,872 bytes. Its registered direct-read results
remain 5.632 and 5.280 GiB/s for adaptive prefixes and 5.388 and 5.201 GiB/s
for the whole-expert control. These are storage-only measurements, not serving
speedups.

## Step 1: complete-model exact baseline

The frozen command was:

```bash
scripts/run_kimi_exact_baseline.sh
```

Before launching, it verified SHA-256 for all fourteen Unsloth
`Kimi-K3-UD-IQ1_S` shards. The build completed in Release mode for CUDA
architecture 86 with four build jobs. The attempted configuration used the
existing file-backed routed-expert provider, graphics device 0, a 512-token
maximum context, four requested generated tokens, and the frozen prompt
`According to all known laws`.

The first run failed while loading shard 8:

```text
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3777.69 MiB on device 0: cudaMalloc failed: out of memory
[kimi-k3] model load failed: Kimi-K3: unable to allocate resident tensor buffer for shard 8
```

This is a **MEASURED** placement failure:

| Measurement | Value |
| --- | ---: |
| elapsed before failure | 15.756 s |
| peak graphics memory | 24,199 MiB |
| process high-water resident memory | 6,762,472 KiB |
| model-drive bytes read | 25,229,496,320 |
| model-drive busy time | 11,670 ms |
| sampled graphics energy | 2,086.41 J |

No prompt token identifiers, final logits, generated identifiers, prefill
timing, or decode timing were produced because the model never completed
initialization. Therefore a second repeatability run was neither possible nor
scientifically meaningful.

Machine-readable records are in
`results/kimi_h16_exact_baseline_blocked.json` and
`results/kimi_h16_exact_baseline_blocked.csv`. Raw logs and telemetry remain
under `/mnt/kimi-k3/results/kimi-exact-baseline/`.

## Diagnosis and falsification control

Checkpoint damage was falsified: every shard passed the pinned checksum before
the allocation attempt. The failing allocation came after the graphics device
had reached 24,199 MiB, on a 24,576 MiB RTX 3090.

The current backend initializes one CUDA backend and allocates every resident
non-routed tensor into its default CUDA buffer. It does not implement CPU or
mixed host/device placement. Consequently, merely increasing WSL host memory
does **not** unblock this exact command. The already-known 128 GiB Lucebox host
was inspected as a control: it has sufficient unified memory but only 325 GiB
free and does not contain the frozen 594 GB checkpoint, so it cannot execute
this teacher without first solving storage.

## H16 results

| Question | Result |
| --- | --- |
| Are 96 slabs viable by final-logit KL? | **OPEN** |
| Are 144 slabs viable by final-logit KL? | **OPEN** |
| Does Fisher beat residual norm at equal bytes? | **OPEN** |
| Minimum safe exact byte fraction | **OPEN** |
| Is aggregate-tail training justified? | **NO** at this stage; terminal behavior and the limiting error are unmeasured |

No H16 intervention, metric fit, hyperparameter selection, or tail training was
run after the gate failed. This prevents validation leakage and preserves the
exact teacher path.

## Proposed roadmap diff (not applied)

```diff
- Next: run H16 Fisher slabs after increasing WSL host memory.
+ Next gate: obtain a complete exact K3 run on a backend that can place the
+ ~58 GiB non-routed core (unified-memory Lucebox with enough model storage,
+ or an explicit tested mixed host/device placement path), then lock two
+ byte-identical exact runs.
+ Only after that gate: run H16 providers A-E at layer 1 and collect terminal
+ KL before fitting any behavioral metric.
```

## Reproduction

The single current command is the exact gate shown above. A successful H16
command is intentionally not added until that command completes twice and
produces byte-identical logits; creating a parallel approximation runner before
then would weaken the regression lock rather than advance the experiment.
