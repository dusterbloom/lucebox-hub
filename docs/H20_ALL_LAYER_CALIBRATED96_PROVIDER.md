# H20: honest all-layer calibrated96 provider substrate

## Status

**IMPLEMENTED SUBSTRATE — PILOT calibration, insufficient for quality
certification. No all-layer quality or speed claim.**

The provider can now apply a separate calibrated slab order and separate mean
cards at each of Kimi K3's 92 routed layers. It requests a nominal global
budget of 96 slab records per token/layer and performs one arithmetic-stable,
full-width down reduction for each approximated expert.

This is deliberately fail-safe:

- a missing, malformed, or provenance-free layer is evaluated exactly;
- an expert with fewer than the exported minimum calibration hits is evaluated
  exactly;
- exact fallback records are excluded from the calibrated selector;
- if fewer than eight calibrated experts are active, fewer than 96 slab
  records may be selected rather than pretending the budget was filled;
- runtime traffic reports selected sidecar bytes and exact fallback bytes
  separately, then reports their sum.

The runtime never substitutes layer-1 statistics for another layer.

## Artifact and provenance contract

`export_kimi_all_layer_calibrated_runtime.py` writes one v2 `.k3aux` per
layer. Each contains that layer's ordered means, importance, calibrated-expert
mask, and `uint32` calibration hit counts. Its header binds the artifact to
four SHA-256 identities:

1. the source layer fit-state;
2. the source layer capture;
3. the natural-order slab sidecar;
4. the registered 14-shard model checksum list.

The JSON manifest additionally maps each gate/up/down tensor name to its GGUF
shard, byte size, and registered shard SHA-256. The natural sidecars are reused
in place. The provider accepts both the original uniform v1 layout and the
real v2 mixed layout where gate/up may be IQ1_S while down is IQ2_XXS.

The ordinary export trusts the already registered sidecar hashes after checking
headers and exact file lengths. Add `--verify-sidecar-sha256` for a full
545.35 GB checksum reread.

## Why the current calibration remains a pilot

The surviving capture has only 2,048 tokens across 26 sequences. Coverage is
uneven and some experts are unseen. The default export threshold is eight
calibration routes per expert; this is an operational fallback threshold, not
a statistical quality guarantee. Experts below it stay exact, but experts
above it are still only pilot-calibrated.

Consequently this substrate enables an honest two-prompt end-to-end screen. It
does not establish that 96 slabs is safe across domains, prompts, or all 92
layers.

## CPU preflight

The non-writing preflight reads only metadata and small NPZ arrays:

```bash
python3 scripts/export_kimi_all_layer_calibrated_runtime.py \
  /mnt/kimi-k3/fit-state/kimi-h18-slab-calibration-2048 \
  /mnt/kimi-k3/captures/kimi-h18-all-layer-2048-chunk8 \
  /mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars \
  /tmp/kimi-calibrated96-preflight \
  --preflight-only
```

The recovered artifact preflight found 92/92 layers: 10 v1 uniform-layout
sidecars and 82 v2 mixed-layout sidecars, totaling 545,352,335,360 bytes.

## Remaining runtime export

This is the exact artifact-generation command (do not run it concurrently with
latency-sensitive generation):

```bash
python3 scripts/export_kimi_all_layer_calibrated_runtime.py \
  /mnt/kimi-k3/fit-state/kimi-h18-slab-calibration-2048 \
  /mnt/kimi-k3/captures/kimi-h18-all-layer-2048-chunk8 \
  /mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars \
  /mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime \
  --minimum-expert-hits 8
```

Expected new storage is 14,188,431,360 bytes (13.21 GiB), plus small JSON
manifests. It reads approximately 15.37 GB of fit states and 1.3 GiB of
captures and writes 13.21 GiB. On the installed NVMe this should take roughly
2–5 minutes; that is an estimate until measured. Adding
`--verify-sidecar-sha256` rereads another 545.35 GB and should be budgeted at
roughly 8–20 additional minutes.

## Runtime activation and traffic accounting

```bash
DFLASH_KIMI_LAYER1_PROVIDER=all-layers-calibrated96 \
DFLASH_KIMI_CALIBRATED96_AUX_DIR=/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime \
DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=/mnt/kimi-k3/artifacts/kimi-h17-natural-sidecars \
DFLASH_KIMI_CALIBRATED96_METRICS_OUT=/tmp/kimi-calibrated96-traffic.tsv \
  <normal Kimi server command>
```

The traffic table records per layer: tokens, requested nominal slabs, actual
selected slab records, calibrated routes, exact fallback routes, selected
sidecar bytes, exact fallback bytes, and total provider bytes.

## Road ahead

1. Export the 13.21 GiB runtime artifacts after the current generation job.
2. Run one easy and one hard end-to-end prompt and inspect both text and the
   traffic table. Treat this only as a smoke screen.
3. If coherent, expand to the frozen prompt suite and compare full-vocabulary
   logits against exact.
4. Capture a much broader, domain-balanced calibration corpus, refit all 92
   independent layers, and raise the minimum-hit policy based on coverage.
5. Only after quality gates pass, optimize I/O/computation and measure speed.
