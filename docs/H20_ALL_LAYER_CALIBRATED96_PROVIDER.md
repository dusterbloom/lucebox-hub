# H20: honest all-layer calibrated96 provider substrate

## Status

**IMPLEMENTED AND SMOKE-TESTED — PILOT calibration, insufficient for quality
certification. One official-template prompt is token-exact to native; no broad
all-layer quality or speed claim.**

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

## Runtime export

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

The completed export occupies 14,188,431,360 bytes (13.21 GiB), plus small JSON
manifests, at
`/mnt/kimi-k3/artifacts/kimi-h20-calibrated96-runtime`. Its aggregate manifest
SHA-256 is
`ed321b400b99234522583d7ea279cca8ba2b053257daa8dd713137beb7546bc1`.
Adding
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

## Official-template matched smoke

The earlier raw-string smoke was not a valid Kimi K3 chat request. The
checkpoint embeds a model-specific Jinja template using `<|open|>`, `<|sep|>`,
`<|close|>`, and `<|end_of_msg|>` markers. A matched deterministic control
rendered this embedded template with thinking disabled, then passed the same
50 prompt token IDs and 32-token cap to native exact and calibrated96.

Question:

```text
A price rises by 25%, then falls by 20%. What is the net percentage change?
Return only the signed percentage.
```

Both modes generated the identical ten output IDs:

```text
10 20 4 163588 12092 163589 163588 2778 163589 163586
```

Both decode to `+5%` followed by the same Kimi response/message terminators.
This answer is mathematically wrong—the correct answer is `0%`—but it is a
token-exact behavioral match to the native quantized teacher. Therefore the
earlier hard-prompt `5` cannot be used as evidence that calibrated slabs caused
the failure.

| measurement | native exact | all-92 calibrated96 |
| --- | ---: | ---: |
| output IDs | reference | 10/10 identical |
| elapsed | 1,219.81 s | 1,692.12 s |
| measured disk reads | 3,256.27 GB | 3,140.53 GB |
| graphics-board energy | 141.41 kJ | 196.40 kJ |
| logical expert bytes | 564.83 GB full exact | 375.93 GB provider |
| logical saving | — | 33.44% |

The calibrated provider used 510,852 selected slab records, 71,191 calibrated
routes, and 14,185 exact-fallback routes over 58 evaluated token steps per
routed layer. The physical prototype is still a systems failure: it took
38.72% longer and used 38.89% more graphics-board energy despite reading 3.55%
fewer physical disk bytes. No serving speedup is claimed.

The complete machine-readable comparison is
`/mnt/kimi-k3/results/kimi-h20-chat-template-math/comparison.json`. The exact
rendered prompt SHA-256 is
`0b3a1eacd64dc1e40dfec292a1c929d6e2a42a27da33209288e2cfbcdba5ac6a`.

## Road ahead

1. Replace raw-string tests with the checkpoint's embedded official chat
   template throughout the frozen suite.
2. Expand to prompts on which native exact is independently known to succeed;
   compare full-vocabulary logits as well as generated tokens.
3. Run the frozen suite and compare full-vocabulary
   logits against exact.
4. Capture a much broader, domain-balanced calibration corpus, refit all 92
   independent layers, and raise the minimum-hit policy based on coverage.
5. Only after quality gates pass, optimize I/O/computation and measure speed.
