# Layer-12 matched-byte full-width VQ control

## Verdict: NO-GO for this static 0.78125-bit codec

This is a deliberately narrow, held-out rate/distortion test.  It asks whether
keeping all 192 active 256-neuron blocks live, while encoding their weights at
one half of the deployed routed-bank byte budget, beats the existing 96/192
exact-block approximation with its mean tail.

It does not measure final-logit KL, generation, NVMe serving throughput, or a
packed runtime.  It must not be described as a general lower bound for every
activation-aware sub-one-bit quantizer.

## Fixed byte contract

For one K3 routed expert, gate, up, and down contain 33,030,144 scalar weights.
The codec was fixed before looking at held-out outputs:

| item | representation | bytes/expert |
| --- | --- | ---: |
| vector labels | 64-entry codebook over 8-weight vectors, 6 bits/label | 3,096,576 |
| scales | FP16 root-mean-square scale for each 512 weights | 129,024 |
| **codec total** | **0.78125 bits/weight** | **3,225,600** |

With sixteen routed experts/token, the codec reads 51,609,600 logical bytes.
That is exactly 50% of the current 103,219,200-byte active expert payload and
matches the 96-of-192 exact slab budget.  Three FP16 64x8 codebooks add 3,072
resident bytes per layer; they are explicitly excluded from the per-token
traffic ledger and counted separately.

The codebooks were trained only on deterministic equal-expert samples of the
deployed checkpoint's already quantized/dequantized static weights.  No capture
latent, expert response, or held-out output chose a codebook or a code.

## Protocol

* Layer: 12 (zero-based K3 routed layer; the first layer of the second AttnRes
  block).
* Data: existing 10,000-token capture with whole-sequence split; 1,800 untouched
  validation tokens.
* Teacher: native routed-expert response records reconstructed with the original
  router weights.
* Control: Python full dequantization of the same IQ1_S layer before applying
  the VQ codec.
* Comparator: existing held-out calibrated 96-slab mean-tail and its
  non-deployable true-residual selector oracle.
* Interval: 2,000 paired whole-sequence bootstrap replicates.

## Measured results

| method | mean cosine to native routed aggregate | p05 cosine | mean relative L2 |
| --- | ---: | ---: | ---: |
| full Python dequantization control | 0.999824 | 0.999590 | 0.017933 |
| 96 exact slabs + mean tail | 0.865533 | 0.801300 | 0.493996 |
| 96 selector oracle | 0.879895 | 0.824682 | 0.469203 |
| **all live neurons, matched-byte VQ** | **0.460347** | **0.424925** | **0.890135** |

The all-width VQ codec loses 0.405185 mean cosine versus the actual 96-slab
provider (95% paired interval [-0.408515, -0.401279]) and 0.419548 versus the
selector oracle ([-0.422197, -0.416740]).  Its error remains essentially the
same after routed normalization and the routed up projection, so the latter
does not rescue this quantization noise locally.

The full-dequantization control is close to native (`0.999824`), establishing
that the failure is the 0.78125-bit VQ representation, not response assembly.

## Artifacts and reproduction

Result root on the NVMe volume:

```text
/mnt/kimi-k3/results/kimi_layer12_halfbit_vq_retry02.{json,csv,npz}
```

SHA-256:

```text
json  4e5937c91905fb4143790ce13946addd03003dfe1eef3bd3a14745d16a3bec40
csv   87a89b6d745bde5d734591bfa308c23d39c9da51d4b1c4797a2066de06384d9f
npz   2131c11f36e5bf07d7bd01847b56a246596127a66583d34f701fecc8ddca05df
```

Reproduction command:

```bash
KIMI_LAYER12_HALFBIT_PREFIX=/mnt/kimi-k3/results/kimi_layer12_halfbit_vq_retryNN \
  bash scripts/run_kimi_layer12_halfbit_vq.sh
```

The runner refuses to overwrite an existing prefix and holds the cooperative
GPU lease.  It records host/GPU/NVMe telemetry but makes no physical read-rate
or serving-speed claim because it does not yet write a packed sidecar.

## Decision

Do not build a runtime for this static vector codec, and do not promote it to
terminal-KL replay.  The productive follow-up remains correctness-hardened
progressive exact computation plus an AttnRes-phase transmission experiment.
Any future sub-one-bit route must first show a held-out local improvement over
the 96-slab selector oracle under the same complete byte ledger.
