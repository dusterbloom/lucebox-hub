# P24 — Persistent sparse expert workspace

VERDICT: GO

## Question

Can the calibrated96 scratch provider retain its full-width native arithmetic
while avoiding one graph construction and device allocation for every routed
expert evaluation?

## Change

`SparseDeviceExpertEvaluator` caches one GGML graph and its device tensors for
each expert geometry and activation-mask mode. Each call still:

- clears the complete gate, up, and down tensors on the device;
- uploads only selected authoritative slab bytes;
- applies the unchanged activation mask and native full-width down reduction;
- returns one expert result before the caller performs the frozen expert-ID
  ordered accumulation and mean-tail addition.

The optimization is opt-in through:

```text
DFLASH_KIMI_P23_PERSISTENT_SCRATCH=1
```

The default remains the one-shot allocation path.

## Reproduction

Base commit: `b794548d0548096ccaacf434070e99955500b080`

```bash
KIMI_P23_PERSISTENT_SCRATCH=0 KIMI_P23_STAGE_PROFILE=1 \
  scripts/run_kimi_p23_core_family_smoke.sh latent,shared \
  /mnt/kimi-k3/results/kimi-p24-ab-baseline-two-token-20260816 2

KIMI_P23_PERSISTENT_SCRATCH=1 KIMI_P23_STAGE_PROFILE=1 \
  scripts/run_kimi_p23_core_family_smoke.sh latent,shared \
  /mnt/kimi-k3/results/kimi-p24-ab-persistent-two-token-20260816 2
```

The baseline arm was repeated after the persistent arm at
`kimi-p24-ab-baseline-r2-two-token-20260816` to control for run order and page
cache warming.

## Semantic gate — MEASURED PASS

The baseline, persistent arm, repeated baseline, and prior frozen P23 reference
all produced the same logits file:

```text
SHA-256 f4f694d31d6d00c3b0d941b66440f9c271be60e560f8c0e87ea25d230e3848af
```

The longer eight-row persistent run also matched the prior eight-row logits
byte for byte:

```text
SHA-256 8daa924c13dd94489541f5d259eb2b72873b9cd49a074aee348aecd5dae90ca7
output IDs 11 374 4936 261 814 2742 316 374
```

## Controlled two-row A/B — MEASURED

| arm | decode transition | expert-provider stage | direct I/O | peak VRAM |
|---|---:|---:|---:|---:|
| baseline, first | 9.068 s | 6.232 s | 1.570 s | 15,949 MiB |
| persistent | **7.452 s** | **5.142 s** | 1.580 s | 15,989 MiB |
| baseline, repeated | 9.166 s | 6.171 s | 1.664 s | 15,949 MiB |

Against the two bracketing baselines, persistence reduced transition latency by
17.8–18.7% and increased transition throughput by 21.7–23.0%. The expert stage
fell by 16.7–17.5%. Direct-I/O time did not improve, as expected; the change
removes graph/device allocation rather than storage work.

## Eight-row check — MEASURED

Compared with the frozen pre-change eight-row run:

| metric | pre-change | persistent | change |
|---|---:|---:|---:|
| decode total for 7 transitions | 65.191 s | 58.932 s | -9.6% |
| seconds per transition | 9.313 s | 8.419 s | -9.6% |
| transitions/s | 0.1074 | 0.1188 | +10.6% |
| prefill | 18.967 s | 9.454 s | cache-sensitive |
| peak VRAM | 15,958 MiB | 15,989 MiB | +31 MiB |

Host/process disk reads differed materially across these two runs, so the
prefill improvement is not attributed solely to persistence. The bracketing
two-row A/B is the cleaner latency result.

## Interpretation

This is a real systems gain with no model-quality trade. It confirms that a
meaningful fraction of the current expert-provider cost is framework lifecycle
overhead, not storage or K3 mathematics. It does not reduce authoritative
bytes, arithmetic width, or the remaining per-expert synchronizations.

Next: retain the proven persistent tensors while batching selected-slab uploads
and expert graph execution at the layer level, preserving deterministic
expert-ID accumulation.
