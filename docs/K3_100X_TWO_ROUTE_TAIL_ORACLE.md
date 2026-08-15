# K3 100x two-route directional-tail oracle

**VERDICT: NO-GO FOR THE REGISTERED TWO-ROUTE/RANK-128 OBSERVER COMBINATION.**

## Question

Can K3 retain only two dynamically selected routed experts, using six of their
twelve calibrated slabs on ordinary layers and both complete experts at
AttnRes block starts, while a rank-128 aggregate correction restores the
omitted routed direction?

The projected average authoritative traffic is 0.595 GiB/token. At P20's
measured 5.257 GiB/s, that corresponds to an expert-I/O roofline of 8.84
tokens/s and therefore satisfies the byte geometry needed for a 100x attempt.

## Registered first gate

Layer 12 is the first routed AttnRes block start and was deliberately selected
as the adversarial gate. The policy is reinforced there: both chosen experts
are complete, rather than six-of-twelve slabs.

The two experts are selected without held-out answers by:

```text
router weight * calibration mean native-response norm
```

All other experts use their calibration mean. No exact fallback is permitted
in the registered arm. A PCA basis is fit on whole training sequences only.
The validation coefficients are oracle projections of the true held-out
aggregate correction, so this is a strict fixed-basis ceiling, not a
deployable predictor.

The directional oracle targets the shortest correction from the retained
aggregate to the teacher's positive ray. This gives the proposal the benefit
of K3's routed RMSNorm scale invariance.

Predeclared continuation gate, measured after routed RMSNorm and the native
shared up-projection:

```text
rank-128 mean cosine >= 0.99
rank-128 p05 cosine  >= 0.95
```

## Data

- real Kimi K3 `UD-IQ1_S`, zero-indexed layer 12;
- 10,000 captured states with the native top-16 router;
- 7,502 training, 698 development (unused), and 1,800 held-out validation
  tokens, split by whole sequence;
- all 160,000 individual exact expert responses;
- 856 calibrated experts and 40 without calibration observations.

Only two routes in the complete 10K corpus involved an uncalibrated expert;
neither was selected as a live route. An optimistic control nevertheless keeps
every uncalibrated route exact.

## Result

Post-routed-up-projection metrics on untouched validation sequences:

| method | mean cosine | p05 cosine | mean relL2 |
| --- | ---: | ---: | ---: |
| two exact experts + mean tail | 0.71063 | 0.53216 | 0.70043 |
| same, all uncalibrated routes exact | 0.71068 | 0.53216 | 0.70039 |
| rank-64 Euclidean PCA oracle | 0.72792 | 0.57045 | 0.67958 |
| rank-128 Euclidean PCA oracle | 0.74339 | 0.59218 | 0.66112 |
| rank-256 Euclidean PCA oracle | **0.76421** | **0.62386** | **0.63551** |
| rank-128 directional-ray PCA oracle | 0.71114 | 0.54207 | 0.69847 |
| rank-128 directional oracle, exact coverage control | 0.71119 | 0.54207 | 0.69844 |

The registered rank-128 directional oracle misses the mean gate by 0.2789 and
the p05 gate by 0.4079. The coverage control is numerically immaterial. Even a
larger rank-256 Euclidean oracle remains very far from the gate.

## Decision

Do not train the proposed rank-128 block Observer. Do not implement its
two-route all-layer runtime. A causal coefficient predictor cannot outperform
the same fixed basis when that basis is given the true held-out coefficients.

This result closes only the registered combination:

```text
two live routes
+ block-start complete-prefix reinforcement
+ expert means
+ rank-128 private block-recurrent aggregate correction
```

It does not falsify a storage-heavy nonlinear response atlas, a nearly
full-rank correction, or 100x aggregate throughput from continuous batching.
Those are different hypotheses with different byte/compute contracts.

## Reproduction

```bash
python3 scripts/probe_kimi_100x_tail_oracle.py \
  /mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf \
  /mnt/kimi-k3/captures/kimi_layer12_10000.bin \
  /mnt/kimi-k3/results/kimi_layer12_panel_10000.teacher.f32 \
  /mnt/kimi-k3/responses/kimi_layer12_10000 \
  /mnt/kimi-k3/fit-state/kimi_layer12_neuron_slabs_calibration.npz \
  results/kimi_layer12_100x_tail_oracle_coverage_control.json \
  --layer 12 --device cuda
```

The first registered run completed in 8.80 seconds; the exact-coverage control
completed in 9.03 seconds. Raw telemetry is retained under
`/tmp/kimi-k3-100x-tail-oracle-layer12*` because the experiment NVMe is
currently mounted read-only.
