# H18 Task B — cross-expert functional redundancy microscope

**VERDICT: WEAK / NO-GO for extreme cross-expert neuron pooling.**

## Predeclaration

Before any redundancy measurement, the experiment fixed exactly 32 layer-one
experts:

- the 16 experts with the most routes in the registered 10K capture;
- 16 deterministic random experts from the ascending expert-ID population with
  at least 50 routes on registered held-out sequences.

The random draw used `numpy.random.default_rng(260814).choice(..., size=16,
replace=False)`.  Every neuron was evaluated on the same 512 inputs, selected
round-robin from 87 calibration sequences.  Registered validation sequences
were excluded from the common probe.

This gives 98,304 native neurons.  For a source neuron `i` and a candidate
neuron `j` in a different expert, the measured replacement was

```text
alpha_ij = E[a_i a_j] (d_i^T d_j) / (E[a_j^2] ||d_j||^2)

C_ij = E||a_i d_i - alpha_ij a_j d_j||^2
       / E||a_i d_i||^2

     = 1 - rho(a_i,a_j)^2 cos(d_i,d_j)^2
```

Here `a` is the post-gate/up SiTU activation over the common probes and `d` is
the dequantized native down-projection column.  Candidates from the source
expert were excluded structurally.

## Exact search

The GPU held about 1.6 GiB of normalized activation and down-column geometry.
The search streamed all 496 unordered expert pairs as 3,072-by-3,072 matrix
products; it never materialized a 98,304-square matrix.  Primary products were
FP32 with TensorFloat-32 disabled.  Total execution, including shard hashing,
geometry construction, search, direct controls, and serialization, took 61.9
seconds.

The algebraic shortcut was checked against 16 explicitly materialized
512-by-3,584 rank-one contribution comparisons: eight deterministic random
pairs and eight best matches.  Maximum absolute cost discrepancy was
`4.31e-10`, passing the `2e-5` tolerance.

## Results

### Best different-expert replacement cost per neuron

| statistic | normalized cost `C` |
|---|---:|
| minimum | 0.971415 |
| median | 0.999771 |
| p75 | 0.999857 |
| p90 | 0.999902 |
| p95 | 0.999917 |
| p99 | 0.999934 |

Coverage was zero at every preregistered threshold:

| threshold | fraction of 98,304 neurons |
|---|---:|
| `C <= 0.01` | 0 |
| `C <= 0.05` | 0 |
| `C <= 0.10` | 0 |
| `C <= 0.20` | 0 |

The weak/no-go gate required fewer than 50% at `C <= 0.10`.  The measured
fraction is zero, so medoid rate-distortion is not earned.

### Why the matches fail

Activation-only similarity is already weak: the best different-expert
activation correlation squared has mean `0.14332` and median `0.13193`.  Only
`0.0122%` of neurons reach activation-only cost `C <= 0.10`.

For the candidates that maximize the true joint score:

| component | mean | median | maximum |
|---|---:|---:|---:|
| activation correlation squared | 0.07873 | 0.06791 | 0.91258 |
| down-column cosine squared | 0.003677 | 0.003388 | 0.23912 |
| joint product | 0.000277 | 0.000229 | 0.028585 |

Even an unusually similar activation is generally paired with an unrelated
down direction, and vice versa.  The best possible pair explains at most
`2.86%` of one source neuron's rank-one contribution energy.

Matches are not concealed in one exceptional expert pair.  All 992 possible
directed expert pairs receive at least one best match.  The largest pair holds
only `0.213%` of source neurons, the top five hold `0.883%`, and the directed
pair Herfindahl index is `0.001053`, close to diffuse.

## Decision

Do not run medoid clustering, build a shared-neuron pool, or implement a pooling
runtime from this hypothesis.  The real layer-one geometry rejects an extreme
cross-expert sharing factor much more strongly than the preregistered no-go
threshold.  This reinforces progressive exact slab hydration: useful savings
must preserve expert-specific native directions rather than substituting
neurons from other experts.

## Reproduction

```bash
flock -x /tmp/lucebox-gpu-0.lock bash -lc \
  'PYTHONPATH=scripts python3 \
    scripts/probe_kimi_h18_cross_expert_redundancy.py \
    /mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00002-of-00014.gguf \
    /mnt/kimi-k3/captures/kimi_layer01_10000.bin \
    results/kimi_h18_cross_expert_predeclaration.json \
    results/kimi_h18_cross_expert_redundancy.json \
    results/kimi_h18_cross_expert_best_matches.npz \
    --threads 8 --device cuda'
```

The complete per-neuron best-match assignments and costs are preserved in the
compressed NPZ artifact referenced and hashed by the summary JSON.
