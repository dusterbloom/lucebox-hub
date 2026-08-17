# H23 broader native-success quality gate

Status: `MEASURED MAJOR GO — BENCHMARK ALIGNMENT NEXT`

This gate was frozen after the measured 1.2201-GiB six-task result and before
opening any candidate output.  It reuses the official GGUF chat template, exact
native K3 teacher, 10K calibration, progressive provider, mean tail, fallback
rules, and 1.2201-GiB layer-budget table unchanged.

## Frozen suite

- Fixture: `server/test/fixtures/kimi_k3_h23_broad_native_success_v1.jsonl`
- SHA-256: `6d1c0583df52738820559bef66f6a96839bcde44c0bae7bdc4bb7bbe7332d4cc`
- Prompts: 12, two each for factual, code, arithmetic/reasoning, grammar,
  multilingual translation, and extraction/retrieval
- Generated-token cap: 24 per prompt
- Chat template: GGUF Jinja, thinking disabled

The candidate is not run unless native K3 passes all twelve registered task
predicates.  A native failure stops this version; it is not counted as a
compression failure.

## Native baseline — measured pass

Native K3 passed all 12/12 registered predicates.  The run scored 552 prompt
tokens and 144 generated tokens.  Aggregate prefill was 1,867.36 seconds
(0.2956 token/s); aggregate decode was 436.47 seconds (0.3299 token/s).  The
long code case used the full 24-token cap and the decoy-filled retrieval case
returned `QUARTZ-918`.

- Suite manifest SHA-256:
  `ebe0448fbb4525949be4a5a4de6b5909e61852c8b5d39097244801cd7a0e1699`
- Native analysis SHA-256:
  `ad0161eeb95312fd893aa88fb1014f467419f9fe5881317e0bc813f7e0efc40c`
- Storage errors: zero
- Measured active expert I/O: 4.747 GiB/s

## Candidate gate

Run the frozen 1.2201-GiB policy without changing its calibration, selector,
tail, or exact fallback behavior.

- Primary: retain at least 11/12 native-success tasks and show no repetitive or
  degenerate answer.
- Secondary: generated-token agreement, first divergence, aligned-history
  terminal KL, top-1 agreement, exact provider bytes, fallback rate, and timing.
- Boundary: passing remains a useful medium-width screen, not a broad benchmark
  or long-context quality certificate.

## 1.220-GiB candidate — measured pass

The unchanged 10K-calibrated policy retained all 12/12 native-success tasks at
1.2223 logical routed GiB/model position.  Nine of twelve generated sequences
were token-identical.  The three divergences were behaviorally harmless:

- `CO₂` instead of native `Carbon dioxide (CO₂)`;
- `Buongiorno.` instead of native `Buongiorno`;
- `Muchas gracias.` instead of native `Muchas gracias`.

The initial scorer marked the Unicode `CO₂` answer wrong because it normalized
ASCII digits but not Unicode subscripts.  The literal registered prompt allowed
either the gas name or chemical formula.  A narrow subscript-digit
normalization control accepts `CO2`, `CO₂`, and `carbon dioxide` while rejecting
`oxygen`; no model output or policy changed.

| Metric | Native | 1.220-GiB candidate |
|---|---:|---:|
| task successes | 12/12 | 12/12 |
| token-identical generations | — | 9/12 |
| prefill | 0.2956 tok/s | 0.6760 tok/s |
| decode | 0.3299 tok/s | 0.7407 tok/s |
| model compute wall | 2,303.83 s | 1,008.21 s |
| end-to-end telemetry wall | 2,313.90 s | 1,018.46 s |

Candidate terminal KL mean/median/p95/max is
0.6849/0.3460/2.8427/5.3593 over 660 aligned-history rows; top-1 agreement is
435/660.  Exact fallback is 0.97% of route occurrences and 7.25% of provider
bytes.  The candidate uses 1.2223 GiB/position, 86.5% below the exact routed
baseline.

- Candidate manifest SHA-256:
  `bfc2cdfe1f4771d5ec438428307009963b65ef0841766dc548ca36bcf752f2a0`
- Candidate analysis SHA-256:
  `c4b71281d16627be9ec180c6fa09ace27c4574f64cf43ffab1d14ae2679d5a7c`
- Telemetry SHA-256:
  `a6ba50372401d71e6e85834da9a411aa63bef5e3e319c22f5e33edc277f84382`

## Benchmark alignment

The official K3 model card and current Lucebox scored suites have no exact
benchmark-name intersection.  K3 reports GPQA Diamond, HLE, DeepSWE and many
agentic/vision suites; Lucebox currently has reusable HumanEval, GSM8K and
Math500 scoring.  The shortest defensible bridge is a fixed preregistered
GPQA-Diamond subset, with Lucebox's existing Math500 subset as a separately
labelled companion.

Official K3 results use thinking enabled at maximum reasoning effort.  This H23
gate intentionally disables thinking for deterministic product-mode comparison,
so it must not be presented as model-card score reproduction.  GPQA therefore
requires a separate thinking-enabled protocol rather than changing this frozen
gate after the fact.

The suite runner now exposes that separate protocol as an opt-in only:

```bash
DFLASH_KIMI_H16_CHAT_TEMPLATE=1 \
DFLASH_KIMI_H16_ENABLE_THINKING=1 \
  server/build-k3-p20-cuda126b/run_kimi_k3_h16_suite ...
```

The default remains thinking-off, preserving every frozen H23 reproduction.

## Reproduction

```bash
KIMI_H23_FIXTURE=server/test/fixtures/kimi_k3_h23_broad_native_success_v1.jsonl \
KIMI_H23_N_GEN=24 \
KIMI_H23_NATIVE_ROOT=/mnt/kimi-k3/results/kimi-h23-broad-native-v1-20260817 \
KIMI_H23_NATIVE_ANALYSIS_OUTPUT=results/h23_broad_native_success_v1.json \
bash scripts/run_kimi_h23_quality.sh native
```

Only after the native gate passes, use the same fixture and token cap with the
registered 1.2201-GiB budget table and 10K runtime package in candidate mode.
