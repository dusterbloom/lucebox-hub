# H23 broader native-success quality gate

Status: `MEASURED NATIVE 12/12 PASS — 1.2201-GiB CANDIDATE NEXT`

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
