# K3 DSpark broad-suite result

## Verdict

**MEASURED — always-on speculation is a suite-wide NO-GO; selective long-output
acceleration remains a real signal.**

This matched comparison used the revision-40 all-layer adaptive slab policy,
the official GGUF chat template with thinking disabled, a 24-token generation
cap, and the real RadixArk Q8_0 DSpark draft on the RTX 3090. Full-logit tracing
was disabled in both arms because it disables the speculative runtime.

| Metric | Autoregressive | Width-4 DSpark |
|---|---:|---:|
| True generated transitions | 129 | 129 |
| Sum decode time | 187.089 s | 197.947 s |
| Transition rate | 0.6895/s | 0.6517/s |
| Paired speedup | 1.000x | **0.9451x** |
| Native-success tasks retained | 12/12 | **11/12** |
| Token-identical outputs | — | **10/12** |
| Peak GPU memory | 15.69 GiB | 18.01 GiB |

The draft took 49 verification steps, accepted 141 of 241 proposed tokens
(58.51%), and committed 2.878 tokens per verification step. The short answers
were generally slower because draft overhead dominated. The two answers with
at least 16 generated tokens were faster: 1.329x for grammar and 1.464x for
code. The code answer was cut off at the fixed 24-token cap before it became a
valid complete function, so it is a task failure despite the useful speed
signal. The Italian difference was punctuation only and still passed.

## Interpretation

- **MEASURED:** always enabling this draft is 5.5% slower on the mixed suite.
- **MEASURED:** long continuations can amortize the verifier and become faster.
- **FAILED:** this result does not justify making speculation the default.
- **OPEN:** delayed or length-aware draft activation may retain short-answer
  latency while capturing the long-answer gain; it needs a fresh quality and
  timing gate before integration.

The registered machine-readable result is
`results/k3_dspark_r40_broad_suite.json`. It binds both suite manifests,
telemetry files, repository commit, prompt suite, model, draft, provider, and
layer-budget table by path and checksum.
