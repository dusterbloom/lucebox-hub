# Prompt-lookup drafting: offline estimate (CPU-only, no box job)

## Data used
- Output token IDs (`emitted_ids`, 512 tokens/run, greedy, spec=2 but
  draft_offered=draft_accepted=0 so this is plain AR ground truth): from
  `qwen4exp-prefix-cache/docs/handoffs/strata-valid-comparison/evidence/
  strata-v4-20261007/strata-latest-matrix-20261007-ar-v4/rep0-{workload}-{len}/
  result.json`, one run per workload x length (code/counting/prose x 16K/64K).
- **Prompt token IDs were NOT available.** `prompt-manifest.json` only has
  prompt *text files* (`/tmp/...`, now gone) and the actual gguf vocab lives
  on the box only (no `tokenizer.json`/sentencepiece for qwen4exp found
  locally; the only local gguf, `Qwen3.8-Flash-Next-...experimental-speed-
  projection.gguf`, is a different checkpoint and its vocab was not assumed
  to match — not used). So lookup matching below is **output-history-only**:
  at step t it can only match against tokens 0..t-1 of the *already decoded
  output*, never the prompt. This is a hard lower bound on real coverage:
  code/counting prompts are built from ~430-1724x literal filler repeats, so
  real prompt-side lookup hit-rate is almost certainly much higher than what
  follows.

## Lookup simulation (greedy ground truth, n=4 falling back to 3,2)
| workload | D | coverage | mean accepted | tok/verify-step |
|---|---|---|---|---|
| code16K/64K | 3 | 0.27/0.26 | 0.73/0.71 | 1.20/1.19 |
| counting16K/64K | 3 | 0.354 | 0.039 | 1.014 |
| prose16K/64K | 3 | 0.125/0.113 | 0.359/0.241 | 1.045/1.027 |

D=7 never beats D=3 net (extra verify-width cost > extra accepted gain), so
D=3 is used below.

## Cost model and tok/s estimates (Q8, 8K, eager verify; from VERIFY-COST.md)
verify(k)=43.2+11.4(k-1) ms, draft=5.9 ms/token, plain-AR k=1=38.25 ms.
MTP d=2 per-draft acceptance: code/prose=0.775 (prose unmeasured, **assumed
= code**, flagged), counting=1.0 (from mtp-lookup REPORT).

| workload | MTP-only d=2 | lookup-only D=3 | hybrid (lookup, else MTP d=2) | Strata MTP measured |
|---|---|---|---|---|
| code16K | 30.5 | 24.5 | 28.3 | 46.75 |
| code64K | 30.5 | 24.5 | 28.4 | - |
| counting16K | 38.6 | 19.5 | 29.7 | 43.35 |
| counting64K | 38.6 | 19.5 | 29.7 | - |
| prose16K | 30.5 | 24.2 | 28.9 | - |
| prose64K | 30.5 | 24.1 | 28.9 | - |

## Decision
Lookup-only is **worse than plain AR** (26.1 tok/s at 38.25ms/tok) on every
workload here, and hybrid never beats pure MTP-only. Adding prompt-lookup
drafting on top of MTP is not justified by this data: output-side repetition
alone doesn't clear the extra verify-width cost. The cost model also
undershoots Strata's measured MTP numbers (30.5 vs 46.75 measured for code),
so this is a conservative/lower-bound model, not a tuned one — but the gap
between policies (hybrid < MTP-only) is model-independent since both use the
same cost function. The one real unknown is prompt-side repetition (430-1724x
filler), which this run couldn't test (no prompt tokens, no local tokenizer).
**Verdict: don't build prompt-lookup from this evidence alone.** If still
wanted, the next cheapest step is a box-side hit-rate probe over the real
prompt+output stream (not a full build) before committing engineering time.
