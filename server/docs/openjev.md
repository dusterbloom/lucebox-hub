# openjev: ecosystem notes and compatibility with `/v1/systemone`

This note tracks what "openjev"/"Jev" is in the wider ecosystem, and
whether existing openjev finetunes on Hugging Face can be dropped in
against lucebox's own `/v1/systemone` endpoint (see
[`API.md`](./API.md#post-v1systemone-openjev-structured-classification)).

## What "Jev" / "openjev" is

Jev is TypeSafe's closed-source runtime service for structured semantic
decisions: hand it a premise and a small set of possible judgments, get
back a probability-normalized answer, at the cost of one forced token of
inference instead of a full generation. `/v1/systemone` in this repo is
lucebox's own reproduction of that interface shape — see `API.md` for
the request/response contract and which backends have first-token logit
capture wired up.

"openjev" is the open-source umbrella for reproductions of that pattern.
Relevant projects found on GitHub/Hugging Face:

- [`ekzhang/openjev-sglang`](https://github.com/ekzhang/openjev-sglang) —
  the SGLang-based reference this repo's `/v1/systemone` design followed.
- [`AlexWortega/openjev`](https://huggingface.co/AlexWortega/openjev) —
  an actual finetuned checkpoint (see below).
- [`TheoLeeCJ/openjev`](https://github.com/TheoLeeCJ/openjev) — running a
  Jev-shaped model on consumer hardware (3090).
- [`sypherin/jev-trace-classifier`](https://github.com/sypherin/jev-trace-classifier) —
  applies the "noul" (binary judgment) primitive to a real classification
  task (agent- vs human-authored text).
- `awesome-typesafe` (multiple forks, e.g.
  [`Friedjof/awesome-typesafe`](https://github.com/Friedjof/awesome-typesafe)) —
  curated link list for TypeSafe/System One/Jev resources.

## `AlexWortega/openjev`: a Qwen3.5 Jev-style finetune

This is the concrete, loadable checkpoint most relevant to lucebox, since
lucebox runs Qwen3.5-family models natively (`qwen35`, `qwen35moe`
backends).

- **Task**: 3-way natural language inference (NLI) — reads a premise and
  a hypothesis, answers `entailment` / `contradiction` / `neutral`.
- **Small variant**: `qwen3.5-4b-nli/` subfolder — base model
  **Qwen3.5-4B**, architecture `Qwen3.5ForSequenceClassification`.
- **Large variant**: `mlp_heads_35b/` — Qwen3.5-35B-A3B (MoE) backbone
  with per-task MLP heads.
- Loadable via `AutoModelForSequenceClassification.from_pretrained(
  "AlexWortega/openjev", subfolder="qwen3.5-4b-nli")`, or their own
  `OpenJevCrossEncoder` (`predict` / `rerank` / `grade` / `latents`).
  MIT licensed.

## Does it map onto `/v1/systemone`'s `choice` type? — No, not as-is

`/v1/systemone`'s `choice` type works by:

1. Rendering the chat prompt plus a question suffix that ends exactly
   where a plain causal LM would emit its answer as the *next token*.
2. Tokenizing each option label (e.g. `"billing"`, `"support"`) to get
   candidate vocab token ids.
3. Reading the raw **vocab logits** (the model's normal `lm_head`
   output) at that one forced position, restricting them to just the
   candidate token ids, and softmax-normalizing over that restricted
   set.

This only works because a plain causal LM's `lm_head` logits are
trained end-to-end to be meaningful as next-token probabilities — no
extra machinery is needed beyond what lucebox's backends already run
for ordinary chat completion.

`AlexWortega/openjev`'s Qwen3.5-4B checkpoint is architecturally
different: `Qwen3.5ForSequenceClassification` adds a **separate
classification head** (a linear `score` layer) on top of the last-token
hidden state, trained from scratch with plain cross-entropy over the
3 NLI labels. The useful trained signal — the actual entailment /
contradiction / neutral judgment — lives in that head's 3 output
logits, **not** in the base model's vocab `lm_head` logits. The vocab
logits at that position were never optimized to say anything about
NLI at all.

Concretely, this means:

- lucebox's backends (qwen35 included) only ever compute and expose the
  vocab `lm_head` logits (`GenerateResult::first_token_logits`, added
  for `/v1/systemone` — see `API.md`). None of them load or run a
  sequence-classification head.
- Restricting `first_token_logits` to the token ids for the words
  "entailment" / "contradiction" / "neutral" and softmaxing would
  produce *some* number, but it would not reproduce
  `AlexWortega/openjev`'s trained judgment — that model's weights for
  this task live entirely in the untouched classification head, which
  `/v1/systemone` never loads or evaluates.
- Loading this specific checkpoint's real behavior into lucebox would
  need a new, separate code path: load the extra `score` head weights
  alongside the base Qwen3.5 weights, run the forward pass through to
  the last hidden state, apply the head, and expose its 3 logits — a
  different integration surface from `want_first_token_logits`, closer
  in shape to the qwen35/qwen35moe backend loaders that already exist.
  Not attempted here.

### What *does* map cleanly

The **task shape** — few fixed labels, one forced decision, probability
over the label set — is exactly what `/v1/systemone`'s `choice` type was
built for, and it works today with any of lucebox's six supported
backends using their existing instruction-tuned chat weights (no extra
checkpoint needed):

```json
POST /v1/systemone
{
  "model": "qwen35",
  "messages": [
    {"role": "user", "content": "Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat."}
  ],
  "questions": [
    {"id": "nli", "type": "choice", "prompt": "Does the hypothesis follow from the premise?",
     "options": ["entailment", "contradiction", "neutral"]}
  ]
}
```

This is a **prompted approximation**, not the trained
`AlexWortega/openjev` classifier — expect lower accuracy than the actual
finetuned head, since it relies on the base chat model's zero-shot NLI
judgment rather than a model trained specifically for this task. It is,
however, immediately usable with no new integration work.

## Summary

| | `/v1/systemone` choice, prompted | `AlexWortega/openjev` (as released) |
|---|---|---|
| Works with lucebox today | ✅ yes | ❌ no — needs a classification-head code path |
| Uses openjev's trained NLI weights | ❌ no (zero-shot prompting) | ✅ yes (if that path were built) |
| Integration effort | none | new loader + head-eval path, not built |
