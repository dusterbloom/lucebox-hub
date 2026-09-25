# systemone & djev: structured decisions on lucebox

A short map of what `/v1/systemone` is, the two mechanisms behind it, and how
to use, extend, or modify it. For the raw HTTP contract see
[`API.md`](./API.md#post-v1systemone-openjev-structured-classification); for
the ecosystem and checkpoint compatibility see [`openjev.md`](./openjev.md);
for the Strix-Halo djev-spark port see [`djev-halo-plan.md`](./djev-halo-plan.md).

## What it is

- **System One / Jev** (TypeSafe, closed): structured semantic decisions —
  hand it a premise and a small set of possible judgments, get back a
  probability-normalized answer, at the cost of **one forced position of
  inference** instead of a full generation.
- **openjev** — the open umbrella of reproductions of that pattern.
- **djev-spark** (`mmastrac/djev-spark`) — serves the same Jev-style
  decisions (noul / choice / score) directly off **DiffusionGemma's
  denoise-canvas** mechanism, rather than off a causal LM's next-token logits.

`/v1/systemone` is lucebox's own reproduction of the endpoint shape. It
supports three question types — `noul` (yes/no), `choice` (N-way), `score`
(1..levels).

## The two answer mechanisms

The endpoint is backend-agnostic: it renders a prompt that ends exactly where
the answer belongs, then asks the backend for a distribution over the
candidate labels.

1. **Causal first-token logits** (`qwen3`, `qwen35`, `qwen35moe`,
   `deepseek4`, `gemma4`, `laguna`). The backend runs one forward and exposes
   the raw `lm_head` logits at the first post-prefill position
   (`GenerateResult::first_token_logits`). The endpoint restricts those logits
   to the candidate label token ids and softmaxes over just those.
2. **Diffusion structured read** (`diffusion-gemma`). There is no
   next-token `lm_head` read — a diffusion model *denoises a canvas*. The
   read seeds `read_canvas` noise slots after the causal prompt prefix, runs
   the entropy-bound denoise loop (`set_sc` self-conditioning + temperature
   schedule), and reads the distribution at the canvas slot where the answer
   label actually lands. DiffusionGemma emits a thinking block
   (`<|channel>thought\n<channel|>`) before the answer, so scoring slot 0
   would score the channel marker; the scorer skips to the answer slot.

## Using it

See [`API.md`](./API.md#post-v1systemone-openjev-structured-classification) for
the full request/response. Minimal example:

```json
POST /v1/systemone
{
  "model": "diffusion-gemma",
  "messages": [{"role": "user", "content": "Premise: The cat sat on the mat."}],
  "questions": [
    {"id": "q1", "type": "noul",   "prompt": "Is this about an animal?"},
    {"id": "q2", "type": "choice", "prompt": "Which animal?",
     "options": ["cat", "dog", "bird"]},
    {"id": "q3", "type": "score",  "prompt": "How confident?", "levels": 5}
  ]
}
```

Notes specific to the diffusion path:

- Each label is matched in **both** surface forms — `" cat"` (leading space,
  the causal/mid-sentence spelling) and `"cat"` (the bare form DiffusionGemma
  emits after `<channel|>`) — and its probability is the **sum** over its
  forms of the softmax over the candidate union.
- The answer is read at the first canvas slot whose argmax is any label form,
  so the leading thinking block is skipped. If no label is found, the answer
  is an explicit **abstention** (`valid: false`), never a silent slot-0 score.

Contract notes (all responses):

- `probabilities` is a proper distribution over the labels (sums to 1);
  `candidate_mass` is how much probability the model put on the candidates at
  all (a low value means it preferred a non-candidate token).
- A label that is not a single token is rejected with `400` — pass unique
  single-token aliases (`A`/`B`/`C`) for arbitrary names and map back. There is
  no silent multi-token collapse.
- `confidence` is `1 - H(p)/ln K`: concentration over the offered labels, not
  the probability the answer is correct. Do not gate on it until calibrated
  (see `djev-halo-plan.md`). Probabilities of 1.000 are common on a decisive
  slot and are expected.

## Code map

| Concern | Where |
|---|---|
| Endpoint, question suffix, response assembly | `server/src/server/http_server.cpp` (`handle_systemone`, `systemone_question_suffix`) |
| Label resolution + candidate scoring | `server/src/common/systemone_score.h` |
| Request/result fields | `server/src/common/generation_types.h` (`want_first_token_logits`, `first_token_logits`, `first_token_slot_count`) |
| Causal backends: first-token logit capture | each backend's first-token-after-prefill site (e.g. `src/qwen3/qwen3_backend.cpp`) |
| Diffusion read | `server/src/diffusion/diffusion_backend.cpp` (the `want_first_token_logits` branch) → `run_diffusion_structured_read` in `server/src/diffusion/diffusion_decoder.cpp` |
| Diffusion graph seam | `server/src/diffusion/diffusion_model.h`, `.../diffusiongemma/` |

## Extending / modifying

**Add a new causal backend.** Populate `GenerateResult::first_token_logits` at
its first-token-after-prefill site when `want_first_token_logits` is set, then
add the arch to `kSystemoneSupportedArches` in `http_server.cpp` and to
`kArchCapabilities` in `server/src/common/model_capabilities.h`.

**Add a new diffusion family.** Implement `DiffusionModelGraph`
(`forward_block`, `set_sc`, the snapshot hooks) and register it in
`server/src/diffusion/diffusion_registry.cpp` + `common/backend_factory.cpp`;
see [`../src/diffusion/README.md`](../src/diffusion/README.md). The endpoint
and scorer need no change — they consume whatever the graph's forward
returns.

**Change how answers are scored.** Edit `server/src/common/systemone_score.h`
(label → token forms, answer-slot pick, per-label probability). It is
header-only and ggml-free by design, so it is unit-testable on CPU; the
tests live in `server/test/test_diffusion_decoder.cpp` (see the `17d`
`sysone-score` cases). Keep the causal `first_token_slot_count == 1` path
behaving exactly as before.

**Tune the diffusion read.** Knobs on `DiffusionConfig`
(`server/src/diffusion/diffusion_types.h`): `read_canvas` (canvas width, 32 =
djev-spark's width), `read_steps` (denoise steps), `read_slots_returned`
(leading slots handed to the scorer). Env overrides `DG_READ_SLOTS`,
`DG_READ_STEPS`, `DG_READ_RETURN` (integers, validated; invalid values are
logged and ignored).

## Known limitations / follow-ups

- Reference parity for the diffusion forward against the llama.cpp oracle is
  still open (the plan's Phase 0) — current correctness is agreement with
  plain generation on a small question set, not an oracle match.
- The diffusion read runs a **fixed** step count (no adaptive early-stop),
  always samples the intermediate canvas (does not honour `do_sample`), and
  picks the **first** slot whose argmax is any label form — a label word
  appearing inside the thinking block could still win; preferring the first
  form-slot after the last `<channel|>` is the next hardening.
- Multi-token labels are rejected (`400`), not collapsed to a sub-token.
- `confidence` is concentration over the offered labels and is **not yet
  calibrated**; do not gate routing/safety on it (Astra review, see
  `djev-halo-plan.md`).
- The causal and diffusion paths share `systemone_score.h`; the union-softmax /
  max-over-forms scoring is slightly different from the original causal
  single-id-per-label behaviour and deserves a causal regression check.

## Tests

- CPU: `server/test/test_diffusion_decoder.cpp` (`sysone-score`, `read-*`,
  `read-canvas`, `read-sc`) — build with the documented `g++` line in
  `../src/diffusion/README.md`.
- Server smoke: `server/tests/test_server_smoke.py`.
- Benchmark harness: `server/scripts/bench_diffusiongemma_agentic_pair.py`.
