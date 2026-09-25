# DFlash Server API Reference

HTTP server exposing OpenAI-compatible, Anthropic, and Responses API endpoints
for inference with speculative decoding. Model-independent — works with any
backend (qwen35, qwen3, gemma4, laguna).

---

## Endpoints

| Method | Path | Description | Status |
|--------|------|-------------|--------|
| GET | `/health`, `/` | Health check | ✅ |
| GET | `/v1/models` | List available models | ✅ |
| POST | `/v1/chat/completions` | OpenAI Chat Completions | ✅ |
| POST | `/v1/messages` | Anthropic Messages | ✅ |
| POST | `/v1/responses` | OpenAI Responses API | ✅ |
| POST | `/v1/systemone` | openjev structured classification (prefill-only) | ✅ all backends |

---

## Agent Turn Cache

Start the server with `--agent-turn-cache` to extend the existing in-memory
prefix cache through generated tool calls. After the model emits a valid tool
call, the server reuses its deepest compatible prefix checkpoint, replays the
uncached tail once, and saves the canonical completed turn. On the next OpenAI
Chat Completions or Responses request, only the appended tool result and new
suffix need prefill.

This is a server-wide optimization; request bodies do not change. It requires
`--prefix-cache-slots` to be nonzero. Compressed or token-rewritten prompts and
requests without a compatible checkpoint safely fall back to ordinary prefix
caching.

The replay moves prefill work out of the follow-up request when tool execution
is long enough to overlap it; it does not eliminate that work. Paged attention
and `--max-concurrency` do not yet support shared prefix blocks, so they cannot
be combined with Agent Turn Cache.

The exact full-prompt cache may remain enabled for identical-request hits.
Disable it only when benchmarking the incremental Agent Turn Cache benefit.

Successful Chat Completions and Responses requests expose the measured result
under `usage.timings`:

- `agent_turn_cache_hit`: the restored prefix includes a generated agent turn.
- `cached_prefix_tokens`: backend-confirmed tokens restored from KV state.
- `prefilled_tokens`: prompt tokens computed for this request.

---

## POST `/v1/chat/completions` (OpenAI-compatible)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `messages` | array | required | Conversation messages | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_tokens` | int | 4096 | Max output tokens | ✅ |
| `max_output_tokens` | int | 4096 | Alias for max_tokens | ✅ |
| `max_completion_tokens` | int | 4096 | Alias for max_tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling threshold | ✅ |
| `top_k` | int | 0 | Top-k filtering (0 = disabled) | ✅ |
| `seed` | int | 0 | Random seed for reproducibility | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens (range: -2.0 to 2.0) | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens (range: -2.0 to 2.0) | ✅ |
| `repetition_penalty` | float | 1.0 | HF-style multiplicative penalty (>1 penalizes) | ✅ |
| `rep_pen` | float | 1.0 | Alias for repetition_penalty | ✅ |
| `rep_window` | int | 256 | Token lookback window for penalties | ✅ |
| `tools` | array | none | Tool/function definitions | ✅ |
| `reasoning` | object | — | Reasoning effort control (`{"effort":"medium"}`) | ✅ |
| `chat_template_kwargs` | object | — | Direct template control (`{"enable_thinking":true}`) | ✅ |
| `stop` | string/array | — | Stop sequences | ✅ |
| `n` | int | — | Number of completions | ❌ TODO |
| `logprobs` | bool | — | Return log probabilities | ❌ TODO |
| `top_logprobs` | int | — | Number of top logprobs per token | ❌ TODO |
| `response_format` | object | — | JSON mode / structured output | ❌ TODO |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
| `logit_bias` | object | — | Per-token logit adjustments | ❌ TODO |
| `user` | string | — | End-user identifier (tracking) | ❌ TODO |
| `stream_options` | object | — | Streaming options (e.g. include_usage) | ❌ TODO 🔴 |

### Response Fields

| Field | Status |
|-------|--------|
| `id` | ✅ `chatcmpl_<hex>` |
| `object` | ✅ `"chat.completion"` / `"chat.completion.chunk"` |
| `model` | ✅ |
| `choices[].message.role` | ✅ |
| `choices[].message.content` | ✅ |
| `choices[].message.tool_calls` | ✅ |
| `choices[].finish_reason` | ✅ (`stop`, `length`, `tool_calls`) |
| `choices[].delta` (streaming) | ✅ |
| `usage.prompt_tokens` | ✅ |
| `usage.completion_tokens` | ✅ |
| `usage.total_tokens` | ✅ |
| `choices[].logprobs` | ❌ TODO |

---

## POST `/v1/messages` (Anthropic-compatible)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `messages` | array | required | Conversation messages | ✅ |
| `system` | string/array | — | System prompt (top-level) | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_tokens` | int | 4096 | Max output tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling | ✅ |
| `top_k` | int | 0 | Top-k filtering | ✅ |
| `seed` | int | — | Random seed | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens | ✅ |
| `thinking` | object | — | Thinking mode (`{"type":"enabled"}`) | ✅ |
| `tools` | array | — | Tool definitions | ✅ |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
| `stop_sequences` | array | — | Stop sequences | ✅ |
| `metadata` | object | — | Request metadata (tracing) | ❌ TODO |

### Response Structure

Follows Anthropic Messages API structure with `content` blocks:
- `type: "text"` — text content
- `type: "thinking"` — reasoning/thinking content
- `type: "tool_use"` — tool call

---

## POST `/v1/responses` (OpenAI Responses API)

### Supported Request Parameters

| Parameter | Type | Default | Description | Status |
|-----------|------|---------|-------------|--------|
| `model` | string | server default | Model identifier | ✅ |
| `input` | string/array | required | Input messages or text | ✅ |
| `instructions` | string | — | System instructions | ✅ |
| `stream` | bool | `false` | SSE streaming | ✅ |
| `max_output_tokens` | int | 4096 | Max output tokens | ✅ |
| `temperature` | float | 0.0 | Sampling temperature | ✅ |
| `top_p` | float | 1.0 | Nucleus sampling | ✅ |
| `seed` | int | — | Random seed | ✅ |
| `frequency_penalty` | float | 0.0 | Penalize frequent tokens | ✅ |
| `presence_penalty` | float | 0.0 | Penalize present tokens | ✅ |
| `reasoning` | object | — | Reasoning effort | ✅ |
| `tools` | array | — | Tool definitions | ✅ |
| `tool_choice` | string/object | — | Tool choice / force tool usage | ✅ |
| `parallel_tool_calls` | bool | — | Allow parallel tool calls | ❌ TODO 🔴 |
| `store` | bool | — | Persist response | ❌ TODO 🔴 |
| `include` | array | — | Include extra response fields | ❌ TODO 🔴 |
| `text` | object | — | Structured output / JSON schema | ❌ TODO 🔴 |
| `service_tier` | string | — | Routing hint | ❌ TODO 🔴 |
| `previous_response_id` | string | — | Multi-turn chaining | ❌ TODO |

---

## POST `/v1/systemone` (openjev structured classification)

> Introduction, mechanism overview, and how to use/extend/modify:
> [`systemone.md`](./systemone.md).

Implements the "openjev"/Jev prefill-only classification protocol
([ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)):
each question costs exactly one forced token of inference. Instead of
generating text, the server renders the chat prompt plus a
question-specific suffix that ends right where the model would emit its
answer, reads the raw logits at that single position, restricts them to
the token(s) for each valid answer label, and renormalizes with softmax
over just those candidates. No decode loop runs, so this is much cheaper
than a normal chat completion.

First-token logit capture is wired up for all six causal backends (qwen3,
qwen35, qwen35moe, deepseek4, gemma4, laguna). The request forces AR
decode (`force_ar_decode=true`) so each backend's speculative-decode
path — which isn't hooked for logit capture — is bypassed in favor of
its plain AR-decode / hybrid-decode first-token site (see
`server/src/common/generation_types.h`'s `want_first_token_logits` /
`first_token_logits`, and `server/src/qwen3/qwen3_backend.cpp` for a
reference implementation).

`diffusion-gemma` is also supported, but via a different mechanism: a
canvas-seeded **structured read** (`run_diffusion_structured_read`) that
denoises a block after the prompt and scores the label at the canvas slot it
lands on (diffusion emits a `<|channel>thought<channel|>` block before the
answer, so slot 0 is not the answer). See [`systemone.md`](./systemone.md)
and [`djev-halo-plan.md`](./djev-halo-plan.md).

For how this relates to the wider openjev ecosystem — including whether
existing Hugging Face openjev finetunes (e.g. `AlexWortega/openjev`, a
Qwen3.5-4B NLI classifier) can be used with this endpoint — see
[`openjev.md`](./openjev.md).

### Request

```json
{
  "model": "qwen3",
  "messages": [
    {"role": "user", "content": "Subject: URGENT invoice overdue, click here now!"}
  ],
  "questions": [
    {"id": "q1", "type": "noul",   "prompt": "Is this spam?"},
    {"id": "q2", "type": "choice", "prompt": "Which category?",
     "options": ["billing", "support", "sales"]},
    {"id": "q3", "type": "score",  "prompt": "Rate urgency 1-5", "levels": 5}
  ]
}
```

`messages` is the same chat message array `/v1/chat/completions` takes —
it forms the shared context every question is asked against. Each entry
in `questions` needs:

| Field | Type | Required | Description |
|-------|------|----------|--------------|
| `id` | string | yes | Echoed back in the response; caller-chosen |
| `type` | string | yes | `"noul"` (yes/no), `"choice"` (multi-way), or `"score"` (ranked level) |
| `prompt` | string | yes | The question text |
| `options` | string[] | `choice` only | ≥ 2 labeled options |
| `levels` | int | `score` only | Number of levels, default 5, range 2–20 |

### Response

```json
{
  "id": "sysone-...",
  "model": "qwen3",
  "answers": [
    {"id": "q1", "type": "noul", "valid": true, "answer": true,
     "confidence": 0.81, "candidate_mass": 0.991,
     "probabilities": {"Yes": 0.94, "No": 0.06}},
    {"id": "q2", "type": "choice", "valid": true, "answer": "billing",
     "confidence": 0.55, "candidate_mass": 0.972,
     "probabilities": {"billing": 0.71, "support": 0.20, "sales": 0.09}},
    {"id": "q3", "type": "score", "valid": false, "answer": null,
     "reason": "model placed negligible probability on the candidate labels",
     "confidence": 0.0, "candidate_mass": 0.0004,
     "probabilities": {"1": 0.10, "2": 0.55, "3": 0.20, "4": 0.10, "5": 0.05}}
  ]
}
```

Each answer carries:

| Field | Meaning |
|---|---|
| `valid` | `false` when the read did not resolve a decision; treat as **abstain**, do not read `answer`. |
| `answer` | bool for `noul`, option string for `choice`, 1-based level for `score`. `null` when `valid` is false. |
| `probabilities` | Proper distribution over the labels (sums to 1). Only meaningful together with `candidate_mass`. |
| `candidate_mass` | Total probability the model put on the candidate tokens; the rest went to non-candidate tokens. A near-zero mass means the model wanted something outside the offered options. |
| `confidence` | `1 - H(p)/ln K` — concentration over the offered labels, **not** a probability the answer is correct. Do not gate on it until calibrated (see `systemone.md`). |

`valid: false` (with `reason`) replaces the previous silent fallbacks: a
label that is not a single token is rejected with `400` at request time (pass
unique single-token aliases such as `A`/`B`/`C` and map back yourself); a
canvas with no candidate slot, or a negligible `candidate_mass`, returns an
explicit abstention rather than scoring slot 0 or an all-zero vector. A `400`
is returned for malformed questions and a `501` for unsupported backends.

---

## Sampling Chain

The sampler applies penalties and sampling in this order:

```
logits (from GPU)
  → repetition_penalty (multiplicative, HF-style)
  → frequency_penalty + presence_penalty (additive, OpenAI-style)
  → top_k filtering
  → temperature softmax
  → top_p (nucleus) filtering
  → random draw (or argmax if temp=0)
```

All penalties respect `rep_window` (default 256 tokens lookback).

When `temperature = 0`:
- Penalties are still applied to logits
- Argmax is used (deterministic, no random sampling)
- Speculative decode is enabled only when no logit processing is needed

---

## Streaming

- **OpenAI format**: `text/event-stream` with `data: {...}\n\n` chunks, terminated by `data: [DONE]\n\n`
- **Anthropic format**: SSE with `event: message_start`, `content_block_start`, `content_block_delta`, `message_stop`
- **Responses format**: SSE with `response.created`, `response.output_item.*`, `response.completed`
- During long prefill or cleanup phases with no token data, the server emits a
  valid SSE comment (`: keep-alive`) every 15 seconds. Clients ignore the
  comment as payload while HTTP body-idle timers remain active. If the peer
  closes, the server propagates cancellation into the backend at its next safe
  prefill or decode boundary.

---

## Features

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-turn conversation | ✅ | Full message history |
| Tool/function calling | ✅ | XML and JSON tool parsing |
| Stop sequences | ✅ | OpenAI `stop` and Anthropic `stop_sequences` |
| Thinking/reasoning | ✅ | OpenAI `reasoning.effort`, Anthropic `thinking.type` |
| Prefix cache (memory) | ✅ | Automatic KV cache reuse |
| Prefix cache (disk) | ✅ | Persistent across restarts |
| PFlash (speculative prefill) | ✅ | Compresses long prompts |
| Client disconnect detection | ✅ | Aborts generation on disconnect |
| CORS | ✅ | Enabled by default |
| Tool memory | ✅ | Caches tool call results |

---

## TODO

### 🔴 High Priority (used by Codex and/or Claude Code)

These parameters are sent by real Codex CLI or Claude Code clients. Missing
support causes errors or silent feature degradation.

| Feature | Used By | Notes |
|---------|---------|-------|
| **`parallel_tool_calls`** | Codex (Responses API) | Codex always sends `true`. Can accept and ignore (we serialize calls). |
| **`store`** | Codex (Responses API) | Controls response persistence. Accept field; can be no-op locally. |
| **`include`** | Codex (Responses API) | Controls what's included in response events. Accept field. |
| **`text` (structured output)** | Codex (Responses API) | JSON schema output formatting (`{"format":{"type":"json_schema","schema":{...}}}`). Needed for structured tool outputs. |
| **`service_tier`** | Codex (Responses API) | Routing hint (e.g., `"default"`). Accept and ignore. |

### 🟡 Medium Priority

| Feature | Used By | Notes |
|---------|---------|-------|
| **`response_format`** | Chat Completions API | JSON mode / structured output (OpenAI Chat format). |
| **`metadata`** | Claude Code (Anthropic) | Request metadata for tracing. Accept and ignore. |
| **`stream_options`** | Some OpenAI clients | `{"include_usage": true}` — usage in final streaming chunk. |
| **Input validation** | — | Clamp penalty ranges [-2,2], reject invalid params with 400. |
| **`previous_response_id`** | Codex (Responses API) | Multi-turn response chaining (we handle via `input` already). |

### 🟢 Low Priority

| Feature | Notes |
|---------|-------|
| **`logprobs` / `top_logprobs`** | Token probabilities in response. Debugging/analysis only. |
| **`n` (multiple completions)** | Generate N choices per request. No known agent uses this. |
| **`logit_bias`** | Per-token logit adjustments. |
| **`user`** | End-user identifier (tracking only). |
| **`prompt_cache_key`** | Codex sends this for server-side caching hints. We have our own prefix cache. |
