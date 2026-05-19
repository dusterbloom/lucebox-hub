# Memo — Running `sapientinc/HRM-Text-1B` on Luce

Quick assessment of what it would take to integrate the Hierarchical
Reasoning Model 1B as a Luce target (i.e. via `dflash/` server with KV
cache, spec decode, PFlash compression, etc.). Written 2026-05-19 after
inspecting `config.json` + `modeling_hrm_text.py` from the HF repo.

## What HRM-Text-1B is

A 1B-parameter dual-timescale recurrent transformer from Sapient
Intelligence. **Not** a standard Llama-family model. Apache 2.0,
pre-alignment (no SFT/RLHF), trained from scratch on structured public
data with a PrefixLM objective.

| Field | Value |
|---|---|
| Parameters | ~1 B (16 layers × 2 stacks × ~32 M ea) |
| Hidden size | 1536 |
| Intermediate (MLP) | 4096 |
| Layers per stack (H and L) | 16 |
| Attention heads | 12 (no GQA — `n_kv_heads = n_heads`) |
| Head dim | 128 |
| Vocab | 65 536 |
| Max position | 4096 |
| RoPE theta | 10 000 |
| Tie embeddings | No (separate `lm_head`) |
| Embedding scale | ~39.19 (input embeddings are multiplied before forward) |
| `H_cycles` × `L_cycles` | 2 × 3 (= 8 effective forward passes per token) |
| `prefix_lm` | True (bidirectional prefix block via `token_type_ids`) |

## The unique part — H/L cycle recurrence

HRM's forward is two transformer stacks (`H_module` slow, `L_module`
fast) iterated `H_cycles × (L_cycles + 1)` times. Pseudocode of the
working loop from `modeling_hrm_text.py:HrmTextModel.forward`:

```
inputs_embeds = embed_tokens(input_ids) * embedding_scale
z_H = inputs_embeds          # slow stream
z_L = z_L_init.expand_as(z_H) # fast stream init from a learned vector

for h in range(H_cycles):                # 2
    for l in range(L_cycles):            # 3
        cycle_offset = (h*(L_cycles+1) + l) * num_layers_per_stack
        z_L = L_module(z_L + z_H, kv[cycle_offset:cycle_offset+16], …)
    cycle_offset = (h*(L_cycles+1) + L_cycles) * num_layers_per_stack
    z_H = H_module(z_H + z_L, kv[cycle_offset:cycle_offset+16], …)

return z_H   # last-hidden  →  lm_head
```

KV cache layout: **128 effective slots** = `num_layers_per_stack=16 ×
H_cycles=2 × (L_cycles+1)=4`. Each L-stack invocation at `(h, l)` and
the trailing H-stack at `(h, L)` get their own 16-slot block. The
`cycle_offset` indexes into that flat 128-slot KV cache.

## What's easy on Luce

- **Weight load**: safetensors → ggml tensors. Luce already has
  `load_draft_safetensors` and `load_target_safetensors` paths in
  `dflash/src/draft/` and analogues for the qwen35 hybrid target.
- **Per-layer primitives**: RMSNorm, RoPE (theta=10000, standard
  NeoX-style), SwiGLU MLP, GQA-less multi-head attention. All exist
  in `dflash/src/qwen3/` already.
- **Tokenizer**: standard HF tokenizer with `tokenizer.json`. The
  server's tokenization layer is HF-driven so swapping in the HRM
  tokenizer is the same as any new target.
- **No GGUF available yet** for this model. Would need a custom
  conversion or use the safetensors directly (we already do this for
  the dflash drafter).

## What's hard

1. **H/L cycle recurrence is a fundamental contract change**. Luce's
   graph builders (`build_qwen35_graph`, `build_target_step`, etc.) all
   assume a single-pass transformer forward: input → N layers → output.
   HRM is `H_cycles × (L_cycles + 1) = 8` forward passes per token,
   with state injection (`z_L + z_H`) between stacks. Need a new graph
   builder per arch: `build_hrm_text_graph`.

2. **KV cache addressing**. Standard luce KV is `[n_layers, max_ctx,
   n_kv_heads, head_dim]` for each of K and V. HRM needs 128 slots,
   not 16. The cache index is `(h*(L_cycles+1) + l) * 16 + layer`. Not
   hard mathematically but every cache access in the graph builder
   needs to be parameterised by `(cycle_idx, layer_idx)` instead of
   just `layer_idx`. Multi-slot snapshots/restores (prefix cache) get
   8× more state to copy.

3. **Embedding scaling (×39.19)**. Trivial as a single multiply, but
   downstream numerical paths (FA, RMS norm thresholds, anything
   tuned for unit-scale embeddings) need to be checked.

4. **Prefix-LM bidirectional mask**. HRM uses `token_type_ids` to mark
   a bidirectional block; positions with `token_type_ids == 1` form a
   single block that attends bidirectionally to itself, and everything
   else is causal. Luce's mask builder
   (`dflash/src/common/...build_causal_mask`) is purely causal. Need a
   `build_prefix_lm_mask(token_type_ids)` variant. The PrefixLM
   convention is similar to Qwen's chat-template causal-with-prefix
   pattern, but expressed via per-position type IDs rather than a
   single prefix length.

5. **Spec decode compatibility is the showstopper for Luce's value
   proposition.**
   - **DFlash**: no drafter exists for HRM. Would need a small drafter
     trained against HRM hidden states (the z-lab Qwen3.6 drafter was
     specifically trained against Qwen3.6 hidden states). This is a
     training run, not an engineering task. Without a drafter, DFlash
     decode is target-only AR — i.e. no speedup.
   - **MTP**: HRM doesn't ship NextN heads. Could train them (similar
     to how Qwen3.6 MTP heads were added by Qwen team / unsloth), but
     again that's a training task. The H/L recurrence also means MTP
     heads would need to attach to `z_H` after all cycles, which is
     more or less how MTP works against the final hidden state on
     standard models — should be compatible.
   - **PFlash compression**: drafter (Qwen3-0.6B) tokenizer mismatch
     (HRM uses a 65 536-vocab tokenizer, drafter is Qwen3 151 936
     vocab). PFlash works in DRAFTER token space, decodes back to
     target tokens via tokenizer round-trip with word-boundary
     recovery. The round-trip already handles vocab mismatch
     (`scripts/laguna_pflash_niah.py` proves this for Laguna). So
     PFlash COULD work without retraining anything, as long as the
     drafter+target round-trip is wired up.

6. **prefix-cache / WARM hit / snapshot**. Each cache slot is 8×
   larger; per-conversation snapshots scale accordingly. Probably fine
   on a 24 GB 3090 at HRM's 4 K max-ctx (KV is small for a 1 B model)
   but worth measuring.

7. **PR #195 (draft loader)** is in the same family of work but
   doesn't directly apply — HRM doesn't have a DFlash draft sibling
   yet.

## Effort estimate (sane engineering pace, single contributor)

| Phase | Time | Output |
|-------|------|--------|
| 0. Convert safetensors → ggml in-process loader | 0.5 d | weights load on GPU |
| 1. `build_hrm_text_graph` single-cycle: just one H+L iteration | 1.5 d | first forward, validates against `HrmTextModel.forward` on CPU torch |
| 2. Loop to all `H_cycles × (L_cycles + 1)` iterations + KV slot addressing | 1.5 d | bit-accurate parity with HF reference at temp=0 |
| 3. Prefix-LM mask + `token_type_ids` plumbing through `GenerateRequest` | 0.5 d | bidirectional prefix block works |
| 4. KV-quant + FA window + chunked prefill on the new graph | 1 d | matches dflash performance baseline |
| 5. server.py arch dispatch + OpenAI/Anthropic compat | 0.5 d | end-to-end via existing daemon protocol |
| 6. Smoke benches (one-shot, no spec decode) on RTX 3090 | 0.5 d | baseline AR tok/s, accuracy on a held-out suite |
| | **~6 days** | **Working HRM target, AR decode only, no spec decode** |

| Phase (optional, research) | Time | Output |
|----------------------------|------|--------|
| 7. PFlash compression + MTP-less prefill on HRM long-ctx | 2 d | TTFT win at long ctx; no decode-side speedup |
| 8. Train a HRM-specific spec-decode drafter (DFlash-style) | weeks | DFlash decode for HRM |
| 9. Train HRM NextN heads (MTP-style) | weeks | MTP decode for HRM |

So **~6 engineer-days for AR-only HRM on Luce**; spec decode is
multi-week + needs training compute.

## Should we actually do this?

Reasons FOR:
- HRM is novel; punches above its weight on math/reasoning benchmarks
  for a 1 B model.
- 1 B at HRM's effective depth (~128 layers' worth of compute per
  token) could be interesting for cheap-VRAM agentic-reasoning
  workloads if the recurrence translates to good code/reasoning at
  cheap cost.
- Luce gets a non-Qwen arch in tree, demonstrates the generality of
  the prefix-cache / PFlash / KV-quant infrastructure.
- Validates the dispatch/abstraction we already have in
  `ModelBackend` — useful exercise for picking up Laguna later.

Reasons AGAINST:
- **Pre-alignment**: HRM is not chat or instruction-tuned. Useless for
  agentic coding via hermes/opencode/pi without your own SFT.
- **No spec-decode story**: AR-only is 30-50 tok/s at 1 B on a 3090
  via plain transformers — Luce buys us nothing here at 1 B size. The
  Luce stack's whole point is spec-decode + long-ctx; HRM at 1 B has
  neither.
- **Engineering cost is significant** (~6 days) for an arch that won't
  benefit from our differentiating features without further research.
- **Better alternative**: run HRM via vanilla `transformers` for
  evaluation; if it proves out, then commit to porting.

## Recommendation

Don't port to Luce now. Do this instead:

1. Run HRM via HF `transformers` directly on the downloaded weights to
   characterize it — generation quality, reasoning benchmarks vs your
   needs.
2. If HRM turns out to be useful, the right path is to ask Sapient to
   ship MTP heads (they own the training), then porting the recurrence
   to ggml costs ~6 days and Luce becomes immediately competitive.
3. The interesting cross-pollination is the OPPOSITE direction:
   HRM's H/L recurrence as a SPECULATOR for a larger target. Use a
   1 B HRM with 8 cycles to draft tokens for a 27 B Qwen3.6 target —
   the dual-timescale recurrence may produce drafts that align with
   target hidden-state evolution better than a single-pass Qwen3-0.6B
   does. **This is a research direction worth a separate spike,
   independent of running HRM as a target.**

## What is downloaded right now

`hf download sapientinc/HRM-Text-1B --local-dir /home/peppi/models/hrm-text-1b`
in flight at the time of this memo. Files: config.json,
configuration_hrm_text.py, modeling_hrm_text.py (custom_code),
model.safetensors (~2 GB BF16), tokenizer.json, tokenizer_config.json,
README.md, LICENSE.

## Quick repro to validate the model first (no Luce work needed)

```python
# requires transformers from main branch (HRM arch not in stable yet)
# pip install --upgrade "git+https://github.com/huggingface/transformers.git@main"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

m = AutoModelForCausalLM.from_pretrained(
    "/home/peppi/models/hrm-text-1b",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
t = AutoTokenizer.from_pretrained("/home/peppi/models/hrm-text-1b",
                                   trust_remote_code=True)
# Composite condition for CoT-ish reasoning:
prompt = "<|im_start|><|quad_end|><|object_ref_end|>What is 17 × 23? Show working.<|im_end|>"
ids = t(prompt, return_tensors="pt").to("cuda:0")
out = m.generate(**ids, max_new_tokens=128, do_sample=False)
print(t.decode(out[0]))
```

This will tell us if HRM is actually useful before any Luce work.
