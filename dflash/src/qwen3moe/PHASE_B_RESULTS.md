# Phase B (DFlash spec decode) — Results & Limitations

Branch: `feat/qwen3moe-backend` | Last validated: 2026-05-26

## Status: Architecturally complete, blocked on lineage-matched drafter

The DFlash speculative-decode pipeline is wired end-to-end and produces
correct outputs. Architecture has been verified against z-lab's `dflash.py`
reference + SGLang DFLASH worker via 3 independent research passes.

Acceptance and throughput on the validated drafter pair, however, are
capped below the autoregressive baseline due to:
1. Drafter↔target fine-tune lineage mismatch (see below)
2. Implementation uses restore-then-replay verify (2× target forward),
   not llama.cpp's single-pass crop pattern

## Acceptance / throughput sweep (SR2AM-IQ2_XXS target + Qwen3-Coder-30B-A3B DFlash drafter)

| `DFLASH_BLOCK_SIZE` | per-pos accept | avg commit | tok/s | vs AR |
|---|---|---|---|---|
| 2  | **72.7 %** | 2.00 | 41.1 | -38 % |
| 3  | 47.1 % | 2.31 | 50.4 | -24 % |
| 4  | 36.3 % | 2.42 | **52.0** | -21 % |
| 8  | 16.6 % | 2.33 | 43.6 | -34 % |
| 16 | 7.2 % | 2.15 | 30.0 | -55 % |
| AR (no spec) | — | — | **66.0** | baseline |

The 72.7 % acceptance at `bs=2` empirically validates that the
implementation reads target captures correctly, projects them through
the draft, and runs the verify loop with sound accept arithmetic. The
drop from 72 → 16 → 7 % as `bs` grows reflects the drafter's
out-of-distribution generalization on a non-matching fine-tune — the
first prediction is accurate, but error cascades over longer horizons.

## Drafter / target lineage

| Component | Model | Base |
|---|---|---|
| Drafter | `z-lab/Qwen3-Coder-30B-A3B-DFlash` | trained against `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Our target | `sailing-lab/SR2AM-v1.0-30B` | fine-tuned from `Qwen/Qwen3-30B-A3B-Thinking-2507` |

Both share the Qwen3-30B-A3B architecture (hidden=2048, 48 layers,
n_head=32, n_head_kv=4, head_dim=128, rope_theta=1e7, vocab=151936) and
the drafter is structurally compatible — it loads, projects, and
generates coherent token candidates. But the captured residual-stream
activations at layers [1, 12, 23, 34, 45] differ across fine-tunes
(Coder vs Thinking), so the drafter's `fc` projection (10240→2048) is
out-of-distribution.

No DFlash drafter exists publicly for any Qwen3-30B-A3B Thinking or base
variant as of 2026-05-26.

## What's needed to unlock the speed-up

In rough order of effort:

1. **Single-pass crop verify** — replace `snapshot_kv → verify → restore_kv
   → replay` with `verify → crop KV to accept+1`. This is the canonical
   llama.cpp pattern. Eliminates one full target forward per spec round
   → roughly cuts the verify cost in half → spec decode becomes
   net-positive even with current ~36-72 % acceptance. Estimated ~1 day
   of work in `Qwen3MoeDFlashTarget::verify_batch` and the surrounding
   `do_spec_decode` loop. This alone should ship a real speed-up
   regardless of drafter quality.

2. **Train a Thinking-lineage drafter** — re-fit just the drafter's `fc`
   projection (~21 M params) against SR2AM-paired captures using z-lab's
   distillation pipeline. Estimated 1-3 days end-to-end. Would raise
   acceptance from current 7-72 % (depending on `bs`) toward the
   reference 47-50 % at `bs=16`. Multiplicatively compounds with fix #1.

3. **Block-size auto-tuning** — at runtime, monitor first-position
   acceptance and dynamically pick the bs that maximizes
   `avg_commit / (1 + verify_overhead_ratio)`. Trivial after fix #1.

## How to invoke

```bash
./dflash_server \
  /path/to/qwen3moe-target.gguf \
  --draft /home/peppi/models/qwen3-30B-A3B-dflash/Qwen3-Coder-30B-A3B-DFlash.gguf \
  --port 18080 --max-ctx 8192
```

Optional diagnostic knob: `DFLASH_BLOCK_SIZE=4 ./dflash_server ...` to
override the GGUF-encoded block size at runtime (allowed range 2-32;
default reads `dflash.block_size` metadata, typically 16).

Spec decode auto-activates when both `--draft` is provided AND the
sampler is greedy (no presence_penalty / top-k / top-p that would force
the AR fallback). Otherwise the AR path runs.

## Files

- `qwen3moe_backend.cpp` — `do_spec_decode` (~lines 605-1010) + prefill
  feature-ring populate (added 2026-05-26)
- `qwen3moe_dflash_target.cpp` — `DFlashTarget` adapter (9 methods)
- `qwen3moe_verify_graph.cpp` — multi-token forward with intermediate
  layer captures at `[1, 12, 23, 34, 45]`
- `scripts/convert_dflash_30b_a3b_to_gguf.py` — converter for z-lab
  HF safetensors → dflash-compatible GGUF (`arch=qwen35-dflash-draft`)
