# Q8_0 MTP heads recover ~8pp accept rate

## Finding

Switching MTP head GGUF from Q4_K_M to Q8_0 (with target backbone unchanged at Q4_K_M) raises chain accept rate from 0.73 to 0.81 on Qwen3.6-27B, with no measurable throughput cost.

## Bench data (n_sample=5 per suite, Qwen3.6-27B-MTP-Q4_K_M target, RTX 3090)

| Config | Chain HE | Chain GSM | Chain MATH | Chain Agent | Chain accept | Tree HE | Tree GSM | Tree MATH | Tree Agent | Tree accept |
|---|---|---|---|---|---|---|---|---|---|---|
| Q4 target + Q4 MTP | 61.1 | 55.0 | 53.7 | 54.5 | 0.73 | 39.0 | 37.1 | 36.0 | 42.5 | 0.23-0.31 |
| Q4 target + Q8 MTP | 60.4 | 53.9 | 53.9 | 53.6 | 0.81 | 39.4 | 37.5 | 37.0 | 42.4 | 0.32-0.48 |

Tokens per second within ±2% across all suites; the win is in accept rate.

## Mechanism

The MTP head's logits flow through the target's shared LM head matrix (`qwen36_mtp.cpp:1289`, "shared LM head via target's project_hidden_to_logits"). With a Q4 target the LM head matrix stays Q4 regardless of the MTP file precision. However, the MTP head's TRMBlock weights (attn_q/k/v/o, ffn_gate/up/down, norms) are loaded from the `--mtp-gguf` file. Q8_0 weights produce a higher-fidelity `x_normed` (the input to the LM head projection), which produces more accurate logits even through a Q4 LM head matrix.

## Recommended deployment

Use the MTP-only GGUF from `havenoammo/Qwen3.6-27B-MTP-UD-GGUF` (457 MB, Q8_0). Pair with the standard Q4_K_M target:

```
--target  /path/to/Qwen3.6-27B-MTP-Q4_K_M.gguf
--mtp-gguf /path/to/27B_MTP.gguf
```

The 457 MB overhead is trivial compared to the 17 GB target. No throughput regression, +8pp chain accept rate.

## What this does NOT do

Tree mode at B=2 K=2 still loses to chain on chain-friendly workloads (Tree 0.39 accept vs Chain 0.81 — gap narrowed from ~0.45 to ~0.42 by the Q8 swap, but tree still has higher per-iter verify cost). The tree substrate (selector, runner, telemetry) remains dormant default-off and awaits a workload or model where tree's diversity bet pays off.

## Open questions

- Does Q8 target backbone + Q8 MTP heads improve further? Not tested in this session; would require ~28 GB Q8 target download.
- Does the accept rate gain hold on longer-context workloads (8K+ prompts)? Tested up to 24K in the agent suite; held.

## Source experiments

- `feat/mtp-adaptive` branch — selector + telemetry substrate.
- Bench logs (transient, not committed): `/tmp/f16_chain_bench.json`, `/tmp/f16_tree_bench.json` from the Q8 hybrid run.
