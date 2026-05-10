# Phase 0 baselines — regression gate for γ>1 MTP work

Date: 2026-05-10 23:50 CEST
Branch: feature/gemma4-support
Binary: `dflash/build/test_gemma4_dflash` (rebuilt 23:48 after removing stray TQ3-DEQ printf in `fattn-chunked.cuh:391-395`)

## Setup

- Model: `/home/peppi/Dev/lucebox-hub/models/gemma-4-31B-it-Q4_K_M.gguf` (Dense 31B)
- MTP head: `/home/peppi/Dev/lucebox-hub/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf`
- KV: `tq3_0/tq3_0`
- Ctx: 4096, `--n-predict 256`, `--temp 0 --ignore-eos`
- Prompt: "Write a brief essay about the impact of speculative decoding on consumer LLM inference. Cover the core idea, why it works, and practical limits." (145 prefill tokens after BOS)

## Results

| Run | Path | Decode tok/s | First tok (ms) | Accept rate | VRAM peak (GB) |
|---|---|---|---|---|---|
| M1_none_tq3_4k | no MTP | **18.92** | 52.96 | — | 19.69 |
| M3_mtp_tq3_4k | γ=1 MTP | **18.64** | 65.78 | **0.66** (169/256) | 20.14 |

## Phase 1 regression checks (same binary, with `--gamma N`)

| Run | Result | Notes |
|---|---|---|
| `--gamma 1` (explicit) | 19.02 tok/s, accept 0.94 on 16 tokens | within noise of M3 baseline; γ=1 path unchanged |
| `--gamma 2` | exits with `fprintf` + `GGML_ABORT` at `test_gemma4_dflash.cpp:2264` | stub fires correctly, points at plan file |

## Notes on May-9 → May-10 binary regression

The May-9 baselines on `matrix-v2/` reported 26.69 / 25.28 tok/s. Today's 18.92 / 18.64 is a 29% gap. Two reasons explain it:
1. Different prompt length: May-9 used 40 pre-tokenized IDs; we use a longer 145-token prefill, slightly more KV traffic per decode step.
2. The May-10 binary path picked up commit `694cea5e1` (FWHT fuse). Even after removing the stray printf in `fattn-chunked.cuh`, other perf-relevant changes may still account for some delta.

The 18.92/18.64 numbers are the **new** gates. γ=1 must stay within ±2% of these as Phases 3+ land.

## How to re-run the gates

```bash
# M1
./dflash/build/test_gemma4_dflash \
  --model /home/peppi/Dev/lucebox-hub/models/gemma-4-31B-it-Q4_K_M.gguf \
  --ctx-size 4096 --n-predict 256 \
  --kv-k tq3_0 --kv-v tq3_0 \
  --temp 0 --ignore-eos \
  --prompt "Write a brief essay about the impact of speculative decoding on consumer LLM inference. Cover the core idea, why it works, and practical limits."

# M3 (γ=1 MTP)
./dflash/build/test_gemma4_dflash \
  --model /home/peppi/Dev/lucebox-hub/models/gemma-4-31B-it-Q4_K_M.gguf \
  --draft-method mtp \
  --mtp /home/peppi/Dev/lucebox-hub/models/gemma4-mtp-31B/gemma-4-31B-it-assistant.Q4_K_M.gguf \
  --ctx-size 4096 --n-predict 256 \
  --kv-k tq3_0 --kv-v tq3_0 \
  --temp 0 --ignore-eos \
  --prompt "Write a brief essay about the impact of speculative decoding on consumer LLM inference. Cover the core idea, why it works, and practical limits."
```
