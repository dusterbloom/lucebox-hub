# Current Main Wall Evidence - 2026-07-06

Scope: natural 38-turn agentic trace, `deep_tool_structured_38_cap2048`,
`--max-ctx 131072`, q4_0 KV, one repeat, seed sent. These rows support
natural wall/correctness claims, not equal-output pinned speed claims.

## Dense Qwen3.6 27B

| Arm | Wall s | Prefill s | Decode s | Weighted decode tok/s | Out tok | Fresh prefill tok | Tools | Fair row | Energy J | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `AR_27B_KVF` | 607.3 | 152.6 | 375.9 | 39.495 | 14846 | 115430 | 38/38 | true | 162954.641 | `results/AR_27B_KVF_20260706_120908_full_raw.json` |
| `AR_27B_KVF` best local same binary | 594.0 | 144.8 | 377.6 | 39.317 | 14846 | 115430 | 38/38 | true | 160606.974 | `results/AR_27B_KVF_20260706_114859_full_raw.json` |
| `AR_LLAMA_27B_SLOTCACHE` | 1535.5 | 153.49 | 1286.607 | 26.399 | 33965 | 121915 | 17/38 | false | 436187.654 | `results/AR_LLAMA_27B_SLOTCACHE_20260706_122832_full_raw.json` |

Dense conclusion: dFlash wins wall-to-wall on the same 38-turn trace even using
the slower current-head row: 607.3s vs 1535.5s. It also wins weighted decode
throughput, tool correctness, and measured energy. The llama row is not a usable
natural-correctness row because tool validity is 17/38.

## MoE Qwen3.6 35B-A3B

| Arm | Wall s | Prefill s | Decode s | Weighted decode tok/s | Out tok | Fresh prefill tok | Tools | Fair row | Energy J | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `AR_35B_KVF_FORCE` current fixed, seeded | 211.4 | 58.5 | 130.0 | 129.462 | 16830 | 115430 | 38/38 | true | 62127.038 | `results/AR_35B_KVF_FORCE_20260706_120151_full_raw.json` |
| `AR_35B_KVF_FORCE` best local, unseeded | 177.2 | 59.5 | 95.3 | 126.821 | 12086 | 115430 | 38/38 | n/a older schema | n/a | `results/AR_35B_KVF_FORCE_20260702_193234_full_raw.json` |
| `AR_LLAMA_35B_SLOTCACHE` | 440.5 | 58.718 | 353.721 | 78.497 | 27766 | 121915 | 20/38 | n/a older schema | n/a | `results/AR_LLAMA_35B_SLOTCACHE_20260702_194357_full_raw.json` |

MoE conclusion: dFlash wins wall-to-wall on the same 38-turn trace against the
available llama.cpp slot-cache baseline. Current fixed row is 211.4s vs 440.5s;
best local row is 177.2s vs 440.5s. It also wins weighted decode throughput and
tool correctness.

The 211.4s vs 177.2s gap is natural-output length, not slower decode. Both rows
use the same trace bytes, model, chat template, ctx, q4_0 KV, `--kvflash-force`,
fresh prefill tokens (115430), and cache hit ratio (0.941). The current seeded
run generated 4744 more output tokens (16830 vs 12086), adding 34.7s decode
time; prefill was actually 1.0s faster and weighted decode was slightly higher
(129.462 vs 126.821 tok/s). The old 177.2s harness did not send request seeds;
the current rerun did (`--send-seed`).

## Caveats

- Natural rows are allowed to produce different output lengths. Use these for
  wall/correctness claims, not equal-output micro-speed claims.
- The 27B dFlash best local row has the same server binary SHA as the current
  row but stale git provenance (`1f2c990d`), so the reproducible headline should
  use `AR_27B_KVF_20260706_120908`.
- The 35B best local and llama baseline are older-schema rows imported from the
  prior deathmatch worktree so the current branch keeps the comparison evidence
  together.
- The 35B best local 177.2s row is not a current-provenance seeded reproduction.
  Treat 211.4s as the conservative current-head seeded headline and 177.2s as
  the best local unseeded wall result until rerun on the WIP branch under an
  explicitly chosen no-seed contract.
