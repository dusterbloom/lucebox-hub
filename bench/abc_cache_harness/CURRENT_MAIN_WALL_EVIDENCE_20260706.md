# Current Main Deathmatch Evidence - 2026-07-06

Scope: natural 38-turn agentic trace, `deep_tool_structured_38_cap2048`,
`--max-ctx 131072`, q4_0 KV, one repeat, seed sent where supported. These
rows support correctness, decode throughput, cache behavior, and natural
wall-time evidence. They do not support equal-output speed ratios because the
arms emitted different token counts and the llama.cpp rows failed many expected
tool calls.

Pinned upstream reference:

- Luce-Org `origin/main`: `b44c872b725061ca16a546c9d8f332784ddf6ae6`
- Pinned llama.cpp submodule: `02cc5286c431273efb7f4177f733aee042020d00`

## Claim Shape

Use these as the maintainer-facing claims:

- Correctness: dFlash produced all expected tool calls on both 27B and 35B
  traces. The llama.cpp slot-cache rows did not.
- Decode: dFlash has higher weighted decode throughput on both dense and MoE.
- Cache: the stale-boundary inline snapshot fix reduces duplicate fresh prefill
  on 35B from 208038 to 115430 tokens in the observed regression pair.
- Energy: on dense 27B, compare joules per successful expected tool call, not
  total joules, because the baseline failed most tool calls.
- Wall: natural wall time is lower for dFlash in the captured rows, but keep it
  secondary because output lengths and tool validity differ.

Avoid using "2.5x wall" as a headline. It is directionally useful context, but
it is not an equal-output benchmark result.

## Dense Qwen3.6 27B

| Arm | Wall s | Prefill s | Decode s | Weighted decode tok/s | Out tok | Fresh prefill tok | Tools | Fair row | Energy J | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `AR_27B_KVF` | 607.3 | 152.6 | 375.9 | 39.495 | 14846 | 115430 | 38/38 | true | 162954.641 | `results/AR_27B_KVF_20260706_120908_full_raw.json` |
| `AR_27B_KVF` best local same binary | 594.0 | 144.8 | 377.6 | 39.317 | 14846 | 115430 | 38/38 | true | 160606.974 | `results/AR_27B_KVF_20260706_114859_full_raw.json` |
| `AR_LLAMA_27B_SLOTCACHE` | 1535.5 | 153.49 | 1286.607 | 26.399 | 33965 | 121915 | 17/38 | false | 436187.654 | `results/AR_LLAMA_27B_SLOTCACHE_20260706_122832_full_raw.json` |

Dense readout:

- Tool correctness: 38/38 vs 17/38.
- Weighted decode: 39.495 tok/s vs 26.399 tok/s.
- Fresh prefill: 115430 vs 121915 tokens.
- Energy per successful expected tool call: 4288.3 J for dFlash
  (`162954.641 / 38`) vs 25658.1 J for llama.cpp (`436187.654 / 17`), a 5.98x
  improvement by that correctness-normalized energy metric.
- Natural wall: 607.3s vs 1535.5s, but this is confounded by the llama.cpp row
  emitting 33965 tokens and only 17 valid expected tool calls.

## MoE Qwen3.6 35B-A3B

| Arm | Wall s | Prefill s | Decode s | Weighted decode tok/s | Out tok | Fresh prefill tok | Tools | Fair row | Energy J | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `AR_35B_KVF_FORCE` current fixed, seeded | 211.4 | 58.5 | 130.0 | 129.462 | 16830 | 115430 | 38/38 | true | 62127.038 | `results/AR_35B_KVF_FORCE_20260706_120151_full_raw.json` |
| `AR_35B_KVF_FORCE` best local, unseeded | 177.2 | 59.5 | 95.3 | 126.821 | 12086 | 115430 | 38/38 | n/a older schema | n/a | `results/AR_35B_KVF_FORCE_20260702_193234_full_raw.json` |
| `AR_LLAMA_35B_SLOTCACHE` | 440.5 | 58.718 | 353.721 | 78.497 | 27766 | 121915 | 20/38 | n/a older schema | n/a | `results/AR_LLAMA_35B_SLOTCACHE_20260702_194357_full_raw.json` |

MoE readout:

- Tool correctness: 38/38 vs 20/38.
- Weighted decode: 129.462 tok/s vs 78.497 tok/s.
- Fresh prefill: 115430 vs 121915 tokens.
- Natural wall: 211.4s vs 440.5s for the current seeded row, and 177.2s vs
  440.5s for the older unseeded best-local row. Keep this secondary because the
  rows are not equal-output and the llama.cpp baseline failed many tool calls.

The 211.4s vs 177.2s gap is natural-output length, not a decode regression.
Both dFlash rows use the same trace bytes, model, chat template, ctx, q4_0 KV,
`--kvflash-force`, fresh prefill tokens (115430), and cache hit ratio (0.941).
The current seeded run generated 4744 more output tokens (16830 vs 12086),
adding 34.7s decode time; prefill was 1.0s faster and weighted decode was
slightly higher (129.462 vs 126.821 tok/s). The old 177.2s harness did not send
request seeds; the current rerun did (`--send-seed`).

## Cache Regression Evidence

The current-main-ish 35B row before the stale-boundary inline snapshot fix was:

| Arm | Wall s | Prefill s | Decode s | Weighted decode tok/s | Out tok | Fresh prefill tok | Mean hit ratio | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `AR_35B_KVF_FORCE` pre-fix | 273.8 | 95.3 | 158.8 | 130.931 | 21097 | 208038 | 0.919 | `results/AR_35B_KVF_FORCE_20260706_095520_full_raw.json` |
| `AR_35B_KVF_FORCE` fixed | 211.4 | 58.5 | 130.0 | 129.462 | 16830 | 115430 | 0.941 | `results/AR_35B_KVF_FORCE_20260706_120151_full_raw.json` |

The important movement is duplicate prefill removal: fresh prefill dropped by
92608 tokens and prefill time dropped by 36.8s while preserving 38/38 expected
tool-call correctness.

## Pinned-Submodule Rebench Status

The clean pinned-submodule rebench branch cannot currently build the full WIP
server. Reproduced with:

```text
cmake --build server/build-pinned-gcc11-cuda126 --target dflash_server test_server_unit -j 4
```

Failure:

```text
server/src/qwen35/qwen35_target_graph.cpp:945:15: error:
'ggml_gated_delta_net_inplace' was not declared in this scope
```

The WIP branch's superproject gitlink still points at llama.cpp `02cc5286`, but
the working submodule contains local dirty ggml changes:

```text
ggml/include/ggml.h
ggml/src/ggml-cuda/gated_delta_net.cu
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml.c
```

Diff stat for those local submodule changes: 119 insertions, 67 deletions.

This means the performance stack is not yet upstream-grade as a clean Luce-Org
main rebench. Either the ggml in-place GDN change must be split and carried as
part of the contribution, or the server-side call path must be gated/removed
before a pinned-main benchmark can be considered reproducible.

## Caveats

- Natural rows are allowed to produce different output lengths. Use them for
  wall/correctness context, not equal-output speed.
- The 27B dFlash best local row has the same server binary SHA as the current
  row but stale git provenance (`1f2c990d`), so the reproducible headline should
  use `AR_27B_KVF_20260706_120908`.
- The 35B best local and llama baseline are older-schema rows imported from the
  prior deathmatch worktree so the current branch keeps the comparison evidence
  together.
- The 35B best local 177.2s row is not a current-provenance seeded
  reproduction. Treat 211.4s as the conservative current-head seeded headline
  and 177.2s as the best local unseeded wall result until rerun under an
  explicitly chosen no-seed contract.
