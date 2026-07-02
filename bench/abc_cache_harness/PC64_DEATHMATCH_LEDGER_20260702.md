# PC64 Deathmatch Ledger - 2026-07-02

This ledger pins the current evidence to committed source and raw artifacts. The
large result JSON/markdown files are intentionally left outside git unless a
reviewer asks for them.

## Provenance

- Branch: `bench/upstream-pr469-473-plus-468`
- Commit: `bd981d81f905abd84931cc943ff6dca2529cae15`
- dflash binary: `server/build-pr468-int-cuda126/dflash_server`
- dflash sha256: `945062af480acbd2d2bb4a5de32b7f132ef5f7c2d5540b6e79585e3bee9c87e8`
- llama binary: `/home/peppi/llama.cpp/build-cuda/bin/llama-server`
- llama sha256: `feedd55326b13fd4156dd0c7d7086fb94201cceeda5ef3eabc43fb26e2adc06b`
- Model: `/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Chat template: `/home/peppi/models/qwen3-coder-chat-template.jinja`
- KV cache: `q4_0/q4_0`
- Max context: `131072`

## Trace Inventory

- Deep tool trace: `bench/abc_cache_harness/traces/deep_tool_structured_38.jsonl`
  - sha256: `304323148614e0e1004aeba118efeb315a932528e962e6af7634cfca36f21af9`
  - 38 turns, every row carries `tools`, `tool_choice`, `expect_tool_call`, and
    `expected_tool_name`.
- Cap trace used for faithful tool-stop runs:
  `bench/abc_cache_harness/traces/deep_tool_structured_38_cap2048.jsonl`
  - sha256: `812fb810d70546eabba4b7df641e54e32382f775e81392c4c373de8ae91dcca0`
  - Derived mechanically with:
    `jq -c '.max_tokens = 2048' deep_tool_structured_38.jsonl > deep_tool_structured_38_cap2048.jsonl`
  - Same content as the deep tool trace except `max_tokens`.
- Guard-retirement trace: `bench/abc_cache_harness/traces/real_session_long_38.jsonl`
- Local quality probe trace: `bench/abc_cache_harness/traces/charbench_code_tool.jsonl`

## Current Evidence

| Run | Trace | Wall s | Prefill s | Decode s | Out toks | Tool valid | Fresh pf tok/s | Decode tok/s | Last-8 decode med | Raw artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `AR_35B_KVF_FORCE` | `deep_tool_structured_38_cap2048` | 212.8 | 55.1 | 124.3 | 16787 | 34/38 | 2094.9 | 135.1 | 135.3 | `results/AR_35B_KVF_FORCE_20260702_104011_full_raw.json` |
| `AR_35B_KVF_FORCE_OPENAI` | `deep_tool_structured_38_cap2048` | 214.7 | 55.0 | 130.4 | 17621 | 33/38 | 2098.7 | 135.1 | 135.4 | `results/AR_35B_KVF_FORCE_OPENAI_20260702_105240_full_raw.json` |
| `AR_LLAMA_35B_SLOTCACHE` | `deep_tool_structured_38_cap2048` | 430.2 | 55.326 | 330.702 | 27766 | 20/38 | 2203.6 | 84.0 | 73.1 | `results/AR_LLAMA_35B_SLOTCACHE_20260702_104448_full_raw.json` |
| `AR_35B_KVF` guard-off env | `real_session_long_38` | 100.4 | 48.9 | 37.3 | 3518 | n/a | 1826.5 | 94.3 | 83.7 | `results/AR_35B_KVF_20260702_105844_full_raw.json` |
| `AR_35B_KVF_FORCE` quality probe | `charbench_code_tool` | 1.1 | 0.5 | 1.0 | 136 | 1/1 tool prompt | 852.0 | 136.0 | n/a | `results/AR_35B_KVF_FORCE_20260702_105807_quality_raw.json` |

Failed expected tool turns on the cap trace:

- dflash native: `4:Agent:42`, `6:Agent:1229`, `7:Agent:39`, `22:Agent:2048`
- dflash OpenAI endpoint: `4:Agent:42`, `6:Agent:1229`, `7:Agent:91`,
  `22:Agent:665`, `25:Agent:2048`
- llama slot-cache: `2:Workflow:144`, `4:Agent:1283`, `6:Agent:190`,
  `7:Agent:636`, `9:TaskCreate:1727`, `10:TaskCreate:124`,
  `16:Agent:179`, `20:TaskUpdate:384`, `21:Agent:176`, `22:Agent:322`,
  `23:Bash:2048`, `24:Bash:1767`, `27:Agent:99`, `29:TaskUpdate:58`,
  `33:Agent:246`, `34:Bash:253`, `35:AskUserQuestion:63`,
  `36:TaskUpdate:1528`

## Gate Status

- P1 tool-schema root cause: native dflash and dflash OpenAI endpoint are close
  to each other on the deep tool trace, so the remaining failures are not mainly
  a native endpoint wrapper issue. The earlier 0.447 native result is stale after
  `bd981d81` stopped decode on complete tool buffers and consumed AR tool hints.
- P2 canonical deep-context tool trace: present and exercised. The cap2048 run is
  faithful to long tool arguments but is not pinned equal decode work, so it is
  not a decisive victory claim.
- Guard retirement: `DFLASH_QWEN35_KVPAD_MAX_ROW` / row-88000 guard is absent on
  this branch. A non-forced 38-turn run with `DFLASH_QWEN35_KVPAD_MAX_ROW=0`
  completed 38/38, no crash, turn 38 at 89K context decoded at 82.3 tok/s.
- Quality: local `charbench_code_tool` probe passed, including the single tool
  prompt. This is a small local gate, not a full external charbench suite.
- Real-client Open WebUI tools: both client modes are green after fixing the
  harness request/validation shape.
  - Native mode: `rc=0`, non-streaming request, returned OpenAI `tool_calls`
    for `get_lucebox_harness_marker`.
    Artifact: `.harness-work/runs/20260702-openwebui-tools-native2/`.
  - Default execution mode: `rc=0`, streaming request, Open WebUI executed the
    tool and wrote `OPENWEBUI_TOOL_OK` to `openwebui-tool-exec.log`; server log
    shows the initial tool-call turn and the post-tool follow-up turn.
    Artifact: `.harness-work/runs/20260702-openwebui-tools-default-stream2/`.
  - Harness note: Open WebUI v0.10.2 executes tools in its streaming chat
    middleware. Non-streaming requests can validate returned native tool calls,
    but do not run the tool execution loop.

## Claim Discipline

Current fair claim: on the deep-context tool-schema cap2048 trace, dflash forced
KVFlash has much higher decode throughput than llama and better tool validity in
this one run. It is not yet a decisive deathmatch claim because output lengths
differ, the run is temp 0.7/N=1, and dflash still has 4/38 invalid expected tool
calls.

The `harness/benchmarks` README covers small generation/correctness/speed
checks. The `harness/clients` README covers real client protocol paths,
including OpenAI Chat Completions tools, OpenAI Responses/Codex, Anthropic
Messages/Claude Code, and Open WebUI tool execution. The replay harness remains
the model-output deathmatch gate; `harness/clients/run_openwebui_tools.sh` is
the separate real-client tool execution gate.
