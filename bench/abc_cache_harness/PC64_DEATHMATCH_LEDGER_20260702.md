# PC64 Deathmatch Ledger - 2026-07-02

This ledger pins the current evidence to committed source and raw artifacts. The
large result JSON/markdown files are intentionally left outside git unless a
reviewer asks for them.

## Provenance

- Branch: `bench/upstream-pr469-473-plus-468`
- Binary source commit: `e0e8573ac59a43fdb108005ef2bf9082dec3c629`
  - Later commits in this branch update only this ledger; they do not affect the
    recorded `dflash_server` binary.
- dflash binary: `server/build-pr468-int-cuda126/dflash_server`
- dflash sha256: `60b410d695af802ecb391d387220562a37fd3c2005a69a087c0b092bc80e3887`
- Harness claim-scope commit: `a2bd6d60`
- llama binary: `/home/peppi/llama.cpp/build-cuda/bin/llama-server`
- llama sha256: `feedd55326b13fd4156dd0c7d7086fb94201cceeda5ef3eabc43fb26e2adc06b`
- Model: `/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Chat template: `/home/peppi/models/qwen3-coder-chat-template.jinja`
- KV cache: `q4_0/q4_0`
- Max context: `131072`
- Trace rows set `temperature: 0`. The raw artifact provenance still reports
  the harness default temperature field; request bodies are the source of truth.

## Trace Inventory

- Deep tool trace: `bench/abc_cache_harness/traces/deep_tool_structured_38.jsonl`
  - sha256: `304323148614e0e1004aeba118efeb315a932528e962e6af7634cfca36f21af9`
  - 38 turns, every row carries `tools`, `tool_choice`, `expect_tool_call`, and
    `expected_tool_name`.
  - Uniform `max_tokens: 256`; usable for pinned equal-workload speed runs.
- Cap trace: `bench/abc_cache_harness/traces/deep_tool_structured_38_cap2048.jsonl`
  - sha256: `812fb810d70546eabba4b7df641e54e32382f775e81392c4c373de8ae91dcca0`
  - Derived mechanically with:
    `jq -c '.max_tokens = 2048' deep_tool_structured_38.jsonl > deep_tool_structured_38_cap2048.jsonl`
  - Same content as the deep tool trace except `max_tokens`.
  - Used to avoid clipping long natural tool calls.
- Guard-retirement trace: `bench/abc_cache_harness/traces/real_session_long_38.jsonl`
- Local quality probe trace: `bench/abc_cache_harness/traces/charbench_code_tool.jsonl`
- Mixed real-session tool trace: local `/tmp/luce_mixed_candidate_0_fixed_38.jsonl`
  - sha256: `dda519be228be47c2076725c215c467e9d38230ba2538876b366934e17664244`
  - Source transcript: `b324020e-f90c-45f3-8055-55dd5fe723c3.jsonl`
    from `replay_inventory.md`, source sha256
    `481ad5bff31a0e95d11945b08e3fd7e7e8a5c1c3d9723664b39d465f5608b1ce`.
  - Generated with `structure_claude_replay_trace.py --source-kind raw-session
    --turns 38 --max-tokens 256`.
  - 38 turns: 29 expected tool-call rows, 9 non-tool rows, exact `tool_choice`
    on expected rows only. The structurer now adds permissive schemas for any
    observed source-session tool names before writing `tool_choice`, so a trace
    cannot name a missing tool by construction.
  - Cap2048 variant: local `/tmp/luce_mixed_candidate_0_fixed_38_cap2048.jsonl`,
    sha256 `e67a612bce5e97a50689a0fc49d044befe22af48dba1813f639272daf62efed4`.
    Derived mechanically by changing only `max_tokens` from 256 to 2048. This
    is the valid natural tool-behavior trace; the 256-token version can clip
    long XML tool calls before their closing tags.

## Parser Fix

Commit `e0e8573a` fixes the Qwen XML tool-call parser/emitter for real outputs
seen in the deep trace:

- `<function=Agent>...</agent>`
- `<function=Agent>...</agent_info>`

The parser now recognizes named close tags that match the function name
case-insensitively, including suffixes separated by `_`, `-`, or `.`. The SSE
emitter uses the same completion detector, so generation stops once these calls
are complete instead of running to `max_tokens` and suppressing the buffer.

Verification:

- `server/build-pr468-int-cuda126/test_server_unit`: 2130 assertions, 0 failures
- Four-row repro of old failing turns: 4/4 valid after the parser fix
- Full native cap2048 run: 38/38 expected tool calls valid

## Current Evidence

| Workload | Arm | Wall s | Prefill s | Decode s | Out toks | Pin | Tool valid | Fresh pf tok/s | Decode tok/s | Last-8 decode med | Raw artifact |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Natural, cap2048 | `AR_35B_KVF_FORCE` | 177.2 | 59.5 | 95.3 | 12086 | false | 38/38 | 1940.0 | 126.8 | 127.7 | `results/AR_35B_KVF_FORCE_20260702_193234_full_raw.json` |
| Natural, cap2048 | `AR_35B_KVF_FORCE_OPENAI` | 212.0 | 60.9 | 124.5 | 15602 | false | 37/38 | 1895.4 | 125.3 | 124.9 | `results/AR_35B_KVF_FORCE_OPENAI_20260702_193847_full_raw.json` |
| Natural, cap2048 | `AR_LLAMA_35B_SLOTCACHE` | 440.5 | 58.718 | 353.721 | 27766 | false | 20/38 | 2076.3 | 78.5 | 68.8 | `results/AR_LLAMA_35B_SLOTCACHE_20260702_194357_full_raw.json` |
| Pinned 256 | `AR_35B_KVF_FORCE` | 157.9 | 59.6 | 76.6 | 9728 | true | 26/38 | 1936.7 | 127.0 | 125.5 | `results/AR_35B_KVF_FORCE_20260702_195412_full_raw.json` |
| Pinned 256 | `AR_LLAMA_35B_SLOTCACHE` | 195.4 | 58.363 | 121.073 | 9728 | true | 20/38 | 2088.9 | 80.3 | 69.0 | `results/AR_LLAMA_35B_SLOTCACHE_20260702_195722_full_raw.json` |
| Mixed real session, cap256 natural diagnostic | `AR_35B_KVF_FORCE` | 146.0 | 64.5 | 61.7 | 7840 | false | 15/29 clipped, unexpected 2/9 | 1934.7 | 127.1 | 124.1 | `results/AR_35B_KVF_FORCE_20260702_205039_full_raw.json` |
| Mixed real session, cap256 natural diagnostic | `AR_LLAMA_35B_SLOTCACHE` | 168.2 | 64.436 | 89.851 | 7797 | false | 19/29 clipped, unexpected 6/9 | 2068.4 | 86.8 | 68.6 | `results/AR_LLAMA_35B_SLOTCACHE_20260702_205341_full_raw.json` |
| Mixed real session, cap2048 natural | `AR_35B_KVF_FORCE` | 255.8 | 60.2 | 170.7 | 21953 | false | 26/29, unexpected 2/9 | 2072.9 | 128.6 | 128.9 | `results/AR_35B_KVF_FORCE_20260702_211640_full_raw.json` |
| Mixed real session, cap2048 natural | `AR_LLAMA_35B_SLOTCACHE` | 272.0 | 63.918 | 186.381 | 15543 | false | 19/29, unexpected 6/9 | 2085.1 | 83.4 | 68.5 | `results/AR_LLAMA_35B_SLOTCACHE_20260702_212133_full_raw.json` |
| Mixed real session, pinned 256 | `AR_35B_KVF_FORCE` | 161.1 | 62.5 | 75.5 | 9728 | true | 15/29, unexpected 2/9 | 1996.6 | 128.8 | 126.8 | `results/AR_35B_KVF_FORCE_20260702_205707_full_raw.json` |
| Mixed real session, pinned 256 | `AR_LLAMA_35B_SLOTCACHE` | 193.7 | 64.194 | 112.594 | 9728 | true | 19/29, unexpected 6/9 | 2076.2 | 86.4 | 66.8 | `results/AR_LLAMA_35B_SLOTCACHE_20260702_210021_full_raw.json` |
| Guard-off, non-forced cap2048 | `AR_35B_KVF` | 388.7 | 68.3 | 296.2 | 24613 | false | 37/38 | 1671.5 | 83.1 | 71.3 | `results/AR_35B_KVF_20260702_201047_full_raw.json` |
| Local quality smoke | `AR_35B_KVF_FORCE` | 1.4 | 0.6 | 1.0 | 136 | false | 1/1 tool prompt | 710.0 | 136.0 | n/a | `results/AR_35B_KVF_FORCE_20260702_200342_quality_raw.json` |

## Pinned Tool-Stop Contract

The deep tool traces have `expect_tool_call: true` on all 38 turns. Exact
fixed-length pinning and natural tool-stop correctness are therefore separate
claims on this trace: forcing exactly 256 output tokens is useful for decode
and wall equal-work speed, but it is not a decisive tool-valid claim.

Harness commit `a2bd6d60` makes this explicit in future artifacts by reporting:

- `pin_decode_claim_scope`
- `pin_decode_tool_turns`
- `pin_decode_non_tool_turns`
- `pin_decode_tool_stop_conflict`
- `pin_decode_non_tool_ok`

Validation smoke:
`results/AR_35B_KVF_FORCE_20260702_204157_smoke_raw.json`

Result: `pin_decode_ok=true`, `tool_expected_valid=3/3`, and
`pin_decode_claim_scope=speed_only_tool_stop_conflict`.

On the mixed real-session trace, pinned rows are now classified as
`mixed_tool_and_non_tool` instead of `speed_only_tool_stop_conflict`.
Both engines emitted exactly 9728/9728 tokens and `pin_decode_non_tool_ok=true`
on the pinned pair, but expected tool rows are still a tool-stop conflict under
forced length. Treat these rows as equal-output speed evidence, not decisive
tool correctness.

The mixed cap256 natural rows are also diagnostic only for tool correctness.
A six-turn debug repro showed turn 6 generated
`<function=Bash><parameter=command>...` but hit `max_tokens=256` before closing
the XML, so the emitter correctly suppressed the incomplete tool buffer. The
cap2048 mixed rows supersede cap256 for natural tool behavior.

## In-Tree OpenAI Client Bench Smoke

The checked-in `harness/client_test_runner.py bench` suite was run against the
same forced-KVFlash OpenAI-compatible server profile:

```
python3 harness/client_test_runner.py bench \
  --url http://127.0.0.1:19099 \
  --model luce-dflash \
  --suite all \
  --json-out .harness-work/runs/forced_kvflash_openai_bench_20260702.json
```

Summary:

| Suite | OK | Mean wall s | Mean TTFT s | Mean prefill tok/s | Mean output tok/s | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `he` | 10/10 | 1.372 | 0.170 | 816.5 | 138.06 | 10/10 |
| `gsm` | 10/10 | 2.371 | 0.121 | 622.5 | 137.57 | 6/10 |
| `math` | 10/10 | 5.073 | 0.122 | 535.8 | 135.94 | 7/10 |
| `agent` | 6/6 | 3.182 | 0.562 | 2052.3 | 138.11 | n/a |

This is a useful endpoint/streaming/usage/correctness smoke over existing
project prompts. It is not the full charbench threshold gate and should not be
cited as that.

## HumanEval+ Code Quality Smoke

The existing HumanEval+ dataset and grader under `server/eval` /
`server/scripts/quality_humaneval_plus.py` were run against the same forced
KVFlash OpenAI-compatible profile.

Summary artifact:
`.harness-work/runs/humanevalplus_forced_kvf_full_20260702_summary.json`

| Bench | Passed | Total | pass@1 | Request errors | Completion tokens | Generation wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HumanEval+ full | 143 | 164 | 0.872 | 0 | 42392 | 336.373 |

The first-20 smoke passed 20/20 before the full run. This is stronger than the
two-row local charbench smoke for code-generation quality, but it still does
not replace the missing charbench `code_complete` / `tool_call` threshold gate.

## Gate Status

- P1 tool-schema root cause: improved. The old native `0.447` result is stale.
  Native dflash is now 38/38 on the deep cap2048 trace. The OpenAI-compatible
  dflash path is 37/38; the remaining miss is turn 35, a forced
  `AskUserQuestion` row that generated a malformed 2048-token tool-like buffer.
  llama is 20/38 on the same trace.
- P2 equal-workload speed: green as a speed row. On pinned 256, both arms emit
  exactly 9728 tokens. dflash wall is 157.9s vs llama 195.4s, and weighted
  decode is 127.0 vs 80.3 tok/s. This is a 19.2% wall win and 1.58x decode win.
  The mixed real-session pinned row repeats the result with non-tool turns in
  the trace: dflash wall is 161.1s vs llama 193.7s, and weighted decode is
  128.8 vs 86.4 tok/s. This is a 16.8% wall win and 1.49x decode win with
  exactly 9728 output tokens on both arms.
- Pinned tool-valid decisive claim: not green. Forcing exactly 256 output tokens
  distorts natural tool-call stopping. dflash drops to 26/38 tool-valid under
  pinning; llama remains 20/38. Harness commit `a2bd6d60` now labels these
  pinned all-tool runs as `speed_only_tool_stop_conflict` in newly generated
  artifacts.
- Natural tool-valid claim: green for dflash native on this run, but not an
  equal-output workload. Natural cap2048 has dflash 38/38 tool-valid and 126.8
  tok/s decode, while llama has 20/38 and 78.5 tok/s. Wall also favors dflash
  here, but output totals differ: 12086 vs 27766.
- Mixed natural tool behavior: green on the corrected cap2048 trace. Dflash is
  26/29 expected tools with 2/9 unexpected tool calls; llama is 19/29 expected
  tools with 6/9 unexpected tool calls. Dflash also wins natural wall
  (255.8s vs 272.0s) despite producing more output tokens
  (21,953 vs 15,543), and weighted decode is 128.6 vs 83.4 tok/s.
- Guard retirement: `DFLASH_QWEN35_KVPAD_MAX_ROW` / row-88000 guard is absent on
  this branch. A non-forced 38-turn run with `DFLASH_QWEN35_KVPAD_MAX_ROW=0`
  on the deep cap2048 tool trace completed 38/38 with zero crashes and every
  turn recorded. The run crossed the old crash band: turn 23 reached 88,186
  prompt tokens at 79.7 tok/s; turn 38 reached 113,944 prompt tokens at 71.0
  tok/s. There was no hidden half-speed fallback after 88K.
- Quality: the local two-row `charbench_code_tool` smoke passes
  `charbench_valid_rate=1.0`, the in-tree OpenAI client bench smoke above
  completed 36/36 requests with HE 10/10, and HumanEval+ full scored 143/164
  pass@1 with zero request errors. The full external charbench gate with the
  85.2%/53.1% thresholds is still missing here.

## Claim Discipline

Valid claims now:

- On the deep-context tools-on natural trace, dflash native forced KVFlash is
  38/38 tool-valid and decodes 1.62x faster than llama in this run.
- On the deep-context pinned 256 equal-workload trace, dflash forced KVFlash has
  lower wall time than llama and 1.58x weighted decode throughput.
- On the mixed real-session pinned 256 trace, dflash forced KVFlash has lower
  wall time than llama and 1.49x weighted decode throughput with equal output
  tokens.
- On the corrected mixed real-session cap2048 natural trace, dflash has higher
  expected-tool validity, lower unexpected-tool rate, lower wall time, and
  1.54x weighted decode throughput than llama.

Not yet valid:

- No single-row decisive victory claim, because the equal-output run is pinned
  and therefore conflicts with natural tool stopping, while the corrected
  natural tool run has unequal output lengths.
- No default-ship quality claim, because the full charbench quality suite is not
  present in this worktree.

Next required work:

1. Preserve claim discipline: use the mixed cap2048 trace for natural tool
   behavior and the mixed pinned-256 trace for equal-output speed. Do not cite
   the cap256 natural row as a tool-quality result.
2. Bring in or reconstruct the full charbench quality suite and run the forced
   KVFlash arm against the 85.2% code-complete / 53.1% tool-call floor.
3. After those two gates are clean, start the dFlash/spec-decode campaign.
