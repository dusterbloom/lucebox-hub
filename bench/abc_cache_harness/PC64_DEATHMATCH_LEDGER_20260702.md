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
| Guard-off, non-forced cap2048 | `AR_35B_KVF` | 388.7 | 68.3 | 296.2 | 24613 | false | 37/38 | 1671.5 | 83.1 | 71.3 | `results/AR_35B_KVF_20260702_201047_full_raw.json` |
| Local quality smoke | `AR_35B_KVF_FORCE` | 1.4 | 0.6 | 1.0 | 136 | false | 1/1 tool prompt | 710.0 | 136.0 | n/a | `results/AR_35B_KVF_FORCE_20260702_200342_quality_raw.json` |

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
- Pinned tool-valid decisive claim: not green. Forcing exactly 256 output tokens
  distorts natural tool-call stopping. dflash drops to 26/38 tool-valid under
  pinning; llama remains 20/38.
- Natural tool-valid claim: green for dflash native on this run, but not an
  equal-output workload. Natural cap2048 has dflash 38/38 tool-valid and 126.8
  tok/s decode, while llama has 20/38 and 78.5 tok/s. Wall also favors dflash
  here, but output totals differ: 12086 vs 27766.
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

Not yet valid:

- No decisive victory claim, because the single run that is pinned/equal-output
  is not tool-valid on dflash (26/38).
- No default-ship quality claim, because the full charbench quality suite is not
  present in this worktree.

Next required work:

1. Decide how the harness should define pinned tool-call workloads. Exact fixed
   decode length and early tool stopping are in tension; the current pinned
   speed row is scientifically useful, but it is not a tool-valid claim.
2. Bring in or reconstruct the full charbench quality suite and run the forced
   KVFlash arm against the 85.2% code-complete / 53.1% tool-call floor.
3. After those two gates are clean, start the dFlash/spec-decode campaign.
