# Replay inventory for agentic cache/decode benchmarking

Created: 2026-07-02  
Branch context: `bench/upstream-pr469-473-plus-468`

This inventory preserves the replay assets that matter for the current Luce vs
llama.cpp work without blindly committing raw Claude Code transcripts. The
small/direct harness traces below are committed in this directory. The large
Claude session logs are listed by local path, counts, and SHA-256 so they can be
converted later without losing provenance.

## Committed direct harness traces

These files are already in the harness JSONL request format and can be passed to
`bench/abc_cache_harness/replay_harness.py --trace ...`.

| Trace | Turns | Tools rows | max_tokens | Temperature | SHA-256 | Use |
| --- | ---: | ---: | --- | --- | --- | --- |
| `traces/real_session_long_38.jsonl` | 38 | 0 | 256 | 0 | `4d7dffc9cfcbfa03ca6b3b9df0bc33c98f21bd80446499775eaecfa516840a6b` | Pinned equal-output 38-turn replay used for current tools-off Luce vs llama pair. |
| `traces/luce_tool_schema_6_direct.jsonl` | 6 | 6 | 128 | 0 | `45fab3c44bf6951a56e06c8f46854145760582c47adbded1a2bbae05b6dc4f51` | Tool-schema smoke where every turn directly asks for a shell tool call. |
| `traces/luce_tool_6turn_replay.jsonl` | 6 | 6 | 128 | 0 | `2557e260655d9a52cc7deb24bfde0b19507a85cf2019ff94e0bfc359cd7f3bee` | More realistic six-turn coding replay with shell tool schema. |
| `traces/luce_t5_only.jsonl` | 1 | 1 | 128 | 0 | `ea8b3f371ee8327e5c65e4080d30e1240aa6fc85fc625f90f6817140abba24c2` | Single-turn tool-call debug slice from the six-turn replay. |

## Local Claude transcript candidates

These are real local Claude Code JSONL transcripts under
`/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub`. They are not
committed because they may contain private session content and tool outputs.
They are the source pool for building 38/60/120/240+ turn traces with real tool
schemas.

The counts below use the stricter benchmark-relevant definition from
`server/scripts/bench_agent_loop.py`: Claude `type=user` records whose content
is human-readable text and not a command/meta record. A separate inventory agent
also scanned 1,698 JSONL/NDJSON files and found no direct request trace with
both `>=38` rows and tool schemas.

| Local path | User-text turns | `tool_use` refs | `tool_result` refs | Lines | Size KiB | SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/b324020e-f90c-45f3-8055-55dd5fe723c3.jsonl` | 135 | 5598 | 1125 | 3866 | 14693 | `481ad5bff31a0e95d11945b08e3fd7e7e8a5c1c3d9723664b39d465f5608b1ce` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/1fd47871-cfe8-4510-ba38-b0a85bad8c2a.jsonl` | 119 | 4475 | 551 | 5134 | 14709 | `d4fcbf40f8f1073b70b202f1ae87de4c9267599467bddc41582658322abf6e43` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/81da99aa-175d-49f1-a724-bf543c187b18.jsonl` | 119 | 3731 | 487 | 4209 | 10843 | `f582bc057f1bb021c0e3c1ba03ff6988225b2523788faf9d0a4bd54bb673991c` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/299d9975-a2f5-4260-afaf-61f405179944.jsonl` | 106 | 2506 | 363 | 3119 | 4804 | `6dc069b4b017e383d2bb2a93689c0627d8b5261177d90cc39d2295a81c7f3efc` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/2ac82d25-daa9-4443-84f5-f8595b098f0a.jsonl` | 91 | 3748 | 560 | 3596 | 9504 | `8f1096af3145f4f7e0f122c4657667cae52d52622841bd5d7e241d39f7261791` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/baa98be5-46f7-4c1f-9b3c-21b288b9cc6d.jsonl` | 81 | 16741 | 4160 | 4156 | 25774 | `75b401428b95b23b89b84d6e10a435bb72ec556352765829ea8922163d374bc2` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/d4e962f3-dc87-4cbb-8200-bdb664ec1dca.jsonl` | 79 | 8984 | 1324 | 8653 | 26723 | `e7ce2a6153c447d3a262f0de8f13e370a673e6046d78040561ac2adec5920733` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/4bcca13f-421a-4cbc-8bda-eac7706b2055.jsonl` | 73 | 15871 | 4652 | 2079 | 23107 | `2bb3250ac71b7556b946f97996c411e6caa7978388f3e238e2ea770694e340ba` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/ad923003-9398-4d1c-b04a-a6eed3187171.jsonl` | 72 | 4412 | 768 | 4448 | 9182 | `5a5fc1e669fbd9bca90d0241419697e74f22e9c03879ca1c6e5b6e3ab1af276a` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/73b79311-df9b-45b6-be1a-1ce02a963dcb.jsonl` | 56 | 2804 | 570 | 1396 | 6145 | `3bf1898373f2d550ae1cb4d66de6456339471bde9ce5dbc4a4d895ad334cb1b3` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/621487a4-18f1-4186-9f3b-a2fc90133ff9.jsonl` | 55 | 3110 | 406 | 3357 | 6634 | `60d10b8ca5733cfaa62a9eaa36daede0d61f9dff93c7ee1e22c612bf5f54a7ac` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/61eaa5f3-0278-4efe-b2ba-4bf29a213c3c.jsonl` | 49 | 1508 | 259 | 1686 | 3194 | `55834c1f96d72fac05f702569b15377e59cb0ed0f1198b5bf6e57a0cf776a38e` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/7384ec0a-82d3-4dac-8386-a24e6c5dfcef.jsonl` | 48 | 1479 | 187 | 1480 | 3362 | `891049ce3dcfb5d70510871093cebc9060fe8dcdef0ab6633cc9d5806e3ff394` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/6fbd77f5-3c5a-4d09-8859-8a1eb6735de5.jsonl` | 44 | 3894 | 629 | 2777 | 7103 | `8e766dcbaa35bb7b8b102f1556417720ab2811cca82492d23c72907b1da4a386` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/9cfbb120-0481-4aaf-96b2-1a3826671cec.jsonl` | 42 | 4360 | 645 | 4016 | 8767 | `91094cd9fb9a1466488007a557c455ce7d23414282659c088c6bcc9ea312dac9` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/3a57db4c-be28-42ca-8cc9-a1202f26ef8b.jsonl` | 41 | 1059 | 123 | 1131 | 4236 | `6b2afd06cf4d98ece083299df0ad2f1f44e146f18c7ab3bc321cb6d943a4f8c8` |
| `/home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/35a490f4-1666-4fb3-b04a-09df6db513a9.jsonl` | 38 | 7066 | 1716 | 1614 | 10331 | `d6e9ba46ee0658e1a1cce8bc65f4eca8f76aa8eed0637f4213e4ae81f0009634` |

## Replay/export notes

- `replay_harness.py` posts Luce arms as Anthropic-style `/v1/messages`
  requests and posts llama arms as OpenAI-compatible `/v1/chat/completions`
  requests, converting Anthropic tool schemas to OpenAI function tools for the
  llama arm.
- The server also has OpenAI-compatible chat handling, and
  `server/scripts/bench_agent_loop.py` already demonstrates replaying real
  Claude user turns through `/v1/chat/completions`.
- The exporter for long real sessions can therefore target either shape:
  Anthropic-style JSONL as the canonical harness trace, or OpenAI-compatible
  chat JSONL for direct compat testing. For the deathmatch, keep one canonical
  trace and apply identical per-turn `max_tokens`, temperature, seed, model
  path, KV type, and cache checks to both engines.
- Tool-call validity must come from structured `tool_use` / `tool_calls`, not
  text parsing. The six-turn tool traces are smoke tests only; decisive claims
  need a 38+ turn trace with real tool schemas and nonzero valid tool calls on
  both arms.
- Raw Claude transcripts should not be pushed unless explicitly reviewed for
  privacy. This manifest keeps enough provenance to find and convert them
  later.
