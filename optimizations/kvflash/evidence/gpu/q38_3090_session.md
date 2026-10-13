# Qwen3.8-27B GPU checks (RTX 3090, WSL) - 2026-10-13

Model: /mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf (sha256 verified)
Binary: /mnt/kimi-k3/build/kvflash-am-cuda/luce_server  md5 da89992e458d1bc9c4989578a9ff7613 (cmake --build . up to date at worktree HEAD b966c2a9, includes d2da55ed)
Worktree branch worktree-kvflash-am. Temperature 0, seed 1, default KV type on all arms, same --max-ctx within each comparison.
All artifacts under /mnt/kimi-k3/artifacts/kvflash-am-gpu/q38/ ; scripts in scripts/ (copies of kvlib.py, gate4_live.py, growth_gate2.py, lvg.py pointed at the 3.8 model, plus new toolloop.py and sse_nl.py).
Env for live arms: KVFLASH_STATE_LADDER=1 KVFLASH_LIVE_RESTORE=1 (task 3 also KVFLASH_GEN_CKPT=1, LUCE_LOG_TOOL_CALLS=1). Pool flag --kvflash 8192.

## Task 1 - 10-turn identity gate: PASS (10/10 identical)
Commands (cwd scripts/): `KV_ROOT=.../q38/t1 python3 -I gate4_live.py A` (ladder only, cold re-prefill reference) then `... gate4_live.py B` (ladder + live restore, replays A's prompts), `python3 -I t1/cmp.py`. --max-ctx 131072, --kvflash 8192.
| arm | median prefill s | median decode tok/s | median wall s |
|---|---|---|---|
| A cold re-prefill | 9.45 | 57.4 | 10.89 |
| B live restore | 0.50 | 57.7 | 1.18 |

Per turn (prompt tokens, A prefill s, B prefill s, B decode tok/s, restored_prefix_tokens L / replay): see t1/cmp.out
t1: 11606 8.5/8.5 (no prefix) | t2: 11958 8.6/0.6 L=11602 | t3 12404 9.0/0.5 L=11954 | t4 12613 9.3/0.3 L=12400 | t5 12898 9.4/0.5 L=12609 | t6 13265 9.5/0.4 L=12894 | t7 13711 10.2/0.8 L=13261 | t8 13921 10.2/0.5 L=13707 | t9 14205 10.3/0.3 L=13917 | t10 14597 10.8/0.7 L=14201. Decode A 55.7-58.4, B 56.4-58.5.
Output identical on all 10 turns.

## Task 2 - pool crossing (growth_gate2.py): PASS
Command: `python3 -I scripts/growth_gate2.py` (A flags off, B ladder+live restore, C cold = fresh server per recorded prompt, flags off). Sizes 3000..14000, pool 8192, max-ctx 131072. Output: t2/growth.out, t2/result.json.
| turn | prompt tok | new tok | A prefill s | B prefill s | B tokens prefilled | C cold prefill s |
|---|---|---|---|---|---|---|
| 0 | 2754 | 2754 | 2.6 | 2.7 | 2754 | 2.6 |
| 1 | 4123 | 1369 | 1.2 | 1.3 | 1375 | 3.6 |
| 2 | 5500 | 1377 | 1.2 | 1.2 | 1383 | 4.5 |
| 3 | 6877 | 1377 | 1.2 | 1.2 | 1383 | 5.6 |
| 4 (crossing, 8255) | 8255 | 1378 | 5.7 | 1.1 | 1384 | 6.5 |
| 5 | 9633 | 1378 | 6.8 | 1.2 | 1441 | 7.3 |
| 6 | 11010 | 1377 | 8.1 | 1.4 | 1794 | 8.5 |
| 7 | 12387 | 1377 | 9.0 | 1.3 | 1635 | 9.7 |
Median B prefill: before crossing (turns 1-3) 1.2 s, crossing+after (turns 4-7) 1.25 s. Median A: after crossing 7.5 s, C cold 7.9 s (turns 4-7).
Crossing turn: B prefilled 1384 tokens for 1378 new (gate bound 512+new, pass); A (ladder off) re-prefilled and paid 5.7 s. Post-crossing B prefilled 1441/1794/1635 (<= new+512 bound). Identical A==C and B==C on all 8 turns; the question asks for Record 20 of part 0 (early context) and all answers matched cold (26). Log lines: "adopt dense: 6871 rows", "resume: live cut 8192/9216/10752 beats server cut 6871". Decode not measured in this harness (non-stream, 64 tokens).

## Task 3 - think-marker newline fix (d2da55ed), thinking on: PASS for echoed-reasoning path
Commands: `python3 -I scripts/lvg.py cpp tool_on_echo,tool_on_noecho,tool4_on_echo,plain_on_echo`, same with `jinja` (lvg.py adds `--chat-template-file tmpl/qwen36_template.jinja`), then `t3/an.py`; SSE check `python3 -I scripts/sse_nl.py cpp|jinja`. 6 turns per scenario, F1=9000 filler, pool 8192, max-ctx 131072.
| path | turns after a tool-call turn with L>=G | turns after a final/stop turn with L>=G |
|---|---|---|
| C++ renderer (all echo/noecho tool scenarios) | 8/8 | 0/12 (plain/tool) |
| jinja echo (reasoning replayed) | 6/6 | 0/8 |
| jinja noecho (client drops reasoning) | 0/3 | 0/3 |
Interpretation: every tool-call turn on the C++ renderer and on jinja-with-reasoning-echo has L==G (the extra newline is gone; before the fix these were first-diff on 1358 vs 271). L<G after final-answer turns is the known "think block dropped" template behaviour (the template strips <think> of earlier turns, history starts 248068), unrelated to newline. jinja noecho cannot match because the client does not send the reasoning back.
SSE newline check (t3/sse_cpp.json, t3/sse_jinja.json): for a tool-call turn, a plain turn and a post-tool turn, streamed reasoning, content and tool_calls are byte-equal to the non-stream response on both paths; content has 0 leading newlines; reasoning ends with exactly one model-generated "\n". No newline gained or lost.
Per-turn decode (from perf lines): ~38-44 tok/s with thinking+tools at ~9K ctx (tool turns short, so noisy); full lines in t3/*_*.json.

## Task 4 - tool-call format past the pool: no degradation observed (0 malformed), pin arm NOT AVAILABLE
Command: `TL_PAD=1000 python3 -I scripts/toolloop.py pool|dense t4b` (output t4b/). Thinking off, tools get_weather/add, tool results padded ~1000 tokens, loop to ~16.7K prompt tokens (34 turns). Both arms --max-ctx 20480. pool = --kvflash 8192 + ladder + live restore + gen ckpt; dense = no --kvflash, no KVFLASH env. Counting from the response tool_calls (parsed OK + args dict) and leaked "<tool_call"/"<function=" text in content.
| arm | turns | well-formed | malformed | median prefill s | median decode tok/s | median wall s |
|---|---|---|---|---|---|---|
| pool 8192, no pin | 34 | 17 | 0 | 0.55 | 48.2 | 1.12 |
| dense control | 34 | 17 | 0 | 0.80 | 46.4 | 1.37 |
By band (pool, well/malformed): <8192 tok 9/0, 8192-12000 4/0, >=12000 4/0 (max 16728). Dense: 9/0, 3/0, 5/0.
Earlier run with 600-token pads (t4/, stopped at 12.1K): pool 19/0, dense 20/0.
Pin arm: not run. The only pin is KVFlashPager::pin_range() in server/src/common/kvflash_pager.h; it has no caller in server/src (only the unit tests test_kvflash_pager_truncate/serde), and no env var or HTTP field exposes it (KVFLASH_* env: STATE_LADDER, LIVE_RESTORE, GEN_CKPT, POLICY, TAU, MAX_POOL, ...; "pin_end" is the prefix-cache pin, unrelated). Pager keeps sink_chunks=1 (64 tokens) fixed. Not writing server code, per instruction. Caveat: this loop did not reproduce the old Qwen3.6 degradation at all, so a pin would have nothing to fix here; the earlier effect may need the old harness or thinking-on tool turns.

## Task 5 - Qwen3.8 drafter: NONE
Searched /home/peppi/models and /mnt/kimi-k3/home-models (fd for qwen3.8/qwen38): only qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf. The dflash/draft ggml models present are for 3.5/3.6 (qwen3.6-27b-dflash, qwen3.5-27b-dflash, qwen3.5-0.8b-draft); not run (different family, parity void). Spec-decode variant of task 1 skipped.

## Notes
- lvg.py wait_gpu() uses `pgrep -f luce_server`; any concurrent shell whose command line contains that string blocks it for 300 s polls. Not a server issue.
- Nothing written to /tmp by my scripts (TMPDIR=/mnt/kimi-k3/build/tmp). Nothing committed.
