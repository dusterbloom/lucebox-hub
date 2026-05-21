# 1. CONTROL LOOP

Current pFlash compression already maps `keep_ratio` into a chunk budget with `n_keep = max(1, n_chunks * keep_ratio)` in both the qwen35 scoring path and the skip path ([qwen3_drafter.cpp:477](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/qwen3/qwen3_drafter.cpp#L477)) ([qwen3_drafter.cpp:840](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/qwen3/qwen3_drafter.cpp#L840)). The sketch is directionally right, but `last 3 turns < 0.75` is too blunt as the actual decision rule. The controller should use a token-weighted EMA of per-turn acceptance, with the three-turn window only as a minimum sample gate.

Recommendation: keep the deadband `0.75-0.85`, clamp `keep_ratio` to `[0.025, 0.20]`, and move in `0.005` steps by default. Use `0.01` only when the EMA is far outside the band or after two consecutive misses. Warm up for the first two completed turns, or until the session has at least a few hundred proposed tokens, before changing anything. That reduces oscillation without making the controller so sluggish that it cannot recover from a bad starting point. If MTP is off, freeze the adaptive loop rather than inventing a fake reward. If the prompt body size jumps sharply between turns, reset the EMA or treat it as a new sub-session.

# 2. SESSION STATE

Put the map on `HttpServer`, not `Qwen3Backend`. The server already owns request parsing, job dispatch, and the single worker thread that processes jobs sequentially ([http_server.h:84](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.h#L84)) ([http_server.h:145](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.h#L145)) ([http_server.cpp:505](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L505)). That makes the adaptation state an HTTP concern, not a model concern.

Use explicit `extra_body.session_id` as the key. It is stable across turns, client-controlled, and already matches the existing `extra_body` pattern used for `pflash_mode` overrides ([http_server.cpp:424](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L424)). Do not key by connection, cookies, or a hash of `system+first-user-message`: those all break on keep-alive reuse, browser assumptions, or legitimate prompt edits. The state map can therefore be `std::unordered_map<std::string, AdaptiveKeepRatioState>`, with no locking required in the current single-worker design. TTL should be short and boring: expire after 30 minutes of inactivity, or when the map exceeds a fixed cap. Keep it in-memory only; persistence adds versioning, privacy, and cross-release compatibility problems for a control loop that can safely cold-start.

# 3. SIGNAL SOURCE

`GenerateResult` does not currently carry accept telemetry; it only has `ok`, `error`, `tokens`, `prefill_s`, and `decode_s` ([model_backend.h:68](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/common/model_backend.h#L68)). So the acceptance signal must be plumbed if the worker is going to read it structurally. Today, acceptance is only emitted to stderr in two backend paths: the DFlash chain path logs `accepted=%d/%d (%.1f%%)` ([qwen35_backend.cpp:932](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/qwen35/qwen35_backend.cpp#L932)), and the MTP path logs `accept_rate=%.2f` ([qwen35_backend.cpp:1225](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/qwen35/qwen35_backend.cpp#L1225)).

For control-loop use, add a small telemetry struct or scalar fields to `GenerateResult` and stop scraping stderr in production. When MTP is off, there is no true accept rate, so use fallbacks only as guardrails: prefix-cache hit rate from `[pc] lookup hit` and `[pc] full-cache hit` logs ([prefix_cache.cpp:263](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/prefix_cache.cpp#L263)) ([prefix_cache.cpp:371](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/prefix_cache.cpp#L371)), anchor density from the skip-drafter log (`forced incl. anchors`) ([qwen3_drafter.cpp:798](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/qwen3/qwen3_drafter.cpp#L798)), and throughput from `decode_s` or tokens/sec. Those signals do not replace accept rate; they only tell us when to freeze or slow the controller.

# 4. INTEGRATION POINTS

The read hook belongs in `HttpServer::worker_loop()` immediately after `const auto & req = job->req` and before the pFlash compression block starts ([http_server.cpp:510](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L510)) ([http_server.cpp:546](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L546)). That is where the worker can resolve `session_id`, fetch the current session state, and substitute the session’s `keep_ratio` into `ModelBackend::CompressRequest`.

The update hook belongs right after `backend_.generate(...)` or `restore_and_generate(...)` returns, before cache confirmation and response serialization ([http_server.cpp:675](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L675)) ([http_server.cpp:682](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L682)). `ParsedRequest` does not need `keep_ratio_override` for v1; the session map owns the adaptive value. It does need a `session_id` field so the worker can key the state. The typed compress request already has `keep_ratio` and `pflash_mode` ([model_backend.h:118](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/common/model_backend.h#L118)), and the server already fills `keep_ratio` today from config ([http_server.cpp:580](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L580)). So the first change is source-of-value, not API shape.

# 5. OBSERVABILITY

Use one human-readable stderr line per turn, plus one JSONL row per turn for offline analysis. The stderr line should be compact enough to scan in live logs:

`[pflash-adapt] sid=<id> turn=<n> keep=<before>-><after> acc=<ratio> ema=<ratio> src=<mtp|fallback|frozen> prompt=<raw_tokens> eff=<effective_tokens>`

The JSONL schema should be append-only and lossless enough for A/B analysis, but still avoid raw prompt text. A single row should include `ts_ns`, `session_id`, `turn_index`, `keep_before`, `keep_after`, `accept_rate`, `ema_accept_rate`, `accepted_tokens`, `proposed_tokens`, `emitted_tokens`, `raw_prompt_tokens`, `effective_prompt_tokens`, `pflash_mode`, `mtp_enabled`, `prefix_cache_hit`, `full_cache_hit`, `decode_s`, `prefill_s`, and `decision_reason`. If the request body is large, bucket its size rather than logging content. The point is to compare treatment vs baseline at the session level without reconstructing prompts or token streams.

# 6. TEST PLAN

Unit tests should hit the controller as a pure state machine first. Feed synthetic accept-rate sequences and assert the expected keep-ratio path: monotone increase on repeated low acceptance, monotone decrease on repeated high acceptance, no change inside the deadband, clamp at `[0.025, 0.20]`, warm-up freeze on the first two turns, and reset/freeze on large prompt-size jumps. Add one test for MTP-disabled turns to prove the controller does not mutate on missing acceptance telemetry.

The integration test should run a 5-turn session twice: once with fixed `keep_ratio=0.10`, once with the adaptive policy. A fake backend can return scripted `GenerateResult` telemetry and record the `CompressRequest.keep_ratio` passed in by the worker. The go/no-go gate is session-level, not turn-level: treatment must Pareto-dominate baseline on at least 70% of sessions, and the build should fail if the bandit hits either clamp on more than 30% of sessions after warm-up. That catches both over-aggressive compression and an adaptation loop that is stuck at the boundaries.

# 7. OPEN QUESTIONS

- Should the update happen after the model finishes streaming, or immediately after `GenerateResult` comes back and before cache confirmation?
- Should anchor parameters stay fixed for now, with adaptive anchor tuning deferred until we know the keep-ratio loop is stable?
- Should we split by `session x task-type` later, once we have a reliable task classifier, or keep a single bandit per session for v1?
- What is the cold-start default: always the global server default, or a prompt-length bucket table that seeds the first session turn?

# 8. HONEST CRITIQUE

The sketch is useful, but it is not robust enough as written. The biggest problem is using `last 3 turns` as the main statistic. Turn lengths vary, proposal counts vary, and a short noisy turn can dominate the decision. An EMA weighted by proposed tokens is a better fit. The second issue is step size: `+/-0.01` is not absurd, but on a `[0.025, 0.20]` range it is large enough to overshoot on short sessions if you update every turn. `0.005` with hysteresis is safer, and `0.01` can still be used when the controller is far off target.

The `0.75-0.85` band is plausible, but it is still a proxy. It should be treated as a default for the MTP-backed path, not a universal truth across every prompt family. Thompson sampling or UCB are probably the wrong tools here: the action space is one scalar, the environment is non-stationary, and we care more about stable session-local adaptation than exploration. The real risk is confounding from prompt-shape changes. Mitigate that with warm-up, prompt-size bucketing, and resets on large body jumps, plus explicit freezing when the reward signal is missing.

# 9. FIRST PR TO LAND

Minimum viable first PR: one pure controller unit plus one server hook path, under 200 LOC total. Add `AdaptiveKeepRatioState` and `step_adaptive_keep_ratio(...)` in the HTTP layer, add `session_id` to `ParsedRequest`, add `std::unordered_map<std::string, AdaptiveKeepRatioState>` to `HttpServer`, and wire read/update hooks into `worker_loop()` at the two existing decision points ([http_server.cpp:510](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L510)) ([http_server.cpp:675](/home/peppi/Dev/lucebox-hub/.claude/worktrees/pflash-auto/dflash/src/server/http_server.cpp#L675)). Add exactly two log lines: one on read, one on update, both with `sid`, `turn`, `keep_before`, `keep_after`, and `accept_rate`.

The test should be a tiny fake-backend integration: one session ID, two turns, scripted acceptance on turn 1, and an assertion that turn 2 compresses with the updated keep ratio. Do not touch drafter code, do not add persistence, do not add task-type bucketing, and do not chase anchor tuning in this PR. The point of the first change is only to prove the closed loop works end-to-end on one session and can be observed in logs.
