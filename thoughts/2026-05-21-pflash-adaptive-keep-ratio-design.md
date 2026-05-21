# Per-session adaptive `keep_ratio` for PFlash

PFlash already has the right actuation point: `keep_ratio` reaches the skip-drafter compressor and directly controls how many chunks survive (`qwen3_drafter.cpp:746` and `qwen3_drafter.cpp:772`). What is missing is a session-scoped feedback loop that reads decode acceptance, updates one scalar, and applies that scalar on the next turn. This design keeps the controller in the HTTP layer, not the backend, because the HTTP server already owns request parsing and the pflash gate (`http_server.cpp:238` and `http_server.cpp:505`), while the backend should stay reusable and session-agnostic (`model_backend.h:68` and `qwen35_backend.h:91`).

## 1. Control loop

Use a deterministic hysteretic controller, not a stochastic policy. Keep a 3-sample sliding window over valid MTP turns, but compute it from raw counts (`accepted/proposed`) rather than a plain average of per-turn ratios, because turn depth varies and `MtpChainStats` already records the counts (`mtp_chain_runner.h:28` and `mtp_chain_runner.cpp:145`). The rule can stay simple: target accept_rate in `[0.75, 0.85]`, bump `keep_ratio` by `+0.01` when the last 3 valid turns fall below `0.75`, decrement by `-0.01` when they rise above `0.85`, and clamp to `[0.025, 0.20]`.

`±0.01` is the right first step size. At long-context chunking it changes a meaningful number of chunks, but the hysteresis band keeps it from ping-ponging. EMA is not worth the extra opacity in v1; a 3-turn window is easier to reason about, easier to log, and already short enough for agentic sessions. I would also add one warm-up rule: initialize new sessions at the server default (`ServerConfig::pflash_keep_ratio`, currently `0.05f` in `http_server.h:56`) and do not adjust until 3 valid samples exist.

Failure modes are straightforward. Oscillation around the thresholds is handled by the deadband; if it still appears in data, widen the deadband before changing the step. If MTP is disabled, there is no accept_rate signal, so freeze the bandit and keep the last ratio instead of inventing a surrogate. If the current turn did not actually exercise pflash, do not learn from it. And if body size jumps sharply between turns, clear the 3-sample window while keeping the current ratio; otherwise the controller mixes two different prompt regimes.

## 2. Session state

Put `std::unordered_map<string, SessionState>` on `HttpServer`, not on `Qwen35Backend`. The state belongs to the HTTP conversation layer, not the model layer, and the worker already sits in the server object (`http_server.h:84` and `http_server.cpp:505`). I would still protect the map with a mutex even though `worker_loop` is single-threaded today (`http_server.h:7` and `http_server.h:145`), because detached client threads already exist and a future multi-worker refactor should not silently turn this into a race.

Key sessions by explicit `extra_body.session_id`. That is the only stable choice here. Hashing `system+first-user-message` would merge distinct conversations, connection-level keying breaks across reconnects and proxies, and cookies are the wrong transport abstraction for API clients. If `session_id` is absent, stay stateless: use the configured default keep ratio and create no session entry.

For cleanup, use time-based TTL, not an LRU cap, as the primary policy. A `last_access > 30 min` sweep matches the actual semantics of a conversation and avoids evicting active sessions just because the server saw a burst of other traffic. An LRU cap of 100 can be a secondary safety net later, but it should not be the primary rule. Persistence should be in-memory only for the first pass. Losing the state on restart is acceptable because the controller is a conservative hint, not user data; persistence adds versioning and model-compatibility complexity for little benefit.

## 3. Signal source

The accept_rate signal is already computed, but it is stranded inside the backend. `GenerateResult` currently only carries `ok`, `error`, `tokens`, `prefill_s`, and `decode_s` (`model_backend.h:68`), while `MtpChainRunner::stats()` already accumulates the needed counts (`mtp_chain_runner.h:28`). The actual `[mtp_decode]` lines are emitted in `mtp_orchestrator.cpp:158` and in `qwen35_backend.cpp:1223`; `dflash_spec_decode.cpp:196` has its own `accept_pct` summary, but that is a local log, not something `HttpServer` can read.

So the signal should be plumbed through `GenerateResult` or a sibling telemetry struct. The minimal useful shape is `std::optional<double> mtp_accept_rate` plus the raw proposal/accept counts; the optional lets non-MTP paths stay silent instead of pretending a zero exists. `worker_loop` already has the returned `GenerateResult` in scope after `backend_.generate()` returns (`http_server.cpp:675`), so once the field exists the control loop can consume it without any extra callback plumbing.

Fallback signals when MTP is off should stay diagnostic, not control inputs. Prefix-cache hit ratio, decode tok/s, anchor-hit density, and any explicit user-side acknowledgement are useful for analysis, but they are too confounded to drive the controller safely. If MTP is off or the backend returns no telemetry, the safest behavior is to keep the current ratio unchanged.

## 4. Integration points

`ParsedRequest` already has `pflash_mode_override` (`http_server.h:79` and `http_server.cpp:424`), but I would not add `keep_ratio_override` in v1. That would create two sources of truth for the same scalar and make the controller test harder to interpret. Instead, add `extra_body.session_id`, resolve `SessionState.keep_ratio` in the worker, and let that value be the single source for `CompressRequest::keep_ratio` (`model_backend.h:118`).

The control loop should wrap the existing worker path in two places. Before `backend_.compress()` in `worker_loop`, resolve the session and copy its current ratio into `creq.keep_ratio` instead of the server-wide default (`http_server.cpp:580`). After `backend_.generate()` returns, read the telemetry, update the session state, and then emit the bandit log line. That placement keeps the update causal: the current request uses the old ratio, and the next request sees the new one.

One important caveat: the current pflash gate still keys off `config_.pflash_mode` at `http_server.cpp:551`, while the typed compress request receives the per-request override at `http_server.cpp:585`. The session bandit should therefore be treated as living inside the existing compress path, not as a mechanism for re-enabling pflash when the server was launched with global OFF. The envelope runner already documents that limitation explicitly (`run_envelope_sweep.py:4`).

## 5. Observability

The first human-readable signal should be a per-turn stderr line like: `[pflash-bandit] session=<id> turn=<n> keep=0.10→0.09 (last_accept=0.83, target=0.75-0.85)`. That line should be emitted after the session update so it reflects the actual next-turn keep ratio, not the stale one. Include `prompt_tokens`, `pflash_applied`, and `window_accept` if space allows; those three values make it obvious whether the controller is reacting to real compression or just to prompt length.

For offline comparison, add append-only JSONL metrics with `session_id`, `turn`, `prompt_tokens`, `keep_ratio_used`, `accept_rate`, `anchor_hits`, `pflash_applied`, `updated_keep_ratio`, and `decode_s`. That gives you a clean A/B split between fixed-keep and adaptive sessions. If the accept_rate field is missing, write `null` rather than `0.0`; otherwise the analysis will confuse “no MTP telemetry” with “hard reject”.

## 6. Test plan

The controller should have a pure unit test path. The best shape is a tiny helper that takes the current `SessionState` plus an observation and returns the next ratio, so the unit tests can cover warm-up, boundary conditions, accept_rate `0`, accept_rate `1`, and the “prompt jump resets the window” case. `dflash/test/test_server_unit.cpp` already carries pflash config tests in the same style (`test_server_unit.cpp:467`), so that is the right home for the first assertions.

The integration test should reuse the existing multi-turn harness. `dflash/bench/run_multiturn_proper.sh` already extracts `accept_rate` from `[mtp_decode]` log lines (`run_multiturn_proper.sh:240`) and already runs real multi-turn clients against the local server (`run_multiturn_proper.sh:1`). For validation, compare fixed keep=0.10 against adaptive keep on the same 5-turn session and require the treatment to Pareto-dominate on at least 70% of sampled sessions. If the controller drives to a corner value (`0.025` or `0.20`) on more than 30% of sessions, treat that as unstable and revert.

## 7. Open questions

The most important choice is whether to update keep_ratio only for requests that actually compressed. I recommend yes: per-request-only, no retroactive recomputation. Once the prompt has already been compressed and generation has started, there is nothing causal to re-run; the best you can do is use the result on the next turn. That keeps the controller honest and avoids double-counting a late observation.

Per-session anchor parameters should stay fixed in the first pass. `head_chunks`, `tail_chunks`, `query_tokens`, and `anchor_radius` all shape the selection policy, not just its budget, so trying to adapt them on the same accept_rate signal will make the system harder to debug. If the controller fails even with a good keep_ratio, that is the signal to revisit anchor policy, not to let the bandit mutate everything at once.

The bandit should also be per-session, not shared across sessions or task types. A global controller would average together unrelated workflows and lose the very signal we care about. If you want transfer learning later, seed the session from a global prior, but keep the updates local. Cold-start state loss on restart is acceptable for the first pass; it simply reverts the server to the conservative default.

## 8. Honest critique

This sketch is not a true bandit; it is a hysteretic controller with a measured proxy. That is fine if the goal is “cheap and predictable,” but it is not the end of the story. If we later want discrete exploration over several `keep_ratio` arms, Thompson sampling or UCB would be the principled upgrade. I would not pay that complexity tax yet because the action space is one scalar and the system is already nonstationary turn to turn.

`±0.01` is probably neither absurdly tiny nor dangerously large. It is small enough that a bad update does not crater the session, and large enough that it changes the chunk budget materially on long prompts. The bigger risk is not step size; it is proxy drift. `accept_rate` is a decoder-side signal, not a direct user-quality measure. If the model starts answering the wrong question confidently, the controller can still look healthy. The mitigation is to keep the loop opt-in by `session_id`, log everything, and preserve a fixed-keep escape hatch.

The “decrease when `accept_rate > 0.85`” rule is intentionally conservative. It sacrifices some compression efficiency to avoid quality cliffs. That is the right bias for the first release, because the worst failure mode here is overcompression that looks good in the telemetry and bad to the user.

## FIRST PR TO LAND

Ship one causal loop, one session, one telemetry field.

- `dflash/src/common/model_backend.h`: extend `GenerateResult` with MTP telemetry fields, at minimum `std::optional<double> mtp_accept_rate` plus the raw proposal/accept counts.
- `dflash/src/common/mtp_orchestrator.cpp` and `dflash/src/qwen35/qwen35_backend.cpp`: populate those fields from `runner.stats()` after the MTP decode path finishes.
- `dflash/src/server/http_server.h` and `dflash/src/server/http_server.cpp`: add `extra_body.session_id`, a `SessionState` map on `HttpServer`, parse the session id in `route_request`, resolve `keep_ratio` from the session before `backend_.compress()`, and update the session after `backend_.generate()`.
- `dflash/test/test_server_unit.cpp`: add pure unit tests for the controller helper covering warm-up, thresholds, boundaries, and a prompt-size jump reset.

Keep this first PR under about 200 LOC of new logic by skipping persistence, JSONL export, fallback surrogates, and adaptive anchor parameters. The proof is simple: two turns with the same `session_id` should show the second turn using the updated `keep_ratio` in the stderr bandit line, while the first turn still uses the configured default.
