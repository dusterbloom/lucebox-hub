# Where KVFlash turns lose wall time outside prefill and decode

Data: rlru4/off/QK16KND (1106 requests). Rerun: `python3 -I analyze_overhead.py > analyze_overhead.out`.
Also `cross_arm_periodicity.py` (same test on every evidence file) and `jitter_probe.py` (idle host stall check).

## Result in four lines
1. The join is exact: 1106 rows, 1106 log blocks, effective_prompt == prompt_tokens on all of them.
2. The slow tail (100 requests over 1 s, median 2.2 s each, 7% of all wall time) is NOT caused by anything KVFlash does.
3. It is a stall of about 2 s that hits one request roughly every 34 s of clock time, whatever the request is.
4. The ordinary 0.1-0.5 s of overhead is request parsing, template rendering and tokenizing before the server's timer starts.

## Evidence the tail is not a feature of the request
- The server's own timer (the "chat DONE" line, started at the top of process_job) is within 0.1 s of prefill+decode
  even on the worst requests. Client wall minus that timer is 2.0 s median on the 100 spikes. So the lost time is outside
  every server phase: restore, ladder copies, cache insert, response write are all inside the timer and all small.
- Spikes per 100 s of wall are flat across groups: primer 3.30, probe 3.28; 1 ladder checkpoint 3.51, 4 checkpoints 3.24;
  trajectory switch 3.34, no switch 3.28; prompt over the pool 3.22, under 3.67. The apparent links to checkpoints, prefill size and
  "primer" only exist because those requests run longer, so a clock event is more likely to land in them.
- Spike rate by request length matches "chance a 34 s tick lands inside": 0.41 observed vs 0.37 for 8-15 s requests, 0.67 vs 0.54 for 15 s and longer.
- Short probes that spike are phase-locked to a 34.1 s period (R=0.78, random 0.17); gaps between them are 1, 2, 3, 5, 12 times 34 s.
- In the main-vs-branch pair (R5MAIN, R5M71) the spikes land on random requests with identical prompts (same 12335 tokens, 0.07 s vs 2.2 s).
- Same pattern on FULL (no KVFlash at all) and every other arm, about one spike per 30 s of wall (see cross_arm_periodicity.out).
- No disk cache is enabled in these runs (no --kv-cache-dir), no page/evict/insert lines exist in the log, so that path is ruled out.

## Ordinary overhead (the non-spike 0.1-0.5 s)
Time outside the server timer grows with request body size: 0.014 s under 50 KB, 0.16 s at 120-160 KB, 0.49 s over 160 KB (r=0.76).
That is JSON parse + chat template + tokenizer, run before process_job starts. The tokenizer (server/src/server/tokenizer.cpp:461-480)
rebuilds a string for every symbol pair per merge, and Tokenizer::encode (tokenizer.cpp:678-720) rescans the whole text with 33 special-token
finds for every segment. Inside the timer the non-prefill/decode part is 0.1 s or less on 99% of requests (0.1 s is the log's resolution).

## Code path (outside prefill_ms/decode_ms, inside or outside the timer)
- prefill_ms starts at qwen35_backend.cpp:1726 and ends at :1763; it includes the ladder checkpoints (state_ladder_capture, called at :2516).
- decode_ms is :1771-:1835. Not timed: kvflash_live_restore (:1700, body :1073-1138: truncate, 149.6 MiB state restore, feature sync), reset at :1701.
- Server side: started_at at http_server.cpp:4801; response written at :5150; log_done at :5158; finish_job after. Parsing, template (:2540-2575) and tokenize (:2576) run earlier, on the connection thread.
- Bytes: ladder checkpoint 149.6 MiB in pageable host memory (snapshot_target_cache, snap_backend_=CPU, "alloc right-sized" lines), 1-4 per request = 0.15-0.6 GiB,
  about 20-60 ms each at 3-8 GB/s. Live restore is one 149.6 MiB upload, about 30 ms. KV is 18 KiB/token: a 16384 pool is 288 MiB, a 36K prompt 650 MiB.
  Even 1.3 GiB at pageable speed is under 0.5 s. No KVFlash copy can account for 1-3 s.

## Proposed instrumentation (smallest)
Add to the timings JSON (http_server.cpp:5081 GenTimings, sse_emitter.cpp:75-81): `queue_wait_ms` (enqueue to process_job start), `parse_ms`, `template_ms`, `tokenize_ms`,
`restore_ms` (around :1700), `ladder_ms` (inside state_ladder_capture), `finalize_ms` (finalize_generation_cache), `send_ms`, and a monotonic `t_recv_ns` at the first byte read.
Also log one timestamp per phase in the existing START/CACHE/DONE lines. Client side: record `t_connect`, `t_first_byte` and CLOCK_MONOTONIC.
Zero-GPU check first: a 3-minute loop of GET /v1/models every 100 ms against an idle server. If 2 s gaps appear every 34 s the stall is in the HTTP layer or the host, not the engine.
Also test with the replay client on the same box but the server pinned to its own cores, and watch WSL2 memory reclaim.

## Fix candidates
1. Tail: find the 34 s stall with the timestamps above before changing code. Likely host or HTTP-layer, not KVFlash. Do not touch the ladder or restore for it.
2. Ordinary overhead: tokenize and render incrementally (cache the token ids of the unchanged message prefix between turns), replace the per-pair string concat in BPE with an index-based merge, and scan for special tokens once.
3. Only if the new timings later show restore_ms or ladder_ms over 100 ms: pinned host buffers for ladder snapshots, skip checkpoints already held at the same position.
