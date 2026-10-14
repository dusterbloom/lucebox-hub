# PFlash 2.0 (#746) with KVFlash 2.0a: the admission gap, R9700 (lucebox6 GPU0)

GPU test 3. Question: with KVFlash on, does `HttpServer::restorable_prefix_tokens()` (RAM prefix-cache snapshots only)
let PFlash compress a long no-tools chat that the live KVFlash session could restore exactly, and does asking the
backend (`ModelBackend::restorable_prefix`, commit 30c09935) close the gap?

## Verdict

1. GAP REPRODUCED on arm A (merge of 30c09935 and PR #746 head eca7265b): at turn 5 the admission counted only the last RAM snapshot
   (13819 tokens, saved at turn 2; turns 3 and 4 outgrew the 16384-token KVFlash pool and skipped their snapshots), so
   uncached droppable = 20028 >= 16384 and the 39456-token prompt was COMPRESSED to 6691 tokens, throwing away a live session that, as arm B shows, covered 31063 of them.
2. FIX WORKS on arm B (same + the http_server.cpp change): restorable = 31063 (the live cut), uncached droppable = 2843, decision under_threshold, prompt served raw
   (39456 tokens), KVFlash live restore `L=31063 C=31063 replay=8393`, i.e. only the new tokens are prefilled.
3. B answers are byte-identical to the exact reference C (compression off) on all 5 turns; A's turn-5 answer differs from C.
   Wall at turn 5: A 12.9 s (4.8 s drafter scoring + 6.9 s prefill + 1.1 s decode), B 12.2 s, C 12.2 s: compressing was lossy and not faster here (n = 1 per arm).

## Parity contract (every arm)

| axis | value |
|---|---|
| target | `~/models/Qwen3.8-27B-UD-Q4_K_M.gguf`, md5 76f527022f010d42bc0dff5905c60444 |
| drafter (PFlash scorer and KVFlash residency scorer) | `~/models/Qwen3.5-0.8B-BF16.gguf`, md5 16671eda429ac1aaff071a3f903380da, passed with `--prefill-drafter` on all three arms |
| KV cache | default `q4_0` / `q4_0` (HIP default, printed in each server log) |
| `--max-ctx` / `--kvflash` | 65536 / 16384 |
| sampling | temperature 0, seed 1, max_tokens 64, non-streaming |
| prompts | identical script and articles on every arm (QuALITY article ids 22073, 22346, 22876, 22590, 22867; system anchor message; one question each) |
| caching state | `--no-kv-cache` (no disk cache) on all arms; RAM prefix cache and KVFlash live session at their defaults; admission/restore state read from the logs for every turn (tables) |
| flags | identical except `--prefill-compression` (A, B: auto; C: off) and the binary (A: merge; B, C: merge+fix) |
| env | `PFLASH_ADMISSION=cache PFLASH_SELECT_MODE=budget_only PFLASH_SELECT_SCORER=legacy HIP_VISIBLE_DEVICES=0` (see deviations 1) |
| session | one session on lucebox6 (A, B, C back to back, one server at a time under `~/kvflash-gate/server.lock`), run dir `final/` |

Binaries: A `luce_server-A` md5 39d0d160e05973ede56aa69dc94b71be (kvflash/pf746-merge da9347580d91767430a03930f40221101fa17ef3);
B and C `luce_server-B` md5 ab232f231e91cbf938191b9a5edd1c4e (kvflash/pf746-merge-fix 78b483bb8e763ec0a6d7765e4243c7db663df503). Details in PROVENANCE.md.

## Final run, per turn (run dir `final/`)

Columns: `prompt tok` = request prompt tokens; `admission` = PFlash decision (`usage.timings.pflash.admission.decision`; `-` = no admission evaluated: prompt below the 16384 threshold in turns 1 and 2, or compression off in arm C); `served tok` = `effective_prompt_tokens`; `restorable (admission)` = tokens the admission
believed restorable (`cached_tokens`); `uncached droppable` as logged by `[pflash-admission]`; `prefix_len` = the `chat DONE` prefix_len;
`live restore` = the `[kvflash] live restore: L= C= replay=` line; `cached_prefix` / `prefilled` = `usage.timings`; prefill tok/s = prefilled / prefill s;
`compress s` = drafter scoring time (`pflash.compress_ms`, not part of prefill s); wall = client wall seconds per request.

### Arm A (merge, compression auto)

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.4 | 1041 | 0.0 | 24.6 | 7.5 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.8 | 907 | 0.0 | 24.5 | 9.0 |
| 3 | 22236 | under_threshold | 22236 | 13819 | 4668 | 13819 | - | 13819 | 8417 | 9.9 | 848 | 0.0 | 24.5 | 11.1 |
| 4 | 31067 | under_threshold | 31067 | 13819 | 12860 | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.5 | 845 | 0.0 | 23.9 | 12.7 |
| 5 | 39456 | compress (COMPRESSED) | 6691 | 13819 | 20028 | 0 | L=2048 C=-1 replay=6691 | 0 | 6691 | 6.9 | 973 | 4.8 | 24.3 | 12.9 |

### Arm B (merge+fix, compression auto)

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.5 | 1037 | 0.0 | 24.5 | 7.5 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.9 | 905 | 0.0 | 24.4 | 9.0 |
| 3 | 22236 | under_threshold | 22236 | 13819 | 4668 | 13819 | - | 13819 | 8417 | 10.0 | 845 | 0.0 | 24.5 | 11.1 |
| 4 | 31067 | under_threshold | 31067 | 22232 | 4478 | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.5 | 844 | 0.0 | 23.8 | 12.7 |
| 5 | 39456 | under_threshold | 39456 | 31063 | 2843 | 13819 | L=31063 C=31063 replay=8393 | 31063 | 8393 | 10.0 | 841 | 0.0 | 23.8 | 12.2 |

### Arm C (merge+fix, compression off, exact reference)

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.5 | 1035 | 0.0 | 24.5 | 7.5 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.9 | 902 | 0.0 | 24.4 | 9.0 |
| 3 | 22236 | - | 22236 | - | - | 13819 | - | 13819 | 8417 | 10.0 | 843 | 0.0 | 24.4 | 11.1 |
| 4 | 31067 | - | 31067 | - | - | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.5 | 841 | 0.0 | 23.8 | 12.7 |
| 5 | 39456 | - | 39456 | - | - | 13819 | L=31063 C=31063 replay=8393 | 31063 | 8393 | 10.0 | 840 | 0.0 | 23.8 | 12.2 |

Resume lines: A turn 4 `live cut 22232 beats server cut 13819 (slot 2)`; B and C turn 4 the same, turn 5 `live cut 31063 beats server cut 13819 (slot 2)`.
Server-log excerpt for the skip (arm A, turns 3 and 4): `[kvflash] pooled prefill: 8417 tokens through a 16384-token pool` followed by
`[kvflash] boundary snapshot skipped: pooled prefill relocates chunks` (this is the skip string that appears in this run; the `snapshot_save` message
`[kvflash] snapshot skipped: cur_pos .. exceeds pool` never printed because the saves here are boundary saves in pooled prefill).

Prompt tokens per turn 6712, 13825, 22236, 31067, 39456: turn 2 fits the pool (RAM snapshot 13819 saved), turns 3 to 5 outgrow it; at turn 5 the uncached droppable
count against the last saved RAM snapshot is 20028 (>= 16384) while the tokens new since the previous turn's live state are 8393 (well below 16384).
T = 5 (the last turn, so no turn after the compression is observed).

### Answers

Identical on all arms for turns 1 to 4. Turn 5:

- A (compressed): "(C) He was terrified of the confrontation with Torkleson regarding the disastrous sales and dividend reports waiting on his desk."
- B and C (identical): "Walter was anxious because the sales and production reports on his desk were the worst in seven years, indicating the company was failing under union management. He feared that Torkleson would use these dismal figures to fire him for incompetence."

Full answers per arm in `tables_final.md`; raw responses in `final/resp_<ARM>_t<N>.json`, server logs `final/server_<ARM>.log`, per-turn log slices `final/log_<ARM>_t<N>.txt`.

## Calibration attempts (all reported; arm A only)

1. `attempts/att1_pool24576_p28000`: the suggested start, pool 24576, ~7K-token pastes (articles cut to 28000 chars: ids 22073, 22218, 22346, 22462, 22524; prompts 6712, 13264, 20381, 27456, 34933).
   Gap NOT reproduced: snapshots saved through turn 3 (20375 tokens), so at turn 5 the uncached droppable count was 9409 < 16384 (under_threshold).
   Reason: with 7K pastes the last saved snapshot sits too close to turn 5 for the count to reach 16384 within 5 turns. Disk KV cache was on (default) in this attempt.

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 0 | L=1 C=-1 replay=6712 | 0 | 6712 | 6.5 | 1038 | 0.0 | 22.2 | 7.6 |
| 2 | 13264 | - | 13264 | - | - | 6706 | - | 6706 | 6558 | 7.3 | 902 | 0.0 | 22.1 | 8.7 |
| 3 | 20381 | under_threshold | 20381 | 13258 | 3181 | 13258 | - | 13258 | 7123 | 8.9 | 796 | 0.0 | 22.2 | 10.2 |
| 4 | 27456 | under_threshold | 27456 | 20375 | 2241 | 20375 | - | 20375 | 7081 | 9.5 | 743 | 0.0 | 22.2 | 10.9 |
| 5 | 34933 | under_threshold | 34933 | 20375 | 9409 | 20375 | L=27452 C=27452 replay=7481 | 27452 | 7481 | 10.1 | 738 | 0.0 | 22.1 | 11.4 |

2. `attempts/att2_pool16384_big345`: pool 16384, turn 1 and 2 pastes cut to 28000 chars, turns 3 to 5 the three longest articles (ids 22876, 22590, 22867, ~8.4K tokens each).
   Gap reproduced (turn 5 compress, uncached droppable 20028). Disk KV cache was still on: turn 1 and 2 restored from disk files written by attempt 1 (`cached_prefix 6706` at turn 1), so it is not
   a clean run; it only fixed the sizes. The numbers from turn 3 on equal the final run.

| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6712 | - | 6712 | - | - | 6706 | - | 6706 | 6 | 0.1 | 47 | 0.0 | 24.6 | 1.4 |
| 2 | 13825 | - | 13825 | - | - | 6706 | - | 6706 | 7119 | 7.8 | 912 | 0.0 | 24.6 | 9.1 |
| 3 | 22236 | under_threshold | 22236 | 13819 | 4668 | 13819 | - | 13819 | 8417 | 9.9 | 849 | 0.0 | 24.5 | 11.1 |
| 4 | 31067 | under_threshold | 31067 | 13819 | 12860 | 13819 | L=22232 C=22232 replay=8835 | 22232 | 8835 | 10.5 | 845 | 0.0 | 23.9 | 12.7 |
| 5 | 39456 | compress (COMPRESSED) | 6691 | 13819 | 20028 | 0 | L=2048 C=-1 replay=6691 | 0 | 6691 | 7.7 | 871 | 4.9 | 24.3 | 13.8 |

3. `final/`: attempt 2 sizes, `--no-kv-cache` added, arms A, B, C. Same admission numbers as attempt 2 for arm A, confirming the reproduction.

## Deviations from the task text

1. `PFLASH_SELECT_MODE=budget_only` and `PFLASH_SELECT_SCORER=legacy` were added. Without them (no head, legacy selection) the cache-admission branch, which is where
   `restorable_prefix_tokens()` is called, is not reached: it sits inside `if (experiment.selection_active && query_window.valid())` in `apply_pflash_compression`, and strict selection with the
   default `head` scorer needs a scoring head. `legacy` is the built-in attention-mass scorer. The task's `PFLASH_ADMISSION=cache` and `--prefill-threshold 16384` are set as asked.
2. Pool 16384 and per-turn paste sizes (6.7K, 7.1K, 8.4K, 8.8K, 8.4K tokens) instead of pool 24576 and ~7K pastes (attempt 1 above).
3. `--no-kv-cache` on every arm of the final run (a default disk KV cache carries state across runs, as attempt 2 showed). The four `.dkv` files written by attempts 1 and 2 into
   `~/.cache/lucebox/kv-cache/0142299ecbc6d429bc431aa167e5933c/` were deleted by name after the final run.
4. `--prefill-skip-park` on every arm (release-profile recommendation for a 32 GB card); a short warm-up request (about 760 tokens, 4 max tokens) before turn 1, as in the gate scripts.
5. Built in a separate clone `~/pf746/src` (not `~/lucebox-sr`) with the cmake flags of `~/kvflash-gate/build.sh` (`build.sh` here); the merge commit was created with
   `git merge-tree --write-tree` + `git commit-tree`, and the edit for arm B was made in a worktree under `/home/peppi/Dev/lucebox-builds/`.
6. One run per arm for the timing columns (a functional admission test, not a speed benchmark): decode tok/s differs by about 0.5 between runs of different generation length.

## Files

`pf746_gap.py` (client and server launcher), `run_arm.sh`, `run_all.sh`, `build.sh`, `build_A.log`, `build_B.log`, `make_table.py`, `tables_final.md`, `tables_att1.md`, `tables_att2.md`,
`arm_B_fix.patch` (the http_server.cpp change, B minus A), `PROVENANCE.md`, `final/`, `attempts/`.
