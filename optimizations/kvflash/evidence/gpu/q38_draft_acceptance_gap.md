# Qwen3.8-27B DFlash 2: why A (cold) and B (live restore) differ in draft acceptance

Verdict: not a restore bug, no code change. The gap is a harness mismatch. Arm A ran without `KVFLASH_TURN_CKPT`, arm B with it. TURN_CKPT cuts the pooled prefill at each assistant header (`kvflash_turn_ckpt_enabled`, prefill_helpers.h:235; `turn_cuts` at qwen35_backend.cpp:2299). With the pool evicting, the cut changes the target feature rows the drafter reads. Adding TURN_CKPT to A makes the feature ring bit-identical and acceptance equal on every turn.

Artifacts: /mnt/kimi-k3/artifacts/kvflash-am-gpu/draftgap/ (runs/r1..r3, scripts/, debug_draft.patch). Debug instrumentation (KVFLASH_DEBUG_DRAFT, hashes every feature-ring row of the drafter window at the first 3 draft steps of each request, plus draft-KV state and the proposed block) was a temporary patch, not committed. Binary during runs md5 b24d854b423614f0465e5fcd5f18cbf9 (= c1e748b5 plus the patch). Same target/drafter/prompts as q38_3090_dflash2_session.md (max-ctx 131072, temp 0, 10 turns).

## 1. Per-turn acceptance (original gate logs, reproduced in r1)
The gap is on turn 1 only. Turns 2..10 and warm-up are identical per turn.

| turn | A | B |
|---|---|---|
| 1 | 15/24 (3 steps) | 14/32 (4 steps) |
| 2..10 | 27/32, then 28/32 x8 | same |
| 11th request (turn 10) | 30/32 | 30/32 |

Pooled 268/312 vs 267/320 is that one turn. Turn 1 is a cold prefill in both arms (`live restore: L=1 C=-1`), so restore never ran for it.

## 2. What differs: the feature-ring rows (r1: pool 8192, A = LADDER, B = LADDER+LIVE_RESTORE+GEN_CKPT+TURN_CKPT)
Per-row hashes of the ring rows over the drafter window [committed-4096, committed) (ring cap 4096, kv cap 4096, F32 storage). Draft-KV length and window range are identical (kv_next_pos == committed, same win=[lo,hi) in A and B on every turn); the first drafted block is identical at step 0 every turn.
- warm-up (req 0): ring hash identical.
- turn 1 (req 1): rows 11602..11605 differ, 4 of 4096, everything below is bit-identical. B cut the prefill at 11602 (turn cut, batch 11264-11602 then 11602-11606); A ran one batch 11264-11606.
- turn 2 (req 2): rows 11264..11957 differ (694 rows), everything below 11264 bit-identical. Ring hash differs on every turn 1..10.
- Size of the difference is not FP noise: turn 1 per-row relative L2 up to 0.21 (max abs 10.75), turn 2 up to 1.08 (max abs 104 on rows with max 256); the error grows with layer (capture 5: 0.5%, capture 61: 17-32%).

## 3. Why (code)
Pooled prefill allocates all slots of a batch up front, evicting as needed (qwen35_backend.cpp:2378-2392, `slot_for`), then each query row attends only the resident chunks below kv_pos plus its own batch causally (slot-space mask, ~:2430-2455). Evicted chunks are invisible. So the visible context of a token depends on where the batch boundaries fall: a batch cut changes how many chunks are evicted before the tokens after the cut compute. That moves target hidden states at the capture layers, and with them the drafter's input. The pooled prefill grid comment says the same ("chunk boundaries change the numerics", :2322-2327). Turn cuts are derived from the prompt tokens (assistant headers), so a cold prefill with TURN_CKPT=1 cuts exactly where a restore does. The prefill_helpers.h:231 comment: "Needs no live restore, so a cold reference can run the same cuts."

## 4. Controls
- r2: pool 16384 (nothing evicts), same A and B as r1: all 33 hashed (request, step) ring states identical (rows differ 0 for req 0,1,2 incl. raw max abs diff 0), per-turn acceptance identical (turn 1 15/24 in both), toks identical. Restore + replay writes exactly the cold prefill's rows when there is no eviction.
- r3: pool 8192, A = LADDER+TURN_CKPT (cold re-prefill every turn with the same cuts), B as before: all 33 (request, step) ring hashes identical, raw rows max abs diff 0 for req 1 and 2, drafted tokens identical, acceptance identical on every turn (turn 1 14/32 in both, turns 2..10 27/32, 28/32 x8, 30/32), 4 steps each. Output identical to r1 as before.

So the live restore re-sync (:1115-1128), replay through the S6/S9b batches, draft_kv_ reset per request (:3187) and the window assembly are exact: zero differing rows once the prefill cuts match. Only the harness differed.

## 5. Consequence
Compare restore against a cold reference that uses the same cuts (KVFLASH_STATE_LADDER=1 KVFLASH_TURN_CKPT=1). Against a no-cut cold prefill, acceptance can differ by a few turns' worth of drafts because the target features differ, though the greedy output does not. In the original gate that was 15/24 vs 14/32 on turn 1, one extra step.
