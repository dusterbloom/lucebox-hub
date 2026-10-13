# Qwen3.8-27B 3090: DFlash 2 session gate and cross-session determinism

Binary: luce_server md5 c1e748b53b05bf2d80a79979453e207b, built from 95c4fa6e + cherry-pick of 88473180 (DFlash 2 GGUF loader + non-causal SWA masks), clean pick.
Unit suites after merge: test_server_unit 729 passed / 0 failed / 3 skipped (all KvflashLadder/RestorePlan/Ubatch fixtures included), test_dflash2_upstream_schema 42 checks, test_draft_attn_mask 1770 checks. test_kvflash needs a model argument and was not run.
Artifacts: /mnt/kimi-k3/artifacts/kvflash-am-gpu/q38dflash2/ (gate4_{A,B}.{json,log,out}, cmp.out, xsess/).

## Step 2: 10-turn session gate, drafter = Qwen3.8-27B-DFlash2-Q8_0.gguf
Target Qwen3.8-27B-UD-Q4_K_M, temp 0, max-ctx 131072, pool 8192, same prompts as q38spec (turn-1 about 11.6k tokens, 10 turns, max_tokens 80).
Arm A: KVFLASH_STATE_LADDER=1 (no restore). Arm B: STATE_LADDER, LIVE_RESTORE, GEN_CKPT, TURN_CKPT all =1.

- A == B byte-identical on all 10 turns (cmp.out: IDENTICAL).
- Restore pattern C = L on turns 2..10 (e.g. L=11602 C=11602 replay=356 ... L=14201 C=14201 replay=396). Turn 1 is cold (L=1 C=-1).
- Server-side (chat DONE / spec-decode lines, warm-up excluded), medians:

| arm | decode tok/s | accepted/proposed (pooled) | prefill s | wall s |
|---|---|---|---|---|
| A | 83.6 | 268/312 = 85.9% | 9.75 | 10.15 |
| B | 83.2 | 267/320 = 83.4% | 0.45 | 1.05 |

(per-request median acceptance 87.5% in both arms). Decode is within noise; B saves the prefill.
- Checkpoint size: 349.6 MiB per ladder checkpoint (includes the drafter feature ring; 199.6/214.2 MiB for the small pre-pool warm-up ckpts).
- Budget KVFLASH_STATE_LADDER_MIB=3072 did evict: 30 checkpoints taken, held count plateaus at 8 (2797.0 MiB; a ninth would be 3146 MiB > 3072). Restores still hit C = L every turn, so the evicted rungs were not needed.

## Step 3: cross-session determinism (plain_on_noecho turn 1, no drafter)
Earlier 3000 vs 1971 tokens, from the s9b logs (in=8320 request, thinking on, temp 0 seed 1, max_tokens 3000):

| log | out | notes |
|---|---|---|
| before, cold (17:55-17:59) | 1689 | older binary |
| after, after2, coldcut (18:40-19:19) | 1971 | no KVFLASH_DEBUG_RESIDENT |
| dbg_coldcut, dbg_after, dbg2_after (19:34-19:44) | 3000 | LVG_DBG=1 sets KVFLASH_DEBUG_RESIDENT=1 |

So the two runs did differ in flags (DEBUG_RESIDENT), but the binary also changed in between (595b40b9, the truncate(C) pool-refill fix, landed around then; lvg.py was edited 19:32). Binary md5 was never logged, so DEBUG_RESIDENT vs the fix is confounded. The 8320-token prompt exceeds the 8192 pool, so the pooled prefill path is exercised.

Fresh run: lvg.py LVG_ARM=coldcut (STATE_LADDER=1, TURN_CKPT=1, no DEBUG_RESIDENT, no drafter), LVG_TURNS=1, 3 separate server sessions, current binary: 3000 tokens (finish=length) in all 3, message JSON byte-identical across the 3. No cross-session nondeterminism on this binary; the 1971 result matches the pre-fix binary, not a flag-only effect (DEBUG_RESIDENT with the current binary was not re-tested).
