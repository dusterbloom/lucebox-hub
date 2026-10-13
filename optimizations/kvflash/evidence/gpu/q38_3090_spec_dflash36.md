# Qwen3.8-27B speculative-decode session gate (RTX 3090, WSL) - 2026-10-13

Result: PASS. A, B and C are byte-identical on all 10 turns; B's restore pattern matches q38/t1.

## Setup / parity contract
- Target: /mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf
- Drafter (arms A, B): /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q8_0.gguf (DFlash 1 drafter trained for Qwen3.6, used across versions). Loaded without error ("[draft]  loaded"), no server rejection.
- Binary: /mnt/kimi-k3/build/kvflash-am-cuda/luce_server md5 5ca681c84875bdafacc82aa400f2aa85, same binary on all arms. Built 18:29 from the worktree (branch worktree-kvflash-am, dirty tree). `cmake --build` on my side reported "up to date" (md5 99ca29ec at that time); another session rebuilt it at 18:29 before my runs started (19:27). Worktree HEAD is now 8975fb7d (was 3f7ddfbf at task start); the binary was not rebuilt by me after that.
- All arms: temperature 0, seed 1, --max-ctx 131072, --kvflash 8192, default KV type, port 18190, one server at a time, GPU free (no compute apps on 2 polls 5 min apart: 19:21:44, 19:27:14).
- Arm flags: A = KVFLASH_STATE_LADDER=1 + drafter (cold re-prefill). B = LADDER + LIVE_RESTORE + GEN_CKPT + drafter. C = LADDER + LIVE_RESTORE + GEN_CKPT, no drafter. (q38/t1 used no GEN_CKPT; C here uses B's flags for parity.)
- B and C replay A's prompts verbatim (10 turns, 11.6K to 14.6K tokens). Scripts in scripts/ (gate4_live.py, kvlib.py with KV_EXTRA for --draft, cmp3.py, run_all.sh). Logs: gate4_{A,B,C}.log/.json/.out, cmp.out.
- nvidia-smi memory.used overflows on this WSL driver, so the GPU-free check used `--query-compute-apps`.

## Per-arm medians (10 turns)
| arm | median prefill s | median decode tok/s | median wall s | draft acceptance |
|---|---|---|---|---|
| A cold, drafter | 10.0 | 73.7 | 10.96 | 42.2% (223/528 drafted tokens; median per-request 41.1%) |
| B live restore, drafter | 0.6 | 60.9 | 1.54 | 42.2% (223/528; identical to A) |
| C live restore, no drafter | 0.5 | 53.7 | 1.23 | n/a |

Decode tok/s is client-side (completion tokens over stream time after the first token) and noisy on 29-token answers: A decodes faster than B although both draft the same tokens (identical acceptance); the cause is not isolated (B carries restore/ckpt overhead on the decode path or run-to-run variance), so treat the A-vs-B decode gap as unexplained. C turn 5 client tps 7.3 is a client timing outlier (server-side decode 46.7 tok/s on that turn); the median is unaffected. Drafter gain over C: about +13% (B vs C) to +37% (A vs C) client-side decode.

## Identity
A==B and A==C on all 10 turns (cmp.out: IDENTICAL). Greedy spec decode is lossless here.

## B restore pattern vs q38/t1
Restored prefix L per turn (B): t1 none (L=1, replay 11606) | t2 11602 | t3 11954 | t4 12400 | t5 12609 | t6 12894 | t7 13261 | t8 13707 | t9 13917 | t10 14201. These equal q38/t1 exactly (11602, 11954, 12400, 12609, 12894, 13261, 13707, 13917, 14201). C (no drafter) has the same L, C and replay values as B on every turn. B prefill 0.3-0.8 s per restored turn vs A 9.1-11.3 s.
