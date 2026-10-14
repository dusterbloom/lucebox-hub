# Gate A on the 3090, rebased 2.0a stack (fc5563d1 on main af64ae9f)

Run 2026-10-14 23:51 on the local RTX 3090, with `optimizations/kvflash/tests/gate/run_gate.sh` and `run_scenarios.sh` from fork branch `kvflash/2.0a-E-turn-checkpoints` (tip fc5563d1, the A-E stack on origin/main af64ae9f). Binary `luce_server` md5 6ef2c99b367754e5a768cd675fa27d54, model `Qwen3.8-27B-UD-Q4_K_M.gguf`, same binary and flags on both arms. Arm A sets `LUCE_KVFLASH_SESSION=0`, arm B `LUCE_KVFLASH_SESSION=1`.

| Arm | Prefill s | Decode tok/s | Wall s |
|---|---|---|---|
| A (session off) | 9.50 | 46.3 | 10.10 |
| B (session on) | 0.40 | 46.9 | 1.05 |

Medians over 10 turns, server side.

- Answers: IDENTICAL 10/10.
- prefix_len: A is 0 on every turn. B is 0 on turn 1, then 11602, 11954, 12400, 12609, 12894, 13261, 13707, 13917, 14201. On each turn B's live-restore length equals that turn's prefix_len.
- Scenarios: 1 (pooled restore past the pool) PASS; 2 (512-token boundary) PASS; 3 (image) SKIP, no mmproj file given.
- Against the pre-rebase run at 27336cc3 (A 9.20 s / 47.3 / 9.80 s; B 0.40 s / 47.1 / 1.05 s), B is the same. A's prefill and wall are about 3% slower and its decode about 2% slower. That is one run per arm, so it is not a speed claim either way.

Files: `cmp.out` (per-turn comparison), `gate4_A.out`, `gate4_B.out`, `provenance.txt`, `scenarios.out`. Full logs: `/mnt/kimi-k3/kvflash-gate-rebased-fc5563d1/`.
