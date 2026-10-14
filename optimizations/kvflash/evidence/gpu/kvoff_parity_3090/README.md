# KVFlash off: main af64ae9f vs the 2.0a stack fc5563d1, RTX 3090

Run 2026-10-15 on the local RTX 3090. The question: with KVFlash off, does the stack change anything compared with main?

Workload: the Gate A chat (10 turns, an 11,606-token first turn, temperature 0, seed 1, 80 max tokens), with no `--kvflash` and no `LUCE_KVFLASH_*` variables. Scripts are in `scripts/` (copies of the gate driver without `--kvflash`).

| Axis | Value |
|---|---|
| Model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, md5 76f527022f010d42bc0dff5905c60444 |
| Binaries | main af64ae9f md5 46cf157fc1f6406261f46b69d0531379; stack fc5563d1 md5 6ef2c99b367754e5a768cd675fa27d54 |
| Server flags | default KV type, `--max-ctx 131072`, `--no-kv-cache`, otherwise the same on both |
| Caching | RAM prefix cache on both; `prefix_len` matches turn for turn (0, 11600, 11952, ... 14199); no disk hits |

Result:
- Answers byte-identical on 10/10 turns.
- Medians:

| Binary | Prefill tok/s | Decode tok/s | Wall s |
|---|---|---|---|
| main | 722 | 45.55 | 1.30 |
| stack | 737 | 45.50 | 1.20 |

Prefill tok/s is new tokens over prefill time. The cold first turn is one sample: 8.1 s on main, 8.2 s on the stack.

Main's decode dipped to about 38 tok/s on turns 8-10 while the stack held 45. That lowers main's mean (43.53 against 45.46) and its worst turn (37.70 against 44.80). The medians are equal, and main held about 46 on those turns in the void run below. One run per binary, so this is not a speed claim either way.

Why `--no-kv-cache`: the first pair of runs used the default on-disk KV cache (`~/.cache/lucebox/kv-cache`), which both binaries share. The stack then got disk hits from main's entries on turns 1 and 10 (prefix_len 11600 against main's 0), so that pair is void. Its answers were also 10/10 identical. Its logs are kept in `/mnt/kimi-k3/kvflash-gate-kvoff/run1-diskpolluted-*`.

Files: `cmp.out` (per-turn table), `main-af64/driver.out`, `stack-fc5563d1/driver.out`. Full logs are in `/mnt/kimi-k3/kvflash-gate-kvoff/`.
