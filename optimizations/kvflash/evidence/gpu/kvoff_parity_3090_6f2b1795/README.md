# KVFlash off: main 6f2b1795 vs the 2.0a stack 09bde623, RTX 3090

Run 2026-10-15 on the local RTX 3090. This repeats `../kvoff_parity_3090/` (main af64ae9f vs fc5563d1) at the current base and tip. The question: with KVFlash off, does the stack change anything compared with main?

Workload: the Gate A chat (10 turns, an 11,606-token first turn, temperature 0, seed 1, 80 max tokens), with no `--kvflash` and no `LUCE_KVFLASH_*` variables. Order: main, E, main, E, so each arm ran twice. Scripts are in `scripts/` (`run_all.sh` drives the four runs; `kvlib.py` and `gate4_live.py` are the same as in the earlier directory; `cmp.py` builds `cmp.out`).

| Axis | Value |
|---|---|
| Model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, md5 76f527022f010d42bc0dff5905c60444 |
| Binaries | main 6f2b1795 md5 3ef3e74c4ae3868e305fb5e8037a7d36; E 09bde623 (gate build) md5 7301e208232e723efdc3bc0f84f42eba |
| Server flags | default KV type, `--max-ctx 131072`, `--no-kv-cache`, otherwise the same on both |
| Caching | RAM prefix cache on both; `prefix_len` matches turn for turn on all four runs (0, 11600, 11952, 12398, 12607, 12892, 13259, 13705, 13915, 14199); `disk_hit=false` on every request, no `disk_hit=true` anywhere |
| GPU | one server at a time under `flock /mnt/kimi-k3/build/3090.lock`; no compute apps on the GPU before each start (see `provenance.txt`) |

Result:
- Pair 1 (main-run1 vs E-run1): IDENTICAL 10/10. Pair 2 (main-run2 vs E-run2): IDENTICAL 10/10.
- Input token counts match on all turns.
- Medians, 10 turns per run (prefill tok/s is new tokens over prefill time):

| Run | Prefill tok/s | Decode tok/s | Wall s |
|---|---|---|---|
| main run 1 | 722 | 43.70 | 1.30 |
| E run 1 | 685 | 43.40 | 1.30 |
| main run 2 | 722 | 43.75 | 1.30 |
| E run 2 | 722 | 43.75 | 1.25 |
| main pooled (20 turns) | 722 | 43.75 | 1.30 |
| E pooled (20 turns) | 720 | 43.45 | 1.30 |

Prefill time in the server log has 0.1 s resolution, so per-turn prefill tok/s on 0.4 to 0.7 s prefills is coarse; E run 1 read 0.1 s slower on turns 2, 3 and 5, and E run 2 did not. Pooled decode means are 43.66 (main) and 43.77 (E). The cold first turn (one sample per run): main 8.5 and 8.6 s, E 8.6 and 8.4 s. The differences are within the run-to-run spread between the two main runs and the two E runs, so this is no speed claim either way.

Decode here is about 43.7 tok/s against 45.5 in the earlier directory. Both arms were slower in this session, so the pairs stay comparable, but numbers must not be compared across the two directories.

Process note: a first attempt overlapped with another job that was using the GPU without the lock. It was stopped and its output discarded; the four runs here were started afterwards, each after an empty compute-apps check.

Files: `cmp.out` (per-turn tables, prefix_len, disk_hit counts, medians per run and pooled), `<arm>-run<N>/driver.out`, `provenance.txt`. Full logs are in `/mnt/kimi-k3/kvflash-gate-kvoff-6f2b1795/`.
