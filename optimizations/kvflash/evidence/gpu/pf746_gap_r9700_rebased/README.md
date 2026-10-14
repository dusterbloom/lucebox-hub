# PFlash admission gap, re-run on the rebased KVFlash 2.0a stack (lucebox6 GPU0, R9700)

Same protocol, chat and arms as `optimizations/kvflash/evidence/gpu/pf746_gap_r9700/`. Date 2026-10-14. Committed as evidence only; nothing pushed upstream or commented anywhere.

## Verdict

1. The gap is still there without the fix (arm A). At turn 5 the admission counted only the last saved RAM prefix snapshot
   (13819 tokens), so uncached droppable = 20028 (>= 16384), decision `compress`, and the 39456-token prompt was compressed to 6688 tokens (6691 prefilled with the template).
2. The fix works (arm B). At turn 5 the admission asked the backend: restorable = 31063, uncached droppable = 2843, decision `under_threshold`,
   prompt served raw (39456 tokens). The live session restored `L=31063 C=31063 replay=8393`.
3. B and C answers are byte-identical on all 5 turns. A differs from B and C on turn 5 only (the compressed prompt).
4. Every admission, restore and prefix number equals the earlier run on the old merge. The rebase on main (PFlash #746, #840, #843, #845) changed none of them.

## Arms

| arm | source | server binary md5 |
|---|---|---|
| A (gap expected) | local branch `local-revert-d11ba7a5` = fc5563d1 + `git revert --no-edit d11ba7a5` = d44387a7f97910633e2ad1e4de504a3e879f9fab (never pushed) | 296a72b88314194afbaa4ef1c768ca75 |
| B (fix) | fc5563d1e61af3daadbf6017a2b525678b44d0eb (fork branch `kvflash/2.0a-E-turn-checkpoints`, on origin/main af64ae9f) | e64839e22f98c896d4b87fb1b9342901 |
| C (reference) | same binary as B, `--prefill-compression off` | e64839e22f98c896d4b87fb1b9342901 |

The source reached lucebox6 as a git bundle (`stack.bundle`) of the fork branch; arm A is a second clone with the local revert. Built with
`build.sh` (HIP, gfx1201, same cmake flags as the earlier run). The build dirs and clones were deleted afterwards; the two binaries stay in `~/pf746r/bin/`.

## Parity contract (every arm)

| axis | value |
|---|---|
| target | `~/models/Qwen3.8-27B-UD-Q4_K_M.gguf`, md5 76f527022f010d42bc0dff5905c60444 (all arms) |
| drafter | `~/models/ggml-org/Qwen3.5-0.8B-BF16.gguf` from Hugging Face `ggml-org/Qwen3.5-0.8B-GGUF`, file `Qwen3.5-0.8B-BF16.gguf`, 1,557,662,496 bytes, sha256 9a7bed4041b7975e0f71fa34670d1e9025213bc92905ac0db75d36c4fa3fa623 (equals the Hugging Face file hash), md5 c1a081ce2cac247a659ff03225f6def1 |
| KV cache | default q4_0 / q4_0 (printed in the server log) |
| context | `--max-ctx 65536`, `--kvflash 16384` |
| sampling | temperature 0, seed 1, max_tokens 64, not streamed |
| chat | the same script and articles (QuALITY ids 22073, 22346, 22876, 22590, 22867; paste caps 28000, 28000, 40000, 40000, 40000 characters); prompts 6712, 13825, 22236, 31067, 39456 tokens on every arm |
| caching | `--no-kv-cache` on all arms; RAM prefix cache and KVFlash session at defaults; `prefix_len` and restore lines read per turn from the logs |
| flags | identical to the earlier run; only `--prefill-compression` (A, B: auto; C: off) and the binary differ. No flag was renamed on main |
| env | `PFLASH_ADMISSION=cache PFLASH_SELECT_MODE=budget_only PFLASH_SELECT_SCORER=legacy HIP_VISIBLE_DEVICES=0` |
| session | one session, arms A, B, C back to back, one server at a time under `~/kvflash-gate/server.lock`, GPU0 idle before the start (no GPU processes listed) |

## Turn 5, per arm

| arm | restorable (admission) | uncached droppable | decision | compressed? | served tokens | prefix_len | live restore line |
|---|---|---|---|---|---|---|---|
| A | 13819 | 20028 | compress | yes | 6691 | 0 | `L=2048 C=-1 replay=6691` |
| B | 31063 | 2843 | under_threshold | no | 39456 | 13819 | `L=31063 C=31063 replay=8393` |
| C | not evaluated (compression off) | - | - | no | 39456 | 13819 | `L=31063 C=31063 replay=8393` |

Admission lines (turn 5): A `[pflash-admission] compress cached=13819 cached_droppable~13764 uncached_droppable~20028 droppable~33792 threshold=16384`;
B `[pflash-admission] under_threshold cached=31063 cached_droppable~30949 uncached_droppable~2843 droppable~33792 threshold=16384`.

Full per-turn tables (prompt, prefix_len, admission, restore line, prefill, decode, wall) for every arm: `tables.md`.

## Speeds (5 turns per arm, one run each; medians)

| arm | prefill tok/s | decode tok/s | wall s |
|---|---|---|---|
| A | 912.7 | 24.5 | 11.0 |
| B | 845.5 | 24.5 | 11.1 |
| C | 844.8 | 24.5 | 11.1 |

Mean, min and max in `medians.txt`. Turn 5 wall: A 12.9 s (includes 4.8 s of compression), B 12.2 s, C 12.1 s. A's median prefill is higher only because its turn 5
prefilled 6691 tokens instead of 8393; this is a functional test, not a speed comparison.

## Answers

B vs C: identical on turns 1 to 5. A vs B and A vs C: identical on turns 1 to 4, different on turn 5 (A: "(C) He was terrified of the confrontation with Torkleson ...";
B and C: "Walter was anxious because the sales and production reports on his desk were the worst in seven years ...").

## GPU memory after the weights (arm B)

The server log of this build does not print free VRAM, a pool size or compute buffer sizes at startup. These are the lines it does print:

```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 32624 MiB):
  Device 0: AMD Radeon Graphics, gfx1201 (0x1201), VMM: no, Wave Size: 32, VRAM: 32624 MiB
[target] target loaded: layers [0,64) output=1, 850 tensors on GPU 14.33 GiB, tok_embd 682 MiB CPU-only (q4_K)
[kvflash] resident pool 16384 tokens (logical max_ctx 65536), tau=64, policy=drafter (attaches on first reselect)
[server] model luce automatic decode KV offload budget: 0 bytes
[vram] released scratch buffers; trimmed 44.2 MiB from device pools     (after the warm-up request; later turns trim 145.4, 474.1, 195.9, 253.0, 250.1 MiB)
```

To get the numbers anyway I repeated arm B once with a sampler polling `rocm-smi --showmeminfo vram` for GPU0 every second (`run_vram_probe/vram.txt`; same answers as the main arm B run):

- idle GPU before start: 57 MiB used
- server ready (weights + 16384-token KVFlash pool, before any request): 14,932 MiB used of 32,624 MiB, so about 17,692 MiB (17.3 GiB) free
- during the chat (compute buffers allocated): 18.3 to 18.9 GiB used; peak 18,908 MiB, so at least 13,716 MiB (13.4 GiB) stayed free through a 39,456-token turn

The 27B weights take 14.33 GiB; the pool plus fixed buffers add about 258 MiB at ready. Compute buffers add roughly 3.4 to 4.0 GiB on the first requests (sampled, not logged by the server).

## Deviations from the earlier protocol

1. Drafter: ggml-org's BF16 Qwen3.5-0.8B file (as asked), not the earlier 1,516,744,736-byte conversion (md5 16671eda...). Admission numbers did not change.
2. Binaries: built from fc5563d1 (arm A: with the d11ba7a5 revert) in two fresh clones under `~/pf746r`, not from the old merge branches. The old tree used the PFlash PR #746 head; this stack is on main with #746 merged.
3. Scripts: only the paths changed (`~/pf746` to `~/pf746r`, drafter path). `pf746_gap.py` and `make_table.py` are otherwise the earlier files.
4. One extra arm-B run with a VRAM sampler, for the memory lines above. It is not part of the three-arm comparison.
5. Same as before: one run per arm, so speeds are indicative only.

## Files

`README.md`, `tables.md` (make_table.py output), `medians.txt`, `medians.py`, `run/` (rows `turns_<ARM>.json`, responses, server logs, per-turn log slices, command lines, provenance),
`run_vram_probe/` (extra B run with `vram.txt`), `pf746_gap.py`, `make_table.py`, `run_arm.sh`, `run_all.sh`, `build.sh`, `vram_sample.sh`, `stack.bundle`.
