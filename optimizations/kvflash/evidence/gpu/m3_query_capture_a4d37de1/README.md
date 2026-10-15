# KVFlash 2.0b M3, query capture: fix after review, evidence

Commit under test: `a4d37de1` (branch `kvflash/2.0b-m3-query-capture`), which supersedes `cc437a44`
(`../m3_query_capture_cc437a44/`, kept as the "before" record). Parent M2 tip: `fd1a7b96`.

## Why the fix

Review of `cc437a44` found two majors:
1. Capture cut every prefill chunk at the 640-token block boundary after the normal planner, so a restore at
   position 639 produced a 1-token chunk. That breaks the minimum-chunk and off-grid-lead invariants of
   `qwen35_prefill_chunk_tokens` (short chunks gave NaN logits on R9700).
2. Capture-on turn-1 prefill was -4.5% against M2 (limit 3%). The 512 grid unioned with the 640 cuts gave
   8 chunks per 2560 tokens instead of 5, and the graph is rebuilt per chunk.

## What changed

- Capture never changes the chunk plan: the backend takes `n_tokens` from the planner only, with capture on or off.
- `TargetCache::q_blk_ring`: a ring of block planes, each F16 `[head_dim, 640, n_head, n_fa]`
  (`q_blk` is `ring[0]`). Slots = `q_block_slots(prefill_ubatch, 640)` = 2 for the default 512 ubatch, so
  240 MiB (251.7 MB) with capture on. Block b lives in `ring[b % slots]`.
- A chunk carries one `QBlockWindow {first, count, dst, slot}` per block it touches
  (`plan_q_block_segments`, pure). The graph writes each window into that block's plane.
- A block is sealed (`kvflash_am_seal_block(block, slot)`, the M4/M5 hook) after the chunk that contains its last
  token. A block that began before the prefill (restored partial block) is skipped, never sealed.
- The ring is allocated once per prefill (first chunk that touches a captured block) and freed when the prefill
  ends. It is no longer freed at each seal; the slot is overwritten by block + slots.
- Per-layer plane layout for M4/M5 is unchanged: `[head_dim, 640, n_head]` F16, layer l at `nb[3] * l` of the slot tensor.

## Unit tests

`server/test/unit/test_qwen35_q_block_capture.cpp`, 8 cases (4 kept from before, 4 new or replacing the two block-cut planner cases), all pass on CPU
and on the 3090 (`LUCE_TEST_QBLK_GPU=1`): worst error vs the scalar reference CPU 2.93e-4 / 6.92e-4, 3090
4.95e-4 / 1.23e-3 (layer 0 / layer 1, thresholds 1e-3 / 2e-3).
Cases added or changed:
- `ring_slots_cover_every_block_a_chunk_touches`
- `a_straddling_chunk_gets_one_segment_per_block`
- `restore_at_639_keeps_the_planner_chunks_and_skips_the_partial_block`: replays a restore at 639 with the planner;
  no chunk below `kQwen35MinChunkTokens`, first chunk at least `kQwen35OffGridLeadTokens`, block 0 gets no rows
  and no seal, blocks 1..4 get 640 rows and are sealed, block 5 gets 439 rows and is not, and the chunk count equals
  the planner-only count.
- `a_chunk_straddling_two_blocks_fills_both_planes_against_the_reference`: one 14-token ubatch over two 8-row blocks,
  both ring planes compared with the scalar reference, unwritten rows keep their sentinel.
- The old `chunks_are_cut_at_block_boundaries` case is removed together with `q_block_chunk_tokens`.
Red (`unit/red_stub_planner.log`): with `plan_q_block_segments` stubbed to return nothing, the planner case and the
restore-at-639 case fail (6 pass, 2 fail). The straddle graph case passed in that run because the graph code was
already real; it guards the graph side. Green: `unit/green_cpu.log`, `unit/green_gpu_3090.log`.
Full `test_server_unit`: 1024 passed, 0 failed, 7 skipped (`unit/test_server_unit_full.log`, under flock).
No new warnings in the touched files.

## 3090 A/B, same parity contract as before

Same GGUF `Qwen3.8-27B-UD-Q4_K_M`, default KV type, `--max-ctx 131072`, `--no-kv-cache` (`disk_hit=false` on all
150 chat lines), no `--kvflash`, greedy chat client, `gate4_live.py A` (turn 1 = 11,606-token cold prefill),
5 interleaved runs per arm, under `flock /mnt/kimi-k3/build/3090.lock`, `nvidia-smi --query-compute-apps` empty
before every run (`provenance.txt`), `prefix_len` identical on all 15 runs, answers byte-identical for all 15 runs
(capture on included).

| md5 | file |
|---|---|
| 7690af4619341f6571531d2920926a62 | M2 `luce_server` |
| c5cd15cbfcdd1731f8388b564a4aae78 | M3 `luce_server` at `a4d37de1` (was bed2fd3aef709243cf5e82c729cafcd4 at `cc437a44`) |
| ggml libs | unchanged from `../m3_query_capture_cc437a44/README.md` (ggml sources untouched) |

Medians (turn 1 over 5 runs; decode and wall over all 50 turns):

| arm | turn-1 prefill tok/s | turn-1 wall s | decode tok/s | wall s |
|---|---|---|---|---|
| M2 tip, this run | 1381.67 | 8.80 | 45.60 | 1.20 |
| M3 capture off, this run | 1381.67 | 8.70 | 45.45 | 1.30 |
| M3 capture on, this run | 1398.31 | 8.60 | 45.40 | 1.30 |
| before the fix (`cc437a44` run): M2 | 1381.67 | 8.80 | 44.60 | 1.30 |
| before: M3 off | 1398.31 | 8.70 | 44.50 | 1.30 |
| before: M3 on | 1318.86 | 9.20 | 45.05 | 1.30 |

Gate: capture on vs M2 turn-1 prefill +1.2% (needs better than -3%): pass. Capture off vs M2: prefill 0.0%, decode
-0.3% (limit -2%): pass. Turn-1 prefill is quantised to 0.1 s of 8.2-8.9 s and the spread inside an arm is
about 5% (min to max in `cmp.out`), so the capture-on and capture-off medians are equal within noise.
Each capture-on run logs 20 `sealed block` lines (2 in the warm-up, 18 for turn 1).
