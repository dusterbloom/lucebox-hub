# qwen4exp prefill handoff — state at 2026-09-18 (session end)

Repo: `/home/peppi/Dev/lucebox-qwen4exp`, branch `feat/qwen4exp-strix-halo`.
Goal: reproduce then beat the pwilkin strix-halo journey on the hand-written
qwen4exp graph. Target **>= 1000 t/s prefill**. Reference on this box: 738 t/s
(pp16384, IQ4_NL 3-shard). **We are at ~950 t/s @ 13.6k, all planted-correct.**
Chunked-prefill QSA landed (indexer-K cache): 32K 576 -> ~893, 64K now runs at
925 with `--chunk 16384` (was OOM/dense). Reference CIRU v4.2.0 hits 984-990 @
12,960 on the same GPU, so the remaining gap is ~4% of engine/kernel work.

### Update 2026-09-18 (cont.) — indexer-K cache / chunked QSA landed

- **HC16 mark fix (fusion, big win).** With `QWEN4EXP_MMB_CUBLAS=5`,
  `LLAMA_MMB_HC16=2` was a **no-op**: `LLAMA_HC16_DEBUG=1` showed every xn mark
  blocked (`ok=0`, blocked 12/12) because the HC down/up `MUL_MAT`s are
  cuBLAS-routed and the route read src1 as f32. Added
  `ggml_cuda_mmb_bf16_src()` (root + view_offs + data) in `mmb.cu`, used it in
  the mmb src1 lookup, and taught the cuBLAS route to hand the in-place bf16
  src1 to hipBLASLt (`ggml-cuda.cu` route + marking pass). Result: 13.6K ~900
  -> ~937, 32.8K ~800 -> ~893, 64K 849 -> 925 t/s; `convert_unary` f32->bf16
  829 -> 223 ms, `hc_combine_norm` 972 -> 849 ms (fresh sum 19.36 -> 17.97 s).
  Planted 5/5 at 13.6K, KEY+CEIL at 32K/64K. (commit `849b9336`)
- **Small-chunk indexer fix (review-driven).** A GLM-5.3 review of `2e94878d`
  found that with `--chunk < 2048` the first chunk is dense (off+nb < budget)
  and wrote no `indexer_k`, so the first QSA chunk read uninitialised prefix
  columns. `indexer_store_ok` now pools+persists on every aligned prefill chunk
  regardless of QSA selection, and `Qwen4ExpCache::indexer_blocks` bounds QSA to
  the written prefix. (commit `0529d25f`.) Note: chunk 1024/1536 are
  semantically correct but the planted `KEY_OK` literal flips ~25% because
  chunking shifts a near-tie greedy first token (`CEIL_OK` 100%); production
  chunks 2048/16384 are 8/8.
- **HC inject to cuBLAS — tried, reverted.** The injects
  (`blk.N.hc_attn_inject`, `blk.N.hc_ffn_inject`) are `[10240,4]` **BF16** (not
  IQ4_NL), so `mmb_small_n_bf16` (623 ms, ~51 GB/s for M=4) owns them. Extending
  `ggml_cuda_mmb_cublas_shape_ok` (mode 5, K=10240/N=4) and the route to accept
  BF16 weights gave **no speedup** (~950 unchanged) plus one suspect gate miss,
  so it was reverted. If revisited, cuBLAS M=4 is the suspect (tensor-op kernels
  want M>=16).
- **Remaining non-GEMM buckets @16K** (sum ~17.5 s): `convert_unary` bf16->f32
  515 ms (needs downstream bf16 consumption for the K=2560 projection outputs),
  `hc_gate_mix` 1377 ms (short K=320, ~7.4 TFLOP/s), `mmb_f32split` (F32 router)
  466 ms, `hc_combine_norm` 855 ms (fuse-across-streams would reuse `block_out`
  hc=4x -> 1x, ~250 ms).

### Update 2026-09-19 — 2A landed, 2B rejected

- **2A (landed, `34bef491`):** `convert_unary` is scalar; added vectorized
  contiguous `convert_bf16_to_f32_vec` / `convert_f32_to_bf16_vec` (8 elems per
  thread) with an alignment fast path in `convert_unary_cont_cuda`. Gate 5/5,
  ~938-941 t/s vs ~928-933 (run-to-run varies +-1-2%; treat as ~1%).
- **2B (evaluated, reverted):** marking the `hc_gate_mix` output bf16-only to
  skip its redundant f32 `Out` (671 MB/call) works, but only **12 of 96** mixes
  qualify: the ffn mix is consumed by the F32 MoE router (`ffn_gate_inp`) and the
  linear-attn mix by the F32 `ssm_alpha`, both of which read src1 as f32
  (f32split). Only the full-layer attn mixes are markable -> ~40 ms, within
  noise, at the cost of marking-pass complexity/risk. Reverted,
  `LLAMA_HC16_DEBUG` shows `mixdst ok=1 blocked=11` (blocked by `ssm_alpha` /
  `ffn_gate_inp`, type f32). Not worth it unless those F32 GEMMs learn to read
  the bf16.
- **GLM-5.3 reviews** (zai-coding-plan, non-blocking) found no concrete bug in
  the HC16 cuBLAS src1 pass-through; the indexer-cache review produced the
  small-chunk fix above. The `getrows.cu` hunk was not in the first review
  prompt.

- `build_full_attn` now always pools the chunk's indexer keys
  (`build_indexer_pooled`, kernel `qwen4exp_graph.cpp:325`) into a new per-full-
  layer cache `Qwen4ExpCache::indexer_k` `[128, ceil(max_ctx/ratio)]` f32, and
  `build_qsa_attn` scores against the cached prefix + current chunk with the
  DS4 kernel's `kv_start`. `qsa_ok` dropped `pos0 == 0`; it now needs
  `ratio % 4 == 0 && pos0 % ratio == 0`, `off+nb_cur >= budget`
  (fixes the latent `ggml_top_k` `k <= ne[0]` assert for T < 2048) and enough
  cache columns. Visibility/tail rows in `build_qsa_attn` are absolute
  (`pos0 + row`); algebraically identical at `pos0 == 0`.
- Measured (QSA=1, MMB_CUBLAS=5, SHADOW=1, HC16=2, chunk 16384):
  | pt | chunks (qsa) | t/s | before |
  |---|---|---|---|
  | 13,664 | 1 | 932-943 | ~870 |
  | 23,984 | 2 | ~890 | dense chunk2 |
  | 32,792 | 2 | 893 | 576 @ 32K |
  | 65,528 | 4 | 925 | OOM |
  Planted gate 5/5 at 13.6k, and KEY+CEIL correct at 24K/32K/64K; FA counter
  confirms `qsa=12` on every prefill chunk (decode still dense).
- **Mask elision.** `qwen4exp_forward` now skips building/uploading the dense
  `[kv_len, T]` f16 causal mask when `qsa_layer_ok` holds for every full layer
  (QSA carries its own visibility). `build_full_attn` aborts if it ever gets a
  maskless dense graph. At T=16384 `mask+upload` fell 124.7 -> 15.3 ms and
  `build+alloc` 457 -> 325 ms; 64K 784 -> 826 t/s (the 4 per-chunk masks were
  ~5.4 GB total). No-QSA dense still builds the mask and stays correct.
- **Decode QSA (`qsa_decode`) is *not* wired** (T < 128 stays dense); the cache
  is a prerequisite, not the whole job.
- **Fresh 16K trace** (current build, `--kernel-trace`, sum 19.36 s @ 16,366):
  | ms | x | kernel |
  |---|---|---|
  | 2352 | 48 | `mul_mat` K=6144 N=2560 (`wo`) — cuBLAS route |
  | 1798 | 48 | `mmb_routed_glu_kernel<64,128,32,32>` |
  | 1743+693+458 = **2894** | | cuBLAS `Cijk` bf16 |
  | 1529 | 97 | `mul_mat` K=10240 N=320 (HC down) |
  | 1404 | 96 | `hc_gate_mix` |
  | 1259+909+306+130 = **2604** | | `mmb_dense_kernel` variants |
  | 1148 | 37 | `mul_mat` K=2560 N=10240 (qkv) |
  | 972 | 380 | `hc_combine_norm_f32_b256` (~930 prefill) |
  | 967 | 36 | `gated_delta_net_tiled` |
  | 933 | 48 | `mmb_routed_kernel<128,128,32,64>` |
  | 913 | 12 | `qsa3_attn_kernel` |
  | 829+534 = 1363 | 199 | `convert_unary` f32<->bf16 (cuBLAS route) |
  | 581 | 96 | `mmb_small_n_bf16_kernel<8,32,128>` |
  | 568 | 273 | `mmb_cvt_f32_bf16` |
  | 464 | 12 | `k_get_rows_float<int,int>` (top-k block sort) |
  | 450 | 168 | `mmb_f32split_kernel` |
  A/B of the cuBLAS route at 16,366 (best of 4): mode 1 = 19.19 s, mode 3 =
  19.06 s, **mode 5 = 18.51 s** — the extended route still pays; no regression
  to reclaim. GEMMs run at ~27 TFLOP/s; the remaining time is many small
  elementwise/norm/HC/conversion kernels, so the road to 1000 is fusion, not a
  single misrouted shape.
- **Packed scalar `get_rows`** (`getrows.cu k_get_rows_scalar`). The QSA top-k
  block sort gathers with `ne00 == 1`, and the generic kernel launched 256
  threads (255 idle) per single element. Consecutive threads now take
  consecutive `ne10` rows (coalesced src1/dst), one grid over all
  `ne10*ne11*ne12`. `k_get_rows<int,int>` 464 ms -> ~few ms; end to end
  +3% (13.6K 873 -> ~900, 64K 826 -> 849).

### CIRU v4.2.0 comparison (ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4)

At a near-identical length CIRU measures **984-990 t/s prefill @ 12,960** on
gfx1151 (v4.2.0 release; the 948-974 figure is the **64K** model-card number).
So ~1000 at ~13K is achievable on this silicon and our same-length gap to CIRU
is ~9% (our decode already leads theirs, ~27 vs 21 tok/s). The earlier
"both competitors run ROCm 10.0" claim does **not** hold here: the AMD apt repo
tops out at 7.2.4 (no ROCm 8/9/10), and the on-box pwilkin reference is linked
against ROCm 7.2.2 while our graph is ~22% faster than it on that same
toolchain. The residual gap is engine/kernel work, not the compiler. Their
named v4.2 items do **not** map to our prefill gap:
- "float32 accumulation for 256-wide attention" is in their **generic fallback**
  for batches QSA3 declines; the QSA3 kernel is explicitly unchanged. Our
  `qsa3_attn_kernel` already accumulates f32
  (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`), and we have no
  selected-key fallback path (decode is dense over the whole cache).
- "scratch masks ... 64-query strips" is their fallback's mask. We avoided the
  single 65K graph by chunking, and now elide the mask entirely on QSA chunks
  (above) — a partial port of that idea.
Remaining gap: engine/kernel fusion (mmb/cuBLAS/hc/conversions), item 3-5.
ROCm 10 is not an option on this box (repo max 7.2.4).

## Environment

- Box: `ssh duster@lucebox4.tail97592a.ts.net`, tree `~/lucebox-qwen4exp`,
  ROCm 7.2.2, gfx1151 (Radeon 8060S, 40 CU, max 2900 MHz). Build dir
  `server/build-hip` (`ninja -C server/build-hip dflash_server -j 24`).
- Model (genuine IQ4_NL, 3-shard):
  `~/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf`
  (there is also a mislabeled `qwen4exp-iq4nl-unc/...uncensored_raw_iq4nl.gguf`
  that is actually Q8_0 — do **not** use it for IQ4_NL comparisons).
- Reference source checkout: `/tmp/opencode/strix-ref` (pwilkin). Built ref:
  `~/llama-qwen4/build/bin/llama-bench`.
- **DPM clock**: `power_dpm_force_performance_level=high` is now set on
  `card2`; it was sitting on the ~600/880 MHz state. Verify with
  `cat /sys/class/drm/card2/device/pp_dpm_sclk` (want `2900Mhz *`). It did
  **not** change throughput (prefill is not clock-bound), but keep it.

## Best config and numbers (all planted-correct)

Env (all off by default unless noted):
```
QWEN4EXP_QSA=1          # QSA sparse attention
QWEN4EXP_MMB_CUBLAS=5   # cuBLAS route: 1=K=2560, 3=+ssm_out, 5=+HC down/up
DFLASH_MMB_SHADOW=1     # IQ4_NL bf16 weight shadow (mode 1)
LLAMA_MMB_HC16=2        # journey step 9 marks
```
| prompt | t/s |
|---|---|
| 4K (pt 4078) | 774 |
| 13.6k (pt 13664) | ~870 |
| 16K (pt 16366, pt%4!=0) | 847 |
| 32K (pt 32662) | **576 — chunk 2+ is dense; open item #1** |
| 64K | OOM (27 GB graph alloc) |

Before this session's QSA work, 16K non-%4 was **605** (dense FA). The single
largest fix so far is `ffbd074f`.

## Fresh kernel-trace composition @ 16,366 tokens (build `660bdc25`)

`rocprofv3 --kernel-trace`, sum 19.07 s ≈ wall 19.3 s (the one instrument that
agrees with wall time). This supersedes every earlier ranking (those were taken
at 13.6k pre-`ffbd074f`, with 8.1 s of dense FA that no longer exists).

| ms | x | kernel |
|---|---|---|
| 1765 | 48 | `mmb_routed_glu` |
| 1734+711+453 = **2898** | | cuBLAS `Cijk` bf16 |
| 1367 | 96 | `hc_gate_mix` |
| 1238+916 = **2154** | | `mmb_dense` |
| 969 | **380** | `hc_combine_norm` |
| 950 | 36 | `gated_delta_net` |
| 924 | 48 | `mmb_routed` |
| **807** | 12 | `qsa3_attn` (was 8.1 s dense) |
| 783+550 = 1333 | 199 | `convert_unary` f32<->bf16 (cuBLAS route) |
| 574 | 96 | `mmb_small_n_bf16` (inject) |
| 530 | 273 | `mmb_cvt_f32_bf16` |
| 506 | 12 | `k_get_rows` (QSA top-k gather) |

Buckets: mmb ~5.4 s, cuBLAS ~2.9 s, hc ~2.3 s, conversions ~1.9 s, QSA ~1.3 s,
GDN 0.95 s.

**Target analysis (decide which 1000).**
- **1000 @ 16K** (=pwilkin pp16384, on-box ref 738): need ~3.0 s out of 19.3 s.
  The profile is **flat** — no bucket is 3 s and none is obviously wasteful.
  The "toolchain difference" hypothesis from the prior review was **tested and
  refuted**: (a) `repo.radeon.com/rocm/apt/` tops out at **7.2.4**; `8.0/9.0/10.0`
  all 404, so there is no ROCm 10 to build against; (b) the on-box pwilkin
  reference `~/llama-qwen4/build/bin/llama-bench` is linked against **ROCm
  7.2.2** (RUNPATH `/opt/rocm-7.2.2/lib`) and gets **738 @ pp16384** — on the
  same toolchain our graph is already ~22% faster than pwilkin's. The published
  pwilkin 1187 is therefore not reproducible on this box for toolchain reasons.
  CIRU (987 @ 12,960) is a different engine, and our remaining ~9% gap to it is
  engine-level (decode we already lead: ~27 vs their corrected 21 tok/s).
  **Primary next item reverts to the kernel/engine fusion work below.**
- **1000 @ 64K** (CIRU v4.2.0 @12,960 = 984-990; model card 64K = 948-974):
  needs the **64K graph-alloc OOM fixed** *and* the **indexer-K cache**
  (chunks 2+, decode). Multi-hour, known-good reference. NOTE: the 984-990
  figure is the **12,960-token** release datapoint, not 64K — do not compare
  our 16K number to a 64K reference.
- The indexer-K cache is **irrelevant at 16K** (`--chunk 16384`: both 13,664 and
  16,366 tokens are a single chunk, `pos0 == 0`, QSA already full).

**Two new cheap checks before any 300-line port:**
- `hc_combine_norm` **x380** — RESOLVED, not a bug. Trace split: 95 launches
  with grid.y = 16366 (the prefill graph) + 285 with grid.y = 1 (3 decode steps
  × 95 ops). One launch per op (grid.x = hc covers all 4 streams); `n_layer` is
  48, so 2/layer ≈ 96 ops, not 72. No per-stream and no redundant launch.
- `k_get_rows` x12 / 506 ms — RESOLVED via `k_get_rows_scalar` (see update).

## Commit log (oldest -> newest, all on `feat/qwen4exp-strix-halo`)

`c1525bf8` sparse selected attention + fused HC/MoE prefill;
`02c3f8e7` trim QSA selection; `bcd627a9` cuBLAS route WIP;
`f3586303` restrict cuBLAS to K=2560; `15a7f648` standalone mmb-vs-GEMM difftest;
`a76a1d05` fuse GDN depthwise conv (prefill +18%);
`84a31e5e` fuse PLE depthwise conv;
`c55ec22a` shadow **only** cuBLAS-routed weights (+24% vs shadowing all);
`448a1a0d` extend cuBLAS route to ssm_out/wo;
`4d10e4fe` per-shape MUL_MAT profiling + `QWEN4EXP_MMB_CUBLAS=5` mode;
`4c6ccbd3` small-N bf16 dense kernel for the HC inject (916->~450 ms);
`358e8933` bf16-only HC normalized stream (**journey step 9**, default off);
`d8ece2a0` marked bf16 stored **in place** (was shared cache slot -> race);
`572a893f` abort when an unfused op reads a bf16-only tensor;
`ffbd074f` **pad QSA to whole key blocks** so QSA engages for any length;
`660bdc25` route the QSA indexer through the fused **DS4** kernel;
+ handoff docs.

## Key mechanisms (where to look)

- **QSA** (`server/src/qwen4exp/qwen4exp_graph.cpp:325` `build_qsa_attn`,
  `:438` `build_full_attn`). Gated by `qsa_ok` at `:501`:
  `pos0 == 0 && kv_len == T && T >= 128 && indexer_head_size == 128 && ...`.
  Pads K/V to `kv_pad` (`ffbd074f`) so `kv_len % lcm(4,ratio)==0` and
  `qsa_supported` passes. `qsa_ok` still requires `pos0 == 0` => **only
  chunk 1 uses QSA; chunks 2+ are dense**.
- **QSA eligibility (backend)**:
  `qsa.cu ggml_cuda_flash_attn_ext_qsa_supported` (needs `k->ne[1] % 4`,
  `ids->ne[0] <= 2560`, `q->ne[2] == 12*k->ne[2]`);
  `qsa-decode.cuh:80` `..._qsa_decode_supported` (needs `src[5]` ids,
  `src[6]/src[7]` null).
- **DS4 indexer** (reuse, in-tree): `GGML_OP_DS4_INDEXER_SCORE`
  (`ds4-indexer.cu:922` op, kernel `:158`/`:266`), constructor
  `ggml_ds4_indexer_score(ctx,q,head_weights,index_comp,kv_start,ratio)`
  (`ggml.c:9377`, declared `ggml.h:2792`). Computes
  `sum_h relu(q_h . comp)*w_h` with causal block visibility
  `comp < (kv_start+t+1)/ratio` — identical to QSA's; **kv_start-aware**.
- **HC16 (step 9)**: `ggml-cuda.cu` marking pass after
  `ggml_cuda_mmb_begin_graph()` (~line 5705); dispatch at
  `GGML_OP_HC_COMBINE_NORM` (case ~3700); marks are keyed on the **tensor
  pointer** (`mmb.cu g_mmb_bf16_only`), the marked bf16 lives **in place**
  (`cache_lookup` returns `t->data` for a marked tensor).
- **cuBLAS route**: `ggml-cuda.cu ggml_cuda_mul_mat` (~2963), mode via
  `ggml_cuda_mmb_cublas_shape_ok`. Uses the bf16 weight shadow; per call it
  converts `src1` f32->bf16 and `dst` bf16->f32 (`ggml_cuda_op_mul_mat_cublas`
  ~1847). ~1.5 s of prefill is these conversions.

## Measurement tooling (trustworthy numbers)

- **rocprofv3** was installed user-side, no sudo:
  `apt-get download rocprofiler-sdk7.2.2 rocprofiler-sdk-roctx rocprofiler-sdk-rocpd hsa-amd-aqlprofile7.2.2`
  then `dpkg -x` into `/tmp/sdkfull` (and `/tmp/aqlp`).
  Run: `export PATH=/tmp/sdkfull/opt/rocm-7.2.2/bin:$PATH;
  export LD_LIBRARY_PATH=/tmp/aqlp/opt/rocm-7.2.2/lib:/tmp/sdkfull/opt/rocm-7.2.2/lib:/opt/rocm-7.2.2/lib;
  rocprofv3 --kernel-trace -f csv -d OUT -o t -- <server cmd>` (SIGTERM the
  app so rocprof flushes; do not SIGKILL). **`--pmc FETCH_SIZE WRITE_SIZE`
  crashes on this box — do not rely on it.**
- **FA launch counter**: `QWEN4EXP_FA_TELEMETRY=1` logs
  `[fa] graph qsa=<n> dense=<n>` per graph
  (`fatn.cu ggml_backend_cuda_get_fattn_{qsa,dense}_launch_count`). This is
  what found the QSA gate: prefill `qsa=12`, decodes `dense=12`.
- **Per-shape MUL_MAT**: `GGML_CUDA_OP_PROF=1` prints times + `[mm] K=.. N=..`.
- **Per-call cuBLAS**: `QWEN4EXP_CUBLAS_PROF` (was added/reverted; re-add if
  needed) — isolate conv_in/gemm/conv_out.

## Correctness protocol (do not weaken)

- Gate = **planted-fact** (`/tmp/planted.py PORT 7400`): must reproduce
  `QUINCE-AMBER-7731` and `54`. Use the **multi-rep** runner
  (`/tmp/rep_hc.sh MODE PORT HC16`, 5 reps incl. decode) — the single-prefill
  gate missed the HC16 cache-slot race. Every mark-path change: 5/5 green.
- Any change under `QWEN4EXP_QSA` must assert the FA counter shows `qsa`.
- Anything touching the bf16 marks: the abort guard (`572a893f`) must not fire.

## Next (prioritized) — open items

0. **RESOLVED (not bugs).** `hc_combine_norm` x380 = 95 prefill launches
   (2/layer x 48 layers, minus one PLE handoff) + 285 from 3 decode steps in the
   same 39.7 s trace window; `n_layer` is 48, so "expected 72" was wrong. No
   redundant launch. `k_get_rows<int,int>` x12 / 506 ms is the ascending top-k
   block sort (`qwen4exp_graph.cpp:409-412`), identical to the reference's
   `qwen4exp_select_complete_blocks`; it is a scalar (`ne0=1`) gather of ~8.4M
   int32 needed because `qsa3_rows_kernel`'s unsorted fallback is O(ns^2). Real
   cost (now paid once per prefill chunk), not a bug; a fused sort/gather kernel
   is the only fix.
1. **Indexer-K cache — DONE for chunked prefill** (see the update at the top).
   `Qwen4ExpCache::indexer_k` per full layer, written by every QSA prefill
   forward and scored with `kv_start`. 32K 576 -> 781, 64K 784, planted-correct.
   **Still open:** `qsa_decode` (T < 128 stays dense) — wire the decode kernel
   against the same cache; and the ds4 indexer cost now grows with context
   (chunk4 scores T=16384 x nb=16384), so per-chunk selection is the next lever
   for 64K.
2. **64K OOM**: avoided with `--chunk 16384` (4 graphs of 16384, no 27 GB alloc).
   A single >=65518-token graph still OOMs; suspect the packed GDN intermediate
   (`build_linear_attn` / `ggml_gated_delta_net_skip_intermediate`). No longer
   blocks 64K prefill.
3. **Conversion elimination** (~1.5 s): blocked. Writing f32 `C` in the
   cuBLAS route regresses badly (cuBLAS drops off tensor cores, 446 t/s);
   reading a marked src1 in cuBLAS is still wrong even with in-place marks
   (guard silent => view-mediated reader). Leave until the mark propagation
   is understood.
4. **HC down (K=10240 N=320)**: unquantified. The clock experiment showed
   ~3% for a 3.3x sclk increase => **memory-bound as implemented**, not
   compute-bound, so the earlier "tall kernel +60 t/s" projection is void.
   Mode 5 already routes it to cuBLAS and gains ~5 t/s over mmb.
5. **Plumbing, unexamined**: `hc_combine_norm_f32_b256` (828 ms prefill) and
   `mmb_f32split_kernel` (468 ms).

## Repro commands (box)

```
# server
pkill -9 -x dflash_server; sleep 4
env HIP_VISIBLE_DEVICES=1 DFLASH_HIP_NO_AUTO_UMA=1 GGML_CUDA_MMB=1 \
    QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5 DFLASH_MMB_SHADOW=1 LLAMA_MMB_HC16=2 \
    ~/lucebox-qwen4exp/server/build-hip/dflash_server \
    ~/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf \
    --host 127.0.0.1 --port 8800 --target-device hip:0 --max-ctx 24000 --chunk 16384
# correctness
python3 /tmp/planted.py 8800 7400
```

## Gotchas learned the hard way

- `ggml_can_fuse_subgraph_ext` needs the reference's constant-view-src fix
  (already in `ggml.c`).
- Conv fusion matchers require the PLE/GDN subgraph to be **contiguous**;
  `build_ple` expands its result early to keep it so.
- `mmb_shadow` default mode is 2 (Q6_K only); IQ4_NL shadows need
  `DFLASH_MMB_SHADOW=1`. Shadow **only routed** weights — shadowing all costs
  ~18%.
- Shadows and marks are keyed on **data pointers**; anything keyed on tensor
  pointers cannot resolve through views (res/xn share one fused buffer).
- `ggml_ds4_indexer_score` requires `q->ne[0] == 128` (indexer_head_size).
- Prefill benchmarks are prompt-length sensitive: QSA engaged only when
  `pt % 4 == 0` before `ffbd074f`; always check the FA counter, not just t/s.
