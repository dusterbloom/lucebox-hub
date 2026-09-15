# paged_attn_wmma — assembly handoff (GLM-5.3 reviewed, rev 2)

Status: design complete, contract extracted, adversarial-reviewed. All source
lines to copy and all deltas enumerated; remaining work is mechanical
assembly + box iteration. Evidence: PAGED_ATTN_WMMA_PLAN.md (the design
section of that doc is SUPERSEDED by this file — see correction 7).

## The kernel = fattn-mma's own code + 6 deltas

Copy verbatim from server/deps/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:
- `flash_attn_ext_f16_iter` (lines 471-971) — KQ mma, mask, softmax, VKQ.
- `flash_attn_ext_f16_process_tile` (lines 1010-1554) — Q load, iter loop,
  rowsum reduce, np-combine, write-back.
- Copy `mma_tile_sizes<ncols>` verbatim (fattn-mma-f16.cuh:991-998). The
  tiles are `tile<16,16,float>` and `tile<16,8,half2>` with DEFAULT
  I_MAJOR — on RDNA4 the I_MAJOR C layout already IS the transposed A&B
  layout. Do NOT declare J_MAJOR tiles.
- mma.cuh: 2-arg `mma(D,A,B)` ACCUMULATES on AMD; tiles zero-init via
  default member initializers (`T x[ne] = {0}`), so KQ_C/VKQ_C start
  zeroed by construction — still zero them explicitly for clarity.

Config: D=256, ncols2=8, ncols1=4, ncols=32, nwarps=8, np=4, nbatch_fa=64,
nbatch_K2=nbatch_V2=128, nbatch_combine=64, Q_in_reg=true, single-stage
(cp.async is NVIDIA-only). Grid: (n_head_kv, ceil(n_rows/4), n_partitions).

## The 6 deltas

1. **Q load**: paged q is F32 [256, n_rows, n_head]; column (j,c) →
   row paged_row0+j, head kv_head*gqa_ratio+c with the RUNTIME gqa_ratio
   (never hardcoded 6). float2 stride = nb/8:
   `Q_f2[j*(q_nb1/8) + c*(q_nb2/8) + k]`, scale_h2 = scale*log2(e)
   (LOG2 DOMAIN). Guards: `j < group_rows` and `c < gqa_ratio`; else zero.
   Head-slot deadness (c >= gqa_ratio) is handled by this zero-Q plus the
   write-back skip — NOT by the mask (the mask tile is per-row only).

2. **K/V tile loads**: per 64-token chunk, per seq-pass s: resolve each
   token with the block-table VALIDATION decode performs (paged-attn.cu
   :452-477: `physical_block < 0 || >= pool_tokens/block_size` → no row,
   zeros). Use the RUNTIME `block_size` op param (gate `block_size == 16`
   in the launcher). row = k + phys*k_nb1 + kv_head*k_nb2.
   F16: direct half2 copy.
   Q8_0: block_q8_0 = {half d(2B); int8 qs[32]} = 34B. For half2 index kk
   (dim pair 2*kk): b = (2*kk)/32, l = (2*kk)%32;
   `d = *(const half *)(row + b*34)`; val = make_half2(d*qs[2l], d*qs[2l+1])
   with qs = row + b*34 + 2. (Row stride k_nb1 is BYTES.)

3. **Mask**: fattn wide-layout mask tile [ncols1][nbatch_fa/2+4] half2s,
   indexed by row only. Paged version: row j visible iff
   `row_seq[j] == seq_s && token < row_extent[j]`, else -FLT_MAX.
   Causal clamp + dead-row pinning from paged-attn.cu:275-330; the
   dead-row SENTINEL WRITE is at :328-348 (an earlier rev pointed at the
   wrong range — the exit block matters).

4. **expf → exp2f** everywhere (softmax :690/:770, KQ_max_scale :784,
   KQ_cms :1404). Drop FATTN_KQ_MAX_OFFSET (log2 domain). Drop sinks.
   INVARIANT: keep `KQ_max` init at `-FLT_MAX/2.0f` (fattn-mma:1088).
   This is load-bearing: paged passes can be FULLY masked (foreign-seq
   passes, dead rows), and `-FLT_MAX/2` makes the exponent
   -inf - finite = -inf → exp2 = 0, never -inf - (-inf) = NaN. Do NOT
   "harmonize" with paged decode's -FLT_MAX init (decode guards
   `score > -FLT_MAX/2 ? exp2f : 0` instead — different mechanism).

5. **Per-sequence passes**: rows in a block may belong to different
   sequences. Enumerate the ≤4 unique seqs ONCE into shared memory
   (block-uniform; a per-warp divergent enumeration deadlocks the
   syncthreads). Per pass s: stage K/V/mask for seq_s, run iter; other
   columns masked. Per-seq active_partitions: inside the block, a row
   whose seq is inactive at THIS partition must take the
   `(-FLT_MAX, 0)` sentinel write (:332-336) even while other rows run —
   paged_attn_combine treats meta.y > 0 as live, and an unwritten
   sentinel reads recycled pool memory as a plausible float2.

6. **Write-back** — the most likely source of a silent bug; copy the guard
   set, not just the addresses:
   (a) SKIP dead columns: `c >= gqa_ratio || row >= n_rows || row invalid`
       (fattn-mma:1506 translates to: head >= n_head || row >= n_rows ||
       row_seq invalid). Writing them OOB corrupts neighboring rows
       silently.
   (b) Single-partition divide: use decode's guard
       `inv_sum = qk_sum > 0 ? 1/qk_sum : 0` (paged-attn.cu:570) — dead
       rows yield 0, not 0/0=NaN.
   (c) Invalid-slot rows → dst = 0 (the :328-348 exit behavior).
   (d) Multi-partition: partial_acc/partial_meta EXACTLY as
       paged_attn_decode :571-602 (pre-normalized acc, log2 meta,
       sentinel). Reuse `paged_attn_partitions` (:24) and
       `paged_attn_combine` (:606) verbatim.

## Launcher + hook

- `try_launch_paged_attn_wmma(ctx, dst)`: env `DFLASH27B_PAGED_WMMA`
  (read-once static, default off). Gates: non-tree, K/V ∈ {F16, Q8_0},
  supported() passes, **gqa_ratio <= 8** (ncols2=8; without this gate any
  ratio > 8 silently never computes heads 8+ — deterministic silent bug
  on model swap), **block_size == 16**. Compute n_partitions like
  try_launch_paged_attn :958-995 INCLUDING the
  GGML_CUDA_PAGED_ATTN_FORCE_PARTITIONS override at :1002-1012 (copy it;
  otherwise whether kv=512 stays single-partition varies by box and the
  direct-write path is untested nondeterministically). When n_partitions > 1
  allocate partials and launch paged_attn_combine after. Bump the counter.
- Hook at the top of ggml_cuda_paged_attn (:1166): if
  try_launch_paged_attn_wmma(...) return;
- Counter: g_paged_attn_wmma256_launch_count + extern "C" record/get,
  getter in ggml-cuda.h (mirror g_fattn_mma256).

## Qualification (test_paged_attn_wmma.cpp)

Differential vs the V_DOT2 kernel (CPU backend ABORTS on GGML_OP_PAGED_ATTN
— confirmed ggml-cpu.c:2247-2249): two-mode program like test_fattn_mma256
— env=1 asserts the wmma counter and dumps outputs, env=0 asserts
counter==0 and dumps; a comparator checks the pair. Matrix:
- rows 64 ragged prefill, pin NON-4-ALIGNED mixes that form mixed-seq
  blocks (e.g. 30+20+14 across slots) — the NaN/garbage classes fire only
  when a seq boundary is not 4-aligned.
- rows 1 decode (no query_positions).
- kv_len 512 (single partition, deterministic only with the
  FORCE_PARTITIONS override or explicit env) and 8192 (multi-partition).
- MIXED LENGTHS at n_partitions>1: one long seq + one short seq — covers
  the per-seq sentinel class.
- K/V {F16, Q8_0} (4 combos).
- A two-mode dump at kv >= 32768 f16 (the f16-acc VKQ drift class grows
  ~sqrt(iterations); 44K ctx ≈ 687 rescales vs 128 at kv 8192).
- Tolerance: q8_0 vs V_DOT2 compares dp4a-integer dequant against
  WMMA half-dequant — a wider gap than the contiguous 2e-3-vs-CPU test;
  expect borderline failures, use ~3e-3 for q8_0 with justification or a
  common f32 reference.
CMake: dflash_add_ggml_gpu_executable + GGML_USE_HIP.

## Verify on lucebox4

build-hip: two-mode unit test + the 12K/44K paged ladder A/B (env 1 vs 0)
+ concurrency harness C1/2/5. Then a GLM-5.3 pass on the actual kernel
diff, fix findings, PR.

## Gotchas learned (do not rediscover)

- 2-arg `mma(D,A,B)` on AMD ACCUMULATES (zero the accumulators anyway).
- `2*k0` in the VKQ A-load is fattn's token-pair stride — copy exactly.
- np-combine meta layout: rows strided by tile_stride=68 half2s, meta at
  float2 offset nbatch_combine/2, nmeta=2 when np*cols_per_warp=64>=32.
- The box's pkill self-match: run server scripts via bash files, never
  inline pkill -f chains.
