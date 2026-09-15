# paged_attn_wmma — assembly handoff

Status: design complete, contract extracted, all source lines to copy and all
deltas enumerated. Remaining work is mechanical assembly + box iteration.
Evidence + design: PAGED_ATTN_WMMA_PLAN.md (same branch).

## The kernel = fattn-mma's own code + 6 deltas

Copy verbatim from server/deps/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:
- `flash_attn_ext_f16_iter` (lines 471-965) — KQ mma, mask, softmax, VKQ.
- `flash_attn_ext_f16_process_tile` (lines 1005-1545) — Q load, iter loop,
  rowsum reduce, np-combine, write-back.
- From mma.cuh: tiles `tile<16,8,half2,I_MAJOR>`, `tile<16,16,float,J_MAJOR>`,
  `tile<16,8,half2,J_MAJOR>`, `mma` (RDNA4 f32 accumulate builtin, 2-arg form
  ACCUMULATES: zero the accumulators explicitly — fattn relies on an
  uninitialized-register trick), `get_half2`, `make_identity_mat`,
  `load_ldmatrix` (= load_generic on AMD).

Config: D=256, ncols2=8, ncols1=4, ncols=32, nwarps=8, np=4, nbatch_fa=64,
nbatch_K2=nbatch_V2=128, nbatch_combine=64, Q_in_reg=true, single-stage
(cp.async is NVIDIA-only). Grid: (n_head_kv, ceil(n_rows/4), n_partitions).

## The 6 deltas

1. **Q load** (process_tile Q staging): paged q is F32 [256, n_rows, n_head];
   column (j,c) → row paged_row0+j, head kv_head*6+c. float2 stride = nb/8:
   `Q_f2[j*(q_nb1/8) + c*(q_nb2/8) + k]`, scale_h2 = scale*log2(e)
   (LOG2 DOMAIN). Guards: `j < group_rows` and `c < gqa_ratio`; else zero.

2. **K/V tile loads** (replace `flash_attn_ext_f16_load_tile` calls): per
   64-token chunk, per seq-pass s: resolve each token via the block table
   (PAGED_BLOCK_SIZE=16 → each column's 16-token strip is contiguous):
   `phys = block_table[(t/block_size)*bt_nb0 + seq_s*bt_nb1]*block_size + t%block_size`;
   row = k + phys*k_nb1 + kv_head*k_nb2; F16: direct half2 copy;
   Q8_0: block b = kk/16, scale = row[b*17+16], val = d*qs[2l], d*qs[2l+1].
   OOB tokens → 0.

3. **Mask**: fattn wide-layout mask tile [ncols1][nbatch_fa/2+4] half2s.
   Paged version: row j visible iff `row_seq[j] == seq_s && token < row_extent[j]`,
   else -FLT_MAX. Head slots c>=gqa_ratio and dead rows get extent -1.
   (Causal clamp + dead-row pinning copied from paged_attn_decode lines 275-330.)

4. **expf → exp2f** everywhere (softmax, KQ_max_scale, KQ_cms). Keep
   FATTN_KQ_MAX_OFFSET out (log2 domain). Drop sinks and fixup branches.

5. **Per-sequence passes**: rows in a block may belong to different
   sequences. Loop pass s over the ≤4 unique seqs; stage K/V/mask per pass
   and run iter; columns of other seqs are masked (mma work is wasted only
   in mixed blocks — chunk rows share one seq so the hot case is 1 pass).

6. **Write-back**: single-partition → dst[ row*dst_nb1 + head*dst_nb2 + dim ]
   divided by rowsum. Multi-partition → partial_acc/partial_meta EXACTLY as
   paged_attn_decode (lines 571-602): partial_row = (head*n_rows + row)*
   n_partitions + partition, acc pre-normalized by inv_qk_sum, meta =
   (qk_max, qk_sum) in log2 domain, dead partition sentinel (-FLT_MAX, 0).
   Reuse `paged_attn_partitions` (line 24) and `paged_attn_combine` (line 606)
   verbatim.

## Launcher + hook

- `try_launch_paged_attn_wmma(ctx, dst)`: env `DFLASH27B_PAGED_WMMA`
  (read-once static, default off), gates: non-tree, K/V ∈ {F16, Q8_0},
  supported() passes. Compute n_partitions like try_launch_paged_attn
  (lines 958-995); when n_partitions > 1 allocate partials and launch
  paged_attn_combine after. Bump the counter.
- Hook at the top of ggml_cuda_paged_attn (line 1166): if
  try_launch_paged_attn_wmma(...) return;
- Counter: g_paged_attn_wmma256_launch_count + extern "C" record/get,
  getter in ggml-cuda.h (mirror g_fattn_mma256).

## Qualification (test_paged_attn_wmma.cpp)

Differential vs the V_DOT2 kernel (CPU backend ABORTS on GGML_OP_PAGED_ATTN):
two-mode program like test_fattn_mma256 — env=1 asserts the wmma counter
and dumps outputs, env=0 asserts counter==0 and dumps; a comparator
(tolerance 2e-3) checks the pair. Shapes: rows 64 ragged prefill + rows 1
decode (no query_positions), kv_len 512 (1 partition) and 8192 (multi),
K/V {F16, Q8_0}. CMake: dflash_add_ggml_gpu_executable + GGML_USE_HIP.

## Verify on lucebox4

build-hip (flag OFF): unit test two-mode + the 12K/44K paged ladder A/B
(env 1 vs 0) + concurrency harness C1/2/5. Then GLM-5.3 adversarial review,
fix findings, PR.

## Gotchas learned (do not rediscover)

- 2-arg `mma(D,A,B)` on AMD ACCUMULATES → zero KQ_C and VKQ_C explicitly.
- `2*k0` in the VKQ A-load is the token-pair stride of fattn's V tile —
  copy the exact load, do not "simplify".
- np-combine meta layout: rows strided by tile_stride=68 half2s, meta at
  float2 offset nbatch_combine/2, nmeta=2 when np*cols_per_warp=64>=32.
- The box's pkill self-match: run server scripts via bash files, never
  inline pkill -f chains.
