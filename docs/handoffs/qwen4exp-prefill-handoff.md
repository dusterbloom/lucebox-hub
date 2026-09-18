# qwen4exp prefill handoff — state at 2026-09-18 (session end)

Repo: `/home/peppi/Dev/lucebox-qwen4exp`, branch `feat/qwen4exp-strix-halo`.
Goal: reproduce then beat the pwilkin strix-halo journey on the hand-written
qwen4exp graph. Target **>= 1000 t/s prefill**. Reference on this box: 738 t/s
(pp16384, IQ4_NL 3-shard). **We are at ~870 t/s @ 13.6k / 847 @ 16K, all
planted-correct.** The two big remaining items are named in "Next".

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

1. **Indexer-K cache + store/select** (THE item). Enables **chunked-prefill
   QSA** (chunks 2+ currently dense: 32K 576 -> should approach ~850) **and**
   `qsa_decode` (see `docs/handoffs/qwen4exp-decode-qsa-reminder.md`).
   Reference: `strix-ref src/models/qwen4exp.cpp:963 build_qsa_store_k`,
   `:1014 build_qsa_top_k`, `:1248 build_attn_qsa`. Why: for `pos0>0` (and
   decode) the indexer keys of `[0,pos0)` are not in `cur` — they must be
   cached per full-attn layer `[idim, max_ctx/r]`. Requires also
   **absolute-position visibility/tail rows** in `build_qsa_attn` (the
   `tv`/`tvt`/`br` terms currently use the local row, must use `pos0+row`).
   ~200-300 lines; validate at `pos0>0` and decode with the FA counter.
2. **64K OOM**: `graph alloc failed (T=65518)`, 27 GB. Suspect the packed
   GDN intermediate (`build_linear_attn` / `ggml_gated_delta_net_skip_intermediate`).
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
