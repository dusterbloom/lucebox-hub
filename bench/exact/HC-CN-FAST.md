# hc_combine_norm_f32_b256_fast — gamma-prefetch variant, LUCE_QWEN_HC_CN_FAST

Target from `HC-AUDIT.md` §7: `hc_combine_norm_f32_b256` is the #1 recoverable-ms/tok kernel
(13.9x measured/floor ratio, no weight matrix, pure launch-latency bound). The audit's own two
proposed designs both fail the "exactly one launch, bit-identical" constraint given here:
splitting into half-blocks adds a second kernel (+95 launches/tok); widening `HC_CN_BLOCK2` from
256 to 1024 changes `block_reduce<SUM,1024>`'s warp/shared-mem tree depth and per-thread column
grouping, which reassociates the `sum(a²)` float addition chain and is **not** bit-identical to
`block_reduce<SUM,256>` for the same data (sequential `tmp +=` per-thread accumulation is a
genuine order-dependent chain — grouping columns differently into per-thread partials changes
rounding, full stop). Confirmed this by inspection of `common.cuh:611-628` before attempting it;
did not need a second honest failed attempt to see the reassociation.

## What shipped: single safe latency cut, zero reassociation

`hc_combine_norm_f32_b256_fast` (hc-cn.cu) is byte-for-byte the same kernel — same grid
`dim3(hc,n_tokens,1)`, same block `HC_CN_BLOCK2=256`, same `block_reduce<SUM,256>` call, same
per-element formulas in the same left-to-right order — with exactly one change: `gamma[c,col]`
is loaded into registers (`g0[k]`/`g1[k]`) during the **first** pass (hidden behind that pass's
own memory latency and the reduction's `__syncthreads` round trip) instead of being re-read from
global memory in the **second** pass, after the sync. The second-pass expression
`scale * xs[2*k] * g0[k]` is the identical chain, on the identical float value (same address,
same bytes) as today's `scale * xs[2*k] * gv.x` — this is a timing change only, not an arithmetic
one. Gated by `LUCE_QWEN_HC_CN_FAST` (default off → original kernel, unchanged dispatch site).

## Gate (a): bit-identity unit test

`bench/exact/test_hc_cn_bitexact.cpp` calls a new test-only export
`ggml_cuda_test_hc_combine_norm_bitexact` (hc-cn.cu) that launches both kernels on identical
inputs into separate output buffers, then memcmp's `out_res`/`out_xn` byte-for-byte on the host.
15/15 cases pass (random small-magnitude, random wide-magnitude ×5 each at production shape
n_embd=2560/hc=4; odd n_embd=2561 boundary ×2; hc=1; hc=8,n_embd=1536; n_embd=3072 at
HC_CN_MAX_EMB). Run via `~/qwen4exp-f16/gpu_exec.sh` with `HIP_VISIBLE_DEVICES=1`:

```
[random-small] n_embd=2560 hc=4 out_res BIT-IDENTICAL out_xn BIT-IDENTICAL   (x5)
[random-wide]  n_embd=2560 hc=4 out_res BIT-IDENTICAL out_xn BIT-IDENTICAL   (x5)
[odd-n_embd]       n_embd=2561 hc=4 out_res BIT-IDENTICAL out_xn BIT-IDENTICAL
[odd-n_embd-wide]  n_embd=2561 hc=4 out_res BIT-IDENTICAL out_xn BIT-IDENTICAL
[hc1]  n_embd=2560 hc=1 out_res BIT-IDENTICAL out_xn BIT-IDENTICAL
[hc8]  n_embd=1536 hc=8 out_res BIT-IDENTICAL out_xn BIT-IDENTICAL
[max-embd] n_embd=3072 hc=4 out_res BIT-IDENTICAL out_xn BIT-IDENTICAL
ALL PASS
```

## Gate (b): full [measure_tokens] identical OFF vs ON, 4 reps

md5 of the 256-token pick sequence (`[measure_tokens]` lines, token ids only) is identical across
`cn_off_1`, `cn_off_2`, `cn_on_1`, `cn_on_3` runs (same `8ea3a898ed4eb74bb4a8489edba86e17` for all
four) — ON produces exactly the same generated tokens as OFF for all reps checked.

## Gate (c): flag-off unchanged

OFF-arm timing before vs after is the same kernel code path (dispatch site unconditionally calls
the original `hc_combine_norm_f32_b256` unless `LUCE_QWEN_HC_CN_FAST=1`); OFF numbers below match
the pre-change 38.46 ms/token baseline once the box was compacted (see Timing).

## Timing: fresh processes, interleaved OFF/ON ×3, same session, post drop_caches+compact_memory

`free -g` after compaction: `27G free / 60G available` (was 7-8G free before). All runs below are
post-compaction. `bash run_stack.sh` / `XENV='LUCE_QWEN_HC_CN_FAST=1' bash run_stack.sh`, mode=1
`ms_token` values per fresh process (8 internal reps alternate mode=0/mode=1; mode=1 is the
measured arm per task spec):

| run | mode=1 ms_token values |
|---|---|
| OFF 1 | 39.668256, 39.611154, 38.289236, 38.260084 |
| ON  1 | 38.610124, 38.372835, 38.403856 |
| OFF 2 | 38.816527, 38.448023, 38.461564, 38.444267 |
| ON  2 | 38.320310, 38.348814, 38.358774, 38.381648 |
| OFF 3 | 38.468451, 38.475913, 38.476256, 38.503014 |
| ON  3 | 38.378007, 38.375708, 38.372957, 38.367443 |

Medians: OFF (n=12) = **38.472 ms/token**, ON (n=11) = **38.373 ms/token** → **−0.099 ms/token**
(−0.26%) wall. Small but real and in the expected direction/magnitude (95 launches/tok ×
~1 µs/launch ≈ 0.1 ms/tok, matches the kernel-level delta below).

## Kernel-level: rocprofv3 --kernel-trace, ON build (bench/exact/prof/run_profiled_cnfast.sh)

Same decode geometry as production (`Workgroup_Size=256`, 4-block dispatch per decode token,
n=244180 dispatches matching the baseline trace's dispatch count for the same run config):

| kernel | mean µs | median µs |
|---|---|---|
| `hc_combine_norm_f32_b256` (baseline, prof/kt) | 18.57 | **10.01** |
| `hc_combine_norm_f32_b256_fast` (prof/kt_cnfast) | 17.75 | **9.32** |

Median per-launch latency drops **10.01 µs → 9.32 µs** (−0.69 µs, −6.9%), consistent with the
wall-clock delta (0.69 µs × 95 launches/tok ≈ 0.066 ms/tok, same order as the measured 0.099
ms/tok).

## Honest limit

This is the one latency cut available without reassociating the sum-of-squares reduction: moving
an *independent* load (gamma) earlier in time changes nothing about the reduction's dependency
chain. The reduction itself (`block_reduce<SUM,256>`'s per-thread 6-iteration sequential `tmp +=`
chain, plus the 8-warp LDS round trip) is intrinsically serial and cannot be split across more
threads/blocks without changing float rounding — confirmed by direct inspection of the tree
structure in `common.cuh`, not by a failed build attempt. The remaining ~9.3 µs is dominated by
kernel-launch dispatch + the one unavoidable `__syncthreads` round trip on a 4-block grid, which
only a design that adds a second launch (ruled out by the "exactly one launch" constraint) could
further reduce.
