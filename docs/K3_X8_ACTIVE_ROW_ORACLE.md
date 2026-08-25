# K3 X8 active-row perfect-token oracle

Date: 2026-08-25
Verdict: **NO-GO for the current exact-scalar/current-IQ1_S q=4 verifier**

This discriminator asked whether perfect future tokens make the existing exact
Kimi K3 target stack at least 15% faster.  It did not.  The active-row path was
byte exact and substantially reduced expert delivery work, but end-to-end
decode was 2.67% slower than scalar AR.

## Frozen closure

- Worktree base: `51daa448c164ec11e48f98f7d4d72dca51e1d6e0`
- Experimental diff: `1dbdbad4e74261d2294553bc752285ffd7a3258b4f4891b3dc7581ee255e1226`
- Binary: `/tmp/k3-wide24-build/smoke_kimi_k3_forward`
- Binary SHA-256: `73021a2f6b9dff41ff280ca2c5f01f822af6bde8e67040d56268fa9da4a0e342`
- Model: `/home/duster/kimi-k3-deploy/p32-core/Kimi-K3-KDA-HYBRID-Q2-MIDLATE-00001-of-00014.gguf`
- First-shard SHA-256: `93cf4fa551660cbc390e6bdea12caf67403cdb0091d82a17b14e39d253708fc4`
- Prompt: 53 IDs, SHA-256 `fd835f624d053f0f2da04114215461906430685463e99fb5e94cdf17115acbb0`
- Frozen output: 36 IDs, SHA-256 `2a90dd5334722853fe64997b313c0c9489f60e858bd40eeca292fafe07ffcb74`
- Physical device: gfx1151, exposed as logical `hip:0` by
  `HIP_VISIBLE_DEVICES=1,0`
- Platform profile: `performance`; both AMDGPU performance levels: `high`

The accepted startup closure was exact scalar prompt prefill, single-owner
gfx1151, calibrated96 `valid-layers=92/92`, slab budget 24, 16-GiB P30,
P40/macro-union off, and P41/P42/P45/P46 on.

## Exact invocation

The scalar control used the same command with the oracle variable omitted and
the three output paths changed from `long-oracle` to `long-scalar`.

```sh
KIMI_GPU_LEASE_LOCK=/home/duster/.kimi-k3-gpu.lock \
  /home/duster/lucebox-k3-p40/scripts/gpu_lease.sh run \
  k3-x8-active-oracle-long -- /bin/bash -c \
  'exec /usr/bin/env -i \
  HOME=/home/duster USER=duster LOGNAME=duster \
  LANG=en_US.UTF-8 LC_ALL=C.UTF-8 \
  PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  TMPDIR=/tmp LD_LIBRARY_PATH=/opt/rocm/lib \
  HIP_VISIBLE_DEVICES=1,0 ROCBLAS_USE_HIPBLASLT=0 \
  DFLASH_KIMI_PRODUCTION_DEFAULTS=1 \
  DFLASH_KIMI_CALIBRATED96_AUX_DIR=/home/duster/kimi-k3-deploy/aux \
  DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR=/home/duster/kimi-k3-deploy/streamed-bank/natural-sidecars \
  DFLASH_KIMI_SMOKE_MAX_CTX=128 DFLASH_KIMI_STAGE_PROFILE=1 \
  DFLASH_KIMI_X8_ORACLE_TOKENS=/home/duster/kimi-k3-deploy/k3-x8-active-oracle-20260825/long-target.ids \
  DFLASH_KIMI_CALIBRATED96_METRICS_OUT=/home/duster/kimi-k3-deploy/k3-x8-active-oracle-20260825/long-oracle.traffic.tsv \
  DFLASH_KIMI_X8_BOUNDARY_LOGITS_OUT=/home/duster/kimi-k3-deploy/k3-x8-active-oracle-20260825/long-oracle.boundary.f32 \
  DFLASH_KIMI_LOGITS_OUT=/home/duster/kimi-k3-deploy/k3-x8-active-oracle-20260825/long-oracle.final.f32 \
  /tmp/k3-wide24-build/smoke_kimi_k3_forward \
  /home/duster/kimi-k3-deploy/p32-core/Kimi-K3-KDA-HYBRID-Q2-MIDLATE-00001-of-00014.gguf \
  0 36 \
  @/home/duster/kimi-k3-deploy/k3-x8-active-oracle-20260825/long-prompt.ids \
  > /home/duster/kimi-k3-deploy/k3-x8-active-oracle-20260825/long-oracle.log 2>&1'
```

Two path-only preflights were INVALID and contributed no timed model work:

1. The supplied `/mnt/kimi-k3/models/...` model alias did not exist.  The
   tokenizer failed before model/GPU initialization.
2. After selecting the content-identical retained model, the supplied
   `/mnt/kimi-k3/artifacts/...` aux and sidecar aliases did not exist.  Model
   weights loaded, but stream initialization failed closed with all 92 aux
   layers missing and zero P20/P30/P41/P45 work.  No generation ran.

The failed log was overwritten only after verifying that it contained no
qualified result artifacts.

## Result

| Metric | Scalar AR | Perfect q=4 active rows | Change |
|---|---:|---:|---:|
| Decode wall | 16.537 s | 16.991 s | +2.75% |
| Useful true AR | 2.116425 tok/s | 2.059934 tok/s | -2.67% |
| Logical provider bytes | 111,530,926,080 | 111,530,926,080 | identical |
| Physical direct bytes | 41,928,515,584 | 42,594,680,832 | +1.59% |
| Direct-I/O wall | 14.595559 s | 13.783965 s | -5.56% |
| Effective direct-I/O | 2.675 GiB/s | 2.878 GiB/s | +7.57% |
| P45 H2D bytes | 111,648,584,800 | 67,645,471,184 | -39.41% |
| P45 jobs | 33,218 | 20,191 | -39.22% |
| P45 device window | 6.410648 s | 4.017804 s | -37.33% |

The oracle executed seven target steps: q=4, physical capacity 8, exactly five
active rows per step, 35 committed state rows, 21 inactive rows, no scalar tail
step, zero oracle fallbacks, and 36 emitted tokens including the terminal token.
Provider exact fallbacks remained 658 routes in both arms and are distinct from
the zero oracle-fallback counter.

The seven oracle target steps took 16,987.011 ms.  Their classified stage sum
was:

| Stage | Total | Per target step |
|---|---:|---:|
| Exact scalar causal core | 7,916.012 ms | 1,130.859 ms |
| Experts | 8,369.231 ms | 1,195.604 ms |
| Join | 509.453 ms | 72.779 ms |
| Output | 155.458 ms | 22.208 ms |
| Other | 5.426 ms | 0.775 ms |
| Classified total | 16,955.582 ms | 2,422.226 ms |

The scalar runner did not emit the same P58 stage decomposition, so its P45
device-window counter is retained as a separate subcounter and is not relabeled
as scalar expert wall.

## Exactness

- All 36 emitted IDs equal the frozen scalar IDs.
- Seven boundary-logit rows are byte identical:
  `43938021a3a97b31f9ddbc92d57aa27160ef48919bf928ef785f3b7fab49eff8`.
- Terminal logits are byte identical:
  `382ca5a264a3c5749fedf0be4e4b468df4d170c4eb9e81255888b6b220390d6d`.
- Logical traffic TSVs are byte identical:
  `5ae3a3f0a921162f2bf3241481570077d98cddaa0e951c5e6828c17add545dcd`.
- Qualified oracle log:
  `954fe4492794878bb1a3e3508aa76abff0594431767f576136be9a70ad9d4d1e`.

## Gates and ceiling

The +15% gate was 2.433889 true AR tok/s, or at most 14.380 s for 35
transitions.  The oracle needed to remove 2.611 s from its measured 16.991 s;
instead it added 0.454 s relative to scalar.

At q=4, five committed state rows per target step imply a 500-ms target-step
budget for 10 useful tok/s.  The measured target step was 2,426.716 ms, requiring
4.85x acceleration.  The exact scalar core plus join/output/other alone took
1,226.621 ms per step, already 2.45x over that entire budget even if expert work
were free.  The current scalar causal core is therefore the immediate binding
term for a q=4 verifier.

This NO-GO closes only the perfect-token q=4 active-row verifier using the
current exact-scalar causal core, current IQ1_S expert kernels, budget-24 P30,
and P40/macro-union disabled.  It does **not** close batched decode generally,
q=7 with no interior padding, a grouped/fused causal core, a different direct
IQ1_S multirow kernel, or later verification after those target costs improve.
