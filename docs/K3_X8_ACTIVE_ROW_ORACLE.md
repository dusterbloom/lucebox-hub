# K3 X8 perfect-token verifier gates

Date: 2026-08-25
Verdict: **q=4 scalar-core NO-GO; full-width q=7 exact GO**

This discriminator asked whether perfect future tokens make the existing exact
Kimi K3 target stack at least 15% faster.  It did not.  The active-row path was
byte exact and substantially reduced expert delivery work, but end-to-end
decode was 2.67% slower than scalar AR.

A later full-width q=7 arm reused the qualified Core8/MLA8/Tail8/Router8 and
macro-union target stack with no inactive rows.  It reached 3.714881 true AR
tokens/s versus a same-binary 2.248906 control, with byte-identical traffic,
logits, and a one-step recurrent/MLA state dump.  This is a material verifier
architecture win, not yet a trained-draft or production-server result.

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

## Width-5 macro-union discriminator

A follow-up tested whether the qualified P40 layer-epoch macro-union/direct-
async delivery path could rescue the same perfect q=4 verifier.  The physical
replay capacity remained eight, while embedding, core, router, provider,
expert, join, and output work consumed exactly the five active rows.  The
provider admitted logical widths 2 through 8, `routes.n_tokens` was exactly 5,
and the active routes entered the existing `evaluate_layer()` boundary.  No
new scheduler, queue, padded route, or API was added.

The same binary (`031a273ffdee4358895953db977491298785db75583b4b746f1a4a653ade1a5f`)
ran union-off control followed by union-on candidate.  Both used the same
frozen 53-ID prompt, 36-ID perfect-token fixture, exact-scalar core, budget-24
P30, platform `performance`, AMDGPU `high`, and single-owner gfx1151 closure.
The only candidate changes were P40 device cache + layer epoch + exact macro
union + direct async upload.  Wide async join and union prefetch stayed off.

| Metric | Union off | Union on | Change |
|---|---:|---:|---:|
| True AR | 2.050628 tok/s | 2.071759 tok/s | +1.03% |
| Target wall / step | 2,437.707 ms | 2,412.850 ms | -1.02% |
| Expert wall / step | 1,207.009 ms | 1,186.179 ms | -1.73% |
| Physical direct bytes | 42,594,680,832 | 56,740,052,992 | +33.21% |
| Direct-I/O wall | 12.648801 s | 14.572278 s | +15.21% |
| Effective direct-I/O | 3.136 GiB/s | 3.626 GiB/s | +15.63% |
| Expert-graph device time | 2.205847 s | 1.079462 s | -51.06% |
| P45 H2D bytes | 67,645,471,184 | 67,648,381,392 | effectively flat |

Seven-step classified totals were:

| Stage | Union off | Union on |
|---|---:|---:|
| Exact scalar causal core | 7,915.626 ms | 7,908.327 ms |
| Experts | 8,449.064 ms | 8,303.253 ms |
| Join | 506.740 ms | 488.230 ms |
| Output | 154.975 ms | 155.527 ms |
| Other | 5.771 ms | 3.277 ms |
| Classified total | 17,032.176 ms | 16,858.610 ms |

The candidate completed all 644 macro unions: 8,527 groups, 51,645 physical
records, 12,824 routed expert rows, and 25,581 direct-async weight calls carrying
28,476,277,760 bytes.  P40 completed 203 requests with 34 hits, 169 cold fills,
zero fallbacks/aborts/evictions, and 1,126,035,456 H2D bytes.  P41 work fell from
33,218 to 20,191 completed expert layouts, and raw expert-graph device time was
halved.  However, loss/bypass of P30 record reuse raised physical service by
14,145,372,160 bytes and 1.923477 s.  The exposed expert interval therefore fell
only 145.811 ms over all seven steps.

The +15% gates were target wall at most 2,054.326 ms/step and expert wall at
most 823.214 ms/step.  The candidate missed them by 358.524 and 362.965 ms/step
respectively.  This is a performance NO-GO.

Exactness remained complete: output IDs, seven boundary-logit rows, terminal
logits, and logical traffic were byte identical.  Their hashes remained
`43938021...`, `382ca5a...`, and `5ae3a3f0...`; control and candidate log hashes
were `b68b1d50...` and `07240381...`.  Both arms reported 35 active/committed
state rows, 21 inactive physical slots, zero oracle fallbacks, zero P41
fallbacks/invalids, and identical logical provider bytes.  The 889-MiB short
fact state artifacts from the reused active-row implementation remain byte
identical (`12021f44...`).  A seven-boundary state dump was omitted from timed
arms because it would add several GiB of host readback outside the production
path.

This follow-up closes only width-5 macro union at q=4 with the current P30/P40
policies and current exact-scalar/IQ1_S kernels.  It does not close a policy
that retains P30 reuse while deduplicating the width-5 payload, q=7, or a faster
causal/expert kernel.  The temporary implementation remains in the shared
worktree solely for the separately bounded q=7 discriminator; it is not a
production promotion.

## Full-width q=7 grouped verifier

The next gate used seven perfect proposals plus the target seed, exactly
filling the existing physical width-eight replay.  It enabled the already
qualified Core8, MLA8, Tail8, Router8, P40 layer epoch, exact macro union, and
direct-async delivery paths.  There were no padded or inactive rows.

The source delta changed the temporary oracle proposal count from four to
seven and aligned the scalar evidence-capture cadence from five to eight.  It
did not add a scheduler, queue, cache, or provider.  The frozen closure was:

- worktree HEAD: `2655ac23783eaa830cba716f557881bd28f33b87`;
- experimental source diff SHA-256:
  `bc42f9749e1994ce1aa726bad410d50a30c2ec440eef38d4585193b5f0f80bc0`;
- binary SHA-256:
  `e2bf799173259b2ae453bffca14547649335c304ed9eed34a200e3c03b24f5ab`;
- same frozen model, 53-ID prompt, and target IDs as the earlier oracle;
- artifacts: `/home/duster/kimi-k3-deploy/k3-x8-q7-20260825/`.

### One-step exact state gate

Nine emitted tokens contain eight true transitions and one complete q=7
target step.

| Metric | Scalar | Full-width q=7 |
|---|---:|---:|
| Decode wall | 3.945 s | 2.162 s |
| True AR | 2.028059 tok/s | 3.700976 tok/s |
| Speedup | 1.000x | **1.825x** |
| Target step | n/a | 2,092.424 ms |

The candidate stage was 2,078.746 ms: 576.878 ms causal core, 1,362.395 ms
experts, 102.586 ms join, 35.307 ms output, and 1.580 ms other.  All eight
rows were active and committed; inactive rows and oracle fallbacks were zero.

Traffic, the eight-transition boundary logits, terminal logits, and the full
recurrent/MLA state dump were byte identical.  Their SHA-256 values were:

- traffic: `b450b9ea6e359d0192a5251d04a04bf766b46bbcf18833a48e82f57f272554f4`;
- boundary and terminal logits:
  `29ada024163ac03014c9c96db7874b484e2f86e65c310a001fb5574a9b6f4a2f`;
- state: `2e14919246ade1040c7d997f4b7c514ab940b8e674a00d374257df924500a6f9`.

### Thirty-two-transition performance gate

The no-state matched run used 33 emitted tokens, four full target steps, and
32 true transitions.

| Metric | Scalar | Full-width q=7 | Change |
|---|---:|---:|---:|
| Decode wall | 14.229 s | 8.614 s | -39.46% |
| True AR | 2.248906 tok/s | **3.714881 tok/s** | **1.6519x** |
| Sparse authoritative H2D bytes | 107,623,305,216 | 92,305,604,608 | -14.23% |
| Physical direct bytes | 40,588,738,560 | 51,914,293,248 | +27.91% |
| Direct-I/O wall | 13.581136 s | 12.126992 s | -10.71% |
| P45 H2D bytes | 107,736,940,288 | 67,647,822,288 | -37.21% |

The logical traffic TSVs themselves were byte identical because they record
the frozen calibrated route plan rather than H2D realization; both hash to
`92ff3ab7a9bb3fec4462f57af5cc17811bd5864d01e27521f16a2b00af05effb`.
Four boundary-logit rows and terminal logits were also byte identical, hashing
to `0ab319c6...` and `3cdf0dd9...` respectively.

The four candidate steps totaled 8,583.745 ms:

| Stage | Total | Per step |
|---|---:|---:|
| Grouped exact causal core | 2,309.968 ms | 577.492 ms |
| Experts | 5,714.929 ms | 1,428.732 ms |
| Join | 413.924 ms | 103.481 ms |
| Output | 141.111 ms | 35.278 ms |
| Other | 3.810 ms | 0.953 ms |
| Classified total | 8,583.742 ms | 2,145.936 ms |

The target interval was 8,610.677 ms, or 2,152.669 ms/step including replay
snapshot/commit and unclassified boundary work.  At perfect acceptance, 10
true tokens/s requires at most 800 ms/step, so the target still needs 2.69x
acceleration.  The expert interval alone is 1.79x over that entire budget.
At 90% independent proposal acceptance, expected committed tokens saturate
near 5.7 for q=7, making the approximate 10-token/s target-step budget about
570 ms before draft cost; proposing 80 tokens would mostly compute rejected
rows and is not earned by this result.

This GO establishes full-width grouped verification as the production
architecture to pursue.  It does not promote the temporary oracle code, prove
a trained-draft speedup, or meet 10 tokens/s.  The next target-side work is to
retain the grouped core while reducing expert physical service and graph wall;
q should then be selected from measured target cost and acceptance rather than
increased blindly.

## Existing P30 reuse inside q=7 macro union

The macro-union reader was bypassing the already-populated 16-GiB P30 cache.
The production fix reuses the existing borrowed-record API at that read site;
it adds no cache, queue, scheduler, or ownership policy.  The uncached direct
reader remains the fallback when borrowed records are disabled.

The one-step correctness gate was byte-identical for boundary logits, terminal
logits, logical provider traffic, and the full recurrent/MLA state image.  In
the matched long reversal, the two candidate arms produced 4.831074 and
4.828583 true AR tok/s, versus 3.389670 for the fresh frozen control.  Physical
reads fell deterministically from 51,914,293,248 to 40,146,558,976 bytes and
the mean expert interval fell from 1,639.514 to 943.002 ms/step.  The
candidate's boundary logits, terminal logits, and logical traffic remained
byte-identical to the frozen control.

This is a target-verifier ceiling, not deployable draft throughput.  The mean
candidate result is 4.829829 true AR tok/s before draft cost, so reducing the
remaining target cost and measuring the current draft remain necessary.  The
machine-readable closure is
`results/k3_x8_q7_p30_macro_reuse.json`.
