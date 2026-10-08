# hc_upmix_row8_exact / hc_down_inject_mixed fast variants — result: regression, not shipped

Box: lucebox4 (gfx1151 Strix Halo, 40 CU, `HIP_VISIBLE_DEVICES=1` -- note the box also
exposes a second GPU agent, `gfx1201`/64 CU, as HIP device 0; every invocation in this
doc uses `HIP_VISIBLE_DEVICES=1` to target the real gfx1151 iGPU, matching `run_stack.sh`).
Tree `~/qwen4exp-hc-gemv/src` (fresh clone of `fork/qwen4exp-hc-gemv`, branched from
`qwen4exp-exact-stack`@8ffa5afb), fresh `cmake` configure in `~/qwen4exp-hc-gemv/build1151`
(cache options copied from `~/qwen4exp-exact-stack/build1151/CMakeCache.txt`, never `cp -a`'d).
Model/prompt: same as `PROFILE-38.md` (`Qwen3.8-Flash-Next-UD-Q4_K_XL`, `prompt.ids` 7235
tokens, `follow.ids`).

## 0. Summary

Both `LUCE_QWEN_HC_GEMV_FAST` variants from the previous commit (50486a24) are a **net
regression** in the real decode path, not an improvement:

| kernel | isolated device avg (rocprofv3, census n=64) | delta | full-decode effect (combined) |
|---|---|---|---|
| `hc_upmix_row8_exact` -> `_fast` (launch_bounds 256,1 -> 256,2) | 20.903 us -> 21.124 us | **+0.221 us (+1.1%)** | included in +0.6 ms/token below |
| `hc_down_inject_mixed` -> `_fast` (44-block -> 40-block wave fusion) | 20.936 us -> 27.537 us | **+6.601 us (+31.5%)** | included in +0.6 ms/token below |

Full-decode (native mode, 256 tok x 8 reps, fresh process per rep, `HIP_VISIBLE_DEVICES=1`):
OFF (baseline) mode=1 median ~38.43-38.53 ms/token across 3 reps; ON (`LUCE_QWEN_HC_GEMV_FAST=1`,
both kernels) mode=1 median ~39.06 ms/token across 2 reps -- a **+0.6 ms/token regression**,
not the hoped-for recovery. `free -g` before starting: 125G total / 64G used / 23G free / 60G
available -- OFF stayed at 38.4-38.5 ms/token (well under the 39.5 ms/token compaction
trigger), so the box did not need compaction; the regression is real, not a noisy-box artifact.

**Decision: do not enable `LUCE_QWEN_HC_GEMV_FAST` / `_UPMIX` / `_DOWNINJECT` in production.**
Default path (all three flags unset) is unchanged and remains the shipped behavior. Bit-identity
of the fast kernels (when explicitly enabled) still holds -- see §1 -- so there is no
correctness risk in leaving the gated code in the tree, but there is no performance reason
to ever set these flags. Per the updated bar (default byte-identical, gated fast path needs
matching `[measure_tokens]` + tolerance-bound logits, reordering allowed) this variant family
simply doesn't clear the speed bar, so the relaxed-order (`LUCE_QWEN_HC_GEMV_FAST=2`) idea was
not attempted -- no concrete reason it would beat the old kernel once the geometry-only change
already lost.

## 1. Bit-identity (correctness gate, unchanged from prior commit)

`bench/exact/test_hc_gemv_bitexact.cpp`, built against this tree and run with
`HIP_VISIBLE_DEVICES=1` (device 0 is the unrelated gfx1201 GPU and segfaults on a gfx1151
codeobj -- not a box problem, just the wrong HIP device index):

```
[upmix-random] mixed[2560] BIT-IDENTICAL   (x5)
[down-inject-random] down_dst[320] BIT-IDENTICAL  inject_dst[4] BIT-IDENTICAL   (x5)
ALL PASS
```

10/10 cases bit-identical, as designed (geometry-only changes, same per-output accumulation
order). This gate passing is necessary but not sufficient -- it does not predict speed.

## 2. Full-decode timing (native mode, `LUCE_QWEN_HC_GEMV_FAST=1`, both kernels)

Driver `driver-hc-gemv` built against this tree (`driver_shared_epilogue.cpp`, same harness
as `driver-stack`), same env/args as `run_stack.sh`, fresh process per rep, interleaved
OFF/ON, `HIP_VISIBLE_DEVICES=1`:

```
free -g (before starting): total=125G used=64G free=23G available=60G
```

| arm | rep | mode=1 values (ms/token) | mode=1 median |
|---|---|---|---|
| OFF | 1 | 38.425949, 38.431159, 38.434524, 38.456504 | 38.4328 |
| ON  | 1 | 39.058526, 39.074629, 39.052743, 39.391381 | 39.0666 |
| OFF | 2 | 38.534481, 38.467483, 38.520302, 38.501332 | 38.5108 |
| ON  | 2 | 40.244818*, 39.066485, 39.044625, 39.059977 | 39.0632 |
| OFF | 3 | 38.501430, 38.494402, 38.549764, 39.954173* | 38.5256 |

\* one outlier per ON-2 and OFF-3 (40.24 / 39.95), ~1.5-2x the typical 5-20us launch-overhead
gap seen in `PROFILE-38.md` -- consistent with the rare rocprofv3-adjacent host-dispatch
hiccups documented there, not a systematic effect (every other mode=1 value in that rep is
tight around the arm's median).

OFF medians (3 reps): 38.4328, 38.5108, 38.5256 ms/token -- matches the committed 38.41-38.58
ms/token baseline in `PROFILE-38.md`/`KNOCKOUT-HC.md`, confirms no box drift.
ON medians (2 reps): 39.0666, 39.0632 ms/token -- consistently **+0.58 ms/token slower** than
OFF, not faster. This single number already falsifies both kernels' "recover ~1.2ms/token"
hypothesis from `HC-AUDIT.md`; the per-kernel split below explains why.

## 3. Per-kernel wiring check (rocprofv3 kernel-trace, census mode n=64/reps=1)

Confirms the gate cleanly switches kernels with no mixing (no baseline+fast overlap):

| arm | `hc_upmix_row8_exact` | `hc_upmix_row8_exact_fast` | `hc_down_inject_mixed` | `hc_down_inject_mixed_fast` |
|---|---|---|---|---|
| base (`v2base`) | 6144 | 0 | 6080 | 0 |
| upmix-only (`LUCE_QWEN_HC_GEMV_FAST_UPMIX=1`) | 0 | 6144 | 6080 | 0 |
| downinject-only (`LUCE_QWEN_HC_GEMV_FAST_DOWNINJECT=1`) | 6144 | 0 | 0 | 6080 |

(counts are launches over the 64-token census window; 6144/64=96 and 6080/64=95 launches/token,
matching `PROFILE-38.md`'s per-token launch counts exactly).

New sub-flags (`luce-knockout.cuh`) added this session so each kernel's change can be timed
independently; both OR with the existing combined `LUCE_QWEN_HC_GEMV_FAST=1` for backward
compatibility with the already-run full-decode numbers in §2:

```
luce_hc_gemv_fast_upmix()      := LUCE_QWEN_HC_GEMV_FAST_UPMIX=1      || LUCE_QWEN_HC_GEMV_FAST=1
luce_hc_gemv_fast_downinject() := LUCE_QWEN_HC_GEMV_FAST_DOWNINJECT=1 || LUCE_QWEN_HC_GEMV_FAST=1
```

## 4. Per-kernel isolated device timing (rocprofv3, (End_Timestamp - Start_Timestamp) per
   dispatch, census mode n=64/reps=1, averaged over all launches in the window)

| arm | kernel | n launches | avg us/launch |
|---|---|---|---|
| v2base | `hc_upmix_row8_exact` | 6144 | 20.903 |
| v2base | `hc_down_inject_mixed` | 6080 | 20.936 |
| v2upmix | `hc_upmix_row8_exact_fast` | 6144 | **21.124** |
| v2upmix | `hc_down_inject_mixed` (unaffected, control) | 6080 | 20.932 |
| v2downinject | `hc_down_inject_mixed_fast` | 6080 | **27.537** |
| v2downinject | `hc_upmix_row8_exact` (unaffected, control) | 6144 | 20.003 |

(control-kernel numbers in each isolated arm track the base arm's number within ~1us,
confirming the arms are otherwise comparable and the delta is attributable to the changed
kernel, not session-to-session drift.)

**(a) upmix (`__launch_bounds__(256,1)` -> `(256,2)`): +0.221 us/launch (+1.1%), a wash, not
a win.** ISA-level check (`-Rpass-analysis=kernel-resource-usage` against the same compile
command CMake uses for `ggml-hip`) shows **zero codegen difference**: both the baseline and
the fast variant report `VGPRs: 18`, `SGPRs Spill: 0`, `VGPRs Spill: 0`, `Occupancy [waves/SIMD]: 16`
-- identical in every resource field. The `(256,2)` hint did not change the compiler's
register allocation at all (18 VGPRs already fits far more than 2 blocks/CU on this
hardware's register file, so the hint was a no-op instruction-wise); the +0.2us is launch-to-
launch measurement noise, not a resource-driven effect. **Prime suspect ("register spills
cutting the unroll") is ruled out by direct ISA inspection.**

**(b) down_inject (44-block two-phase -> 40-block fused-wave): +6.601 us/launch (+31.5%), a
real regression.** ISA check: baseline `VGPRs: 16`, fast `VGPRs: 18` (+2, from carrying the
down-projection's live state across into the folded inject phase) but **occupancy is
unchanged at `16 waves/SIMD` for both** and **neither variant spills** (`VGPRs Spill: 0`,
`SGPRs Spill: 0` in both). So this is *not* a spill/occupancy regression either -- the two
extra VGPRs did not cost any occupancy headroom. The regression is a real increase in
critical-path *work*: folding the inject phase into blocks 0..3 makes those 4 blocks execute
the down-projection loop **and then** the inject loop (with its own two `__syncthreads`
round trips) serially, back-to-back, inside one block's lifetime. The original hypothesis
(`HC-AUDIT.md`/this family's own prior analysis: "4 idle-wave-tail blocks cost ~6.4us of
pure dispatch latency with 36/40 CUs idle") turns out to be wrong on real hardware: the
baseline's tiny second wave is in practice cheap (the 4 inject blocks dispatch and retire
quickly once a CU frees, overlapped with HIP-graph-replay's own per-launch overhead), while
*lengthening the critical-path blocks'* instruction stream by a full second reduction phase
costs more (+6.6us) than the eliminated wave-tail ever did. **Root cause of the regression:
the fix traded a cheap, mostly-hidden extra wave for a more expensive serialized extension
of the already-longest-running blocks -- the opposite of the intended effect.**

## 5. Decision

Neither half is faster. (a) upmix is a statistical wash with no codegen difference (not
worth the sub-flag's complexity in production). (b) down_inject is a clear, ISA-confirmed-
non-spill regression. **Keep the default (unflagged) kernels in production.** No reordered
/ relaxed-accumulation-order variant (`LUCE_QWEN_HC_GEMV_FAST=2`) was attempted, per explicit
scope cut -- there is no concrete reason to believe reordering would reverse a regression
whose root cause (critical-path lengthening from serialized extra work, not launch-count or
register pressure) is unrelated to accumulation order.

The hc_combine_norm_f32_b256 -> `_fast` gamma-prefetch variant (commit 8ffa5afb,
`LUCE_QWEN_HC_CN_FAST`) is unaffected by this result and remains a separate, independently
gated change (not re-evaluated in this session).

## Files

- `bench/exact/test_hc_gemv_bitexact.cpp` -- bit-identity + host-side microbench (committed
  previously, unchanged this session).
- `server/deps/llama.cpp/ggml/src/ggml-cuda/luce-knockout.cuh` -- added
  `luce_hc_gemv_fast_upmix()` / `luce_hc_gemv_fast_downinject()` sub-flags this session.
- `server/deps/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` -- production launch sites switched to
  the per-kernel sub-flag gates (`luce_hc_gemv_fast_upmix()` / `_downinject()`) so each
  kernel's change can be timed independently; `hc_upmix_row8_exact_fast` /
  `hc_down_inject_mixed_fast` kernel bodies unchanged from commit 50486a24.
- On lucebox4 (not committed, reproducible from the commands in this doc): `~/qwen4exp-hc-gemv/`
  tree, `driver-hc-gemv`, `prof/kt_v2base`, `prof/kt_v2upmix`, `prof/kt_v2downinject`
  (rocprofv3 kernel-trace CSVs backing §3/§4), `logs/{off,on}{1,2,3}_rep1.log` (full-decode
  backing §2).
