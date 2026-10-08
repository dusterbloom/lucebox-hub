# Knockout upper bound: hc_* hyper-connection family and quantize_q8_1 producer

Box: lucebox4 (gfx1151, `duster@100.115.193.112`), driver `~/qwen4exp-exact-stack/driver-ko`
built from a **separate** build dir `~/qwen4exp-exact-stack/build-ko` (fresh `cmake -S src/server
-B build-ko`, reconfigured from scratch with the same cache options as `build1151` — never `cp -a`'d,
per the `CMAKE_HOME_DIRECTORY` pitfall). Same model, same `run_stack.sh` args/env
(`<gguf> prompt.ids 256 8`, native mode, `MEASURE_FOLLOW=follow.ids` to keep the step count fixed
under output divergence), just pointed at `driver-ko`. This is a **timing upper bound only** —
knocked-out kernels leave their output buffers stale/garbage; token output diverges from the
real decode (confirmed: forced-follow kept 256 steps, no crash/NaN-trap, `rc=0` in every run).

## What changed (patch, gated off by default)

New header `server/deps/llama.cpp/ggml/src/ggml-cuda/luce-knockout.cuh`: two env-gated,
cached-via-`getenv` switches, read once per process (safe under HIP graph capture since the
decision is baked into the captured graph at capture time, same as every other `LUCE_QWEN_*` gate
in this tree):

- `LUCE_KO_HC=1` — skip the hc_* launch entirely (no kernel, no empty kernel; dst buffers stale).
  Applied at all 4 hc_* launch sites: `hc_down_inject_mixed`, `hc_upmix_row8_exact`,
  `hc_combine_norm_f32_b256` (decode path only, not the MoE-combine variant), `quantize_hc_lo_q8_1`.
- `LUCE_KO_EMPTY_HC=1` — same 4 sites, but launch a `<<<1,32>>>` no-op kernel instead of skipping,
  isolating launch+dispatch-gap cost from the kernels' own compute.
- `LUCE_KO_EMPTY_Q8=1` — same no-op substitution, applied only to the plain `quantize_q8_1`
  launch in `quantize_row_q8_1_cuda` (the 256/tok vanilla producer from the profile's top-15,
  distinct from `quantize_hc_lo_q8_1` which is already covered by the hc switches above).
  No skip-mode was added for this one (task only asked for the empty-kernel isolation here).
  The `quantize_mmq_q8_1`/`quantize_mmq_mxfp4` MoE-routing variants were left untouched — they
  don't appear in the profiled top-15 `quantize_q8_1` line and are out of scope.

All three flags default off; with none set, `driver-ko` reproduces the committed `driver-stack`
timing exactly (see baseline sanity check below).

## Commands

```
# fresh build dir, same cmake cache as build1151 (gen_build_ko_configure.sh + build-ko-configure.sh)
bash ~/qwen4exp-exact-stack/build-ko-configure.sh && cd ~/qwen4exp-exact-stack/build-ko && make -j32
bash ~/qwen4exp-exact-stack/build_driver_stack_ko.sh   # -> driver-ko

# baseline sanity (driver-ko, no knockout env)
TAG=ko_base REPS='1 2 3' bash ~/qwen4exp-exact-stack/run_stack_ko.sh

# A: skip hc_* entirely
XENV='LUCE_KO_HC=1' TAG=ko_hc_A REPS='1 2 3' bash ~/qwen4exp-exact-stack/run_stack_ko.sh

# B: empty-kernel in place of hc_*
XENV='LUCE_KO_EMPTY_HC=1' TAG=ko_hc_B REPS='1 2 3' bash ~/qwen4exp-exact-stack/run_stack_ko.sh

# C: empty-kernel in place of quantize_q8_1
XENV='LUCE_KO_EMPTY_Q8=1' TAG=ko_q8_C REPS='1 2 3' bash ~/qwen4exp-exact-stack/run_stack_ko.sh
```

Each `TAG`/arm run above is 3 fresh processes (reps), each process internally alternating
mode 0/1 across 8 measured runs (native driver schedule `{0,1,0,1,1,0,1,0}`); only `mode=1`
lines are used for the median (same convention as `PROFILE-38.md`). All 4 arms (baseline + A/B/C)
were run interleaved in this one session on the same binary/box.

## Results (median of mode=1 `ms_token=`, 15 values per arm = 3 reps x 5 mode=1 runs/rep)

| arm | env | median ms/token | delta vs baseline |
|---|---|---|---|
| baseline | (none, driver-ko) | 38.458 | — |
| A — skip hc_* | `LUCE_KO_HC=1` | 32.172 | **-6.286** |
| B — empty-kernel hc_* | `LUCE_KO_EMPTY_HC=1` | 33.094 | **-5.364** |
| C — empty-kernel quantize_q8_1 | `LUCE_KO_EMPTY_Q8=1` | 38.057 | **-0.401** |

No crashes, no NaN-traps, `rc=0` on every run; `GPU_EXEC_DONE rc=0` for all 12 processes.

## Interpretation

- **A - baseline = -6.29 ms/token**: total upper-bound cost of the hc_* family on the critical
  path if it were free (zero launches, zero work). This is somewhat larger than PROFILE-38's
  raw-kernel-time estimate (5.11 ms/token raw sum) because A also removes the per-launch
  host-dispatch gap for all 383 hc launches/token, which the raw-kernel-time sum doesn't capture.
- **B - baseline = -5.36 ms/token**: the launch/dispatch-gap part alone (same 383 launches/token,
  now doing nothing) — this is the part attributable to *having 383 extra kernel launches*,
  independent of what those kernels compute.
- **A - B = -0.92 ms/token**: the hc_* kernels' own compute work, isolated from launch overhead.
  Small relative to the launch-overhead component — consistent with PROFILE-38's framing that
  launch-overhead idle (the 5-20 µs gap bucket) and the hc_* family were "roughly tied" buckets;
  this knockout shows the hc_* family's bound (6.29 ms) is itself *mostly* launch-overhead
  (5.36 of the 6.29 ms, ~85%), with its actual arithmetic work contributing only ~0.92 ms/token.
- **C - baseline = -0.40 ms/token**: upper bound for eliminating quantize_q8_1's launch overhead
  alone (256 launches/token, ~2 µs each per PROFILE-38) — small and in line with the profile's
  534 µs/token raw-time estimate for this kernel.

These are upper bounds on a fused/eliminated-kernel win, not a projected real speedup — fusing
hc_down_inject_mixed + hc_upmix_row8_exact + hc_combine_norm_f32_b256 + quantize_hc_lo_q8_1 into
1-2 real kernels would still have to do the ~0.92 ms/token of actual work plus whatever reduced
launch count remains, so the realistic win is somewhere between 0.92 ms (B-style, work-only) and
6.29 ms (A-style, zero-cost fantasy) per token, closer to the 5-6 ms end if a fusion gets down to
1-2 launches instead of 383.
