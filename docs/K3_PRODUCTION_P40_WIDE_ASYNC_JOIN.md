# K3 production P40 wide asynchronous join

## Result

Commit `d1dec9fce11891f4f5af3af491d1d164deff7c35` makes the
qualified P40 wide path use the already-established P45 asynchronous expert
queue and P42 canonical device join. On the production M1024 profile it reduces
the expert interval from **200.358 to 154.994 seconds** and raises prefill from
**2.263908 to 2.516772 positions/s** (`1.1117x`). Total stage wall falls by
45.453 seconds.

This is an engineering **GO, DEFAULT OFF**. The implementation is narrowly
qualified and fails closed when its exact P40/P41/P42/P45/P46 envelope is not
present.

## Implementation and closure

The old exact-macro path evaluated cached experts synchronously and returned
each expert output to the host before joining. The winner retains each P40
lease through execution, feeds the cached device tensors directly into the
persistent evaluator, batches the expert graphs through P45, joins routes in
canonical order through P42, and reads one joined output per token. The P42
resident means also remove the corresponding host auxiliary-mean reads. No
approximate arithmetic or alternate route ordering was introduced.

- branch: `perf/k3-production-ponytail`
- parent P46 commit: `014aaf4a5ac4b5c11af2461c69e61b810391a517`
- documentation parent: `e17b76e7d5e4f0a65d754d01ba5818b8b341f715`
- winner: `d1dec9fce11891f4f5af3af491d1d164deff7c35`
- candidate source-file SHA-256:
  `97d2c945c09338f6167b9c4321aea91b5b1ed6a16be343a12cf4951040079fc7`
- clean measured smoke SHA-256:
  `9a4f281dd74d896bc91c57bbb3ce12514fed620fb9fcf81fa6e4d3faf5ff67cb`
- source patch SHA-256:
  `8d1c8554d796bc4dc9b37adf5898cae5fba50beb2f40820f4d608975301f54c3`
- switch: `DFLASH_KIMI_P40_WIDE_ASYNC_JOIN=0|1`, default `0`
- machine: Lucebox4, HIP/ROCm, physical GPU1 `gfx1151`, logical device 0,
  performance platform profile, single-owner topology, 18 CPU workers
- policy: calibrated96/fixed96, authoritative sidecars, direct-pread, P30
  16 GiB, P40 8 GiB with layer epochs, and P41/P42/P45/P46 enabled

M64 is a same-clean-binary A/B. M1024 compares the immediately preceding P46
winner with the new commit, so it is an adjacent-commit production comparison,
not a same-binary A/B.

## Same-binary M64 A/B

| Metric | P40 async join off, reversal | P40 async join on | Change |
| --- | ---: | ---: | ---: |
| stage total | 57.527 s | **47.801 s** | **-9.726 s** |
| prefill | 1.112204 pos/s | **1.338407 pos/s** | **`1.2034x`** |
| causal core | 14.365 s | 14.367 s | +0.003 s |
| experts | 42.170 s | **32.457 s** | **-9.713 s** |
| direct I/O | 28.281 s | 24.380 s | -3.901 s |
| physical bytes | 115,133,669,376 | 115,107,717,120 | -25,952,256 |

After subtracting the direct-I/O difference, the expert interval still saves
**5.812 seconds** (`90.8 ms/position`). The output IDs were identically
`5801,114820`, and both traffic files have SHA-256
`898a397272075995ba469d4db79526c7d099b7c6ed148516f9cf6d154dc1a9fd`.

The 25,952,256-byte physical difference is only **0.0225%**. The candidate
eliminates roughly 4.84 GB of logical auxiliary-mean reads by consuming the
P42 resident means; that changes P30 cache occupancy and leaves this small
secondary reduction in direct sidecar service. Selected-slab and
exact-fallback logical traffic is unchanged, as proven by the identical
traffic hash. This is not a changed expert-weight plan.

## Exact partial-commit gate

The existing `commit_n=3` probe was run with the switch off and on using the
same candidate binary. Both arms reported `exact=1`:

- committed-prefix full logits byte-identical;
- next-step full logits byte-identical;
- convolution, SSM, and MLA state hashes identical across every layer after
  commit 3 and after the next ordinary step;
- all mismatch indices `-1` (no mismatching layer);
- next-logits hash `7899592251675440270` in both arms;
- traffic SHA-256
  `d0436d265e1b04585481f46e33b45c84064ee27d7a65e42d16b4e1856f9b68f1`
  in both arms.

The control and candidate probe logs hash to
`7ea055aaf91ff12a559cb715d7d6522d135e1f26cf7ade9da10dbafc73672f0c`
and `dd39df5d3e0fd19bdf853097a6909426e2d317123c98d8aa64c67ed445d16da2`,
respectively.

## M1024 production result

| Term | Prior P46 | Wide async join | Change |
| --- | ---: | ---: | ---: |
| prefill wall | 452.315 s | **406.870 s** | **-45.445 s** |
| positions/s | 2.263908 | **2.516772** | **`1.1117x`** |
| stage total | 452.274679 s | **406.821367 s** | **-45.453312 s** |
| causal core | 236.624296 s | 236.874767 s | +0.250471 s |
| experts | 200.357841 s | **154.993998 s** | **-45.363843 s (`-22.6%`)** |
| join | 10.543054 s | 10.215118 s | -0.327936 s |
| output | 4.507782 s | 4.489633 s | -0.018149 s |
| physical bytes | 169,143,877,632 | 169,143,877,632 | identical |
| payload H2D | 168,212,803,584 | 168,212,803,584 | identical |

The winner's direct-I/O wall was 43.799217 seconds. P45 enqueued 1,244,278
expert jobs in a 110.009188-second device window; P42 performed 94,208
canonical join launches. P40 recorded 34,014 cold fills, zero evictions,
1,244,278 completions, zero aborts, and zero fallbacks. Process swap remained
zero. The traffic SHA-256 is
`c0dfd31205484facf903d1e55cd75ccf05338359583b1f9d76fbaa032b6cea15`,
exactly matching the preceding P46 run.

Correctness is deliberately bounded. The separate partial-commit probe is a
fresh byte-exact full-logit and per-layer-state gate. The M1024 smoke generated
no output token, so that run itself does not contribute a fresh full-logit or
state hash. Its identical logical traffic, physical bytes, H2D, P40 completion
count, and zero-fallback execution establish the production performance
closure without being mislabeled as a second exactness proof.

## Updated critical path

| M1024 winner term | Seconds | Share |
| --- | ---: | ---: |
| causal one-row core | 236.875 | 58.2% |
| experts | 154.994 | 38.1% |
| join | 10.215 | 2.5% |
| output | 4.490 | 1.1% |
| other | 0.248 | 0.1% |
| total | 406.821 | 100% |

The measured result is **2.516772 positions/s**. Even an impossible zero-cost
expert stage reaches only **4.066 positions/s**; an impossible zero-cost core
reaches only **6.025 positions/s**. Ten positions/s requires a 102.4-second
M1024 prefill, or another **304.421 seconds removed**—77.7% of the remaining
core-plus-expert wall. The causal core is now the largest term, but both core
and exact expert service must improve for 10 positions/s.

## Verdict

**ENGINEERING GO, DEFAULT OFF** for the fully qualified wide exact profile.
Retain the synchronous exact fallback. Promotion to default requires matched
long-fixture full-logit/state qualification and broad operational testing; it
does not require reopening the already-passed canonical-join exactness gate.
