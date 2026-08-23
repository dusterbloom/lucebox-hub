# K3 production exact macro-union service

## Result

The default-off production macro-union path raises the retained M1024 exact
prefill profile from **2.516772 to 2.954033 positions/s** (`1.1737x`). Stage
wall falls from 406.821 to 346.606 seconds, a **60.215-second** reduction.

This is an engineering **GO, DEFAULT OFF**. It is a coupled storage/compute
win, not a claim that K3 has reached 10 positions/s. The current internal
stage still needs 5.273 seconds to cross 3 positions/s and 244.206 seconds to
reach 10; the end-to-end smoke gaps are about 5.31 and 244.25 seconds.

## What changed

The implementation extends the existing P58 causal-micro / MoE-macro seam
without widening recurrent arithmetic:

1. Build the authoritative calibrated plan for every completed row.
2. Group selected work by expert and form one component-major slab union.
3. Read that immutable union in physical-offset order through the existing
   direct-I/O pool.
4. Upload each expert payload once and replay exact widths 1, 2, or 8.
5. Preserve every output in a provider-owned arena.
6. Batch calibrated-only canonical joins while retaining the original scalar
   P42/P45 path for any token with an exact fallback.
7. Bind terminal graph outputs directly into the owned arena, avoiding a
   redundant device copy while restoring gallocr bindings after each
   synchronous replay.

The path is enabled only by `DFLASH_KIMI_EXACT_MACRO_UNION=1`. Unsupported
quant tuples, shapes, ownership, cache policy, or join policy fail closed. The
ordinary width-one and non-union paths are unchanged.

## Measured progression

All M1024 arms use physical GPU1 `gfx1151`, the performance platform profile,
single-owner execution, 18 CPU workers, calibrated96/fixed96, P30 16 GiB,
P40 8 GiB with layer epochs, P41/P42/P45/P46, direct-pread, and hipBLASLt
disabled.

| M1024 production arm | Prefill | Stage | Core | Experts | Direct I/O |
| --- | ---: | ---: | ---: | ---: | ---: |
| retained wide async join | 2.516772 pos/s | 406.821 s | 236.875 s | 154.994 s | 43.799 s |
| ordered union executor | 2.882735 pos/s | 355.181 s | 236.659 s | 103.348 s | 29.150 s |
| batched clean-token join | 2.935830 pos/s | 348.750 s | 236.917 s | 96.782 s | 29.013 s |
| direct owned outputs | **2.954033 pos/s** | **346.606 s** | **236.756 s** | **94.747 s** | **29.045 s** |

The ordered union executor supplies the large win: 51.640 seconds. Batched
canonical joins save another 6.431 seconds. Direct terminal-output binding
saves another 2.144 seconds. The physical payload remains exactly
169,143,877,632 bytes in every arm.

At M1024, the union path executes 32,065 expert groups, 279,696 selected
records and 1,207,489 routed rows. Batched join converts 69,539 clean
token-layers into 92 layer launches; 24,669 token-layers with exact fallbacks
retain the scalar path. Direct binding removes all 1,207,489 union-output
scratch copies. The remaining 345,649 expert D2D copies belong to the scalar
join path for fallback-containing token-layers: 36,789 exact-fallback outputs
and 308,860 calibrated union outputs staged beside them.

Storage is no longer the immediate standalone target. The final union arm
serves the unchanged trace at about 5.40 GiB/s; its 29.045 seconds is close to
the roughly 28.23-second raw sequential floor. The dominant remaining term is
the 236.756-second causal core.

## Correctness boundary

The physical ordered-join sentinel compares raw float bits against both the
existing scalar GPU path and an independent CPU teacher. It covers transient
and resident rows, variable operation counts through the 208-operation limit,
signed zero, denormals, cancellation, NaN propagation, and a non-FMA rounding
trap.

The union sentinel covers production IQ1/IQ1/IQ1, IQ2/IQ2/IQ1 and
IQ1/IQ1/IQ2 tuples; widths 1, 2 and 8; graph reuse; and padded tails. The live
M8 ladder retained output IDs `48430,10867` and text ` Ryzen`. The M64 ladder
retained output IDs `5801,114820` and text ` exact causal`. All four compared
M64 ladder arms and all compared M1024 arms hash to
`c0dfd31205484facf903d1e55cd75ccf05338359583b1f9d76fbaa032b6cea15`
at M1024 and
`898a397272075995ba469d4db79526c7d099b7c6ed148516f9cf6d154dc1a9fd`
at M64.

The M1024 fixture generated zero output tokens, so it does not provide a fresh
full-logit or recurrent-state hash. It is therefore incorrect to call that
specific long run a new full-logit/state qualification. Exactness comes from
the bitwise component sentinels, the live M8/M64 output checks, and the
already-qualified P58/P42 semantic seam. Promotion to default still requires
a matched long-fixture full-logit/state gate and broad operational coverage.

## Source and evidence closure

- branch: `perf/k3-production-ponytail`
- parent production winner: `8f5fc168`
- union bridge commits: `106c1f0f` through `3eba362c`
- batched join commits: `60e3dae5`, `cca571cd`
- direct-output commit: `7451559b`
- final measured smoke SHA-256:
  `19d81ce8f0c8df1030cf5285090fb72ffed8e36cf389fd21f69eac54855908a1`
- final provider source SHA-256:
  `396026b5b8622d4e74d99f2bece0bdddc61372135b20625f0be548f21f7300a2`
- M1024 control/union/macro/direct log SHA-256:
  `9940820ee9b56b10f025fc471d63bc77fcc2085cc6cf4baf228a7fbdebb7bfe3`,
  `65d1307e7d60db6ec95107fdd7a98f73cae102c263323ffa08218dcb87f52129`,
  `fe96624312f2be2f12219b728b5a0913915f5ebec1e70d083f41e26280e82b99`,
  `49ea8f016263997f80742c1146d2e679a377476b2956068798ea22d515765bac`
- machine-readable record: `results/k3_production_exact_macro_union.json`

Raw logs and traffic files remain on Lucebox4 under `/tmp/k3-union-*`.
Process swap was zero in every retained run.

## Verdict and next dependency

**ENGINEERING GO, DEFAULT OFF.** Keep the exact fallback and the narrow
production envelope. Direct widths 16/32 are not promoted: they dispatch
gate/up from exact MMVQ to differently-associated MMQ on gfx1151. A separate
analytical, unmeasured call-overhead estimate puts an exact tiled version at
only about 1–2 seconds, so no wider-width code was added.

The next bounded systems change is to overlap the already-determined union
reads with the still-running same-layer causal token loop. Expert GPU work and
canonical join must remain after the complete authoritative route set; this
does not reopen P62's closed storage-to-device overlap lane.
