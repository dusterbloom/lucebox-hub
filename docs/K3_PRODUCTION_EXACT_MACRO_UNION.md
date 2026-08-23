# K3 production exact macro-union service

## Result

The default-off production macro-union path raises the retained M1024 exact
prefill profile from **2.516772 to 3.079098 positions/s** (`1.2234x`). Stage
wall falls from 406.821 to 332.522 seconds, a **74.300-second** reduction.

This is an engineering **GO, DEFAULT OFF**. It is a coupled storage/compute
win, not a claim that K3 has reached 10 positions/s. The current run clears
3 positions/s by 8.77 seconds of prefill wall. Reaching 10 still requires
about 230.12 seconds to be removed from the internal stage.

## What changed

The implementation extends the existing P58 causal-micro / MoE-macro seam
without widening recurrent arithmetic:

1. Build the authoritative calibrated plan for every completed row.
2. Group selected work by expert and form one component-major slab union.
3. Read that immutable union in physical-offset order within each service
   batch or 64-row prefetch epoch through the existing direct-I/O pool.
4. Upload each expert payload once and replay exact widths 1, 2, or 8.
5. Preserve every output in a provider-owned arena.
6. Batch calibrated-only canonical joins while retaining the original scalar
   P42/P45 path for any token with an exact fallback.
7. Bind terminal graph outputs directly into the owned arena, avoiding a
   redundant device copy while restoring gallocr bindings after each
   synchronous replay.
8. At M1024 only, observe each completed exact route row and submit newly
   discovered immutable union records every 64 rows. The authoritative plan
   is recomputed and validated before expert execution and canonical join.

The union path is enabled by `DFLASH_KIMI_EXACT_MACRO_UNION=1`; causal/read
overlap additionally requires `DFLASH_KIMI_EXACT_MACRO_UNION_PREFETCH=1`.
Both remain default off. Unsupported
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
| causal/read overlap control | 2.948901 pos/s | 347.211 s | 236.960 s | 95.009 s | 28.865 s |
| causal/read overlap | **3.079098 pos/s** | **332.522 s** | **239.804 s** | **78.271 s** | 34.270 s\* |

\*Under overlap, direct-I/O time is physical-service wall and is not additive
with stage wall.

The ordered union executor supplies the large win: 51.640 seconds. Batched
canonical joins save another 6.431 seconds. Direct terminal-output binding
saves another 2.144 seconds. The provider-counted direct physical payload
remains exactly 169,143,877,632 bytes in every arm.

At M1024, the union path executes 32,065 expert groups, 279,696 selected
records and 1,207,489 routed rows. Batched join converts 69,539 clean
token-layers into 92 layer launches; 24,669 token-layers with exact fallbacks
retain the scalar path. Direct binding removes all 1,207,489 union-output
scratch copies. The remaining 345,649 expert D2D copies belong to the scalar
join path for fallback-containing token-layers: 36,789 exact-fallback outputs
and 308,860 calibrated union outputs staged beside them.

The same-binary overlap A/B saves 14.689 seconds of stage wall (`1.0442x`):
experts fall by 16.738 seconds while causal core rises by 2.844 seconds under
contention. Provider direct physical bytes remain exactly 169,143,877,632.
The prefetched stable union is 156,125,675,520 bytes; the remaining
13,018,202,112 bytes serve exact fallbacks. Only 153.598 ms of the 30.833-s
stable-union service envelope remains as blocking wait, so 99.50% is hidden.

This is an overlap win, not faster storage. Epoch service slows total provider
throughput from about 5.457 to 4.597 GiB/s, and the new serial validation/copy
tail costs 6.178 seconds. Whole-process file input also rises by 4.726 GiB and
maximum RSS by about 1.84 GiB. Process swap remains zero. The dominant term is
now the 239.804-second causal core.

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
The overlap A/B's identical traffic TSV proves logical route/byte equality,
not identical physical request order. Its `dde49a22156d1df5` fingerprint is
the epoch submission plan, not concurrent device-arrival order.

## Source and evidence closure

- branch: `perf/k3-production-ponytail`
- parent production winner: `8f5fc168`
- union bridge commits: `106c1f0f` through `3eba362c`
- batched join commits: `60e3dae5`, `cca571cd`
- direct-output commit: `7451559b`
- overlap commits: `1d49dfc8`, `52b2f02b`, `67e0b982`
- qualified M1024 source commit, integrated-equivalent commit, and provider
  SHA-256:
  `b4776f3dcd263ef3cccb4f419dae21e5785c34d0`,
  `52b2f02b0982916584ae382ac0b06e92d4f2660e`,
  `15d7e339c72de2038f67ac2ffe1ab7ba1b456aa0102b06d8a0ea3b88c6cb50d6`
- qualified M1024 smoke SHA-256:
  `862545e4f8136c1216a9de39c48321fdd2fb5d526daaa1e9ba290ff21f9c7278`
- final trimmed smoke SHA-256 (M64 requalified):
  `ab995f2defd4a2b8471b6db7672423266e0a7bd4907b13246a2228b4a271860d`
- final provider source SHA-256:
  `7a1317b4cfd0e32fa0f821aa5c5977ae961c675ee58c2569cf4bf9dae071f2e6`
- M1024 control/union/macro/direct log SHA-256:
  `9940820ee9b56b10f025fc471d63bc77fcc2085cc6cf4baf228a7fbdebb7bfe3`,
  `65d1307e7d60db6ec95107fdd7a98f73cae102c263323ffa08218dcb87f52129`,
  `fe96624312f2be2f12219b728b5a0913915f5ebec1e70d083f41e26280e82b99`,
  `49ea8f016263997f80742c1146d2e679a377476b2956068798ea22d515765bac`
- overlap control/candidate log SHA-256:
  `9a15ea1ca1b2f5f888678b373a32ed715f73360c5b314e342ede270f11f40c6b`,
  `d8a82128ed8f5806b00e8016769b2c600cc530de5eb0da938e01a545a71afd68`
- machine-readable record: `results/k3_production_exact_macro_union.json`

Raw logs, traffic files, and the qualified binary are preserved on Lucebox4
under `~/kimi-k3-deploy/k3-production-union-prefetch-20260824/`.
Process swap was zero in every retained run.

## Verdict and next dependency

**ENGINEERING GO, DEFAULT OFF.** Keep the exact fallback and the narrow
production envelope. Direct widths 16/32 are not promoted: they dispatch
gate/up from exact MMVQ to differently-associated MMQ on gfx1151. A separate
analytical, unmeasured call-overhead estimate puts an exact tiled version at
only about 1–2 seconds, so no wider-width code was added.

The overlap objective is complete. The next large prize is the measured
239.804-second causal core. The tested M64 HIP graph replay arm is a NO-GO: it
changed core wall from 14,372.985 to 14,417.792 ms despite 10,918 replays. Use
the existing ROCm profiler to name the largest remaining projection/conv/gate
kernel before editing it. The 6.178-second serial prefetch packing tail is a
smaller, bounded cleanup opportunity, not the route to 10 positions/s.
