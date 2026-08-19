# P55 — GPU1 device chain broad-12 qualification

## Verdict

**BYTE-EXACT BROAD-12 GO AT 1.856934565 TRUE AUTOREGRESSIVE TRANSITIONS/S,
WITHOUT SPECULATIVE DECODE.**

The orderly current-main branch now has the longer evidence that the P47–P51
campaign was missing. On Lucebox4, the exact P51 topology completes all twelve
frozen prompts and 129 true decode transitions in 69.469329936 seconds. This is
1.856934565/s, 33.63% above the matched historical P40-off HIP broad reference
and only 6.76% below the best 1.991574475/s short fact.

This is not a measured 2/s result and it is not steady-state certification. It
does show that the near-2 short fact was not merely a one-prompt accident.

## Frozen topology

- Lucebox4 ROCm/HIP, with the full 37.56-GiB core, recurrent state, resident
  means and canonical join on GPU1 (`gfx1151`).
- `all-layers-calibrated96`, authoritative sidecars, P27 direct-pinned compact
  delivery, P41 compact execution, P42 ordered device join, P45 asynchronous
  compact queue and P46 persistent routed preparation.
- Qualified 16-GiB P30 host cache; P40, P43b, P52 and P53 disabled.
- Thinking off, 12 frozen prompts, at most 24 emitted tokens, no speculative
  decode.
- Source `2ce8a28b654473257d3d0d4acddd87da79ddefaa`; runner SHA-256
  `e35bb15d822690c9583e976b6e930c43056b6850525d0cccd6d11421386592a7`.

Before the model gate, the exact branch passed the provider sentinel, ordered
join, common MoE stream tests and sparse-K 68/68 gate on both Lucebox4 GPUs.

## Runtime

| measure | P55 broad-12 |
|---|---:|
| prompts | 12 |
| prompt positions | 552 |
| emitted tokens | 141 |
| true decode transitions | 129 |
| prefill | 339.059029324 s |
| prompt positions/s | 1.628035098 |
| decode | 69.469329936 s |
| **true AR transitions/s** | **1.856934565** |
| wall | 444.405873081 s |

Canonical throughput uses
`sum(max(0, emitted_i - 1)) / sum(decode_seconds_i)`. The runner-style
`141 / decode` value is not used because each prompt's first emitted token is
produced by its final prefill logit.

Mean decode-stage timing is 538.359829 ms: 223.344364 ms routed preparation,
286.940264 ms experts, 17.171287 ms join and 4.461248 ms output. Reaching an
honest 2/s now requires at least 38.36 ms/transition of additional reduction.
Reaching 4–5/s still requires a joint expert and routed-preparation redesign;
neither batching alone nor a KDA-only kernel can supply that whole gain.

## Exactness and quality

All twelve prompt-token sequences, emitted-token sequences, texts and full
logits are byte-identical to the P40-off HIP broad reference. Concatenating the
twelve logits traces in fixture order yields SHA-256
`f761bb7e9ffb614ce48fb5fc02c8cd2da44ff092ecb1d7d9bd318d14cd4fd06b`
for both runs. Logical traffic is also byte-identical at SHA-256
`54b76f99439d90ecb35d033c12dd1d1f219e30007248aaf77f913fc69d76f383`.

The fixed-cap task result therefore remains the known HIP baseline: 11/12. The
`code-function` answer ends after `x %` at the 24-token cap, before the required
`2`; the GPU1 device chain introduces no quality change. Exact extraction still
returns `LIME-742` and `QUARTZ-918` through its byte-identical reference output.

## Lifecycle and resource evidence

- P41: 275,008 attempted and completed; zero fallback and zero invalid.
- P45: 275,008 jobs/graph enqueues under 62,652 layer flushes; zero aborts.
- P42: 275,008 expert D2D copies and 62,652 joins/output publications.
- P46: 68 persistent graphs and 46,308 executions.
- Resident means: zero hot reads and zero hot H2D.
- Physical direct reads: 335,504,441,344 bytes; logical provider traffic:
  892,584,075,264 bytes, or 1.220680909 GiB/model position.
- Peak GPU1 memory: 61,174.211 MiB; process swap: zero; sampled GPU1 energy:
  29,250.443 J.

The full machine-readable result is
`results/k3_p55_gpu1_device_chain_broad12_runtime.json`; the durable remote
root is
`/home/duster/kimi-k3-deploy/p51-orderly-broad12-2ce8a28-20260819`.

## Decision

P47–P51 remain default-off but are now qualified across both the exact short
fact and the frozen broad-12 gate. Keep the full stateful core and canonical
ordering domain on GPU1. The next bounded work should first recover the final
38.36 ms needed for a measured 2/s, then combine expert-side reduction with a
Kimi-specific wave32 KDA/routed-preparation improvement toward 4–5/s before
speculative decode.
