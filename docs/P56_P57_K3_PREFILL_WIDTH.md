# P56/P57 — K3 prefill census and exact width-two activation

## Verdict

**P56 MEASURED THE PREFILL BOUNDARY. P57 RECOVERED BYTE-EXACT WIDTH-TWO
PREFILL, BUT MISSED THE 1.8× ACTIVATION GATE.**

On Lucebox4, P56 separates prefill from decode over the frozen broad-12 suite.
It records 552 prompt positions, 223,579 compact expert jobs and
291,245,891,584 physical direct-read bytes during prefill. The trace-enabled
diagnostic takes 375.560925302 seconds, or 1.469801470 positions/s. This is not
the canonical throughput result: the trace itself adds overhead, so P55's
1.628035098 positions/s remains the registered cold-prefill baseline and the
10× target remains 16.28035098 positions/s.

P57 activates bounded width-two prefill. The first implementation produced
correct first-chunk logits but diverged at the first row after that chunk. The
Wafer/AITER K3 discussion prompted a padding discriminator. On this unsharded
96-head GGML path the problem was not AITER's 12-to-16 per-rank head padding;
it was a related causal reduction-width issue: an earlier query with true
causal length two was aggregated over a stored width-three value matrix. The
future probability was exactly zero, but the longer HIP reduction changed
floating-point association.

The exact repair has two parts:

1. For multi-token MLA under the existing serial-core qualification mode,
   compute each `V × probabilities` row over its true causal length.
2. Capture recurrent KDA state for each prompt chunk and commit it through the
   already-qualified ReplaySSM path. Direct batched state mutation was the
   reason the next chunk diverged.

The final cleaned source is commit `50399e8d6011242841bf3f53be640986b6ff3dd3`.
Temporary internal MLA tracing was removed before qualification; relative to
the initial P57 activation boundary, the retained implementation is 44 added
and 4 deleted production lines across the backend and graph.

## P56 broad-12 census

| measure | prefill | decode |
|---|---:|---:|
| positions | 552 | 129 |
| compact jobs | 223,579 | 51,429 |
| physical direct reads | 291,245,891,584 B | 44,258,549,760 B |
| logical provider bytes | 724,784,406,528 B | 167,799,668,736 B |
| routed preparation | 223.575 ms/position | 224.032 ms/position |
| experts | 427.751 ms/position | 341.918 ms/position |
| join | 17.751 ms/position | 18.035 ms/position |
| fallback / invalid / async abort | 0 / 0 / 0 | 0 / 0 / 0 |

At macro width 32, trace replay reduces deduplicated/coalesced request count
from 187,654 to 150,340 (19.88%) but physical bytes only from
290,115,694,592 to 289,687,969,792 (0.15%). Width 64 reduces requests 26.37%
and bytes 3.25%. Therefore prompt batching has a real submission opportunity,
but it does not by itself solve storage density.

## P57 adjacent exact fact

The matched width-one and width-two arms use the same binary, one-line fact
fixture, P41 compact executor and all-layer calibrated96 policy. P42/P45/P46
and the later one-token GPU1 chain are disabled because they are currently
one-token-only.

| measure | width 1 | exact width 2 | change |
|---|---:|---:|---:|
| prompt positions | 34 | 34 | — |
| forward calls | 34 | 17 | −50.0% |
| prefill | 26.415983065 s | 22.718395184 s | −13.998% |
| positions/s | 1.287099553 | 1.496584584 | **+16.276%** |
| physical direct reads | 18,606,039,040 B | 18,627,551,232 B | +0.116% |
| direct-I/O service | 9.509405758 s | 8.297712481 s | −12.742% |
| compact pack | 0.109641772 s | 0.107173868 s | −2.25% |
| expert graph | 4.474600095 s | 4.300139685 s | −3.90% |
| expert readback | 0.292900372 s | 0.282468983 s | −3.56% |
| compact jobs | 14,495 | 14,495 | identical |

Both arms produce the same Tokyo tokens/text, full logits SHA-256
`cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`
and logical traffic SHA-256
`e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`.
P41 completes 17,917/17,917 jobs with zero fallback or invalid jobs.

The gain is real for the matched diagnostic path, but it is below the
preregistered 1.8× P57 gate and the absolute rate does not exceed the optimized
P55 broad prefill baseline. Most observed wall improvement also coincides with
lower direct-I/O service time rather than a proportional reduction in
arithmetic or bytes. P57 therefore remains an exact research substrate, not a
production default and not a broad throughput claim.

## Final gates

The cleaned commit passes on Lucebox4:

- width-two S0: logits and argmax byte-equal; recurrent, convolution, SSM and
  MLA hashes equal; zero numerical error; committed verifier ratio 1.121147×;
- provider and ordered-join tests on HIP;
- common MoE stream tests, 2/2 on each GPU;
- sparse-K MMVQ, 68/68 on `gfx1201` and 68/68 on `gfx1151`;
- local CUDA Release build and core-placement test.

The final S0 root is
`/home/duster/kimi-k3-deploy/p57-s0-final-50399e8-20260820`. The matched fact
roots are
`/home/duster/kimi-k3-deploy/p57-fact-width1-control-0b6742e-20260820` and
`/home/duster/kimi-k3-deploy/p57-fact-width2-replay2-0b6742e-20260820`.

## Decision

Keep the exact width-two machinery default-off. Do not run broad P57 because
the activation gate failed. Multi-token prediction is not a substitute for
prefill: speculative decoding begins only after prompt ingestion. The next
prefill decision should be a bounded P58 experiment that separates exact KDA
microtile width from the larger position-wise MoE width. P59 grouped compact
execution is justified only if P58 exposes enough exact rows per routed layer
and the measured all-resident/IO rooflines remain above the target.
