# P61A — K3 recurrent split census

## Verdict

**DECISION INCONCLUSIVE; QUALIFICATION NO-GO.**

P61A measured cumulative, byte-exact preparation subgraphs for all 68 recurrent
routed layers on Lucebox4's physical GPU1 (`gfx1151`). The central whole-model
median assigns **104.108312 ms/transition** to the paired KDA interval
`F2 - F1`, and a hypothetical **1.8x speedup of that entire interval** has a
zero-overhead optimistic saving of **46.270361 ms/transition**. That falls
between the registered 38.521937-ms hard floor and 50-ms continue threshold, so
the KDA decision is **inconclusive**.

This was not a qualified timing result. Large same-run outliers fail every
registered stability/reconstruction check, and the production-adjacent sample
misses the frozen P60 dependency by 6.109%, above the 5% gate. The official
runner therefore exited 1 and P61A is a qualification **NO-GO**. It was not
repeated.

The 1.8x calculation is an optimistic envelope over the **whole measured KDA
interval**, not a projection for AITER/FlyDSL GDR decode. An AITER-inspired
kernel can cover only a subset of this interval and would leave replacement,
launch, state-publication and surrounding graph work.

## Registered boundary

P61A was measurement-only and default-off. It built warmed, stable cumulative
GGML/HIP graphs around the existing P46 preparation construction without
changing arithmetic or placing synchronization in production:

| stage | cumulative endpoint | whole-model median |
|---|---|---:|
| F0 | first AttnRes | 4.196169 ms |
| F1 | attention normalization | 4.014483 ms |
| F2 | KDA, state update and prefix publication | 108.086196 ms |
| F3 | second AttnRes | 114.310665 ms |
| F4 | FFN-normalized pre-MoE seam | 114.388377 ms |
| L | routed latent projection | 121.531008 ms |
| R | router and exact top-16 | 125.943421 ms |
| S | shared expert | 150.653539 ms |
| ALL | complete recurrent routed preparation | 168.953449 ms |

Whole-model quantities are medians of eleven same-index sums across all 68
layers, not sums of per-layer medians. The paired aggregate quantities are:

- whole KDA, `F2 - F1`: **104.108312 ms**;
- combined post-F4 preparation, `ALL - F4`: **54.464255 ms**;
- timed P46 teacher compute-only: **169.249156 ms**;
- production-adjacent device-hidden D2D plus eager P46 graph:
  **169.131499 ms**;
- paired production-minus-teacher D2D estimate: **-0.074014 ms**.

The negative D2D estimate is timer noise in a paired difference, not a negative
copy cost or a saving claim. Route D2H, checkpoint work, P42 publication,
storage and expert execution are outside this census.

## Exactness and schema gates

The one isolated census covered exactly the 68 recurrent routed layers:

```text
1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,
36,37,38,40,41,42,44,45,46,48,49,50,52,53,54,56,57,58,60,61,62,64,65,66,
68,69,70,72,73,74,76,77,78,80,81,82,84,85,86,88,89,90
```

The canonical CSV SHA-256 is
`3b4bab585e9106b4d76eac06b75b0f2353ce425d08d66aa4b0f9554895a36f30`.
Every layer reports `exact_flags=511`: prefix, cross-variant F4 seam, latent,
router IDs, router-weight bits, shared output, convolution state, SSM state and
node/output identity all match the synchronous P46 teacher. No identity or
address fingerprint is bad.

The schema is complete:

- 612 cumulative graphs (`68 layers * 9 variants`);
- 6,732 cumulative timed submissions (`612 * 11 trials`);
- 748 timed empty-event, teacher, production and D2D samples apiece
  (`68 * 11`);
- five warmups and eleven trials per layer/stage;
- all samples finite, no negative KDA trials;
- zero timed H2D bytes and zero timed storage bytes;
- eager GGML/HIP on physical GPU1 `gfx1151`, capture compiled=false,
  capture enabled=false, graph-disable environment absent and P46's skip
  property true.

The candidate `ALL` graph reconciles to the timed P46 compute-only teacher at
**0.174717%**, passing the 2% exact-boundary timing gate.

## Qualification failures

The preregistered timing gates did not survive the single run:

| gate | measured | limit | verdict |
|---|---:|---:|:---:|
| maximum aggregate stage spread | 123.0813% | 2% | FAIL |
| KDA paired-delta spread | 53.8357% | 2% | FAIL |
| combined-preparation spread | 170.0793% | 2% | FAIL |
| paired reconstruction max relative error | 102.9852% | 2% | FAIL |
| P60 production reconciliation error | 6.1093% | 5% | FAIL |
| candidate ALL vs teacher error | 0.1747% | 2% | PASS |

The most visible outliers are in trials 4–6. The exact raw vectors are retained
in `results/k3_p61a_recurrent_split_census.json`; representative central vectors
include:

```text
ALL: 168.5231, 168.7547, 168.8832, 168.9836, 261.0713, 232.8711,
     168.7885, 168.9844, 168.8226, 169.1164, 168.9534 ms
KDA: 104.1083, 105.0022, 103.8422, 159.8781, 104.1071, 104.9901,
     104.0370, 105.0522, 103.8306, 104.9945, 103.9915 ms
prep: 54.1771, 54.2591, 54.4007, 54.3478, 146.8095, 118.3314,
      54.5537, 54.3478, 54.4643, 54.7823, 54.5651 ms
```

The production-adjacent median is 169.131499 ms versus P60's frozen
159.393603-ms dependency, a 6.1093% mismatch. It excludes route D2H,
checkpoint and P42 work by contract, but its failure still prevents treating
this one run as a qualified decomposition of P60.

## Decision math

The registered decision applies a hypothetical 1.8x speedup to the complete
paired KDA interval:

```text
saving = 104.108311694 * (1 - 1/1.8) = 46.270360753 ms
```

- saving below **38.521937 ms**: hard-stop;
- saving at or above **50 ms**: continue;
- otherwise: inconclusive.

The measured central interval would need a whole-interval speedup of
**1.587346643x** merely to reach the hard floor. It does not prove that any
particular kernel can provide that speedup. Because timing qualification failed,
even the central 46.270361-ms envelope is diagnostic rather than a promotion
result.

## Run and artifact envelope

The reviewed Release/HIP source was built with ccache disabled and `-j4`.
The default-off self-test passed; serialized event tests passed on GPU0
`gfx1201` and GPU1 `gfx1151`. Exactly one model-isolated 68-layer census then
ran on physical GPU1. There was no broad suite, generation, repeat or model fact
beyond that isolated census.

The census wall was 40.35 s with maximum RSS 6,002,980 KiB. `/usr/bin/time`
and process telemetry report zero process swaps. The machine-wide historical
swap-used counter rose by 4 KiB (129,540,096 to 129,544,192 bytes); this is
recorded as host background history, not process swap. No KFD process was
present before or after the census and the GPU was idle afterward.

Evidence is rooted at
`/home/duster/kimi-k3-deploy/p61a-official-20260820`. The source snapshot,
binary, stdout/JSON, stderr, gate logs and 32-entry artifact-manifest hashes are
bound in the machine-readable result. The exact reviewed scoped patch SHA-256
is `ccf6c6deb6ad277b884759118ef88c904908512914fd0fa17f23400979b3a7c8`.

## Source removal

P61A was a temporary discriminator. Closure removes all **858 added lines**:

- 350 compiled additions across the private HIP timing hook and K3 graph seam;
- 31 CMake integration lines; and
- 477 lines in the standalone P61 runner.

All five touched tracked source/CMake files match HEAD exactly, the P61 runner
is absent, and the existing P46 runner remains byte-identical to HEAD. No P61
runtime symbol or build option is retained. Only this report, its result JSON
and roadmap revision 72 remain.

## Next decision

Do not begin KDA/AITER kernel work from this noisy decomposition. P62 first
measures whether same-layer NVMe delivery overlaps the existing P45 device
window, preserving exact arithmetic and the GPU1 ownership chain. That census
decides whether expert I/O/execution overlap is the larger bounded prize.

P58's exact recurrent-microtile / MoE-macro multi-token comparison remains
open. After P62, compare its attainable gain with a requalified recurrent split
before choosing kernel work versus multi-token verification.
