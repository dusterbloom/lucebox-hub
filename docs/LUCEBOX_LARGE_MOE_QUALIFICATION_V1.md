# Lucebox large-MoE qualification v1

## Decision

Hy4 should be treated as a useful **process prototype**, not as a recipe to
copy.  The reusable idea is to spend almost all compression effort on routed
experts, protect small control/residual tensors, calibrate candidate formats
from the native source, and separate decode from prefill execution.

The Lucebox standard should therefore be a small qualification contract:

```text
native model + official template
        -> semantic tensor inventory
        -> route-aware native calibration
        -> measured codec curves
        -> byte allocator
        -> provider-oriented package
        -> decode/prefill-specific execution
        -> terminal quality and physical-performance gates
```

It should not be a universal list of regular expressions, layer numbers, or
quant types.

This document calls that contract the **Lucebox MoE ladder**. Its scope is a
model whose highest-authority released weight set exceeds 300 GB. The final
serving artifact may be much smaller. That distinction matters: a 500-GB
source that becomes a qualified 98-GB image is a resident deployment, while a
2.8-T-parameter source may remain a provider deployment after aggressive
compression.

The standardized object is the evidence and serving contract, not the codec.
Every target must produce the same classes of records and pass the same kinds
of gates. It is expected to produce a different tensor allocation.

## Two reference implementations, not one

Hy4 and Lucebox's DeepSeek4 work supply complementary halves of the process:

- **Hy4 is the encoder reference.** It starts from the highest-authority
  source, uses activation importance during fitting, spends precision by
  semantic role, and publishes the tensor recipe with the codec patch.
- **DeepSeek4 is the resident runtime reference.** Its adaptive ROCmFP2/3
  formats have per-expert codebooks and modes, a host reference decoder,
  strict metadata validation, compressed-domain one-row kernels, and separate
  larger-batch fallbacks. The final Strix artifact is one checksummed 98.29-GB
  GGUF rather than an untracked weight/sidecar pair. It retained 82/92 on the
  full evaluation and 17/17 COMPSEC at 18.1 tok/s; a faster top-4 policy reached
  22.3 tok/s but lost one COMPSEC item, so quality mode remained the default.
- **K3 is the provider/BWS reference.** It makes authoritative logical and
  physical bytes part of correctness, supports route/slab reads below an
  all-expert representation, and keeps exact causal-state adoption separate
  from approximate expert fidelity.

The first two references also expose a useful implementation order. DeepSeek4
initially served its adaptive format through dequantization plus dense GEMV.
Only after profiling showed that expansion dominated decode was a fused
compressed GEMV written; on gfx1151 that changed roughly 7.8 to 17.9 tok/s.
That is the default order for a new codec. Correctness and a usable artifact
come before a specialized decode kernel.

## GSQ-RCO update: refine standard formats, then allocate them

The `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF` artifact at revision
`888cc868537099e09a9c4f41a2b9a421b346f88b` adds two useful, separable
techniques:

- **GSQ** learns discrete scalar grid assignments and group scales, then
  materializes ordinary GGUF scalar formats. It is therefore a mandatory
  native-source codec arm before Trellis or a new HIP decoder is considered.
- **RCO** chooses a mixed tensor allocation under an exact global byte budget
  using the actual non-decomposable objective. Its idea matches the eventual
  `route x slab x fidelity` problem better than independent local scores.

The reviewed artifact is a 27B dense/hybrid model, not a greater-than-300-GB
MoE, and supplies no K3 terminal-KL, tool-boundary, or Lucebox measurement.
Its phrase "task-lossless" applies only to the published AIME25, GPQA-Diamond,
and LiveCodeBench evaluations. It is not evidence of distributional, tool, or
universal output equivalence.

The public IQ2_S plan nevertheless contains an important structural result:
it uses 12 distinct GGUF types, makes highly irregular choices within the
same projection role, and retains 449 small tensors in BF16/F32. This agrees
with Hy4 and DeepSeek4 independently. Semantic roles should impose floors and
priors, while evidence chooses individual fidelity; a model-wide bpw label is
only an artifact summary.

RCO is not yet an earned K3 subsystem. Its true-loss optimizer requires a
differentiable forward or validated differentiable terminal surrogate, while
the current captured-state llama.cpp intervention path is not autograd based.
Until Phase B passes its held-out correlation gate, use the simple discrete
allocator over measured utilities. If the gate passes, test RCO first on a
representative layer with byte-discretization and tractability measurements.
The immutable source review is
`results/lucebox_gsq_rco_artifact_review_20260901.json`.

## Model intake matrix

The common inventory must represent the following known differences without
pretending they are interchangeable:

| target | authoritative scale / routing | sensitive non-MoE structure | likely first deployment class | first adapter work |
|---|---|---|---|---|
| DeepSeek V4 Flash | 284B parameters, 256 experts, top-6 | MLA, indexer/compressors, four HC streams | resident after mixed ROCmFPX | use as the resident control |
| Hy4-preview | 770B / 49B active, 256 experts, top-8 + shared | gated DSA, IndexCache, iHC, native MTP | provider/BWS on current Lucebox4 at the published 213.66 GiB; reclassify only if a smaller qualified artifact fits the full serving ledger | map split attention/indexer and HC roles; reproduce STQ from source |
| GLM-5.3-Flash | 320B / 18B active | hybrid sparse/linear attention, mHC, multimodal path | determine from the qualified artifact and context-memory ledger | add linear-state, mHC, and vision roles before quantization |
| Kimi K3 | 2.8T / 104B active, 896 experts, top-16 + two shared | 69 KDA + 24 gated MLA, AttnRes, native vision | provider/BWS | use the existing K3 semantic and byte-accounting adapter |

This table is a routing guide, not a claim that an unmeasured target fits.
Artifact bytes, usable UMA/VRAM after cache allocation, device ownership, and
the requested context decide the deployment class.

## Keep the compression axes identifiable

A target can reduce cost along five different axes:

1. **fidelity:** codec/bit depth for a complete tensor or tile;
2. **routes:** number of routed experts evaluated;
3. **intra-expert coverage:** slabs/tiles retained within each selected route;
4. **residency and layout:** where bytes live and how often they are loaded;
5. **temporal execution:** batching, prefill reuse, caching, or speculation.

Each axis gets an isolated control before combinations are scored. Route
pruning is not described as quantization; sparse BWS is not described as a
smaller artifact; prefix restore is not described as cold prefill. Pairwise
interaction tests are required before a joint allocator assumes additive
damage. This keeps the result scientifically attributable while still
allowing a final plan over route, tile, and fidelity when the measurements
earn it.

## Common scorecard, class-specific traffic gate

All classes use the same quality scorecard: native-eligible task retention,
tool/structured validity, full-vocabulary terminal KL distribution, top-1
agreement, and generated IDs. They also publish measured decode and prefill
throughput rather than extrapolating from a kernel or storage benchmark.

Traffic gates differ by deployment class:

- **resident:** artifact and peak-serving bytes must fit the qualified
  allocation; report actual DRAM/UMA ownership and bandwidth counters;
- **resident base + cold residual:** report base bytes, residual logical and
  physical bytes per position, rescue incidence, and average sequence cost;
- **provider/BWS:** report authoritative logical and physical direct-read
  bytes per position, exact fallback/rescue bytes, and expert/slab union per
  prompt.

For K3 on the current approximately 5.3-GiB/s NVMe path, the decode north star
remains at most 0.53 GiB of physical authoritative traffic per position and a
measured 10 tok/s. Prefill's 100 tok/s north star is separately measured at
named prompt lengths with exact terminal-state hashes. A resident target does
not inherit K3's cold-byte gate, and a provider target cannot claim success
from resident bandwidth arithmetic.

## Bounded-memory conversion

The conversion itself must not require the source model to fit in Lucebox
memory. Use three resumable passes:

```text
metadata-only inventory
        -> calibration/statistics capture
        -> tensor-or-stripe streaming encode and checksummed append
```

The encoder reads at most one tensor or explicitly bounded stripe plus its
calibration statistics, writes an immutable payload record, verifies it with
the scalar reference decoder, and checkpoints the manifest. A restart skips
only records whose source and output hashes match. Whole-model materialization,
an unbounded in-memory imatrix, and a final unverified concatenation are not
part of the standard.

## Immutable source review

The Hy4 material reviewed here is Hugging Face revision
`779242edccdedc2109a0b36b164263a88f015bfa`:

- `0002-stq1_0-quant-and-cuda.patch` SHA-256
  `b6deb5d1eda8cc241c417c28725c426dfe36d280ed744dc40ac9aa4472748ec1`;
- `Hy4-preview-STQ1_0.tensortypes` SHA-256
  `6ccb89ba093ece88becac2c920ca74e18cb332a27059243db1ca57407a6244d0`;
- model API capture SHA-256
  `ed40c6cc1c6f42c85fcf1e0d45d7c2d324035491b0d3c4806ed78c87e1a9a78b`.

The published STQ build is 213.66 GiB / 2.38 bpw for a 770B model.  Its
routed gate/up tensors use STQ1_0 on 29 layers and IQ2_XXS on 48.  Routed down
uses IQ3_XXS, with the final three layers raised to IQ4_XS.  Small tensors
that control routing, DSA visibility, attention, residual flow, recurrent
mixing, and the vocabulary head are deliberately much higher precision.

STQ1_0 stores one scale and a ternary 3:4 pattern per 256-weight block, or
1.3125 bpw.  Hy4's important PTQ change is weighted least-squares scale plus
importance-aware zero placement, alternated for three rounds.  The patch has
a compressed MMVQ implementation but no STQ MMQ.  Its reported throughput is
fully resident on eight H20 GPUs; it is not evidence about cold Lucebox
traffic or gfx1151 kernels.

Sources: [Hy4 serving artifact](https://huggingface.co/AngelSlim/Hy4-preview-GGUF),
[official Hy4 model card](https://huggingface.co/tencent/Hy4-preview),
[DeepSeek4 Lucebox artifact](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3),
[official GLM-5.3-Flash card](https://huggingface.co/zai-org/GLM-5.3-Flash),
[official Kimi K3 card](https://huggingface.co/moonshotai/Kimi-K3),
[ROCm Core SDK 10.0 release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html),
and [Moonshot FlashKDA](https://github.com/MoonshotAI/FlashKDA).

## What transfers from Hy4

1. **Compress parameter mass, not every tensor equally.** Hy4's three routed
   expert families hold 97.7% of its parameters, so it can afford generous
   floors elsewhere.
2. **Tensor role matters.** A down projection writes into the residual stream
   and is not interchangeable with gate/up.  Routers, indexers, recurrent
   state transforms, and vocabulary heads have small byte cost but large
   behavioral leverage.
3. **Native-source calibration matters.** STQ's scale and zero pattern are
   fitted decisions.  Requantizing a deployment quant compounds error.
4. **Layer choice must be measured.** The published file says its STQ/IQ2
   layer split is imatrix-derived, but does not publish the score curve or a
   transferable threshold.  The layer-number list is not a general method.
5. **Codec and executor are separate decisions.** A compact format can be
   excellent for storage and still need different kernels for one-row decode
   and many-row prefill.

## What K3 falsified

The local K3 screen tested an exact IQ1_S core plus a Hy4-style STQ gate/up
tail at route12/B16 bytes.  The frozen held-out comparison used 12 rows from
10 validation sequences:

| arm | payload bytes / routed layer | mean local relative L2 | median |
|---|---:|---:|---:|
| 16 exact IQ1_S records | 8,601,600 | 0.711892 | 0.710868 |
| 8 exact + 9 STQ gate/up tails | 8,623,104 | 0.728829 | 0.729720 |

The candidate used `1.0025x` bytes, lost all 12 rows, and increased mean error
by `2.379%`.  Per the preregistration this is **NO-GO** for cascaded
IQ1_S-to-STQ1_0 as K3's leading complement.  It does not test STQ calibrated
directly from K3 BF16/native weights.

The immutable decision is
`results/k3_stq1_exact_core_holdout_decision_20260901.json`.

## Qualification ladder

Each rung is independently useful and is the gate for the next one. A target
stops at the cheapest failed rung and retains the negative result.

### R0 — source and native closure

Freeze the upstream revision, file hashes, license, tokenizer, processor,
official chat template, generation settings, and highest-authority released
weights. "Native" can be BF16/FP8 or an upstream QAT format such as K3's
MXFP4; it does not mean inventing a BF16 teacher that was never released.

Before quantization, record native output IDs and logits for the fixtures the
model actually passes. For recurrent/hybrid models also hash the complete
terminal attention/recurrent state. No later arm is scored on a fixture the
native control fails.

### R1 — semantic and memory feasibility

Inventory every tensor, byte, role, layer, expert axis, block constraint, and
runtime consumer. Compute three honest footprints:

```text
authoritative source bytes
candidate artifact bytes
peak serving bytes = artifact residency + KV/recurrent state + workspace + OS margin
```

Then classify the candidate as resident, resident-base-plus-residual, or
provider/BWS. Reject any plan whose peak allocation cannot close before doing
quality calibration. A nominal 128-GB machine is not a 128-GB model budget.

### R2 — native-source codec screen

From the same authoritative tensor samples and route-aware calibration, fit a
small common arm set:

1. the current qualified scalar/IQ control;
2. the same standard GGUF target format refined with GSQ, at identical stored
   bytes;
3. adaptive per-expert scalar codebooks where evidence supports them;
4. STQ for eligible gate/up surfaces;
5. a Trellis proxy only if an existing reference implementation is cheap to
   adapt.

Test gate, up, and down separately at equal stored bytes. Local weighted error
and cosine are screening measurements only. A format that is not self-
describing must ship its codebook/mode metadata in the hashed package manifest
and must have a scalar reference decoder before runtime work starts.

### R3 — captured-state terminal attribution

Inject the candidate expert outputs into aligned native trajectories. Record
local residuals, full-vocabulary terminal KL, top-1 and margin changes, tool or
grammar boundary effects, and authoritative bytes. Use calibration traces to
fit selectors, a separate validation split to choose the plan, and a sealed
holdout for the decision.

The selector may use terminal labels offline. A runtime rescue policy may use
only information resident before the additional read. This prevents an
oracle score from silently becoming an impossible serving policy.

### R4 — allocation and narrow generation

Compile the measured curves into a discrete allocation over
`(role, layer, expert, tile, fidelity)`. Structural floors are constraints;
the remaining bytes maximize measured terminal information saved per byte.
Run short native-success, exact-copy, structured-output, and known boundary
fixtures before a broad suite. Static route-count or slab-count reductions
are separate approximation axes and cannot hide inside a quantization result.

Use exact dynamic programming or a simple constrained allocator when the
objective is a sum of measured utilities. A non-decomposable RCO-style
allocator is eligible only after the terminal surrogate predicts exact
interventions better than local residual ranking on held-out layers. Preserve
its candidate database and emitted allocation as immutable artifacts; the
allocation file is compiled evidence, not a hand-authored codec recipe.

### R5 — broad quality

Use the official template and source-matched history on coding, multi-step
reasoning, factual, extraction, multilingual, long response, JSON, native tool
calling, and at least one longer coding-agent turn. A natively multimodal
target also needs source-matched image/video fixtures. Publish terminal KL
mean/median/p95/max, top-1 agreement, output IDs, task success, and native
tool validity. Token equality is reported as token equality, not
distributional equivalence.

### R6 — package and execute

Produce one immutable manifest and payload set. Resident images should be
self-contained when practical, as the final DeepSeek4 artifact is. Provider
images may use multiple files, but every payload and metadata object is
addressed and checksummed from the manifest.

Bring up the simplest exact executor first:

- decode: reference dequantization plus dense GEMV, then compressed GEMV only
  if end-to-end profiling earns it;
- prefill: dequantize each routed expert/tile once and serve all matching rows
  with dense GEMM, then consider compressed MMQ only if it wins the measured
  crossover;
- recurrent/attention core: qualify separately from expert execution.

### R7 — production promotion

On Lucebox hardware record logical and physical bytes, direct-I/O time,
storage wait, provider time, routed compute, total prefill/decode throughput,
memory high-water marks, thermals/clocks, ROCm/build identity, and exact output
hashes. Run a repeated soak and the sealed quality suite. A passing primitive
is narrowly transplanted to `perf/k3-production-ponytail` (or the relevant
production branch) and requalified there; the experiment branch is never
merged wholesale.

## Immutable bundle per rung

Existing scripts may produce the files; v1 does not require a new orchestration
framework. Every target must preserve equivalent immutable records:

```text
source.json          upstream revisions and every input hash
inventory.json       semantic tensor map and byte totals
fixtures.jsonl       prompts/history/template hashes and native eligibility
calibration.json     trace IDs, route coverage, statistics, split assignment
codec_curves.csv     equal-byte role/layer/expert measurements
plan.json            exact selector/allocation and structural floors
artifact.json        payload index, codec metadata, alignment and checksums
quality.json         logits, KL, IDs, task/tool outcomes
performance.json     logical/physical bytes, timing, memory and hardware
decision.json        preregistered gate and GO/NO-GO interpretation
```

Every record includes source branch/commit, dirty-tree and patch hash,
executable hash, exact command, model/template/calibration hashes, hardware,
and whether each value is measured or projected. Results are append-only.

## Reproducible calibration split

Prompt diversity and route coverage are both required. Freeze three disjoint
sets before looking at a candidate result:

- **fit:** estimate activation importance, codebooks, scales, and candidate
  terminal predictors;
- **validation:** choose codec/allocation/hyperparameters;
- **sealed holdout:** make the promotion decision once.

Sampling should stratify by task family, sequence position, layer family,
route frequency, and rare-expert coverage. Route frequency affects expected
bytes and expected damage, but it is never a substitute for intervention.
An expert with no calibration observations is marked unknown and retains a
conservative floor; absence of samples is not evidence of irrelevance. The
native-success filter is applied independently to every split.

## The minimum portable contract

### 1. Semantic inventory

Each supported model gets a thin adapter that maps source tensor names to
roles.  The common manifest needs only:

```text
model identity and source hashes
layer and expert identity
role: router | expert_gate | expert_up | expert_down | shared_expert |
      attention | recurrent | indexer | embedding | vocabulary | other
logical shape and axis meaning
native dtype and bytes
routing geometry and activation function
official chat-template hash
```

Role mapping remains model-specific.  Byte accounting, calibration records,
codec curves, allocation, packaging, and qualification remain common.
Regex-first-match recipe files may be emitted for llama.cpp compatibility,
but they are compiled output and must be checked against the exact inventory.

### 2. Native, route-aware calibration

Calibration starts from the highest-authority released weights and native
teacher trajectories.
For each expert projection it records at least route count, router-weight
moments, input second moments, output norms, and task/sequence identity.
Gate/up and down require different activation statistics.  A pooled imatrix
is acceptable only as a screening feature; it is not proof that rare experts
or terminally sensitive directions are protected.

Candidate codec error is first measured on captured routed states.  Promotion
uses aligned full-vocabulary terminal KL, top-one/margin effects, and fixtures
the native model passes.  K3 has already shown that local error ranking is a
weak terminal-value proxy and that lower KL at one token does not guarantee
the right discrete tool decision.

### 3. Measured byte allocation

For every eligible `(role, layer, expert, codec)` choice, the planner consumes
measured or explicitly validated predicted damage and authoritative bytes.
The optimization is a discrete budget allocation, not a global bpw target.

Hard structural floors protect routers, token/indexer gates, recurrent-state
transforms, residual-stream projections, embeddings, and the vocabulary head
until an intervention experiment earns a lower floor.  Rarely routed experts
are not automatically unimportant: route frequency is multiplied by damage,
not used as the objective alone.

### 4. Provider-oriented package

A monolithic GGUF remains a compatibility view.  The serving artifact indexes
each payload by:

```text
(layer, expert, role, tile, fidelity)
codec and block geometry
file offset, stored length, logical authoritative length
alignment, checksum, source tensor hash
optional parent/base payload
```

Learned codebooks, mode bits, rotations/transforms, and any expert permutation
are part of that same indexed record. A decoder must fail closed if metadata
is missing, duplicated, dimensionally incompatible, or unsupported on the
selected execution path. Silent fallback to a fixed-codebook decoder is data
corruption, not compatibility.

Payloads are independently readable and at least 4-KiB aligned.  One payload
is stored once; decode and prefill use different index traversals before any
duplicated layout is considered.  Tile size is derived from tensor geometry,
codec blocks, direct-I/O granularity, and measured kernel shapes; K3's
256-neuron slab is not made a universal constant.

### 5. Two execution paths

Decode uses a compressed-domain GEMV/MMVQ for one or a few rows.  A gfx1151
STQ kernel should decode packed ternary groups directly and use wave32 integer
dot operations where profitable; it is not earned until native-source STQ
wins the quality screen.

Prefill uses a reuse threshold measured per `(codec, shape, hardware)`:

```text
load compressed expert/tile once
        -> dequantize once to FP16/BF16
        -> hipBLASLt/WMMA over every matching routed row
        -> discard or retain according to the layer cache policy
```

This deliberately avoids requiring a bespoke low-bit MMQ for every codec.
It can lose for short prompts, tiny route groups, insufficient workspace, or
when dequantization is not amortized; those cases fall back to compressed MMQ
or MMVQ based on a measured crossover, not a fixed `M >= 16` rule.

KDA and other recurrent cores are separate.  Weight reuse cannot rescue K3's
current M1024 causal-core floor.  K3 prefill still requires a true
chunkwise/DPLR algorithm with exact terminal-state adoption.

## Residency classes

The same package supports three deployment classes:

1. **Resident:** selected representation plus runtime state fits the qualified
   GPU/UMA allocation.
2. **Resident base + cold residual:** a compact all-expert base fits; exact or
   higher-fidelity residual payloads hydrate selectively.
3. **Provider/BWS:** no useful all-expert base fits, so authoritative
   route/expert/tile payloads are read on demand.  K3 is in this class.

Classification is made from measured usable bytes, not nominal system memory.
The gfx1151 and gfx1201 memories are not treated as one zero-cost pool; any
cross-device policy must include transfers and ownership in its ledger.

## ROCm 10.0 implications

Lucebox4's recorded production runs use ROCm 7.2.2.  ROCm Core SDK 10.0.0 was
released on 2026-08-26 and officially supports gfx1151 and vLLM 0.27.  Relevant
changes include HIP event-overhead reductions, cooperative-group scans,
better gfx1151 roofline/ATT profiling, HIP graph-node attribution, LLVM 24,
hipBLASLt 1.4.1, and rocWMMA 2.2.1.

These changes earn a side-by-side toolchain A/B, not a throughput claim.
`hipFile` remains listed for Instinct rather than Ryzen/Radeon, so ROCm 10 does
not provide an assumed gfx1151 GPU-direct-storage solution.  Cooperative scan
is a possible building block inside KDA preprocessing; it is not the
chunkwise DPLR algorithm itself.

Production must not be upgraded in place.  Build the same commit separately,
verify exact output/state closure, then compare relevant microkernels and one
end-to-end fixture under the same clocks and ownership.

## First three discriminators

### A. Native-source codec deathmatch

Stream a small set of highest-authority expert tensors from at least
gate/up/down roles for two materially different MoEs. Fit STQ1_0, a
GSQ-refined standard scalar arm, and the current standard/IQ control from the
same route-aware calibration. The GSQ arm and its control must have exactly
the same stored format and bytes. Inject each into held-out native
trajectories.

Use "highest-authority source" rather than BF16 for a QAT-native model. The
first two targets should be Hy4 and one structurally different source for which
we can obtain authoritative weights and terminal trajectories; K3's current
IQ1_S deployment file is not an acceptable source for this discriminator.

GO for a codec/backend effort only if it reduces median terminal KL by at
least 20% at no greater authoritative bytes than the control, does not worsen
held-out top-one/tool correctness, and repeats on both models.  Otherwise the
codec remains model-specific research.

### B. gfx1151 execution crossover

For the winning codec and real K3/Hy4 tensor shapes, measure `M = 1, 2, 4, 8,
16, 32, 64, 128` under ROCm 7.2.2 and 10.0:

- compressed GEMV/MMVQ;
- compressed MMQ if available;
- compressed load + dequantize + hipBLASLt FP16/BF16 GEMM.

Include load, dequantization, workspace, and dispatch.  Adopt ROCm 10 or a new
kernel only with exact output closure and at least a 10% end-to-end improvement
in the execution regime that consumes it.  A microkernel-only win does not
qualify serving.

### C. Package/layout A/B without changing quantization

Repack an existing qualified model's expert payloads into the common indexed,
aligned provider container.  Keep tensor bytes and computation identical.
Require identical logits/output IDs, logical byte equality, physical bytes
within 5% of logical payload bytes, and a measured storage-wait reduction of
at least 20% before replacing the current provider layout.

## TIPPS stop list

The best expert can already reject the following, so v1 does not build them:

- a universal Hy4 tensor-type file;
- a large format-agnostic quantization framework;
- a full RCO optimizer before a differentiable or validated terminal surrogate;
- an STQ HIP kernel calibrated from K3 IQ1_S;
- a low-bit MMQ before dequant-once plus dense GEMM is measured;
- a duplicated decode/prefill expert bank;
- a production ROCm 10 upgrade based on release notes;
- performance projections from fully resident H20 measurements;
- a single scalar quality score that replaces terminal KL and real fixtures.

The next implementation is only the smallest harness needed by discriminator
A.  The common package/compiler starts after a native-source codec wins.
