# K3 terminal-BWS v2

## Provenance lock

The experiment branch is `experiment/k3-terminal-kl-bws-v2` at
`fac048c090c74e5f8f989bffcda3aadc0bc8c266`, created from the clean commit
instead of the dirty production worktree.  The research-only reference is
`perf/k3-layer-major-prefill` at `102cf35ab2d86dc37e63faeece180267b2e9b2e6`;
its terminal-state adoption parent is `04ce074a17f37285b9de0e7169b1ee307f9f1000`.

The imported evidence objects are immutable Git blobs:

| Object | Blob SHA-1 |
| --- | --- |
| `docs/H16_PROGRESSIVE_FISHER_SLABS.md` | `ec28130efc354dfb9b6dc291b51bebd85f390ca6` |
| `docs/K3_H22_LAYER_ADAPTIVE_ALLOCATION.md` | `8f7ea9083da71e6299da6530f7279697c093dd23` |
| `docs/H23_ADAPTIVE_BYTE_FRONTIER.md` | `049a369fdcd771bd131e6ecd98edee9d6862ccdb` |
| `docs/H23_10K_CALIBRATION_RUNBOOK.md` | `432b73b7a72d33b3fd45cc12641ab063f558e1f3` |
| `docs/K3_LAYER_MAJOR_RANGE_M64.md` | `388f2ab8e1ef5d9fcb79741b88d969e652df5acd` |
| `results/h23_10k_projected_byte_frontier.json` | `604392ceb60ed4e1e93e89a23319fcadad9d7fa6` |
| `results/h23_10k_aggressive1p8_quality.json` | `81f82ce50dce6eca958234d4ace12236e8a6cccf` |
| `results/h23_10k_moonshot1p2_quality.json` | `1960f193ce812f73f35c6c8c99ed4b2521f52c38` |

## Production promotion boundary

This branch is research-only.  A passing experiment does not make
`experiment/k3-terminal-kl-bws-v2` a production branch and does not authorize
merging it wholesale.  Each earned primitive must be narrowly transplanted or
reimplemented on `perf/k3-production-ponytail`, then re-earn binary closure,
traffic accounting, behavioral quality and performance gates there.  The same
rule excludes a wholesale merge of `perf/k3-layer-major-prefill`; that branch
is an immutable source of research artifacts and proven causal/state
invariants only.

## Decision objective

There is no global `10x` KL-reduction or `0.01`-KL promotion gate.  Terminal KL
is the distributional risk signal used to discover selectors and identify
rescue positions; it is not a substitute for behavior.  A candidate advances
only when it improves the measured byte/time Pareto frontier while retaining
native-success coding, reasoning, structured-output and tool-call fixtures.
Mean KL is reported with median, p95 and maximum because a modest mean can hide
a decisive identifier-boundary failure.  Conversely, harmless probability
redistribution need not block a materially faster candidate whose outputs and
tasks remain reliable.

The explicit trade-off is accepting some measured distribution drift for
speed, while using progressive fidelity at the sparse positions where the
drift changes behavior.  The known `get_weather` identifier boundary remains a
hard rescue gate; low entropy or a confident top-one prediction does not waive
it.

## Decisive first experiment

For each representative isolated routed layer, retain the existing exact
teacher trajectory and evaluate one changed slab group at a time.  Each plan
row must name the layer, route/expert, natural slab IDs, byte count, local
residual/cosine, selector score, and paired full-vocabulary trace paths.
`scripts/analyze_kimi_terminal_slab_screen.py` rejects unaligned traces and
produces mean/median/p95/max terminal KL, top-1 changes, logit-margin change,
and conditional KL-recovered-per-byte.  It reports rank correlation only after
three real interventions; missing estimators remain `null`, never zero.

Screen independent marginals first, then use the measured winners for a greedy
conditional restoration series at budgets 8/12/16/24/32.  The only Phase-A
GO criterion is lower measured terminal KL at equal-or-lower authoritative
bytes than the frozen local Budget24 control.  No projection is a GO.

## Frozen comparison facts

The 10K trace has exact geometry 9.069580078125 GiB/position, a measured
fallback floor 0.009017229080200195 GiB/position, and a registered Budget24
floor 1.1427147388458252 GiB/position.  The measured 1.8-GiB H23 candidate
used 1.8344892914762203 GiB/position with mean KL 0.2963024675818419 and 6/6
exact sequences on the small suite.  The measured 1.2-GiB candidate used
1.2200937172801225 GiB/position with mean KL 0.7417508484726244 and 5/6 exact
sequences.  These are controls, not Phase-A results.

M64 layer-major was an exact, storage-fast end-to-end NO-GO: only 0.65% total
improvement.  Its causal/state invariants may be reused later; its shape must
not be tuned as a prefill claim.

## Lucebox4 scalar control (2026-08-31)

The complete deployed K3 teacher was found and used on Lucebox4:
`Kimi-K3-KDA-HYBRID-Q2-MIDLATE-00001-of-00014.gguf` (first-shard SHA-256
`93cf4fa551660cbc390e6bdea12caf67403cdb0091d82a17b14e39d253708fc4`), with
the calibrated96 manifest SHA-256
`1189c85743a1e5f41901f09b090ea3a4a567c2020bce53400e9ae5f039409f56` and
natural-sidecar manifest SHA-256
`86216b66256f61bdcbe8d796dc2e9316e3dccdeba801ea296b2e779cf86394e9`.

The production smoke harness must run as `HIP_VISIBLE_DEVICES=1` with logical
GPU 0.  Addressing physical GPU 1 directly reproducibly faulted inside ROCm
graph capture's fused RMSNorm before a forward result.  This is a hardware
topology/harness limitation, not a BWS observation.

On the fixed five-token raw fixture `According to all known laws`, an exact
terminal capture predicted token ID 11 (`,`).  The frozen H23 moonshot table
(all 92 layers at Budget24) measured 5.668487549 GiB logical provider bytes
for the five prefill positions, or 1.133697510 GiB/position; its terminal KL
was 1.158164302 and its top token was ID 318 (` of`).  This is a one-row
plumbing control and must not be interpreted as a quality-suite score.

The new immutable plan trace showed that at the final token, terminal-sensitive
layer 92 spent all 24 records on slabs 0--11 of experts 382 and 4.  Forcing
one omitted-route candidate (layer 92, expert 369, rank 11) at the same bytes
made terminal KL **worse**, 1.193684550 (+0.035520248), while retaining the
same wrong top token.  It falsifies the simple claim that an arbitrary omitted
route/slab is a terminal-quality rescue.  It is one conditional datum only;
no terminal ranking, rank correlation, or below-Budget24 conclusion is earned.

A uniform local-selector Budget8 point was also measured at 0.377899170
GiB/position, with terminal KL 0.371722866 (still wrong top ID 318).  The
exact terminal vector was byte-identical on a rerun.  This surprising
non-monotonic point is **not** Phase-A evidence: its plan trace reaches layer
92 with a different route set than Budget24, so it is an on-policy terminal
comparison, not an ablation on a frozen native trajectory.  It establishes
that sub-24 representations are runnable and worth studying, not that they
are quality-safe or terminal-ranked.

`results/k3_terminal_bws_v2_lucebox4_scalar_20260831.json` is the immutable
result summary; the raw F32 logits, plan TSV, traffic TSV, stdout/stderr and
SHA256SUMS live under the absolute artifact roots named there.

## Phase A discovery result: direct terminal slab value

The paired probe was narrowed to the final prompt row and layer 92 while the
rest of the model followed the exact frozen KDA-hybrid trajectory.  Because
the deployed hybrid GGUF and the calibrated natural sidecars do not share a
source quantization identity, all 192 natural sidecar slabs are the
representation-matched routed teacher for this selector test.  The measured
native-to-sidecar difference is not charged to the selector.

Every one of the 192 layer-92 candidates was screened at equal Budget24 bytes:
24 selected records by drop-and-replacement and 168 omitted records by
force-and-eviction.  Only 22 of the omitted records improved the local
Budget24 baseline; 146 made it worse.  The existing local calibration score
was a weak terminal-value proxy (Pearson `0.212`, Spearman `0.305`).  A locally
attractive omitted record, E769:R11, was terminal rank 166 and increased KL.

The terminal-marginal ranking produced a direct, exactly equal-byte Budget24
measurement:

| selector | isolated layer bytes | terminal KL | top ID |
|---|---:|---:|---:|
| local Budget24 | 14,278,656 | 0.102755626 | 11 |
| terminal-ranked Budget24 | 14,278,656 | **0.051154509** | 11 |

This is a measured `50.22%` reduction, not a projection.  The terminal-ranked
curve was also monotonic from Budget8 through Budget32 and beat the local
selector at every equal-byte point by `25.70%`, `35.94%`, `37.65%`, `50.22%`,
and `62.32%`, respectively.  A conditional leave-one-out check around the new
Budget24 set found all 24 members useful relative to the next local-selector
replacement.

This earns a **research GO** for direct terminal sensitivity and falsifies the
claim that local residual rank is sufficient.  It does not establish a cheap
selector, cross-layer or cross-prompt generalization, official-template
quality, tool recovery, model-wide bytes/token, or throughput.  The bytes in
the table are for one isolated layer provider call; they must not be reported
as full-model bytes/token.

Following the TIPPS decision rule, further coordinate descent on the same
layer/prompt is deferred: an expert would reject it as discovery-set
overfitting.  The next gate is held-out replication on H22-selected early,
middle and late high/medium/tolerant layers, followed by the known tool-name
boundary.  The explicit trade-off is leaving possible layer-92 KL gains
unclaimed in exchange for external validity.

The immutable summary is
`results/k3_terminal_bws_v2_l92_phase_a_20260831.json`; all raw run directories
retain command/environment captures and checksum manifests on Lucebox4.

## Held-out prompt transfer and Phase-B metadata gate

A complete 192-intervention layer-92 screen was run on the held-out coding
prompt `Implement a Python function that parses JSON and returns sorted unique
keys.`.  The local Budget24 control had terminal KL `0.106299085`, chose token
`17763`, and disagreed with teacher token `646`.  Of 168 omitted-slab forces,
40 lowered KL, but the best single equal-byte swap lowered it by only `9.69%`
to `0.096000216`; no single swap recovered teacher top-one.  Local score
generalized better here than on discovery (Spearman `0.4441`), reinforcing
that slab value is prompt-conditioned.

The preregistered metadata-only ridge is a **NO-GO**.  It selected lambda `10`
and improved discovery Spearman from `0.3048` to `0.4074`, but held-out
Spearman fell to `0.2990`, versus `0.4441` for the existing local score.  Its
held-out gain was `-0.1451`, not the required `+0.15`.  Per the preregistration,
no additional metadata features will be added to rescue this model.

Post-heldout oracle interactions were retained as mechanistic discovery, not
validation.  At exactly 24 records, 14,278,656 logical authoritative bytes and
zero fallback, the best measured terminal KL values were:

| conditional set | terminal KL | relative reduction | teacher top-one |
|---|---:|---:|---:|
| local Budget24 | 0.106299085 | -- | no |
| oracle top-2 | 0.085052882 | 19.99% | no |
| oracle top-4 | 0.073302854 | 31.04% | no |
| oracle top-8 | 0.061147494 | 42.48% | no |
| positive-gain crossover-14 | 0.060754512 | 42.85% | no |

The crossover-14 set had lower KL but a worse teacher-token margin than top-8,
directly falsifying independent marginal additivity and demonstrating that
lower mean terminal KL does not guarantee identifier/token-boundary recovery.
This phase therefore earns no production transplant and establishes no
below-Budget24 result.  The next highest-value experiment is a narrow
prompt-conditioned terminal Fisher/JVP discriminator on these captured states,
validated against the completed exact intervention screen.  A general
second-order framework and GSQ/Trellis kernels remain unearned.

The immutable result summary is
`results/k3_terminal_bws_v2_heldout_code_phase_b_20260901.json`.

## Held-out margin-capacity Gate A

The post-heldout margin oracle was tested exactly as preregistered: four
nested force groups of 2, 4, 8 and 24 records, each retaining the Budget24
contract of 24 selected records, 14,278,656 logical authoritative bytes,
14,352,384 physical bytes and zero exact fallback.  All four runs reproduced
the frozen exact trajectory hash.

| group | terminal KL | KL change vs local B24 | teacher margin | teacher top-one |
|---|---:|---:|---:|---:|
| local B24 | 0.106299085 | -- | -0.575365 | no |
| top-2 margin | 0.098126259 | -7.69% | -0.460027 | no |
| top-4 margin | 0.086541398 | -18.59% | -0.366367 | no |
| top-8 margin | **0.080981479** | **-23.82%** | **-0.343472** | no |
| positive-margin crossover-24 | 0.211096682 | +98.59% | -0.940985 | no |

Gate A is therefore a measured **NO-GO**.  Even a held-out-label oracle did
not find an equal-byte static B24 set that restored teacher token 646 or made
its margin positive.  The 24-way arm also reversed every projected independent
gain, nearly doubled KL and worsened the boundary margin.  This is strong
evidence that finite slab interventions interact non-additively and that the
known boundary is not safely repaired by more static layer-92 ranking.

Per the preregistered stop rule, no captured-tail/Fisher framework is earned
from this gate.  The next discriminator is an offline low-bit approximation
of the omitted complement or a progressive richer-pass rescue, evaluated by
terminal logits on captured native states before any HIP decoder is written.
The immutable result is
`results/k3_terminal_bws_v2_margin_gate_20260901.json`; raw roots r58--r61
and analysis r62 remain on Lucebox4.

## Progressive tool-boundary rescue gate

The known official-template `get_weather` failure was rerun on Lucebox4 with
one executable and the H23 moonshot 1.2-GiB table as the base policy.  Prompt
alignment is exact: all arms contain 147 prompt token IDs with frozen i32le
SHA-256 `f1ed3971af8259f3b9241d92404ec8d45f34137dec027fd96bc8d23b91b9773c`.
The v2 control reproduced the invalid identifier boundary
`618,21055,10666` (`get`, `_we`, wrong continuation), and the HTTP layer
suppressed the malformed tool payload.

A label-derived mechanistic oracle raised only base position 158 from the H23
table to Budget96 across the routed layers.  It changed exactly one of 45
generated IDs: boundary token `10666` became native/teacher token `2800`.
The remaining 44 IDs were identical.  The response then contained exactly one
valid `get_weather` call with JSON argument `{"location":"San Francisco"}`
and finish reason `tool_calls`.

| arm | valid tool | logical GiB/position | physical GiB/position | prefill | decode |
|---|---:|---:|---:|---:|---:|
| H23 base | no | 1.34037 | 0.46749 | 130.81 s | 40.08 s / 45 tokens |
| position-158 B96 | **yes** | 1.35993 | 0.47834 | 137.26 s | 39.64 s / 45 tokens |

The rescue added 6,624 selected records, 3.73651 GiB logical traffic and
2.07259 GiB physical reads over the complete 191-position sequence.  Averaged
over the whole sequence, the premiums are 0.01956 logical GiB/position and
0.01085 physical GiB/position: +1.46% and +2.32%, respectively.  There was no
request-wide Budget96 marker.  The preregistered four-position window is
therefore skipped.

This is a measured **progressive-rescue GO**, not a deployable policy.  It
falsifies the need to pay global Budget96 for this failure and shows that a
sparse higher-fidelity decision can repair a confident wrong identifier at a
small sequence-average byte premium.  It does not identify that decision
without labels, establish broad tool/coding reliability, or measure terminal
full-vocabulary KL.  The cold token-sequential timings are not serving
throughput.  The next earned experiment is a runtime grammar/tool-schema
boundary trigger followed by the same fixture and non-tool false-positive
controls; no production transplant is earned yet.

The immutable summary is
`results/k3_terminal_bws_v2_progressive_tool_result_20260901.json`; raw roots
r64/r65 and analysis r67 remain on Lucebox4.  The earlier r63 arm is retained
as an excluded pilot because its branch lacked the promised generated-token
trace.
