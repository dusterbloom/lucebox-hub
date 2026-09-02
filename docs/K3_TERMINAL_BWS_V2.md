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

## Runtime tool-schema discriminator

The position oracle was replaced with a preregistered runtime signal.  For
each effective declared tool whose name tokenizes to at least two IDs, the
research hook raises the next routed forward to Budget96 only when committed
output ends with all name tokens except the final one.  `get_weather`
tokenizes as `[618,21055,2800]`, so `[618,21055]` is the measured boundary
signal.  The mechanism is enabled only by
`DFLASH_KIMI_EXPERIMENT_TOOL_SCHEMA_RESCUE=1`; production defaults remain
unchanged.

On the frozen tool fixture the runtime discriminator configured one prefix,
emitted exactly one rescue marker at base position 158, and returned the valid
`get_weather` call.  It used no static position environment value and no
request-wide Budget96 floor.  Its 45 generated IDs, terminal full-vocabulary
logit bytes and 92-row traffic TSV are byte-identical to the successful
position-158 singleton.  Thus the runtime signal reproduces the mechanism,
not merely the visible tool name.

A separate official-template no-tools control (`Reply with exactly OK.`) was
run off/on with the same executable.  Both arms produced the same 26 prompt
IDs, the same eight generated IDs and visible `OK`, byte-identical terminal
logits and traffic, and exactly 16,109,527,040 physical provider bytes.  The
enabled arm configured no prefix and emitted no rescue marker.  Direct-I/O
time varied and is intentionally not treated as a hook cost.

This is a measured **schema-rescue gate GO**.  It earns a broader
native-success tool suite and tool-declared false-positive controls, still on
the experiment branch.  It does not yet earn a production transplant:
one-token tool names are unsupported by this first discriminator, a no-tools
control cannot measure false positives inside a tool-declared turn, only one
real tool boundary has been rescued, and terminal KL was not captured for the
generated sequence.

The immutable summary is
`results/k3_terminal_bws_v2_schema_rescue_result_20260901.json`; raw roots
r68--r70 and analysis r71 remain on Lucebox4.

## Sub-24 base plus schema rescue

The next curve asked whether the same runtime trigger could make a uniform
sub-24 base viable.  Budget16 ran first under a preregistered stop rule;
Budget12 and Budget8 were contingent on its tool success.

Budget16 reduced logical authoritative traffic from 1.35993 to
**0.93371 GiB/provider-position** (-31.34%) and physical reads from 0.47834 to
**0.30537 GiB/provider-position** (-36.16%).  It therefore crossed the traffic
component of the <1.2-GiB milestone.  It failed the joint gate: generation
diverged at the first output token, never reached the declared-name suffix,
emitted no rescue marker, and returned ordinary text claiming sunny weather
instead of a tool call.

This is a measured **Budget16 progressive NO-GO** on the known tool fixture.
It falsifies the simple hypothesis that preserving only the final tool-name
boundary is enough below Budget24.  The damaging information is already
missing before the first generated token, so a downstream schema trigger
cannot recover it.  Per preregistration, Budget12 and Budget8 were not run and
the trigger tokens/positions were not tuned.

The next useful discriminator must cover terminal prompt state or detect
cheap/rich disagreement before the first emitted token.  A larger static
downstream rescue window is not justified by this result.  Any such gate must
account for its prompt-side bytes over the full sequence and preserve exact
KDA/MLA state semantics.

The immutable summary is
`results/k3_terminal_bws_v2_sub24_schema_result_20260901.json`; raw Budget16
root r72 and analysis r75 remain on Lucebox4.

## Budget20 interpolation

A preregistered one-shot Budget20 arm tested whether the Budget16 failure was
merely too far below the Budget24 quality threshold.  Budget20 used
**1.13254 logical GiB/provider-position** and **0.39364 physical
GiB/provider-position**, reductions of 16.72% and 17.71% from the valid
Budget24-plus-schema arm.  It therefore crossed the `<1.2 GiB` traffic
milestone.

Quality still failed.  Budget20 emitted the same first six wrong IDs as
Budget16, returned invented weather prose instead of a tool call, and never
reached the declared-name suffix, so no schema rescue fired.  This is a
measured **Budget20 schema NO-GO**.  Per the one-shot stop rule, Budget21--23
will not be tuned on this fixture.

The result closes uniform-budget interpolation as the shortest route below
Budget24.  The next earned experiment is a captured pre-first-token
discriminator: measure terminal logits at the frozen prompt under cheap and
rich policies, then test whether a narrow prompt-tail rescue is justified
before changing runtime state scheduling.

The immutable summary is
`results/k3_terminal_bws_v2_budget20_schema_result_20260901.json`; raw Budget20
root r76 and analysis r77 remain on Lucebox4.

## Prompt-tail causal discriminator

Four preregistered one-token arms isolated the terminal prompt distribution:
uniform Budget20, the behaviorally successful H23 Budget24 policy, Budget20
with only prompt position 146 raised to Budget24, and (conditionally) Budget20
with positions 139--146 raised to Budget24.  All used the same 147-token
official-template tool prompt and the same executable.  Every expected
position marker fired.

The final-row intervention was effectively inert: it reduced
`KL(P_B24 || P_arm)` from 0.23599 to 0.23409 (0.8%) and retained the wrong
first token.  The last-eight intervention was much stronger.  It reduced KL
to **0.06136** (74.0%) at **1.15200 logical / 0.40749 physical
GiB/provider-position**, but still selected token 1008 (`The`) rather than the
Budget24 tool-protocol token 163588.  The Budget24 token's margin improved
from -0.842 to -0.272, but remained negative.

This is a measured **prompt-tail NO-GO**, not a quality pass.  It falsifies a
one- or eight-row fixed tail as sufficient on the discovery fixture, while
showing that recent prompt rows carry a large share of the distributional
damage.  Per preregistration, the tail length will not be tuned further on
this fixture.  A longer/periodic refresh must be treated as a new hypothesis
and validated on held-out native-success fixtures rather than declared from
this near miss.

The immutable summary is
`results/k3_terminal_bws_v2_prompt_tail_result_20260901.json`; raw arms r78--r81
and analysis r82 remain on Lucebox4.

## Periodic fidelity refresh

A separate preregistered hypothesis tested whether uniform-Budget24 rows could
periodically limit recurrent-state drift under a Budget20 base.  Period 8
(positions 7 mod 8) and, conditionally, period 4 (positions 3 mod 4) were the
only allowed densities.  This was a label-free schedule, not a longer tail or
a tool-position oracle.

Period 8 reduced `KL(P_B24 || P_arm)` by 25.1%, to 0.17674, at 1.16488
logical GiB/provider-position.  Period 4 reduced it by **69.7%**, to 0.07151,
at **1.18524 logical / 0.42491 physical GiB/provider-position**.  Neither arm
restored the Budget24 first token.  Period 4 made the decision nearly tied,
but still wrong: the Budget24 token margin was -0.0862 and token 1008 retained
a +0.0862 top-1 margin.

This is a measured **periodic-refresh NO-GO**.  Occasional rich rows improve
the distribution but do not reset accumulated cheap-state error.  Denser
refresh would consume the remaining `<1.2 GiB` margin and approach Budget24
under another name, so this static schedule is closed.  The next experiment
must allocate information inside routed layer/route/slab computation or use a
real disagreement/risk observation; it should not tune modulo phase on the
discovery fixture.

The immutable summary is
`results/k3_terminal_bws_v2_periodic_refresh_result_20260901.json`; raw arms
r83--r84 and analysis r85 remain on Lucebox4.

## Equal-Budget20 route-prefix screen

The missing route axis was tested with a research-only width-one hook.  At the
same uniform Budget20 slab capacity, only the router's first 12, 8, 6 or 4
descending top-k routes were eligible for slabs/fallback.  Retained router
weights were unchanged and omitted routes contributed zero.  The rebuilt
route16 default first passed exact closure against the retained Budget20
control: prompt/generated IDs, full-vocabulary logits and traffic were
byte-identical.

Top-12 is the important result.  It restored the Budget24 first token with a
+0.587 margin, reduced `KL(P_B24 || P_arm)` from 0.23599 to **0.14083**
(-40.3%), and reduced logical traffic to **1.08289 GiB/provider-position**.
Selected sidecar bytes were unchanged; the 5.1% logical reduction came from
lower exact-fallback bytes.  Top-8 and top-6 also restored first-token top-1
at 1.02884 and 1.00600 GiB/position, but worsened full-distribution KL to
0.32730 and 0.44897.  Top-4 failed top-1 with KL 0.55657.

The preregistered strong gate required at least a 50% KL reduction, so this is
formally a **route-prefix gate NO-GO** and is not relabeled after seeing the
data.  Nevertheless, top-12 is the first sub-1.2 representation in this
campaign to restore the correct pre-tool behavior while improving KL.  Under
the stated priority of reliable behavior over a literal KL multiple, it earns
a separately preregistered full-generation validation with the existing
schema rescue.  That follow-up remains discovery evidence, not production
promotion.

The immutable summary is
`results/k3_terminal_bws_v2_route_prefix_result_20260901.json`; raw route16
closure r86, route arms r87--r90 and analysis r91 remain on Lucebox4.

## Route12 full-tool validation

Because route12 restored the discovery fixture's first token while materially
improving KL and bytes, a separate preregistration tested it across the full
tool sequence with the already-validated declared-name schema rescue.  The
formal route-screen NO-GO label was retained.

Route12 returned exactly one valid `get_weather` call for San Francisco.  The
schema rescue fired once at base position 158, and all **45 generated IDs were
identical** to the valid Budget24-plus-schema control.  Final logits were not
byte-identical, so token equality is not reported as distributional
equivalence.

Average traffic was **1.10947 logical / 0.39270 physical
GiB/provider-position**, versus 1.35993 / 0.47834 for the valid Budget24
control.  Selected sidecar plus fallback traffic totaled 227,535,917,056
bytes over 191 positions.  The cold run reported 96.864 s prefill and 31.114 s
for 45 decode tokens (~1.4 tok/s); this is an improvement over the cold
Budget24 control but remains neither the 10 tok/s target nor a serving
throughput measurement.

This is a measured **route12 tool GO** and the first sub-1.2 policy in this
campaign to preserve the known full tool sequence.  It earns held-out
native-success coding/reasoning/multilingual/extraction tests plus additional
tool-declared controls.  It does not earn a production transplant from one
discovery fixture.

The immutable summary is
`results/k3_terminal_bws_v2_route12_tool_result_20260901.json`; raw arm r92
and analysis r93 remain on Lucebox4.

## Route12 native-success alignment gate

The frozen six-task H23 native-success suite was replayed without tuning at
route12, uniform Budget20 and schema rescue enabled.  All six candidate texts
were behaviorally correct: Tokyo, 10, 42, the corrected apples sentence,
Buongiorno, and LIME-742.  Against the old native output IDs, five of six
sequences matched; Italian added a period.  This is useful descriptive
evidence, but the preregistered gate remains **invalid**, not GO: the current
official-template tokenizer disagreed with the immutable native capture on
`grammar-apples` (38 versus 37 prompt tokens) and `extract-code` (50 versus
48).  The other four prompt sequences aligned exactly, with three of four
generated sequences exact.

Measured candidate traffic over all 288 provider positions was **1.00369
logical / 0.38537 physical GiB/position**, including 0.05894 GiB/position of
exact fallback.  No no-tool request configured or fired the schema rescue.
Thus route12 has crossed the `<1.2 GiB` behavioral milestone on these short
tasks, but this run cannot supply native terminal KL or a valid six-task
exactness count.

The first attempted binary closure reran only the two misaligned prompts
through the same executable's `native-exact` provider.  Its prompt IDs and
binary identity closed, but the model path was the sparse P32 package: all 276
routed tensor payloads are filesystem holes.  The exact provider therefore
read holes instead of routed weights and emitted prompt-tail fragments.  This
is an **INVALID CONTROL**, not a native-model NO-GO.  The immutable result is
preserved, and the harness now refuses `provider=exact` when any GGUF shard is
sparse.

Fresh asset inventory found the complete pinned UD-IQ1_S source on ds-033 at
`/mnt/kimi-k3`, even though it is no longer resident on Lucebox4.  The proven
P38 packer produced a new exact non-routed sparse core: 14 shards, 2,573
tensors, 276 routed holes and 48,692,428,800 allocated bytes.  Its manifest
SHA-256 is `37fa1184...`.  The canonical output-hash map for all 92 natural
sidecars is byte-identical between the source bank and Lucebox4
(`14257820...`).  A pinned 290-range fetch plan now transfers only those 48.69
GB and verifies every range before Full192 closure.  This replaces the false
claim that a new teacher model must be sourced.

The historical next gate remained behavioral and used previously unseen
textual fixtures which immutable native artifacts had already passed.  Prompt
alignment was recorded, and no KL or token-distribution claim was made when it
did not align.  Native terminal KL becomes measurable only after the recovered
Full192 source-matched core passes its registered closure.

Immutable summaries are
`results/k3_terminal_bws_v2_route12_native_success_invalid_20260901.json`
(raw r94--r99, analysis r100) and
`results/k3_terminal_bws_v2_route12_native_closure_result_20260901.json`
(raw r101--r102, analysis r103).

## Recovered native Full192 teacher closure

The source-matched sparse core was fetched directly on Lucebox4 from 290
pinned UD-IQ1_S byte ranges using four workers.  Every range SHA-256 and every
combined shard range SHA-256 matched the ds-033 source.  The transfer took
812.36 seconds and allocated exactly 48,692,428,800 bytes while preserving the
594,040,923,616-byte logical GGUF geometry.  Together with the already
identical 92-layer natural sidecar bank, Full192 is the exact stored K3
representation under test.

The preregistered closure is **3/3 GO**.  `fact-capital` matched both the old
native prompt and all eight native output IDs exactly and returned `Tokyo`.
The two controls corrupted by the prior hole-reading run now return
`She doesn't like apples.` and `LIME-742`; both pass their frozen behavioral
validators.  This confirms that r101/r102 was an invalid sparse-provider
control, not a K3 quality failure.

Full192 consumes exactly 9.069580078125 logical GiB/provider-position.  Its
cold short-request prefill measured 293.8--497.8 seconds and decode only
0.1--0.2 tok/s.  Those figures characterize the teacher/control and are not a
serving target.  The recovered teacher now earns aligned H23 and route12/B20
terminal-KL measurements on the unchanged tool boundary; candidate traffic
and performance must still be measured separately.

The immutable closure summary is
`results/k3_native_sidecar_closure_result_20260901.json` (SHA-256
`d095f3d4a298a217b47a13c012706fe29f10caffafe071c21264f0888f2fddec`);
raw arms r137--r139, range-fetch result `e2b2699f...` and analysis r140 remain
on Lucebox4.

## Unseen route12 behavioral holdout

The next gate froze the six tasks from the immutable 12/12 broad
native-success suite which route12 had not seen: a photosynthesis fact, a
Python list-comprehension function, rate reasoning, subject/verb agreement,
Spanish translation and decoy-resistant code extraction.  The gate was
behavioral because the then-used sparse exact control was invalid and the
source-matched core had not yet been recovered on Lucebox4.

The first analysis incorrectly scored the valid chemical formula `CO₂` as a
failure because the copied normalizer accepted ASCII `CO2` but had omitted
the original H23 scorer's subscript-digit translation.  That 5/6 result was
preserved unchanged.  An analysis-only amendment restored exactly the old
normalization rule, pinned the changed scorer hash and reran **zero** model
arms.

The corrected result is **6/6 behavioral GO**.  The code function and three
other outputs matched frozen native output IDs, while science answered the
shorter `CO₂` and Spanish added a period.  Four prompts aligned to the old
native tokenization; two of those four had exact generated sequences.  These
counts are diagnostic and are not called distributional equivalence.

Across 371 provider positions, traffic was **1.00675 logical / 0.37613
physical GiB/position**, with 0.06201 GiB/position exact fallback.  No no-tool
request configured or fired schema rescue.  This clears the `<1.2 GiB`
behavioral milestone on twelve short native-success tasks plus the full
`get_weather` discovery sequence.  It remains about 1.9x above the 0.53-GiB
decode target, cold request timing is not serving throughput, and native
terminal KL remains unresolved.

The corrected immutable result is
`results/k3_terminal_bws_v2_route12_behavioral_holdout_result_20260901.json`
(SHA-256 `bf55bddb724b15cfcff1156f5b47b639a113882a875d78faddf231dba5af13cf`);
raw arms r104--r109 and analyses r110--r111 remain on Lucebox4.  The next gate
is tool-declared false-positive behavior plus JSON/structured and longer
coding-agent tasks.  Production remains untouched.

## Tool-declared false-positive controls

Two `tool_choice=auto` requests declared different tools but explicitly
required a plain answer: `OK` with `get_weather` available, and `42` with
`lookup_customer_order` available.  Each was run at frozen route12/Budget20
with schema rescue off and on.

Both pairs passed.  The on arms configured their declared-name prefixes but
fired zero rescue markers and emitted no tool calls.  Off versus on was
byte-identical in prompt IDs, generated IDs, final full-vocabulary logits,
traffic TSV, logical bytes, fallback bytes and physical direct-read bytes.
The OK control measured 1.07385 logical / 0.38622 physical GiB/position; the
math control measured 1.06390 / 0.38298.

This is a measured **tool false-positive GO** for these two names and prompt
shapes.  It validates the narrow trigger mechanism, not every possible
shared-prefix grammar state.  The immutable result is
`results/k3_terminal_bws_v2_route12_tool_false_positive_result_20260901.json`
(SHA-256 `d3876477ca2e2e98d094a2d606137d3c5d1fb7cb03a18fbdea69c78b9eb4522d`);
raw arms r112--r115 and analysis r116 remain on Lucebox4.

## Structured JSON and coding-agent gate

The last cheap reliability discriminator compared H23 aggressive-1.8 with
the frozen route12/Budget20 candidate on strict JSON emission and a small
`merge_intervals` repository repair.  The validator required exact JSON
semantics/key order and executable Python that passed five restricted
functional and input-mutation checks.  Controls had to pass before route12
could be scored.

Two infrastructure defects were preserved rather than silently retried.  The
first campaign clamped the registered 128-token coding response to 64 tokens;
its immutable analysis is
`results/k3_terminal_bws_v2_route12_structured_agentic_invalid_20260901.json`.
After raising only that server cap, the unchanged 144-token prompt plus
128-token completion exceeded the runner's 256-token context and was rejected
before inference.  That attempt is recorded in
`results/k3_terminal_bws_v2_route12_structured_agentic_cap256_infra_failure_20260901.json`.
The final preregistered campaign used a 512-token context and otherwise
unchanged model-facing inputs.

Strict JSON passed both policies.  Route12 emitted the requested object at
**1.01268 logical / 0.36354 physical GiB/provider-position**, versus H23's
1.85036 / 0.71965, a 42.5% total-provider-byte reduction.  Route12 prefill was
41.0 s and its 25-token decode 12.8 s (2.0 tok/s); H23 measured 49.0 s and
20.8 s (1.0 tok/s).  These are cold request measurements, not serving
throughput.

The coding fixture is formally **INVALID_CONTROL**.  With the registered
128-token allowance, H23 still ended at `finish_reason=length` one assignment
short of syntactic completion.  Route12 first emitted a complete plausible
implementation, then began emitting a duplicate function and was also cut at
128 tokens; the registered whole-response validator therefore rejected it as
well.  Because the control failed, this is neither a route12 GO nor NO-GO, and
the token limit/prompt will not be tuned after seeing the outputs.  Diagnostic
traffic was 1.83089 / 0.61507 GiB per position for H23 and **0.99050 /
0.29898** for route12; corresponding cold prefill/decode times were
131.8/107.0 s and 100.0/72.6 s.

The immutable final analysis is
`results/k3_terminal_bws_v2_route12_structured_agentic_result_20260901.json`
(SHA-256 `38b888e5b80df27c49d9b1c406198305236fb0b2e48160abbbb6e4763a7d465c`);
raw arms r126--r129 and analysis r130 remain on Lucebox4.  This strengthens
structured-output evidence but does not establish coding-agent retention.
Production remains untouched.  The next earned discriminator is joint
route/slab reduction below the route12/Budget20 ~1.0-GiB regime, starting at
route12/Budget16 on the known tool boundary before any broad rerun.

## Route12 plus uniform Budget16

The first joint route/slab discriminator compared route16/Budget16 with
route12/Budget16 at the frozen first `get_weather` output position.  Both used
the same executable and uniform Budget16 policy; the only intervention was
the native descending route prefix.  The preregistered GO required the
Budget24-reference top-1 with positive margin, at least 20% KL recovery versus
route16/Budget16, fewer bytes, and at most 0.90 logical GiB/position.

Route12 met the byte threshold at **0.88991 logical / 0.30973 physical
GiB/provider-position**, 5.7% less logical traffic than route16/Budget16 and
0.193 GiB below route12/Budget20.  It failed behaviorally: both B16 arms chose
token `1008` instead of the Budget24 reference token `163588`.  Route12 made
the teacher-token margin worse from -0.260 to **-1.060** and increased
`KL(P_B24 || P_arm)` from 0.20835 to **0.25886** (+24.2%).

This is a measured **route12/Budget16 NO-GO**.  Route removal recovered the
boundary at Budget20 but cannot replace the information lost by lowering the
uniform slab budget to 16.  No full tool generation or broad quality run is
earned.  The immutable result is
`results/k3_terminal_bws_v2_route12_budget16_result_20260901.json`; raw arms
r131--r132 and analysis r133 remain on Lucebox4.  The next useful discriminator
must change where the 16-slab-average capacity is spent (layer-adaptive
allocation) or its representation; tuning the global route count is closed at
this budget.

## H22-ranked equal-average Budget16 allocation

The frozen H22 isolated Budget96 layer atlas was used only as a rank prior.
Its source was pinned to `perf/k3-layer-major-prefill` commit `102cf35a`, blob
`9efa6628`, SHA-256 `a492f084...`; no low-budget projected KL was treated as
measured.  The 23 most tolerant, 46 middle and 23 most sensitive layers were
assigned either conservative `12/16/20` budgets or sharp `8/16/24` budgets.
Both policies total exactly 1,472 nominal slabs, equal to uniform Budget16,
and both used route12.

Neither policy restored the Budget24-reference top-1.  The conservative arm
reduced KL from uniform route12/B16's 0.25886 to **0.22907** (-11.5%) and
improved the teacher-token margin from -1.060 to **-0.268**, at **0.88819
logical / 0.30844 physical GiB/position**.  The sharp arm reached the closer
margin **-0.092**, but KL was 0.25540 (only 1.3% better), at 0.88960 / 0.30542.
Both still selected token `1008`.

This is a measured **H22 average-B16 rank-transfer NO-GO** under the
preregistered 20% KL plus top-1 gate.  It shows that layer sensitivity is
useful but insufficient: the isolated Budget96 ordering partially transfers,
while composed low-budget interactions change enough that coarse quartiles do
not recover behavior.  Quartile membership will not be tuned on this fixture.
The immutable result is
`results/k3_terminal_bws_v2_route12_h22_avg16_result_20260901.json`
(SHA-256 `c2222b1061890f83f8b785ec01e107449de4e39a3676656620fe55f6ce47d629`);
raw arms r134--r135 and analysis r136 remain on Lucebox4.

## Source-matched native tool-boundary KL

The recovered Full192 teacher and all three frozen comparison policies were
run on the identical 147-token official-template `get_weather` history using
one executable and source commit.  The current H23 aggressive-1.8 and
moonshot/Budget24 controls both missed native top-one ID `163588`; they chose
ID `65447`.  Route12/Budget20 alone selected the native token:

| arm | logical GiB/position | physical GiB/position | KL(native || arm) | native-token margin | top-one |
|---|---:|---:|---:|---:|---|
| native Full192 | 9.06958 | 0 (mmap reference) | 0 | +0.07571 | native |
| H23 aggressive-1.8 | 1.95014 | 0.75472 | 0.39840 | -0.99215 | wrong |
| H23 moonshot/Budget24 | 1.33002 | 0.48327 | 0.47871 | -0.06470 | wrong |
| route12/Budget20 | **1.08185** | **0.39288** | 0.65615 | **+0.33290** | **native** |

This is the preregistered **behavioral GO**, not the strong distributional GO.
Route12 uses 44.5% fewer logical bytes than aggressive-1.8 and repairs the
known boundary, but its terminal KL is 64.7% higher.  The result directly
falsifies using a scalar KL threshold such as `0.13` as a universal safety
rule: native itself has only a +0.0757 top-token margin here, and a lower
whole-vocabulary KL can still move the wrong competitor across that narrow
boundary.  Route pruning is therefore a useful behavioral intervention, not
evidence that route12 is generally more native-like.

Full192 cold prefill took 1,465.99 s; aggressive-1.8, moonshot and route12 took
140.67, 120.46 and 105.26 s respectively for this 147-token request.  These
are token-sequential cold measurements, not serving throughput.  The next
earned step is broader source-matched native KL on already-captured teacher
fixtures, with exact history alignment required before any KL is scored.

The immutable summary is
`results/k3_native_tool_first_token_result_20260901.json` (raw arms r141--r144,
failed NumPy-only analysis r145, standard-library analysis r146).  The
analysis-only amendment reran zero model arms.  Production remains untouched.

## Reused-native terminal suite

The three already-closed Full192 fact, grammar and extraction teachers were
reused without another native run.  H23 aggressive-1.8, H23
moonshot/Budget24 and route12/Budget20 were rerun against the source-matched
non-routed core.  A terminal vector was scored only when all eight generated
IDs matched its teacher history.

All nine candidate arms aligned.  Every policy retained all 3/3 native
sequences and final top-one decisions:

| policy | logical GiB/position | physical GiB/position | mean KL | median KL | p95 KL | max KL |
|---|---:|---:|---:|---:|---:|---:|
| H23 aggressive-1.8 | 1.84465 | 0.73744 | 0.000227 | 0.000001 | 0.000611 | 0.000679 |
| H23 moonshot/Budget24 | 1.22954 | 0.48342 | 0.002462 | 0.002653 | 0.004503 | 0.004708 |
| route12/Budget20 | **1.00953** | **0.39436** | 0.010998 | 0.003963 | 0.026487 | 0.028989 |

This is another preregistered **behavioral GO**, not a strong distributional
GO.  On these ordinary, high-margin terminal positions route12 is comfortably
below KL `0.13`; its worst value is `0.02899`.  On the hard tool boundary it
was `0.65615` yet selected the native token while both lower-KL H23 policies
failed.  KL risk is therefore sharply position-dependent, and a single global
tolerance cannot certify reliability.  The data favor a cheap route12 base
plus a sparse boundary-risk mechanism; they do not favor globally paying for
H23 fidelity.

Across the three route12 requests, 119 prompt tokens took 89.36 s
(~1.33 prefill tok/s) and 24 decoded tokens took 12.11 s (~1.98 tok/s).  These
remain cold token-sequential request measurements, not the 100/10 tok/s serving
targets.  The immutable result is
`results/k3_native_terminal_reuse_suite_result_20260901.json` (SHA-256
`caa095e6cb42dc94f4fc50d8176732fcb9efbdbeda879c5c2cac0bf1a152980b`);
raw candidate arms r147--r155 and analysis r156 remain on Lucebox4.

## Route12/B16 base and B16/B20 disagreement

Four source-matched B16 arms reused the existing native and route12/B20
artifacts.  On fact, grammar and extraction, B16 retained all 3/3 native
eight-token sequences and final top-one decisions at **0.82179 logical /
0.32517 physical GiB/provider-position**.  Mean/median/p95/max terminal KL
were `0.01304/0.00303/0.03228/0.03553`, close to B20's
`0.01100/0.00396/0.02649/0.02899` while using 18.6% fewer logical bytes.

At the source-matched `get_weather` boundary, B16 selected wrong ID `1008`
with native-token margin `-0.24390` and KL `0.97838`; B20 selected native ID
`163588` with margin `+0.33290` and KL `0.65615`.  B16 and B20 agreed on every
ordinary final top-one and disagreed at the known failure.  This passes the
preregistered **B16/B20 disagreement GO** and establishes B16 as a promising
cheap base, not as a globally safe policy.

The captured B16 top-one margins were 8.57, 6.61 and 4.99 on the three
ordinary rows versus only 0.244 on the tool failure.  This is a useful
discovery observation, not a validated threshold: entropy/margin alone is
explicitly insufficient, and the tool row was already known.  Two complete
B16 and B20 passes would also erase the traffic win.  The next earned work is
a held-out one-pass resident-state risk predictor or an incremental
B16-to-B20 hydration experiment; no broad progressive subsystem is earned.

Across all four rows B16 measured 0.85682 logical / 0.31711 physical
GiB/position, but that aggregate includes the failed tool token and is not a
quality-preserving frontier point.  The immutable result is
`results/k3_b16_b20_disagreement_result_20260901.json` (SHA-256
`56f84a2c064a8a73e992dcf3473fd2556ab9f85ce69dca20826d7b520e931828`);
raw B16 arms r157--r160 and analysis r161 remain on Lucebox4.

## Hy4/STQ low-bit complement screen

Hy4 revision `779242edccdedc2109a0b36b164263a88f015bfa` was reviewed as a
native-source PTQ and mixed-precision process.  Its STQ1_0 format is 1.3125
bpw and protects routed down projections at higher precision.  Blanket STQ
can reduce one K3 gate/up/down route-slab record by only 16% versus the current
IQ1_S bytes, so it is not an independent traffic solution.  Its possible K3
value was additional approximate tail coverage at Budget16 bytes.

The exploratory four-row exact-core screen narrowly favored eight exact
IQ1_S records plus nine Hy4-style STQ gate/up tails, but the frozen held-out
test reversed it.  Across 12 validation rows from 10 sequences, exact B16 had
mean local relative L2 `0.711892`; the mixed arm had `0.728829` at `1.0025x`
bytes and lost all 12 rows.  The held-out preregistration therefore gives a
**NO-GO** for STQ derived from the already-lossy K3 IQ1_S source.  No HIP
kernel is earned.

This does not close STQ fitted directly from BF16/native K3 weights with a
route-aware imatrix.  Such an arm must first beat scalar alternatives at equal
bytes on held-out captured states and then terminal KL before runtime work.
The immutable decision is
`results/k3_stq1_exact_core_holdout_decision_20260901.json`; the raw harness
artifact is preserved even though its older exploratory classifier printed
`RETAIN_LOW_PRIORITY`.

The reusable Hy4 lessons, provider package contract, ROCm 10 qualification
gate, and cross-model process are recorded in
`docs/LUCEBOX_LARGE_MOE_QUALIFICATION_V1.md`.  Production
`perf/k3-production-ponytail` remains untouched at
`fac048c090c74e5f8f989bffcda3aadc0bc8c266`.

## GSQ-RCO allocator watch

The Qwen3.8-27B GSQ-RCO release was frozen at Hugging Face revision
`888cc868537099e09a9c4f41a2b9a421b346f88b`. It is not a K3 or large-MoE
result, but it contributes two distinct candidates. GSQ refines scalar
assignments and scales while emitting standard GGUF formats, so an equal-byte
same-format GSQ arm is now preferred over earning a new codec kernel. RCO
optimizes a non-decomposable objective under an exact byte budget, which is a
plausible future allocator for `route x slab x fidelity`.

RCO is deferred: current K3 captured interventions are not differentiable,
and GSQ itself is trained against layer reconstruction rather than terminal
KL. The next earned test is a route-aware, held-out, same-format GSQ screen on
representative gate/up/down tensors or slabs. It advances to terminal
intervention only after beating the current quantizer at identical stored
bytes without catastrophic individual-row regressions. RCO advances only
after the Phase B terminal predictor beats residual/norm ranking on held-out
layers. No GSQ-specific decoder, HIP kernel, or large allocator subsystem is
earned by the published artifact alone. The immutable review is
`results/lucebox_gsq_rco_artifact_review_20260901.json`.

## Route12/B16 V16 target-only oracle

The V16 gate resumed on Lucebox4 from clean experiment commit `51152192`.
Before model work, a transcription error in the preregistered 53-ID prompt
hash was corrected by an immutable amendment; target IDs, Budget16 policy,
protocol and gates did not change. Targeted HIP executables and the focused
provider, ordered-join and stream tests passed on physical `gfx1151`.

The scalar route12/B16 arm froze 18 candidate IDs and complete terminal logits
and state. The V16 arm then failed the preregistered runtime gate before one
target graph executed:

```text
Kimi-K3 persistent exact Core8 failed at layer 0:
persistent exact Core8 shape mismatch at layer 0
```

The experimental guard admitted widths divisible by eight, but the persistent
Core8 evaluator still requires `token_begin == 0` and a hidden vector of
exactly `8 * n_embd`. The width-16 vector therefore fails at the first call;
there is no target timing to interpret, and automatic clocks do not affect the
NO-GO. This closes the current single-call V16 patch, not a correctly
state-chained composition of two Core8 transactions. Per the registered stop
rule, it will not be repaired on the discovery fixture.

The next earned lane is the smallest incremental B16-to-B20 hydration and
held-out one-pass risk discriminator. The immutable result is
`results/k3_v16_route12_budget16_oracle_result_20260902.json`; raw roots r162
and r163 remain on Lucebox4. Production received no change.

## Incremental B16 to B20 composition

The host-reference discriminator evaluated ordinary Budget20 and the exact
same 20 selected records as a Budget16 base followed by four corrections of
`exact slab output - stored slab mean`.  The first layer-1 preflight exposed a
cross-model hash error in the original preregistration: its frozen teacher hash
came from the older p32-core artifact while the registered model was the
native-IQ1S slim core.  That preregistration remains immutable, layer 1 is
classified as development only, and untouched layers 46 and 92 were registered
separately before execution.

Both held-out layers passed more strongly than required.  Control and
incremental terminal vectors were byte-identical: maximum logit delta `0`,
composition KL `0`, equal top-one, equal selected plans, equal logical bytes,
and equal exact-teacher trajectories.  Their terminal KL versus the exact
teacher was therefore unchanged (`0.0021161811` at layer 46 and `0.4437261734`
at layer 92).  This proves the four-record correction arithmetic for the
tested path; it does not measure selective rescue traffic or serving speed.

The next earned experiment is an all-layer, on-policy B20 control versus
B16-plus-four composition on one ordinary sequence and the registered tool
boundary.  Production remains untouched.  The immutable held-out result is
`results/k3_incremental_b16_b20_heldout_result_20260902.json`; raw roots
r166-r170 remain on Lucebox4.
