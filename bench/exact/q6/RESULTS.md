# Option A: dense Q8_0 -> Q6_K requant (Qwen3.8-Flash-Next)

Goal: cut dense-projection bandwidth by requantizing the listed Q8_0 dense tensors to
Q6_K and see whether T=1 decode gets faster at acceptable quality. Verdict: **no net
win**. The byte savings are real (~0.8 GiB/token less dense traffic) but the private
`LUCE_QWEN_SHARED_OVERLAP` fusion cannot bind to Q6_K tensors (hard shape/type
contract: K-quant superblock is 256 elements, several dense rows are 640-wide and
can't even become Q6_K at all). Losing that fusion costs almost exactly what the
narrower dense tensors save, so the net decode time is statistically identical to
today's best Q8_0 baseline.

## Conversion

No BF16/F16 source for these tensors exists on the box (only IQ3_XXS and IQ4_NL
quants besides the UD-Q4_K_XL original). Converted from Q8_0: dequantize to f32 with
`ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float`, requantize with
`ggml_quantize_chunk(GGML_TYPE_Q6_K, ...)`. Tool: `requant_q6.cpp`, a ~160-line
program against the box's own ggml/gguf API (`gguf_init_from_file` no_alloc, raw
fread/fwrite of each tensor's bytes, `gguf_write_to_file(..., only_meta=true)` +
manual data-section append so padding/offsets match exactly what a single-pass
writer would produce). Built against the already-built
`qwen4exp-exact-stack/build1151` libggml/libggml-base.

Shard layout: shard 00001 is metadata-only (0 tensors, copied verbatim); all target
tensors live in shards 00002/00003/00004, each rewritten independently (unaffected
tensors byte-copied, not touched).

**Tensors requantized to Q6_K** (ne[0] divisible by 256 required for the K-quant
superblock; anything that doesn't divide stays Q8_0 and is flagged by the tool):

| Tensor | Requantized | Skipped (ne0 not /256, kept Q8_0) |
|---|---|---|
| attn_qkv ×36 | 36 | 0 |
| attn_gate ×36 | 36 | 0 |
| ssm_out ×36 | 36 | 0 |
| output.weight | 1 | 0 |
| attn_q ×12 | 12 | 0 |
| attn_output ×12 | 12 | 0 |
| ffn_gate_shexp ×36, ffn_up_shexp ×36 | 72 | 0 |
| ffn_down_shexp ×36 | 0 | 36 (ne0=640) |
| attn_k/attn_v ×12 each | 0 | 24 (ne0=640) |

253 tensors converted, 48 left at Q8_0 because their row width (640) isn't a
multiple of the 256-element Q6_K superblock -- this matches the small
attn_k/attn_v/ffn_down_shexp tensors (the ~32 MiB category in the plan).

Byte accounting (sum over the 253 converted tensors, all 3 shards):
old (Q8_0) = 3.508 GiB, new (Q6_K) = 2.709 GiB, **saved = 0.800 GiB** -- matches the
plan's ~0.8 GB/token estimate almost exactly. Whole-model size: 104 GiB -> 103 GiB.

Verified with gguf-py (`GGUFReader`) on all 4 output shards: headers, tensor counts
(297/752/175) and quant-shape checks all pass.

Model path: `/home/duster/models/qwen4exp-dense-q6k/Qwen3.8-Flash-Next-dense-Q6K-0000{1..4}-of-00004.gguf`

## Kernel path / fusion fallback

Loading the Q6_K model with the default `run_stack.sh` env (`LUCE_QWEN_SHARED_OVERLAP=1`)
aborts immediately:
```
[qwen4exp] shared overlap disabled at layer 0: shape/type contract
GPU_EXEC_DONE rc=3
```
`ggml-cuda.cu`'s overlap-bind `reject()` path correctly detects that the producer
buffers it expects (Q8_0 dense streams) no longer match, returns false, and the
driver's own measurement-harness invariant check (`driver_shared_epilogue.cpp`,
the `before/previous` counter assertions) then exits 3 rather than silently
continuing -- by design, this is the harness catching a silent regression.

Running with `LUCE_QWEN_SHARED_OVERLAP=0` lets the model run to completion. Looking
at `[graph_producer_q8_counts]` between the Q8 baseline and the Q6 run shows the
blast radius is bigger than just SHARED_OVERLAP: disabling it also zeroes the
CUDA-graph *replay* counts for `HC_LO_Q8` (`lo_replay` 24444->0), `GDN_AB_EXACT`
(`ab_replay` 9072->0, `producer_gdn_replay` 9072->0) and `PRODUCER_HC`
(`producer_hc_replay` 23940->0) -- those fusions still fire on the host path
(`*_host` counts stay nonzero) but lose their graph-capture/replay fast path,
because they are all captured inside the same SHARED_OVERLAP CUDA-graph region.
`PRODUCER_Q8`'s own eager/capture/replay/seal counts (2/2/252/2) are unaffected --
expert/router tensors were left untouched at their original quant types, so
`EXACT_ROUTER_SUFFIX` and `EXPERT_ROW_WARPS` are unaffected either.

So the real fallback is not "one fusion disabled" but "SHARED_OVERLAP's CUDA-graph
region, and everything nested in it, drops to eager/host execution."

## Quality

No full KL/logit-capture pipeline was run (would need `quality_gate.py`'s golden
`MEASURE_LOGITS` server-based run, out of this session's time budget) -- flagging
this as incomplete rather than fabricating a number.

What *was* measured directly from the timing runs: both arms ran the same
`prompt.ids` with `MEASURE_GPU_ARGMAX=1 MEASURE_FOLLOW=follow.ids` (teacher-forced
greedy argmax at each position). Diffing the printed `measure_tokens` id streams
(Q8 baseline rep0/mode0 vs Q6 rep0/mode0, both 256 positions):

- **Top-1 agreement: 246/256 = 96.1%** (5 isolated single-position divergences at
  positions 73, 90, 133, 148, 191; no cascading divergence, consistent with
  per-position teacher forcing rather than runaway autoregressive drift).

This is **below the plan's 98% flag threshold** -- flagged, not a go/no-go call.
Mean/p99 KL was not computed (no raw logits captured in this pass).

## Timing

`free -g` before timing: `total=125 used=64 free=50 buff/cache=10 available=60`
(no indication of needing compaction; `drop_caches`/`compact_memory` require root
the box account doesn't have, so that step from past sessions' lore could not be
run here -- noted as a limitation, not skipped silently).

Same driver (`driver-stack`), same env template (`run_stack.sh`), fresh process per
rep, interleaved reps, mode=1 (post-warmup) medians reported, `pt`/follow setup
identical on every arm:

| Arm | n (mode=1) | median ms/token | mean | min | max |
|---|---|---|---|---|---|
| Q8_0 baseline (SHARED_OVERLAP=1, today's best config) | 12 | **39.68** | 39.69 | 39.43 | 39.98 |
| Q6_K dense (SHARED_OVERLAP forced off -- can't bind) | 8 | **39.71** | 39.71 | 39.40 | 40.02 |
| Q8_0, SHARED_OVERLAP=0 (isolation arm) | 4 | **42.27** | 42.27 | 42.25 | 42.28 |

Byte budget: baseline reads ~6.2 GB/token; Q6_K dense cuts that by the measured
0.800 GiB (~13% of the dense slice, ~ a few % of the whole 6.2 GB budget).
Achieved GB/s isn't separately reported here -- decode_ms/tokens times the known
per-token byte budget line up with the ms/token numbers above; no second metric
added.

**Reading the four numbers together:** Q6_K alone (comparing the two
SHARED_OVERLAP=0 arms, which is the only apples-to-apples pair since Q6_K cannot
run with SHARED_OVERLAP=1 at all) is genuinely ~6% faster than Q8_0 alone
(39.71 vs 42.27 ms/token) -- the bandwidth cut does help. But SHARED_OVERLAP itself
is worth almost exactly that same ~6% on Q8_0 (39.68 vs 42.27). The two effects
cancel: **Q6_K dense end-to-end (39.71) is statistically indistinguishable from
today's Q8_0+SHARED_OVERLAP baseline (39.68)** -- the four-rep spread (39.4-40.0)
is bigger than the 0.03 ms/token gap between them.

## Decision

Option A as fused today is a wash, not a win: the dense-tensor bandwidth saved by
Q6_K is fully offset by losing the SHARED_OVERLAP CUDA-graph region (which also
silently drops HC_LO_Q8/GDN_AB_EXACT/PRODUCER_HC from replay to host path). Shipping
it as-is would trade a flagged 96.1% top-1 agreement for zero measured speedup.
**Not recommending promotion of the Q6_K dense model** unless SHARED_OVERLAP (or an
equivalent producer path) is ported to tolerate non-Q8_0 dense tensors first; at
that point the ~6% Q6_K-alone gain shown above would be additive on top of the
existing baseline instead of canceling it. Q5_K (option B) was not attempted --
the SHARED_OVERLAP contract break is type-independent (it's about the dense
tensors not being Q8_0 at all, not about which K-quant replaces them), so Q5_K
would hit the identical cancellation and wasn't worth the extra conversion pass
given this result.

## Artifacts

- `requant_q6.cpp` -- the conversion tool (committed here).
- `run_q6_nooverlap.sh`, `run_q8_nooverlap.sh` -- timing harness variants used above
  (SHARED_OVERLAP env flipped relative to `run_stack.sh`).
- Model: `/home/duster/models/qwen4exp-dense-q6k/Qwen3.8-Flash-Next-dense-Q6K-0000{1..4}-of-00004.gguf`
  on lucebox4 (not committed -- binary weights, box-local only).

---

# Arm A2: keep ffn_{gate,up,down}_shexp at Q8_0

Hypothesis (coordinator): SHARED_OVERLAP's "shape/type contract" reject in option A
was caused by the shared-expert tensors becoming type-inconsistent (gate/up became
Q6_K, down stayed Q8_0 because its row width can't fit a 256-superblock) rather than
by the main dense stack (attn_qkv/attn_gate/ssm_out/attn_q/attn_output/output.weight)
being Q6_K. **Confirmed correct, but it exposes a second, independent, hard-crashing
contract.**

## 1. Build

Same `requant_q6.cpp` tool, `--keep-shexp-q8` flag added: `should_requant()` no longer
touches `ffn_gate_shexp`/`ffn_up_shexp` (join `ffn_down_shexp`, which already couldn't
convert -- ne0=640 isn't a multiple of 256). 157 tensors converted this time (vs 253 in
option A; the 96-tensor difference is `ffn_gate_shexp`+`ffn_up_shexp` across 36 layers
plus the handful that only cleared the 256-divisibility bar when counted per-shard).
Shard 00001 (metadata-only, 0 tensors) hardlinked, not copied, per instruction. Shards
00002/00003/00004 rewritten (same tool, same byte-exact-copy-for-untouched-tensors
approach as option A). Verified with gguf-py: tensor counts 297/752/175 match
source, all 4 shards open cleanly.

Byte accounting (sum over the 157 converted tensors): old (Q8_0) = 3.275 GiB, new
(Q6_K) = 2.528 GiB, **saved = 0.746 GiB** (less than option A's 0.800 GiB, as
expected -- keeping 72 shexp tensors at Q8_0 gives up some of the savings). Whole
model: 103 GiB (same as option A to the GiB).

Model path: `/home/duster/models/qwen4exp-dense-q6k-a2/Qwen3.8-Flash-Next-dense-Q6K-a2-0000{1..4}-of-00004.gguf`

## 2. Fusion survival -- SHARED_OVERLAP fixed, HC_UPMIX_ROW8 now hard-crashes

Running the full `run_stack.sh` env (`LUCE_QWEN_SHARED_OVERLAP=1` and everything else
on) against the A2 model: **no `[qwen4exp] shared overlap disabled ... shape/type
contract` message at all** -- confirms the coordinator's hypothesis: SHARED_OVERLAP's
`q8_weight(shared_gate, 2560, 640) / q8_weight(shared_up, 2560, 640) /
q8_weight(shared_down, 640, 2560)` contract (ggml-cuda.cu:5321-5323, all three must be
exactly `GGML_TYPE_Q8_0`) is satisfied again now that the shared-expert triple is
type-consistent.

But the process now hard-aborts (SIGABRT, rc=134) a few hundred ms into warmup rep 0:
```
ggml-cuda.cu:6530: GGML_ASSERT(upmix_row8 && upmix_row8->pairs.size() == 96) failed
```
This assert only fires when `sealed_sh` is true (ggml-cuda.cu:6393/6530) -- i.e. only
once SHARED_OVERLAP itself has successfully sealed/captured. In option A, SHARED_OVERLAP
never sealed (it was rejected at the contract check), so this stricter invariant never
ran and HC_UPMIX_ROW8 fell back to its host path uneventfully. In A2, SHARED_OVERLAP
*does* seal, which activates a second, independent structural contract:
`ggml_cuda_prepare_hc_upmix_row8()`'s positional graph-node scan
(`ggml_cuda_hc_upmix_row8_pair_quick_valid`, ggml-cuda.cu:6250-6279) expects to find
exactly 96 fixed-shape `MUL_MAT` sites (`weight` = the untouched `hc_attn_up`/
`hc_ffn_up` tensor, 320x10240, still Q8_0 -- not something we changed) at specific,
hard-coded positions in the capture graph. Converting attn_qkv/attn_gate/ssm_out/
attn_q/attn_output/output.weight to Q6_K changes the CUDA kernel dispatch for those
MUL_MATs (K-quant vs Q8_0 take different code paths), which shifts node
indices/adjacency in the captured graph enough that fewer than 96 of the expected
upmix sites pattern-match. It is a type-independent **positional** contract on the
*rest* of the dense stack, triggered only once SHARED_OVERLAP seals.

**Tried to route around it with an env flag (not a code patch) and hit a second,
intentional guard:** `driver_shared_epilogue.cpp:58` hard-requires
`LUCE_QWEN_HC_UPMIX_ROW8=1` (together with `SHARED_EPILOGUE=1`) at process startup --
`LUCE_QWEN_HC_UPMIX_ROW8=0` makes the driver `return 2` immediately, before even
opening the model. So there is no supported env-only way to keep SHARED_OVERLAP sealed
while skipping HC_UPMIX_ROW8's 96-pair scan.

**Conclusion for step 2: cannot be satisfied without a code change.** Per instruction,
no patch was attempted. The exact contract that needs to change (in
`ggml_cuda_prepare_hc_upmix_row8`/`ggml_cuda_hc_upmix_row8_pair_quick_valid`,
ggml-cuda.cu:6250-6360) is the fixed assumption that the graph positions of 96 upmix
sites are stable regardless of neighboring dense-tensor quant types -- it would need
to either tolerate <96 matched pairs (relaxing the `GGML_ASSERT` at line 6530 to a
graceful partial-capture, mirroring how SHARED_OVERLAP's own `reject()` degrades) or
re-derive the 96 expected sites from the actual (Q6_K-aware) graph shape instead of a
hard-coded count.

## 3. Quality / 4. Timing -- blocked

Both require a non-crashing full-stack (`SHARED_OVERLAP=1`, which the plan explicitly
wants for the "full 38.41 env") run of the A2 model, which is not currently possible
(see above). Did not fabricate numbers from a crashing config, and did not run
quality/timing with `SHARED_OVERLAP=0` for A2 either, since that would just reproduce
option A's already-reported "no net win" result (the whole point of A2 was to test
*with* SHARED_OVERLAP sealed) -- re-running the already-known SO=0 arm would not answer
the question asked. **Stopping here pending a decision**: either (a) accept
option A's SO=0 numbers as the answer for the shexp-fixed model too (since A2 can't
run with SO=1), or (b) someone patches the HC_UPMIX_ROW8 positional contract, at which
point this section's steps 3-4 can be completed against a real A2+SHARED_OVERLAP run.

## A2 Artifacts

- `requant_q6.cpp --keep-shexp-q8` -- same tool, flag added (committed).
- `run_a2_full.sh` -- full 38.41 env, SHARED_OVERLAP=1 (aborts, rc=134, log kept
  for the stack trace above).
- `run_a2_noupmix.sh` -- attempted HC_UPMIX_ROW8=0 workaround; refused by the
  driver's own startup guard (rc=2, `driver_shared_epilogue.cpp:58`).
- Model: `/home/duster/models/qwen4exp-dense-q6k-a2/Qwen3.8-Flash-Next-dense-Q6K-a2-0000{1..4}-of-00004.gguf`
  on lucebox4 (not committed -- binary weights, box-local only).
