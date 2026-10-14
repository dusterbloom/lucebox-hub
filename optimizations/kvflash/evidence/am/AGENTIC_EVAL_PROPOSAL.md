# Proposal: agentic long-session evaluation for AM-compacted KV cache

Status: proposal only. Nothing paid was run. Dataset statistics below were measured on public data by downloading one shard (or the single file) and tokenizing a sample with the Qwen3 tokenizer (Qwen3-0.6B tokenizer files, same vocabulary as Qwen3-4B); where only characters were counted this is said. Cost numbers are estimates from stated assumptions, not measurements; the pilot in section 4 replaces them.

Context: R0/R1b measured AM on QuALITY (single-document reading): Qwen3.8-27B loses about 3 points at 0.10 (R1b.md, F floor 86.6 vs 89.6 original), Qwen3-4B none measurable (R0_summary.md, S1 -1.68 [-4.49, +1.12] held-out). Engine config "F" = context-prefill queries, HighestAttnKeys, per-head budgets, on-policy. PLAN.md "Evaluation for agentic coding" items 1-2 (teacher-forced replay, recall probes) are what this document makes concrete; item 3 (SWE-bench outcome) is out of scope.

## 1. Datasets

Measured on Hugging Face (API + parquet), 2026-10-14. Token counts: Qwen3 tokenizer on a sample of whole trajectories (messages + tool calls; the tool-schema column is not counted, add ~2K).

| Dataset | Rows | License | Format | Generator | Resolved label | Median tok | >16K / >32K / >64K |
|---|---|---|---|---|---|---|---|
| **nebius/SWE-rebench-openhands-trajectories** | 67,074 | cc-by-4.0 | OpenAI-style messages (system/user/assistant+tool_calls/tool), `tools` column, one parquet 2.08 GB | Qwen3-Coder-480B-A35B + OpenHands 0.54, function calling, avg 64 turns, max 100 | `resolved` (32,161 true) | 53K (n=100, 4.2 chars/token) | 100% / 96% / 26% |
| SWE-bench/SWE-smith-trajectories (`tool` split) | 24,100 (76,002 over 3 formats) | mit | `messages` as JSON string, function calling in `tool` split, 8 shards | claude-3-7-sonnet, claude-3-5-sonnet, gpt-4o; SWE-agent; synthetic bugs | `resolved` | 21K (n=300, shard 0, 3.07 chars/token) | 71% / 25% / 6% |
| nvidia/SWE-Hero-openhands-trajectories | 34,269 | cc-by-4.0 | like rebench (`trajectory` with tool_calls) | Qwen3-Coder-480B + OpenHands | none (SFT-curated) | not measured | not measured |
| nvidia/SWE-Zero-openhands-trajectories | 318,115 | cc-by-4.0 | like Hero | not verified | none | not measured | not measured |
| SWE-Gym/OpenHands-Sampled-Trajectories | 6,055 | none in card (SWE-Gym repo is MIT) | messages + tools, 3 shards | gpt-4o, maxiter 30 | `resolved` (7/150 true in shard 0) | 4K (n=150) | 31% / 8% / 0.7% |
| nebius/SWE-agent-trajectories | 80,036 | cc-by-4.0 | `trajectory` list of {role, text}; ReAct text, not function calling | SWE-agent with Llama-3 70B/8B/405B | `target` (13,389 true) | 10.8K (n=150) | 35% / ~1% / 0% |
| SWE-Gym/OpenHands-SFT-Trajectories | 491 | mit | messages | resolved only | all true | not measured | small |

Notes and corrections:
- PLAN.md names `nebius/SWE-agent-trajectories` filtered to >= 32K. That leaves roughly 1-4% of 80K (0.7% by tokenized sample n=150; 4% by character estimate at 3.5 chars/token over 6,670 rows, an upper bound), mostly unresolved runs by Llama models in a text-action format foreign to Qwen3. Demote it.
- SWE-rebench-openhands is the primary: almost every trajectory exceeds 32K, ~26% exceed 64K, real tool calls with long file views and test output, 61 median assistant turns, resolved and unresolved both present, and the generator is in the Qwen3 family (smallest off-policy gap for Qwen3-4B and Qwen3.8-27B). Eligible pool at >= 32K is about 64K trajectories. Caveat: the 100-row token sample came from the first rows of the file, so the 64K figure carries a roughly +-9 point interval.
- SWE-smith `tool` split is the secondary (different generator family, Claude, different scaffold, 25% >32K), to check the result is not an OpenHands/Qwen-Coder artifact.
- Not verified: Terminal-Bench traces (`harborframework/terminal-bench*` are task sets, not trajectories; `mlfoundations-dev/terminal-bench-traces-local` exists, ~3K downloads, not inspected), Agentless (pipeline, not multi-turn), mini-SWE-agent logs in the SWE-bench experiments repo (GitHub, sizes not checked).
- Off-policy: all trajectories come from other models, which is what teacher forcing needs gold text for. To keep conclusions on-policy for the model under test, metrics b and c are defined relative to the model's own full-cache behaviour, and metric a is reported only as a delta against the full cache on the same gold text, never as an absolute loss.

Format handling: render with the Qwen3 chat template (tools from the `tools` column, tool calls as `<tool_call>` JSON, observations as `<tool_response>`), non-thinking mode. Cap each trajectory at 32,768 tokens total (Qwen3-4B native window; no YaRN, which would confound), cut at a turn boundary.

## 2. Metrics

Common setup per (trajectory, compaction point P): prefill tokens [0, P), compact with method M to 10% of the prefix tokens (the system prompt/tool-schema head is kept verbatim, as R1b kept prompt tokens; state the exact kept count), then everything after P runs on the compacted cache. P in {12K, 20K} at a turn boundary. Arms: full, F, random keys (same value fit, as R0 vs-random), truncate (keep the most recent 10%). Unit of resampling for every CI is the trajectory (paired cluster bootstrap, 10,000 resamples, as R1b); McNemar exact as secondary.

Sample-size rule for paired binary outcomes: N = 3.84 * d * DE / delta^2, with d the discordance rate between two arms, DE the design effect from clustering, delta = 0.02. R1b gives d: F vs original on QuALITY 21 discordant of 357 = 5.9%; O1 vs original 43/357 = 12%. Planning values below use d between 8% and 10%; the pilot re-estimates d and DE and freezes N.

### a. Teacher-forced loss and top-1 agreement on late turns
- Build: feed gold tokens from P to 32K through the compacted cache (chunked), mask to assistant tokens only (tool-call arguments and prose), compute logits only at masked positions (a 32K x 152K fp32 logit tensor is 19 GB; never materialize it). Report per turn: mean NLL, delta NLL = NLL(M) - NLL(full), top-1 agreement with the full-cache argmax, and KL on a 1-in-4 token subsample. Bucket by distance from P and report the last 8K tokens ("late turns") separately.
- N: continuous, paired by turn. If per-turn sd of delta NLL is ~0.08 nats (assumption), resolving 0.02 nats at 95% needs ~60 independent turns; with DE ~3.7 (ICC 0.3, ~10 turns per trajectory) about 230 turns, i.e. ~25 trajectories. For top-1 agreement (d ~4% of tokens, 60 tokens/turn, ICC 0.1, DE ~7): ~2,700 tokens, ~45 turns. Cheapest of the three; resolved well inside the planned 60 trajectories x 2 points.
- Failure modes: loss is dominated by boilerplate tokens that never depend on old context, diluting the signal (mitigate: also report on tokens inside tool-call arguments and on tokens whose full-cache probability is above a threshold fixed before running); off-policy gold text gives a high absolute loss (only deltas are valid); teacher forcing hides error compounding (metric c's job); a compaction that breaks position handling shows as a uniform loss jump at P, so check the first 100 tokens after P separately.

### b. Exact-match recall probes
- Build (deterministic, no LLM): scan tool observations before P for facts that do not appear anywhere in [P, probe) and not in the first user message: (1) file paths from directory listings/view headers, (2) `def`/`class` names from file views, (3) last line of a traceback (`ErrorType: message`), (4) failing test ids from pytest output (`FAILED path::test_name`), (5) line-anchored constants (a number or string literal on a viewed line). Template examples: "In the earlier `grep` of the repository, which file defined `X`? Answer with the path only." / "What was the exact error message of the first failing test run? Answer with the message only." Insert as a final user turn at P on a forked cache (never persisted into later turns). Greedy, non-thinking, 48 new tokens, normalize whitespace, exact match.
- Calibration filter: keep a probe only if the full-cache arm answers it correctly (otherwise it measures the 4B's reading ability, not retention); generate 16 candidates per unit to keep ~10. Report retention = P(correct | full correct) per arm, bucketed by fact type and distance from P. Truncate must score near 0 on facts outside its retained tail (floor check); full defines the ceiling.
- N: d ~8%, DE ~1.5 (probes clustered within trajectory) gives ~1,150 retained probes for a +-2 point half-width. Plan 1,200 retained (120 units x 10).
- Failure modes: answer guessable from the question or repeated project structure (mitigate: drop probes whose answer also occurs in the question or in the system/tools head; add a shuffled-fact control where the asked fact comes from a different trajectory, expected accuracy ~0); exact match penalizes formatting drift (instruct "path only", normalize quotes/whitespace, nothing more lenient); facts the full 4B cannot retrieve at 20K+ shrink the retained set; regex extraction biases toward test and traceback text (report per type, never average types silently).

### c. Next-tool-call agreement
- Build: at decision points (the start of a gold assistant turn after P, 12 sampled per unit, stratified over the last 8K tokens, fixed seed), roll the cache forward with gold tokens, then run greedy generation up to 128 tokens on a snapshot and parse the first tool call. Agreement = tool name equal and key argument equal after normalization (`path` for view/edit; first token plus first path/identifier of `command` for bash; hash of `old_str` prefix for str_replace). Reference is the full-cache model's own parsed call at the same point, not the gold call. Also log agreement with gold for context.
- N: binary, paired against full. d ~10%, DE 1.5 gives 1,440 decision points for +-2 points (12 per unit x 120 units). For comparing F against random or truncate (the question that matters) the discordance is larger, so the same N resolves 2 points there with room.
- Failure modes: free-text before the call differs while the action is the same (match only on the parsed call); many equivalent actions (view vs bash cat) deflate agreement for every arm equally and hide differences (report tool-name-only agreement separately); greedy decode of a 4B at 20-32K can loop (count parse failures per arm as their own outcome, do not drop); full-vs-gold agreement below ~70% means the 4B is too weak here, which is itself a finding. Needs snapshot/restore, not crop (section 5).

## 3. Memory fit, Qwen3-4B

Qwen3-4B: 36 layers, 8 KV heads, head_dim 128 (from the model config as I recall it, not re-fetched), so KV = 2*36*8*128*2 B = 144 KiB/token: 4.5 GiB at 32K, 9 GiB at 64K. Weights bf16 ~8.0 GB.
- Decode and teacher-forced eval at 32K: 8.0 + 4.5 + chunked-prefill activations ~ 14-16 GB. Fits 24 GB.
- AM fit peak is the unknown: query capture costs 36 layers * 32 heads * 128 * 2 B = 295 KB per query token (2.4 GB per 8K queries, 5.9 GB per 20K) on top of weights and KV. With all 20K prefix queries held on GPU: 8.0 + 2.9 (20K KV) + 5.9 + score chunks ~ 20-24 GB, too tight for 24 GB cards; fine on 40/48/80 GB. I did not measure the repo's query handling or peak memory (R0/R1 logs record time, not peak memory; all prior 4B runs used the H100 / A100-80GB list in `am_repro.py`, line 33). 64K does not fit 24 GB for the fit and is out of the 4B plan (native window 32K).
- The pilot must log `torch.cuda.max_memory_allocated` for fit, teacher-forced pass and generation.

## 4. Cost plan (Qwen3-4B)

Modal list prices fetched from modal.com/pricing on 2026-10-14 (per second x 3600): T4 $0.59/h, **L4 $0.80/h, A10 $1.10/h** (Modal lists "A10", no "A10G"), **L40S $1.95/h, A100-40GB $2.10/h, A100-80GB $2.50/h, H100 $3.95/h**. CPU/memory are billed separately; I add 5%.

Workload: N = 60 trajectories x 2 compaction points = 120 units. Per unit and arm: prefill + teacher-forced pass, 16 probe candidates (~25 decoded tokens each), 12 decision points (~80 decoded tokens); fits for F and random (truncate has none).
- Decoded tokens per unit: 4 arms * (16*25 + 12*80) = 5,440.
- Compute-bound seconds per unit on H100 (assumptions, replaced by the pilot): F fit 45 s and random fit 40 s at 12-20K (anchors: R0 ~29 s for S1 on a 6.6K article, R1b 7 s for F on the 27B at 6.5K; I assumed super-linear growth with length), prefill + teacher-forced 5 s per arm, total 105 s.
- Decode rate assumed launch-bound HF eager: ~30 tok/s H100, 28 A100, 27 L40S, 25 A10, 20 L4 (L4 is bandwidth-limited: 8 GB weights at 300 GB/s).
- Compute slowdown vs H100 for fit/prefill: A100 2.6x, L40S 2.7x, A10 6.5x, L4 13x.

| GPU | s/unit | 120 units (h) | $/h | Est. $ (+5%) | Memory for fit | Verdict |
|---|---|---|---|---|---|---|
| L4 24GB | ~1,640 | 54.6 | 0.80 | 46 | tight/likely OOM | no |
| A10 24GB | ~905 | 30.2 | 1.10 | 35 | tight | no (memory risk, slow wall) |
| **L40S 48GB** | ~485 | 16.1 | 1.95 | **33** | ample | **recommended** |
| A100-40GB | ~467 | 15.6 | 2.10 | 34 | ok | close second |
| A100-80GB | ~467 | 15.6 | 2.50 | 41 | ample | over budget |
| H100 80GB | ~286 | 9.5 | 3.95 | 39 | ample | works, no margin; what R0/R1b used |

Reading: decode is 40-60% of unit time and is launch-bound, so a cheaper-per-hour GPU costs little extra time while H100's price premium only speeds the fits. L40S wins on dollars with 48 GB headroom for the query-capture peak. Sensitivity: if the F fit is 3x my assumption (135 s), L40S rises to ~$44 and H100 to ~$48; the pilot decides.

**Recommendation: L40S 48GB, N = 60 trajectories x 2 compaction points (120 units), ~$33, plus a ~$2 pilot, total ~$35, under $40.** Run with `max_containers` = 8 (about 2 h wall), cost log to the results volume as in `r1b.py` (name, wall_s, usd_est, running total), hard stop at $40.

Pilot gate (5 trajectories x 2 points, one L40S container, ~$2): measure fit seconds and peak memory vs prefix length, d and DE for b and c, per-turn sd for a; re-solve N with the section 2 formula; fall back to a smaller N or to H100 if any assumption pushes the full run above $40. Pre-register in `PREREG.md` before any result is read: dataset slice (first K trajectories with >= 32K tokens by instance_id sort, resolved and unresolved stratified 1:1), P values, seeds, kept-head size, margin (2 points), arms, stop rule.

Validity checks at no extra cost: (1) the uncompacted path through the harness equals plain HF forward (identical greedy tokens on 5 trajectories, as R1a); (2) arm ordering must reproduce the QuALITY ordering (full >= F > random >> truncate) on a, b and c, otherwise the harness is not sensitive and the result is void; (3) random keys must be separable from F by >= 3 points on probe retention, as on QuALITY (+5.4 points, R0).

## 5. Transfer to Qwen3.8-27B (hybrid DeltaNet)

Put the model-specific part behind one backend interface: `prefill(ids) -> cache`, `compact(cache, method, ratio) -> cache`, `extend(cache, ids, logit_positions) -> logits`, `generate(cache, max_new) -> ids`, `snapshot(cache)` / `restore(snapshot)`.
- 4B backend: the repo's unpatched Qwen3 modeling (transformers 4.57.1) with `CompactedPrefixCache`, as in `modal/am_qwen35/r1b4.py`.
- 27B backend: the adapter in `modal/am_qwen35/` (transformers 5.19.0, only the 16 full-attention layers compacted, Gated DeltaNet layers unchanged, `CompactedPrefixCache` carries conv/recurrent state through), reusing `r1b_core.py` helpers, the per-head budget file `R1b/r1c/optimized_agnostic.json`, and the cost log.
- Use snapshot/restore (deep copy of the cache) rather than `crop` from day one: DeltaNet recurrent state cannot be truncated, so probes and decision points on the 27B must restore a copy. Build it that way on the 4B so the code path is shared.
- Semantics differ and must be reported as such: DeltaNet layers still carry a lossy summary of the whole prefix, so on the 27B the truncate arm is a weaker control (full-attention layers lose the prefix, recurrent layers do not) and recall probes measure the combination. Label it "full-attention truncated, DeltaNet intact". Do not compare absolute retention across 4B and 27B; compare each model's F and random deltas to its own full cache (the matched-protocol approach of R1b).
- Context: the 27B can use 64K (26% of rebench trajectories exceed it): P in {24K, 48K}, cap 64K. bf16 weights (~54 GB) need H100/H200 80 GB or larger, as in R1b; decode is several times slower per token than the 4B and fits are heavier. Rough cost at the same 120 units: $150-250, so run it reduced: all three metrics on 30 trajectories x 2 points, about $40-60. Gate: run the 27B only if the 4B run separates F from random and truncate on all three metrics.
- Interpretation anchor: R1b put the 27B at -3 points with F on QuALITY at 0.10 and the matched 4B at ~0. If the 27B agentic loss on retention is much greater than 3 points, single-document QuALITY understated the cost for the engine workload.

Reuse for lucebox (R4): the extracted trajectory slices, the probe set (JSONL of prompt, answer, fact type, distance) and the decision-point set are model-independent and deterministic, so the same files can drive the C++ engine replay on lucebox6 for metrics a (via logits) and c.
