# KVFlash + Attention Matching: work plan (2026-10-13)

Goal (Davide): "una versione migliore di kvflash con la compaction". Long agentic
sessions, one user and small teams, constant decode and prefill cost per token,
old context kept as Attention Matching (AM) summaries instead of being dropped.

Four tracks. S and C are engine work that pays off with or without AM. R decides
whether AM is worth building; E builds it only after R says yes.

```
S  sessions survive turns   ──────────────► (needed by E and by every long session)
R  reproduce AM, one axis per rung ──► go/no-go ──► E  AM inside lucebox
C  concurrency (paged pool)  ─────────────► (needed by E for many users)
B0 baseline of today's system ──► the number every later step is compared to
```

## Machines

| Where | What runs there |
|---|---|
| Local RTX 3090 | Unit tests, B0 on Qwen3.6-27B Q4_K_M (`/mnt/kimi-k3/home-models/qwen3.6-27b-q4km`), tensor dumps for R2. Builds go on `/mnt/kimi-k3` (local `/` is full) |
| lucebox6 R9700 (`duster@`) | HIP build, GPU tests of S, B0 on Qwen3.8-27B UD-Q4_K_M (`~/models`), later E kernels |
| lucebox3 R9700 (`dusterbloom@`) | Fallback when lucebox6 is busy |
| Modal (workspace davide-83641) | Everything in Python/torch: R0, R1, R2 fitting. Results in volume `kvflash-am-results` |

## Rules for scientific fidelity (R track)

1. **Pre-register.** Before each rung, write hypothesis, metric, sample size,
   seeds and pass threshold in `evidence/am/PREREG.md`. Anything changed after
   seeing data is logged there as a deviation.
2. **One axis per rung.** Each rung changes exactly one thing (model, runtime,
   quantization, implementation). A failure is then attributable.
3. **Provenance per run.** `manifest.json` next to every result: repo sha,
   pinned packages, model id + revision, dataset + split, seed, GPU, command,
   wall time, cost.
4. **Controls in the same run.** Uncompacted baseline and a trivial compactor
   (truncate or random-subset keys) at the same budget.
5. **Statistics.** Report n, mean, 95% CI; paired comparisons across seeds.
   "Reproduced" means the paper's number lies inside our CI. AIME has 30
   problems (about ±9 points per seed), so it needs 4 seeds.
6. **Select on one split, report on another.** The method that wins on the
   selection articles is reported on held-out articles. One primary
   comparison is pre-registered: best online candidate vs ceiling.
7. **Cost next to quality.** Every method records GPU-seconds per compaction.
   Hiding compaction in the background is not free under concurrency; Python
   H100 timings rank methods, they do not predict C++ on the R9700.
8. **Keep everything.** Raw outputs stay on the Modal volume; summaries and
   manifests are committed under `evidence/am/`. Negative results too.

## Track R: reproduce Attention Matching

| Rung | Changes vs previous | Where | Pass |
|---|---|---|---|
| **R0** paper as published | nothing: their repo, model (Qwen3-4B), scripts, configs | Modal H100 | Full method matrix (query sources context-prefill / repeat-prefill / self-study × key selection HighestAttnKeys / OMP / OMP-fast × on-policy × uniform vs per-head budgets) and their leave-one-out ablation: QuALITY at 5% and 10% cache, AIME with 1/3/8 compactions, all within our CI of the paper's numbers; AM beats the trivial compactor |
| **R0s** stale queries | fit with only the queries available when the fit starts vs all queries (models background, non-pausing compaction) | Modal H100 | quality gap reported with CI; decides whether non-pausing compaction is safe |
| **R1** our model, their code | model → Qwen3.8-27B (hybrid; only the 16 full-attention layers compacted, DeltaNet state untouched). Adapter ~800–1,100 lines, transformers 5.x image | Modal H200 | (a) adapter without compaction matches HF transformers: identical greedy tokens on 5 prompts; (b) four configurations: floor (context-prefill + HighestAttnKeys), ceiling (self-study + OMP + on-policy + per-head budgets), two online candidates; degradation vs its own baseline no worse than Qwen3-4B's at the same ratio; (c) head-sensitivity curves → per-head budget file |
| **R1(e)** segment-local fit | same adapter: AM fitted per S-token segment into a fixed block vs the paper's global per-head fit, same total budget | Modal H200 | segment-local within CI of global on QuALITY and AIME; if not, E3 needs variable-size blocks |
| **R1(d)** retrieval | same adapter: summary only vs summary + top-k spans swapped back to originals (scored by summary attention mass) vs uncompacted, same token budget on GPU | Modal H200 | retrieval closes ≥ half of the summary-only gap on QuALITY and needle tasks, within CI; the paper never mixes summaries and originals, so this is new evidence |
| **R2** our tensors | inputs → K/V/Q dumped from lucebox (Q4 weights, quantized KV, FWHT basis) for the same prompts | dump local/lucebox6, fit on Modal | no amplification: err(AM, engine) ≤ err(AM, bf16) + err(quantization alone), within bootstrap CI; logit KL vs the uncompacted engine no worse than R1's on bf16 |
| **R3** our implementation | fitter → C++ `am_fit` | local | on R2 fixtures: same selected keys ≥ 99%, outputs within 1e-3 relative |
| **R4** end to end | lucebox with E1–E3 | lucebox6 / 3090 | same QuALITY subset within CI of R1; 120-turn trace 120/120 with 0 degenerate turns; harness 36/36; NIAH ≥ today; decode median within noise |

R1 adapter, exactly (repo `adamzweiger/compaction` @ 2020f6d3):
`models/qwen3_5/modeling_qwen3_5.py` subclasses the transformers 5.x Qwen3.5
model and patches only the full-attention layer: the β-into-mask block of
`models/qwen3/modeling_qwen3.py:223-260` (β added after the softmax scale,
before the output gate, so the gate is untouched). Gated DeltaNet layers run
unchanged. `models/generate.py:get_sliding_layer_info` gains a third set,
`linear_layer_indices` (`layer_types == "linear_attention"`), which every
compaction loop skips (`global_highest_attention_keys.py:165/256/377/541`);
`models/cache.py:CompactedPrefixCache` carries the DeltaNet conv/recurrent
state through as-is. Query capture hooks only the 16 full-attention layers.

Evaluation for agentic coding (after R1, before E):
1. Teacher-forced replay of real long agent trajectories (our own 120-turn
   traces plus `nebius/SWE-agent-trajectories` filtered to ≥ 32K tokens):
   per-turn tool-call exact match, KL vs full cache, first divergent turn.
   Same data on Modal (R1) and lucebox (R4), deterministic, cheap.
2. Recall probes inside those sessions: questions about earlier tool outputs
   (file lines, error messages), plus RepoQA and RULER for calibration.
3. Outcome: SWE-bench Verified subset (≥ 50 tasks, mini-SWE-agent) and
   Terminal-Bench; resolve rate full vs KVFlash vs KVFlash+AM, paired by task.

Go/no-go for E: R0 reproduced and R1(b) passes. If R1(b) fails, stop and report.

## Track S: sessions survive turns (port of fork code)

| Patch | Donor | Status | Proof |
|---|---|---|---|
| S1 pager serialize/restore, pins, score ledger | 902b99de | committed b64d1630 | 9 unit cases, ASan/UBSan clean |
| S2 pager blob inside prefix snapshots | fd166b39 | committed b03f6ea7 | 2 unit cases; callers compile |
| S3 pooled snapshot at chunk boundary | fd166b39, 70f2a4b5 | committed 6b9f8b59, GPU-verified | lucebox6, Qwen3.8-27B, `--kvflash 8192` |
| S4 restore + prefill only new tokens, `KVFLASH_RESTORE_CONSUME` off by default | 56168347, 3156aec9 | committed 6b9f8b59, GPU-verified | turn-3 output identical with and without restore at 20K/35K/64K; turn 2–3 prefill 19/33/62 s → 0.5/0.5–1.8/0.2 s; decode unchanged (27–28 tok/s); turn 1 +1.5 s |

**Gap found by the GPU run.** The server asks for a snapshot at the prompt end
(18813) but S3 can only save at the chunk boundary below it (18432). Later turns
log "boundary snapshot skipped: pooled prefill relocates chunks", so only the
turn-1 snapshot is ever reused and the re-prefilled suffix grows every turn.
Each snapshot also copies all KV to host RAM. S1–S4 are a correct stepping
stone, not the end state.

### Decision: one live session store, aligned compute grid (S5–S9)

Design reviewed against Codex's alternative; Codex's version adopted. Goal:
an unaligned cut is impossible by construction, and full re-prefill from 0
happens only for a new conversation.

Principles:
- **KV is never snapshotted per turn.** The session keeps its live pager and
  pool; going back to position P is `truncate(P)`, which works for any P
  because full-attention KV at a position does not depend on later tokens.
- **Only DeltaNet state needs checkpoints** (149.6 MiB each: 144 SSM + 5.6 conv).
- **Every KVFlash prefill runs on an absolute grid of U = 512 tokens**
  (multiple of the 64-token chunk). Batches never start off-grid, so the
  unaligned-start batch accounting (qwen35_backend.cpp:2154,
  kvflash_pager.h:627) and the 2-token-batch NaN on R9700
  (prefill_helpers.h:15/:51) cannot occur.
- **Restore = greatest checkpoint C ≤ common prefix L**, `truncate(C)`,
  replay [C, prompt end). Missing checkpoint → fall back to an earlier one,
  never to 0 while a session exists.
- **Publish the cut actually made**, not the one requested
  (http_server.cpp:4185, prefix_cache.cpp `commit_at` :555).

| Patch | What | Test / gate |
|---|---|---|
| S5 `pager.truncate(P)` (5cd2a923, 7 unit cases; GPU: used by S8a, passes) | drop logical positions ≥ P, free their blocks, keep pins/ledger consistent | unit: truncate at chunk edge, mid-chunk, inside tail, after eviction; serialize ↔ truncate round trip |
| S6 aligned prefill grid (03ea42af, 9 unit cases; R9700: remainders 1–64 after 9216 tokens, no NaN) | KVFlash prefill batches start on multiples of U; the remainder below U runs as the last batch | unit: batch plan for prompt lengths 1, 511, 512, 513, 18813; GPU: no NaN on R9700 for every remainder size 1–64 |
| S7 DeltaNet checkpoint ladder (7154dfb1, 5 unit cases; R9700: rungs as expected, output identical, prefill +2–2.5%, 349.6 MiB per checkpoint incl. 200 MiB drafter ring) | checkpoints at the system/tools head, every 8192 tokens, and at the grid point below each prompt end; LRU-bounded (~2.9 GiB at 131K) | unit: selection of C for synthetic L; memory bound respected |
| S8 per-session store + restore (S8a single live session: 1e853bfd, 8 unit cases; R9700 10-turn: identical output at all turns, turn 2–10 prefill 11–12.7 s → 0.3–0.9 s, decode unchanged 27.8 tok/s, mid-history edit restores below the edit; S8b many sessions: after C) | live pager + checkpoints keyed by session id, under one global GPU-block and host-RAM budget (idle sessions page out whole, LRU across sessions); restore as above; actual cut published | GPU: 10-turn session, every turn re-prefills < U + new tokens; output identical to cold prefill at turns 3, 5, 10 |
| S9 generation-end checkpoint | checkpoint after decode so the next turn skips re-reading the answer | gated: enabled only if DeltaNet state after generation equals cold prefill of the same tokens (greedy tokens identical over 5 sessions) |

The pager blob (S1/S2) stays only for export and disk persistence. Refcounted
copy-on-write blocks are the follow-up if sibling conversations (forks of one
session) become a requirement; that belongs to track C.

Default-on for S4 is dropped: S8 replaces it.

## Track B0: baseline of today's system

120-turn real session trace, temperature 0, same GGUF and KV type on every arm,
`pt`/`prefix_len` read on both engines (bench-parity rule).

| Arm | Engine | Settings |
|---|---|---|
| a | llama-server | `--max-ctx 131072` |
| b | lucebox, KVFlash off | `--max-ctx 131072` |
| c | lucebox, KVFlash `8192` qk | `--max-ctx 196608` |
| d | arm c + S5–S8 | after S8 |

Per turn: prefill, decode, wall (median), `pt`, degenerate yes/no.
Needs the replay client and `degeneration_scan.py` copied into this repo
without `/tmp` locks or fork-only env vars.

## Track E: AM in lucebox (only after the R go)

| Step | What | Test |
|---|---|---|
| E1 | `am_fit` in C++ (HighestAttnKeys, NNLS, least squares, fp32); reference-query reservoir from prefill | R3 |
| E2 | flash-attention per-head bias: drop the `fattn.cu:4433` gate; head offset in vec/tile/wmma/mma using the group's first query head (β is per KV head, so GQA grouping stays) | random per-head masks match the CPU backend; decode with a 1-head mask unchanged |
| E3 | segment map + compacted blocks (Davide): the full cache is cut into segments of S = 64 / r tokens (S = 1280 at r = 5%), each fitted into exactly one 64-slot pool block (Ck = selected original keys, RoPE baked in; fitted Cv; β per KV head). The map segment [lo,hi) ↔ block is the pager's chunk table, so page_out, scoring, pins, serialize and truncate work unchanged on compacted blocks; offload/retrieval moves compacted blocks (20× smaller than originals). Recent tail and pinned chunks stay uncompacted. One mask per layer (F16 decode and F32 verify/prefill); fit on a CPU worker | R4 |
| E3r | optional exact tier, only if R1(d) passes: originals of each segment kept on host/disk; a block's attention mass (Σ exp(q·Ck + β), fitted to the segment's true mass) triggers a swap back to originals. Per segment exactly one of {compacted, originals} is visible in the mask | R1(d); then 120-turn trace with the tier on vs off |
| E4 | budget overflow: refit from host originals (no error compounding, unlike the paper's summary-of-summary); merge summaries only when host RAM is also full (then originals go to disk via the S1 blob) | 120-turn trace, rounds counted |

## Track C: concurrency (after S)

C0 one-page design of one pager per `seq_*` slot over the shared pool;
C1 refcount + COW over sealed 64-token chunks + chained salted hash in `paged_kv_pool` (shared system/tools prefix stored once; forks and AM compaction copy only the chunks they change); C2 KVFlash per slot
(lifts the `--max-concurrency` refusal). Gate: 2 and 4 concurrent replays,
each identical to its solo run, decode median per session flat.

## Budget (estimate, revised after the first runs)

R0 full matrix ≈ 20–30 H100-hours ($80–120). R1 four configurations ≈ 30–60
H200-hours ($140–270).
