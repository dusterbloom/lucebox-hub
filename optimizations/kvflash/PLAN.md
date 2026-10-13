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
| **R2** our tensors | inputs → K/V/Q dumped from lucebox (Q4 weights, quantized KV, FWHT basis) for the same prompts | dump local/lucebox6, fit on Modal | no amplification: err(AM, engine) ≤ err(AM, bf16) + err(quantization alone), within bootstrap CI; logit KL vs the uncompacted engine no worse than R1's on bf16 |
| **R3** our implementation | fitter → C++ `am_fit` | local | on R2 fixtures: same selected keys ≥ 99%, outputs within 1e-3 relative |
| **R4** end to end | lucebox with E1–E3 | lucebox6 / 3090 | same QuALITY subset within CI of R1; 120-turn trace 120/120 with 0 degenerate turns; harness 36/36; NIAH ≥ today; decode median within noise |

Go/no-go for E: R0 reproduced and R1(b) passes. If R1(b) fails, stop and report.

## Track S: sessions survive turns (port of fork code)

| Patch | Donor | Status | Proof |
|---|---|---|---|
| S1 pager serialize/restore, pins, score ledger | 902b99de | committed b64d1630 | 9 unit cases, ASan/UBSan clean |
| S2 pager blob inside prefix snapshots | fd166b39 | committed b03f6ea7 | 2 unit cases; callers compile |
| S3 pooled snapshot at chunk boundary | fd166b39, 70f2a4b5 | committed (with S4), GPU gate running | GPU |
| S4 restore + prefill only new tokens, `KVFLASH_RESTORE_CONSUME` off by default | 56168347, 3156aec9 | committed, 9 unit cases; GPU gate running | GPU: turn-3 output identical with and without restore at 20K/35K/64K (donor saw divergence above 35K, never explained); suffix-only prefill visible in logs |

Default-on only after the S4 identity gate passes. Disk persistence of pooled
snapshots comes after S4.

## Track B0: baseline of today's system

120-turn real session trace, temperature 0, same GGUF and KV type on every arm,
`pt`/`prefix_len` read on both engines (bench-parity rule).

| Arm | Engine | Settings |
|---|---|---|
| a | llama-server | `--max-ctx 131072` |
| b | lucebox, KVFlash off | `--max-ctx 131072` |
| c | lucebox, KVFlash `8192` qk | `--max-ctx 196608` |
| d | arm c + S1–S4 | after S4 |

Per turn: prefill, decode, wall (median), `pt`, degenerate yes/no.
Needs the replay client and `degeneration_scan.py` copied into this repo
without `/tmp` locks or fork-only env vars.

## Track E: AM in lucebox (only after the R go)

| Step | What | Test |
|---|---|---|
| E1 | `am_fit` in C++ (HighestAttnKeys, NNLS, least squares, fp32); reference-query reservoir from prefill | R3 |
| E2 | flash-attention per-head bias: drop the `fattn.cu:4433` gate; head offset in vec/tile/wmma/mma using the group's first query head (β is per KV head, so GQA grouping stays) | random per-head masks match the CPU backend; decode with a 1-head mask unchanged |
| E3 | pager summary blocks; one mask per layer (F16 decode and F32 verify/prefill); host dequant; fit at `page_out` on a CPU worker; compact when the session budget is full | R4 |
| E4 | merge old summaries so the budget never overflows | 120-turn trace, rounds counted |

## Track C: concurrency (after S)

C0 one-page design of one pager per `seq_*` slot over the shared pool;
C1 refcount + COW + chained salted hash in `paged_kv_pool`; C2 KVFlash per slot
(lifts the `--max-concurrency` refusal). Gate: 2 and 4 concurrent replays,
each identical to its solo run, decode median per session flat.

## Budget (estimate, revised after the first runs)

R0 full matrix ≈ 20–30 H100-hours ($80–120). R1 four configurations ≈ 30–60
H200-hours ($140–270).
