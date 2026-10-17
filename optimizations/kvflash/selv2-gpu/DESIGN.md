# Selector v2 near recall-OFF latency

Recommend **the current v2 algorithm on GPU, with a GPU query ring and packed page movement**. Keep mode-4 replay. Optimize its checkpoint after timing it. This preserves the evidence behind v2's quality; per-layer or previous-turn selection does not.

**Estimated target:** 1.10–1.25 s median wall on this harness, versus OFF 0.918 s and v2 2.807 s. Acceptance target: **OFF + 250 ms**, with quality equivalent to v2 within ±1 percentage point. This is a design target, not an achieved result. GPU scoring alone projects to about 1.40 s.

## Evidence and conventions

M = measured from supplied logs/rows. D = arithmetic from code or M. E = estimate with stated assumptions. Targets and proposed experiment settings are not measurements. MB is decimal; MiB is binary.

- Source read with `git show kvflash/2.0c-selector-v2:<path>` in the current worktree. Ref resolved to **c09a5ec0**. The inspected engine files are unchanged from **4121bd29**; implementation is **22b59a62**, query dump **7c9180a9**.
- Main inputs: `/mnt/kimi-k3/split-work/selv2/e2e/v2/QK16KND.{jsonl,server.log}`, and `/mnt/kimi-k3/split-work/rlru4k/v2/{on6,off}/QK16KND.{jsonl,server.log}`. Additional isolation evidence: `rlru4k/v2/smoke/{off,noop,on6}/`.
- [mine.py](mine.py) joins each JSONL row to its ordered `chat START` block, asserts request counts and matched reselect populations, and writes [metrics.json](metrics.json), including input hashes. Reproduce with `python3 -B optimizations/kvflash/selv2-gpu/mine.py`. It only reads inputs and writes that sibling JSON.
- M: **951 retained probes**, **1,106 total requests**, **827 reselects**; **710 retained probes reselect**, **241 fit**. Do not mix these populations. The report's scorer median covers all reselects, not just retained probes.
- Quality remains **provisional**, from `selv2/e2e/report_e2e.provisional.json`: v2 **888/951**, ON6 **847/951**; beyond the pool **176/207** versus **138/207**; within **712/744** versus **709/744**. Offline N=4 or N=16 does not replace N=64 (`selv2/README.md`, `offline_mq.out`).
- Clock caveat from `rlru4k/v2/README.md`: monotonic time reportedly runs about **7% slow**, while WSL steps `time.time()` about **2.3 s every ~32 s**. Preserve the original client-wall metric, but collect an unstepped client clock in future. Do not treat wall residual outliers as network latency.

## Where time goes

All entries below are M, medians over the same retained probes. **Column medians do not add.**

| Metric | OFF | ON6 | v2 |
|---|---:|---:|---:|
| Prefill, ms | 398.9 | 819.3 | 2,190.3 |
| Decode, ms | 271.8 | 293.8 | 280.3 |
| Decode, tokens/s | 44.1 | 44.2 | 43.8 |
| Client wall, ms | 918 | 1,370 | 2,807 |
| Per-row wall − prefill − decode, ms | 144.0 | 151.6 | 153.9 |
| Prefilled tokens / cached prefix | 349 / 24,064 | same | same |

D: differences between the displayed medians are **+420.4 ms prefill / +452 ms wall** for OFF→ON6, and **+1,371.0 / +1,437 ms** for ON6→v2. These are not phase durations. Paired per-request differences give a clearer comparison:

| Paired statistic, ms | ON6 − OFF | v2 − ON6 |
|---|---:|---:|
| Median prefill difference, all retained probes (M/D) | 478.5 | 1,216.5 |
| Median prefill difference, reselected probes only (M/D) | 505.6 | 1,803.3 |
| Same, after subtracting each arm's logged scorer (M/D) | 470.3 | 62.7 |

### Phase inventory

The logs contain **no `[prefill-timing]` rows**. First-pass compute, gather, rewind, DMA and replay are not separately timed. The following is the finest defensible split; inventing a precise additive median breakdown would be misleading.

| Phase | What the logs/code establish | Time attributable today |
|---|---|---|
| Restore before prefill | Live/prefix restore precedes backend prefill timer | In wall residual; not isolated |
| Reset to recent-only pool | Required so first-pass queries see OFF's pool. On the 710 reselected probes, median **57.4 MiB H2D** in both ON6 and v2 (M) | Untimed; includes D2H, zeroing and host submission |
| First-pass prefill | Same suffix tokens; mode 4 adds a cut at R and a scratch recurrent snapshot | OFF's **398.9 ms** is a baseline proxy, not an isolated ON first pass |
| Query gather | v2 captures the tail across qualifying batches; synchronous full-buffer D2H, CPU transpose, append/erase | Untimed. Capacity **24 MiB** per gather (D); not in logged scorer |
| Exact scoring | Resident K copied D2H; host K read directly; then CPU math | All 827 reselects: ON6 **34.7 ms** = copy **6.2**, math **28.4** medians; v2 **1,720.2 ms**, copy **5.3**, math **1,714.5** (M; component medians need not sum) |
| Rewind | Restore recurrent/conv state, truncate rows at R, invalidate mask | Untimed; scratch allocation logged about **149.6 MiB** (M) |
| Recall page-in | Selected pool applied after rewind | **60.8 / 61.9 MiB H2D** ON6/v2 median per reselect (M); duration absent |
| Re-prefill user turn | All layers, including DeltaNet, run again under selected pool | Median **81 tokens**, p90 **98**, max **150** on reselected retained probes (M); duration absent |
| Decode | Same resident budget; output lengths differ | Measured above; no material throughput change |
| Tokenizer / template / HTTP / admission / response work | Outside backend prefill/decode timers, mixed with restore and clock steps | Only aggregate residual above; tokenizer and HTTP cannot be separated from these logs |

The scorer's `copy` is **D2H of the resident K pool**, not the recall H2D. Code bulk-copies all resident-pool K tensors: **151 MB** in the log, derived **144 MiB**. `scorer_ms` starts after the last-query read, includes K copy and math, and ends before rewind/page-in. Summed CPU dequant/dot/exp times are thread totals, not elapsed time; cross-chunk softmax/top-plane reduction is also inside math.

### Existing isolation evidence: replay is only part of the tax

Same-binary smoke arms contain **5 reselected probe pairs**, replay lengths **80–99 tokens** (M). The noop arm still scores and rewinds but skips the recall pool change. Subtract copy+math as its scorer proxy; ON6 has an explicit scorer timer. Printed rounding limits precision.

| Paired prefill difference after scorer subtraction | Median, ms | Range, ms |
|---|---:|---:|
| noop − OFF: extra cut + snapshot/restore + truncate + replay | 234.9 | 224.7–247.0 |
| ON6 − noop: recalled-pool transitions and any resulting compute changes | 221.2 | 192.7–269.5 |
| ON6 − OFF: entire non-scorer package | 467.7 | 427.6–513.1 |

These are M/D on a small smoke sample, not pure kernel times or full-harness medians. Noop keeps a recent-only pool; later ON6 requests also reset the preceding recalled pool. Thus the second row includes both pool transitions and altered replay behavior.

**Verdict on the guess:** confirmed that GPU scoring alone leaves latency near ON6. Refuted as a clean claim that the extra cost is mostly second-prefill compute: the smoke split is roughly half replay/cuts/state work and half pool-transition work. The full harness independently shows **470.3 ms** paired non-scorer prefill overhead on reselected ON6 probes.

Pager code explains why bytes alone understate cost. A chunk is **1.125 MiB** (D). Each direction issues **16 × 2 × 4 = 128** separate **9,216-byte** DMA operations, plus the same number of zero operations per outgoing chunk. Median recall **55 page-ins** on v2 means **7,040 H2D submissions** for that swap alone (D). At an **assumed** effective bulk link rate of **12–24 GB/s**, the reselected-probe median recent+recall volume, about **119 MiB**, would take only **5–10 ms** of H2D wire time (E). That excludes D2H, launch overhead, zeroing, synchronization and allocation; the link rate is not measured here.

### GPU-only counterfactual

For each retained row, replace its logged scorer by G only when it reselects, then take the median. Keep all other observed work and its output length. This is an E projection, not a rerun:

| Replacement G per reselect | v2 prefill median | v2 wall median |
|---:|---:|---:|
| 0 ms: ideal scorer, existing gather/pager/replay | 857.7 ms | 1,391.0 ms |
| 5 ms | 862.7 ms | 1,394.1 ms |
| 20 ms | 877.7 ms | 1,404.5 ms |
| 50 ms | 907.7 ms | 1,433.9 ms |

Even the zero-cost scorer leaves **473 ms** above OFF's median wall (D).

## Algorithm and memory contract

Source anchors at the ref above: `server/src/qwen35/qwen35_backend.cpp` (B), `qwen35_target_graph.cpp` (G), `server/src/common/kvflash_exact_fast.h` (F), `kvflash_pager.h` (P), `kvflash_prefill_reselect.h` (R).

- B:2357–2416 defines the two passes, recent-pool reset, R and query retention. B:2687–2711 gathers across batches before rewind. G:2477–2502 captures queries. B:2902–2919 downloads/transposes them.
- B:3080–3186 defines CPU copy/math and timer scope. B:3200–3253 picks/applies units. B:3269–3297 rewinds. P:891–969 contains per-head DMA and zero loops. R:116–134 defines eligible units and newer-first ties.
- Score **all available sealed chunks before R**, including resident chunks, not only paged-out or recall-eligible chunks: `C=floor(R/64)`. Keep the existing post-RoPE and quantization-rotation coordinate system and GQA head mapping. Physical relocation does not change logical positions; do not add KVMem-style re-RoPE to this pager.

For query i, plane p, chunk c:

`z[i,p,c] = logsumexp_t(q[i,p]·dequant(K[c,p,t]) / sqrt(256))`

`m[i,p,c] = softmax_over_all_C_chunks(z[i,p,:])[c]`

`s[c] = max_over_queries(mean_of_largest_8_planes(m[i,:,c]))`

Unit b scores `s[2b] + s[2b+1]`. Select **30 units**, preserving the **60-chunk** recall budget, exclusions before chunk **10**, complete-unit boundary, exact recent/reserve formula and newer-first ties. **Sum of chunk maxima is not max of unit sums.** Preserve actual captured query count on short prefills: logs report 64 on **822/827** reselects and fewer on **5**. Do not replace them with cached tokens' queries.

### GPU allocation, at the logged maximum context 52,224

D: q4_0 stores **18 bytes / 32 values**. K per token = `16 × 4 × (256/32) × 18 = 9,216 bytes`; K+V doubles it. Figures exclude allocator padding.

| Allocation | MB | MiB | Derivation |
|---|---:|---:|---|
| All-history K-only q4 shadow | 481.30 | 459 | 52,224 × 9,216 |
| Only nonresident K, with full 16K pool | 330.30 | 315 | (52,224 − 16,384) × 9,216; resident K read via map |
| All-history dequantized FP16 K | 1,711.28 | 1,632 | 52,224 × 16 × 4 × 256 × 2 |
| FP32 query ring | 25.17 | 24 | 64 × 384 × 256 × 4 |
| FP32 chunk logsumexp workspace | 80.22 | 76.5 | 816 chunks × 64 × 384 × 4 |
| FP16 mean-K index per 128-token unit | 13.37 | 12.75 | 408 × 16 × 4 × 256 × 2 |
| FP16 min/max index per 64-token chunk | 53.48 | 51 | 816 × 16 × 4 × 256 × 2 × 2 |

Use an all-history q4 shadow first: simple indexing and deterministic ownership are worth the **144 MiB** duplication of resident K (D). Do not keep all token logits: FP32 would cost **5,133.83 MB** (D). Tile q4 dequantization and QK, immediately reduce each chunk to logsumexp. Cross-chunk normalization requires a subsequent reduction; top-8 planes and query max follow it. This cannot be one independent per-chunk fused kernel with local normalization.

QK work = `2 × tokens × 64 × 384 × 256`: **0.420 TFLOP** at the reselected-probe median **521 chunks**, **0.657 TFLOP** at max context (D). Assuming **20–80 effective TFLOP/s**, QK takes **5–21 ms** at that median, **8–33 ms** at max (E). Allow additional reductions, dequantization and launches: **G=10–50 ms** is a planning range, not a GPU measurement. First implement FP32 query/accumulation correctness; FP16/TF32 inputs need separate pick checks. Do not budget tensor-core throughput for an unfused scalar kernel.

## Options, ranked

All wall estimates below are retained-probe median projections on this workload, assuming unchanged decode behavior. Approximate selectors may change output length and invalidate that assumption. Effort is an engineering estimate of implementation size, not elapsed delivery time: S = local changes; M = backend plus kernels; L = execution/pager changes; XL = a new attention schedule.

| Expected speed rank | Design | Estimated wall / derivation | Extra GPU memory at max context | Picks and quality risk | Effort; HIP/AMD |
|---:|---|---|---|---|---|
| 1, speculative | Previous-turn tail queries; select/prefetch before next request | **0.94–1.00 s** if background work finishes: OFF + **20–80 ms** handoff/materialization budget (E) | K + Q/LSE workspace; optional prefetched KV **67.5 MiB** = 60 chunks | Different queries and scoring pool; **very high** quality risk on a new question. No evidence of v2 parity | M–L; portable events/kernels, contention differs |
| 2, speculative | Single prefill with per-layer recall before that layer's attention | **0.96–1.07 s**: OFF + **40–150 ms** layer score/page barriers (E) | K + workspace; layer maps and staging | Different reduction and queries; **high** quality/ordering risk | XL; CUDA and HIP attention-graph changes |
| 3, speculative | Keep v2 picks, replay only final token after page-in | **0.99–1.08 s**: OFF + **10–50 ms** score + **10–40 ms** packed movement + about **23 ms** token step, `1000/44.1`, + **20–50 ms** state/control allowance (E) | Same K/Q/workspace as GPU option | Picks close; prompt KV and DeltaNet state lack recalled context. **High** quality risk despite same picks | M; portable; mode-3 mechanism already exists |
| 4, optional | Exact GPU pipeline with full K+V GPU archive and D2D pool gather | **1.08–1.22 s**: OFF + **140–220 ms** replay/state + **10–50 ms** score + **10–30 ms** gather/control (E) | Archive **918 MiB**, Q **24**, LSE **76.5**, staging; optional state **149.6** (D/M) | Same algorithm; rounding and memory pressure. **Low** quality risk, medium resource risk | L; shared indexing plus CUDA/HIP gather |
| **5, recommended** | **GPU K-only scorer + query ring + packed pager; keep replay** | **1.10–1.25 s**, budget below (E) | K **459** + Q **24** + LSE **76.5** + proposed staging **16** = **575.5 MiB**; optional GPU scratch state about **149.6 MiB** | Same algorithm; close arithmetic, exact IDs sought. **Low** quality risk | M–L; reusable runtime shim, portable kernels, tune per GPU |
| 6 | GPU scorer alone, existing gather/pager/replay | **1.39–1.44 s** from row-wise G=5–50 ms projection | K **459** + LSE **76.5** + device queries **24 MiB** | Same algorithm; close arithmetic. Low quality risk | M; CUDA/HIP kernel implementations |
| 7 | Mean-K or Quest shortlist, exact CPU or GPU refinement | CPU **1.74–2.09 s**: scorer-free **1.391 s** + **346–692 ms** CPU work for 64–128 units. GPU **1.40–1.45 s**: replace scorer with **15–60 ms**, including staging (E; same pager) | Index **12.75–51 MiB**; GPU refinement adds K staging **72–144 MiB**, Q/workspace | Uncertified shortlist and changed normalization; medium/high risk | M; summaries portable, CPU refinement unchanged or reuse GPU kernels |
| 8 | Cache/overlap only, keep CPU v2 | **Near current 2.807 s** for novel requests; `2.807 s − hidden/reused work` (E; savings unmeasured) | Policy-dependent; caching all max-context FP32 LSE adds **76.5 MiB per full query-window entry** (D) | Exact only with exact Q and key versions; low risk when validated | S–M; CPU cache portable, GPU overlap competes with model |

Speed ranks overlap within estimate uncertainty. Quality-risk ranking: unchanged-selector pipelines first; certified shortlist next; approximate shortlist next; final-token replay/per-layer/previous-turn policies last. Effort ranking: cache instrumentation first, GPU scorer next, packed exact pipeline next, per-layer schedule last. **Do not trade away the demonstrated quality gain to win a speculative latency rank.**

Implementation-size estimates (E, based on the modules above; exclude unrelated refactors): GPU scorer/ring about **600–1,000 changed/new lines** across backend, graph capture and shared CUDA/HIP kernels; recommended pipeline about **1,000–1,800** after pager batching and checkpoint changes. Full-KV archive adds **200–400** for archive ownership/gather. Per-layer execution is roughly **2,000–4,000** across graph scheduling and layer maps. Previous-tail hints add **100–300** above the GPU base; final-token replay adds **50–150** by reusing rewind machinery. Summary shortlist is **400–800** using the existing CPU refinement; certification adds a separate bound/refinement engine. Exact-repeat cache or measured overlap is **100–300** each. These are planning estimates, not counted patches.

### Recommendation budget and implementation boundary

E critical-path budget for a reselect: **OFF + replay/cuts/state 140–220 ms + GPU score 10–50 ms + packed transfers/control 10–40 ms + incremental query capture 0–10 ms**. This gives **OFF + 160–320 ms**, or about **1.08–1.24 s** using the supplied OFF median; round outward to **1.10–1.25 s** for planning. Mixed fitting/reselected medians require a rerun, so this is not an additive median forecast.

The replay/state budget assumes reducing the smoke package's **234.9 ms** by roughly **15–95 ms**, through reusable scratch storage, avoiding host recurrent round trips where memory allows, and removing diagnostic full-vocabulary logits copies/comparison. These savings are assumptions to test. The page budget assumes replacing the smoke pool-transition package's **221.2 ms** with bulk transfers and scatter; the gather budget assumes eliminating synchronous query downloads and CPU vector churn. Until these assumptions pass, use the measured **1.39 s scorer-free floor**, not the target.

Target flow:

1. Maintain a logical-chunk q4 K shadow, copied from **the final quantized cache bytes**, not independently quantized pre-cache K. Publish it with a version/event when sealed. Truncate, replay overwrite, prefix restore, session switch and partial chunks invalidate the matching versions. Resident chunks count in the score domain too. All **951 retained probes** log a live restore (M), so retain unchanged-prefix shadow versions across those restores. A genuine shadow miss can require up to **459 MiB** upload, **20–40 ms** at the assumed bulk rate (E); record this separately rather than assuming every restore is free.
2. Capture the same first-pass query tail directly into a GPU ring across batch boundaries. Keep the old batch cuts, query coordinates and actual short-prefill count. Avoid second-pass capture and the unused host tail transpose.
3. Preserve the recent-only first-pass attention set. Do not skip this reset based on physical cache hits: that changed first-pass query quality in the earlier experiments. Physical reuse must remain invisible to the attention mask.
4. Score q4 tiles using the existing GQA mapping, FP32 accumulation and stable logsumexp; normalize over **all C**, reduce top planes, then queries. Copy only chunk scores to the CPU and reuse the existing picker/tie logic. That output is only **816 × 4 = 3,264 bytes** at max context (D).
5. Pack outgoing/incoming pages into persistent pinned slabs. Replace thousands of small H2D calls with bulk DMA plus a scatter kernel; likewise gather outgoing bytes before bulk D2H. Preserve raw bytes and slot assignments. Batch zeroing; retain mask and zero guarantees. Existing `gpu_runtime_compat.h` and page-stream ordering work on CUDA/HIP. Do not add a second async framework.
6. Restore the scratch recurrent state, truncate and replay exactly as today. Reuse scratch allocation. If its measured host-copy cost matters, put only this temporary checkpoint on GPU; leave the larger state ladder alone. Its logged size implies roughly **149.6 MiB** extra (M, subject to scratch-layout confirmation).
7. Use per-layer/page-ready events to overlap known page transfers with replay preparation and later layers. Global picks are not available early enough to hide current-request recall behind an arbitrary amount of first-pass compute. Keep this optimization only if measured wall improves.

The full-KV archive is an optional alternative when memory measurements support another **459 MiB** beyond the K-only archive (D). It removes host page movement at this context limit, but needs a bounded-capacity fallback and still needs replay. Logged GPU weights are **14.33 GiB**; this is not a free-memory measurement. Do not enlarge memory based on weights alone. Integrated gfx1151 and discrete R9700 need their own memory and transfer measurements; importing 3090 latency estimates would be unjustified.

### Why the single-prefill ideas change quality

A global v2 pick depends on queries from **all 16 attention layers / 384 planes**. A layer cannot use that pick before producing its own Q; later-layer Q also depends on earlier attention. Selecting per layer from its own planes, or from a partial global score, changes both the selector and later queries. A multi-batch turn adds another dependence: later query tokens need earlier batches processed. Layer-specific picks also need layer-specific slot maps/masks, rather than today's shared pager.

Previous-turn queries can be computed during decode/idle time and drive prefetch, but do not contain the new user question. They are useful **hints only**, followed by current-query selection. If that selection differs, an exact implementation still pays correction/replay. Likewise, paging after first prefill and skipping all replay keeps stale initial logits; replaying only the final token refreshes logits but leaves earlier prompt KV and **48 DeltaNet layers'** states stale. Test these as separate quality hypotheses.

### Two-stage scoring: distinguish speed from exactness

[KVMem](https://arxiv.org/html/2609.04852) uses position-independent mean-K for block retrieval and replays the current query after retrieval. It motivates a small index; it does not establish parity with v2's post-RoPE token logsumexp, top-plane and query-max selector. Averaging can erase the single sharp token that v2 needs. [Quest](https://arxiv.org/abs/2406.10774) uses per-page coordinate min/max to estimate query relevance. Those bounds are attractive for a conservative shortlist, but their tightness here is unmeasured.

Proposed offline shortlist sizes: **64, 96, 128 units**, settings to sweep. The middle setting covers **12,288 tokens** (D). At the supplied **0.66 ms/query/1K tokens**, CPU exact work for that middle shortlist is about **519 ms** (E/D: `0.66×64×12.288`), before overhead: too slow to recommend. The smallest/largest settings give the table's **346–692 ms**. Even refining only the winning **3,840 tokens** costs about **162 ms** at that coefficient (E/D); a tiny shortlist has little room for misses. GPU refinement can stream shortlisted K instead of keeping a full shadow: **8,192–16,384 tokens × 9,216 bytes = 72–144 MiB** (D), about **3–13 ms** at the assumed link rate (E). Its table budget adds **12–47 ms** for summary/refinement, host packing, reductions and control (E). The full GPU scan is already small enough to try first.

**Exact-on-shortlist is not exact v2.** Even if the shortlist contains all winning units, renormalizing its chunks changes every plane's denominator and can change top-8 planes and picks. Keep full-domain normalizers or certify bounds. For a certified variant, coordinate min/max bounds give per-token logit limits L/U, hence chunk LSE limits `log(64)+L/U`. Bound normalized mass using a chunk's lower numerator with other chunks' upper numerators, and vice versa. Propagate these monotone bounds through top-8, query-max and unit sums. Refine until the last winner's lower bound exceeds every loser's upper bound, handling ties explicitly. Bounds must enclose the actual dequantized keys with outward rounding, including arithmetic error; FP16-rounded extrema alone are not certified. Worst case scans everything; loose high-dimensional bounds may add overhead. This is L/XL effort, not the first implementation.

### Async work and caches

- Exact safe overlap: stage out sealed chunks during later prefill/decode, update shadow/index only for new or overwritten chunks, keep pinned slabs warm, and compute a layer's score when its full query-tail slice is ready. Model compute and scoring share GPU resources; layer overlap may increase wall instead of reducing it.
- Speculative overlap: previous-tail hints prefetch likely pages during decode/idle time. Reserve separate physical staging so hints cannot alter the active mask, overwrite live pages, or modify DeltaNet state. Current-query v2 remains authoritative. Unknown next-turn text cannot be scored early.
- Cache keys/data, not merely final chunk scores. New Q requires new QK against all old K. Cache LSE only for **identical actual Q**, positions/coordinate system, model and key generations. A repeated token string is insufficient: attention pool and batch cuts can change its hidden-state query.
- Even identical old Q needs new cross-chunk normalization after adding history. Cache per-plane unnormalized LSE, update new chunks and recompute normalizers/top-8/max. Replay and branching can overwrite keys; logical chunk ID alone is not a valid cache key. No assumed hit rate is used in the recommendation.
- A no-op reselection may skip page application, but not replay solely because the chosen IDs match the previous turn: the current first pass used the recent-only pool. Only an identical first-pass and selected execution view, with matching state and batch semantics, can certify that skip.

## Build plan and pass bars — future work

No builds or GPU experiments are part of this design task. Future work must wait for benchmark ownership of the GPU lock. Use the same source snapshot, model, harness and prompt/cache settings for all arms.

| Checkpoint | Test first | Required result before proceeding |
|---|---|---|
| Phase accounting | Add disjoint timers for restore, recent reset, snapshot save, first-pass build/compute, Q capture/read, score/copy/reductions, restore/truncate, page pack/DMA/scatter/wait, replay, decode, template/tokenizer/HTTP | CPU elapsed intervals reconcile to request boundaries; GPU events measure device intervals without double-counting overlap. Existing v2/OFF behavior unchanged |
| Exact input contract | Fixtures for q4 bytes, 64-token-tail crossing batches, short prefill, GQA, all-sealed normalization, absent chunks, R/reserve, excluded units and newer-first ties | CPU reference score/pick reproduced; no GPU work required for fixture semantics |
| GPU shadow and scorer, shadow-only | Match K bytes and Q tail against CPU; compare LSE, mass, top planes, unit scores and IDs on existing dump inputs where present, plus captured smoke inputs | Seek **100% identical selected IDs** on full harness; log every mismatch and cutoff gap. Never call FP32 reorderings bit-exact. Any changed picks require the quality gate |
| Precision | Start correctness path with FP32 Q/accumulation; then try tiled dequant/tensor cores | Same pick gate or gated quality equivalence; full-domain normalization retained. Near-boundary cases use reference recomputation/fallback until explained |
| Packed paging, independently | Existing pager race/serialization/truncation tests plus raw-byte roundtrip, restore/session-switch and partial-chunk fixtures; smoke noop logits comparison | Identical logical resident sets, mask, K/V bytes and recurrent state; preserved stream dependencies; no pick differences |
| Replay/checkpoint optimization | Compare scratch CPU/GPU state bytes and final logits before changing diagnostic copies; keep identical batch partitioning | Same outputs under reference precision; all lifecycle/error paths covered |
| End-to-end acceptance | Same-binary CPU-v2, GPU-target and OFF arms on the retained harness, including primers; randomized/interleaved arm order by trajectory | Gates below; then separate runs on HIP gfx1151 and R9700 |

End-to-end protocol:

- Freeze settings and acceptance rules before new answers. Preserve **951 probes**, **60 trajectories**, the existing beyond/within bands and the within-excluding-diagnostic guard. Same executable/library hashes for the arms; isolate only intended flags. CPU v2 is the quality reference; OFF measures the achievable baseline. Historical ON6 is contextual evidence, not the final control.
- Report per-arm and paired **prefill/decode/client-wall medians**, decode tokens/s, fixed-output-length replay timing, p90/p99, fitting/reselected subsets, pt/cached prefix, Q count, scorer stages, page bytes/events, peak GPU/pinned memory. Preserve `time.time()` and add `perf_counter_ns()` on the client to diagnose clock steps. Bootstrap by trajectory, using the harness's existing method.
- **Latency goal:** median client wall **≤ same-binary OFF + 250 ms**; with old OFF this is **1.168 s** (D). Also prefill median **≤ OFF + 250 ms** and decode throughput **≥97% of OFF** (proposed gates). Record a stretch goal **OFF + 100 ms**; do not claim it for the exact replay design before measurement. G-only is expected to fail the main latency goal.
- **Quality gate:** proposed equivalence interval **±1 percentage point overall** against CPU v2, with the **95% trajectory-clustered CI entirely inside** that interval. Point difference alone is insufficient. Require no more than **1 point** observed loss in either beyond or within bands; report their clustered CIs and the diagnostic-excluded band. If unchanged picks/answers give exact equality, record it. If CI is too wide, declare inconclusive and add independent trajectories; do not relax the gate post hoc.
- **Structural gate:** zero request errors and restore/residency mismatches; identical per-request cached-prefix/pt values, matching recent-only first-pass visibility, no hidden disk hits, fixed 16K attention pool and 60-chunk recall budget. A GPU archive changes storage, not the visible attention set.
- GPU scorer checkpoint goal **≤50 ms median** at the harness's scoring contexts, with max-context scaling measured separately. It is an internal budget, not a substitute for wall or quality gates. Run only necessary checks after each independent change; benchmark the combined pipeline after its component gates pass.

## Open questions and smallest answering experiments

| Unknown | Smallest future experiment |
|---|---|
| Is replay compute or state handling dominant in the 235 ms package? | Instrument the existing OFF/noop/ON6 smoke probes; separate cut/build, GPU replay, scratch save/restore, truncate and diagnostic-logit reads. No selector change |
| Are page submissions/zeros the 221 ms smoke package? | Replay one recorded pool transition from pinned pages; compare existing per-head copies with one packed batch. Count API calls and time host submission, DMA, zero and scatter separately |
| Can the main target meet OFF+250 ms? | Combine measured phase budgets from the preceding smokes and one GPU scorer timing, then run the full harness if the budget is plausible |
| GPU QK/reductions and precision cost? | One frozen current-query/key input at median scoring size and at max context; compare CPU IDs and stage times for FP32 correctness and accelerated input precision |
| How much does query gathering cost? | Same input, old host gather versus GPU ring with scoring shadowed/disabled; retain batch cuts and checksum the captured tail |
| Is extra GPU memory safe, including AMD? | Record peak allocated/reserved/free memory on the longest existing primer with K/Q/workspace allocated; then optionally add the full-KV archive. Repeat on each AMD device when available |
| Can a shortlist preserve rare sharp-token winners? | CPU-only replay of available Q/K dumps; sweep the proposed shortlist sizes, report winning-unit recall, boundary misses and full-domain versus restricted-normalizer picks. Missing raw keys require later capture, not a claim from mass-only dumps |
| Can replay be shortened without losing the gain? | Same-binary smoke, then a preregistered full 951-probe comparison of full-turn versus final-token replay, current-query picks held fixed. Inspect beyond answers and recurrent-state/logit drift |
| Are previous-tail hints useful and safe? | Replay consecutive real agent turns, measuring hint/current-pick overlap and idle lead time. Existing independent probe branches cannot establish production hit rate |
| Can caching or no-op fast paths matter? | Hash actual Q and key generations, log exact repeats and first-pass/selected-view equality over the harness before implementing a score cache |
| Does layer overlap help on a busy single GPU? | Compare serial and event-overlapped scorer/transfer schedules on one identical frozen request; keep only a wall improvement, check decode contention |
| Does provisional quality survive a same-binary reference? | Read the completed reference report when available, then use CPU-v2 in the new same-binary experiment. No extrapolation from provisional counts |

All latency ranges above are estimates or projections. No GPU timings, builds, or quality reruns were performed to produce this design.
