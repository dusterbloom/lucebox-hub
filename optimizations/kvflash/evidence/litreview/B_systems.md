# KVFlash 2.0 prior-art review (systems side), 2026-10-16

Bottom line: we mostly recycled. The closest prior work is KVMem (arXiv 2609.04852, 4 Sep 2026). It already does exact-block recall from a GPU/host/NVMe tiered pool, scored by attention-space indexes, then re-prefill of the query, on a 24 GB consumer GPU with a Qwen 27B model, and it reports beating compaction. On exact session restore plus spec decode, llama.cpp merged draft-state slot save/restore on 2026-10-15 (PR #30574). What is left that looks new is narrow (see the end).

Verification note: "OPENED" means I fetched the page. "SNIPPET" means I saw it only in a search result and did not open it; do not cite those without opening them. The paper-search MCP returned empty lists for several arXiv queries (KVFlow, Continuum, Marconi, SCBench); I fell back to WebSearch and then fetched the arXiv abs pages. The mcp-agent-mail server failed to connect (not needed here).

## S1. Multi-turn KV reuse and tiering with persistence. Verdict: RECYCLED

| Prior work (all OPENED) | Same | Different |
|---|---|---|
| Stateful Large Language Model Serving with Pensieve, Yu, Lin, Li, arXiv 2312.05516 (Dec 2023) https://arxiv.org/abs/2312.05516 | Cross-turn KV cache, GPU+CPU multi-tier | No SSD, no spec decode, no exactness claim |
| Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention, Gao et al., arXiv 2403.19708 (ATC 2024) https://arxiv.org/abs/2403.19708 | Hierarchical cache over memory and storage, layer-wise preload, async save | Cluster serving, not a local bounded pool |
| LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference, Liu et al., arXiv 2510.09665 https://arxiv.org/abs/2510.09665 | GPU/CPU/storage/network tiers for vLLM and SGLang | Enterprise scale; no spec-decode state |
| Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving, Qin et al., arXiv 2407.00079 https://arxiv.org/abs/2407.00079 | KVCache pooled over DRAM and SSD | Disaggregated cluster |
| SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention), Zheng et al., arXiv 2312.07104 https://arxiv.org/abs/2312.07104 | Prefix-tree KV reuse for multi-turn chat and agents | In-GPU, no persistence by itself |
| CacheGen, Liu et al., arXiv 2310.07240 https://arxiv.org/abs/2310.07240 | Stored KV streamed back | Lossy compressed bitstreams |
| Agent Memory Below the Prompt: Persistent Q4 KV Cache for Multi-Agent LLM Inference on Edge Devices, Shkolnikov, arXiv 2603.04428 https://arxiv.org/abs/2603.04428 | Per-agent KV persisted to disk and reloaded on edge hardware, block pool under a budget, TTFT down up to 136x | Q4, so not exact (perplexity +2.8%, +3.0%); no spec decode |
| KVFlow, Pan et al., arXiv 2507.07400 https://arxiv.org/abs/2507.07400; Continuum, Li et al., arXiv 2511.02230 https://arxiv.org/abs/2511.02230; Marconi, Pan et al., arXiv 2411.19379 https://arxiv.org/abs/2411.19379 | Agentic KV eviction/prefetch, TTL pinning across tool calls, hybrid-model prefix caching | Scheduling and eviction policy, GPU-resident. Marconi matters if the 27B model is hybrid |

Tiers, SSD persistence and per-turn checkpoints are all published or shipped. Our 72 h TTL and 128 GB budget are engineering defaults.

## S2. Exact restore + bounded pool + spec decode in one engine on a consumer GPU. Verdict: KNOWN VARIANT (claim does not survive as worded)

| Evidence (OPENED) | What it shows |
|---|---|
| llama.cpp PR #30574 "server: save the draft state in the slot file", ServeurpersoCom, merged 2026-10-15 by ggerganov https://github.com/ggml-org/llama.cpp/pull/30574 | Slot file now stores target context, checkpoints and the draft context. Draft acceptance on the turn after save/erase/restore: Qwen3.8-27B with draft-mtp 37% to 49%. Author states output is identical in all cases; test `test_slot_restore_draft_state` added |
| llama.cpp issue #28619 (opened 2026-09-08, closed 2026-10-15) https://github.com/ggml-org/llama.cpp/issues/28619 | Before the fix, restore silently left the draft cold. Until yesterday our claim held against llama.cpp; it no longer does |
| KVMem, arXiv 2609.04852 https://arxiv.org/abs/2609.04852 | Bounded GPU page pool + host + NVMe on a 24 GB RTX 5090 Laptop GPU, Qwen 27B, runs with MTP. The paper does not describe persistence across restarts or how MTP interacts with paging |
| 2603.04428 | Persistent disk KV on edge devices, not exact, no spec decode |

Per piece: bounded pool plus host/NVMe tiers (KVMem), spec decode in the same engine (KVMem, llama.cpp), save/restore of target and draft with identical output (llama.cpp). No opened source shows pager-state restore (pool + host copy + checkpoints) plus spec decode with byte-identical warm turns verified. llama.cpp with slots, draft and slot-save is one config away, and only lacks a bounded recall pool. Reword the claim: do not say "no other open stack"; say we verify byte-identical warm restore of a recall-pager state with a block-diffusion drafter.

Not checked: vLLM+LMCache+spec decode determinism (searches found nothing on bit-exactness; the vLLM APC doc, OPENED, mentions spec decode only for Mamba-group checkpoints). ExLlamaV2/V3, TensorRT-LLM, Ollama, LM Studio: not searched specifically, so leave them out of any claim.

## S3. Attention Matching compaction and our negative result. Verdict: KNOWN VARIANT

| Source (OPENED) | Content |
|---|---|
| Fast KV Compaction via Attention Matching, Zweiger, Fu, Guo, Kim, arXiv 2602.16284 (Feb 2026) https://arxiv.org/abs/2602.16284 | Original AM: compact K/V reproducing attention outputs and mass per KV head, up to 50x compaction in seconds |
| Practical Online KV Cache Compaction for LLM Agents: An Empirical Study, Liu, Ji, An, Jain, Polatkan, Zhu, Chang, arXiv 2608.00902 (2 Aug 2026) https://arxiv.org/abs/2608.00902 (HTML read at https://arxiv.org/html/2608.00902v1) | Compares token eviction (TE, keeps original K/V) with AM on BrowseComp-Plus and WideSearch. "TE is often more robust than AM under imperfect proxies"; TE cuts KV 80% with most accuracy. Immediate compaction often hurts (Qwen3.5-4B 46.00% to 32.75% with immediate TE). AM as low as 11.25% on Gemma-4-E4B vs 33.00% uncompressed. Cites SCBench. Limitation stated: code editing not covered. Uses 10,000 example-level bootstrap and 95% CIs; no needle probes |
| SCBench: A KV Cache-Centric Analysis of Long-Context Methods, Li et al., arXiv 2412.10319 (ICLR 2025) https://arxiv.org/abs/2412.10319 | "sub-O(n) memory methods suffer in multi-turn scenarios" |
| KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction, Kim et al., arXiv 2505.23416 https://arxiv.org/abs/2505.23416 | Query-aware eviction degrades under multiple queries even at 90% budget |
| KVMem 2609.04852 | Reports exact-block selection beating compaction-based approaches on LongMemEval, MemoryAgentBench, AgentLongBench; DeepSWE success 43.8% (compaction only) to 48.4% (Qwen3.8-27B) |

Our 11.3% retention failure and "exact recall beats compaction" match 2608.00902 (selection beats AM, AM collapses to about 11% in one cell) and KVMem. What differs: coding-agent trajectories (2608.00902 says code editing is not covered), AM inside a pager, and compact-on-evict as a second failure. This is a useful negative replication on a new workload, not a new finding. Cite 2608.00902 as "often more robust under imperfect proxies", not as a blanket claim.

## S4. Benchmarks and methodology. Verdict: KNOWN VARIANT for agentic KV evaluation; POSSIBLY NEW for needle-in-trajectory probes and preregistration

| Source (OPENED) | Relevance |
|---|---|
| SCBench 2412.10319 | Multi-turn and multi-request shared-context KV benchmark, 12 tasks |
| AgentKV: Phase-Aware KV Eviction for Agentic LLMs, Liu et al., arXiv 2609.14872 (Sep 2026) https://arxiv.org/abs/2609.14872 | BFCL multi-turn and tau2-bench; paired instance-level bootstrap, CI-excludes-zero rule; no needle probes, no cluster-robust, no preregistration |
| KVMem 2609.04852 | LongMemEval-S, MemoryAgentBench, AgentLongBench, DeepSWE (16 tasks x 4 runs); no needle probes, significance tests or preregistration |
| IntentKV, arXiv 2606.09916 https://arxiv.org/abs/2606.09916; CommitKV, arXiv 2608.07855 https://arxiv.org/abs/2608.07855; CodeComp, arXiv 2604.10235 https://arxiv.org/abs/2604.10235; MemDecay, arXiv 2607.10582 https://arxiv.org/abs/2607.10582 | Agent/code KV methods. MemDecay uses synthetic probes at about 450 and 1,700 tokens (abstract page) |
| KVDiagnosis, arXiv 2608.09412 https://arxiv.org/abs/2608.09412 | Diagnostic benchmark for why KV compression fails, full-cache control |
| 2608.00902 | Example-level bootstrap CIs on agent trajectories |

I found no paper that inserts needle probes into real agent trajectories at known depths. I found no preregistered, frozen-rule evaluation in KV-cache or inference-systems work (two searches, below). Trajectory-clustered bootstrap is not reported in any opened paper; the PAGE (2609.22157) and MemDecay cluster-style bootstraps appear only in a search snippet, and PAGE's abstract page does not describe one. This is weak evidence of absence. LongMemEval, RULER and LongBench v2 were not opened.

## S5. PFlash / speculative prefill and selector heads. Verdict: RECYCLED (PFlash); KNOWN VARIANT (selector heads)

| Source | Relevance |
|---|---|
| Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation, Liu, Chen, Zhang, arXiv 2502.02789 (Feb 2025, ICML 2025) https://arxiv.org/abs/2502.02789 (OPENED). It is speculative PREFILL, not speculative decoding | Small model estimates token importance, big model prefills only those tokens. PFlash is this idea |
| Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models, Upasani et al., arXiv 2603.02631 https://arxiv.org/abs/2603.02631 (OPENED) | Cross-family drafters, aimed at agentic workloads |
| Luce PFlash pages https://github.com/Luce-Org/lucebox-hub/tree/main/pflash and https://www.lucebox.com/blog/pflash (SNIPPET, our own project) | States it stacks Cross-Family Speculative Prefill and FlashPrefill |
| IntentKV 2606.09916 (OPENED) | Learned zero-initialised residual head over current-query keys, frozen base model, trained for agent trajectories |
| On the Limits of Learned Importance Scoring for KV Cache Compression, Steele, arXiv 2601.14279 https://arxiv.org/abs/2601.14279 (OPENED) | A learned query-agnostic scorer did not beat position heuristics or prefill attention; a warning for selector heads |

The C++/CUDA engineering is ours; the idea is published. A trained selector head for per-turn recall from host memory: IntentKV is the nearest, but it prunes, it does not recall.

## Queries behind POSSIBLY NEW items
- "benchmark KV cache compression agentic coding trajectories multi-turn tool use needle recall 2026 arXiv"
- "preregistered KV cache eviction evaluation cluster bootstrap confidence intervals LLM inference"
- '"pre-registered" OR "preregistered" LLM inference systems evaluation KV cache frozen decision rules'
- "speculative decoding with prefix caching persistent KV cache SSD offload consumer GPU single engine"
- "vLLM speculative decoding prefix caching KV offload disk LMCache compatibility session resume byte-identical output"
- "learned token importance selector head agent trace prefill compression KV selection recall from host memory per-turn arXiv 2026"
- "llama.cpp server slot save restore KV cache to file speculative decoding draft model slots"
Not run: engine-specific queries for ExLlama, TensorRT-LLM, Ollama, LM Studio; Google Scholar.

## Scoring
- Recycled: tiered multi-turn KV, SSD persistence, speculative prefill, AM and its loss to selection under imperfect proxies, per-turn exact-block recall with re-prefill on a consumer GPU (KVMem).
- Known variants: draft-state restore with identical output (llama.cpp #30574), agentic KV evaluation (AgentKV, KVMem, 2608.00902), selector heads (IntentKV).
- Possibly new, narrow: (a) byte-identical warm restore of a recall-pager (pool + host copy + checkpoints) with a block-diffusion drafter in the same engine; (b) needle probes at known depth inside real coding-agent trajectories; (c) preregistered frozen rules plus trajectory-clustered bootstrap plus the structural resident-set-equals-baseline check; (d) the negative result that AM and compact-on-evict fail on coding-agent traces.

## Does anything already do our whole stack?
No single source does all of it. KVMem covers the recall-with-re-prefill pager on a 24 GB GPU with a 27B model and beats compaction. llama.cpp PR #30574 covers target and draft slot restore with identical output. Together they cover everything except the combination, the evaluation protocol and the coding-agent negative result. I read KVMem's HTML only to character 100,000 of 109,400.
