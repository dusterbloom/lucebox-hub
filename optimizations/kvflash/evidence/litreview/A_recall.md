# Prior-art review: KVFlash recall (mode 4 / prefill reselect). Date 2026-10-16

Method note. Every paper below was opened (arXiv abs/html page, NeurIPS page, or the arXiv API record via the paper-search tool). Title, authors and dates are copied from what I read. Where a PDF came back as binary I used the arXiv HTML version. The paper-search tool returned empty results on several calls (the I4 queries, search_semantic, and some name lookups for EPIC, APE, SpecPrefill, Memorizing Transformers, Unlimiformer); I fell back to WebSearch and direct abs-page fetches. I did not read the full text of every paper: method claims come from abstracts unless marked "(full text)". The full-text reads (InfLLM, Quest, KVMem) went through a page summarizer, not my own eyes. ArkVale has no arXiv id (NeurIPS 2024 only).

## Bottom line
We did not invent the architecture. One paper from five weeks ago, KVMem (arXiv 2609.04852, Sep 2026), describes almost our whole pipeline: bounded paged GPU pool, sink + recent always kept, history in host memory, score history blocks from an initial query prefill, rewind to a checkpoint, re-prefill the query over the recalled view, then decode with the set fixed. I1, I2, I3 and I5 are all published. The only thing I could not find anywhere is I4: the contamination feedback loop and the reset-to-recency-before-scoring fix. That is the single candidate for novelty, and it is a system-level finding, not a new algorithm. I6 (exact full-key scoring) is a design choice, not a novelty.

## I1. Sink + recent window + top-k old KV blocks offloaded to CPU
Verdict: RECYCLED.

| Paper | Same | Different |
|---|---|---|
| InfLLM, Xiao et al., arXiv 2402.04617 (2024-02) | Initial tokens + local window always kept; old KV in 128-token blocks on CPU; top-k blocks by query-key score loaded to GPU; GPU block cache (full text) | Lookup every step (512-token chunk in prefill, one token in decode); representative-token scoring (approximate); no multi-turn framing |
| ArkVale, Chen et al., NeurIPS 2024 (no arXiv) | Pages backed up to CPU, evicted pages recalled by importance, top-ranked pages attended | Per-step recall; per-page bounding-volume digests (approximate) |
| KVMem, Chai et al., arXiv 2609.04852 (2026-09-04) | Bounded GPU pool, 24 GB consumer GPU, 27B model, host and NVMe tiers, sink + recent retained, top-scoring blocks restored in chronological order | See I3 |

Same family, read at abstract level: Quest 2406.10774, InfiniGen 2406.19707, ShadowKV 2410.21465, RetrievalAttention 2409.10516 (RTX 4090 24 GB, 128K), MagicPIG 2410.16179, RetroInfer 2505.02922, ClusterKV 2412.03213, FreeKV 2505.13109, ParisKV 2602.07721, Landmark Attention 2305.16300, Memorizing Transformers 2203.08913, Unlimiformer 2305.01625, EM-LLM 2407.09450. StreamingLLM 2309.17453 is the sink + recent baseline, which is our OFF arm. This idea is the mainstream of the field.

## I2. Select from the attention of the last prompt token(s)
Verdict: RECYCLED.

| Paper | Same | Different |
|---|---|---|
| SnapKV, Li et al., arXiv 2404.14469 | Uses an observation window at the end of the prompt to pick important KV | Permanent eviction, per head, from the prompt itself; ours recalls from a host store |
| Lookahead Q-Cache, Wang et al., arXiv 2505.20334 (EMNLP 2025) | Argues prefill-time scores mismatch the queries that matter; builds pseudo queries | Eviction, not recall |
| InfoFlow KV, arXiv 2603.05353 | Attention-norm signal from the query picks tokens to recompute | RAG chunk reuse; selects tokens to recompute, not blocks to page in |
| ActQKV, Xiao et al., arXiv 2502.13542 | Chooses probe queries from the prefill window for KV retrieval | Probe selection via an activation indicator; we use only the last token |

H2O (2306.14048) is the accumulated-attention eviction ancestor. Using the last token alone as the query is a narrow choice (KVMem averages over query tokens). That is a twist, not a new idea.

## I3. Prefill, score, rewind, recall, re-prefill
Verdict: KNOWN VARIANT, bordering RECYCLED. This is the closest hit.

| Paper | Same | Different |
|---|---|---|
| KVMem, arXiv 2609.04852 (full text, HTML) | "The initial query prefill is used to drive retrieval." Working set updated "after the current-step prefill and before decoding", fixed during decoding. "The current query is then re-prefilled over the updated view." A checkpoint taken "immediately before the current query" is restored (the rewind). Sink and recent blocks always retained; remaining budget goes to top-scoring blocks. | Scores Mean-K block vectors (an approximate index) averaged over query tokens and layer-head pairs, softmax across blocks; 32-token blocks; 64K-80K token views; benchmarks LongMemEval, MemoryAgentBench, AgentLongBench, DeepSWE; baselines are full context, sliding window, compaction, compaction+RAG, with no LRU/StreamingLLM numbers; no ablations |
| CacheBlend 2405.16444, EPIC 2410.15332, APE 2502.05431, InfoFlow KV 2603.05353 | Selectively recompute token KV so text attends to reused context | Reuse of independently cached document KV in RAG; direction is reversed (they recompute cached chunks, we recompute the new tokens) |
| Speculative Prefill, Liu, Chen, Zhang, arXiv 2502.02789 | A small model estimates which prompt tokens matter | Prompt-token pruning for TTFT, not KV recall from history |
| UNREAL, arXiv 2610.08463 (2026-10-06) | Two passes: one forward pass forms retrieval queries, a second pass generates over selected chunks | Learned retrieval tokens, selects text chunks, not paged KV |

KVMem is public prior art as of today (it may be concurrent with our work). Our differences from KVMem: exact last-token scoring (I6), 64-token chunks summed into 640-token blocks, and the OFF-vs-ON study on needle probes in real agent trajectories.

## I4. Next-retrieval query contaminated by previously recalled blocks; fix = reset to recency before scoring
Verdict: POSSIBLY NEW. Moderate confidence. The nearest prior art does not contain it.

Closest items, and why none is the same:
- KVMem 2609.04852. Says the initial prefill "drives retrieval before the new working set is assembled" and warns of "a stale historical working set" when recall is infrequent. It does not state what the GPU view holds during that initial prefill, and says nothing about contamination, carry-over, feedback or a reset to recency. Caveat: this was read through a summarizer. Re-read its Sec. 5.2 yourself before claiming priority.
- ParisKV 2602.07721. "Drift" is a distribution shift between prefill-only centroids and later keys. Not a query-context feedback loop.
- ActQKV 2502.13542. Reports InfLLM's retrieval queries are poor under sparse attention (via search summary). Query quality, not feedback from earlier recalls.
- InfLLM (full text). Structurally has the same exposure, since retrieved blocks sit in context when later queries are computed, but I saw no discussion of it.
- IntentKV 2606.09916: cross-turn intent accumulation on purpose, the opposite problem.
- Memorizing Transformers "staleness": training-time distribution shift of stored keys.
- Seen in search summaries only, not opened, so not relied on: TempoFit 2603.07647 (stale history interfering with the present), LoopGuard 2604.10044 (self-reinforcing attention loops, in repetition), text-memory loop posts (CoEvo-Mem 2608.01739). All different level or mechanism.

Queries run, none returned the loop:
1. "arXiv retrieval query drift contaminated context KV cache retrieval feedback loop retrieved tokens influence next retrieval"
2. "memory-augmented transformer retrieval query computed with retrieved memory in context stale memory bias next retrieval KV cache retrieval self-reinforcing"
3. "KV cache recall multi-turn stale recalled blocks occupy GPU cache pollution reset before retrieval agent sessions arXiv 2026"
4. "KV retrieval query contaminated by previously retrieved blocks recency reset before scoring" (arXiv tool, empty)
5. Same idea on the Semantic Scholar tool (empty)
6. "retrieval feedback loop self-reinforcing recalled KV blocks bias subsequent retrieval long-context agent KV cache recall ablation recency pool"
7. "retrieval-conditioned query: compute retrieval query with local context only, then re-encode query with retrieved memory, KV cache memory retrieval two-pass encode"
8. "KV cache recall multi-turn dialogue attention drift query tokens recent-only window score history blocks topic switch agents 2026 sink recent window plus recalled blocks"
9. "retrieval KV cache recall previously retrieved blocks query shifted attention distribution retrieved context changes query vectors layers InfLLM limitation"

Not run: Google Scholar, OpenReview full-text search, citation-chasing forward from KVMem. A search-summary tool failing to find something is not proof it is absent. Also, "score from a clean recency context" is the obvious repair once you see the problem, so what is ours is the diagnosis and the measured effect (814 vs 847 at k=6, lost accuracy on recent material), not the mechanism.

## I5. Static per-turn selection vs per-step dynamic selection
Verdict: KNOWN VARIANT.

| Paper | Same | Different |
|---|---|---|
| KVMem 2609.04852 | Recall per agent step, working set "stays fixed during decoding"; motivates it with attention KL of 0.070 bits within a step vs 2.59 across steps | Same as I3 |
| RocketKV, Behnam et al., arXiv 2502.14051 (RocketKV-MT) | Multi-turn variant keeps all KV history across turns | Still selects dynamically within decode |
| SnapKV 2404.14469; SAGE-KV 2503.08879 | One-time selection after prefill, static through decode | Permanent eviction |
| Quest, InfLLM, ArkVale, InfiniGen | Per-step reselection (the contrast case) | Dynamic |
| SCBench 2412.10319 (ICLR 2025) | Shows sub-O(n)-memory methods degrade in multi-turn | Benchmark, not a method |
| EpiCache 2509.17396 | Multi-turn conversation, fixed budget, per-episode eviction | Eviction by episode clustering |

Static per-turn recall for agent serving is published (KVMem).

## I6. Exact scoring of all history chunks on CPU
Verdict: KNOWN VARIANT. A design choice, not a result.

KVMem scores exactly over an approximate Mean-K index; InfLLM uses representative tokens; Quest uses min/max bounds; ArkVale uses digests; RetrievalAttention, MagicPIG and RetroInfer use ANN/LSH/clustering indexes; ReTopK 2607.27692 reranks with exact scores for the current query. Exact q.K over all keys is the "oracle top-k" these papers compare against. Running it once per turn rather than once per token is what makes 35 ms affordable. I found no paper that presents per-turn exact full-key scoring as a contribution. Summing 10 chunk masses into a block is exact over keys, unlike Mean-K.

## Does any paper do the whole pipeline?
KVMem (arXiv 2609.04852) does I1 + I2 + I3 + I5 almost verbatim at system level. It differs by the approximate Mean-K index, query-token-averaged scoring, 32-token blocks, different benchmarks, no OFF=LRU comparison, and no contamination analysis. Any write-up must cite it and position ours as exact last-token scoring, plus the contamination finding, plus an ablated comparison against a plain recency pool.

## References (all opened unless marked)
- StreamingLLM: Xiao, Tian, Chen, Han, Lewis. "Efficient Streaming Language Models with Attention Sinks". arXiv 2309.17453 (2023). https://arxiv.org/abs/2309.17453
- InfLLM: Xiao, Zhang, Han, Xiao, Lin, Zhang, Liu, Sun. "InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory". arXiv 2402.04617 (2024). https://arxiv.org/abs/2402.04617
- Quest: Tang, Zhao, Zhu, Xiao, Kasikci, Han. "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference". arXiv 2406.10774 (2024). https://arxiv.org/abs/2406.10774
- ArkVale: Chen et al. "ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction". NeurIPS 2024. https://papers.nips.cc/paper_files/paper/2024/hash/cd4b49379efac6e84186a3ffce108c37-Abstract-Conference.html
- InfiniGen: Lee, Lee, Seo, Sim. arXiv 2406.19707 (2024). https://arxiv.org/abs/2406.19707
- ShadowKV: Sun et al. arXiv 2410.21465 (2024). https://arxiv.org/abs/2410.21465
- RetrievalAttention: Liu et al. arXiv 2409.10516 (2024). https://arxiv.org/abs/2409.10516
- MagicPIG: Chen et al. arXiv 2410.16179 (2024). https://arxiv.org/abs/2410.16179
- RetroInfer: Chen et al. arXiv 2505.02922 (2025). https://arxiv.org/abs/2505.02922
- ClusterKV: Liu, Li, Zhao, Zhang, Guo. arXiv 2412.03213 (2024). https://arxiv.org/abs/2412.03213
- FreeKV: Liu et al. arXiv 2505.13109 (2025). https://arxiv.org/abs/2505.13109
- ParisKV: Qi et al. arXiv 2602.07721 (2026). https://arxiv.org/abs/2602.07721
- Landmark Attention: Mohtashami, Jaggi. arXiv 2305.16300 (2023). https://arxiv.org/abs/2305.16300
- Memorizing Transformers: Wu, Rabe, Hutchins, Szegedy. arXiv 2203.08913 (2022). https://arxiv.org/abs/2203.08913
- Unlimiformer: Bertsch, Alon, Neubig, Gormley. arXiv 2305.01625 (2023). https://arxiv.org/abs/2305.01625
- EM-LLM: Fountas et al. "Human-inspired Episodic Memory for Infinite Context LLMs". arXiv 2407.09450 (2024). https://arxiv.org/abs/2407.09450
- SnapKV: Li et al. arXiv 2404.14469 (2024). https://arxiv.org/abs/2404.14469
- H2O: Zhang et al. arXiv 2306.14048 (2023). https://arxiv.org/abs/2306.14048
- Lookahead Q-Cache: Wang et al. arXiv 2505.20334 (2025). https://arxiv.org/abs/2505.20334
- ActQKV: Xiao et al. arXiv 2502.13542 (2025). https://arxiv.org/abs/2502.13542
- RocketKV: Behnam et al. arXiv 2502.14051 (2025). https://arxiv.org/abs/2502.14051
- SAGE-KV: Wang et al. arXiv 2503.08879 (2025). https://arxiv.org/abs/2503.08879 (arXiv API record)
- SCBench: Li et al. arXiv 2412.10319 (2024). https://arxiv.org/abs/2412.10319
- EpiCache: Kim et al. arXiv 2509.17396 (2025). https://arxiv.org/abs/2509.17396
- CacheBlend: Yao et al. arXiv 2405.16444 (2024). https://arxiv.org/abs/2405.16444 (arXiv API record)
- EPIC: Hu et al. arXiv 2410.15332 (2024). https://arxiv.org/abs/2410.15332
- APE: Yang, Chen, Chen. arXiv 2502.05431 (2025). https://arxiv.org/abs/2502.05431
- Speculative Prefill: Liu, Chen, Zhang. arXiv 2502.02789 (2025). https://arxiv.org/abs/2502.02789
- InfoFlow KV: Teng et al. arXiv 2603.05353 (2026). https://arxiv.org/abs/2603.05353
- IntentKV: Li, Lou, Li. arXiv 2606.09916 (2026). https://arxiv.org/abs/2606.09916
- ReTopK: Yao et al. arXiv 2607.27692 (2026). https://arxiv.org/abs/2607.27692
- Tessera: Fang et al. arXiv 2609.32999 (2026). https://arxiv.org/abs/2609.32999
- UNREAL: Kinderman et al. arXiv 2610.08463 (2026). https://arxiv.org/abs/2610.08463
- KVMem: Chai, Wang, Su, Xia, Yu. "KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU". arXiv 2609.04852 (2026-09-04). https://arxiv.org/abs/2609.04852
- Not opened (search-result mentions only, not relied on): ProphetKV 2602.02579, TempoFit 2603.07647, LoopGuard 2604.10044, CTkvr 2512.15550, HySparse2 2609.26368, OasisKV 2608.08097, CoEvo-Mem 2608.01739.
