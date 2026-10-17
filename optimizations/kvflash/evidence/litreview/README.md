# KVFlash 2.0: did we innovate or recycle? (literature review, 2026-10-16)

Sources:
- `A_recall.md`, the recall design;
- `B_systems.md`, session restore, compaction and evaluation.

Both were written by research agents that opened every paper they relied on. I re-checked the four claims that carry the verdict myself:
- KVMem's abstract and method text;
- llama.cpp PR #30574 (merged 2026-10-15T13:24Z);
- TempoFit;
- LoopGuard.

## Verdict: mostly recycled

The engineering is ours. The core ideas are published, and the closest paper predates our design.

**KVMem** does nearly our whole recall pipeline. It is "KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU", Chai, Wang, Su, Xia and Yu, [arXiv 2609.04852](https://arxiv.org/abs/2609.04852), submitted 4 Sep 2026, six weeks before our rlru4 design. Like us, it has:
- paged KV on GPU, host and NVMe, under a bounded pool;
- the sink and recent blocks always kept;
- one recall per agent step, driven by an initial prefill of the query;
- a restore to a checkpoint taken just before the query, then a re-prefill of the query over the recalled view;
- the recalled set held fixed during decode;
- a 27B Qwen model on a 24 GB GPU, with MTP.

Any write-up must cite it.

## Idea by idea

| Our idea | Verdict | Closest prior work | What is ours |
|---|---|---|---|
| Sink, recent window and top-k old blocks recalled from host | Recycled | InfLLM ([2402.04617](https://arxiv.org/abs/2402.04617)), ArkVale (NeurIPS 2024), KVMem | nothing |
| Choosing blocks from the last prompt token's attention | Recycled | SnapKV ([2404.14469](https://arxiv.org/abs/2404.14469)) | nothing |
| Prefill, score, rewind, recall, re-prefill the turn | Recycled | KVMem ("restoring a checkpoint taken immediately before the current query … re-prefills the current query over the updated execution view") | nothing |
| Static per-turn selection | Known variant | KVMem ("remains fixed throughout decoding"), RocketKV ([2502.14051](https://arxiv.org/abs/2502.14051)) | nothing |
| Exact q·K over every history chunk | Known variant | exact top-k is the usual oracle baseline | the choice, not the idea; we never compared it with KVMem's Mean-K index |
| **Recalled blocks contaminate the next turn's scoring; reset the pool to recency before scoring** | **Possibly new**, moderate confidence | KVMem only mentions a "stale historical working set" and has no reset. LoopGuard ([2604.10044](https://arxiv.org/abs/2604.10044)) shows attention-ranked caches reinforcing themselves, but in decode-time eviction and repetition, not recall. | The diagnosis and the measurement: 814 → 847 at k = 6, and the within-16K loss disappears. The fix is obvious once the problem is seen. |
| Multi-turn KV tiering and persistence (GPU, host, SSD) | Recycled | Pensieve ([2312.05516](https://arxiv.org/abs/2312.05516)), CachedAttention ([2403.19708](https://arxiv.org/abs/2403.19708)), LMCache ([2510.09665](https://arxiv.org/abs/2510.09665)) | nothing |
| Exact session restore plus bounded pool plus spec decode in one engine | Known variant | llama.cpp PR [#30574](https://github.com/ggml-org/llama.cpp/pull/30574) (draft state in the slot file, merged yesterday); KVMem (bounded pool, tiers and MTP on 24 GB) | Byte-identical restore of a *recall pager's* state with a drafter is narrow but not found elsewhere |
| Attention Matching compaction | Recycled, as intended | "Fast KV Compaction via Attention Matching" ([2602.16284](https://arxiv.org/abs/2602.16284)) | nothing |
| Compaction fails on agent traces; exact blocks win | Known variant | [2608.00902](https://arxiv.org/abs/2608.00902) (eviction beats Attention Matching under imperfect proxies, AM as low as 11.25% against 33.00% in one cell, coding agents not covered); KVMem | The coding-agent setting, and compact-on-evict failing too |
| PFlash | Recycled | Speculative Prefill ([2502.02789](https://arxiv.org/abs/2502.02789)) | nothing |
| **Evaluation method** | **Possibly new** for KV-cache work | AgentKV ([2609.14872](https://arxiv.org/abs/2609.14872)) uses a paired bootstrap; 2608.00902 uses an example-level bootstrap | Needle probes at known depth inside real coding-agent trajectories; preregistered frozen rules; a trajectory-clustered bootstrap; the resident-set structural check. None of these was found in any KV paper opened. |

## What this changes

1. **Our uniqueness claim does not survive.** `tracking/doc_rev153.md` says "no other open stack does exact session restore, a bounded KV pool and spec decode". llama.cpp now restores draft state, and KVMem runs a bounded pool with MTP on 24 GB. The defensible claim is narrower: exact, byte-identical restore of a recall pager's state with a drafter, measured on coding-agent traces with preregistered rules.
2. **Two things are worth claiming:**
   - the contamination finding, with its ablation (flag off 814 against flag on 847 at k = 6);
   - the evaluation method.
   Both need a citation search forward from KVMem, plus a full read of its section 5.2, before we claim them.
3. **We do not know whether our scoring beats KVMem's.** Exact last-token q·K over 640-token blocks has never been compared with Mean-K over 32-token blocks. One arm on our probe set, with the same pool, k and recency pool, would answer it.

## Limits

- The paper-search MCP returned empty results on many calls, so the agents used web search and fetched arXiv pages directly.
- The following were not searched:
  - Google Scholar;
  - OpenReview full text;
  - forward citations of KVMem;
  - ExLlama, TensorRT-LLM, Ollama and LM Studio, for the restore-plus-drafter combination.
- Papers seen only in search snippets are listed in `A_recall.md` and `B_systems.md`, and were not relied on.
