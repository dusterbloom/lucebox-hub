# PFlash: Paper Plan + Scaling Roadmap (Draft)

**Date:** 2026-05-21
**Status:** draft, written while empirics are still in flight

## TL;DR

We have a query-conditioned 4-gram anchor prefill compressor with a recall floor. It is novel vs all surveyed prior art. At 32K context on Qwen3.6-27B-Q4 it ships ~13× TTFT speedup with 100% NIAH retrieval. The real prize is scaling to 64K-1M where the technique becomes an *enabler*, not just an optimization. Yesterday's gemma4 work already proved the mechanism at 1M context on a 24 GB consumer GPU — we just haven't measured it for the Qwen3.6-27B family yet.

## Paper title candidates

1. **PFlash: Million-Context LLM Inference on a Single 24 GB Consumer GPU via Query-Conditioned Compression with Recall Floor** — the scaling-headline (preferred if Phase 3 lands)
2. PFlash: Query-Conditioned Prefix Compression Composes with Speculative Decoding for 13× TTFT at 32K — the MVP-headline
3. Anchor-Gated Prefix Compression with Recall Floor: A Pareto-Better Long-Context Inference Path

## Abstract (current evidence, before Phase 3)

> Long-context LLM inference is dominated by prefill cost. Existing prefix-compression methods (SnapKV, PyramidKV, H2O, Quest, SpecPrefill, vllm-mlx PR #180) use unconditional attention-score gating with a fixed budget — they cannot fall back to dense computation when their gating signal is weak, risking silent retrieval failures on abstract queries.
>
> We present **PFlash**, a query-conditioned prefix compression mechanism that (i) skips the drafter forward at long context, (ii) builds the keep-set from forced head/tail chunks plus 4-gram anchor matches between the last 96 query tokens and the body, and (iii) falls back to dense drafter forward when no anchor matches exist — a recall floor preserving retrieval correctness.
>
> On a single RTX 3090 with Qwen3.6-27B-Q4_K_M, PFlash achieves 13× wall-time speedup at 16K-32K context on NIAH single-needle while preserving 100% retrieval across a 48-cell (ctx × keep × mode) sweep. On 168 real Claude Code agentic turns from 65 captured sessions, anchor-zero rate is 6.5% overall and 0% above 32K body length — the band where PFlash's energy savings matter most.
>
> PFlash composes with **DFlash chain speculative decoding** (3/3 multi-turn OK_DONE under real compression). MTP composition is currently blocked on a re-init bug. We extend with a per-session bandit on accept-rate feedback for closed-loop compression that adapts per turn. Compressed KV at decode also reduces attention bandwidth proportionally to 1/keep, predicting a compounded prefill + decode speedup at long context.

## Paper structure (target 8-10 pages)

1. **Introduction** — prefill dominates TTFT; existing compression unconditional; PFlash adds query-conditioning + recall floor + composition with spec-decode + environmental framing
2. **Background & Related Work** — speculative decoding (DFlash chain, MTP γ-heads), KV compression (SnapKV/PyramidKV/H2O/Quest), SpecPrefill, vllm-mlx
3. **The PFlash Mechanism** — skip+anchor algorithm; recall floor; AUTO threshold; pseudocode of `compute_anchor_hits`
4. **Empirical Evaluation**
   - 4.1 NIAH single-needle envelope (48 cells, 100% accuracy throughout)
   - 4.2 Multi-turn agentic (claude_code + DFlash chain)
   - 4.3 Real-workload anchor coverage (168-turn study, the "0% above 32K" finding)
   - 4.4 Composition with DFlash chain (validated)
   - 4.5 Production server-log evidence (100% anchor coverage on 20 real requests)
5. **Extensions**
   - 5.1 Multi-resolution anchors (2+4-gram union, 100% rescue at 6.2 pp precision cost)
   - 5.2 Per-session bandit on accept-rate feedback (closed-loop adaptive keep_ratio)
   - 5.3 Compressed-decode hypothesis (theoretical 20×; bounded by FFN per Momus's analysis; predicted 3-6× empirical)
   - 5.4 **Scaling to 256K-1M context** — recall floor reactivates H1 cosine-pool backstop at extreme ctx where drafter doesn't fit
6. **Discussion & Limitations**
   - NIAH ceiling at ≤32K, opencode tool-loop variance, VMM behavior on <32 GB GPUs
   - MTP re-init bug (current P0 block)
   - Energy framing (~1.4 Wh per 32K-request saved at 350W)
7. **Conclusion** — single-card 27B inference at 32K-1M becomes practical for agentic workloads

## Scaling roadmap (the actual paper backbone)

### Phase 3A — energy + correctness at 64K-256K (NEW, after MVP)
- NIAH single-needle at 64K, 128K, 256K — ALWAYS only — does retrieval hold?
- Multi-turn agentic at 64K via claude_code — does accept rate hold? (uses DFlash chain since MTP currently broken)
- Energy per request at 64K vs 32K vs 16K — does per-token energy curve flatten?

### Phase 3B — anchor distribution at extreme ctx
- Anchor-zero rate on real workload at 64K+ (we have only 2 turns above 64K in the 168-turn sample — sparse, need more)
- If anchor-zero rate rises at extreme ctx → H1 backstop becomes load-bearing

### Phase 3C — cosine-pool backstop at extreme ctx
- Build H1, test on anchor-zero corpus at 64K+
- Cost: how many FLOPs? Fits in a single decode-step time budget?

### Phase 3D — 1M reasoning chain demo
- Long-form reasoning task at 1M context
- End-to-end latency, energy, correctness
- Direct comparison: gemma4 numbers exist (26 ± 5 tok/s at 1M) — can we match or beat for Qwen3.6-27B?

## Open questions (the 5.5 of the paper)

- **Adaptive query window**: currently fixed at last 96 tokens. Could be detected from chat-template last-user-turn boundary. Worth measuring.
- **Cross-tokenizer composition**: PFlash drafter uses Qwen3-0.6B tokenizer; target uses Qwen3.6-27B (Qwen3.5 family). Tokenizer cross-mapping is byte-level BPE bijective; verified in `dflash/scripts/laguna_pflash_niah.py`. Same approach for Qwen3 ↔ Qwen3.6.
- **TF-IDF anchor weighting**: ladder back to 1-gram for rare-token cases (file paths, UUIDs).
- **Suffix-array longest-match**: variable-length matching at O((N+M) log N).

## Status (2026-05-21 evening)

- 32K NIAH envelope: ✅ done
- Multi-turn DFlash composition: ✅ done
- MTP composition: ❌ broken (re-init bug)
- Real-transcript anchor coverage: ✅ done
- H2 multi-resolution validation: ✅ PROCEED verdict
- Codex adaptive-keep design: ✅ doc written
- Decode-rate theoretical model: ✅ done (20× theoretical, 3-6× expected)
- TTFT investigation v2: ⏸️ pending GPU
- 64K+ tests (Qwen3.6-27B family): ❌ not measured yet
- Cosine backstop empirics: ❌ not started
- vllm-mlx baseline comparison: ❌ not started

## Estimate to paper-ready

- 6-8 GPU-hours: Phase 3A (64K-256K) + remaining MVP gaps (MTP bug, TTFT v2, recall-floor ablation)
- 4 hours: vllm-mlx baseline comparison
- 8 hours writing
- **Total: ~2 working days from MTP-fix landing.**
