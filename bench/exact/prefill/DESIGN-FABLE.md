# Prefill gap design: Fable (2026-10-13)

Design brief: remove lucebox's measured extra prefill kernel time against gufo (p=4096, matched text, lucebox4 iGPU; see GUFO-PARITY.md). Each change is scored on time-to-result and readability. Codex was given the same brief, and its design goes in DESIGN-CODEX.md.

Ranked change set. ms ranges marked "assumption" were not derived from code or measurement.

| # | Change | Hook | LOC +/- | ms saved | Decode | Hours | Readability |
|---|---|---|---|---|---|---|---|
| 1 | GDN row-split layout. Replace the tiled kernel with 4 lanes/row, a 2-step DPP reduction, 2 rows/lane and a 192x2-wave grid (port of gufo kernels.hip.cpp:1516-1760, MIT). | gated_delta_net.cu:666-812, :850-864 | +230/-150 | -100..-140 (assumption) | none | 10 | 3 |
| 2 | Fuse the q/k RMS_NORM+SCALE into GATED_DELTA_NET. | ggml-cuda.cu:7120, gated_delta_net.cu:729-740 | +35/0 | -40..-55 (assumption) | none | 4 | 4 |
| 3 | Routed GLU tile 64 -> 128 rows (down's shape). | mmb.cu glu dispatch | +2/-2 | -50..-100 (assumption) | none | 2 | 5 |
| 4 | Row-group rasterization in mmb_q8f16 for K>=4096 (gufo kRowGroup=5). | mmb-q8f16.cuh:65-68 | +6/-3 | -30..-40 (assumption) | none | 2 | 4 |
| 5 | HC combine with one block per token, the residual row in registers and a single xn store. | hc-cn.cu:81-150 | +60/-40 | -20..-40 (assumption) | keep the T=1 variant | 5 | 4 |
| 6 | Reuse the GLU call's route maps for the down call. | mmb.cu two mm_ids sites | +15/-5 | -12..-15 | none | 2 | 4 |
| 7 | W8A8 for ssm_out. Not exact; dropped in favour of #4. | mmb.cu:87-94 | +30/0 | -22 | none | 4 | 3 |

Root causes, with file:line evidence on both sides:
- **GDN:** two serial 32-lane reductions per token per column, and a 96-block grid that leaves a tail on 40 CUs.
- **Norms:** separate q/k norm passes, separate conv output, and F32->F16 activation conversions. gufo folds these into its producers.
- **Dense:** the same kernel lineage, but with a different rasterization, and the narrow-deep shapes go int8 in gufo.
- **hc_*:** one block per stream re-reads block_out 4x, and xn is stored as both F32 and bf16.

Overlap: #2 goes inside #1. #4 and #7 overlap.

Order by ms/hour: 3 -> 4 -> 6 -> 2 -> 1 -> 5.

The subset 1-6 gives an expected -250..-390 ms against a ~490 ms gap. That closes 50-80% of it, not parity. The remainder is gufo's model-specific GEMM-epilogue fusions.

Rejected:
- host/graph work (0.3% of wall);
- env knobs;
- dense Q6_K requant;
- kSsmConv/kAttention epilogue forks;
- sort-gate relaxation (in noise);
- per-expert tile ladder.
