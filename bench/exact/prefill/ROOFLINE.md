# Per-kernel roofline ledger, p≈4096 prefill: lucebox vs gufo

Analysis only. CPU-side, no GPU use, no server runs. Source traces (measured,
not committed): `lucebox4:~/qwen4exp-f16/scratch/gufo/trace2/{lucebox,gufo}_out_csv/`
(rocprofv3 kernel traces, same files documented in `GUFO-PARITY.md`'s
"Matched-text kernel trace (p=4096)" section). Isolation script and per-kernel
aggregate CSVs: `lucebox4:~/qwen4exp-f16/scratch/roofline/` (`isolate.py`,
`extract.py`, `{lucebox,gufo}_kernel_agg.csv`). Measured window boundaries
were re-derived independently here (largest-gap method) and validated: wall
3652.444 ms (lucebox) / 2995.206 ms (gufo) match `GUFO-PARITY.md`'s profiled
wall numbers exactly; GDN/attention call counts (36/12 lucebox,
108/36 gufo) also match.

## Hardware constants (assumption, cited)

- Peak FP16 WMMA = 40 CU × 512 FLOP/clk/CU × 2.9 GHz = **59.4 TFLOP/s**.
  The 512 FLOP/clk/CU figure is **derived** from AMD's public RDNA3 spec: the
  7900 XTX (96 CU, 2.5 GHz boost) is rated 123 TFLOP/s FP16 WMMA →
  123e12/(96×2.5e9) = 512.5 FLOP/clk/CU, consistent with this task's number.
  gfx1151 (Strix Halo) uses the same WMMA ISA (RDNA3.5), so the formula
  carries over unchanged (assumption: no reduced issue rate on gfx1151).
- DRAM BW = 220 GB/s (measured, given).
- All weight-only-quantized GEMMs (Q8_0/Q4_K/Q5_1) are modeled as
  decode-to-FP16-then-WMMA, i.e. compute uses the FP16 peak above
  (assumption — real kernels may fuse dequant into the MMA differently).

## Per-op FLOPs/bytes and roofline (derived, T=4096 nominal)

All dense (non-routed) GEMMs below are **compute-bound at T=4096**: weight+
activation bytes are 2-3 orders of magnitude below what 220 GB/s would need
to dominate (e.g. attn_q: 155 MB/call → 0.7 ms BW-floor vs 52 ms compute-floor
over 12 layers). This is expected for prefill (large M dimension).

| op | layers | GFLOP/call | MB/call (w+act) | roofline ms (all layers) |
|---|---:|---:|---:|---:|
| attn_q 2560→12288 Q8_0 | 12 | 257.70 | 155.1 | 52.06 |
| attn_k 2560→512 Q8_0 | 12 | 10.74 | 26.6 | 2.17 |
| attn_v 2560→512 Q8_0 | 12 | 10.74 | 26.6 | 2.17 |
| attn_output 6144→2560 Q8_0 | 12 | 128.85 | 88.0 | 26.03 |
| qsa_q_proj 2560→512 BF16 | 12 | 10.74 | 27.8 | 2.17 |
| qsa_k_proj 2560→128 BF16 | 12 | 2.68 | 22.7 | 1.24 |
| flash-attn score (QK^T+AV, causal, assumption: head_dim 256, 48 Q-heads) | 12 | 34.4 | — | 83.0 |
| gdn_attn_qkv 2560→10240 Q8_0 | 36 | 214.75 | 132.7 | 130.15 |
| gdn_attn_gate 2560→6144 Q8_0 | 36 | 128.85 | 88.0 | 78.09 |
| gdn_ssm_out 6144→2560 Q8_0 | 36 | 128.85 | 88.0 | 78.09 |
| moe_gate 2560→640 Q4_K ×10 experts | 48 | 134.22 | 82.6 | 108.46 |
| moe_up 2560→640 Q4_K ×10 experts | 48 | 134.22 | 82.6 | 108.46 |
| moe_down 640→2560 Q5_1 ×10 experts | 48 | 134.22 | 84.7 | 108.46 |
| shared expert (gate/up+down) Q8_0 | 48 | 40.27 | — | 32.54 |
| router 2560→512 F32 | 48 | 10.74 | 30.4 | 8.68 |
| hc_attn_up/down 320×10240 Q8_0 ×2 | 48 | 53.69 | 180.0 | 43.38 |
| hc_ffn_up/down 320×10240 Q8_0 ×2 | 48 | 53.69 | 180.0 | 43.38 |

Σ derived compute-floor, all GEMMs + flash-attn score ≈ **908 ms**.
GDN recurrence (`gated_delta_net_tiled_cuda`, 48 heads × 128×128 state,
chunked causal scan) is **not** a GEMM-shaped roofline: naive per-token
state-update FLOPs (~4×head_dim²) give only ~8 ms compute-floor, two orders
below its 252-393 ms measured cost — the real bottleneck is sequential scan
depth across chunks, not FLOPs or bytes (**assumption**: this kernel's
"efficiency" number below is not meaningful under the simple roofline model).

## (a) Theoretical floor for p4096

- **Compute floor** ≈ 908 ms GEMM+attention-score (derived) + GDN recurrence
  (assumption, ~8-50 ms depending on scan-parallelism model) ≈ **~920-960 ms**.
- **Weight-bytes floor**: dense+attn+hc weights ≈ 3.5 GB (derived); MoE
  weight table, assuming all 512 experts get touched at least once across
  4096 tokens (assumption, reasonable for real text) ≈ 73 GB. Total ≈ 76.5 GB
  / 220 GB/s ≈ **348 ms**.
- **Activation-bytes floor**: sub-GB per class, never the binding constraint
  at T=4096 (derived).
- → compute-bound overall; floor ≈ **~950 ms** (derived+assumption blend).

## (b) Distance from floor

| | measured wall (profiled) | floor | ratio |
|---|---:|---:|---:|
| lucebox | 3652.44 ms | ~950 ms | **3.8x** |
| gufo | 2995.21 ms | ~950 ms | **3.2x** |

Both engines run at roughly 26-32% of the theoretical floor — expected,
since the floor ignores dequant overhead, irregular MoE routing, norms, and
idle/launch time.

## (c) Kernels far below their own roofline (inefficiency candidates)

- **Dense GEMM** class (attn/GDN/hc Q8_0 projections): measured 1024-1167 ms
  vs derived floor (attn-dense 85.8 + GDN-dense 286.3 + hc 86.8) ≈ 459 ms →
  **efficiency ~40-45%**. Consistent with Q8_0-decode-in-kernel overhead
  (`mmb_q8f16_kernel`, `mmb-q8f16.cuh:254-256`) rather than reaching WMMA
  peak.
- **Routed gate/up + Routed down** (MoE): measured 950-1022 ms combined vs
  naive full-batch-dense floor 325 ms → **efficiency ~32-34%** —
  but this floor is optimistic: real per-expert tiles average ~80 rows
  (4096×10/512), not one dense 40960-row GEMM, so launch/occupancy overhead
  per expert is structural, not a bug (**marked uncertain** — the "far below
  roofline" read may be a modeling artifact rather than a kernel defect).

## (d) At-roofline-but-redundant (unfused passes)

- **hc_* kernels**: measured 338.9-456.4 ms vs derived floor 86.8 ms (~4-5x
  over floor — also inefficient in absolute terms, but conceptually) exist
  **only** because of the 4-stream hyper-connection design: hc_attn and
  hc_ffn each add one full-width (T×10240) up+down GEMM pass per layer that
  a single-residual-stream model would not need at all. Pass count: **2
  extra full-width activation passes/layer × 48 layers = 96 system-wide**,
  vs a fused minimum of **0** — this is redundant-by-architecture, not a
  kernel bug.

## (e) Top 5 ms opportunities (ranked, causal chain) + reconciliation

Reusing `GUFO-PARITY.md`'s already-exact per-class deltas (measured, not
re-derived here) and adding the roofline framing:

1. **Routed gate/up, Δ177.71 ms (lucebox slower)** — `mmb_routed_glu_kernel<64,128,32,32,44>` (`mmb.cu:1146-1147`) tile shape underperforms gufo's `RoutedF16GEMMKernel` at this routing distribution.
2. **Dense GEMM, Δ142.53 ms (gufo faster)** — `mmb_q8f16_kernel<256,128,2,8,1,x>` (`mmb-q8f16.cuh:254-256`) below its own roofline (item c).
3. **GDN, Δ140.17 ms (lucebox slower)** — `gated_delta_net_tiled_cuda` recurrence, not GEMM-bound; needs scan-parallelism work, not tile retune.
4. **hc_*, Δ117.52 ms (lucebox slower)** — redundant-by-design passes (item d); real fix is fusing/eliminating, not rebalancing.
5. **Norms/elementwise/quantize, Δ97.78 ms (lucebox slower)** — many small (<1 ms) launches (`rms_norm_f32`, `k_argsort_f32_i32`, `gated_rms_norm*`); launch-overhead-bound at T=4096, not compute/BW-bound.

**Reconciliation table** (measured, ms, from `GUFO-PARITY.md`, re-verified to sum exactly):

| class | Δ (luce−gufo) |
|---|---:|
| Dense GEMM | +142.53 |
| Routed gate/up | +177.71 |
| Routed down | −249.82 |
| Expert sort/scatter/gather | +23.11 |
| GDN | +140.17 |
| Attention | +22.79 |
| hc_* | +117.52 |
| Norms/elementwise/quantize | +97.78 |
| Other | −15.58 |
| **Σ kernels** | **+456.22** |
| Idle (wall − Σkernels) | +201.03 |
| **Wall (profiled)** | **+657.23** |

Top-5 |Δ| sum to 675.71 ms; net class sum (including Routed down's −249.82
and the two small residuals) reconciles exactly to +456.22 ms kernel-time
delta, and +201.03 ms idle brings it to the full +657.23 ms profiled-wall
delta — matching the unprofiled text-parity wall gap (~490 ms) in order of
magnitude, per `GUFO-PARITY.md`.
