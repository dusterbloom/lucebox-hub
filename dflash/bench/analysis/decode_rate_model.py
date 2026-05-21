#!/usr/bin/env python3
"""
Theoretical decode tok/s model for PFlash compressed KV cache on RTX 3090.

Architecture: Qwen3-27B-Q4_K_M
  n_layer   = 64
  n_head    = 64   (Q heads)
  n_kv_head = 8    (GQA)
  head_dim  = 128

KV quantization: tq3_0
  block_tq3_0 = 14 bytes per 32 elements (struct from ggml-common.h)
  => 3.5 bits/value => 0.4375 bytes/value

Hardware: RTX 3090, 936 GB/s memory bandwidth
"""

import math

# ── Architecture ──────────────────────────────────────────────────────────────
N_LAYER   = 64
N_HEAD    = 64
N_KV_HEAD = 8
HEAD_DIM  = 128

# tq3_0: 14 bytes per 32 elements
TQ3_BLOCK_BYTES   = 14
TQ3_BLOCK_ELEMS   = 32
BYTES_PER_ELEM    = TQ3_BLOCK_BYTES / TQ3_BLOCK_ELEMS   # 0.4375

# RTX 3090 peak memory bandwidth (GB/s)
BW_GB_S = 936.0

# ── Context / compression sweep ───────────────────────────────────────────────
CTX_FULL_TOKENS = [4096, 8192, 16384, 32768, 65536, 131072]
KEEP_RATIOS     = [1.00, 0.20, 0.10, 0.05, 0.025]


def kv_cache_bytes_per_layer(ctx_eff: int) -> float:
    """Bytes for K+V for one layer at ctx_eff tokens (tq3_0)."""
    # K cache: ctx_eff tokens × n_kv_head heads × head_dim elements
    # V cache: same
    # Each element = BYTES_PER_ELEM bytes
    # 2 for K and V
    return 2 * ctx_eff * N_KV_HEAD * HEAD_DIM * BYTES_PER_ELEM


def total_kv_cache_bytes(ctx_eff: int) -> float:
    return N_LAYER * kv_cache_bytes_per_layer(ctx_eff)


def bytes_per_decode_step(ctx_eff: int) -> float:
    """
    Bandwidth consumed by the attention computation in one decode step.

    For GQA with n_head Q heads and n_kv_head KV heads:
    Each Q head attends over all ctx_eff KV positions.  Under GQA the KV
    heads are shared: n_head Q heads share n_kv_head KV heads, so each
    KV head is read by (n_head / n_kv_head) Q heads — but the KV data is
    only read ONCE per KV head per layer (broadcast/shared fetch on chip).

    Bandwidth = 2 (K+V) × ctx_eff × n_kv_head × head_dim × bytes_per_elem × n_layer
    This is the minimum (perfect cache sharing within a warp group). In
    practice the GPU may read it more than once, so this is a lower bound.
    """
    bytes_attn = 2 * ctx_eff * N_KV_HEAD * HEAD_DIM * BYTES_PER_ELEM * N_LAYER
    return bytes_attn


def decode_tok_s(ctx_eff: int) -> float:
    bw_bytes = BW_GB_S * 1e9
    step_bytes = bytes_per_decode_step(ctx_eff)
    return bw_bytes / step_bytes


# ── Tables ────────────────────────────────────────────────────────────────────

def kv_cache_mib_table():
    """(ctx_full × keep_ratio) -> KV cache size in MiB."""
    rows = []
    for ctx_full in CTX_FULL_TOKENS:
        row = {}
        for kr in KEEP_RATIOS:
            ctx_eff = max(1, round(ctx_full * kr))
            mib = total_kv_cache_bytes(ctx_eff) / (1024 ** 2)
            row[kr] = (ctx_eff, mib)
        rows.append((ctx_full, row))
    return rows


def decode_toks_table():
    """(ctx_full × keep_ratio) -> theoretical decode tok/s."""
    rows = []
    for ctx_full in CTX_FULL_TOKENS:
        row = {}
        for kr in KEEP_RATIOS:
            ctx_eff = max(1, round(ctx_full * kr))
            toks = decode_tok_s(ctx_eff)
            row[kr] = toks
        rows.append((ctx_full, row))
    return rows


def speedup_table():
    """keep_ratio vs 1.0 speedup at each ctx_full."""
    rows = []
    for ctx_full in CTX_FULL_TOKENS:
        baseline = decode_tok_s(ctx_full)  # keep=1.0
        row = {}
        for kr in KEEP_RATIOS:
            ctx_eff = max(1, round(ctx_full * kr))
            toks = decode_tok_s(ctx_eff)
            row[kr] = toks / baseline
        rows.append((ctx_full, row))
    return rows


def arithmetic_intensity_check(ctx_eff: int) -> float:
    """
    Flops per byte for the softmax attention step.
    Attention: O(n_head × ctx_eff × head_dim) FMAs = 2× flops.
    If intensity < RTX3090 peak FP16 TFLOPS / BW_GB_S, decode is BW-bound.
    """
    flops = 2 * N_HEAD * ctx_eff * HEAD_DIM * N_LAYER  # dot products only
    bw    = bytes_per_decode_step(ctx_eff)
    return flops / bw


# ── ASCII chart ───────────────────────────────────────────────────────────────

def ascii_chart(toks_table):
    """Two lines: keep=1.0 and keep=0.05."""
    print("\nDecode tok/s vs context length (theoretical, tq3_0 KV, RTX 3090)\n")
    print(f"{'ctx_full':>10}  {'keep=1.00':>12}  {'keep=0.05':>12}  {'speedup':>9}")
    print("-" * 52)
    for ctx_full, row in toks_table:
        t_full = row[1.00]
        t_comp = row[0.05]
        su     = t_comp / t_full
        print(f"{ctx_full:>10,}  {t_full:>12.1f}  {t_comp:>12.1f}  {su:>8.1f}x")
    print()


# ── SVG spark chart (two curves) ─────────────────────────────────────────────

def make_svg(toks_table) -> str:
    w, h = 600, 340
    pad_l, pad_r, pad_t, pad_b = 80, 30, 30, 60

    xs = [r[0] for r in toks_table]
    vals_full = [r[1][1.00] for r in toks_table]
    vals_comp = [r[1][0.05] for r in toks_table]
    all_vals  = vals_full + vals_comp

    y_max = max(all_vals) * 1.08
    y_min = 0.0

    def sx(idx):
        n = len(xs) - 1
        return pad_l + (w - pad_l - pad_r) * idx / n

    def sy(v):
        return pad_t + (h - pad_t - pad_b) * (1 - (v - y_min) / (y_max - y_min))

    pts_full = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals_full))
    pts_comp = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals_comp))

    x_labels = "\n  ".join(
        f'<text x="{sx(i):.1f}" y="{h - pad_b + 18}" text-anchor="middle" '
        f'font-size="11" fill="#555">{xs[i]//1024}K</text>'
        for i in range(len(xs))
    )

    # y-axis ticks
    y_ticks_vals = [int(y_max * t / 4) for t in range(5)]
    y_ticks = "\n  ".join(
        f'<text x="{pad_l - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" '
        f'font-size="11" fill="#555">{v:,}</text>'
        f'<line x1="{pad_l}" y1="{sy(v):.1f}" x2="{w - pad_r}" y2="{sy(v):.1f}" '
        f'stroke="#ddd" stroke-width="1"/>'
        for v in y_ticks_vals
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
  <rect width="{w}" height="{h}" fill="white"/>
  <!-- y-axis grid + labels -->
  {y_ticks}
  <!-- x-axis labels -->
  {x_labels}
  <!-- axis lines -->
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="#333" stroke-width="1.5"/>
  <line x1="{pad_l}" y1="{h-pad_b}" x2="{w-pad_r}" y2="{h-pad_b}" stroke="#333" stroke-width="1.5"/>
  <!-- keep=1.0 line (uncompressed) -->
  <polyline points="{pts_full}" fill="none" stroke="#e74c3c" stroke-width="2.5"/>
  <!-- keep=0.05 line (PFlash 5%) -->
  <polyline points="{pts_comp}" fill="none" stroke="#2ecc71" stroke-width="2.5"/>
  <!-- legend -->
  <rect x="{pad_l + 10}" y="{pad_t + 8}" width="14" height="4" fill="#e74c3c"/>
  <text x="{pad_l + 28}" y="{pad_t + 13}" font-size="12" fill="#333">keep=1.0 (uncompressed)</text>
  <rect x="{pad_l + 10}" y="{pad_t + 24}" width="14" height="4" fill="#2ecc71"/>
  <text x="{pad_l + 28}" y="{pad_t + 29}" font-size="12" fill="#333">keep=0.05 (PFlash 5%)</text>
  <!-- axis labels -->
  <text x="{(pad_l + w - pad_r) // 2}" y="{h - 5}" text-anchor="middle" font-size="12" fill="#333">Context length (tokens)</text>
  <text x="{14}" y="{(pad_t + h - pad_b) // 2}" text-anchor="middle" font-size="12" fill="#333"
        transform="rotate(-90 14 {(pad_t + h - pad_b) // 2})">Decode tok/s (theoretical)</text>
  <text x="{w // 2}" y="{pad_t - 10}" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">
    Theoretical decode speed — Qwen3-27B tq3_0 KV, RTX 3090
  </text>
</svg>"""
    return svg


# ── Main ──────────────────────────────────────────────────────────────────────

def fmt_ctx(n: int) -> str:
    return f"{n//1024}K" if n % 1024 == 0 else str(n)


def print_kv_table(rows):
    header = f"{'ctx_full':>8}" + "".join(f"  keep={kr:.3f} (ctx_eff / MiB)" for kr in KEEP_RATIOS)
    print(header)
    print("-" * (len(header) + 10))
    for ctx_full, row in rows:
        line = f"{fmt_ctx(ctx_full):>8}"
        for kr in KEEP_RATIOS:
            ctx_eff, mib = row[kr]
            line += f"  {fmt_ctx(ctx_eff):>5} / {mib:>7.1f} MiB"
        print(line)
    print()


def print_toks_table(rows):
    header = f"{'ctx_full':>8}" + "".join(f"  keep={kr:.3f}" for kr in KEEP_RATIOS)
    print(header)
    print("-" * (len(header)))
    for ctx_full, row in rows:
        line = f"{fmt_ctx(ctx_full):>8}"
        for kr in KEEP_RATIOS:
            line += f"  {row[kr]:>9.1f}"
        print(line)
    print()


def print_speedup_table(rows):
    header = f"{'ctx_full':>8}" + "".join(f"  keep={kr:.3f}" for kr in KEEP_RATIOS)
    print(header)
    print("-" * len(header))
    for ctx_full, row in rows:
        line = f"{fmt_ctx(ctx_full):>8}"
        for kr in KEEP_RATIOS:
            line += f"  {row[kr]:>9.2f}x"
        print(line)
    print()


if __name__ == "__main__":
    import os

    kv_rows    = kv_cache_mib_table()
    toks_rows  = decode_toks_table()
    su_rows    = speedup_table()

    print("=== KV cache size (MiB) — tq3_0 encoding, Qwen3-27B ===\n")
    print_kv_table(kv_rows)

    print("=== Theoretical decode tok/s (bandwidth-bound model, RTX 3090 936 GB/s) ===\n")
    print_toks_table(toks_rows)

    print("=== Predicted speedup vs keep=1.0 ===\n")
    print_speedup_table(su_rows)

    ascii_chart(toks_rows)

    # Headline numbers
    ctx_32k = 32768
    toks_full = decode_tok_s(ctx_32k)
    ctx_eff_5 = max(1, round(ctx_32k * 0.05))
    toks_comp = decode_tok_s(ctx_eff_5)
    speedup   = toks_comp / toks_full
    print(f"HEADLINE: At 32K context, keep=0.05 (ctx_eff={ctx_eff_5}) speedup: {speedup:.1f}×")
    print(f"  keep=1.0 → {toks_full:.1f} tok/s")
    print(f"  keep=0.05 → {toks_comp:.1f} tok/s")
    print()

    # Arithmetic intensity check at 32K
    ai = arithmetic_intensity_check(ctx_32k)
    # RTX 3090 FP16 tensor: ~35.6 TFLOPS, roofline ridge = 35600 GF / 936 GB/s = ~38 FLOP/byte
    ridge = 35600 / 936
    print(f"Arithmetic intensity at 32K ctx: {ai:.2f} FLOP/byte  (ridge ≈ {ridge:.0f} FLOP/byte)")
    print("=> Decode is BW-bound at all context lengths in this table.\n")

    # Write SVG
    svg = make_svg(toks_rows)
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    svg_path = os.path.join(out_dir, "decode_rate_chart.svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    print(f"SVG chart written to {svg_path}")
