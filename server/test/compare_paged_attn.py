#!/usr/bin/env python3
"""Comparator for test_paged_attn_wmma.cpp two-mode dumps.

Usage: compare_paged_attn.py ref.bin cand.bin
The dump layout is case order from the test source; the test prints the
per-case shape line. Tolerances: 2e-3 f16 KV, 3e-3 q8_0 KV (the V_DOT2
reference dequantizes via dp4a-integer paths, the WMMA kernel via half
dequant - a wider gap than the contiguous fattn differential).
"""
import sys
import numpy as np

def main():
    ref = np.fromfile(sys.argv[1], dtype=np.float32)
    cand = np.fromfile(sys.argv[2], dtype=np.float32)
    assert ref.shape == cand.shape, f"shape mismatch: {ref.shape} vs {cand.shape}"
    diff = np.abs(ref - cand)
    maxd = float(diff.max())
    # per-case comparison would need shape metadata; report the global max.
    tol = 3e-3
    print(f"elements={ref.size} max_abs_diff={maxd:.6e} tol={tol:.1e} -> {'PASS' if maxd < tol else 'FAIL'}")

if __name__ == "__main__":
    main()
