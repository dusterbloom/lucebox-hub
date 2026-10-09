#!/usr/bin/env python3
"""Compute KL(P_ref || P_cand) mean/p99/max and top-1 agreement between two
golden-mode logit dumps (raw float32, n_vocab per step, n steps back-to-back,
as written by driver_shared_epilogue.cpp's MEASURE_LOGITS path).

usage: kl_gate.py <ref.modeN> <cand.modeN> <n_vocab> <n_steps> [--label L]
"""
import sys
import numpy as np


def load(path, n_vocab, n_steps):
    a = np.fromfile(path, dtype=np.float32)
    assert a.size == n_vocab * n_steps, f"{path}: {a.size} != {n_vocab}*{n_steps}"
    return a.reshape(n_steps, n_vocab)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return z - lse


def main():
    ref_path, cand_path, n_vocab, n_steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else f"{ref_path} vs {cand_path}"
    ref = load(ref_path, n_vocab, n_steps)
    cand = load(cand_path, n_vocab, n_steps)

    log_p = log_softmax(ref.astype(np.float64))
    log_q = log_softmax(cand.astype(np.float64))
    p = np.exp(log_p)
    kl = (p * (log_p - log_q)).sum(axis=-1)  # KL(P_ref || P_cand), per step

    top1_ref = ref.argmax(axis=-1)
    top1_cand = cand.argmax(axis=-1)
    agree = (top1_ref == top1_cand)

    print(f"[{label}] n_steps={n_steps} n_vocab={n_vocab}")
    print(f"  KL(ref||cand): mean={kl.mean():.6e} p99={np.percentile(kl, 99):.6e} max={kl.max():.6e} min={kl.min():.6e}")
    print(f"  top-1 agreement: {agree.sum()}/{n_steps} = {100.0*agree.mean():.2f}%")
    if not agree.all():
        bad = np.where(~agree)[0]
        print(f"  divergent positions: {bad.tolist()[:20]}{'...' if len(bad) > 20 else ''}")


if __name__ == "__main__":
    main()
