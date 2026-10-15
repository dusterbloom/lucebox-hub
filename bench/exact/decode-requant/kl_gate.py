#!/usr/bin/env python3
"""Quality gate for the decode-only requantized copies (LUCE_DECODE_REQ): next-token distribution of a candidate
arm vs the exact Q8_0 baseline over the same teacher-forced positions.

Input files are LUCE_TEACHER_LOGITS dumps: int32 n_vocab header, then per step [int32 forced token][n_vocab f16].
Both arms must have committed the identical forced stream (checked). Reports, over all positions:
KL(base || cand) mean / median / p99 / max (nats), top-1 agreement, top-5 overlap, and each arm's own
next-token accuracy / NLL on the forced text (both arms are teacher-forced on real text). Chunked: the dumps are
1.2 GB each at 2400 positions.
usage: kl_gate.py <base.bin> <cand.bin> [skip_first=1]"""
import sys
import numpy as np

def open_dump(path):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    n_vocab = int(np.asarray(raw[:4]).view(np.int32)[0])
    rec = 4 + 2 * n_vocab
    n = (len(raw) - 4) // rec
    body = raw[4:4 + n * rec].reshape(n, rec)
    toks = np.asarray(body[:, :4]).copy().view(np.int32).reshape(n)
    return toks, body, n_vocab

def logits(body, lo, hi, n_vocab):
    return np.asarray(body[lo:hi, 4:]).copy().view(np.float16).reshape(hi - lo, n_vocab).astype(np.float64)

def log_softmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))

base_t, base_b, nv = open_dump(sys.argv[1])
cand_t, cand_b, nv2 = open_dump(sys.argv[2])
assert nv == nv2
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 1   # step 0's logits come from the prefill path (Q8_0 on both arms)
n = min(len(base_t), len(cand_t))
assert n > skip, "empty dump"
assert (base_t[:n] == cand_t[:n]).all(), "forced token streams differ: not the same teacher-forced positions"
toks = base_t[:n]
kl, top1_b, top1_c, top5, nll_b, nll_c = [], [], [], [], [], []
CH = 128
for lo in range(skip, n, CH):
    hi = min(n, lo + CH)
    lp, lq = log_softmax(logits(base_b, lo, hi, nv)), log_softmax(logits(cand_b, lo, hi, nv))
    kl.append((np.exp(lp) * (lp - lq)).sum(axis=1))
    tb, tc = lp.argmax(axis=1), lq.argmax(axis=1)
    top1_b.append(tb); top1_c.append(tc)
    fb, fc = np.argpartition(-lp, 5, axis=1)[:, :5], np.argpartition(-lq, 5, axis=1)[:, :5]
    top5.append([len(set(a) & set(b)) / 5 for a, b in zip(fb, fc)])
    # record i holds the logits that predicted the forced token committed at step i (toks[i])
    nxt = toks[lo:hi]
    m = len(nxt)
    nll_b.append(-lp[np.arange(m), nxt]); nll_c.append(-lq[np.arange(m), nxt])
kl = np.concatenate(kl); top1_b = np.concatenate(top1_b); top1_c = np.concatenate(top1_c)
top5 = np.concatenate(top5); nll_b = np.concatenate(nll_b); nll_c = np.concatenate(nll_c)
nxt = toks[skip:n]
acc_b = (top1_b == nxt).mean(); acc_c = (top1_c == nxt).mean()
print(f"positions {len(kl)} (skip {skip})  n_vocab {nv}")
print(f"KL(base||cand) nats: mean {kl.mean():.5f}  median {np.median(kl):.5f}  p99 {np.percentile(kl, 99):.5f}  max {kl.max():.5f}")
print(f"top-1 agreement {np.mean(top1_b == top1_c) * 100:.2f}%  top-5 overlap {top5.mean() * 100:.2f}%")
print(f"next-token accuracy on the forced text: base {acc_b * 100:.2f}%  cand {acc_c * 100:.2f}%   NLL base {nll_b.mean():.4f}  cand {nll_c.mean():.4f}")
