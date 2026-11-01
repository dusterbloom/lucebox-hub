"""Reader for ds41_bridge_probe top-k dumps (PROBE_DUMP_TOPK), numpy only.

Layout (little-endian): "DS41TOPK", u32 version=1, u32 K, u32 N, u32 vocab,
i32 L, i32 R, char arm, 3 pad (36 B); then ids int32[K,N]; then logprobs float32[K,N].
"""
import struct
import sys
import tempfile

import numpy as np

_HDR = struct.Struct("<8s4I2i4s")


def read(path):
    with open(path, "rb") as f:
        magic, ver, k, n, vocab, L, R, tail = _HDR.unpack(f.read(_HDR.size))
        if magic != b"DS41TOPK" or ver != 1:
            raise ValueError(f"{path}: bad magic/version {magic!r} {ver}")
        ids = np.fromfile(f, "<i4", k * n).reshape(k, n)
        lp = np.fromfile(f, "<f4", k * n).reshape(k, n)
    return dict(version=ver, K=k, N=n, vocab=vocab, L=L, R=R, arm=chr(tail[0]), ids=ids, logprobs=lp)


def kl_topk(teacher_ids, teacher_lp, student_logprobs_full):
    """Per-position KL(teacher || student) restricted to the teacher's top-N, as the
    probe's KL (sum_v p_t (lp_t - lp_s), p_t = exp(lp_t) over the full vocab) with the
    sum truncated to those N tokens. Not renormalized, so it lower-bounds the full KL
    when every dropped term is nonnegative and converges to it as N -> vocab.
    student_logprobs_full: [K, vocab] log-softmax (any float dtype). Returns float64 [K]."""
    ids = np.asarray(teacher_ids)
    lp_t = np.asarray(teacher_lp, dtype=np.float64)
    lp_s = np.take_along_axis(np.asarray(student_logprobs_full, dtype=np.float64), ids, axis=1)
    return (np.exp(lp_t) * (lp_t - lp_s)).sum(axis=1)


def write(path, ids, lp, L, R, arm, vocab):
    k, n = ids.shape
    with open(path, "wb") as f:
        f.write(_HDR.pack(b"DS41TOPK", 1, k, n, vocab, L, R, arm.encode() + b"\0\0\0"))
        f.write(np.ascontiguousarray(ids, "<i4").tobytes())
        f.write(np.ascontiguousarray(lp, "<f4").tobytes())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K, V, N = 5, 50, 8
    logits = rng.normal(size=(K, V))
    lps = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    ids = np.argsort(-lps, axis=1, kind="stable")[:, :N].astype(np.int32)
    lp = np.take_along_axis(lps, ids, axis=1).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        write(t.name, ids, lp, 2048, 128, "B", V)
        d = read(t.name)
    assert (d["K"], d["N"], d["vocab"], d["L"], d["R"], d["arm"]) == (K, N, V, 2048, 128, "B")
    assert (d["ids"] == ids).all() and (d["logprobs"] == lp).all()
    assert np.allclose(kl_topk(d["ids"], d["logprobs"], lps), 0.0, atol=1e-6)
    full = kl_topk(np.tile(np.arange(V), (K, 1)), lps, lps - 0.1)
    assert np.allclose(full, np.exp(lps).sum(1) * 0.1)
    print("ds41_topk_read self-check OK", file=sys.stderr)
