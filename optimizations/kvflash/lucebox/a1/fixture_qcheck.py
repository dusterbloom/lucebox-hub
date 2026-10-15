"""Do the fixture's captured queries (q, grouped head-major per kv head) reproduce the fixture's scores and idx?  python3 -I fixture_qcheck.py <case dir> [bf16]
Uses fitq (the queries the fit received) when the case has it; bf16 emulates the reference bf16 logits. Needs numpy only."""
import json
import sys

import numpy as np

d = sys.argv[1]
m = json.load(open(f"{d}/manifest.json"))
F = {(f["layer"], f["name"]): f for f in m["files"]}


def load(layer, name):
    f = F[(layer, name)]
    return np.fromfile(f"{d}/{f['file']}", dtype=np.int32 if f["dtype"] == "int32" else np.float32).reshape(f["shape_torch"])


def bf16(x):
    """Round float64 -> float32 -> bf16 (round to nearest even), back to float64."""
    b = x.astype(np.float32).view(np.uint32)
    b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return b.view(np.float32).astype(np.float64)


ROUND = len(sys.argv) > 2 and sys.argv[2] == "bf16"  # emulate bf16 logits: (q @ K.T) in bf16, then fp32, as torch does for bf16 inputs
H, n_kv, hd, slots = m["n_head"], m["n_kv"], m["head_dim"], m["slots"]
worst, allr, allo = (0, None), [], []
for L in m["layers"]:
    q, k, sc_ref, idx = load(L, "q").astype(np.float64), load(L, "k").astype(np.float64), load(L, "scores"), load(L, "idx")
    g = load(L, "fitq").astype(np.float64) if (L, "fitq") in F else q.reshape(n_kv, (H // n_kv) * q.shape[1], hd)
    for h in range(n_kv):
        s = g[h] @ k[h].T
        s = (bf16(s) if ROUND else s) / np.sqrt(hd)
        w = np.exp(s - s.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        sc = np.sqrt((w ** 2).mean(0))
        rel = np.abs(sc - sc_ref[h]).max() / np.abs(sc_ref[h]).max()
        ov = len(set(np.argsort(-sc, kind="stable")[:slots].tolist()) & set(idx[h].tolist())) / slots
        allr.append(rel)
        allo.append(ov)
        if rel > worst[0]:
            worst = (rel, (L, h))
        if ov < 1 or rel > 1e-2:
            print(f"L{L} kv{h}: normalized score error {rel:.2e}, top-{slots} overlap with idx {ov:.3f}")
print("fitq used:", any(nm == "fitq" for _, nm in F), "checks:", m.get("checks"))
print(f"normalized score error over {len(allr)} heads: median {np.median(allr):.2e} p90 {np.percentile(allr, 90):.2e}; "
      f"top-{slots} overlap with idx: min {min(allo):.3f} mean {np.mean(allo):.4f}, heads with full overlap {sum(o == 1 for o in allo)}")
print("worst normalized score error", f"{worst[0]:.2e}", "at (layer, kv head)", worst[1])
