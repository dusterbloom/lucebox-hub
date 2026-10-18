"""What the GPU scorer would do to the E2E v2 arm, per request: prefill - logged CPU scorer ms + bench GPU ms at that
request's sealed count (q4_0, 64 queries, linear between the measured points; fewer queries get the 64-query cost).
Medians over the 951 probes, against on6ref. Stdlib only.   python3 -I predict_gpu.py <e2e dir>"""
import json
import os
import re
import statistics as st
import sys

GPU = [(514, 96.7), (576, 113.2), (816, 167.6)]   # bench.out, q4_0 N 64 mmq: sealed chunks -> total ms (median)
SCORER = re.compile(r"prefill reselect mode=4 scorer=exact .* scorer ([\d.]+) ms")
SPLIT = re.compile(r"exact-scorer fast split: n_q=(\d+) sealed=(\d+)")


def gpu_ms(s):
    (a, ta), (b, tb) = (GPU[0], GPU[1]) if s <= GPU[1][0] else (GPU[1], GPU[2])
    return max(0.0, ta + (tb - ta) * (s - a) / (b - a))


def arm(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "QK16KND.jsonl"))]
    blocks = open(os.path.join(d, "QK16KND.server.log")).read().split("] chat START ")[1:]
    assert len(rows) == len(blocks)
    return [(r, b) for r, b in zip(rows, blocks) if r.get("kind", "probe") == "probe" and "error" not in r]


def main(e2e):
    out = {}
    for name in ("on6ref", "v2"):
        pre, wall, pre_g, wall_g, cpu, gpu = [], [], [], [], [], []
        for r, b in arm(os.path.join(e2e, name)):
            p, w = r["timings"]["prefill_ms"], r["client_wall_s"]
            sc, sp = SCORER.findall(b), SPLIT.findall(b)
            assert len(sc) == len(sp), (r["id"], len(sc), len(sp))
            delta = 0.0
            for ms, (_, sealed) in zip(sc, sp):
                g = gpu_ms(int(sealed)) if name == "v2" else float(ms)   # on6ref keeps its 1-query CPU scorer
                cpu.append(float(ms))
                gpu.append(g)
                delta += float(ms) - g
            pre.append(p); wall.append(w); pre_g.append(p - delta); wall_g.append(w - delta / 1000)
        out[name] = {"probes": len(pre), "prefill_ms": st.median(pre), "client_wall_s": st.median(wall),
                     "scorer_ms_cpu": st.median(cpu), "scorer_ms_gpu_model": st.median(gpu),
                     "scorer_ms_gpu_model_max": max(gpu),
                     "prefill_ms_with_gpu_scorer": st.median(pre_g), "client_wall_s_with_gpu_scorer": st.median(wall_g)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
