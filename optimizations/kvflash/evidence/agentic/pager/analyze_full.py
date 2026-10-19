"""Pager evidence: packed vs legacy copy path, 951-probe QK16KND harness, same binary.
    python3 -I analyze_full.py <full dir with packed/ and legacy/> <on6ref dir>"""
import json
import re
import statistics as st
import sys

FULL, REF = sys.argv[1], sys.argv[2]
PH = ["reset", "snap", "first", "qread", "score", "rewind", "page_submit", "page_wait", "replay", "total", "pre", "apply", "resid"]


def rows(d):
    return [json.loads(l) for l in open(d + "/QK16KND.jsonl")]


def med(v):
    return st.median(v) if v else float("nan")


def key(r):
    return (r["id"], r["content"], r["reasoning"], json.dumps(r["tool_calls"], sort_keys=True), r["finish"], r["completion_tokens"])


def pf(r):
    return r["timings"]["prefill_ms"]


ref = rows(REF)
refd = {r["id"]: r for r in ref}
out = {}
for arm in ["packed", "legacy"]:
    d = f"{FULL}/{arm}"
    rs = rows(d)
    log = open(d + "/QK16KND.server.log").read()
    probes = [r for r in rs if r["kind"] == "probe"]
    same = sum(key(r) == key(refd[r["id"]]) for r in rs if r["id"] in refd)
    same_p = sum(key(r) == key(refd[r["id"]]) for r in probes)
    ph = [dict((k, float(v)) for k, v in re.findall(r"(\w+)=([\d.\-]+)", m.group(1) + " " + m.group(2)))
          for m in re.finditer(r"mode4 phases ms: (.*?) \((.*?)\)", log)]
    cl = [dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", m.group(1))) for m in re.finditer(r"mode4 calls: (.*)", log)]
    pls = [int(x) for x in re.findall(r"chat DONE \S+ .*? restore=\S+ slot=\S+ prefix_len=(\d+) ", log)]
    dh = re.findall(r"disk_hit=(\w+)", log)
    out[arm] = dict(rs=rs, probes=probes)
    print(f"## {arm}")
    print(f"answers identical to on6ref: probes {same_p}/{len(probes)}, all rows {same}/{len(rs)}")
    print(f"prefilled_tokens med {med([r['timings']['prefilled_tokens'] for r in probes])}, "
          f"cached_prefix_tokens med {med([r['timings']['cached_prefix_tokens'] for r in probes])}, "
          f"prefix_len (server log) med {med(pls)}; disk_hit: {sorted(set(dh))}")
    print(f"prefill ms med {med([pf(r) for r in probes]):.1f}, decode tok/s med "
          f"{med([r['timings']['decode_tokens_per_sec'] for r in probes]):.1f}, "
          f"client wall ms med {med([r['client_wall_s'] * 1000 for r in probes]):.0f}")
    print(f"reselects {len(ph)}; phase medians ms: " + " ".join(f"{k}={med([x[k] for x in ph]):.1f}" for k in PH))
    print("call medians: " + " ".join(f"{k}={med([x[k] for x in cl]):.0f}" for k in cl[0]))
a = {r["id"]: r for r in out["packed"]["probes"]}
b = {r["id"]: r for r in out["legacy"]["probes"]}
ids = [i for i in a if i in b]
dp = [pf(a[i]) - pf(b[i]) for i in ids]
dw = [(a[i]["client_wall_s"] - b[i]["client_wall_s"]) * 1000 for i in ids]
print(f"## paired packed - legacy over {len(ids)} probes: prefill ms med {med(dp):.1f}, client wall ms med {med(dw):.1f}")
