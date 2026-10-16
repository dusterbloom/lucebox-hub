import json, re, statistics as st, sys, glob, os
R = "/mnt/kimi-k3/build/scorer-run"

def shadow(d):
    log = open(f"{R}/{d}/QK16KND.server.log").read()
    rows = re.findall(r"scorer shadow: ref=\[(.*?)\] fast=\[(.*?)\] (SAME|DIFFERENT) max\|dscore\| (\S+) \(rel (\S+)\) k-th/next block gap rel (\S+); ref (\S+) ms fast (\S+) ms", log)
    return rows

def med(v): return st.median(v) if v else float("nan")
def p90(v):
    v = sorted(v); return v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))] if v else float("nan")

tot_ref, tot_fast = [], []
for d in ["s_k2_shadow", "s_k10_shadow", "d_k2_shadow", "d_k10_shadow"]:
    rows = shadow(d)
    same = sum(r[2] == "SAME" for r in rows)
    ref = [float(r[6]) for r in rows]; fast = [float(r[7]) for r in rows]
    tot_ref += ref; tot_fast += fast
    print(f"{d}: {len(rows)} reselects, SAME {same}, DIFFERENT {len(rows)-same}; max|dscore| {max(float(r[3]) for r in rows):.3g}, max rel {max(float(r[4]) for r in rows):.3g}, "
          f"min k-th/next gap rel {min(float(r[5]) for r in rows):.3g}; ref ms med {med(ref):.1f} p90 {p90(ref):.1f}; fast ms med {med(fast):.1f} p90 {p90(fast):.1f}")
    for r in rows:
        if r[2] != "SAME": print("   DIFF", r)
print(f"ALL shadow: {len(tot_ref)} reselects  ref med {med(tot_ref):.1f} p90 {p90(tot_ref):.1f} | fast med {med(tot_fast):.1f} p90 {p90(tot_fast):.1f}")

def scorer_ms(d):
    log = open(f"{R}/{d}/QK16KND.server.log").read()
    return [float(x) for x in re.findall(r"picked=\[.*?\].*? scorer ([\d.]+) ms", log)]

def rows(d): return {r["id"]: r for r in map(json.loads, open(f"{R}/{d}/QK16KND.jsonl")) if "id" in r}
def chat(d):
    log = open(f"{R}/{d}/QK16KND.server.log").read()
    return re.findall(r"chat CACHE (\S+) restore=(\S+) slot=(\S+) prefix_len=(\d+) effective_prompt=(\d+) .*?disk_hit=(\S+)", log)

for d in ["s_k2_new", "s_k2_ref"]:
    rr = rows(d); t = [r["timings"] for r in rr.values() if r.get("timings")]
    print(d, "scorer ms", scorer_ms(d), "med", med(scorer_ms(d)))
    for k in ("prefill_ms", "decode_tokens_per_sec", "decode_ms"):
        v = [x[k] for x in t if k in x]; print("  ", k, "median", med(v), "values", v)
    wall = [r["client_wall_s"] for r in rr.values()]; print("   client wall s median", med(wall), wall)
    print("   pt/prefix_len/disk_hit:", [(c[3], c[4], c[5]) for c in chat(d)])

# byte identity vs prs6 ON answers
prs6 = {r["id"]: r for r in map(json.loads, open("/mnt/kimi-k3/split-work/rlru4/smoke/on/QK16KND.jsonl")) if "id" in r}
new = rows("s_k2_new"); ref = rows("s_k2_ref")
for name, o in (("new", new), ("ref", ref)):
    ident = [i for i in prs6 if i in o and prs6[i].get("content") == o[i].get("content") and prs6[i].get("tool_calls") == o[i].get("tool_calls")]
    print(f"{name} vs prs6 ON: {len(ident)}/{len(prs6)} byte-identical (content+tool_calls); differing: {[i for i in prs6 if i not in ident]}")
# prs6 recorded numbers
t = [r["timings"] for r in prs6.values() if r.get("timings")]
print("prs6 ON recorded: prefill_ms med", med([x["prefill_ms"] for x in t]), "decode tps med", med([x["decode_tokens_per_sec"] for x in t]),
      "wall med", med([r["client_wall_s"] for r in prs6.values()]), "scorer ms", scorer_ms.__call__ and None)
log = open("/mnt/kimi-k3/split-work/rlru4/smoke/on/QK16KND.server.log").read()
sm = [float(x) for x in re.findall(r"picked=\[.*?\].*? scorer ([\d.]+) ms", log)]
print("prs6 ON scorer ms", sm, "med", med(sm))
