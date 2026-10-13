"""Aggregate sharded AIME result JSONs: python aggregate.py <dir with run subdirs> (e.g. _dl/qwen3-4b)."""
import glob, json, re, sys, collections

root = sys.argv[1]
runs = collections.defaultdict(list)
for f in glob.glob(f"{root}/*/*.json"):
    if f.endswith("manifest.json"):
        continue
    name = f.split("/")[-2]
    base = re.sub(r"_p\d+$", "", name)
    d = json.load(open(f))
    if "results" in d and d["results"] and "is_correct" in d["results"][0]:
        runs[base] += d["results"]
print(f"{'run':36s} n  acc   avg_tok  stopped_nat  avg_gen_s  avg_comp_s  GPU-h(gen+comp)")
for k, rs in sorted(runs.items()):
    n = len(rs)
    acc = sum(r["is_correct"] for r in rs) / n
    tok = sum(r.get("total_reasoning_tokens", r.get("reasoning_tokens", 0)) for r in rs) / n
    nat = sum(bool(r["stopped_naturally"]) for r in rs)
    gen = sum(r["generation_time"] for r in rs)
    comp = sum(r.get("total_compaction_time", 0) for r in rs)
    print(f"{k:36s} {n:2d} {acc:.3f} {tok:8.0f} {nat:5d} {gen/n:10.0f} {comp/n:8.1f} {(gen+comp)/3600:8.2f}")
