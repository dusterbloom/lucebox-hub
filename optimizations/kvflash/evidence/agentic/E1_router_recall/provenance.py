"""E1 provenance: caching state (pt / prefix_len) of the E1 probes against B0 POOL16K, and the speeds, from the rows.
Usage: python3 -I provenance.py <E1 rows.jsonl> <B0 POOL16K.jsonl> <server.log> <out.json>"""
import json
import re
import statistics as st
import sys

e1_p, b0_p, log_p, out_p = sys.argv[1:5]
e1 = [json.loads(l) for l in open(e1_p)]
b0 = {r["id"]: r for r in map(json.loads, open(b0_p))}


def stats(rows, key):
    xs = [key(r) for r in rows]
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "median": st.median(xs), "mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs)}


probes = [r for r in e1 if r.get("kind") == "probe" and "error" not in r]
pr = lambda r: (r.get("timings") or {}).get("prefilled_tokens")
out = {
    "n_probe_rows": len(probes),
    "n_primer_rows": sum(1 for r in e1 if r.get("kind") == "primer"),
    "errors": sum(1 for r in e1 if "error" in r),
    "E1_probe_prompt_tokens": stats(probes, lambda r: r["prompt_tokens"]),
    "E1_probe_cached_tokens": stats(probes, lambda r: r["cached_tokens"]),
    "E1_probe_prefilled_tokens": stats(probes, pr),
    "E1_probe_cache_hit_fraction": sum(r["cached_tokens"] for r in probes) / sum(r["prompt_tokens"] for r in probes),
}
bp = [b0[r["id"]] for r in probes if r["id"] in b0]
out["B0_POOL16K_same_probes_prefilled_tokens"] = stats(bp, pr)
out["B0_POOL16K_same_probes_cache_hit_fraction"] = sum(r["cached_tokens"] for r in bp) / sum(r["prompt_tokens"] for r in bp)
tm = lambda k: (lambda r: (r.get("timings") or {}).get(k))
for name, rows in (("E1", probes), ("B0_POOL16K", bp)):
    out[name + "_probe_prefill_ms"] = stats(rows, tm("prefill_ms"))
    out[name + "_probe_decode_tok_s"] = stats(rows, tm("decode_tokens_per_sec"))
    out[name + "_probe_client_wall_s"] = stats(rows, lambda r: r.get("client_wall_s"))
log = open(log_p, errors="replace").read()
out["server_log"] = {
    "chat_requests": len(re.findall(r"\[server\] chat CACHE", log)),
    "disk_hit_false": len(re.findall(r"disk_hit=false", log)),
    "disk_hit_true": len(re.findall(r"disk_hit=true", log)),
    "kv_cache_dir_flag": "--kv-cache-dir" in log.split("\n", 1)[0],
    "server_starts": log.count("### start"),
    "live_restore_lines": len(re.findall(r"\[kvflash\] live restore:", log)),
    "cold_prefills": len(re.findall(r"live restore: L=0 C=-1", log)),
}
sample = re.findall(r"\[server\] chat DONE (\S+) ok=\S+ in=(\d+) effective_in=\d+ out=(\d+) [^\n]*?prefix_len=(\d+) prefill=([\d.]+)s", log)
out["server_log_sample_pt_prefix_len"] = [{"req": a, "prompt_tokens": int(b), "out": int(c), "prefix_len": int(d), "prefill_s": float(e)}
                                         for a, b, c, d, e in sample[:3] + sample[500:503] + sample[-3:]]
json.dump(out, open(out_p, "w"), indent=1)
print(json.dumps(out, indent=1))
