import json, statistics as st
R="/mnt/kimi-k3/build/scorer-run"
def load(p): return [r for r in map(json.loads, open(p)) if "id" in r]
for n,p in (("prs6 ON (recorded)","/mnt/kimi-k3/split-work/rlru4/smoke/on/QK16KND.jsonl"),("ref (this binary)",R+"/s_k2_ref/QK16KND.jsonl"),("fast (this binary)",R+"/s_k2_new/QK16KND.jsonl")):
    rs=[r for r in load(p) if (r.get("prompt_tokens") or 0)>16384]
    f=lambda k:[r["timings"][k] for r in rs]
    print(n,len(rs),"reselect rows: prefill_ms med %.1f"%st.median(f("prefill_ms")),"decode tps med %.1f"%st.median(f("decode_tokens_per_sec")),"wall s med %.3f"%st.median([r["client_wall_s"] for r in rs]),
      "| warm rows only (drop first, cold): prefill %.1f wall %.3f"%(st.median(f("prefill_ms")[1:]),st.median([r["client_wall_s"] for r in rs][1:])))
