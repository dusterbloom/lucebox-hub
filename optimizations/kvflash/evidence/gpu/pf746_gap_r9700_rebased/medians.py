"""python3 -I medians.py RUNDIR   median (and mean/min/max) of prefill tok/s, decode tok/s, wall over the 5 turns, per arm; and byte-identity B vs C"""
import json, statistics as st, sys
run = sys.argv[1]
for arm in "ABC":
    d = json.load(open(f"{run}/turns_{arm}.json"))["turns"]
    pf = [t["timings"]["prefilled_tokens"] / (t["timings"]["prefill_ms"] / 1000) for t in d]
    dec = [t["timings"]["decode_tokens_per_sec"] for t in d]
    wall = [t["wall_s"] for t in d]
    for n, v in (("prefill tok/s", pf), ("decode tok/s", dec), ("wall s", wall)):
        print(f"arm {arm} {n}: median {st.median(v):.1f} mean {st.mean(v):.1f} min {min(v):.1f} max {max(v):.1f}")
b = json.load(open(f"{run}/turns_B.json"))["turns"]; c = json.load(open(f"{run}/turns_C.json"))["turns"]
print("B vs C byte-identical per turn:", [x["answer"].encode() == y["answer"].encode() for x, y in zip(b, c)])
