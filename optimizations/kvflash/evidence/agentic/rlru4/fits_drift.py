"""Post-hoc: (1) OFF is deterministic across runs and binaries; (2) ON probes that mode 4 skipped ("skip, fits", a strict no-op)
still answer differently from OFF in some cases, so state from earlier ON requests in the same server reaches later no-op requests.
    python3 -I fits_drift.py"""
import json

S = "/mnt/kimi-k3/split-work"


def probes(p):
    return {r["id"]: r.get("content") for r in map(json.loads, open(p)) if r.get("kind") == "probe"}


off4, off3, dump = probes(f"{S}/rlru4/off/QK16KND.jsonl"), probes(f"{S}/rlru/off/QK16KND.jsonl"), probes(f"{S}/rlru/dump/QK16KND.jsonl")
print(f"OFF prs6 (rlru4) vs OFF prs5 (rlru): {sum(off4[i] != off3[i] for i in off4)} of {len(off4)} answers differ; "
      f"OFF prs5 vs dump arm prs-cuda: {sum(off3[i] != dump[i] for i in off3)}")
on = [json.loads(l) for l in open(f"{S}/rlru4/on/QK16KND.jsonl")]
blocks = open(f"{S}/rlru4/on/QK16KND.server.log").read().split("] chat START ")[1:]
assert len(on) == len(blocks)
offr = {r["id"]: r for r in map(json.loads, open(f"{S}/rlru4/off/QK16KND.jsonl"))}
fits = [(j, r) for j, (r, b) in enumerate(zip(on, blocks)) if r.get("kind") == "probe" and "mode=4: skip, fits" in b]
diff = [(j, r) for j, r in fits if r.get("content") != offr[r["id"]].get("content")]
print(f"ON probes skipped as fits: {len(fits)}; answer differs from OFF: {len(diff)}")
for j, r in diff:
    t, u, pb = offr[r["id"]]["timings"], r["timings"], blocks[j - 1]
    print(f"  {r['id'][-30:]} pt {t['prefilled_tokens']}/{u['prefilled_tokens']} cached {t['cached_prefix_tokens']}/{u['cached_prefix_tokens']} "
          f"| previous request {on[j - 1].get('kind')} {'skip' if 'skip, fits' in pb else 'rewound' if 're-prefill' in pb else 'none'}")
