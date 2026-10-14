"""Per-turn tables for the arms of one run directory.   python3 -I make_table.py RUNDIR [ARM ...] > tables.md

Reads RUNDIR/turns_<ARM>.json (written by pf746_gap.py) and the per-turn server log slices, prints one markdown table
per arm plus the answer comparison across arms (byte-identical or not).
"""
import json, re, sys

run = sys.argv[1]
arms = sys.argv[2:] or ["A", "B", "C"]
ADM = re.compile(r"\[pflash-admission\] (\w+) cached=(\d+) cached_droppable~(\d+) uncached_droppable~(\d+) droppable~(\d+)")
LIVE = re.compile(r"\[kvflash\] live restore: L=(\d+) C=(-?\d+) replay=(\d+)")
RESUME = re.compile(r"\[kvflash\] resume: (.*)")
DONE = re.compile(r"chat DONE .* in=(\d+) effective_in=(\d+) out=(\d+) .* restore=(\w+) slot=(-?\d+) prefix_len=(\d+) "
                  r"prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)")
SKIP = re.compile(r"snapshot skipped")


def load(arm):
    return json.load(open(f"{run}/turns_{arm}.json"))


def rows(arm):
    d = load(arm)
    out = []
    for t in d["turns"]:
        lines = t["log_lines"]
        tm = t["timings"]
        pf = tm.get("pflash") or {}
        adm = pf.get("admission", {})
        m = next((ADM.search(l) for l in lines if ADM.search(l)), None)
        live = next((LIVE.search(l) for l in lines if LIVE.search(l)), None)
        resume = next((RESUME.search(l) for l in lines if RESUME.search(l)), None)
        done = next((DONE.search(l) for l in lines if DONE.search(l)), None)
        skipped = any(SKIP.search(l) for l in lines)
        compressed = "compressed_tokens" in pf
        decision = adm.get("decision") or ("-" if not pf else "?")
        out.append(dict(
            turn=t["turn"], prompt=t["prompt_tokens"], decision=decision,
            served=tm.get("effective_prompt_tokens"), compressed=compressed,
            rest_cached=adm.get("cached_tokens"), unc_drop=adm.get("uncached_droppable"),
            cached_drop=adm.get("cached_droppable"), droppable=adm.get("droppable"),
            prefix_len=int(done.group(6)) if done else None,
            live=f"L={live.group(1)} C={live.group(2)} replay={live.group(3)}" if live else "-",
            resume=resume.group(1) if resume else "-",
            cached_prefix=tm.get("cached_prefix_tokens"), prefilled=tm.get("prefilled_tokens"),
            prefill_s=tm.get("prefill_ms", 0) / 1000.0, tps=tm.get("decode_tokens_per_sec"),
            compress_s=pf.get("compress_ms", 0) / 1000.0,
            wall=t["wall_s"], answer=t["answer"], skipped=skipped, ct=t["completion_tokens"]))
    return out


def fmt(x, f="{}"):
    return "-" if x is None else f.format(x)


def table(arm):
    r = rows(arm)
    h = ("| turn | prompt tok | admission | served tok | restorable (admission) | uncached droppable | prefix_len | "
         "live restore | cached_prefix | prefilled | prefill s | prefill tok/s | compress s | decode tok/s | wall s |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for x in r:
        h += (f"| {x['turn']} | {x['prompt']} | {x['decision']}{' (COMPRESSED)' if x['compressed'] else ''} | "
              f"{fmt(x['served'])} | {fmt(x['rest_cached'])} | {fmt(x['unc_drop'])} | {fmt(x['prefix_len'])} | "
              f"{x['live']} | {fmt(x['cached_prefix'])} | {fmt(x['prefilled'])} | {x['prefill_s']:.1f} | "
              f"{(x['prefilled'] or 0) / x['prefill_s'] if x['prefill_s'] > 0 else 0:.0f} | "
              f"{x['compress_s']:.1f} | {fmt(x['tps'])} | {x['wall']:.1f} |\n")
    return h


print(f"# run {run}\n")
for a in arms:
    print(f"## arm {a}\n\n{table(a)}")
    for x in rows(a):
        if x["resume"] != "-":
            print(f"- turn {x['turn']} resume: {x['resume']}")
    print()
    for x in rows(a):
        print(f"- turn {x['turn']} answer ({x['ct']} tok): {json.dumps(x['answer'])}")
    print()
if len(arms) > 1:
    base = {a: [x["answer"] for x in rows(a)] for a in arms}
    print("## answer comparison\n")
    for a, b in ((x, y) for i, x in enumerate(arms) for y in arms[i + 1:]):
        same = [ra == rb for ra, rb in zip(base[a], base[b])]
        print(f"- {a} vs {b}: " + ", ".join(f"t{i + 1}={'identical' if s else 'DIFFERENT'}" for i, s in enumerate(same)))
