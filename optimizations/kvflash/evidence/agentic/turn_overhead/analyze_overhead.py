#!/usr/bin/env python3
"""Join KVFlash jsonl rows with server-log request blocks and attribute non-prefill/decode wall time.
usage: python3 -I analyze_overhead.py [arm_base_path ...]   (base = path without .jsonl)"""
import json, re, sys, statistics as st
W = "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/evidence/"
DEF = [W + "agentic/rlru4/off/QK16KND"]


def med(x): return st.median(x) if x else float("nan")


def pct(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p / 100 * len(x)))] if x else float("nan")


def blocks(log):
    cur = None
    out = []
    for ln in open(log, errors="replace"):
        if ln.startswith("[server] request path"):
            cur = [ln.rstrip("\n")]   # keep the request line: it carries body_bytes
            out.append(cur)
        elif cur is not None:
            cur.append(ln.rstrip("\n"))
    return out


def feat(b):
    f = dict(ckpts=0, held=0, held_mib=0.0, skipped_bnd=0, page=0, other=[], L=-1, C=-1, replay=-1)
    for ln in b:
        m = re.search(r"body_bytes=(\d+)", ln)
        if m: f["body"] = int(m[1])
        m = re.search(r"chat CACHE \S+ restore=(\w+) slot=(-?\d+) prefix_len=(\d+) effective_prompt=(\d+).*disk_hit=(\w+) snap_slot=(-?\d+) snap_pos=(\d+) full_snap_slot=(-?\d+)", ln)
        if m:
            f.update(restore=m[1] == "true", pc_slot=int(m[2]), prefix_len=int(m[3]), eff=int(m[4]),
                     disk_hit=m[5] == "true", snap_slot=int(m[6]), snap_pos=int(m[7]), full_snap_slot=int(m[8]))
        m = re.search(r"live restore: L=(-?\d+) C=(-?\d+) replay=(\d+)", ln)
        if m:
            f.update(L=int(m[1]), C=int(m[2]), replay=int(m[3]))
        if "state ladder: ckpt" in ln:
            f["ckpts"] += 1
            m = re.search(r"(\d+) held, ([\d.]+) MiB", ln)
            f["held"] = int(m[1]); f["held_mib"] = float(m[2])
        if "boundary snapshot skipped" in ln:
            f["skipped_bnd"] += 1
        if re.search(r"page|evict|spill|insert", ln, re.I) and "pooled prefill" not in ln and "[server] chat" not in ln:
            f["page"] += 1; f["other"].append(ln[:120])
        m = re.search(r"chat DONE .* (\d+\.\d+)s [\d.]+ tok/s finish=(\S+) .*prefill=([\d.]+)s decode=([\d.]+)s", ln)
        if m:
            f.update(srv_total=float(m[1]), finish_log=m[2], srv_pf=float(m[3]), srv_dec=float(m[4]))
    return f


def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** .5
    sy = sum((y - my) ** 2 for y in ys) ** .5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx * sy else float("nan")


def run(base):
    rows = [json.loads(l) for l in open(base + ".jsonl")]
    bl = blocks(base + ".server.log")
    print(f"\n===== {base}\nrows={len(rows)} log_blocks={len(bl)}")
    fs = [feat(b) for b in bl]
    n = min(len(rows), len(fs))
    bad = sum(1 for r, f in zip(rows, fs) if f.get("eff") != r["prompt_tokens"])
    print(f"join mismatches (effective_prompt != prompt_tokens): {bad}/{n}")
    prev_traj = None
    D = []
    for r, f in zip(rows, fs):
        t = r["timings"]
        ov = r["client_wall_s"] - (t["prefill_ms"] + t["decode_ms"]) / 1000
        sw = r["traj"] != prev_traj
        prev_traj = r["traj"]
        d = dict(ov=ov, switch=sw, cached=r["cached_tokens"], pfl=t["prefilled_tokens"], comp=r["completion_tokens"],
                 kind=r["kind"], wall=r["client_wall_s"], pf_ms=t["prefill_ms"], dec_ms=t["decode_ms"], **f)
        D.append(d)
    ovs = [d["ov"] for d in D]
    print("overhead s: p50 %.2f p90 %.2f p95 %.2f p99 %.2f max %.2f  >1s: %d" % (
        med(ovs), pct(ovs, 90), pct(ovs, 95), pct(ovs, 99), max(ovs), sum(o > 1 for o in ovs)))
    print("kinds:", {k: sum(d['kind'] == k for d in D) for k in set(d['kind'] for d in D)})
    gap = [d["wall"] - d["srv_total"] for d in D if "srv_total" in d]
    print("client_wall - server 'DONE' total (0.1s resolution): p50 %.2f p90 %.2f max %.2f" % (med(gap), pct(gap, 90), max(gap)))
    big = [d["srv_total"] - d["srv_pf"] - d["srv_dec"] for d in D if "srv_total" in d and d["ov"] > 1]
    print(">1s requests: server DONE total - prefill - decode (s): median %.2f min %.2f" % (med(big), min(big) if big else float('nan')))
    small = [d["srv_total"] - d["srv_pf"] - d["srv_dec"] for d in D if "srv_total" in d and d["ov"] <= 1]
    print("<=1s requests: same quantity: median %.2f max %.2f" % (med(small), max(small) if small else float('nan')))

    def row(name, sel):
        s = [d for d in D if sel(d)]
        o = [d["ov"] for d in s]
        print(f"  {name:40s} n={len(s):5d} med={med(o):6.2f} p90={pct(o, 90):5.2f} >1s={sum(x > 1 for x in o):4d} ({100 * sum(x > 1 for x in o) / max(1, len(s)):5.1f}%)")

    print("\nAttribution (overhead = wall - prefill - decode, seconds)")
    row("ALL", lambda d: True)
    row("traj switch", lambda d: d["switch"]); row("no traj switch", lambda d: not d["switch"])
    row("kind=primer", lambda d: d["kind"] == "primer"); row("kind=probe", lambda d: d["kind"] == "probe")
    row("live restore replay==0", lambda d: d["replay"] == 0)
    row("live restore replay>0", lambda d: d["replay"] > 0)
    row("live restore L=0 (cold)", lambda d: d["L"] == 0)
    row("live restore L>0 (warm)", lambda d: d["L"] > 0)
    for k in range(0, 8):
        row(f"ladder ckpts taken = {k}", lambda d, k=k: d["ckpts"] == k)
    row("ladder ckpts taken >= 8", lambda d: d["ckpts"] >= 8)
    row("page/evict/insert log line", lambda d: d["page"] > 0)
    row("boundary snapshot skipped line", lambda d: d["skipped_bnd"] > 0)
    row("pool overflow (prompt > 16384)", lambda d: d["eff"] > 16384)
    row("prompt <= 16384", lambda d: d["eff"] <= 16384)
    for lo, hi in [(0, 1), (1, 8192), (8192, 16384), (16384, 24576), (24576, 32768), (32768, 40000), (40000, 10 ** 9)]:
        row(f"cached_tokens [{lo},{hi})", lambda d, lo=lo, hi=hi: lo <= d["cached"] < hi)
    for lo, hi in [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 10 ** 9)]:
        row(f"prefilled_tokens [{lo},{hi})", lambda d, lo=lo, hi=hi: lo <= d["pfl"] < hi)
    for lo, hi in [(0, 2), (2, 16), (16, 64), (64, 10 ** 9)]:
        row(f"completion_tokens [{lo},{hi})", lambda d, lo=lo, hi=hi: lo <= d["comp"] < hi)
    for fin in sorted(set(d["finish_log"] for d in D if "finish_log" in d)):
        row(f"finish={fin}", lambda d, fin=fin: d.get("finish_log") == fin)
    row("disk_hit", lambda d: d["disk_hit"]); row("prefix-cache restore=true", lambda d: d["restore"])
    row("ladder held >= 898 MiB at end", lambda d: d["held_mib"] >= 897)
    row("ladder held < 898 MiB at end", lambda d: d["held_mib"] < 897)

    print("\nPearson r with overhead:")
    for nm, fn in [("ckpts taken", lambda d: d["ckpts"]), ("cached_tokens", lambda d: d["cached"]),
                   ("prompt_tokens", lambda d: d["eff"]), ("prefilled_tokens", lambda d: d["pfl"]),
                   ("replay", lambda d: d["replay"]), ("completion_tokens", lambda d: d["comp"]),
                   ("held_mib", lambda d: d["held_mib"]), ("prefill_ms", lambda d: d["pf_ms"]),
                   ("decode_ms", lambda d: d["dec_ms"])]:
        print(f"  {nm:32s} {corr([fn(d) for d in D], ovs):6.3f}")
    print(f"  {'previous request overhead (lag1)':32s} {corr(ovs[:-1], ovs[1:]):6.3f}")

    print("\nOverhead vs the PREVIOUS request's features (is it post-work of the last request?)")
    prev = [None] + D[:-1]

    def prow(name, sel):
        o = [d["ov"] for p, d in zip(prev, D) if p and sel(p)]
        print(f"  {name:40s} n={len(o):5d} med={med(o):6.2f} >1s={sum(x > 1 for x in o)}")
    prow("prev ckpts>=2", lambda p: p["ckpts"] >= 2); prow("prev ckpts<=1", lambda p: p["ckpts"] <= 1)
    prow("prev prefilled>=1024", lambda p: p["pfl"] >= 1024); prow("prev prefilled<1024", lambda p: p["pfl"] < 1024)

    print("\nTop 15 overhead requests:")
    for i, d in sorted(enumerate(D), key=lambda x: -x[1]["ov"])[:15]:
        print(f"  #{i:4d} ov={d['ov']:.2f} wall={d['wall']:.2f} pf={d['pf_ms']:.0f}ms dec={d['dec_ms']:.0f}ms cached={d['cached']} pfl={d['pfl']} eff={d['eff']} sw={d['switch']} L={d['L']} C={d['C']} replay={d['replay']} ckpts={d['ckpts']} held={d['held_mib']} comp={d['comp']} {d['kind']}")
    hi = [d for d in D if d["ov"] > 1]; lo = [d for d in D if d["ov"] <= 1]
    print("\nmedians, >1s population vs rest")
    for nm, k in [("cached", "cached"), ("prefilled", "pfl"), ("eff prompt", "eff"), ("replay", "replay"),
                  ("ckpts", "ckpts"), ("completion", "comp"), ("wall", "wall")]:
        print(f"  {nm:12s} >1s: {med([d[k] for d in hi]):9.1f}   rest: {med([d[k] for d in lo]):9.1f}")
    print("  switch fraction  >1s: %.2f  rest: %.2f" % (sum(d['switch'] for d in hi) / max(1, len(hi)), sum(d['switch'] for d in lo) / max(1, len(lo))))
    print("  ckpts-per-request histogram:", {k: sum(d['ckpts'] == k for d in D) for k in range(0, 10)})
    print("  unique 'other' page/evict/insert lines:", sorted(set(x for d in D for x in d['other']))[:10])

    # ---- Second pass: is it a property of the request, or of the clock? ----
    import math
    tot = sum(d["wall"] for d in D)
    nsp = sum(d["ov"] > 1 for d in D)
    print("\n== Spike hazard: spikes (>1 s) per 100 s of request wall time, by feature (a request-caused stall would differ by group)")
    print("   overall: %d spikes in %.0f s = %.2f per 100 s (one per %.1f s)" % (nsp, tot, 100 * nsp / tot, tot / max(nsp, 1)))

    def haz(name, sel):
        g = [d for d in D if sel(d)]
        w = sum(d["wall"] for d in g)
        k = sum(d["ov"] > 1 for d in g)
        print(f"  {name:40s} n={len(g):5d} wall={w:7.0f}s spikes={k:4d} per100s={100 * k / max(w, 1e-9):5.2f}")
    haz("kind=primer", lambda d: d["kind"] == "primer"); haz("kind=probe", lambda d: d["kind"] == "probe")
    for k in range(1, 5): haz(f"ladder ckpts taken = {k}", lambda d, k=k: d["ckpts"] == k)
    haz("traj switch", lambda d: d["switch"]); haz("no traj switch", lambda d: not d["switch"])
    haz("prompt <= 16384", lambda d: d["eff"] <= 16384); haz("prompt > 16384 (pool overflow)", lambda d: d["eff"] > 16384)
    haz("prefilled < 1024", lambda d: d["pfl"] < 1024); haz("prefilled >= 4096", lambda d: d["pfl"] >= 4096)
    print("  (only wall >= 1.2 s requests can show a >1 s spike, so short groups are biased low; compare groups of similar wall)")
    for lo, hi in [(1.2, 3), (3, 8), (8, 15), (15, 40)]:
        g = [d for d in D if lo <= d["wall"] < hi]
        print("  wall [%g,%g) s: n=%d spike rate %.3f; chance a 34 s clock tick lands inside = mean wall/34 = %.3f" % (
            lo, hi, len(g), sum(d["ov"] > 1 for d in g) / len(g), st.mean([d["wall"] for d in g]) / 34.0))
    sp = sorted(d["ov"] for d in D if d["ov"] > 1)
    print("  spike size s: min %.2f p25 %.2f median %.2f p75 %.2f max %.2f; sum %.0f s = %.1f%% of all wall" % (
        sp[0], pct(sp, 25), med(sp), pct(sp, 75), sp[-1], sum(sp), 100 * sum(sp) / tot))

    # periodicity: end times of SHORT spiking probes (timing error < 5 s), circular mean test over candidate periods
    t = 0; ends = []
    for d in D:
        t += d["wall"]
        if d["ov"] > 1 and d["wall"] < 5: ends.append(t)
    gaps = [round(b - a, 1) for a, b in zip(ends, ends[1:])]
    print("\n== Periodicity of spikes (short probes only, %d of them)" % len(ends))
    print("   gaps between consecutive spikes, s:", gaps)
    best = (0, 0)
    for T10 in range(150, 600):
        T = T10 / 10
        R = abs(sum(complex(math.cos(2 * math.pi * e / T), math.sin(2 * math.pi * e / T)) for e in ends)) / len(ends)
        best = max(best, (R, T))
    print("   best period %.1f s, phase concentration R=%.2f (random would be about %.2f)" % (best[1], best[0], 1 / math.sqrt(len(ends))))
    near = [g for g in gaps if g > 20]
    print("   gaps > 20 s as multiples of the best period:", [round(g / best[1], 2) for g in near])

    # where does the time go for NON-spike requests? client wall minus server-side elapsed (process_job) vs body size
    print("\n== Where the overhead sits: client wall vs the server's own request timer (DONE line, 0.1 s resolution)")
    sp_d = [d for d in D if d["ov"] > 1]; ns_d = [d for d in D if d["ov"] <= 1]
    for nm, g in (("spike", sp_d), ("non-spike", ns_d)):
        inside = [d["srv_total"] - d["srv_pf"] - d["srv_dec"] for d in g]
        outside = [d["wall"] - d["srv_total"] for d in g]
        print(f"  {nm:10s} n={len(g):4d} inside request timer (not prefill/decode): median {med(inside):.2f} p90 {pct(inside, 90):.2f} max {max(inside):.2f}"
              f" | outside the timer: median {med(outside):.2f} p90 {pct(outside, 90):.2f} max {max(outside):.2f}")
    print("  non-spike outside-timer time by request body size (parse + template + tokenize + queue, before the timer starts):")
    for lo, hi in [(0, 50000), (50000, 90000), (90000, 120000), (120000, 160000), (160000, 10 ** 9)]:
        g = [d["wall"] - d["srv_total"] for d in ns_d if lo <= d.get("body", 0) < hi]
        if g: print(f"    body [{lo},{hi}) bytes n={len(g):4d} median {med(g):.3f} p90 {pct(g, 90):.3f}")
    xs = [d["body"] for d in ns_d]; ys = [d["wall"] - d["srv_total"] for d in ns_d]
    print("    Pearson r(outside-timer time, body bytes), non-spike: %.2f" % corr(xs, ys))
    return D


if __name__ == "__main__":
    for a in (sys.argv[1:] or DEF):
        run(a.replace(".jsonl", "").replace(".server.log", ""))
