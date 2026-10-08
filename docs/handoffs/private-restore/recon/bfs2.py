"""Fast hash-guided BFS for one file: seeds = every git blob of PATH (git log --raw) + extra files;
edges = every per-file section of every .patch/.diff under ROOTS. Writes recovered/<PATH> on hit.
Usage: bfs2.py PATH TARGET_SHA256 DEPTH ROOT... [--extra FILE...]"""
import hashlib, os, re, subprocess, sys, pathlib
path, target, depth = sys.argv[1], sys.argv[2], int(sys.argv[3])
args = sys.argv[4:]
extra = args[args.index("--extra") + 1:] if "--extra" in args else []
roots = args[:args.index("--extra")] if "--extra" in args else args
sha = lambda b: hashlib.sha256(b).hexdigest()
git = lambda *a: subprocess.run(["git", *a], capture_output=True).stdout
here = pathlib.Path(__file__).resolve().parent; os.chdir(here.parents[3])
secs = {}
for root in roots:
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith((".patch", ".diff")):
                for part in re.split(r"(?m)^(?=--- )", open(os.path.join(dp, f), errors="replace").read()):
                    m = re.search(r"(?m)^\+\+\+ b/(\S+)", part)
                    if m and m.group(1) == path:
                        secs.setdefault(part, os.path.join(dp, f))
raw = git("log", "--all", "--reflog", "--format=", "--raw", "--no-abbrev", "--", path).decode().split("\n")
blobs = {l.split()[3] for l in raw if l.startswith(":")} | {l.split()[2] for l in raw if l.startswith(":")}
states = {}
for b in blobs:
    if set(b) != {"0"}:
        d = git("cat-file", "blob", b); states[sha(d)] = d
if os.environ.get("EXTRA_ONLY"):
    states = {}
for e in extra + ([] if os.environ.get("EXTRA_ONLY") else [path]):
    if os.path.exists(e):
        d = open(e, "rb").read(); states[sha(d)] = d
known = set(open(os.environ["KNOWN"]).read().split()) if os.environ.get("KNOWN") else set()
print("seed-known:", sorted(h[:8] for h in states if h in known))
print(f"{path}: {len(states)} seeds, {len(secs)} sections", flush=True)
work = here / "work2"; work.mkdir(exist_ok=True)
frontier, hit = dict(states), states.get(target)
for lvl in range(depth):
    if hit: break
    nxt = {}
    for data in frontier.values():
        (work / "src").write_bytes(data)
        for (part, pname), rev in [(x, r) for x in secs.items() for r in ([], ["-R"])]:
            r = subprocess.run(["patch", "-s", "-f", *rev, "--no-backup-if-mismatch", "-o", str(work / "res"), str(work / "src")],
                               input=part.encode(), capture_output=True)
            pname = pname + (" (reverse)" if rev else "")
            if r.returncode: continue
            d = (work / "res").read_bytes(); h = sha(d)
            if h == target: hit = d; print("HIT depth", lvl + 1, "via", pname, flush=True); break
            if h not in states and (h in known or not os.environ.get("KNOWN_ONLY")):
                states[h] = nxt[h] = d
                if h in known: print("REACH", h[:8], "depth", lvl + 1, "via", pname, flush=True)
        if hit: break
    frontier = nxt
    print(f"depth{lvl+1} new={len(nxt)}", flush=True)
if hit:
    dst = here / "recovered" / path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(hit)
else:
    print("MISS", path)
