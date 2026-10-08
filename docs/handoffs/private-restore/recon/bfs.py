"""Reconstruct exact 38.41 source files by hash: BFS over (file version) x (per-file hunks of every known patch).

Usage: bfs.py miss.json found.json repo_root patch_root... ; writes recovered files into ./recovered/<path>.
"""
import hashlib, json, os, re, subprocess, sys, pathlib

here = pathlib.Path(__file__).parent
miss = json.load(open(sys.argv[1]))
found = json.load(open(sys.argv[2]))
repo = sys.argv[3]
patch_roots = sys.argv[4:]
sha = lambda b: hashlib.sha256(b).hexdigest()
git = lambda *a, **k: subprocess.run(["git", "-C", repo, *a], capture_output=True, **k).stdout

# per-file sections of every patch
sections = {}  # path -> list of (patchname, text)
for root in patch_roots:
    for dp, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith((".patch", ".diff")):
                continue
            txt = open(os.path.join(dp, f), errors="replace").read()
            for part in re.split(r"(?m)^(?=--- )", txt):
                m = re.search(r"(?m)^\+\+\+ b/(\S+)", part)
                if m:
                    sections.setdefault(m.group(1), []).append((os.path.join(dp, f), part))

out = here / "recovered"
work = here / "work"; work.mkdir(exist_ok=True)
for path, target, _ in miss:
    if target in found or (os.environ.get("ONLY") and not any(s in path for s in os.environ["ONLY"].split(","))) or path.endswith(".orig") or "/" not in path:
        continue
    # seeds: every historical blob of this path + the working tree copy
    seeds = {}
    for c in git("rev-list", "--all", "--reflog", text=True).split():
        o = git("ls-tree", c, "--", path, text=True).split()
        if len(o) >= 3 and o[2] not in seeds:
            seeds[o[2]] = None
    states = {}
    for b in seeds:
        data = git("cat-file", "blob", b)
        states[sha(data)] = data
    wt = pathlib.Path(repo, path)
    if wt.exists():
        states[sha(wt.read_bytes())] = wt.read_bytes()
    secs = sections.get(path, [])
    frontier = dict(states)
    hit = states.get(target)
    for depth in range(int(os.environ.get("DEPTH", "3"))):
        if hit is not None:
            break
        nxt = {}
        for h0, data in frontier.items():
            src = work / "src"; src.write_bytes(data)
            for pname, part in secs:
                res = work / "res"
                r = subprocess.run(["patch", "-s", "-f", "--no-backup-if-mismatch", "-o", str(res), str(src)],
                                   input=part.encode(), capture_output=True)
                if r.returncode != 0 or not res.exists():
                    continue
                d = res.read_bytes(); h = sha(d)
                if h == target:
                    hit = d; print(f"HIT {path} depth{depth+1} via {pname}", flush=True); break
                if h not in states:
                    states[h] = d; nxt[h] = d
            if hit is not None:
                break
        frontier = nxt
        print(f"{path}: depth{depth+1} seeds={len(seeds)} sections={len(secs)} new={len(nxt)}", flush=True)
    if hit is not None:
        dst = out / path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(hit)
    else:
        print(f"MISS {path}", flush=True)
