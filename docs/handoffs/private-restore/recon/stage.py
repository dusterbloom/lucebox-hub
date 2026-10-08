"""Stage the exact 38.41 source tree: every path of the build manifest from the worktree, overlay on top; verify all hashes.
Usage: stage.py BUILD_JSON DEST"""
import hashlib, json, pathlib, shutil, sys
here = pathlib.Path(__file__).resolve().parent
repo, overlay = here.parents[3], here.parent / "exact-overlay"
hashes = json.load(open(sys.argv[1]))["source_hashes"]
dest = pathlib.Path(sys.argv[2])
bad = []
for p, h in hashes.items():
    src = overlay / p if (overlay / p).exists() else repo / p
    dst = dest / p
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest() != h:
        bad.append(p)
print(len(hashes), "files,", len(bad), "mismatch:", bad)
