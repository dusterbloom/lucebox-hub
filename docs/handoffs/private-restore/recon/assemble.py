"""Assemble the exact 38.41 overlay (files differing from the dirty tree) into ../exact-overlay/, verifying each sha256."""
import hashlib, json, os, pathlib, shutil, subprocess
here = pathlib.Path(__file__).resolve().parent
repo = here.parents[3]
os.chdir(repo)
targets = {p: h for p, h, _ in json.load(open(here / "miss.json"))}
found = json.load(open(here / "miss.json.found"))
out = here.parent / "exact-overlay"
sha = lambda b: hashlib.sha256(b).hexdigest()
for path, want in targets.items():
    if path.endswith(".orig"):
        continue
    src = found.get(want)
    if src and src.startswith("blob:"):
        data = subprocess.run(["git", "cat-file", "blob", src[5:]], capture_output=True).stdout
    elif src:
        data = open(src[5:], "rb").read()
    else:
        cands = [here / d / path for d in ("recovered", "recovered2")]
        data = next((c.read_bytes() for c in cands if c.exists()), None)
    ok = data is not None and sha(data) == want
    print("OK  " if ok else "MISS", path)
    if ok:
        dst = out / path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data)
