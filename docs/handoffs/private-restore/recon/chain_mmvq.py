"""Approximate mmvq.cu: apply the 38.41 lineage patches in order (fuzzy) to candidate bases; report per-step result + hash."""
import hashlib, json, os, pathlib, re, subprocess, sys
here = pathlib.Path(__file__).resolve().parent
os.chdir(here.parents[3])
M = "docs/handoffs/strata-valid-comparison/methods/"
P = "server/deps/llama.cpp/ggml/src/ggml-cuda/mmvq.cu"
CHAIN = [M + x for x in sys.argv[2].split(",")]
known = set(open("/home/peppi/.claude/jobs/559c1173/tmp/known-all.txt").read().split())
sha = lambda b: hashlib.sha256(b).hexdigest()
def section(pf):
    for part in re.split(r"(?m)^(?=--- )", open(pf, errors="replace").read()):
        m = re.search(r"(?m)^\+\+\+ b/(\S+)", part)
        if m and m.group(1) == P:
            return part
work = here / "work3"; work.mkdir(exist_ok=True)
for base in sys.argv[1].split(","):
    data = open(base, "rb").read()
    print("BASE", base[-40:], sha(data)[:8])
    for pf in CHAIN:
        sec = section(pf)
        if sec is None:
            print("   (no mmvq section)", pf[len(M):]); continue
        (work / "src").write_bytes(data)
        r = subprocess.run(["patch", "-f", "--no-backup-if-mismatch", "-r", str(work / "rej"), "-o", str(work / "res"), str(work / "src")],
                           input=sec, capture_output=True, text=True)
        data = (work / "res").read_bytes()
        h = sha(data)
        print(f"   rc={r.returncode} {h[:8]}{' KNOWN' if h in known else ''} {pf[len(M):]} :: {' '.join(r.stdout.split())[:200]}")
    out = here / "approx" / (pathlib.Path(base).name + ".mmvq.cu"); out.parent.mkdir(exist_ok=True); out.write_bytes(data)
