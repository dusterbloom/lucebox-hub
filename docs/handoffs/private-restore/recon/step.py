"""Apply mmvq.cu sections of the given patches in order to BASE; print patch output and keep rejects in work4/."""
import os, pathlib, re, subprocess, sys
here = pathlib.Path(__file__).resolve().parent
os.chdir(here.parents[3])
M = "docs/handoffs/strata-valid-comparison/methods/"
P = "server/deps/llama.cpp/ggml/src/ggml-cuda/mmvq.cu"
work = here / "work4"; work.mkdir(exist_ok=True)
data = open(sys.argv[1], "rb").read()
for i, pf in enumerate(sys.argv[2:]):
    sec = next(p for p in re.split(r"(?m)^(?=--- )", open(M + pf, errors="replace").read())
               if re.search(r"(?m)^\+\+\+ b/" + re.escape(P) + r"\s", p))
    (work / "src").write_bytes(data)
    r = subprocess.run(["patch", "-f", "--no-backup-if-mismatch", "-r", str(work / f"rej{i}"), "-o", str(work / "res"), str(work / "src")],
                       input=sec, capture_output=True, text=True)
    print(pf, r.returncode, r.stdout.replace(str(work), ""), sep="\n")
    data = (work / "res").read_bytes()
(work / "final.cu").write_bytes(data)
