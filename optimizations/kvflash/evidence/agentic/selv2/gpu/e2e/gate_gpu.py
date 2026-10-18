"""PREREG_gpu checks 1-2 (stdlib only).
    python3 -I gate_gpu.py shadow <dir>   every reselect: a gpu scorer line and a shadow pick comparison; picks vs the CPU
                                         scorer identical on >= 90%, >= 26 of 30 units in common on all
    python3 -I gate_gpu.py gpu <dir>      PREREG_e2e check 2 (gate_e2e.v2) + one gpu scorer line per reselect, none unavailable
    python3 -I gate_gpu.py ident <a> <b>  check 3: probe answers equal on >= 99%"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2e"))
import gate_e2e  # noqa: E402

GPU = re.compile(r"\[kvflash\] gpu scorer: n_q=\d+ sealed=\d+ resident=\d+ .* total ([\d.]+) ms")
UNAVAIL = "gpu scorer unavailable"


def gpu_lines(d):
    _, blocks = gate_e2e.load(d)
    text = "".join(blocks)
    nsel = len(gate_e2e.SEL.findall(text))
    tot = sorted(float(x) for x in GPU.findall(text))
    print(f"gpu: {nsel} reselects, {len(tot)} gpu scorer lines, {text.count(UNAVAIL)} unavailable; total ms median "
          f"{tot[len(tot) // 2] if tot else None} max {tot[-1] if tot else None}")
    return text, nsel, len(tot) == nsel > 0 and UNAVAIL not in text


def shadow(d):
    text, nsel, ok = gpu_lines(d)
    # kvflash_prefill_apply's shadow line: ref = the CPU scorer's picks, fast = the GPU scorer's (in use)
    pairs = re.findall(r"\[kvflash\] scorer shadow: ref=\[([0-9,]*)\] fast=\[([0-9,]*)\]", text)
    common = [len(set(r.split(",")) & set(f.split(","))) if r and f else 0 for r, f in pairs]
    same = sum(c == 30 for c in common)
    print(f"shadow: {len(common)} comparisons; identical {same}; min common {min(common, default=None)}")
    return 0 if ok and len(common) == nsel and same >= 0.9 * nsel and min(common, default=0) >= 26 else 1


def gpu(d):
    rc = gate_e2e.v2(d)
    _, _, ok = gpu_lines(d)
    return 0 if rc == 0 and ok else 1


def ident(a, b):
    """Check 3: probe answers (content + tool calls) of two arms equal on >= 99% of shared probes."""
    pa, pb = ({r["id"]: r for r, _ in zip(*gate_e2e.load(d)) if r.get("kind", "probe") == "probe"} for d in (a, b))
    ids = [i for i in pa if i in pb]
    eq = sum((pa[i].get("content"), pa[i].get("tool_calls")) == (pb[i].get("content"), pb[i].get("tool_calls")) for i in ids)
    print(f"ident: {eq}/{len(ids)} probes equal")
    return 0 if ids and eq >= 0.99 * len(ids) else 1


if __name__ == "__main__":
    m = sys.argv[1]
    sys.exit(shadow(sys.argv[2]) if m == "shadow" else gpu(sys.argv[2]) if m == "gpu" else ident(sys.argv[2], sys.argv[3]))
