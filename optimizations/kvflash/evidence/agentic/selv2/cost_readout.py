"""Scorer math ms per request from 'exact-scorer fast split' log lines: n_q, sealed chunks, math ms. Medians per n_q, and
ms per query per 1K sealed tokens (the dot cost is linear in both).
    python3 -I cost_readout.py <server.log> [<server.log> ...]"""
import re
import statistics
import sys

pat = re.compile(r"exact-scorer fast split: (?:n_q=(\d+) )?sealed=(\d+) .*?math ([\d.]+) ms")
for p in sys.argv[1:]:
    rows = [(int(m[1] or 1), int(m[2]), float(m[3])) for m in map(pat.search, open(p, errors="replace")) if m]
    for n in sorted({r[0] for r in rows}):
        sel = [r for r in rows if r[0] == n]
        per = [ms / q / (s * 64 / 1000) for q, s, ms in sel]
        print(f"{p}: n_q {n}: {len(sel)} calls, median sealed {statistics.median(r[1] for r in sel):.0f} chunks, "
              f"median math {statistics.median(r[2] for r in sel):.1f} ms, {statistics.median(per):.3f} ms per query per 1K tokens")
