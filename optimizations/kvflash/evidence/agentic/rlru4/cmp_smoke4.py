"""Mode-4 smoke (a) gate (PREREG.md): with LUCE_KVFLASH_REWIND_NOOP=1, every re-prefill reproduces the first pass's final logits
bit for bit; the fitting probe logs `skip, fits` and answers exactly as OFF; no row errors. Reports flips vs OFF (the R4 batch
boundary) for the rest.   python3 -I cmp_smoke4.py <off QK16KND.jsonl> <noop QK16KND.jsonl> <noop server.log> <fitting probe id>"""
import json
import re
import sys

off, noop = ({r["id"]: r for r in map(json.loads, open(p))} for p in sys.argv[1:3])
log, fit = open(sys.argv[3]).read(), sys.argv[4]
diffs = [float(d) for d in re.findall(r"mode=4 @\d+: re-prefill \d+ tokens ok, logits max\|diff\| (\S+) vs first pass", log)]
skips = log.count("mode=4: skip, fits")
errors = [i for i in noop if "error" in noop[i]]
fit_ok = fit in off and fit in noop and off[fit].get("content") == noop[fit].get("content")
flips = [i for i in off if off[i].get("kind") == "probe" and i != fit and off[i].get("content") != noop.get(i, {}).get("content")]
print(f"smoke a: {len(noop)} requests, {len(diffs)} re-prefills, logits max|diff| {diffs}, {skips} skips, fitting probe equals OFF {fit_ok}, errors {errors}")
print(f"flip rate vs OFF (batch boundary at R4): {len(flips)}/{sum(off[i].get('kind') == 'probe' and i != fit for i in off)} probes {flips}")
sys.exit(0 if diffs and not any(diffs) and skips and fit_ok and not errors else 1)
