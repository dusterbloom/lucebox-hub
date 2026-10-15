"""Markdown tables from summary.json. Usage: python3 -I make_tables.py <summary.json>"""
import json
import sys

s = json.load(open(sys.argv[1]))
KS = ("8", "16", "32", "64", "128")
SUB = (("older_than_16K", "older than 16K (primary)"), ("within_16K", "within 16K"), ("all", "all 951"))


def pct(h, n):
    return "%d/%d (%.1f%%)" % (h, n, 100.0 * h / n) if n else "-"


for which, wname in (("fact", "fact span (primary)"), ("lit", "literal PREREG rule: [src_tok, src_tok + gold length]"), ("msg", "whole source message")):
    for variant, vname in (("last", "(a) last prefill token's query"), ("max", "(b) max over the question's prefill tokens")):
        v = s["variants"][variant]
        if isinstance(v, str):
            print("\n### %s, %s: %s\n" % (wname, vname, v))
            continue
        print("\n### Hit = %s; query %s\n" % (wname, vname))
        print("| subset | n | " + " | ".join("recall@%s" % k for k in KS) + " |")
        print("|---|---|" + "---|" * len(KS))
        for key, label in SUB:
            r = v[which][key]
            print("| %s | %d | " % (label, r["n"]) + " | ".join(pct(r["recall"][k]["hits"], r["n"]) for k in KS) + " |")
        print("| chance (all 951) | | " + " | ".join("%.1f%%" % (100 * v[which]["all"]["recall"][k]["chance_rate"]) for k in KS) + " |")
        print("| chance (older than 16K) | | " + " | ".join("%.1f%%" % (100 * v[which]["older_than_16K"]["recall"][k]["chance_rate"]) for k in KS) + " |")

for which in ("fact",):
    for variant in ("last", "max"):
        v = s["variants"][variant]
        if isinstance(v, str):
            continue
        print("\n### Rank distribution, fact span, variant %s\n" % variant)
        keys = list(v[which]["all"]["rank_hist"].keys())
        print("| subset | n ranked | " + " | ".join(keys) + " | median | q25 | q75 | median rank / n_chunks |")
        print("|---|---|" + "---|" * (len(keys) + 4))
        for key, label in SUB:
            r = v[which][key]
            rs = r["rank_stats"] or {}
            fs = r["rank_frac_stats"] or {}
            print("| %s | %d | " % (label, r["n_ranked"]) + " | ".join(str(r["rank_hist"][k]) for k in keys)
                  + " | %s | %s | %s | %s |" % (rs.get("median"), rs.get("q25"), rs.get("q75"), "%.3f" % fs["median"] if fs else "-"))
        print("\n### Resident at prefill end and recall over the non-resident probes, fact span, variant %s\n" % variant)
        print("| subset | n | resident at end (or in tail) | non-resident n | " + " | ".join("nonres recall@%s" % k for k in KS) + " |")
        print("|---|---|---|---|" + "---|" * len(KS))
        for key, label in SUB:
            r = v[which][key]
            nr = r["non_resident"]
            print("| %s | %d | %d | %d | " % (label, r["n"], r["n_resident_at_end"], nr["n"])
                  + " | ".join(pct(nr["recall"][k]["hits"], nr["n"]) for k in KS) + " |")
