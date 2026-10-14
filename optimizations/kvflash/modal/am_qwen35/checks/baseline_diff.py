"""Pair R0 `original` rows (parsed from the committed run stdout modal/sel.log) with R1b 4B `orig` rows.

usage: python3 -I baseline_diff.py <sel.log> <QuALITY dev file> <R1b 4b orig__t0.1__sel.jsonl>
R0 log lines used: 'Evaluating on article: T', 'Method: original', then per question
'Q<n>: ...', '  A: <think>', '<answer head> | Choice: c | Correct: g | mark'. No repo code is run.
"""
import json
import re
import sys
from collections import defaultdict


def parse_log(path):
    lines = open(path, errors="replace").read().splitlines()
    blocks = []  # (title, method, [(qn, choice, gold, mark)])
    cur = None
    for i, ln in enumerate(lines):
        if ln.startswith("Evaluating on article: "):
            cur = [ln[len("Evaluating on article: "):].strip(), None, []]
            blocks.append(cur)
        elif cur is not None and ln.startswith("Method: ") and cur[1] is None:
            cur[1] = ln[len("Method: "):].strip()
        elif cur is not None:
            m = re.search(r"\| Choice: (\S+) \| Correct: (\S+) \| (\S)\s*$", ln)
            if m:
                cur[2].append((m.group(1), m.group(2), m.group(3)))
            elif re.search(r"\| Choice: (None|\S+) \| Correct", ln):
                pass
            elif "Could not parse" in ln:
                pass
    return blocks


def main(log, data, r1b):
    title2id = {}
    for l in open(data):
        d = json.loads(l)
        title2id.setdefault(d["title"], "quality_" + d["article_id"])
    r0 = {}
    for t, m, rows in parse_log(log):
        if m == "original":
            r0[title2id[t]] = rows
    r1 = defaultdict(dict)
    for l in open(r1b):
        x = json.loads(l)
        r1[x["article_id"]][x["qi"]] = x
    n = a = b = both = neither = r0_nolabel = 0
    ex_a, ex_b = [], []
    r0c = r1c = 0
    for aid, rows in r0.items():
        for qi, (ch, gold, mark) in enumerate(rows):
            x = r1[aid][qi]
            c0 = mark == "✓"
            c1 = x["correct"]
            n += 1
            r0c += c0
            r1c += c1
            if c0 and not c1:
                a += 1
                ex_a.append((aid, qi, ch, gold, x["parsed"], x["gold"], x["answer"]))
            elif c1 and not c0:
                b += 1
                ex_b.append((aid, qi, ch, gold, x["parsed"], x["gold"], x["answer"]))
            elif c0 and c1:
                both += 1
            else:
                neither += 1
    print("R0 articles", len(r0), "R0 rows", sum(len(v) for v in r0.values()), "R1b rows", sum(len(v) for v in r1.values()))
    print("paired n", n, "R0 correct", r0c, "R1b correct", r1c)
    print("R0 right/R1b wrong", a, "R1b right/R0 wrong", b, "both right", both, "both wrong", neither)
    # gold agreement sanity
    mism = sum(1 for aid, rows in r0.items() for qi, (ch, gold, mark) in enumerate(rows)
               if str(r1[aid][qi]["gold"]) != gold)
    print("gold mismatches", mism)
    print("R0-right/R1b-wrong examples (aid, qi, R0 choice, R0 gold, R1b parsed, R1b gold, R1b answer)")
    for e in ex_a[:3]:
        print(" ", e)
    print("R1b-right/R0-wrong examples")
    for e in ex_b[:3]:
        print(" ", e)
    none0 = sum(1 for rows in r0.values() for ch, g, mk in rows if ch == "None")
    none1 = sum(1 for d in r1.values() for x in d.values() if x["parsed"] is None)
    print("unparsed R0", none0, "unparsed R1b", none1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
