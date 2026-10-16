"""Offline router bake-off for KVFlash mode 4 (CPU only).  Run:  python3 -I bakeoff.py [--n N] > bakeoff.out
Reuses rank4_cal.py's estimator (window/tail rule, fits-skip, P_WHOLE/P_PART/P_MISS) and swaps the block ranking.
1 block = 10 chunks x 64 tokens.  Candidate blocks = 1..nb-1 (block 0 and the window excluded), as rank4_cal.top()."""
import bisect
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict

W = "/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am"
AG = W + "/optimizations/kvflash/evidence/agentic"
sys.path.insert(0, W + "/optimizations/kvflash/agentic_eval")
sys.path.insert(0, AG + "/rlru4/agentic_eval")
import convert  # noqa: E402  (render + tokenizer, same path as E1 and the server)
import probes  # noqa: E402

CT, CPB, POOL, OLD = 64, 10, 256, 16384
BT = CT * CPB
KS = (1, 2, 4, 6, 8, 10)
P_WHOLE, P_PART = {"within": 0.95, "beyond": 0.95}, {"within": 0.94, "beyond": 0.94}
P_MISS = {"within": 0.29, "beyond": 0.11}
BEYOND_CUT = 0.83  # rank4_cal over-estimates beyond-16K by ~17%
NSEED = 20
ALIGNED = "--aligned" in sys.argv  # robustness run: only probes whose local token count equals the server's prompt_tokens
LIMIT = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else None

tasks = {t["id"]: t for t in map(json.loads, open("/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"))}
traj = {r["trajectory_id"]: r for r in map(json.loads, open("/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl"))}
span = {r["id"]: r["span"] for r in json.load(open(AG + "/E1_router_recall/summary.per_probe.json"))}
reqs = [json.loads(l) for l in open(AG + "/rlru/dump/QK16KND.jsonl")]
dump = [json.loads(l) for l in open(AG + "/rlru/dump/exact_dump/exact_dump.jsonl")]
rows = [(r, d) for r, d in zip(reqs, dump) if r.get("kind") == "probe" and r["id"] in span]
assert len(rows) == 951
if LIMIT:
    rows = rows[:LIMIT]
TOK = convert.tokenizer()
IM_START, IM_END = TOK.token_to_id("<|im_start|>"), TOK.token_to_id("<|im_end|>")
USER = TOK.encode("user").ids[0]


# ---------------------------------------------------------------- text helpers (lex.py ideas)
def words(s):
    out = []
    for w in re.findall(r"[\w./\-]+", s):
        out.append(w.lower())
        out.extend(x.lower() for x in re.split(r"[./\-_]+", w) if len(x) > 1)
    return out


def bm25(qterms, tfs, lens):
    """lex.py formula.  tfs: list of {term: tf} (only query terms needed); returns scores."""
    n = len(tfs)
    if n == 0:
        return []
    avg = sum(lens) / n or 1.0
    df = Counter()
    for d in tfs:
        df.update(d.keys())
    idf = {w: math.log(1 + (n - df[w] + .5) / (df[w] + .5)) for w in qterms if df[w]}
    out = []
    for d, L in zip(tfs, lens):
        s = 0.0
        for w, f in d.items():
            if w in idf:
                s += idf[w] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * L / avg))
        out.append(s)
    return out


def order(cand, score):
    return [b for _, b in sorted(((-score[b], -b) for b in cand), key=lambda x: (x[0], x[1]))] if False else \
        [b for b in sorted(cand, key=lambda b: (-score[b], -b))]


def rrf(orders, k0=60):
    sc = defaultdict(float)
    for o in orders:
        for r, b in enumerate(o):
            sc[b] += 1.0 / (k0 + r + 1)
    return sorted(sc, key=lambda b: (-sc[b], -b))


# ---------------------------------------------------------------- lexical leak
GENERIC = re.compile(r"^[A-Za-z]+$")


def idents(q):
    out = set()
    for w in re.findall(r"[A-Za-z0-9_./\-]{6,}", q):
        w = w.strip(".,;:-/")
        if len(w) >= 6 and (re.search(r"[/_.\d]", w) or re.search(r"[a-z][A-Z]", w)):
            out.add(w)
    return out


def msg_text(m):
    s = m.get("content") or ""
    if not isinstance(s, str):
        s = json.dumps(s)
    tc = m.get("tool_calls")
    if tc:
        s += " " + json.dumps(tc)
    return s


# ---------------------------------------------------------------- estimator pieces
class Acc:
    def __init__(self):
        self.est = {"within": 0.0, "beyond": 0.0}
        self.cnt = [0.0, 0.0, 0.0]
        self.fact = {"within": 0.0, "beyond": 0.0}
        self.msg = {"within": 0.0, "beyond": 0.0}
        self.n = {"within": 0.0, "beyond": 0.0}


R_NAMES = ["exact", "mass_r", "recency", "random", "bm25_tok", "bm25_txt", "msg_bm25", "rrf_tok", "rrf_txt",
           "pre_tok", "pre_txt", "oracle"]
acc = {(n, k): Acc() for n in R_NAMES for k in KS}
off_est = {"within": 0, "beyond": 0}
align = {"n": 0, "tok_mismatch": 0, "dump_mismatch": 0, "fact_bad": 0, "msg_bad": 0, "fact_chk": 0, "msg_chk": 0}
leak = {"within": [0, 0], "beyond": [0, 0]}  # [quoted, n]
leak_all = [0, 0]
tim = {"build": 0.0, "query": 0.0, "tokens": 0, "postings": 0, "n": 0, "dict_terms": 0}
mismatch_ids = []
deltas = []
hits = defaultdict(dict)
risky = [0]

t_start = time.time()
for pi, (r, d) in enumerate(rows):
    i = r["id"]
    t = tasks[i]
    T = traj[t["traj"]]
    band = "beyond" if t["distance_tok"] > OLD else "within"
    ans = r.get("content") or ""
    sc_off = probes.score_exact(ans, t["answer"])
    msgs = list(T["messages"][: t["cut_index"]]) + [{"role": "user", "content": t["question"]}]
    s = convert.render(msgs, T["tools"], add_generation_prompt=True)
    enc = TOK.encode(s)
    ids, offs = enc.ids, enc.offsets
    if ALIGNED and len(ids) != r["prompt_tokens"]:
        continue
    off_est[band] += sc_off
    align["n"] += 1
    if len(ids) != r["prompt_tokens"]:
        align["tok_mismatch"] += 1
        mismatch_ids.append(i)
        delta = r["prompt_tokens"] - len(ids)
        deltas.append(delta)
        # upper bound: the fact span (+-|delta| tokens) straddles a block boundary
        if (max(span[i]["fact"][0] - abs(delta), 0)) // BT != (span[i]["fact"][1] + abs(delta)) // BT:
            risky[0] += 1
    if len(ids) != d["prompt_len"]:
        align["dump_mismatch"] += 1
    fa, fb = span[i]["fact"]
    ma, mb = span[i]["msg"]
    # span checks: fact span decodes to contain the answer; msg span contains the source message content head
    if len(ids) > mb:
        align["fact_chk"] += 1
        if t["answer"] not in s[offs[max(fa - 1, 0)][0]: offs[min(fb + 1, len(ids) - 1)][1]]:
            align["fact_bad"] += 1
        align["msg_chk"] += 1
        src = (T["messages"][t["msg_index"]].get("content") or "")
        head = src.strip()[:60]
        if head and head not in s[offs[ma][0]: offs[mb][1]]:
            align["msg_bad"] += 1
    else:
        align["fact_bad"] += 1
        align["msg_bad"] += 1
    # lexical leak
    src_txt = msg_text(T["messages"][t["msg_index"]])
    ids_q = idents(t["question"])
    q_hit = any(x in src_txt for x in ids_q)
    leak[band][0] += q_hit
    leak[band][1] += 1

    # query = last user turn, without the im_start/user/newline header and im_end/generation prompt
    qa = max(j for j in range(len(ids) - 1) if ids[j] == IM_START and ids[j + 1] == USER)
    qe = max(j for j in range(qa, len(ids)) if ids[j] == IM_END)
    qtok = ids[qa + 3: qe]
    qkeys = set(qtok) | {(a << 20) | b for a, b in zip(qtok, qtok[1:])}
    nblk = -(-len(ids) // BT)
    # per-block token-id tf restricted to query keys
    btf = []
    blen = []
    for b in range(nblk):
        seg = ids[b * BT:(b + 1) * BT]
        c = Counter(x for x in seg if x in qkeys)
        c.update(k2 for k2 in ((a << 20) | bb for a, bb in zip(seg, seg[1:])) if k2 in qkeys)
        btf.append(c)
        blen.append(len(seg))
    # per-block word tf (text variant)
    qwords = set(words(t["question"]))
    wtf, wlen = [], []
    for b in range(nblk):
        a0, a1 = offs[b * BT][0], offs[min((b + 1) * BT, len(ids)) - 1][1]
        ws = words(s[a0:a1])
        wtf.append(Counter(x for x in ws if x in qwords))
        wlen.append(len(ws))
    # per-segment (message-level) word tf: split at turn starts and each tool_response
    cuts = [0] + [m.start() for m in re.finditer(r"<\|im_start\|>|<tool_response>", s) if m.start() > 0] + [len(s)]
    starts = [o[0] for o in offs]
    ends = [o[1] for o in offs]
    segs = []
    for cs, ce in zip(cuts, cuts[1:]):
        ws = words(s[cs:ce])
        ta, tb = bisect.bisect_right(ends, cs), bisect.bisect_left(starts, ce) - 1
        if tb < ta:
            continue
        segs.append((ta // BT, tb // BT, Counter(x for x in ws if x in qwords), len(ws)))

    # BM25 index cost sample (full uni+bigram postings, pure python)
    if pi < 40:
        t0 = time.time()
        inv = defaultdict(list)
        for b in range(nblk):
            seg = ids[b * BT:(b + 1) * BT]
            c = Counter(seg)
            c.update((a << 20) | bb for a, bb in zip(seg, seg[1:]))
            for term, f in c.items():
                inv[term].append((b, f))
        tim["build"] += time.time() - t0
        t0 = time.time()
        N = nblk
        score = defaultdict(float)
        for term in qkeys:
            pl = inv.get(term)
            if pl:
                idf = math.log(1 + (N - len(pl) + .5) / (len(pl) + .5))
                for b, f in pl:
                    score[b] += idf * f * 2.2 / (f + 1.2)
        sorted(score.items(), key=lambda x: -x[1])[:10]
        tim["query"] += time.time() - t0
        tim["tokens"] += len(ids)
        tim["postings"] += sum(len(v) for v in inv.values())
        tim["dict_terms"] += len(inv)
        tim["n"] += 1

    mass_end, mass_r = d["mass_end"], d["mass_r"]
    bsum_e = {b: sum(mass_end[b * CPB:(b + 1) * CPB]) for b in range(nblk)}
    bsum_r = {b: sum(mass_r[b * CPB:(b + 1) * CPB]) for b in range(nblk)}
    for k in KS:
        reserve = -(-r["timings"]["prefilled_tokens"] // CT) + 1
        tail = POOL - 1 - CPB * k - reserve - 1
        fwc = (d["r"] - 1) // CT - tail
        fits = -(-d["prompt_len"] // CT) + reserve <= POOL
        if fits or fb // CT >= fwc:
            for n in R_NAMES:
                acc[(n, k)].est[band] += sc_off
            continue
        nb = fwc // CPB
        cand = list(range(1, nb))
        fact_blocks = {c // CPB for c in range(fa // CT, fb // CT + 1)}
        msg_cs = [c for c in range(ma // CT, mb // CT + 1) if c < fwc]
        msg_blocks = {c // CPB for c in msg_cs}
        o_exact = order(cand, bsum_e)
        # block bm25 (token ids) and text variant, IDF over candidate blocks only
        s_tok = dict(zip(cand, bm25(qkeys, [btf[b] for b in cand], [blen[b] for b in cand])))
        s_txt = dict(zip(cand, bm25(qwords, [wtf[b] for b in cand], [wlen[b] for b in cand])))
        o_tok, o_txt = order(cand, s_tok), order(cand, s_txt)
        # message-level: rank segments that touch a candidate block; take their blocks in message order of rank
        cset = set(cand)
        sg = [x for x in segs if any(b in cset for b in range(x[0], x[1] + 1))]
        sgs = bm25(qwords, [x[2] for x in sg], [x[3] for x in sg])
        ranked = sorted(range(len(sg)), key=lambda j: (-sgs[j], -sg[j][1]))
        o_msg, seen = [], set()
        for j in ranked:
            bl = [b for b in range(sg[j][0], sg[j][1] + 1) if b in cset and b not in seen]
            bl.sort(key=lambda b: (-s_txt[b], -b))
            o_msg += bl
            seen.update(bl)
        o_msg += [b for b in o_txt if b not in seen]
        # oracle: fact blocks, then the other source-message blocks, then exact
        o_or = [b for b in sorted(fact_blocks) if b in cset]
        o_or += [b for b in sorted(msg_blocks) if b in cset and b not in o_or]
        o_or += [b for b in o_exact if b not in o_or]
        orders = {
            "exact": o_exact,
            "mass_r": order(cand, bsum_r),
            "recency": sorted(cand, reverse=True),
            "bm25_tok": o_tok, "bm25_txt": o_txt, "msg_bm25": o_msg,
            "rrf_tok": rrf([o_exact, o_tok]), "rrf_txt": rrf([o_exact, o_txt]),
            "pre_tok": order(o_tok[:16], bsum_e) + [b for b in o_exact if b not in set(o_tok[:16])],
            "pre_txt": order(o_txt[:16], bsum_e) + [b for b in o_exact if b not in set(o_txt[:16])],
            "oracle": o_or,
        }
        runs = [(n, o, 1.0) for n, o in orders.items()]
        for seed in range(NSEED):
            rg = random.Random(seed * 1000003 + i.__hash__() * 0 + pi)
            o = cand[:]
            rg.shuffle(o)
            runs.append(("random", o, 1.0 / NSEED))
        for n, o, w in runs:
            a = acc[(n, k)]
            picked = set(o[:k])
            a.n[band] += w
            fhit = bool(fact_blocks & picked)
            a.fact[band] += w * fhit
            if n != "random":
                hits[(n, k)][i] = fhit
            a.msg[band] += w * bool(msg_blocks & picked)
            if fhit:
                whole = all(c >= fwc or c // CPB in picked for c in range(ma // CT, mb // CT + 1))
                a.est[band] += w * (P_WHOLE[band] if whole else P_PART[band])
                a.cnt[0 if whole else 1] += w
            else:
                a.cnt[2] += w
                a.est[band] += w * P_MISS[band]
    if pi % 100 == 0:
        print(f"[{pi}/{len(rows)}] {time.time() - t_start:.0f}s", file=sys.stderr, flush=True)

# ---------------------------------------------------------------- report
n_all = len(rows)
print(f"probes: {n_all}   OFF (dump arm): within {off_est['within']}, beyond {off_est['beyond']}")
print("\n== STEP 1 alignment ==")
print(f"token count != prompt_tokens: {align['tok_mismatch']}/{align['n']}   != dump prompt_len: {align['dump_mismatch']}/{align['n']}")
print(f"fact span does not decode to the answer: {align['fact_bad']}/{align['fact_chk']}   msg span lacks source-message text: {align['msg_bad']}/{align['msg_chk']}")
if deltas:
    ds = sorted(abs(x) for x in deltas)
    print(f"|server - local| token drift over the {len(ds)} mismatched probes: median {ds[len(ds) // 2]}, p90 {ds[int(len(ds) * .9)]}, max {ds[-1]} "
          f"(block = {BT} tokens); server count is larger in {sum(x > 0 for x in deltas)}, smaller in {sum(x < 0 for x in deltas)}")
    print(f"upper bound on probes whose fact span could straddle a block edge because of the drift: {risky[0]}/{align['n']} = {100 * risky[0] / align['n']:.1f}%")

# reproduction check against rank4_cal.out
ref = {}
for l in open(AG + "/rlru4/rank4_cal.out"):
    m = re.match(r"\s*(\d+) \|\s+(\d+) \|\s+(\d+)\s+(\d+) \|\s+(\d+) \|\s+(\d+) \| (\d+)/(\d+)/(\d+)", l)
    if m:
        ref[int(m.group(1))] = tuple(int(x) for x in m.groups()[2:3] + m.groups()[4:6] + m.groups()[6:])
print("\n== exact vs rank4_cal.out (within, beyond, total, whole/part/miss) ==")
ok = True
for k in (2, 4, 6, 8, 10):
    a = acc[("exact", k)]
    mine = (round(a.est["within"]), round(a.est["beyond"]), round(a.est["within"] + a.est["beyond"]), *(round(c) for c in a.cnt))
    same = (not LIMIT and not ALIGNED) and mine == ref[k]
    ok &= same
    print(f"k={k}: mine {mine}  ref {ref[k]}  {'MATCH' if same else 'DIFF'}")
print("exact reproduces rank4_cal:", ok if not (LIMIT or ALIGNED) else "n/a (subset run)")

LABEL = {"exact": "exact mass_end (today)", "mass_r": "mass_r", "recency": "recency", "random": "random (mean of 20 seeds)",
         "bm25_tok": "block BM25, token ids uni+bi", "bm25_txt": "block BM25, text words", "msg_bm25": "message BM25 -> blocks",
         "rrf_tok": "RRF exact+BM25(tok)", "rrf_txt": "RRF exact+BM25(txt)", "pre_tok": "prefilter BM25(tok) top16 -> exact",
         "pre_txt": "prefilter BM25(txt) top16 -> exact", "oracle": "oracle (fact block always picked)"}
print("\n== STEP 3a: recall of the fact block / of any block of the source message, among probes whose fact is outside the window ==")
print("(n = engaged probes within/beyond; adj = within + 0.83*beyond, the same cut for every router; measured engine k=2: 766, rank4_cal k=2: 770)")
for k in KS:
    a0 = acc[("exact", k)]
    print(f"\n-- k={k} blocks ({k * BT} tokens), engaged probes: within {a0.n['within']:.0f}, beyond {a0.n['beyond']:.0f}")
    print(f"{'router':<38}|{'fact w':>7}{'fact b':>7}{'fact all':>9}|{'msg w':>7}{'msg b':>7}{'msg all':>8}|{'within':>7}{'beyond':>7}{'total':>7}{'adj':>7}|whole/part/miss")
    for n in R_NAMES:
        a = acc[(n, k)]
        nw, nbd = a.n["within"], a.n["beyond"]
        pc = lambda x, y: 100.0 * x / y if y else float("nan")
        tot = a.est["within"] + a.est["beyond"]
        adj = a.est["within"] + BEYOND_CUT * a.est["beyond"]
        print(f"{LABEL[n]:<38}|{pc(a.fact['within'], nw):6.1f}%{pc(a.fact['beyond'], nbd):6.1f}%{pc(a.fact['within'] + a.fact['beyond'], nw + nbd):8.1f}%|"
              f"{pc(a.msg['within'], nw):6.1f}%{pc(a.msg['beyond'], nbd):6.1f}%{pc(a.msg['within'] + a.msg['beyond'], nw + nbd):7.1f}%|"
              f"{a.est['within']:7.0f}{a.est['beyond']:7.0f}{tot:7.0f}{adj:7.0f}|{a.cnt[0]:.0f}/{a.cnt[1]:.0f}/{a.cnt[2]:.0f}")

print("\n== paired fact-block hits vs exact among engaged probes (wins = router hit & exact missed; losses = reverse) ==")
for k in (2, 4, 8):
    for n in ("bm25_tok", "bm25_txt", "rrf_tok", "rrf_txt", "pre_tok", "pre_txt"):
        h, e = hits[(n, k)], hits[("exact", k)]
        wi = sum(h[i] and not e[i] for i in h)
        lo = sum(e[i] and not h[i] for i in h)
        print(f"k={k} {LABEL[n]:<38} wins {wi:3d}  losses {lo:3d}  net {wi - lo:+d} of {len(h)} engaged")

print("\n== lexical-leak caveat ==")
qn = sum(v[0] for v in leak.values())
nn = sum(v[1] for v in leak.values())
print(f"questions quoting verbatim an identifier/path (>=6 chars, has / _ . digit or camelCase) found in the source message: {qn}/{nn} = {100 * qn / nn:.1f}%"
      f"  (within {100 * leak['within'][0] / max(1, leak['within'][1]):.1f}%, beyond {100 * leak['beyond'][0] / max(1, leak['beyond'][1]):.1f}%)")

print("\n== BM25 index cost (token-id unigrams+bigrams, one posting per distinct term per 640-token block) ==")
if tim["n"]:
    kt = tim["tokens"] / 1000
    print(f"{tim['n']} prompts, {tim['tokens']} tokens: {tim['postings'] / kt:.0f} postings per 1K tokens "
          f"(~{tim['postings'] / kt * 6 / 1024:.1f} KiB per 1K tokens at 6 bytes/posting: u32 block + u16 tf), {tim['dict_terms'] / kt:.0f} distinct terms per 1K tokens")
    print(f"pure-python build: {tim['build'] / kt * 1000:.0f} ms per 1K tokens ({tim['build'] / tim['n']:.2f} s per prompt avg);"
          f" query: {tim['query'] / tim['n'] * 1000:.1f} ms per probe (pure python; a C++ version is ~100x faster)")
