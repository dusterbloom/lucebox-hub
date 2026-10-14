"""Deterministic recall-probe extraction, answer normalization, and tool-call keys. Pure functions, no IO.

A probe asks for a fact that appears ONLY in one tool output before the cut (no other message, assistant call,
user turn or tool schema contains the answer string). Types: defname, line, traceback, failed_test,
cmdline_path, cmdline_text."""
import hashlib
import json
import random
import re
import shlex

TYPES = ["defname", "line", "traceback", "failed_test", "cmdline_path", "cmdline_text"]
META_RE = re.compile(r"\n?\[The command completed|\n?\[Current working directory:")
VIEW_RE = re.compile(r"^\s*(\d+)\t(.*)$")
DEF_RE = re.compile(r"^(\s*)(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
ERR_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)\b.*$")
FAILED_RE = re.compile(r"^FAILED\s+(\S+::\S+)")
PATH_RE = re.compile(r"^(?:\./|/)?[\w.\-]+(?:/[\w.\-]+)+(?::\d+)?")
MIN_DIST_TOK = 1500
ANSWER_SUFFIX = " Reply with the answer only: no explanation, no quotes, and do not call any tool."


def bash_body(content):
    m = META_RE.search(content)
    return content[: m.start()] if m else content


def normalize(s):
    """Whitespace collapse, strip, one trailing period, one pair of surrounding quotes/backticks. Nothing else."""
    s = " ".join((s or "").split())
    if s.endswith(".") and not s[:-1].endswith("."):
        s = s[:-1]
    for q in ("`", '"', "'"):
        if len(s) >= 2 and s[0] == q and s[-1] == q:
            s = s[1:-1].strip()
            break
    return s


def score_exact(answer, gold):
    a, g = normalize(answer), normalize(gold)
    if g.endswith("."):
        return a.rstrip(".") == g.rstrip(".")
    return a == g


def _corpus(messages, tools):
    parts = [json.dumps(tools)]
    for m in messages:
        parts.append("\n".join([m.get("content") or ""] + [tc["function"]["arguments"] for tc in m.get("tool_calls") or []]))
    return parts  # per message (plus tools first); index i+1 <-> messages[i]


def _src_tok(boundaries, idx):
    """Token position of tool message idx ~ boundary of the nearest preceding assistant message."""
    for j in range(idx - 1, -1, -1):
        if str(j) in boundaries:
            return boundaries[str(j)]
    return 0


def extract_candidates(messages, tools, boundaries, cut_tok):
    """All eligible probes for the prefix messages (ending at a tool result). boundaries: {str(idx): tokens}."""
    parts = _corpus(messages, tools)
    calls = {}
    cmd_count, view_count, edit_paths = {}, {}, set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            a = json.loads(tc["function"]["arguments"])
            calls[tc["id"]] = (tc["function"]["name"], a)
            if tc["function"]["name"] == "execute_bash":
                c = a.get("command", "")
                cmd_count[c] = cmd_count.get(c, 0) + 1
            elif tc["function"]["name"] == "str_replace_editor":
                p = a.get("path")
                if a.get("command") == "view":
                    view_count[p] = view_count.get(p, 0) + 1
                else:
                    edit_paths.add(p)

    def only_here(ans, i, in_src_once):
        if not ans or any(ans in parts[k] for k in range(len(parts)) if k != i + 1):
            return False
        return parts[i + 1].count(ans) == 1 if in_src_once else True

    out = []
    for i, m in enumerate(messages):
        if m["role"] != "tool" or m["tool_call_id"] not in calls:
            continue
        st = _src_tok(boundaries, i)
        if cut_tok - st < MIN_DIST_TOK or i < 4:
            continue
        name, a = calls[m["tool_call_id"]]
        content = m["content"]
        base = {"msg_index": i, "src_tok": st, "distance_tok": cut_tok - st}
        if name == "str_replace_editor" and a.get("command") == "view":
            path = a.get("path")
            if view_count.get(path) != 1 or path in edit_paths or "cat -n" not in content[:200]:
                continue
            lines = [VIEW_RE.match(l) for l in content.split("\n")]
            lines = [(int(x.group(1)), x.group(2)) for x in lines if x]
            for n, txt in lines:
                dm = DEF_RE.match(txt)
                if dm:
                    ans = dm.group(2)
                    q = f"In your earlier view of the file {path}, what is the name of the function or class defined on line {n}?"
                    if ans not in q and only_here(ans, i, False):
                        out.append({**base, "type": "defname", "question": q + ANSWER_SUFFIX, "answer": ans})
                s = txt.strip()
                if 20 <= len(s) <= 110 and not s.startswith("#") and only_here(s, i, True):
                    q = (f"In your earlier view of the file {path}, what is the exact text of line {n} "
                         "(without the line number, leading indentation not required)?")
                    out.append({**base, "type": "line", "question": q + ANSWER_SUFFIX, "answer": s})
        elif name == "execute_bash":
            cmd = a.get("command", "")
            if cmd_count.get(cmd) != 1 or len(cmd) > 200 or "\n" in cmd:
                continue
            body = bash_body(content).strip("\n")
            lines = body.split("\n")
            for t, l in enumerate(lines):
                if l.startswith("Traceback (most recent call last)"):
                    j = t + 1
                    while j < len(lines) and (lines[j].startswith(" ") or lines[j].startswith("\t")):
                        j += 1
                    if j < len(lines) and ERR_RE.match(lines[j]) and len(lines[j]) <= 160:
                        ans = lines[j].strip()
                        q = (f"The command `{cmd}` that you ran earlier printed a Python traceback. "
                             "What was the exact final line (the error line) of the first traceback in its output?")
                        if ans not in q and only_here(ans, i, True):
                            out.append({**base, "type": "traceback", "question": q + ANSWER_SUFFIX, "answer": ans})
                    break
            for l in lines:
                fm = FAILED_RE.match(l)
                if fm:
                    ans = fm.group(1)
                    q = (f"In the output of the command `{cmd}` that you ran earlier, which test was reported as FAILED "
                         "first? Give the test id (path::name).")
                    if only_here(ans, i, False) and parts[i + 1].count(ans) <= 3:
                        out.append({**base, "type": "failed_test", "question": q + ANSWER_SUFFIX, "answer": ans})
                    break
            nonempty = [l for l in lines if l.strip()]
            for k in range(1, min(len(nonempty), 12) + 1):
                s = nonempty[k - 1].strip()
                if 8 <= len(s) <= 120 and only_here(s, i, True):
                    q = (f"The command `{cmd}` was run earlier. What was line {k} of its output "
                         "(counting non-empty lines from 1, ignoring the trailing exit-code notes)?")
                    typ = "cmdline_path" if PATH_RE.match(s) else "cmdline_text"
                    out.append({**base, "type": typ, "question": q + ANSWER_SUFFIX, "answer": s})
    return out


def sample_probes(cands, seed, k):
    """Up to k probes, round-robin over types, seeded shuffle inside each type."""
    rng = random.Random(seed)
    buckets = {t: [c for c in cands if c["type"] == t] for t in TYPES}
    for b in buckets.values():
        rng.shuffle(b)
    out = []
    while len(out) < k and any(buckets.values()):
        for t in TYPES:
            if buckets[t] and len(out) < k:
                out.append(buckets[t].pop())
    return out


def call_key(name, args):
    """(name_key, full_key): tool name; tool name + key argument (see AGENTIC_EVAL_PROPOSAL.md metric c)."""
    if name == "execute_bash":
        cmd = (args.get("command") or "").strip()
        try:
            toks = shlex.split(cmd)
        except ValueError:
            toks = cmd.split()
        if not toks:
            return name, name + "|"
        arg = next((t for t in toks[1:] if not t.startswith("-")), "")
        return name, f"{name}|{toks[0]}|{arg}"
    if name == "str_replace_editor":
        c, p = args.get("command"), args.get("path")
        extra = ""
        if c == "str_replace":
            extra = hashlib.md5((args.get("old_str") or "")[:60].encode()).hexdigest()[:8]
        return name, f"{name}|{c}|{p}|{extra}"
    return name, name


def parse_response_call(message):
    """First tool call of a chat completion message -> (name_key, full_key) or None when absent/malformed."""
    tcs = message.get("tool_calls") or []
    if not tcs:
        return None
    try:
        f = tcs[0]["function"]
        args = json.loads(f["arguments"]) if isinstance(f["arguments"], str) else f["arguments"]
        if not isinstance(args, dict) or not f.get("name"):
            return None
        return call_key(f["name"], args)
    except Exception:
        return None
