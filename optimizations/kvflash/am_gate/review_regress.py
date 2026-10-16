"""Engine-level regressions for the two Codex reviews of M7 (GPU, run under the 3090 lock; python3 -I).

  hook      an injected allocation failure in the first seal's scratch init leaves that block exact, and a
            later seal still compacts (LUCE_KVFLASH_AM_TEST_FAIL_SCRATCH=1)
  draft     --draft with a compacted state decodes autoregressively to completion; the text equals the same
            run without a draft
  oversize  a disk snapshot written by a larger --kvflash pool is refused at adoption by a smaller pool with compaction on:
            no crash, same text as a fresh prefill
  bypass    a fitting snapshot with a tail to prefill, compaction on: counted as a miss before any restore write,
            same text as a fresh prefill
            with compaction on, counts as a miss: no crash, same text as a fresh prefill

usage: KV_ROOT=... KV_BIN=... KV_MODEL=... TMPDIR=... python3 -I review_regress.py <scenario>"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlib import *

DRAFT = "/mnt/kimi-k3/home-models/qwen3.6-27b-dflash/dflash-draft-3.6-q8_0.gguf"
Q = "\nQuestion: continue this list with the next 12 records in exactly the same format and write nothing else."
AM = ["--kvflash", "8192", "--kvflash-am-ratio", "0.10", "--no-kv-cache"]
fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails.append(what)


def run_one(name, extra, prompt, env=None, max_tokens=48):
    log = f"{ROOT}/{name}.log"
    try:
        with Server(log, env=env or {}, extra=extra):
            o = chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
    except Exception as e:  # server died or request failed
        print(f"{name}: {type(e).__name__}: {e}", flush=True)
        return None, log
    return o, log


def hook():
    p = filler(7000, salt=21) + Q
    o, log = run_one("hook", AM, p, {"LUCE_KVFLASH_AM_TEST_FAIL_SCRATCH": "1"})
    check(o is not None and len(o["text"]) > 0, "request completes")
    check(bool(grep_log(log, r"alloc failure injected")), "the injected failure fired")
    check(bool(grep_log(log, r"block 0 stays exact")), "block 0 stays exact")
    s = grep_log(log, r"\[kvflash-am\] prefill: sealed=(\d+) compacted=(\d+) skipped=\d+ failed=1 ")
    check(bool(s), "exactly one failed seal")
    if s:
        import re
        m = re.search(r"sealed=(\d+) compacted=(\d+)", s[0])
        check(int(m.group(1)) >= 5 and int(m.group(2)) == int(m.group(1)) - 1, "every later seal still compacts")


def draft():
    p = filler(7000, salt=22) + Q
    a, la = run_one("draft_on", AM + ["--draft", DRAFT], p, max_tokens=64)
    b, lb = run_one("draft_off", AM, p, max_tokens=64)
    check(a is not None and len(a["text"]) > 0, "draft + compaction request completes")
    check(b is not None and len(b["text"]) > 0, "no-draft reference completes")
    check(bool(grep_log(la, r"\[kvflash-am\] block \d+ compacted")), "blocks compacted with the draft configured")
    if a and b:
        check(a["text"] == b["text"], "draft run text equals the autoregressive run")


def snapshot_case(name, writer_pool, prompt):
    d = f"{ROOT}/kvdir"
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    a, la = run_one(name + "_writer", ["--kvflash", str(writer_pool), "--kv-cache-dir", d], prompt, max_tokens=8)
    check(a is not None, f"writer server (pool {writer_pool}) completes")
    files = [f for _, _, fs in os.walk(d) for f in fs]
    check(len(files) > 0, f"the writer left a disk snapshot ({len(files)} files)")
    b, lb = run_one(name + "_am", ["--kvflash", "8192", "--kvflash-am-ratio", "0.10", "--kv-cache-dir", d], prompt)
    check(b is not None and len(b["text"]) > 0, "small pool + compaction completes (no crash)")
    c, lc = run_one(name + "_fresh", AM, prompt)
    check(c is not None and len(c["text"]) > 0, "fresh-prefill reference completes")
    if b and c:
        check(b["text"] == c["text"], "answer equals a fresh prefill")
    return lb


def oversize():
    lb = snapshot_case("oversize", 16384, filler(12000, salt=23) + Q)
    check(bool(grep_log(lb, r"snapshot adopt rejected")), "the oversized snapshot was refused at adoption (a miss)")
    check(not grep_log(lb, r"disk_hit=true"), "no hit was taken")


def bypass():
    lb = snapshot_case("bypass", 8192, filler(6000, salt=24) + Q)
    check(bool(grep_log(lb, r"disk_hit=true")), "the snapshot was found (not a vacuous run)")
    check(bool(grep_log(lb, r"restore .*bypassed")), "the restore was bypassed as a miss, before any restore write")


{"hook": hook, "draft": draft, "oversize": oversize, "bypass": bypass}[sys.argv[1]]()
print("RESULT", "FAIL" if fails else "PASS", sys.argv[1], flush=True)
sys.exit(1 if fails else 0)
