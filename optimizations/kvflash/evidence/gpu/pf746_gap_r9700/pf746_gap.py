"""PFlash 2.0 (#746) x KVFlash 2.0a admission-gap chat client (run on lucebox6 with python3 -I).

  python3 -I pf746_gap.py ARM        ARM in A | B | C  (see run_arm.sh, which sets the env and takes the lock)

The same script and the same prompts drive every arm. A 5-turn chat, no tools, temperature 0, seed 1.
Each user turn is one new distinct QuALITY article paste plus one question about it; every assistant reply
is appended verbatim to that arm's own history. The server is launched per run with identical flags on
every arm except --prefill-compression (auto for A and B, off for C) and the binary (A: merge, B/C: merge+fix).

Env (all required unless a default is shown):
  KV_BIN, KV_ROOT (output dir), KV_MODEL, KV_DRAFTER
  KV_POOL (24576)  KV_MAX_CTX (65536)  KV_PORT (18292)  KV_TURNS (5)  KV_MAX_TOKENS (64)
  PASTE_CHARS (28000)  max article characters per paste, one value or a comma list per turn (~4.2 chars/token)
  PASTE_IDS ()         comma list of QuALITY article_id, one per turn (default: first eligible by article_id)
  PASTE_SKIP (0)       number of eligible articles to skip (picks a different deterministic set)
"""
import json, os, re, signal, subprocess, sys, time, urllib.request

ARM = sys.argv[1]
assert ARM in ("A", "B", "C")
ROOT = os.environ["KV_ROOT"]
BIN = os.environ["KV_BIN"]
MODEL = os.environ["KV_MODEL"]
DRAFTER = os.environ["KV_DRAFTER"]
POOL = os.environ.get("KV_POOL", "24576")
MAX_CTX = os.environ.get("KV_MAX_CTX", "65536")
PORT = int(os.environ.get("KV_PORT", "18292"))
TURNS = int(os.environ.get("KV_TURNS", "5"))
MAX_TOKENS = int(os.environ.get("KV_MAX_TOKENS", "64"))
PASTE_CHARS = os.environ.get("PASTE_CHARS", "28000")
PASTE_IDS = os.environ.get("PASTE_IDS", "")
PASTE_SKIP = int(os.environ.get("PASTE_SKIP", "0"))
URL = f"http://127.0.0.1:{PORT}"
DATA = os.path.expanduser("~/r1e/code/repo/data/QuALITY.v1.0.1.htmlstripped.dev")  # read-only
SYSTEM = {"role": "system",
          "content": "You are a careful assistant who reads long articles. Keep every answer short (two sentences)."}

# PFLASH_* env that every arm gets (no trained scoring head: built-in scorer; strict selection is the only
# path that reaches the cache-admission code in HttpServer::apply_pflash_compression).
PFLASH_ENV = {"PFLASH_ADMISSION": "cache", "PFLASH_SELECT_MODE": "budget_only", "PFLASH_SELECT_SCORER": "legacy"}


def pastes():
    """TURNS distinct articles, in order. PASTE_IDS (comma list of article_id) picks them explicitly; otherwise the
    first eligible ones by article_id. PASTE_CHARS (one value or a comma list per turn) caps the article length; a longer
    article is cut at a paragraph break, a shorter one is used whole."""
    rows = {}
    for l in open(DATA):
        r = json.loads(l)
        rows.setdefault(str(r["article_id"]), r)
    caps = [int(c) for c in PASTE_CHARS.split(",")]
    caps = caps * TURNS if len(caps) == 1 else caps
    if PASTE_IDS:
        out = [rows[i] for i in PASTE_IDS.split(",")]
    else:
        out = [r for _, r in sorted(rows.items()) if len(r["article"]) >= min(caps)][PASTE_SKIP:PASTE_SKIP + TURNS]
    assert len(out) == TURNS and len({r["article_id"] for r in out}) == TURNS, "need TURNS distinct articles"
    res = []
    for r, cap in zip(out, caps):
        a = r["article"]
        if len(a) > cap:
            cut = a.rfind("\n\n", 0, cap)
            if cut < cap // 2:
                cut = a.rfind(". ", 0, cap) + 1
            a = a[:cut]
        q = r["questions"][0]
        opts = "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(q["options"]))
        res.append((r["article_id"], r["title"], a.strip(), q["question"], opts))
    return res


def user_message(t, art):
    aid, title, text, q, opts = art
    return (f"Article {t} (\"{title}\"):\n\n{text}\n\n"
            f"Question about Article {t}: {q}\n{opts}\nAnswer with the letter and one short reason.")


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(messages, max_tokens):
    return post("/v1/chat/completions", {"model": "luce", "messages": messages, "temperature": 0, "seed": 1,
                                         "max_tokens": max_tokens, "stream": False})


def server_args():
    return [BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--max-ctx", MAX_CTX, "--kvflash", POOL,
            "--model-name", "luce", "--prefill-drafter", DRAFTER,
            "--prefill-compression", "off" if ARM == "C" else "auto", "--prefill-threshold", "16384",
            "--prefill-skip-park", "--no-kv-cache"]  # no disk KV cache: no state may cross runs


def server_env():
    e = dict(os.environ)
    e["TMPDIR"] = os.path.expanduser("~/pf746/tmp")
    e["HIP_VISIBLE_DEVICES"] = "0"  # R9700 gfx1201; GPU1 is the iGPU and belongs to another experiment
    for k in list(e):
        if k.startswith(("KVFLASH_", "PFLASH_", "LUCE_")):
            del e[k]
    e.update(PFLASH_ENV)
    return e


LOG_KEYS = ("[pflash", "[snap]", "snapshot skipped", "[kvflash]", "chat CACHE", "chat DONE", "[agent-turn-cache]",
            "[server] prefix", "prefix")


def main():
    log = f"{ROOT}/server_{ARM}.log"
    json.dump({"argv": server_args(), "env": {k: v for k, v in server_env().items()
                                              if k.startswith(("PFLASH_", "HIP_", "TMPDIR"))}},
              open(f"{ROOT}/cmdline_{ARM}.json", "w"), indent=1)
    arts = pastes()
    f = open(log, "w")
    p = subprocess.Popen(server_args(), stdout=f, stderr=subprocess.STDOUT, env=server_env(), stdin=subprocess.DEVNULL)
    try:
        for _ in range(600):
            if p.poll() is not None:
                raise RuntimeError("server died; see " + log)
            try:
                urllib.request.urlopen(URL + "/v1/models", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server start timeout")
        # warm-up, as in the gate scripts: a distinct short prompt, not part of the chat
        chat([{"role": "user", "content": "Warm-up. " + "alpha bravo charlie delta " * 150}], 4)
        time.sleep(1.0)
        rec, msgs = [], [SYSTEM]
        for t in range(1, TURNS + 1):
            msgs = msgs + [{"role": "user", "content": user_message(t, arts[t - 1])}]
            pos = os.path.getsize(log)
            t0 = time.time()
            resp = chat(msgs, MAX_TOKENS)
            wall = time.time() - t0
            time.sleep(1.0)
            with open(log, errors="replace") as lf:
                lf.seek(pos)
                seg = lf.read().splitlines()
            json.dump(resp, open(f"{ROOT}/resp_{ARM}_t{t}.json", "w"), indent=1)
            open(f"{ROOT}/log_{ARM}_t{t}.txt", "w").write("\n".join(seg) + "\n")
            answer = resp["choices"][0]["message"].get("content") or ""
            u = resp.get("usage", {})
            tm = u.get("timings", {})
            row = dict(turn=t, article_id=arts[t - 1][0], prompt_tokens=u.get("prompt_tokens"),
                       completion_tokens=u.get("completion_tokens"), timings=tm, wall_s=wall, answer=answer,
                       finish_reason=resp["choices"][0].get("finish_reason"),
                       log_lines=[l for l in seg if any(k in l for k in LOG_KEYS)])
            rec.append(row)
            print(json.dumps({k: row[k] for k in ("turn", "prompt_tokens", "completion_tokens", "wall_s")} |
                             {"pflash": tm.get("pflash"), "cached": tm.get("cached_prefix_tokens"),
                              "prefilled": tm.get("prefilled_tokens"), "eff": tm.get("effective_prompt_tokens"),
                              "prefill_ms": tm.get("prefill_ms"), "tps": tm.get("decode_tokens_per_sec")}),
                  flush=True)
            msgs = msgs + [{"role": "assistant", "content": answer}]
        json.dump(dict(arm=ARM, pool=POOL, max_ctx=MAX_CTX, paste_chars=PASTE_CHARS, paste_ids=PASTE_IDS, paste_skip=PASTE_SKIP, turns=rec),
                  open(f"{ROOT}/turns_{ARM}.json", "w"), indent=1)
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(60)
            except Exception:
                p.kill()
                p.wait()
        f.close()


main()
