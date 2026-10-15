#!/usr/bin/env python3
"""KVFlash 2.0b M7 gate: QuALITY selection split through luce_server (chat endpoint).

One arm per invocation: --arm orig (compaction off) or --arm auto (--kvflash-am-ratio 0.10, P=W=0).
Prompts are the reference's: format_context (system + article) then format_question (enable_thinking
False, Qwen model_name), sent as chat messages; temperature 0, max_tokens 256 (N_NEW).
In the auto arm each request carries kvflash_am_exact_tail = the question's token count, so the
blocks the engine compacts are exactly the reference's sealed context blocks (the tail stays exact).

Rows are appended to --out as they finish (resumable), one JSON object per question.

Env: KV_BIN (luce_server), KV_MODEL (gguf), KV_ROOT (log dir), KV_PORT, TMPDIR, AM_REPO, TOK_JSON.
Run as `python3 run_quality.py ...` (not -I: it imports the reference repo for the parser).
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request

AM_REPO = os.environ.get("AM_REPO", "/mnt/kimi-k3/repo/am-compaction")
TOK_JSON = os.environ.get("TOK_JSON", "/mnt/kimi-k3/build/a1_stats/tok38/tokenizer.json")
MODEL_NAME = "Qwen/Qwen3.8-27B"
N_NEW = 256
SYSTEM = "You are a helpful assistant. Answer a question based on the provided context."
PORT = int(os.environ.get("KV_PORT", "18090"))
URL = f"http://127.0.0.1:{PORT}"

sys.path.insert(0, AM_REPO)
os.chdir(AM_REPO)
from evaluation.datasets import load_dataset  # noqa: E402
from evaluation.utils import format_question, parse_model_choice  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


class QwenTemplate:
    """The Qwen chat template for the two renderings the reference makes (no jinja needed)."""
    last_content = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        self.last_content = messages[-1]["content"]
        return out


def start_server(extra, log):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("LUCE_KVFLASH_"):
            del env[k]
    args = [os.environ["KV_BIN"], os.environ["KV_MODEL"], "--host", "127.0.0.1", "--port", str(PORT),
            "--max-ctx", os.environ.get("KV_MAX_CTX", "131072"), "--model-name", "luce",
            "--no-kv-cache", "--prefix-cache-slots", "0", "--kvflash", os.environ.get("KV_POOL", "8192")] + extra
    f = open(log, "w")
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
    for _ in range(300):
        if p.poll() is not None:
            raise RuntimeError("server died; see " + log)
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=2)
            return p, f
        except Exception:
            time.sleep(2)
    raise RuntimeError("server start timeout")


def chat(messages, extra_body):
    body = {"model": "luce", "messages": messages, "temperature": 0, "seed": 1, "max_tokens": N_NEW,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra_body)
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, ttft, text, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                delta = c.get("delta", {}).get("content")
                if delta:
                    ttft = ttft or time.time() - t0
                    text.append(delta)
    return "".join(text), usage, ttft, time.time() - t0


RX_CACHE = re.compile(r"chat CACHE \S+ restore=(\w+) slot=(-?\d+) prefix_len=(\d+) effective_prompt=(\d+).*disk_hit=(\w+)")
RX_DONE = re.compile(r"chat DONE .* in=(\d+) .* out=(\d+) .*prefill=([\d.]+)s decode=([\d.]+)s\(([\d.]+)tok/s\)")
RX_AM = re.compile(r"\[kvflash-am\] prefill: sealed=(\d+) compacted=(\d+) skipped=(\d+) failed=(\d+) select=([\d.]+)ms "
                   r"fit=([\d.]+)ms io=([\d.]+)ms total=([\d.]+)ms per_block=([\d.]+)ms nan_layers=\[([\d,]*)\]")


def log_facts(chunk):
    """Facts the server logged for one request: caching state, its own timings, the compaction summary."""
    f = {}
    m = RX_CACHE.search(chunk)
    if m:
        f.update(restore=m.group(1) == "true", prefix_len=int(m.group(3)), effective_prompt=int(m.group(4)),
                 disk_hit=m.group(5) == "true")
    m = RX_DONE.search(chunk)
    if m:
        f.update(in_tokens=int(m.group(1)), out_tokens=int(m.group(2)), prefill_s=float(m.group(3)),
                 decode_s=float(m.group(4)), decode_tps=float(m.group(5)))
    m = RX_AM.search(chunk)
    if m:
        g = m.groups()
        f.update(am_sealed=int(g[0]), am_compacted=int(g[1]), am_skipped=int(g[2]), am_failed=int(g[3]),
                 am_select_ms=float(g[4]), am_fit_ms=float(g[5]), am_io_ms=float(g[6]), am_total_ms=float(g[7]),
                 am_per_block_ms=float(g[8]), nan_layers=[int(x) for x in g[9].split(",") if x])
    elif "nan_layers" not in f:
        f["nan_layers"] = []
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["orig", "auto"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="stop after N questions (smoke)")
    ap.add_argument("--prefix", default="0")
    ap.add_argument("--recent", default="0")
    a = ap.parse_args()

    data = load_dataset("quality")[: a.articles]
    tok = Tokenizer.from_file(TOK_JSON)
    tmpl = QwenTemplate()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            done.add((r["article_index"], r["qi"]))
    extra = ["--kvflash-am-ratio", "0.10", "--kvflash-am-prefix-blocks", a.prefix,
             "--kvflash-am-recent-blocks", a.recent] if a.arm == "auto" else []
    log = os.path.join(os.environ["KV_ROOT"], f"server_{a.arm}.log")
    p, lf = start_server(extra, log)
    n = 0
    try:
        with open(a.out, "a") as out:
            for ai, art in enumerate(data):
                aid = art["article_id"]
                ctx_str = tmpl.apply_chat_template([{"role": "system", "content": SYSTEM},
                                                    {"role": "user", "content": art["article"]}])
                ctx_len = len(tok.encode(ctx_str, add_special_tokens=False).ids)
                for qi, q in enumerate(art["questions"]):
                    if (ai, qi) in done:
                        continue
                    q_str = format_question(tmpl, q["question"], q["options"], MODEL_NAME, enable_thinking=False)
                    content = tmpl.last_content
                    tail = len(tok.encode(q_str, add_special_tokens=False).ids)
                    hf_total = len(tok.encode(ctx_str + q_str, add_special_tokens=False).ids)
                    off = os.path.getsize(log)
                    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": art["article"]},
                            {"role": "user", "content": content}]
                    text, usage, ttft, wall = chat(msgs, {"kvflash_am_exact_tail": tail} if a.arm == "auto" else {})
                    time.sleep(0.3)
                    with open(log, errors="replace") as lg:
                        lg.seek(off)
                        facts = log_facts(lg.read())
                    ch = parse_model_choice(text)
                    row = {"arm": a.arm, "article_id": aid, "article_index": ai, "qi": qi, "gold": q["gold_label"],
                           "parsed": ch, "correct": bool(ch and ch == q["gold_label"]), "answer": text.strip()[:300],
                           "ctx_len": ctx_len, "tail_tokens": tail, "hf_prompt_len": hf_total,
                           "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                           "ttft_s": ttft, "wall_s": wall, "finite": True}
                    row.update(facts)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    n += 1
                    print(f"[{a.arm}] a{ai} q{qi} gold={q['gold_label']} parsed={ch} wall={wall:.1f}s "
                          f"prompt={row['prompt_tokens']} hf={hf_total} prefix_len={facts.get('prefix_len')}", flush=True)
                    if a.limit and n >= a.limit:
                        return
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(30)
            except Exception:
                p.kill()
                p.wait()
        lf.close()


if __name__ == "__main__":
    main()
