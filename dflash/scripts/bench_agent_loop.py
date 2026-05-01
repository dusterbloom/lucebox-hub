"""B.6: agent-loop bench using real Claude Code session transcripts.

Faithfully replays a recorded Claude Code session against the dflash
server: at each assistant turn, sends the exact OpenAI-format message
prefix that was originally sent at that point (system + every preceding
user/assistant/tool turn, tool I/O included), measures TTFT + total wall,
then advances state with the recorded assistant turn (NOT a bench-
synthesized one).

Compares cold (--prefix-cache-slots=0) vs warm (--prefix-cache-slots=N).

Usage:
    python3 dflash/scripts/bench_agent_loop.py [--turns N] [--session PATH]

Default session = most recent JSONL under ~/.claude/projects/<workspace>,
where <workspace> is the cwd with `/` replaced by `-`.

Why faithful replay (not synthesised assistant replies)?
    Real agentic sessions accumulate tool results turn-over-turn — typical
    prefix grows from ~5K chars at turn 1 to 60-300K by turn 30, dominated
    by tool_result blocks. A loader that drops tool I/O understates the
    prefix-cache workload by 1-2 orders of magnitude and produces "within
    noise" cold-vs-warm numbers on real sessions even though the cache
    genuinely helps in production.
"""
import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent.parent
TARGET        = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT         = Path.home() / "models/qwen3.6-27b-dflash"
BIN           = ROOT / "dflash/build/test_dflash"
SERVER_SCRIPT = ROOT / "dflash/scripts/server_tools.py"


def _default_session_dir() -> Path:
    """~/.claude/projects/<cwd-with-slashes-as-dashes>."""
    workspace = str(Path.cwd().resolve()).replace("/", "-")
    return Path.home() / ".claude/projects" / workspace


# ── Transcript loader: Anthropic Messages JSONL → OpenAI message array ─
#
# Claude Code stores per-session transcripts at
#   ~/.claude/projects/<workspace>/<session_uuid>.jsonl
#
# Each line is one event in Anthropic Messages format. A real LLM "turn"
# (one /v1/messages API call's response) can span multiple jsonl records:
# typically a `thinking` block, a `text` block, and one or more `tool_use`
# blocks all share a single API response but get serialised as separate
# rows. Same for user turns: each `tool_result` is its own row.

def _load_transcript(path: Path) -> list[dict]:
    """Parse JSONL into ordered (role, blocks) turns, coalescing same-role runs."""
    turns: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") not in ("user", "assistant"):
                continue
            msg = rec.get("message") or {}
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, list):
                blocks = content
            elif isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            else:
                continue
            if turns and turns[-1]["role"] == role:
                turns[-1]["blocks"].extend(blocks)
            else:
                turns.append({"role": role, "blocks": blocks})
    return turns


SYSTEM_PROMPT = (
    "You are a precise coding assistant for the lucebox-hub repo. Answer concisely."
)


def _to_openai_messages(turns: list[dict]) -> list[dict]:
    """Convert Anthropic-format turns → OpenAI messages array.

    Emits proper structured tool messages (the OpenAI-on-the-wire shape):
      tool_use   → assistant.tool_calls[].function.{name,arguments}
      tool_result → role="tool" message with tool_call_id
      text       → user/assistant content
      thinking   → dropped (no OpenAI equivalent)

    Targets `dflash/scripts/server_tools.py`, whose ChatRequest schema
    accepts `content: Any | None`, `tool_calls`, and `tool_call_id` —
    i.e. the real production tool path the daemon uses for agent CLIs.
    """
    out: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in turns:
        role = turn["role"]
        blocks = turn["blocks"]
        if role == "user":
            text_parts: list[str] = []
            for blk in blocks:
                t = blk.get("type")
                if t == "text":
                    text_parts.append(blk.get("text") or "")
                elif t == "tool_result":
                    tc_id = blk.get("tool_use_id") or ""
                    raw = blk.get("content")
                    if isinstance(raw, list):
                        body = "".join(
                            c.get("text", "") for c in raw
                            if isinstance(c, dict) and c.get("type") == "text"
                        )
                    else:
                        body = str(raw) if raw else ""
                    out.append({"role": "tool", "tool_call_id": tc_id,
                                "content": body})
            if text_parts:
                out.append({"role": "user",
                            "content": "\n".join(text_parts)})
        else:  # assistant
            text_parts = []
            tool_calls: list[dict] = []
            for blk in blocks:
                t = blk.get("type")
                if t == "text":
                    text_parts.append(blk.get("text") or "")
                elif t == "tool_use":
                    tool_calls.append({
                        "id": blk.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": blk.get("name") or "",
                            "arguments": json.dumps(blk.get("input") or {}),
                        },
                    })
                # `thinking` blocks dropped — Anthropic-only
            asst: dict = {"role": "assistant"}
            if text_parts:
                asst["content"] = "\n".join(text_parts)
            elif not tool_calls:
                asst["content"] = ""
            if tool_calls:
                asst["tool_calls"] = tool_calls
            out.append(asst)
    return out


def _messages_chars(messages: list) -> int:
    """Char count across an OpenAI message array — proxy for prompt size."""
    n = 0
    for m in messages:
        n += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            n += len(fn.get("arguments") or "")
            n += len(fn.get("name") or "")
    return n


# ── Streaming chat call (TTFT + total wall) ────────────────────────────

def _stream_chat(port: int, payload: dict, timeout: int = 600) -> tuple[float, float, int]:
    """POST a streaming chat completion. Returns (ttft_s, total_s, n_tok)."""
    payload = {**payload, "stream": True,
               "stream_options": {"include_usage": True}}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    t_first: float | None = None
    usage_tok: int | None = None  # from usage chunk if server emits one
    delta_count = 0  # fallback: count of content/reasoning/tool deltas
    buf = b""
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            chunk = resp.read(256)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                line, buf = buf.split(b"\n\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                ev_str = line[5:].strip()
                if ev_str == b"[DONE]":
                    break
                try:
                    ev = json.loads(ev_str)
                except json.JSONDecodeError:
                    continue
                if not ev.get("choices") and ev.get("usage"):
                    usage_tok = ev["usage"].get("completion_tokens", usage_tok)
                    continue
                choices = ev.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                produced = (delta.get("content") or delta.get("reasoning_content")
                            or delta.get("tool_calls"))
                if produced:
                    delta_count += 1
                    if t_first is None:
                        t_first = time.perf_counter()
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    # Prefer usage.completion_tokens when the server emits it; otherwise
    # fall back to counting deltas. PR #59's server does NOT honour
    # stream_options.include_usage, so without this fallback every call
    # appears to generate 0 tokens even when 64 content deltas streamed.
    n_tok = usage_tok if usage_tok is not None else delta_count
    return (t_first - t0, t_end - t0, n_tok)


# ── Server lifecycle ───────────────────────────────────────────────────

def _wait_server_up(port: int, proc: subprocess.Popen, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1).read()
            return True
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            time.sleep(1)
    return False


def _stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Bench driver ───────────────────────────────────────────────────────

def run_config(label: str, port: int, slots: int, turns: list[dict],
               n_gen: int, max_ctx: int, log_path: Path) -> list[dict]:
    """Spin up server with --prefix-cache-slots=slots, replay each assistant turn."""
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(SERVER_SCRIPT),
         "--target", str(TARGET), "--draft", str(DRAFT), "--bin", str(BIN),
         "--max-ctx", str(max_ctx), "--port", str(port),
         "--prefix-cache-slots", str(slots)],
        stdout=log_f, stderr=subprocess.STDOUT, bufsize=1,
    )

    if not _wait_server_up(port, proc):
        log_f.close()
        out = log_path.read_text()[-1500:]
        _stop_server(proc)
        raise RuntimeError(f"{label}: server didn't come up\n{out}")

    print(f"\n--- {label} (slots={slots}) ---", flush=True)

    # Warmup: discard a single tiny call so CUDA graph capture / kernel JIT
    # land outside the measured run. Without this, call 1 cold absorbs
    # ~tens of seconds of one-time cost that has nothing to do with prefill.
    try:
        _stream_chat(port, {"model": "luce-dflash",
                            "messages": [{"role": "user", "content": "ok"}],
                            "max_tokens": 1}, timeout=180)
        print("  (warmup done)", flush=True)
    except Exception as e:
        print(f"  WARN: warmup failed: {e}", flush=True)

    asst_indices = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    per_call: list[dict] = []
    try:
        for n, idx in enumerate(asst_indices, start=1):
            prefix = _to_openai_messages(turns[:idx])
            in_chars = _messages_chars(prefix)
            payload = {"model": "luce-dflash", "messages": prefix,
                       "max_tokens": n_gen}
            try:
                ttft, wall, n_tok = _stream_chat(port, payload, timeout=600)
            except Exception as e:
                print(f"  call {n}: ERROR {e}")
                per_call.append({"call": n, "in_chars": in_chars,
                                 "ttft_s": float("nan"), "wall_s": float("nan"),
                                 "n_tok": 0, "error": str(e)})
                continue
            per_call.append({"call": n, "in_chars": in_chars,
                             "ttft_s": ttft, "wall_s": wall,
                             "n_tok": n_tok, "error": ""})
            print(f"  call {n}: in={in_chars:>7,} ttft={ttft*1000:>6.0f}ms "
                  f"wall={wall:>5.2f}s tok={n_tok}", flush=True)
    finally:
        _stop_server(proc)
        log_f.close()
    return per_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=10,
                    help="Cap LLM calls per replay (default: %(default)s)")
    ap.add_argument("--n-gen", type=int, default=64,
                    help="max_tokens per response (small to bound bench time, "
                         "but large enough that a thinking model emits at least "
                         "one completion token after reasoning)")
    ap.add_argument("--max-ctx", type=int, default=16384,
                    help="Server --max-ctx (default: %(default)s)")
    ap.add_argument("--session", type=Path, default=None,
                    help="Path to session JSONL; default = most recent under "
                         "~/.claude/projects/<cwd-as-dashes>")
    ap.add_argument("--cold-port", type=int, default=18290)
    ap.add_argument("--warm-port", type=int, default=18291)
    ap.add_argument("--warm-slots", type=int, default=4,
                    help="--prefix-cache-slots for warm config (default: %(default)s)")
    args = ap.parse_args()

    if not TARGET.exists() or not BIN.exists():
        print(f"SKIP: prereqs missing (target={TARGET.exists()} bin={BIN.exists()})")
        return 0

    if args.session:
        session = args.session
    else:
        session_dir = _default_session_dir()
        candidates = sorted(session_dir.glob("*.jsonl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"No session JSONL under {session_dir}")
            return 1
        session = candidates[0]
    print(f"Session: {session}", flush=True)

    turns = _load_transcript(session)
    asst_indices = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    if not asst_indices:
        print("No assistant turns in transcript")
        return 1
    if args.turns and len(asst_indices) > args.turns:
        # Truncate at the args.turns-th assistant index (inclusive).
        turns = turns[: asst_indices[args.turns - 1] + 1]
        asst_indices = asst_indices[: args.turns]
    n_user = sum(1 for t in turns if t["role"] == "user")
    print(f"Loaded {len(turns)} turns ({n_user} user, {len(asst_indices)} assistant)")

    cold = run_config("COLD (cache disabled)", port=args.cold_port, slots=0,
                      turns=turns, n_gen=args.n_gen, max_ctx=args.max_ctx,
                      log_path=Path("/tmp/bench_cold.log"))
    warm = run_config(f"WARM (cache slots={args.warm_slots})",
                      port=args.warm_port, slots=args.warm_slots,
                      turns=turns, n_gen=args.n_gen, max_ctx=args.max_ctx,
                      log_path=Path("/tmp/bench_warm.log"))

    print("\n=== Per-call latency (faithful replay) ===", flush=True)
    print(f"{'call':>4} {'in_chars':>9} "
          f"{'cold ttft':>10} {'warm ttft':>10} {'ttft x':>7}  "
          f"{'cold wall':>10} {'warm wall':>10} {'wall x':>7}")
    tot_c_ttft = tot_w_ttft = tot_c_wall = tot_w_wall = 0.0
    for c, w in zip(cold, warm):
        ct_ms = c["ttft_s"] * 1000
        wt_ms = w["ttft_s"] * 1000
        cw = c["wall_s"]
        ww = w["wall_s"]
        ttft_x = (ct_ms / wt_ms) if wt_ms > 0 else float("nan")
        wall_x = (cw / ww) if ww > 0 else float("nan")
        print(f"{c['call']:>4} {c['in_chars']:>9,} "
              f"{ct_ms:>8.0f}ms {wt_ms:>8.0f}ms {ttft_x:>6.2f}x  "
              f"{cw:>9.2f}s {ww:>9.2f}s {wall_x:>6.2f}x")
        if not c.get("error") and not w.get("error"):
            tot_c_ttft += ct_ms; tot_w_ttft += wt_ms
            tot_c_wall += cw;     tot_w_wall += ww
    if tot_w_ttft > 0 and tot_w_wall > 0:
        print(f"\ntotal cold:  ttft={tot_c_ttft/1000:.2f}s  wall={tot_c_wall:.2f}s")
        print(f"total warm:  ttft={tot_w_ttft/1000:.2f}s  wall={tot_w_wall:.2f}s")
        print(f"speedup:     ttft={tot_c_ttft/tot_w_ttft:.2f}x  "
              f"wall={tot_c_wall/tot_w_wall:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
