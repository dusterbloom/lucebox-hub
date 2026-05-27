#!/usr/bin/env python3
"""
NIAH two-arm bench: baseline vs pflash, claude_code client, 32K/64K/128K.

Runs 3 needles per context, one fresh server per case (ggml view bug workaround).
Measures e2e wall, prefill_s, drafter_wall from server log + claude output JSON.

Usage:
    python3 run_niah_twoarm_claude_code.py --out-dir /path/to/results

Output: per-context metrics files + raw_results.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

SERVER_BIN = Path("/home/peppi/Dev/lucebox-hub/dflash/build/dflash_server")
TARGET = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DECODE_DRAFT = Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf")
PFLASH_DRAFTER = Path("/home/peppi/models/Qwen3-0.6B-Q8_0.gguf")
CLAUDE_BIN = Path("/home/peppi/.local/bin/claude")
PORT = 19099
BASE_URL = f"http://127.0.0.1:{PORT}"
CONTEXTS = [32768, 65536, 131072]
NIAH_DATA_DIR = Path("/tmp")

# Server args per arm
BASELINE_CMD_EXTRA = []
BASELINE_ENV_EXTRA = {}

PFLASH_CMD_EXTRA = [
    "--prefill-compression", "always",
    "--prefill-keep-ratio", "0.05",
    "--prefill-drafter", str(PFLASH_DRAFTER),
    "--lazy-draft",
]
PFLASH_ENV_EXTRA = {
    "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
    "PFLASH_DRAFTER_SCORE_LAYERS": "7",
    "PFLASH_COMPRESS_ANCHOR_TRANSITIVE": "1",
}


def build_server_env(arm_env_extra: dict) -> dict:
    env = os.environ.copy()
    # Clear any leftover pflash env from parent
    for k in ("PFLASH_DRAFTER_EARLY_EXIT_N", "PFLASH_DRAFTER_SCORE_LAYERS",
              "PFLASH_COMPRESS_ANCHOR_TRANSITIVE"):
        env.pop(k, None)
    env["GGML_CUDA_NO_VMM"] = "1"
    env["DFLASH27B_KV_K"] = "tq3_0"
    env["DFLASH27B_KV_V"] = "tq3_0"
    env.update(arm_env_extra)
    return env


def start_server(arm: str, log_path: Path, arm_cmd_extra: list, arm_env_extra: dict) -> subprocess.Popen:
    env = build_server_env(arm_env_extra)
    cmd = [
        str(SERVER_BIN), str(TARGET),
        "--draft", str(DECODE_DRAFT),
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", "139264",
        "--max-tokens", "128",
        "--model-name", "luce-dflash",
        "--ddtree",
        "--ddtree-budget", "16",
    ] + arm_cmd_extra

    print(f"  [server] starting arm={arm} port={PORT}", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=env)
    return proc


def wait_server(proc: subprocess.Popen, timeout: int = 180) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    # Phase 1: /health
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                return False
    else:
        return False
    # Phase 2: wait for "listening on" in log (model fully loaded on GPU)
    log_path = None
    # find log from proc
    for _ in range(180):
        # just wait briefly for GPU load
        time.sleep(1)
        if proc.poll() is not None:
            return False
        # re-probe
        try:
            urllib.request.urlopen(f"{BASE_URL}/health", timeout=1)
        except Exception:
            pass
        # check server ready via /v1/models
        try:
            resp = urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=2)
            if resp.status == 200:
                return True
        except Exception:
            pass
    return True  # health was ok, proceed


def wait_server_full(proc: subprocess.Popen, log_path: Path, timeout: int = 300) -> bool:
    """Wait for /health then for 'listening on' line in server log."""
    import urllib.request
    deadline = time.time() + timeout
    health_ok = False

    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
            health_ok = True
            break
        except Exception:
            time.sleep(1)
            if proc.poll() is not None:
                return False

    if not health_ok:
        return False

    # Now wait for "listening on" which means GPU load is done
    while time.time() < deadline:
        try:
            text = log_path.read_text()
            if "listening on" in text:
                return True
        except Exception:
            pass
        time.sleep(1)
        if proc.poll() is not None:
            return False

    return False


def stop_server(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(2)


def run_claude_client(prompt: str, run_dir: Path, case_idx: int) -> dict:
    """Run claude client with the given prompt. Returns dict with wall_s, ok_done, text, input_tokens."""
    home_dir = run_dir / f"claude-home-case{case_idx}"
    home_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / f"claude-case{case_idx}.out"

    # Write prompt to temp file to avoid shell quoting issues with huge prompts
    prompt_file = run_dir / f"prompt-case{case_idx}.txt"
    prompt_file.write_text(prompt)

    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["ANTHROPIC_API_KEY"] = "sk-lucebox"
    env["ANTHROPIC_BASE_URL"] = BASE_URL
    env["CLAUDE_CODE_API_BASE_URL"] = BASE_URL
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    env["CLAUDE_CODE_DISABLE_TELEMETRY"] = "1"
    env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"] = "1"

    # Claude reads prompt from stdin when --print is used without positional arg
    # But for huge prompts we use the file approach via positional arg
    cmd = [
        str(CLAUDE_BIN),
        "--print",
        "--output-format", "json",
        "--model", "luce-dflash",
        "--tools", "none",
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
        f"$(cat {prompt_file})",  # will be replaced below
    ]

    # Actually pass prompt content directly via stdin to avoid shell expansion issues
    # claude --print reads from positional arg or stdin
    # Use positional arg approach with temp file
    with open(out_file, "w") as out_f:
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    str(CLAUDE_BIN),
                    "--print",
                    "--output-format", "json",
                    "--model", "luce-dflash",
                    "--tools", "none",
                    "--permission-mode", "dontAsk",
                    "--no-session-persistence",
                    prompt,  # positional arg = the prompt itself
                ],
                env=env,
                stdout=out_f,
                stderr=out_f,
                timeout=900,
            )
            wall_s = time.perf_counter() - t0
            rc = result.returncode
        except subprocess.TimeoutExpired:
            wall_s = time.perf_counter() - t0
            return {"wall_s": wall_s, "ok_done": False, "text": "", "input_tokens": None, "rc": -1, "error": "timeout"}

    # Parse output
    text = out_file.read_text()
    ok_done = "OK_DONE" in text
    input_tokens = None
    response_text = ""

    # Try to parse JSON result lines
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get("type") == "result":
                response_text = d.get("result", "")
                usage = d.get("usage", {})
                input_tokens = usage.get("input_tokens")
                if input_tokens is None:
                    # Try modelUsage
                    for model_data in d.get("modelUsage", {}).values():
                        it = model_data.get("inputTokens")
                        if it:
                            input_tokens = it
                            break
        except Exception:
            pass

    return {
        "wall_s": wall_s,
        "ok_done": ok_done,
        "text": response_text[:200],
        "input_tokens": input_tokens,
        "rc": rc,
        "error": None,
    }


def parse_server_log(log_path: Path) -> dict:
    """Extract prefill_s, drafter_score_s, keep_ratio from server log."""
    metrics = {
        "prefill_s": None,
        "drafter_score_s": None,
        "keep_ratio": None,
        "prompt_tokens_server": None,
    }
    if not log_path.exists():
        return metrics
    try:
        text = log_path.read_text()
        m = re.search(r"\[prefill\] tokens=(\d+) time=([\d.]+) s", text)
        if m:
            # This is the post-compression effective count in pflash arm,
            # but still useful for baseline prefill_s
            metrics["prefill_s"] = float(m.group(2))

        m2 = re.search(r"\[drafter\] forward\+score in ([\d.]+)s", text)
        if m2:
            metrics["drafter_score_s"] = float(m2.group(1))

        m3 = re.search(r"\[pflash\] \d+ -> \d+ -> \d+ tokens \(([\d.]+)% kept\)", text)
        if m3:
            metrics["keep_ratio"] = float(m3.group(1))

    except Exception:
        pass
    return metrics


def run_one_niah_case(arm: str, ctx: int, case: dict, case_idx: int, out_dir: Path,
                      arm_cmd_extra: list, arm_env_extra: dict) -> dict:
    """Start server, run claude client with NIAH prompt, stop server. Return metrics."""
    log_path = out_dir / f"server_{arm}_{ctx}_case{case_idx}.log"
    proc = start_server(arm, log_path, arm_cmd_extra, arm_env_extra)

    result = {
        "arm": arm, "ctx": ctx, "case_idx": case_idx,
        "wall_s": None, "prefill_s": None, "drafter_score_s": None,
        "keep_ratio": None, "input_tokens": None,
        "found": False, "text": "", "error": None,
    }

    try:
        if not wait_server_full(proc, log_path, timeout=300):
            tail = ""
            try:
                tail = "".join(log_path.read_text().splitlines()[-30:])
            except Exception:
                pass
            result["error"] = f"server_start_failed: {tail[:300]}"
            return result

        client_result = run_claude_client(case["prompt"], out_dir, case_idx)
        result["wall_s"] = client_result["wall_s"]
        result["input_tokens"] = client_result["input_tokens"]
        result["text"] = client_result.get("text", "")[:200]
        result["error"] = client_result.get("error")

        # Check needle found
        answer = str(case.get("answer", ""))
        result["found"] = answer in client_result.get("text", "")

    finally:
        stop_server(proc)

    # Parse server log post-mortem
    srv_metrics = parse_server_log(log_path)
    result["prefill_s"] = srv_metrics["prefill_s"]
    result["drafter_score_s"] = srv_metrics["drafter_score_s"]
    result["keep_ratio"] = srv_metrics["keep_ratio"]

    return result


def run_arm_ctx(arm: str, ctx: int, cases: list, out_dir: Path,
                arm_cmd_extra: list, arm_env_extra: dict) -> dict:
    print(f"\n[bench] arm={arm} ctx={ctx} cases={len(cases)}", flush=True)
    case_results = []
    for i, case in enumerate(cases):
        print(f"  case {i}: n_tokens={case.get('n_tokens', ctx)} answer={case.get('answer')}", flush=True)
        r = run_one_niah_case(arm, ctx, case, i, out_dir, arm_cmd_extra, arm_env_extra)
        case_results.append(r)
        status = "FOUND" if r["found"] else "MISS"
        wall_s = f"{r['wall_s']:.1f}s" if r["wall_s"] else "N/A"
        prefill_s = f"{r['prefill_s']:.2f}s" if r["prefill_s"] else "N/A"
        drafter_s = f"{r['drafter_score_s']:.3f}s" if r["drafter_score_s"] else "N/A"
        print(f"  case {i}: wall={wall_s} prefill={prefill_s} drafter={drafter_s} "
              f"input_tokens={r['input_tokens']} [{status}] text={r['text'][:60]!r}", flush=True)
        if r.get("error"):
            print(f"  case {i}: error={r['error'][:200]}", flush=True)

    walls = [r["wall_s"] for r in case_results if r["wall_s"] is not None]
    prefills = [r["prefill_s"] for r in case_results if r["prefill_s"] is not None]
    drafters = [r["drafter_score_s"] for r in case_results if r["drafter_score_s"] is not None]
    input_toks = [r["input_tokens"] for r in case_results if r["input_tokens"] is not None]
    niah_pass = sum(1 for r in case_results if r["found"])
    keep_ratios = [r["keep_ratio"] for r in case_results if r["keep_ratio"] is not None]

    return {
        "arm": arm, "ctx": ctx,
        "case_results": case_results,
        "wall_mean_s": mean(walls) if walls else None,
        "prefill_mean_s": mean(prefills) if prefills else None,
        "drafter_mean_s": mean(drafters) if drafters else None,
        "input_tokens_mean": mean(input_toks) if input_toks else None,
        "niah_pass": niah_pass,
        "niah_total": len(cases),
        "keep_ratio_mean": mean(keep_ratios) if keep_ratios else None,
    }


def write_ctx_metrics(baseline: dict, pflash: dict, ctx: int, out_dir: Path):
    ctx_k = f"{ctx // 1024}K"
    metrics_path = out_dir / f"metrics_{ctx_k}.txt"

    b_wall = baseline.get("wall_mean_s")
    p_wall = pflash.get("wall_mean_s")
    b_prefill = baseline.get("prefill_mean_s")
    p_prefill = pflash.get("prefill_mean_s")
    b_drafter = baseline.get("drafter_mean_s")
    p_drafter = pflash.get("drafter_mean_s")
    input_toks = pflash.get("input_tokens_mean") or baseline.get("input_tokens_mean")

    speedup = (b_wall / p_wall) if (b_wall and p_wall and p_wall > 0) else None
    pf_speedup = (b_prefill / p_prefill) if (b_prefill and p_prefill and p_prefill > 0) else None
    dr_speedup = (b_drafter / p_drafter) if (b_drafter and p_drafter and p_drafter > 0) else None
    keep_str = f"{pflash.get('keep_ratio_mean', 0):.1f}%" if pflash.get("keep_ratio_mean") else "N/A"

    b_niah = f"{baseline['niah_pass']}/{baseline['niah_total']}"
    p_niah = f"{pflash['niah_pass']}/{pflash['niah_total']}"

    lines = [
        f"context={ctx_k}\n",
        f"prompt_tokens={int(input_toks) if input_toks else 'N/A'}\n\n",
        f"[baseline]\n",
        f"e2e_wall={'%.1fs' % b_wall if b_wall else 'N/A'}    prefill={'%.2fs' % b_prefill if b_prefill else 'N/A'}    drafter_wall={'%.3fs' % b_drafter if b_drafter else 'N/A'}    NIAH={b_niah}\n\n",
        f"[pflash]\n",
        f"e2e_wall={'%.1fs' % p_wall if p_wall else 'N/A'}    prefill={'%.2fs' % p_prefill if p_prefill else 'N/A'}    drafter_wall={'%.3fs' % p_drafter if p_drafter else 'N/A'}    NIAH={p_niah}    tokens_kept={keep_str}\n\n",
        f"[headline]\n",
        f"e2e_speedup={'%.2fx' % speedup if speedup else 'N/A'}   prefill_speedup={'%.2fx' % pf_speedup if pf_speedup else 'N/A'}   drafter_speedup={'%.2fx' % dr_speedup if dr_speedup else 'N/A'}\n",
    ]
    metrics_path.write_text("".join(lines))
    print(f"[bench] wrote {metrics_path}", flush=True)
    print("".join(lines), flush=True)
    return {
        "ctx": ctx_k,
        "prompt_tokens": int(input_toks) if input_toks else "N/A",
        "b_wall": b_wall, "p_wall": p_wall, "speedup": speedup,
        "b_niah": b_niah, "p_niah": p_niah,
        "b_prefill": b_prefill, "p_prefill": p_prefill, "pf_speedup": pf_speedup,
        "b_drafter": b_drafter, "p_drafter": p_drafter, "dr_speedup": dr_speedup,
        "keep_str": keep_str,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    ap.add_argument("--data-dir", default=str(NIAH_DATA_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    cases_by_ctx = {}
    for ctx in args.contexts:
        f = data_dir / f"niah_{ctx}.jsonl"
        if not f.exists():
            print(f"[error] missing {f}", flush=True)
            sys.exit(1)
        with open(f) as fh:
            cases_by_ctx[ctx] = [json.loads(l) for l in fh if l.strip()]
        print(f"[init] ctx={ctx} cases={len(cases_by_ctx[ctx])}", flush=True)

    all_ctx_summaries = []
    all_raw = []

    for ctx in args.contexts:
        cases = cases_by_ctx[ctx]
        ctx_k = f"{ctx // 1024}K"

        # Baseline arm
        print(f"\n=== BASELINE arm, ctx={ctx_k} ===", flush=True)
        baseline = run_arm_ctx(
            "baseline", ctx, cases, out_dir,
            BASELINE_CMD_EXTRA, BASELINE_ENV_EXTRA
        )

        # pflash arm
        print(f"\n=== PFLASH arm, ctx={ctx_k} ===", flush=True)
        pflash = run_arm_ctx(
            "pflash", ctx, cases, out_dir,
            PFLASH_CMD_EXTRA, PFLASH_ENV_EXTRA
        )

        summary = write_ctx_metrics(baseline, pflash, ctx, out_dir)
        all_ctx_summaries.append(summary)
        all_raw.append({"baseline": baseline, "pflash": pflash})

        # Save raw
        with open(out_dir / "raw_results.json", "w") as f:
            json.dump(all_raw, f, indent=2, default=str)

    # Print headline summary
    print("\n=== NIAH HEADLINE SUMMARY ===", flush=True)
    print(f"{'ctx':>6}  {'prompt_toks':>12}  {'base_wall':>10}  {'pflash_wall':>11}  {'e2e_speedup':>12}  {'NIAH b/p':>10}", flush=True)
    for s in all_ctx_summaries:
        b_w = f"{'%.1fs' % s['b_wall']}" if s['b_wall'] else "N/A"
        p_w = f"{'%.1fs' % s['p_wall']}" if s['p_wall'] else "N/A"
        sp = f"{'%.2fx' % s['speedup']}" if s['speedup'] else "N/A"
        print(f"{s['ctx']:>6}  {str(s['prompt_tokens']):>12}  {b_w:>10}  {p_w:>11}  {sp:>12}  {s['b_niah']:>5} / {s['p_niah']}", flush=True)


if __name__ == "__main__":
    main()
