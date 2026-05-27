#!/usr/bin/env python3
"""
NIAH long-context two-arm bench (baseline vs pflash) via claude_code harness.

Runs claude_code client for each NIAH case so e2e timing includes client overhead.
Both arms per context; 3 needles per context depth=50%.
Outputs per-context metrics files in the required format.

Usage:
  python3 run_niah_twarm_harness.py --out-dir /path/to/out
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

# Insert bench dir for _harness_lib
sys.path.insert(0, str(Path(__file__).parent))
from _harness_lib import (
    PFLASH_ENV_OVERRIDES,
    BASELINE_ENV_OVERRIDES,
    build_env,
    harness_for,
    wait_for_health,
)

REPO = Path(__file__).resolve().parents[2]
BINARY = "/home/peppi/Dev/lucebox-hub/dflash/build/dflash_server"
TARGET = "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"
DECODE_DRAFT = "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf"
PFLASH_DRAFTER = "/home/peppi/models/Qwen3-0.6B-Q8_0.gguf"
PORT = 19099
BASE_URL = f"http://127.0.0.1:{PORT}"
CONTEXTS = [32768, 65536, 131072]
CASES_DIR = Path("/tmp")

# PFLASH arm for NIAH: cascade enabled (ANCHOR_TRANSITIVE)
NIAH_PFLASH_ENV = {
    **PFLASH_ENV_OVERRIDES,
    "PFLASH_COMPRESS_ANCHOR_TRANSITIVE": "1",
    # Override EXTRA_SERVER_ARGS to add cascade and correct drafter paths
    "EXTRA_SERVER_ARGS": (
        f"--prefill-compression always --prefill-keep-ratio 0.05 "
        f"--prefill-drafter {PFLASH_DRAFTER} --lazy-draft"
    ),
}


def kill_port(port: int) -> None:
    """Kill any process holding the given port (SIGTERM then SIGKILL)."""
    try:
        r = subprocess.run(
            ["fuser", f"{port}/tcp"], capture_output=True, text=True
        )
        pids = r.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), 15)  # SIGTERM
            except ProcessLookupError:
                pass
        if pids:
            time.sleep(3)
            # SIGKILL stragglers
            r2 = subprocess.run(
                ["fuser", f"{port}/tcp"], capture_output=True, text=True
            )
            for pid in r2.stdout.strip().split():
                try:
                    os.kill(int(pid), 9)
                except ProcessLookupError:
                    pass
            time.sleep(1)
    except FileNotFoundError:
        # fuser not available
        subprocess.run(
            f"pkill -f 'dflash_server.*{port}' || true",
            shell=True,
            capture_output=True,
        )
        time.sleep(3)


def start_server(env: dict, log_path: Path) -> subprocess.Popen:
    """Start dflash_server using the env config; return Popen handle."""
    extra_args_str = env.get("EXTRA_SERVER_ARGS", "")
    extra_args = extra_args_str.split() if extra_args_str else []

    draft = env.get("DRAFT", DECODE_DRAFT)
    draft_args = ["--draft", draft] if draft else []

    cmd = [
        BINARY, TARGET,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-ctx", "139264",
        "--max-tokens", "128",
        "--model-name", env.get("MODEL_ID", "luce-dflash"),
        "--ddtree",
        "--ddtree-budget", env.get("BUDGET", "16"),
    ] + draft_args + extra_args

    proc_env = os.environ.copy()
    proc_env["GGML_CUDA_NO_VMM"] = "1"
    proc_env["DFLASH27B_KV_K"] = "tq3_0"
    proc_env["DFLASH27B_KV_V"] = "tq3_0"
    # Pass through pflash env vars
    for k in [
        "PFLASH_DRAFTER_EARLY_EXIT_N",
        "PFLASH_DRAFTER_SCORE_LAYERS",
        "PFLASH_COMPRESS_ANCHOR_TRANSITIVE",
        "PFLASH_COMPRESS_ANCHOR_MAX_ITERS",
        "PFLASH_COMPRESS_RARE_MAX_FREQ",
        "PFLASH_COMPRESS_ANCHOR_NGRAM",
    ]:
        if k in env:
            proc_env[k] = env[k]

    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=f, env=proc_env)
    print(f"[niah] server PID={proc.pid} cmd={' '.join(cmd[:6])}...", flush=True)
    return proc


def wait_server(proc: subprocess.Popen, timeout: int = 240) -> bool:
    """Wait for server /health and 'listening on' line."""
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
    # Wait for model load
    deadline2 = time.time() + 240
    while time.time() < deadline2:
        if proc.poll() is not None:
            return False
        time.sleep(1)
        # Not checking log file here since it's written by proc; just give it time
        # Actually check the log
        break  # health OK is enough — common.sh wait_lucebox_server also does this
    return True


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    time.sleep(2)


def run_claude_case(prompt: str, env: dict, run_dir: Path, label: str, case_idx: int) -> dict:
    """Run claude CLI directly with the NIAH prompt. Return timing + text.

    Server must already be running on PORT. This drives the claude binary
    directly (not via run_claude_code.sh) so we control the server lifecycle.
    """
    claude_bin = env.get("CLAUDE_BIN", "/home/peppi/.local/bin/claude")
    claude_home = run_dir / f"claude-home-case{case_idx}"
    claude_home.mkdir(exist_ok=True)
    out_file = run_dir / f"claude-code-case{case_idx}.out"

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [
                claude_bin,
                "--print",
                "--output-format", "json",
                "--model", "luce-dflash",
                "--tools", "none",
                "--permission-mode", "dontAsk",
                "--no-session-persistence",
                prompt,  # positional: the prompt text itself
            ],
            env={
                **os.environ,
                "HOME": str(claude_home),
                "ANTHROPIC_API_KEY": "sk-lucebox",
                "ANTHROPIC_BASE_URL": BASE_URL,
                "CLAUDE_CODE_API_BASE_URL": BASE_URL,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_TELEMETRY": "1",
                "CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK": "1",
            },
            capture_output=True,
            text=True,
            timeout=600,
        )
        wall_s = time.perf_counter() - t0
        out_file.write_text(result.stdout + result.stderr)
        combined = result.stdout + result.stderr
        return {"wall_s": wall_s, "combined": combined, "rc": result.returncode}
    except subprocess.TimeoutExpired:
        wall_s = time.perf_counter() - t0
        return {"wall_s": wall_s, "combined": "", "rc": -1, "error": "timeout"}
    except Exception as e:
        wall_s = time.perf_counter() - t0
        return {"wall_s": wall_s, "combined": "", "rc": -1, "error": str(e)}


def parse_claude_json(combined: str) -> dict:
    """Extract text, input_tokens, ttft_ms from claude --output-format json."""
    result_text = ""
    input_tokens = None
    ttft_ms = None
    for line in combined.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "result":
            result_text = obj.get("result", "")
            usage = obj.get("usage", {})
            input_tokens = usage.get("input_tokens")
            ttft_ms = obj.get("ttft_ms")
            break
    return {"text": result_text, "input_tokens": input_tokens, "ttft_ms": ttft_ms}


def parse_server_metrics_all(server_log: Path) -> dict:
    """Extract ALL drafter_score_s, prefill_s, keep_ratio occurrences from server log.

    Returns lists (one entry per request) so callers can compute means.
    """
    metrics: dict = {"drafter_score_s": [], "prefill_s": [], "keep_ratio": []}
    if not server_log.exists():
        return metrics
    text = server_log.read_text()
    for m in re.finditer(r"\[drafter\]\s+forward\+score in ([\d.]+)s", text):
        metrics["drafter_score_s"].append(float(m.group(1)))
    for m in re.finditer(r"prefill=([\d.]+)s", text):
        metrics["prefill_s"].append(float(m.group(1)))
    for m in re.finditer(r"\[pflash\] \d+ -> \d+ -> \d+ tokens \(([\d.]+)% kept\)", text):
        metrics["keep_ratio"].append(float(m.group(1)))
    return metrics


def run_arm_niah(arm_label: str, arm_env_overrides: dict, cases: list, ctx: int,
                 out_dir: Path, server_log: Path) -> dict:
    """Run one arm (baseline or pflash) for all cases at given context.

    Server must already be running. Returns per-case and aggregate metrics.
    Server log accumulates all cases; we parse all occurrences for means.
    """
    arm_dir = out_dir / arm_label
    arm_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(arm_env_overrides)
    env["PORT"] = str(PORT)

    case_results = []
    for i, case in enumerate(cases):
        print(f"  [{arm_label}] case {i}/{len(cases)-1} ctx={ctx} ans={case['answer']}", flush=True)
        cr = run_claude_case(case["prompt"], env, arm_dir, arm_label, i)
        parsed = parse_claude_json(cr.get("combined", ""))

        found = str(case["answer"]) in parsed.get("text", "")
        case_results.append({
            "case_idx": i,
            "answer": case["answer"],
            "found": found,
            "wall_s": cr.get("wall_s"),
            "ttft_ms": parsed.get("ttft_ms"),
            "input_tokens": parsed.get("input_tokens"),
            "text": parsed.get("text", "")[:200],
            "rc": cr.get("rc"),
        })
        status = "FOUND" if found else "MISS"
        wall_s = f"{cr.get('wall_s', 0):.1f}s" if cr.get("wall_s") else "N/A"
        print(f"    {status} wall={wall_s}", flush=True)

    # Parse ALL server metrics after all cases complete
    srv_all = parse_server_metrics_all(server_log)
    drafters = srv_all["drafter_score_s"]
    prefills = srv_all["prefill_s"]
    keeps = srv_all["keep_ratio"]

    walls = [r["wall_s"] for r in case_results if r["wall_s"]]
    ttfts = [r["ttft_ms"] / 1000.0 for r in case_results if r.get("ttft_ms")]
    tokens = [r["input_tokens"] for r in case_results if r.get("input_tokens")]
    niah_pass = sum(1 for r in case_results if r["found"])

    print(f"  [{arm_label}] drafter times: {drafters}", flush=True)
    print(f"  [{arm_label}] prefill times: {prefills}", flush=True)
    print(f"  [{arm_label}] keep_ratios: {keeps}", flush=True)

    return {
        "arm": arm_label,
        "ctx": ctx,
        "niah_pass": niah_pass,
        "niah_total": len(cases),
        "wall_mean_s": sum(walls) / len(walls) if walls else None,
        "wall_median_s": median(walls) if walls else None,
        "ttft_mean_s": sum(ttfts) / len(ttfts) if ttfts else None,
        "drafter_mean_s": sum(drafters) / len(drafters) if drafters else None,
        "prefill_mean_s": sum(prefills) / len(prefills) if prefills else None,
        "keep_ratio_mean": sum(keeps) / len(keeps) if keeps else None,
        "prompt_tokens": int(sum(tokens) / len(tokens)) if tokens else ctx,
        "case_results": case_results,
    }


def write_ctx_metrics(baseline: dict, pflash: dict, ctx: int, out_dir: Path) -> None:
    """Write per-context metrics.txt in the required format."""
    metrics_path = out_dir / f"metrics_{ctx}.txt"

    b_wall = baseline.get("wall_mean_s")
    p_wall = pflash.get("wall_mean_s")
    e2e_speedup = (b_wall / p_wall) if (b_wall and p_wall and p_wall > 0) else None

    b_prefill = baseline.get("prefill_mean_s")
    p_prefill = pflash.get("prefill_mean_s")
    prefill_speedup = (b_prefill / p_prefill) if (b_prefill and p_prefill and p_prefill > 0) else None

    b_drafter = baseline.get("drafter_mean_s")
    p_drafter = pflash.get("drafter_mean_s")
    drafter_speedup = (b_drafter / p_drafter) if (b_drafter and p_drafter and p_drafter > 0) else None

    ctx_label = f"{ctx // 1024}K"
    prompt_tok = baseline.get("prompt_tokens") or pflash.get("prompt_tokens") or ctx

    keep = pflash.get("keep_ratio_mean")
    keep_str = f"{keep:.1f}%" if keep is not None else "N/A"

    b_niah = f"{baseline['niah_pass']}/{baseline['niah_total']}"
    p_niah = f"{pflash['niah_pass']}/{pflash['niah_total']}"

    lines = [
        f"context={ctx_label}",
        f"prompt_tokens={prompt_tok}",
        "",
        "[baseline]",
        (f"e2e_wall={b_wall:.1f}s" if b_wall else "e2e_wall=N/A") +
        (f"    prefill={b_prefill:.2f}s" if b_prefill else "    prefill=N/A") +
        (f"    drafter_wall={b_drafter:.2f}s" if b_drafter else "    drafter_wall=N/A") +
        f"    NIAH={b_niah}",
        "",
        "[pflash]",
        (f"e2e_wall={p_wall:.1f}s" if p_wall else "e2e_wall=N/A") +
        (f"    prefill={p_prefill:.2f}s" if p_prefill else "    prefill=N/A") +
        (f"    drafter_wall={p_drafter:.2f}s" if p_drafter else "    drafter_wall=N/A") +
        f"    NIAH={p_niah}    tokens_kept={keep_str}",
        "",
        "[headline]",
        (f"e2e_speedup={e2e_speedup:.2f}x" if e2e_speedup else "e2e_speedup=N/A") +
        (f"   prefill_speedup={prefill_speedup:.2f}x" if prefill_speedup else "   prefill_speedup=N/A") +
        (f"   drafter_speedup={drafter_speedup:.2f}x" if drafter_speedup else "   drafter_speedup=N/A"),
    ]
    metrics_path.write_text("\n".join(lines) + "\n")
    print(f"[niah] metrics -> {metrics_path}", flush=True)
    print("\n".join(lines), flush=True)


def run_ctx(ctx: int, cases: list, out_dir: Path) -> tuple[dict, dict]:
    """Run baseline + pflash arms for one context. Returns (baseline, pflash)."""
    print(f"\n[niah] ===== ctx={ctx} ({len(cases)} cases) =====", flush=True)

    def _run_single_arm(arm_label: str, arm_env: dict) -> dict:
        kill_port(PORT)
        srv_log = out_dir / f"server_{arm_label}_{ctx}.log"
        print(f"[niah] starting server arm={arm_label} ctx={ctx}", flush=True)

        full_env = build_env(arm_env)
        full_env["PORT"] = str(PORT)

        proc = start_server(full_env, srv_log)

        # Wait for health
        import urllib.request
        deadline = time.time() + 300
        health_ok = False
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{BASE_URL}/health", timeout=2)
                health_ok = True
                break
            except Exception:
                time.sleep(1)
            if proc.poll() is not None:
                print(f"[niah] server crashed early, log tail:", flush=True)
                print(srv_log.read_text()[-2000:], flush=True)
                return {"arm": arm_label, "ctx": ctx, "error": "server_crashed",
                        "niah_pass": 0, "niah_total": len(cases),
                        "wall_mean_s": None, "prompt_tokens": ctx, "case_results": []}

        if not health_ok:
            stop_server(proc)
            return {"arm": arm_label, "ctx": ctx, "error": "server_health_timeout",
                    "niah_pass": 0, "niah_total": len(cases),
                    "wall_mean_s": None, "prompt_tokens": ctx, "case_results": []}

        # Wait for "listening on" log line (model fully loaded)
        dl2 = time.time() + 300
        loaded = False
        while time.time() < dl2:
            if proc.poll() is not None:
                break
            if srv_log.exists() and "listening on" in srv_log.read_text():
                loaded = True
                break
            time.sleep(2)

        if not loaded:
            print(f"[niah] model load timeout for {arm_label} ctx={ctx}", flush=True)
            stop_server(proc)
            return {"arm": arm_label, "ctx": ctx, "error": "model_load_timeout",
                    "niah_pass": 0, "niah_total": len(cases),
                    "wall_mean_s": None, "prompt_tokens": ctx, "case_results": []}

        print(f"[niah] server ready for {arm_label} ctx={ctx}", flush=True)

        try:
            result = run_arm_niah(arm_label, arm_env, cases, ctx, out_dir, srv_log)
        finally:
            stop_server(proc)
            kill_port(PORT)

        return result

    baseline = _run_single_arm("baseline", BASELINE_ENV_OVERRIDES)
    pflash = _run_single_arm("pflash", NIAH_PFLASH_ENV)

    write_ctx_metrics(baseline, pflash, ctx, out_dir)
    return baseline, pflash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default="/tmp/lucebox-bench-pr/server/bench/results/2026-05-27_full_harness/_niah_longctx",
    )
    ap.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    ap.add_argument("--cases-dir", default=str(CASES_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_dir = Path(args.cases_dir)
    all_results = []

    for ctx in args.contexts:
        case_file = cases_dir / f"niah_{ctx}.jsonl"
        if not case_file.exists():
            print(f"[niah] ERROR: case file missing: {case_file} — skipping ctx={ctx}", flush=True)
            continue
        with open(case_file) as f:
            cases = [json.loads(l) for l in f]
        print(f"[niah] loaded {len(cases)} cases for ctx={ctx}", flush=True)

        try:
            baseline, pflash = run_ctx(ctx, cases, out_dir)
            all_results.append((ctx, baseline, pflash))
        except Exception as e:
            print(f"[niah] ERROR ctx={ctx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    # Save raw results
    raw = []
    for ctx, b, p in all_results:
        raw.append({"ctx": ctx, "baseline": {k: v for k, v in b.items() if k != "case_results"},
                    "pflash": {k: v for k, v in p.items() if k != "case_results"}})
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(raw, f, indent=2)

    print("\n[niah] === FINAL SUMMARY ===", flush=True)
    for ctx, b, p in all_results:
        ctx_label = f"{ctx // 1024}K"
        b_wall = b.get("wall_mean_s")
        p_wall = p.get("wall_mean_s")
        speedup = (b_wall / p_wall) if (b_wall and p_wall) else None
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        b_niah = f"{b['niah_pass']}/{b['niah_total']}"
        p_niah = f"{p['niah_pass']}/{p['niah_total']}"
        print(f"  {ctx_label}: e2e_speedup={speedup_str}  NIAH base={b_niah} pflash={p_niah}", flush=True)

    print(f"\n[niah] results in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
