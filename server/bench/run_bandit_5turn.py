#!/usr/bin/env python3
"""run_bandit_5turn.py — drive a single client through 5 turns to prove bandit convergence.

Each turn sends the next prompt in harness/clients/prompts/multiturn_5/turn_{1..5}.txt
via the same PFLASH_SESSION_ID so the C++ bandit can observe accept_rate across the
full conversation and adapt keep_ratio accordingly.

Usage:
  python3 server/bench/run_bandit_5turn.py --client claude_code \\
    --output server/bench/results/2026-05-27_full_harness/claude_code/bandit_5turn

  python3 server/bench/run_bandit_5turn.py --client codex \\
    --output server/bench/results/2026-05-27_full_harness/codex/bandit_5turn

Captures:
  - turn-by-turn keep_ratio evolution from server stderr [pflash-bandit] lines
  - wall + accept_rate per turn
  - <output>/trajectory.csv   — turn,keep_ratio,ema,accept_rate,wall_s,ok_done
  - <output>/server.log       — raw server stderr (all turns, server stays up)
  - <output>/client_turn_N.log — per-turn client stdout
  - <output>/metrics.txt      — summary

Requires PFLASH_SESSION_ID env var or generates one automatically.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO / "harness/clients"
PROMPTS_DIR = HARNESS_DIR / "prompts/multiturn_5"
BENCH_LOCK = "/tmp/lucebox-bench.lock"

VALID_CLIENTS = ["claude_code", "codex", "pi", "hermes", "opencode"]

DEFAULT_ENV = {
    "MODEL_SERVER": "lucebox",
    "LUCEBOX_SERVER_BACKEND": "cpp",
    "DFLASH27B_KV_K": "tq3_0",
    "DFLASH27B_KV_V": "tq3_0",
    "GGML_CUDA_NO_VMM": "1",
    "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
    "PFLASH_DRAFTER_SCORE_LAYERS": "7",
    "TARGET": "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf",
    "DRAFT": "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf",
    "MAX_CTX": "98304",
    "MAX_TOKENS": "512",
    "VERIFY_MODE": "ddtree",
    "BUDGET": "16",
    "REPO_DIR": str(REPO),
    "RUN_DIR": "/tmp/lucebox-bench-runs",
    "EXTRA_SERVER_ARGS": (
        "--prefill-compression always --prefill-keep-ratio 0.10 "
        "--prefill-drafter /home/peppi/models/Qwen3-0.6B-BF16.gguf"
    ),
    "CLAUDE_TIMEOUT": "600",
    "MARKER": "OK_DONE",
    "CLAUDE_TOOLS": "none",
    "PORT": "19099",
    "HOST": "127.0.0.1",
    "MODEL_ID": "luce-dflash",
    "API_KEY": "sk-lucebox",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--client", default="claude_code", choices=VALID_CLIENTS,
                   help="Harness client to drive (default: claude_code)")
    p.add_argument("--output", required=True,
                   help="Output directory for results")
    p.add_argument("--session-id", default=None,
                   help="PFLASH_SESSION_ID; auto-generated if not set")
    p.add_argument("--dflash-server-bin", default=None,
                   help="Path to dflash_server binary (default: <repo>/server/build/dflash_server)")
    return p.parse_args()


def wait_for_lock_and_start_server(env: dict, server_log_path: Path) -> subprocess.Popen:
    """Start dflash_server (C++ binary) under flock; returns the popen handle."""
    server_bin = env.get("DFLASH_SERVER_BIN", str(REPO / "server/build/dflash_server"))
    extra_args = env.get("EXTRA_SERVER_ARGS", "").split()
    cmd = [
        "flock", "-x", BENCH_LOCK,
        server_bin, env["TARGET"],
        "--draft", env["DRAFT"],
        "--host", env["HOST"],
        "--port", env["PORT"],
        "--max-ctx", env["MAX_CTX"],
        "--max-tokens", env["MAX_TOKENS"],
        "--model-name", env["MODEL_ID"],
        "--ddtree", "--ddtree-budget", env["BUDGET"],
    ] + extra_args
    server_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[5turn] starting server on port {env['PORT']}...", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=server_log_path.open("w"))
    return proc


def wait_for_health(base_url: str, timeout_s: int = 120) -> bool:
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_proxy(session_id: str, host: str, proxy_port: int, upstream: str,
                proxy_log: Path) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(HARNESS_DIR / "session_inject_proxy.py"),
        "--host", host,
        "--port", str(proxy_port),
        "--upstream", upstream,
        "--session-id", session_id,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=proxy_log.open("w"))


def run_turn(turn_n: int, client: str, prompt_file: Path,
             client_url: str, env: dict, out_dir: Path) -> dict:
    """Run a single turn via the client harness; return metrics dict."""
    client_log = out_dir / f"client_turn_{turn_n}.log"
    stamp = f"5turn-t{turn_n}-{int(time.time())}"
    turn_env = env.copy()
    turn_env["PROMPT_FILE"] = str(prompt_file)
    turn_env["STAMP"] = stamp
    turn_env["BASE_URL"] = client_url
    # Override PORT so common.sh BASE_URL matches our proxy/server URL
    port = client_url.split(":")[-1].rstrip("/")
    turn_env["PORT"] = port
    # Ensure HOST matches
    host = client_url.split("://")[1].split(":")[0]
    turn_env["HOST"] = host

    harness = HARNESS_DIR / f"run_{client}.sh"
    print(f"[5turn] turn={turn_n} client={client} prompt={prompt_file.name}", flush=True)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["bash", str(harness)],
            env=turn_env,
            capture_output=True,
            text=True,
            timeout=int(env.get("CLAUDE_TIMEOUT", "600")),
        )
        elapsed = time.perf_counter() - t0
        out_text = result.stdout + result.stderr
        client_log.write_text(out_text)
        rc = result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        client_log.write_text(f"TIMEOUT after {elapsed:.0f}s\n")
        return {"turn": turn_n, "wall_s": elapsed, "ok_done": False,
                "accept_rate": None, "rc": -1, "error": "timeout"}

    ok_done = env.get("MARKER", "OK_DONE") in out_text
    # Extract accept_rate from harness output
    ar_m = re.search(r"accepted=\d+/\d+ \(([0-9.]+)%\)", out_text)
    accept_rate = float(ar_m.group(1)) if ar_m else None

    print(f"[5turn] turn={turn_n} elapsed={elapsed:.1f}s ok_done={ok_done} "
          f"accept_rate={accept_rate}", flush=True)
    return {"turn": turn_n, "wall_s": elapsed, "ok_done": ok_done,
            "accept_rate": accept_rate, "rc": rc}


def extract_bandit_lines(server_log: Path, after_byte: int = 0) -> list[str]:
    """Return [pflash-bandit] log lines from server_log after byte offset."""
    try:
        text = server_log.read_text(errors="replace")
        lines = [l for l in text[after_byte:].splitlines() if "[pflash-bandit]" in l]
        return lines
    except Exception:
        return []


def parse_bandit_line(line: str) -> dict:
    """Parse a [pflash-bandit] line into a dict of key=value fields."""
    result = {}
    for m in re.finditer(r"(\w+)=([\d.]+)", line):
        try:
            result[m.group(1)] = float(m.group(2))
        except ValueError:
            result[m.group(1)] = m.group(2)
    return result


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = args.client
    session_id = args.session_id or os.environ.get("PFLASH_SESSION_ID") or str(uuid.uuid4())
    print(f"[5turn] client={client} session_id={session_id}", flush=True)

    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    if args.dflash_server_bin:
        env["DFLASH_SERVER_BIN"] = args.dflash_server_bin
    env["PFLASH_SESSION_ID"] = session_id

    server_log = out_dir / "server.log"
    proxy_log = out_dir / "proxy.log"
    metrics_txt = out_dir / "metrics.txt"
    trajectory_csv = out_dir / "trajectory.csv"

    host = env["HOST"]
    port = int(env["PORT"])
    base_url = f"http://{host}:{port}"
    proxy_port = port - 17  # 19099 -> 19082

    # Start server
    server_proc = wait_for_lock_and_start_server(env, server_log)
    try:
        if not wait_for_health(base_url, timeout_s=120):
            print("[5turn] ERROR: server did not become healthy", file=sys.stderr)
            sys.exit(1)
        print(f"[5turn] server healthy at {base_url}", flush=True)

        # Start session-inject proxy
        proxy_proc = start_proxy(session_id, host, proxy_port, base_url, proxy_log)
        time.sleep(1)  # brief settle
        proxy_url = f"http://{host}:{proxy_port}"
        print(f"[5turn] proxy at {proxy_url} (session={session_id})", flush=True)

        try:
            turn_results = []
            bandit_trajectory = []
            server_log_offset = 0

            for turn_n in range(1, 6):
                prompt_file = PROMPTS_DIR / f"turn_{turn_n}.txt"
                if not prompt_file.exists():
                    print(f"[5turn] ERROR: prompt file missing: {prompt_file}", file=sys.stderr)
                    sys.exit(1)

                # Snapshot server log offset before turn
                try:
                    server_log_offset = server_log.stat().st_size
                except Exception:
                    server_log_offset = 0

                metrics = run_turn(turn_n, client, prompt_file, proxy_url, env, out_dir)
                turn_results.append(metrics)

                # Extract bandit lines emitted during this turn
                bandit_lines = extract_bandit_lines(server_log, after_byte=server_log_offset)
                bandit_data = {}
                if bandit_lines:
                    # Take the last [pflash-bandit] line per turn (most updated)
                    bandit_data = parse_bandit_line(bandit_lines[-1])
                    print(f"[5turn] bandit turn={turn_n}: {bandit_lines[-1].strip()}", flush=True)
                else:
                    print(f"[5turn] bandit turn={turn_n}: no [pflash-bandit] line found", flush=True)

                bandit_trajectory.append({
                    "turn": turn_n,
                    "keep_ratio": bandit_data.get("new_keep", bandit_data.get("keep_ratio", "")),
                    "old_keep": bandit_data.get("old_keep", ""),
                    "ema": bandit_data.get("ema", ""),
                    "accept_rate": metrics.get("accept_rate", ""),
                    "wall_s": f"{metrics['wall_s']:.1f}",
                    "ok_done": "YES" if metrics.get("ok_done") else "NO",
                })

        finally:
            if proxy_proc.poll() is None:
                proxy_proc.terminate()
                proxy_proc.wait(timeout=5)

    finally:
        if server_proc.poll() is None:
            server_proc.terminate()
            server_proc.wait(timeout=10)

    # Write trajectory CSV
    csv_fields = ["turn", "keep_ratio", "old_keep", "ema", "accept_rate", "wall_s", "ok_done"]
    with trajectory_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(bandit_trajectory)
    print(f"[5turn] trajectory written to {trajectory_csv}", flush=True)

    # Write metrics summary
    ok_all = all(r.get("ok_done") for r in turn_results)
    with metrics_txt.open("w") as f:
        f.write(f"client={client}\n")
        f.write(f"session_id={session_id}\n")
        f.write(f"turns_completed={len(turn_results)}\n")
        f.write(f"all_ok_done={'YES' if ok_all else 'NO'}\n")
        for r in turn_results:
            f.write(f"turn_{r['turn']}_wall_s={r['wall_s']:.1f}\n")
            f.write(f"turn_{r['turn']}_ok_done={'YES' if r.get('ok_done') else 'NO'}\n")
            ar = r.get('accept_rate')
            f.write(f"turn_{r['turn']}_accept_rate={ar if ar is not None else 'N/A'}\n")

    print("\n[5turn] === TRAJECTORY ===")
    print(f"{'turn':>5} {'keep_ratio':>12} {'old_keep':>10} {'ema':>8} "
          f"{'accept%':>8} {'wall_s':>8} {'ok_done':>8}")
    for row in bandit_trajectory:
        print(f"{row['turn']:>5} {str(row['keep_ratio']):>12} {str(row['old_keep']):>10} "
              f"{str(row['ema']):>8} {str(row['accept_rate']):>8} "
              f"{row['wall_s']:>8} {row['ok_done']:>8}")
    print(f"\n[5turn] done. all_ok_done={'YES' if ok_all else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
