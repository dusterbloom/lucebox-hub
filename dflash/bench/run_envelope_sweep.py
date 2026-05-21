#!/usr/bin/env python3
"""Multi-server envelope sweep runner.

The C++ server loads the pflash drafter only when started with
--pflash-mode always|auto. Per-request pflash_mode override cannot
switch between OFF and ALWAYS within a single server instance because
the should_compress gate in http_server.cpp reads config_.pflash_mode,
not the per-request override.

Strategy:
  Phase 1: Start server with --pflash-mode off  → run all OFF cells
  Phase 2: For each keep_ratio:
             Start server with --pflash-mode always --prefill-keep-ratio K
             Run all ALWAYS cells for that keep_ratio

Each phase calls sweep_envelope.py with a filtered mini-grid, then the
frontier rows are merged into a single frontier.json.

Usage:
  python dflash/bench/run_envelope_sweep.py \
      --grid dflash/bench/results/2026-05-21_envelope/grid.yaml \
      --out  dflash/bench/results/2026-05-21_envelope \
      [--dry-run]
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("[run_envelope_sweep] pyyaml not installed. Run: pip install pyyaml")

_HERE = Path(__file__).resolve().parent
_WORKTREE = _HERE.parent.parent
_SWEEP_SCRIPT = _HERE / "sweep_envelope.py"

# ─── Paths (resolve from environment or defaults) ────────────────────────────

TARGET_MODEL  = os.environ.get("TARGET",
    "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
PFLASH_DRAFTER = os.environ.get("PFLASH_DRAFTER",
    "/home/peppi/models/Qwen3-0.6B-BF16.gguf")
DFLASH_SERVER_BIN = os.environ.get("DFLASH_SERVER_BIN",
    str(_WORKTREE / "dflash/build/dflash_server"))
SERVER_HOST   = "127.0.0.1"
SERVER_PORT   = 8080
MAX_CTX       = 73728   # 72K — fits 64K input + headroom; drop to 65536 if OOM


def _wait_for_server(log_path: Path, timeout_s: int = 120) -> bool:
    """Tail log_path until server ready or timeout. Returns True if up."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if "listening on" in text or "[server] listening" in text:
                return True
            if "failed\n" in text or "error\n" in text.lower():
                print(f"[runner] server log tail:\n{text[-2000:]}", flush=True)
                return False
        time.sleep(1)
    print(f"[runner] server did not come up within {timeout_s}s", flush=True)
    return False


def start_server(
    mode: str,
    keep_ratio: float,
    log_path: Path,
    dry_run: bool,
) -> int | None:
    """Start dflash_server and return PID, or None on dry-run/failure."""
    cmd = [
        DFLASH_SERVER_BIN, TARGET_MODEL,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--max-ctx", str(MAX_CTX),
        "--pflash-mode", mode,
        "--prefill-skip-park",      # avoid cuMemSetAccess crash on unpark
        "--cache-type-k", "tq3_0",
        "--cache-type-v", "tq3_0",
    ]
    if mode != "off":
        cmd += [
            "--prefill-drafter", PFLASH_DRAFTER,
            "--prefill-keep-ratio", str(keep_ratio),
        ]

    print(f"[runner] starting server mode={mode} keep={keep_ratio}", flush=True)
    print(f"[runner]   cmd: {' '.join(cmd)}", flush=True)

    if dry_run:
        print("[runner]   (dry-run, not launching)", flush=True)
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd, stdout=lf, stderr=lf,
            preexec_fn=os.setpgrp,
        )
    return proc.pid


def kill_server(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(3)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    print(f"[runner] server pid={pid} killed", flush=True)


def write_mini_grid(base: Path, ctx_list, keep_list, mode_list, tasks, n) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    path = base / "_mini_grid.yaml"
    grid = {
        "ctx_tokens": ctx_list,
        "keep_ratio": keep_list,
        "mode": mode_list,
        "n_per_cell": n,
        "tasks": tasks,
    }
    path.write_text(yaml.dump(grid))
    return path


def run_sweep_phase(
    mini_grid_path: Path,
    out_dir: Path,
    server_url: str,
    dry_run: bool,
) -> list[dict]:
    """Run sweep_envelope.py for a mini-grid; return frontier rows."""
    cmd = [
        sys.executable, str(_SWEEP_SCRIPT),
        "--grid", str(mini_grid_path),
        "--out", str(out_dir),
        "--server-url", server_url,
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n[runner] running sweep: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[runner] sweep returned {result.returncode}", flush=True)

    frontier_path = out_dir / "frontier.json"
    if frontier_path.exists():
        with open(frontier_path) as f:
            return json.load(f)
    return []


def main():
    ap = argparse.ArgumentParser(description="Multi-phase envelope sweep runner.")
    ap.add_argument("--grid", type=Path, required=True, help="Grid YAML file.")
    ap.add_argument("--out",  type=Path, required=True, help="Base output directory.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(args.grid) as f:
        grid = yaml.safe_load(f)

    ctx_list   = grid.get("ctx_tokens", [4096, 8192, 16384, 32768, 65536])
    keep_list  = grid.get("keep_ratio",  [0.025, 0.05, 0.10, 0.20])
    n_per_cell = grid.get("n_per_cell",  5)
    tasks      = grid.get("tasks", ["niah_single", "vt", "fwe", "mqa"])
    server_url = f"http://{SERVER_HOST}:{SERVER_PORT}"

    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # ── Phase 1: OFF baseline ─────────────────────────────────────────────
    # Keep_ratio doesn't matter for OFF; run once covering all ctx.
    phase1_log = args.out / "server_off.log"
    phase1_out = args.out / "_phase_off"
    mini_grid = write_mini_grid(phase1_out, ctx_list, [0.10], ["off"], tasks, n_per_cell)

    pid = start_server("off", 0.10, phase1_log, args.dry_run)
    if not args.dry_run:
        up = _wait_for_server(phase1_log)
        if not up:
            print("[runner] FATAL: server (off) failed to start. Aborting.", flush=True)
            kill_server(pid)
            sys.exit(1)
        print("[runner] server (off) is up.", flush=True)

    rows_off = run_sweep_phase(mini_grid, phase1_out, server_url, args.dry_run)
    all_rows.extend(rows_off)
    kill_server(pid)
    pid = None
    if not args.dry_run:
        time.sleep(5)   # let GPU memory drain

    # ── Phase 2: ALWAYS sweep over keep_ratio values ──────────────────────
    for keep in keep_list:
        keep_str = f"{keep:.4f}".rstrip("0").rstrip(".")
        phase_log = args.out / f"server_always_{keep_str}.log"
        phase_out = args.out / f"_phase_always_{keep_str}"
        mini_grid = write_mini_grid(phase_out, ctx_list, [keep], ["always"], tasks, n_per_cell)

        pid = start_server("always", keep, phase_log, args.dry_run)
        if not args.dry_run:
            up = _wait_for_server(phase_log)
            if not up:
                print(f"[runner] server always/keep={keep} failed. Skipping phase.", flush=True)
                kill_server(pid)
                pid = None
                continue
            print(f"[runner] server (always keep={keep}) is up.", flush=True)

        rows = run_sweep_phase(mini_grid, phase_out, server_url, args.dry_run)
        all_rows.extend(rows)
        kill_server(pid)
        pid = None
        if not args.dry_run:
            time.sleep(5)

    # ── Smoke-test step C: run 400-error test ─────────────────────────────
    # Not done here; smoke test is a separate step.

    # ── Merge all rows ────────────────────────────────────────────────────
    frontier_path = args.out / "frontier.json"
    frontier_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n[runner] frontier -> {frontier_path} ({len(all_rows)} rows)")
    print("[runner] complete")


if __name__ == "__main__":
    main()
