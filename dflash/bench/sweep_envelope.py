#!/usr/bin/env python3
"""Operating-envelope sweep driver.

Runs bench_niah_cpp.py (for niah_single) and ruler_diag3.py (for vt/fwe/mqa)
over a grid defined by a YAML config file.

Grid file example (default: dflash/bench/envelope_grid.yaml):
  ctx_tokens:   [4096, 8192, 16384, 32768, 65536]
  keep_ratio:   [0.025, 0.05, 0.10, 0.20]
  mode:         [off, always]
  n_per_cell:   5
  tasks:        [niah_single, vt, fwe, mqa]

Usage:
  python sweep_envelope.py                        # uses default grid + today's date
  python sweep_envelope.py --grid my_grid.yaml --out dflash/bench/results/my_run
  python sweep_envelope.py --dry-run              # print plan, no requests
  python sweep_envelope.py --server-url http://127.0.0.1:8080

The script does NOT launch the server. Assumes the server is already running.
Uses ThreadPoolExecutor(max_workers=1) — single GPU, no concurrency.
"""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("[sweep_envelope] pyyaml not installed. Run: pip install pyyaml")

_HERE = Path(__file__).resolve().parent
_RULER_SCRIPT = _HERE / "ruler_diag3.py"
_NIAH_BENCH   = _HERE.parent.parent / "pflash" / "tests" / "bench_niah_cpp.py"
_NIAH_GEN     = _HERE.parent.parent / "pflash" / "tests" / "niah_gen.py"

DEFAULT_GRID = {
    "ctx_tokens":  [4096, 8192, 16384, 32768, 65536],
    "keep_ratio":  [0.025, 0.05, 0.10, 0.20],
    "mode":        ["off", "always"],
    "n_per_cell":  5,
    "tasks":       ["niah_single", "vt", "fwe", "mqa"],
}


def _load_grid(path: Path | None) -> dict:
    if path is None:
        default_path = _HERE / "envelope_grid.yaml"
        if default_path.exists():
            with open(default_path) as f:
                return {**DEFAULT_GRID, **yaml.safe_load(f)}
        return DEFAULT_GRID
    with open(path) as f:
        return {**DEFAULT_GRID, **yaml.safe_load(f)}


def _cell_dir(base: Path, task: str, ctx: int, keep: float, mode: str) -> Path:
    keep_str = f"{keep:.4f}".rstrip("0").rstrip(".")
    return base / f"{task}_{ctx}_{keep_str}_{mode}"


def _run_ruler(
    task: str, ctx: int, keep: float, mode: str, n: int,
    out_dir: Path, server_url: str, seed: int, dry_run: bool,
) -> dict | None:
    cmd = [
        sys.executable, str(_RULER_SCRIPT),
        "--task", task,
        "--ctx-tokens", str(ctx),
        "--n", str(n),
        "--out", str(out_dir),
        "--seed", str(seed),
        "--mode", mode,
        "--keep-ratio", str(keep),
        "--server-url", server_url,
    ]
    if dry_run:
        print("  DRY-RUN:", " ".join(cmd))
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [warn] ruler_diag3 exited {result.returncode} for {task}/{ctx}/{keep}/{mode}")
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def _run_niah_single(
    ctx: int, keep: float, mode: str, n: int,
    out_dir: Path, server_url: str, seed: int, dry_run: bool,
) -> dict | None:
    # bench_niah_cpp.py talks to the daemon directly (not the HTTP server) and
    # has a different CLI shape. For the HTTP-server sweep we generate cases
    # with niah_gen.py and POST them via a minimal inline client that mirrors
    # ruler_diag3.py's approach.
    # Decision: re-use ruler_diag3.py's HTTP path with a synthetic NIAH prompt
    # built by niah_gen.py, since bench_niah_cpp.py requires the daemon binary
    # and GGUF files which may not be present in all environments.
    # Fallback: if bench_niah_cpp.py args are unavailable, skip with a warning.
    niah_gen = _NIAH_GEN
    if not niah_gen.exists():
        print(f"  [skip] niah_single: {niah_gen} not found")
        return None

    cases_path = out_dir / "cases.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cmd = [
        sys.executable, str(niah_gen),
        "--n", str(n),
        "--ctx", str(ctx),
        "--out", str(cases_path),
        "--seed-base", str(seed),
    ]
    if dry_run:
        print("  DRY-RUN:", " ".join(gen_cmd))
        return None
    result = subprocess.run(gen_cmd, capture_output=False)
    if result.returncode != 0 or not cases_path.exists():
        print(f"  [warn] niah_gen failed for ctx={ctx}")
        return None

    # Post each generated case to the HTTP server.
    import requests as _req
    import time as _time

    with open(cases_path) as f:
        cases = [json.loads(line) for line in f]

    scores: list[float] = []
    walls: list[float] = []
    system = "You are a careful long-context assistant. Answer in one short line, no extra prose."

    for i, case in enumerate(cases):
        url = server_url.rstrip("/") + "/v1/messages"
        payload: dict[str, Any] = {
            "model": "local",
            "system": system,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": 128,
            "extra_body": {"pflash_mode": mode, "keep_ratio": keep},
        }
        t0 = _time.time()
        try:
            resp = _req.post(url, json=payload, timeout=300)
            wall_s = _time.time() - t0
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [warn] niah_single case {i}: {exc}")
            continue

        content = data.get("content", [])
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
        sc = 1.0 if case["answer"] in text else 0.0
        scores.append(sc)
        walls.append(wall_s)
        (out_dir / f"case_{i:04d}.raw.json").write_text(json.dumps({
            "case_idx": i, "prompt_len": case["n_tokens"], "answer": case["answer"],
            "response_text": text, "score": sc, "wall_s": wall_s,
            "mode_used": mode, "keep_ratio": keep, "ctx_tokens": ctx,
        }, indent=2))

    if not scores:
        return None

    accuracy = sum(scores) / len(scores)
    wall_sorted = sorted(walls)
    p50 = wall_sorted[len(wall_sorted) // 2]
    p95_idx = max(0, int(math.ceil(0.95 * len(wall_sorted))) - 1)
    p95 = wall_sorted[p95_idx]
    summary = {
        "task": "niah_single", "ctx_tokens": ctx, "n_cases": len(scores),
        "accuracy": accuracy, "wall_p50": p50, "wall_p95": p95,
        "mode": mode, "keep_ratio": keep,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Sweep driver for operating-envelope study.")
    ap.add_argument("--grid", type=Path, default=None,
                    help="Path to YAML grid file (default: envelope_grid.yaml in script dir).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Base output directory (default: dflash/bench/results/<today>_envelope).")
    ap.add_argument("--server-url", default="http://127.0.0.1:8080",
                    help="Base URL of the running C++ HTTP server.")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed (default 42).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan without making any HTTP requests.")
    args = ap.parse_args()

    grid = _load_grid(args.grid)
    ctx_list   = grid.get("ctx_tokens", DEFAULT_GRID["ctx_tokens"])
    keep_list  = grid.get("keep_ratio", DEFAULT_GRID["keep_ratio"])
    mode_list  = grid.get("mode", DEFAULT_GRID["mode"])
    n_per_cell = grid.get("n_per_cell", DEFAULT_GRID["n_per_cell"])
    tasks      = grid.get("tasks", DEFAULT_GRID["tasks"])

    base_out = args.out or (
        _HERE / "results" / f"{date.today().isoformat()}_envelope"
    )
    base_out = Path(base_out)

    # Build full cell list.
    cells = [
        (task, ctx, keep, mode)
        for task in tasks
        for ctx in ctx_list
        for keep in keep_list
        for mode in mode_list
    ]

    print(f"[sweep] {len(cells)} cells × {n_per_cell} cases each = "
          f"{len(cells) * n_per_cell} total requests")
    print(f"[sweep] output base: {base_out}")
    print(f"[sweep] server: {args.server_url}")
    if args.dry_run:
        print("[sweep] --dry-run: printing commands only\n")

    frontier_rows: list[dict] = []

    def run_cell(cell_args):
        task, ctx, keep, mode = cell_args
        cell_dir = _cell_dir(base_out, task, ctx, keep, mode)
        seed_i = args.seed + hash((task, ctx, keep, mode)) % 10000
        # Skip cells that already have a complete summary.json (resume support).
        if not args.dry_run and (cell_dir / "summary.json").exists():
            try:
                with open(cell_dir / "summary.json") as f:
                    existing = json.load(f)
                print(f"[sweep] skip (cached) task={task} ctx={ctx} keep={keep} mode={mode} "
                      f"acc={existing.get('accuracy')}", flush=True)
                return {
                    "task": task, "ctx": ctx, "keep": keep, "mode": mode,
                    "accuracy": existing.get("accuracy"),
                    "wall_p50": existing.get("wall_p50"),
                    "wall_p95": existing.get("wall_p95"),
                    "n_cases": existing.get("n_cases"),
                }
            except Exception:
                pass  # corrupted summary — re-run
        print(f"[sweep] start task={task} ctx={ctx} keep={keep} mode={mode}", flush=True)
        if task == "niah_single":
            summary = _run_niah_single(
                ctx, keep, mode, n_per_cell, cell_dir, args.server_url, seed_i, args.dry_run
            )
        else:
            summary = _run_ruler(
                task, ctx, keep, mode, n_per_cell, cell_dir, args.server_url, seed_i, args.dry_run
            )
        if summary:
            row = {
                "task": task, "ctx": ctx, "keep": keep, "mode": mode,
                "accuracy": summary.get("accuracy"),
                "wall_p50": summary.get("wall_p50"),
                "wall_p95": summary.get("wall_p95"),
                "n_cases": summary.get("n_cases"),
            }
            print(f"[sweep] done task={task} ctx={ctx} keep={keep} mode={mode} "
                  f"acc={row['accuracy']:.3f}" if row["accuracy"] is not None
                  else f"[sweep] done {task}/{ctx}/{keep}/{mode} (no result)", flush=True)
            return row
        return None

    # max_workers=1: single-GPU server, no concurrency.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = {pool.submit(run_cell, c): c for c in cells}
        for fut in as_completed(futures):
            row = fut.result()
            if row:
                frontier_rows.append(row)

    if frontier_rows:
        frontier_path = base_out / "frontier.json"
        base_out.mkdir(parents=True, exist_ok=True)
        frontier_path.write_text(json.dumps(frontier_rows, indent=2))
        print(f"\n[sweep] frontier -> {frontier_path} ({len(frontier_rows)} rows)")
    else:
        if not args.dry_run:
            print("[sweep] no results collected")

    print("[sweep] complete")


if __name__ == "__main__":
    main()
