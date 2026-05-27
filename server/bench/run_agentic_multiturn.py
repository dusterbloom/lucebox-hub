#!/usr/bin/env python3
"""run_agentic_multiturn.py — drive any harness client through an agentic multi-turn task.

Sends a single prompt that requires 3+ reasoning steps (merge_sorted_lists: write,
edge-case, tests, Rust translation).  Captures wall, OK_DONE marker, accept_rate,
drafter_fwd_wall.  This is the P1 evidence script for cross-client ee7 validation.

Usage:
  python3 server/bench/run_agentic_multiturn.py --client claude_code \\
    --output server/bench/results/2026-05-27_full_harness/claude_code/agentic_multiturn

  python3 server/bench/run_agentic_multiturn.py --client codex \\
    --output server/bench/results/2026-05-27_full_harness/codex/agentic_multiturn

  python3 server/bench/run_agentic_multiturn.py --client all \\
    --output server/bench/results/2026-05-27_full_harness

  # Skip backend_pair (structurally a wrapper; handled separately):
  python3 server/bench/run_agentic_multiturn.py --client all --skip-backend-pair \\
    --output server/bench/results/2026-05-27_full_harness

Captures:
  - client wall, OK_DONE marker, accept_rate, drafter_fwd_wall
  - <output>/<client>/agentic_multiturn/{client.log, server.log, metrics.txt}
  - When --client all: top-level SUMMARY.md with per-client row

Port: 19099 (never touches user's :18099).
Lock: flock /tmp/lucebox-bench.lock held per run.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# _harness_lib lives in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness_lib import (
    ALL_CLIENTS,
    BASELINE_ENV_OVERRIDES,
    BENCH_LOCK,
    DEFAULT_SERVER_ENV,
    HARNESS_DIR,
    PFLASH_ENV_OVERRIDES,
    REPO,
    build_env,
    client_out_filename,
    harness_for,
    run_arm,
    run_client,
    write_two_arm_metrics,
)

PROMPT_FILE = HARNESS_DIR / "prompts/agentic_multiturn.txt"

# backend_pair is structurally a wrapper over other clients; it has no single-script
# harness invocation and is skipped unless explicitly targeted.
_SKIP_BY_DEFAULT = {"backend_pair"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--client",
        default="claude_code",
        help=(
            "Client to drive. Pass 'all' to iterate all 11 clients "
            "(backend_pair skipped by default). "
            "Valid names: " + ", ".join(ALL_CLIENTS)
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory. When --client all this is the base dir; "
            "per-client results land in <output>/<client>/agentic_multiturn/. "
            "When a single client, results land directly in <output>/."
        ),
    )
    p.add_argument(
        "--skip-backend-pair",
        action="store_true",
        default=True,
        help="Skip backend_pair when --client all (default: True)",
    )
    p.add_argument(
        "--include-backend-pair",
        dest="skip_backend_pair",
        action="store_false",
        help="Include backend_pair when --client all",
    )
    p.add_argument(
        "--dflash-server-bin",
        default=None,
        help="Path to dflash_server C++ binary (default: <repo>/server/build/dflash_server)",
    )
    p.add_argument(
        "--prompt-file",
        default=str(PROMPT_FILE),
        help="Prompt file to use (default: harness/clients/prompts/agentic_multiturn.txt)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-client wall timeout in seconds (default: 900)",
    )
    return p.parse_args()


def clients_for(args: argparse.Namespace) -> list[str]:
    if args.client == "all":
        skip = _SKIP_BY_DEFAULT if args.skip_backend_pair else set()
        return [c for c in ALL_CLIENTS if c not in skip]
    if args.client not in ALL_CLIENTS:
        print(
            f"[agentic_multiturn] unknown client '{args.client}'. "
            f"Valid: {', '.join(ALL_CLIENTS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return [args.client]


def run_one(client: str, output_dir: Path, args: argparse.Namespace) -> dict:
    """Run baseline + pflash arms for one client; write combined metrics.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    base_overrides: dict[str, str] = {
        "PROMPT_FILE": args.prompt_file,
        "MARKER": "OK_DONE",
        "PORT": "19099",
    }
    if args.dflash_server_bin:
        base_overrides["DFLASH_SERVER_BIN"] = args.dflash_server_bin

    # --- baseline arm (no compression) ---
    print(f"\n[agentic_multiturn] --- {client}: baseline arm ---", flush=True)
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline = run_arm(
        client=client,
        label="baseline",
        arm_env_overrides=BASELINE_ENV_OVERRIDES,
        base_overrides=base_overrides,
        run_dir=baseline_dir,
        timeout_s=args.timeout,
    )

    # --- pflash arm (compression ON, ee7) ---
    print(f"\n[agentic_multiturn] --- {client}: pflash arm ---", flush=True)
    pflash_dir = output_dir / "pflash"
    pflash_dir.mkdir(parents=True, exist_ok=True)
    pflash = run_arm(
        client=client,
        label="pflash",
        arm_env_overrides=PFLASH_ENV_OVERRIDES,
        base_overrides=base_overrides,
        run_dir=pflash_dir,
        timeout_s=args.timeout,
    )

    write_two_arm_metrics(baseline, pflash, output_dir)

    # Return a summary dict for the outer loop
    return {
        "client": client,
        "ok_done": baseline.get("ok_done") and pflash.get("ok_done"),
        "baseline": baseline,
        "pflash": pflash,
        "run_dir": str(output_dir),
    }


def write_summary(results: list[dict], output_base: Path) -> None:
    summary = output_base / "SUMMARY.md"
    with summary.open("w") as f:
        f.write("# Agentic Multi-turn — Cross-client Summary\n\n")
        f.write(
            "| client | OK_DONE (both) | baseline_wall | pflash_wall | "
            "e2e_speedup | accept_rate | drafter_speedup |\n"
        )
        f.write("|--------|----------------|---------------|-------------|"
                "-------------|-------------|----------------|\n")
        for r in results:
            client = r.get("client", "?")
            ok = "YES" if r.get("ok_done") else "NO"
            b = r.get("baseline", {})
            p = r.get("pflash", {})
            bw = f"{b['wall_s']:.1f}s" if b.get("wall_s") else "N/A"
            pw = f"{p['wall_s']:.1f}s" if p.get("wall_s") else "N/A"
            speedup = (
                f"{b['wall_s'] / p['wall_s']:.2f}x"
                if (b.get("wall_s") and p.get("wall_s") and p["wall_s"] > 0)
                else "N/A"
            )
            ar = p.get("accept_rate") or "N/A"
            bdf = b.get("drafter_fwd_s")
            pdf = p.get("drafter_fwd_s")
            ds = (
                f"{bdf / pdf:.2f}x"
                if (bdf and pdf and pdf > 0)
                else "N/A"
            )
            f.write(f"| {client} | {ok} | {bw} | {pw} | {speedup} | {ar} | {ds} |\n")
    print(f"[agentic_multiturn] summary -> {summary}", flush=True)


def main() -> None:
    args = parse_args()
    clients = clients_for(args)
    output_base = Path(args.output)

    results = []
    for client in clients:
        if len(clients) > 1:
            out_dir = output_base / client / "agentic_multiturn"
        else:
            out_dir = output_base

        print(f"\n[agentic_multiturn] === client={client} ===", flush=True)
        r = run_one(client, out_dir, args)
        results.append(r)

    if len(results) > 1:
        write_summary(results, output_base)

    # Exit non-zero if any client failed OK_DONE
    any_failed = any(not r.get("ok_done") for r in results)
    if any_failed:
        failed = [r["client"] for r in results if not r.get("ok_done")]
        print(
            f"[agentic_multiturn] FAIL: OK_DONE not seen for: {', '.join(failed)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[agentic_multiturn] ALL OK_DONE", flush=True)


if __name__ == "__main__":
    main()
