#!/usr/bin/env python3
"""
Pass B: Agentic harness for ee7 broad validation — baseline / ee14 / ee7 comparison.

Drives a configurable harness client through three conditions to measure
drafter_fwd speedup and OK_DONE retention.  Originally validated on claude_code
(commit 764b18e); extended to all 11 harness clients.

Usage:
  python3 server/bench/run_agentic_ee7_passbv.py --client claude_code
  python3 server/bench/run_agentic_ee7_passbv.py --client codex
  python3 server/bench/run_agentic_ee7_passbv.py --client all

Results land in server/bench/results/<date>_ee7_passbv/<client>/SUMMARY_PASS_B.md.

backend_pair is a wrapper over other clients; it has no single-script invocation
and is skipped when --client all.  Pass --include-backend-pair to override.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# _harness_lib lives in the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness_lib import (
    ALL_CLIENTS,
    DEFAULT_SERVER_ENV,
    REPO,
    build_env,
    run_client,
)

RESULTS_BASE = REPO / "server/bench/results"
_SKIP_BY_DEFAULT = {"backend_pair"}

CONDITIONS = [
    {
        "name": "baseline",
        "env_extra": {
            "PFLASH_DRAFTER_EARLY_EXIT_N": "",
            "PFLASH_DRAFTER_SCORE_LAYERS": "",
        },
    },
    {
        "name": "ee14",
        "env_extra": {"PFLASH_DRAFTER_EARLY_EXIT_N": "14"},
    },
    {
        "name": "ee7",
        "env_extra": {
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
]


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
        "--results-dir",
        default=None,
        help="Override output directory (default: server/bench/results/<date>_ee7_passbv)",
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
        "--conditions",
        nargs="+",
        choices=["baseline", "ee14", "ee7"],
        default=["baseline", "ee14", "ee7"],
        help="Which conditions to run (default: all three)",
    )
    p.add_argument(
        "--dflash-server-bin",
        default=None,
        help="Path to dflash_server C++ binary (default: <repo>/server/build/dflash_server)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-condition wall timeout in seconds (default: 900)",
    )
    return p.parse_args()


def clients_for(args: argparse.Namespace) -> list[str]:
    if args.client == "all":
        skip = _SKIP_BY_DEFAULT if args.skip_backend_pair else set()
        return [c for c in ALL_CLIENTS if c not in skip]
    if args.client not in ALL_CLIENTS:
        print(
            f"[passbv] unknown client '{args.client}'. "
            f"Valid: {', '.join(ALL_CLIENTS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return [args.client]


def run_condition_for_client(
    client: str,
    cond: dict,
    results_dir: Path,
    args: argparse.Namespace,
) -> dict:
    name = cond["name"]
    stamp = f"passbv-{client}-{name}-{int(time.time())}"
    run_dir = results_dir / client / name

    overrides: dict[str, str] = {
        "PORT": "19099",
    }
    if args.dflash_server_bin:
        overrides["DFLASH_SERVER_BIN"] = args.dflash_server_bin

    # Merge condition overrides; empty-string values signal removal
    cond_extra = cond.get("env_extra", {})
    overrides.update({k: v for k, v in cond_extra.items() if v != ""})

    env = build_env(overrides)

    # Remove early-exit vars for baseline so the server runs without them
    if name == "baseline":
        env.pop("PFLASH_DRAFTER_EARLY_EXIT_N", None)
        env.pop("PFLASH_DRAFTER_SCORE_LAYERS", None)

    print(f"\n[passbv] client={client} condition={name} stamp={stamp}", flush=True)
    result = run_client(
        client=client,
        env=env,
        run_name=stamp,
        run_dir=str(run_dir),
        timeout_s=args.timeout,
    )
    result["condition"] = name
    return result


def run_for_client(
    client: str,
    results_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    selected = [c for c in CONDITIONS if c["name"] in args.conditions]
    results = []
    baseline_fwd = None

    for cond in selected:
        r = run_condition_for_client(client, cond, results_dir, args)
        results.append(r)
        if cond["name"] == "baseline" and r.get("drafter_fwd_s"):
            baseline_fwd = r["drafter_fwd_s"]

    write_client_summary(client, results, baseline_fwd, results_dir)
    return results


def write_client_summary(
    client: str,
    results: list[dict],
    baseline_fwd: float | None,
    results_dir: Path,
) -> None:
    print(f"\n=== PASS B TABLE — {client} ===")
    print(
        f"{'condition':>12}  {'drafter_fwd':>12}  {'accept_rate':>12}  "
        f"{'OK_DONE':>8}  {'speedup':>8}"
    )
    rows = []
    for r in results:
        cond = r["condition"]
        df = f"{r['drafter_fwd_s']:.2f}s" if r.get("drafter_fwd_s") else "N/A"
        ar = r.get("accept_rate") or "N/A"
        ok = "YES" if r.get("ok_done") else "NO"
        speedup = "1.00x"
        if cond != "baseline" and r.get("drafter_fwd_s") and baseline_fwd:
            speedup = f"{baseline_fwd / r['drafter_fwd_s']:.2f}x"
        print(f"{cond:>12}  {df:>12}  {ar:>12}  {ok:>8}  {speedup:>8}")
        rows.append(
            {
                "condition": cond,
                "drafter_fwd": df,
                "accept_rate": ar,
                "OK_DONE": ok,
                "speedup": speedup,
            }
        )

    out_dir = results_dir / client
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "SUMMARY_PASS_B.md"
    with out.open("w") as f:
        f.write(f"# Pass B: Agentic Harness — ee7 vs ee14 vs baseline ({client})\n\n")
        f.write(
            "| condition | drafter_fwd | accept_rate | OK_DONE | speedup_vs_baseline |\n"
        )
        f.write("|---|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['condition']} | {row['drafter_fwd']} | "
                f"{row['accept_rate']} | {row['OK_DONE']} | {row['speedup']} |\n"
            )
    print(f"[passbv] summary -> {out}", flush=True)


def write_global_summary(
    all_results: dict[str, list[dict]],
    results_dir: Path,
) -> None:
    out = results_dir / "SUMMARY_PASS_B_ALL.md"
    with out.open("w") as f:
        f.write("# Pass B: Cross-client — ee7 speedup summary\n\n")
        f.write("| client | baseline_fwd | ee7_fwd | speedup | OK_DONE(ee7) |\n")
        f.write("|--------|-------------|---------|---------|---------------|\n")
        for client, results in all_results.items():
            baseline_fwd = next(
                (r["drafter_fwd_s"] for r in results if r["condition"] == "baseline"),
                None,
            )
            ee7 = next(
                (r for r in results if r["condition"] == "ee7"),
                None,
            )
            if ee7:
                df_b = f"{baseline_fwd:.2f}s" if baseline_fwd else "N/A"
                df_e = f"{ee7['drafter_fwd_s']:.2f}s" if ee7.get("drafter_fwd_s") else "N/A"
                speedup = (
                    f"{baseline_fwd / ee7['drafter_fwd_s']:.2f}x"
                    if baseline_fwd and ee7.get("drafter_fwd_s")
                    else "N/A"
                )
                ok = "YES" if ee7.get("ok_done") else "NO"
                f.write(f"| {client} | {df_b} | {df_e} | {speedup} | {ok} |\n")
    print(f"[passbv] global summary -> {out}", flush=True)


def main() -> None:
    args = parse_args()
    clients = clients_for(args)

    date_tag = datetime.now().strftime("%Y-%m-%d")
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else RESULTS_BASE / f"{date_tag}_ee7_passbv"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    for client in clients:
        all_results[client] = run_for_client(client, results_dir, args)

    if len(clients) > 1:
        write_global_summary(all_results, results_dir)

    any_failed = any(
        not r.get("ok_done")
        for rs in all_results.values()
        for r in rs
        if r["condition"] == "ee7"
    )
    if any_failed:
        failed = [
            c
            for c, rs in all_results.items()
            if not any(r.get("ok_done") for r in rs if r["condition"] == "ee7")
        ]
        print(
            f"[passbv] FAIL: OK_DONE not seen for ee7 on: {', '.join(failed)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("[passbv] ALL ee7 OK_DONE", flush=True)


if __name__ == "__main__":
    main()
