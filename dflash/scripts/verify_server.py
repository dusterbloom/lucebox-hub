#!/usr/bin/env python3
"""verify_server.py — verify a running server meets profile floor metrics.

Usage:
    verify_server.py --profile NAME [--base-url URL] [--runs N] [--json-out FILE]

Exit codes:
    0 — all floors met
    1 — config/connection error
    2 — floor(s) failed
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _find_git_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Could not find git root from {start}")


def _http_json(url: str, payload: dict = None, timeout: float = 30.0):
    """Make a JSON HTTP request. Returns (response_dict, elapsed_s, first_byte_s)."""
    body = json.dumps(payload).encode() if payload else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=body, headers=headers)

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            t_first = time.monotonic()
            data = json.loads(resp.read().decode())
            t_end = time.monotonic()
        return data, t_end - t0, t_first - t0
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Request to {url} failed: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Verify server meets profile floor metrics")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-out", metavar="FILE")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    try:
        git_root = _find_git_root(script_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(git_root))

    from dflash.scripts.configlib.loader import load_profile, ProfileError
    from dflash.scripts.configlib.validate import validate_profile

    profiles_dir = git_root / "configs" / "profiles"
    profile_path = profiles_dir / f"{args.profile}.toml"

    try:
        profile = load_profile(profile_path, git_root=str(git_root), profiles_dir=str(profiles_dir))
    except ProfileError as exc:
        print(f"ERROR loading profile {args.profile!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    errors, warnings = validate_profile(profile, profile_name=args.profile, git_root=str(git_root))
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    floors = profile.get("expected_floors", {})
    base_url = args.base_url.rstrip("/")

    # Health check
    try:
        health, _, _ = _http_json(f"{base_url}/health", timeout=5.0)
    except ConnectionError as exc:
        print(f"ERROR: Health check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Completion runs
    prompt = "Hello, world! Please respond briefly."
    decode_rates = []
    ttfts = []

    for i in range(args.runs):
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "stream": False,
        }
        try:
            resp, elapsed, ttft = _http_json(
                f"{base_url}/v1/chat/completions", payload=payload, timeout=60.0
            )
        except ConnectionError as exc:
            print(f"ERROR: Run {i+1} failed: {exc}", file=sys.stderr)
            sys.exit(1)

        usage = resp.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens > 0 and elapsed > 0:
            decode_rates.append(completion_tokens / elapsed)
        ttfts.append(ttft * 1000)  # convert to ms

    avg_decode = sum(decode_rates) / len(decode_rates) if decode_rates else 0.0
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0

    # Compare to floors
    floor_results = {}
    passed = True

    if "decode_tok_s" in floors:
        floor_val = floors["decode_tok_s"]
        ok = avg_decode >= floor_val
        floor_results["decode_tok_s"] = {"measured": avg_decode, "floor": floor_val, "passed": ok}
        if not ok:
            passed = False
            print(f"FAIL: decode_tok_s={avg_decode:.2f} < floor={floor_val}", file=sys.stderr)
        else:
            print(f"PASS: decode_tok_s={avg_decode:.2f} >= floor={floor_val}")

    if "ttft_ms_max" in floors:
        floor_val = floors["ttft_ms_max"]
        ok = avg_ttft <= floor_val
        floor_results["ttft_ms_max"] = {"measured": avg_ttft, "floor": floor_val, "passed": ok}
        if not ok:
            passed = False
            print(f"FAIL: ttft_ms={avg_ttft:.1f} > max={floor_val}", file=sys.stderr)
        else:
            print(f"PASS: ttft_ms={avg_ttft:.1f} <= max={floor_val}")

    result = {
        "profile": args.profile,
        "runs": args.runs,
        "avg_decode_tok_s": avg_decode,
        "avg_ttft_ms": avg_ttft,
        "floors": floor_results,
        "passed": passed,
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.json_out}")

    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
