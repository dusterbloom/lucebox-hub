#!/usr/bin/env python3
"""profile_run.py — run a lucebox-hub profile.

Usage:
    profile_run.py --profile NAME [--override key.path=value ...] [--dry-run] [--print-cmd]
"""
import argparse
import os
import sys
from pathlib import Path


def _find_git_root(start: Path) -> Path:
    """Walk up to find the git root (directory containing .git)."""
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Could not find git root from {start}")


def _dot_path_set(obj: dict, dot_path: str, value):
    """Set a nested dict value given a dot-separated path."""
    keys = dot_path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    obj[keys[-1]] = value


def _coerce(value: str):
    """Auto-coerce a string value to bool, int, float, or leave as str."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Run a lucebox-hub inference profile"
    )
    parser.add_argument("--profile", required=True, help="Profile name (stem of TOML file in configs/profiles/)")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Dot-path override (e.g. runtime.ctx=131072)")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, exit 0")
    parser.add_argument("--print-cmd", action="store_true", help="Print resolved argv, exit 0")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    try:
        git_root = _find_git_root(script_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(git_root))

    from dflash.scripts.configlib.loader import load_profile, ProfileError
    from dflash.scripts.configlib.validate import validate_profile
    from dflash.scripts.configlib.backends import load_backend, build_argv, BackendError

    profiles_dir = git_root / "configs" / "profiles"
    backends_dir = git_root / "configs" / "backends"
    profile_path = profiles_dir / f"{args.profile}.toml"

    # Load profile
    try:
        profile = load_profile(profile_path, git_root=str(git_root), profiles_dir=str(profiles_dir))
    except ProfileError as exc:
        print(f"ERROR loading profile {args.profile!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Apply overrides
    for ov in args.override:
        if "=" not in ov:
            print(f"ERROR: --override {ov!r} must be in KEY=VALUE format", file=sys.stderr)
            sys.exit(1)
        key, _, val = ov.partition("=")
        _dot_path_set(profile, key, _coerce(val))

    # Validate
    errors, warnings = validate_profile(profile, profile_name=args.profile, git_root=str(git_root))
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"Profile {args.profile!r} is valid.")
        sys.exit(0)

    # Load backend
    backend_name = profile.get("backend", "")
    backend_path = backends_dir / f"{backend_name}.toml"
    try:
        backend = load_backend(backend_path, git_root=str(git_root))
    except BackendError as exc:
        print(f"ERROR loading backend {backend_name!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Build argv
    try:
        argv = build_argv(backend, profile)
    except Exception as exc:
        print(f"ERROR building command: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.print_cmd:
        for tok in argv:
            print(tok)
        sys.exit(0)

    # Execute
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
