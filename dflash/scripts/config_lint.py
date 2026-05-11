#!/usr/bin/env python3
"""config_lint.py — validate all profiles and backends.

Usage:
    config_lint.py [--profile NAME] [--strict]

Exit codes:
    0 — all valid (warnings may be printed)
    1 — one or more errors
"""
import argparse
import sys
from pathlib import Path


def _find_git_root(start: Path) -> Path:
    p = start.resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Could not find git root from {start}")


def lint_profile(profile_path: Path, git_root: str, profiles_dir: str, strict: bool):
    """Lint a single profile. Returns (errors, warnings)."""
    from dflash.scripts.configlib.loader import load_profile, ProfileError
    from dflash.scripts.configlib.validate import validate_profile

    name = profile_path.stem
    try:
        profile = load_profile(profile_path, git_root=git_root, profiles_dir=profiles_dir)
    except ProfileError as exc:
        return [f"[{name}] Load error: {exc}"], []

    errors, warnings = validate_profile(
        profile,
        profile_name=profile_path.name,
        strict=strict,
        git_root=git_root,
    )
    return errors, warnings


def lint_backend(backend_path: Path, git_root: str):
    """Lint a single backend. Returns (errors, warnings)."""
    from dflash.scripts.configlib.backends import load_backend, BackendError

    name = backend_path.stem
    try:
        load_backend(backend_path, git_root=git_root)
    except BackendError as exc:
        return [f"[{name}] Backend error: {exc}"], []
    return [], []


def main():
    parser = argparse.ArgumentParser(description="Lint lucebox-hub config profiles and backends")
    parser.add_argument("--profile", help="Lint only this profile name (stem)")
    parser.add_argument("--strict", action="store_true", help="Escalate warnings to errors")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    try:
        git_root = str(_find_git_root(script_dir))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, git_root)

    profiles_dir = Path(git_root) / "configs" / "profiles"
    backends_dir = Path(git_root) / "configs" / "backends"

    total_errors = []
    total_warnings = []

    if args.profile:
        # Lint single profile
        profile_path = profiles_dir / f"{args.profile}.toml"
        if not profile_path.exists():
            print(f"ERROR: Profile {args.profile!r} not found at {profile_path}", file=sys.stderr)
            sys.exit(1)
        errs, warns = lint_profile(profile_path, git_root, str(profiles_dir), args.strict)
        total_errors.extend(errs)
        total_warnings.extend(warns)
    else:
        # Lint all profiles (skip base.toml — template, no provenance)
        for profile_path in sorted(profiles_dir.glob("*.toml")):
            if profile_path.stem == "base":
                # Still parse to catch TOML errors, but skip provenance validation
                try:
                    from dflash.scripts.configlib.loader import load_profile, ProfileError
                    load_profile(profile_path, git_root=git_root, profiles_dir=str(profiles_dir))
                    print(f"  base.toml: OK (template, provenance skipped)")
                except Exception as exc:
                    total_errors.append(f"[base] Parse error: {exc}")
                continue
            errs, warns = lint_profile(profile_path, git_root, str(profiles_dir), args.strict)
            total_errors.extend(errs)
            total_warnings.extend(warns)

        # Lint all backends
        for backend_path in sorted(backends_dir.glob("*.toml")):
            errs, warns = lint_backend(backend_path, git_root)
            total_errors.extend(errs)
            total_warnings.extend(warns)

    for w in total_warnings:
        print(f"WARNING: {w}")
    for e in total_errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if total_errors:
        print(f"\n{len(total_errors)} error(s), {len(total_warnings)} warning(s). FAIL.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n0 errors, {len(total_warnings)} warning(s). OK.")
        sys.exit(0)


if __name__ == "__main__":
    main()
