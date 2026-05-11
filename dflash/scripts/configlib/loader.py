"""Profile TOML loader with inheritance, env expansion, and path validation."""
import os
import re
import copy
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib  # Python 3.11+


class ProfileError(Exception):
    """Raised for any profile loading or validation failure."""


# Regex to match ${VAR} or ${VAR:-default}
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env(value: str, profile_name: str) -> tuple[str, bool]:
    """Expand ${VAR} and ${VAR:-default} in value.

    Returns (expanded, had_env_var) where had_env_var is True if any ${...}
    was present in the original string (even after expansion).
    """
    had_env_var = bool(_ENV_RE.search(value))

    def _replace(m):
        var = m.group(1)
        default = m.group(2)
        val = os.environ.get(var)
        if val is None:
            if default is not None:
                return default
            raise ProfileError(
                f"Unset environment variable ${{{var}}} referenced in profile {profile_name!r}"
            )
        return val

    return _ENV_RE.sub(_replace, value), had_env_var


def _resolve_path(raw: str, git_root: str, profile_name: str) -> str:
    """Resolve a path string according to spec rules.

    1. Expand ${VAR} / ${VAR:-default}.
    2. Expand leading ~.
    3. If resolved starts with / AND raw had no ${...} AND raw did not start with ~ -> raise.
    4. Otherwise resolve relative to git_root.
    """
    had_tilde = raw.startswith("~")
    expanded, had_env_var = _expand_env(raw, profile_name)
    expanded = os.path.expanduser(expanded)

    if expanded.startswith("/") and not had_env_var and not had_tilde:
        raise ProfileError(
            f"Hardcoded absolute path {raw!r} in profile {profile_name!r}. "
            "Use ${{VAR}}/... or a relative path instead."
        )

    if os.path.isabs(expanded):
        return expanded  # env-expanded absolute or tilde-expanded — allowed

    # Relative — resolve against git root
    return str(Path(git_root) / expanded)


def _is_path_key(key: str) -> bool:
    """Heuristic: keys whose values should be treated as paths."""
    path_keys = {"target", "mtp_assistant", "dflash_draft", "source_log"}
    return key in path_keys


def _resolve_paths_in(obj, git_root: str, profile_name: str, resolve_paths: bool = True):
    """Recursively walk obj and resolve path-like string values."""
    if isinstance(obj, dict):
        return {
            k: (
                _resolve_path(v, git_root, profile_name)
                if resolve_paths and isinstance(v, str) and _is_path_key(k)
                else _resolve_paths_in(v, git_root, profile_name, resolve_paths)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_resolve_paths_in(i, git_root, profile_name, resolve_paths) for i in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_profile(
    profile_path,
    git_root: str,
    profiles_dir: str = None,
    _seen: set = None,
) -> dict:
    """Load and merge a profile TOML, resolving inheritance and paths.

    Args:
        profile_path: Path-like to the profile TOML file.
        git_root: Absolute path to the repository root (used for relative paths).
        profiles_dir: Directory containing profiles for extends resolution.
                      Defaults to the directory of profile_path.
        _seen: Internal set for circular dependency detection.

    Returns:
        Merged profile dict with all paths resolved.

    Raises:
        ProfileError: On any loading, parsing, or path validation error.
    """
    profile_path = Path(profile_path)
    profile_name = profile_path.name

    if not profile_path.exists():
        raise ProfileError(f"Profile not found: {profile_path}")

    if _seen is None:
        _seen = set()

    canonical = str(profile_path.resolve())
    if canonical in _seen:
        raise ProfileError(
            f"Circular extends chain detected involving {profile_name!r}"
        )
    _seen = _seen | {canonical}

    # Parse TOML
    try:
        raw_bytes = profile_path.read_bytes()
        data = tomllib.loads(raw_bytes.decode())
    except Exception as exc:
        raise ProfileError(f"TOML parse error in {profile_name!r}: {exc}") from exc

    # Handle inheritance
    extends = data.get("extends")
    if extends and extends != "null":
        if profiles_dir is None:
            profiles_dir = str(profile_path.parent)
        parent_path = Path(profiles_dir) / f"{extends}.toml"
        parent = load_profile(
            parent_path,
            git_root=git_root,
            profiles_dir=profiles_dir,
            _seen=_seen,
        )
        # Merge: parent is base, child overrides
        merged = _deep_merge(parent, data)
        merged["extends"] = extends
    else:
        merged = data

    # Resolve paths in the merged result
    resolved = _resolve_paths_in(merged, git_root, profile_name)
    return resolved
