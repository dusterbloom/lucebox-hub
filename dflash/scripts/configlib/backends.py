"""Backend TOML loader and argv builder."""
import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib


class BackendError(Exception):
    """Raised for any backend loading or validation failure."""


# Required flag keys for each spec type beyond "none"
_SPEC_FLAG_REQUIREMENTS = {
    "mtp": {"spec_model", "spec_gamma"},
    "dflash": {"draft_model", "draft_max"},
}

# Runtime keys that are boolean flags (added as bare flags when True)
_BOOL_RUNTIME_KEYS = {"pflash", "ignore_eos", "flash_attn"}


def load_backend(
    backend_path,
    git_root: str,
) -> dict:
    """Load and validate a backend TOML file.

    Args:
        backend_path: Path-like to the backend .toml file.
        git_root: Repository root for resolving in_tree paths.

    Returns:
        Backend dict with additional key ``resolved_binary``.

    Raises:
        BackendError: On any validation or resolution failure.
    """
    backend_path = Path(backend_path)
    stem = backend_path.stem

    if not backend_path.exists():
        raise BackendError(f"Backend file not found: {backend_path}")

    try:
        data = tomllib.loads(backend_path.read_bytes().decode())
    except Exception as exc:
        raise BackendError(f"TOML parse error in {backend_path.name!r}: {exc}") from exc

    # name == filename stem
    name = data.get("name", "")
    if name != stem:
        raise BackendError(
            f"Backend name {name!r} does not match filename stem {stem!r} in {backend_path.name!r}"
        )

    # binary: exactly one of in_tree or env_var
    binary = data.get("binary", {})
    in_tree = binary.get("in_tree")
    env_var = binary.get("env_var")

    if in_tree and env_var:
        raise BackendError(
            f"[{stem}] binary.in_tree and binary.env_var are mutually exclusive"
        )
    if not in_tree and not env_var:
        raise BackendError(
            f"[{stem}] [binary] must have exactly one of in_tree or env_var"
        )

    # Resolve binary path
    if in_tree:
        resolved = Path(git_root) / in_tree if not Path(in_tree).is_absolute() else Path(in_tree)
        if not resolved.exists():
            raise BackendError(
                f"[{stem}] in_tree binary not found: {in_tree!r} (resolved to {resolved})"
            )
        resolved_binary = str(resolved)
    else:
        # env_var
        val = os.environ.get(env_var)
        if val is None:
            raise BackendError(
                f"[{stem}] env_var {env_var!r} is not set — cannot resolve binary path"
            )
        if not Path(val).exists():
            raise BackendError(
                f"[{stem}] binary from ${env_var}={val!r} does not exist"
            )
        resolved_binary = val

    # Validate required flags for declared spec_types
    spec_types = data.get("supports", {}).get("spec_types", [])
    flags = data.get("flags", {})
    for spec_type in spec_types:
        required = _SPEC_FLAG_REQUIREMENTS.get(spec_type, set())
        missing = required - set(flags.keys())
        if missing:
            raise BackendError(
                f"[{stem}] Missing required flags for spec_type={spec_type!r}: {sorted(missing)}"
            )

    result = dict(data)
    result["resolved_binary"] = resolved_binary
    return result


def build_argv(backend: dict, profile: dict) -> list[str]:
    """Build the command-line argv from a loaded backend and merged profile.

    Args:
        backend: dict returned by load_backend (must have resolved_binary).
        profile: merged profile dict.

    Returns:
        List of strings [binary, flag, value, ...] suitable for os.execvp.
    """
    flags = backend.get("flags", {})
    runtime = profile.get("runtime", {})
    model = profile.get("model", {})
    spec = runtime.get("spec", {})
    method = spec.get("method", "none")

    argv = [backend["resolved_binary"]]

    # model (always)
    if "model" in flags:
        argv += [flags["model"], str(model["target"])]

    # ctx
    if "ctx" in flags:
        argv += [flags["ctx"], str(runtime["ctx"])]

    # kv_k, kv_v
    if "kv_k" in flags:
        argv += [flags["kv_k"], str(runtime["kv_k"])]
    if "kv_v" in flags:
        argv += [flags["kv_v"], str(runtime["kv_v"])]

    # Optional scalar runtime flags
    for key in ("temp", "seed", "n_predict", "batch", "ubatch"):
        if key in flags and key in runtime:
            argv += [flags[key], str(runtime[key])]

    # Boolean flags — add bare flag only when True
    for key in _BOOL_RUNTIME_KEYS:
        if key in flags and runtime.get(key) is True:
            argv.append(flags[key])

    # Speculative decode
    if method == "mtp":
        if "spec_model" in flags:
            argv += [flags["spec_model"], str(model.get("mtp_assistant", ""))]
        if "spec_gamma" in flags:
            argv += [flags["spec_gamma"], str(spec.get("gamma", 1))]

    elif method == "dflash":
        if "draft_model" in flags:
            argv += [flags["draft_model"], str(model.get("dflash_draft", ""))]
        if "draft_max" in flags:
            argv += [flags["draft_max"], str(spec.get("draft_max", 4))]

    return argv
