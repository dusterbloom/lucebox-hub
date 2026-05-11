"""Profile validation: structural rules and provenance checks."""
import os
from pathlib import Path


class ProfileError(Exception):
    """Raised when a profile fails validation."""


def validate_profile(
    profile: dict,
    profile_name: str = "unknown",
    strict: bool = False,
    git_root: str = None,
) -> tuple[list[str], list[str]]:
    """Validate a merged profile dict.

    Returns:
        (errors, warnings) — lists of human-readable strings.
        Caller should treat any non-empty errors list as fatal.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def err(msg):
        errors.append(f"[{profile_name}] {msg}")

    def warn(msg):
        warnings.append(f"[{profile_name}] {msg}")

    # --- provenance ---
    prov = profile.get("provenance")
    if not prov:
        err("Missing [provenance] section (required: source_log, measured_at, hardware_id)")
    else:
        for field in ("source_log", "measured_at", "hardware_id"):
            if not prov.get(field):
                err(f"Missing provenance.{field}")

        source_log = prov.get("source_log", "")
        if source_log == "<NEEDS_RUN>":
            msg = "provenance.source_log is <NEEDS_RUN> — run the benchmark and fill in the real log path"
            if strict:
                err(msg)
            else:
                warn(msg)
        elif source_log and git_root:
            log_path = Path(git_root) / source_log if not os.path.isabs(source_log) else Path(source_log)
            if not log_path.exists():
                warn(f"provenance.source_log points to nonexistent file: {source_log!r}")

    # --- expected_floors ---
    floors = profile.get("expected_floors", {})
    if not floors:
        err("Empty or missing [expected_floors] — at least one of decode_tok_s, prefill_tok_s, ttft_ms_max required")

    # --- spec method cross-checks ---
    runtime = profile.get("runtime", {})
    spec = runtime.get("spec", {})
    method = spec.get("method", "none")
    model = profile.get("model", {})

    if method == "mtp":
        if not model.get("mtp_assistant"):
            err("spec.method=mtp requires model.mtp_assistant to be set")

    if method == "dflash":
        if not model.get("dflash_draft"):
            err("spec.method=dflash requires model.dflash_draft to be set")

    return errors, warnings
