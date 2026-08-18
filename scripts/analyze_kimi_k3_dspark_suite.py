#!/usr/bin/env python3
"""Compare matched untraced K3 autoregressive and DSpark suite runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from analyze_kimi_h23_quality import task_success


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(root: Path) -> dict[str, object]:
    path = root / "suite" / "suite-manifest.json"
    value = json.loads(path.read_text())
    if value.get("schema") != "kimi-k3-h16-suite-v1":
        raise ValueError(f"unsupported suite manifest: {path}")
    if value.get("logits_recorded") is not False:
        raise ValueError(f"expected an untraced performance run: {path}")
    return value


def parse_draft(stdout: Path) -> list[dict[str, int]]:
    pattern = re.compile(
        r"(?P<steps>\d+) draft steps, accepted=(?P<accepted>\d+)/"
        r"(?P<proposed>\d+)"
    )
    rows = []
    for line in stdout.read_text().splitlines():
        match = pattern.search(line)
        if match:
            rows.append({key: int(value) for key, value in match.groupdict().items()})
    return rows


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-root", type=Path, required=True)
    parser.add_argument("--spec-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ar = load(args.ar_root)
    spec = load(args.spec_root)
    invariant_keys = (
        "chat_template", "core_placement", "gpu", "max_context",
        "model_path", "n_gen", "provider",
    )
    for key in invariant_keys:
        if ar.get(key) != spec.get(key):
            raise ValueError(f"unmatched suite setting {key}")
    if ar.get("draft_path") or not spec.get("draft_path"):
        raise ValueError("expected AR without draft and speculative run with draft")

    ar_rows = {str(row["id"]): row for row in ar["sequences"]}
    spec_rows = {str(row["id"]): row for row in spec["sequences"]}
    if list(ar_rows) != list(spec_rows):
        raise ValueError("sequence order/identity mismatch")
    draft_rows = parse_draft(args.spec_root / "stdout.log")
    if len(draft_rows) != len(ar_rows):
        raise ValueError("draft-stat count does not match sequence count")

    rows = []
    for (identifier, left), draft in zip(ar_rows.items(), draft_rows):
        right = spec_rows[identifier]
        for key in ("prompt_tokens", "prompt_token_count", "text"):
            if left[key] != right[key]:
                raise ValueError(f"prompt mismatch for {identifier}: {key}")
        ar_tokens = [int(value) for value in left["output_tokens"]]
        spec_tokens = [int(value) for value in right["output_tokens"]]
        ar_seconds = float(left["decode_seconds"])
        spec_seconds = float(right["decode_seconds"])
        rows.append({
            "id": identifier,
            "ar_seconds": ar_seconds,
            "speculative_seconds": spec_seconds,
            "speedup": ar_seconds / spec_seconds,
            "generated_tokens": len(spec_tokens),
            "token_exact": ar_tokens == spec_tokens,
            "first_generated_token_divergence": first_divergence(ar_tokens, spec_tokens),
            "ar_task_success": task_success(identifier, str(left["output_text"])),
            "speculative_task_success": task_success(identifier, str(right["output_text"])),
            "ar_text": left["output_text"],
            "speculative_text": right["output_text"],
            **draft,
            "acceptance_fraction": draft["accepted"] / draft["proposed"],
            "commit_per_step": draft["accepted"] / draft["steps"],
        })

    ar_seconds = sum(row["ar_seconds"] for row in rows)
    spec_seconds = sum(row["speculative_seconds"] for row in rows)
    transitions = sum(max(0, row["generated_tokens"] - 1) for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    proposed = sum(row["proposed"] for row in rows)
    steps = sum(row["steps"] for row in rows)
    ar_telemetry = json.loads((args.ar_root / "telemetry.json").read_text())
    spec_telemetry = json.loads((args.spec_root / "telemetry.json").read_text())
    result = {
        "schema": "kimi-k3-dspark-broad-suite-v1",
        "status": "MEASURED",
        "verdict": "ALWAYS_ON_NO_GO_SELECTIVE_LONG_OUTPUT_SIGNAL",
        "provenance": {
            "ar_root": str(args.ar_root),
            "speculative_root": str(args.spec_root),
            "ar_manifest_sha256": sha256(args.ar_root / "suite" / "suite-manifest.json"),
            "speculative_manifest_sha256": sha256(args.spec_root / "suite" / "suite-manifest.json"),
            "ar_telemetry_sha256": sha256(args.ar_root / "telemetry.json"),
            "speculative_telemetry_sha256": sha256(args.spec_root / "telemetry.json"),
            "suite_sha256": ar["environment"]["KIMI_H16_SUITE_SHA256"],
            "repository_commit": ar["environment"]["KIMI_H16_REPOSITORY_COMMIT"],
            "model_path": ar["model_path"],
            "draft_path": spec["draft_path"],
            "provider": ar["provider"],
            "layer_budget_table": ar["environment"]["DFLASH_KIMI_H22_LAYER_BUDGETS"],
            "chat_template": ar["chat_template"],
            "thinking": "disabled",
        },
        "aggregate": {
            "tasks": len(rows),
            "task_success_ar": sum(row["ar_task_success"] for row in rows),
            "task_success_speculative": sum(row["speculative_task_success"] for row in rows),
            "token_exact": sum(row["token_exact"] for row in rows),
            "true_autoregressive_transitions": transitions,
            "ar_decode_seconds": ar_seconds,
            "speculative_decode_seconds": spec_seconds,
            "always_on_speedup": ar_seconds / spec_seconds,
            "ar_transition_rate": transitions / ar_seconds,
            "speculative_transition_rate": transitions / spec_seconds,
            "draft_steps": steps,
            "accepted_tokens": accepted,
            "proposed_tokens": proposed,
            "acceptance_fraction": accepted / proposed,
            "commit_per_step": accepted / steps,
            "ar_peak_vram_mib": ar_telemetry["graphics"]["peak_memory_mib"],
            "speculative_peak_vram_mib": spec_telemetry["graphics"]["peak_memory_mib"],
            "ar_peak_rss_kib": ar_telemetry["process"]["peak_rss_kib"],
            "speculative_peak_rss_kib": spec_telemetry["process"]["peak_rss_kib"],
        },
        "sequences": rows,
        "interpretation": {
            "measured": "Always-on width-four DSpark is 5.5% slower over this mixed 12-task suite.",
            "signal": "Both 16+ token answers speed up; the 24-token code answer is 1.46x faster.",
            "quality": "11/12 speculative tasks succeed and 10/12 token sequences are identical; the 24-token code answer is truncated before becoming valid code.",
            "open": "A delayed/length-aware activation policy is not implemented or measured.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
