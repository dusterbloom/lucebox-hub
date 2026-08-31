#!/usr/bin/env python3
"""Freeze equal-B24 margin-oracle groups from an existing terminal screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--teacher-margin-min", type=float, default=1.0e-3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    screen = json.loads(args.screen_analysis.read_text())
    baseline = screen["baseline"]
    teacher_margin = float(baseline["teacher_top1_margin"])
    baseline_margin = float(baseline["candidate_teacher_margin"])
    if teacher_margin <= args.teacher_margin_min:
        raise ValueError(
            f"teacher margin {teacher_margin} does not exceed "
            f"the preregistered floor {args.teacher_margin_min}")

    arms = []
    for row in screen["interventions"]:
        if row["action"] != "force" or not row["calibrated_expert"]:
            continue
        arm = {
            "layer": int(row["layer"]),
            "route": int(row["route"]),
            "expert": int(row["expert"]),
            "ordered_rank": int(row["ordered_rank"]),
            "natural_slab": int(row["natural_slab"]),
            "delta_teacher_margin": (
                float(row["candidate_teacher_margin"]) - baseline_margin),
            "candidate_teacher_margin": float(row["candidate_teacher_margin"]),
            "independent_terminal_kl_recovered": float(
                row["equal_byte_terminal_value"]),
        }
        arm["target"] = (
            f'{arm["layer"]}:{arm["expert"]}:{arm["ordered_rank"]}')
        arms.append(arm)
    arms.sort(key=lambda row: (
        -row["delta_teacher_margin"], row["expert"], row["ordered_rank"]))
    if len(arms) != 168:
        raise ValueError(f"expected 168 calibrated force arms, found {len(arms)}")

    positive = [row for row in arms if row["delta_teacher_margin"] > 0.0]
    sizes = (2, 4, 8, min(24, len(positive)))
    names = ("top2m", "top4m", "top8m", "positive_margin_crossover24")
    groups = []
    for name, size in zip(names, sizes, strict=True):
        members = arms[:size]
        groups.append({
            "name": name,
            "member_count": size,
            "targets": [row["target"] for row in members],
            "force_environment_value": ",".join(
                row["target"] for row in members),
            "members": members,
            "projected_independent_delta_teacher_margin": sum(
                row["delta_teacher_margin"] for row in members),
            "projected_independent_terminal_kl_recovered": sum(
                row["independent_terminal_kl_recovered"] for row in members),
            "projection_warning": (
                "Independent sums are diagnostics only; interactions are "
                "measured by the group run and are not assumed additive."),
        })

    result = {
        "schema": "kimi-k3-terminal-margin-group-prereg-v1",
        "status": "PREREGISTERED_MECHANISTIC_ORACLE_ONLY",
        "source": {
            "branch": "experiment/k3-terminal-kl-bws-v2",
            "commit_before_preregistration": args.source_commit,
            "screen_analysis_path": str(args.screen_analysis),
            "screen_analysis_sha256": digest(args.screen_analysis),
            "screen_root": screen["screen"]["root"],
            "teacher_logits_sha256": screen["teacher"]["sha256"],
            "baseline_logits_sha256": baseline["logits_sha256"],
            "exact_trajectory_sha256": baseline["exact_trajectory_sha256"],
        },
        "teacher_precondition": {
            "teacher_top1": int(baseline["teacher_top1"]),
            "teacher_top1_margin": teacher_margin,
            "minimum_margin": args.teacher_margin_min,
            "passed": True,
        },
        "baseline": {
            "budget": 24,
            "candidate_top1": int(baseline["candidate_top1"]),
            "candidate_teacher_margin": baseline_margin,
            "terminal_kl": float(baseline["terminal_kl"]),
        },
        "ranking": {
            "definition": (
                "descending isolated recovery of teacher-top1 versus the "
                "candidate's strongest contender margin"),
            "eligible_force_arms": len(arms),
            "positive_margin_force_arms": len(positive),
            "tie_break": "expert ascending, ordered rank ascending",
            "labels_used": True,
        },
        "groups": groups,
        "execution_gate": {
            "intended_new_full_model_runs": 4,
            "hard_cap": 6,
            "contingency_runs": (
                "Only invalid/capacity-contended repeats; no alternative "
                "composition may be selected after observing group results."),
            "pass": (
                "At least one exact equal-B24 group recovers teacher top1 "
                "or makes the teacher-versus-contender margin positive."),
            "first_stop": (
                "If none passes, stop static B24 selection work at layer 92 "
                "and move to offline low-bit complement or progressive rescue."),
        },
        "scope": [
            "Held-out labels were used; this can test representation capacity but is not a selector.",
            "All bytes and interventions are for isolated layer 92 on one frozen terminal row.",
            "No group result can be promoted to production or called validation.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "groups": [group["name"] for group in groups],
        "teacher_margin": teacher_margin,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
