#!/usr/bin/env python3
"""Create an immutable screen index over existing single-probe artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    result.update(path.read_bytes())
    return result.hexdigest()


def environment(path: Path) -> dict[str, str]:
    result = {}
    for item in path.read_bytes().split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode()] = value.decode()
    return result


def baseline_targets(path: Path, layer: int, scope: str) -> set[tuple[str, str]]:
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source, delimiter="\t")
                if int(row["model_layer"]) == layer]
    if not rows:
        raise ValueError(f"no layer {layer} rows: {path}")
    terminal_pos = max(int(row["base_pos"]) for row in rows)
    rows = [row for row in rows if int(row["base_pos"]) == terminal_pos]
    if len(rows) != 16:
        raise ValueError(f"expected 16 terminal routes, found {len(rows)}")
    result = set()
    for row in rows:
        expert = int(row["expert"])
        selected = {int(value) for value in row["selected_ranks"].split(",") if value}
        for rank in range(12):
            action = "drop" if rank in selected else "force"
            if scope == "all" or action == scope:
                result.add((action, f"{layer}:{expert}:{rank}"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-plan", type=Path, required=True)
    parser.add_argument("--artifact-glob", action="append", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scope", choices=("all", "force", "drop"), default="all")
    parser.add_argument("--binary-sha256", action="append", required=True)
    parser.add_argument("--pair-script-sha256", action="append", required=True)
    parser.add_argument("--source-commit", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.layer < 1 or args.layer > 92:
        raise ValueError("layer must be in [1, 92]")
    for values, name in ((args.binary_sha256, "binary"),
                         (args.pair_script_sha256, "pair script")):
        for value in values:
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"invalid {name} SHA-256")

    expected = baseline_targets(args.baseline_plan, args.layer, args.scope)
    indexed: dict[tuple[str, str], Path] = {}
    for pattern in args.artifact_glob:
        for raw_path in sorted(glob.glob(pattern)):
            path = Path(raw_path)
            env_path = path / "environment.nul"
            if not env_path.is_file() or not (path / "SHA256SUMS").is_file():
                continue
            env = environment(env_path)
            for action, key in (("force", "DFLASH_KIMI_EXPERIMENT_SLAB_FORCE"),
                                ("drop", "DFLASH_KIMI_EXPERIMENT_SLAB_DROP")):
                for target in filter(None, env.get(key, "").split(",")):
                    identity = (action, target)
                    if identity not in expected:
                        continue
                    if identity in indexed:
                        raise ValueError(f"duplicate artifact for {identity}: {indexed[identity]} {path}")
                    indexed[identity] = path
    missing = sorted(expected - indexed.keys())
    if missing:
        raise ValueError(f"missing {len(missing)} interventions; first: {missing[:5]}")

    args.output_root.mkdir(parents=True)
    plan = args.output_root / "screen-plan.tsv"
    with plan.open("w", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(("action", "target", "artifact_dir"))
        for action, identity in sorted(expected, key=lambda value: tuple(
                map(int, value[1].split(":"))) + (value[0],)):
            writer.writerow((action, identity, indexed[(action, identity)]))
    (args.output_root / "screen-plan.sha256").write_text(f"{digest(plan)}  {plan}\n")
    (args.output_root / "binary.sha256").write_text("".join(
        f"{value}  indexed-legacy-executable-{index}\n"
        for index, value in enumerate(args.binary_sha256)))
    (args.output_root / "pair-script.sha256").write_text("".join(
        f"{value}  indexed-legacy-pair-script-{index}\n"
        for index, value in enumerate(args.pair_script_sha256)))
    (args.output_root / "source-commit.txt").write_text(
        "".join(value + "\n" for value in args.source_commit))
    (args.output_root / "COMPLETE").write_text(f"complete\t{len(expected)}\n")
    print(f"indexed {len(expected)} interventions at {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
