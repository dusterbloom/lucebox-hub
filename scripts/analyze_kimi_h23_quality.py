#!/usr/bin/env python3
"""Score the frozen H23 candidate against native-success official-chat tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from compare_kimi_logits import load_trace, log_softmax


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def task_success(identifier: str, text: str) -> bool:
    value = normalized(text)
    if identifier == "fact-capital":
        return "tokyo" in value
    if identifier == "code-sum":
        return re.search(r"(?<!\d)10(?!\d)", value) is not None
    if identifier == "reasoning-marble":
        return re.search(r"(?<!\d)42(?!\d)", value) is not None
    if identifier == "grammar-apples":
        return "she doesn't like apples" in value or "she does not like apples" in value
    if identifier == "translation-italian":
        return "buongiorno" in value or "buon giorno" in value
    if identifier == "extract-code":
        compact = re.sub(r"\s+", "", value)
        return "lime-742" in compact
    raise ValueError(f"unregistered H23 task {identifier}")


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def logits_path(directory: Path, row: dict[str, object]) -> Path:
    candidate = directory / Path(str(row["output_logits"])).name
    return candidate if candidate.is_file() else Path(str(row["output_logits"]))


def aligned_terminal(
    native_path: Path,
    candidate_path: Path,
    prompt_tokens: int,
    divergence: int | None,
) -> dict[str, object]:
    native_header, native, _ = load_trace(native_path)
    candidate_header, candidate, _ = load_trace(candidate_path)
    if native_header["vocabulary"] != candidate_header["vocabulary"]:
        raise ValueError("native/candidate vocabulary mismatch")
    rows = min(native.shape[0], candidate.shape[0])
    if divergence is not None:
        rows = min(rows, prompt_tokens + divergence)
    if rows <= 0:
        raise ValueError("no aligned rows")
    native_logp = log_softmax(native[:rows].astype(np.float64))
    candidate_logp = log_softmax(candidate[:rows].astype(np.float64))
    probability = np.exp(native_logp)
    kl = np.maximum(
        np.sum(probability * (native_logp - candidate_logp), axis=1), 0.0
    )
    return {
        "rows": rows,
        "mean_kl": float(kl.mean()),
        "median_kl": float(np.median(kl)),
        "p95_kl": float(np.quantile(kl, 0.95)),
        "maximum_kl": float(kl.max()),
        "top1_agreement": int((native[:rows].argmax(1) == candidate[:rows].argmax(1)).sum()),
        "top1_denominator": rows,
        "_kl_values": kl.tolist(),
    }


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads((path / "suite-manifest.json").read_text())
    if manifest.get("schema") != "kimi-k3-h16-suite-v1":
        raise ValueError(f"unsupported suite manifest {path}")
    if manifest.get("chat_template") != "gguf-jinja-thinking-off":
        raise ValueError(f"H23 requires official chat template: {path}")
    return manifest


def traffic_totals(path: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            for key, value in row.items():
                if key != "model_layer":
                    totals[key] = totals.get(key, 0) + int(value)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--traffic", type=Path)
    parser.add_argument("--traffic-process", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--budget-table", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--sidecar-manifest", type=Path)
    parser.add_argument("--warning", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native = load_manifest(args.native)
    native_rows = {str(row["id"]): row for row in native["sequences"]}
    native_successes = {
        key: task_success(key, str(row["output_text"]))
        for key, row in native_rows.items()
    }
    result: dict[str, object] = {
        "schema": "kimi-k3-h23-native-success-quality-v1",
        "status": "MEASURED_NATIVE_ONLY" if args.candidate is None else "MEASURED",
        "provenance": {
            "native_manifest_sha256": sha256(args.native / "suite-manifest.json"),
            "suite_path": native["suite_path"],
            "suite_sha256": native.get("environment", {}).get("KIMI_H16_SUITE_SHA256"),
            "chat_template": native["chat_template"],
        },
        "native": {
            "successes": sum(native_successes.values()),
            "tasks": len(native_successes),
            "all_tasks_succeeded": all(native_successes.values()),
            "sequences": [{
                "id": key,
                "prompt": row["text"],
                "prompt_tokens": row["prompt_tokens"],
                "output_tokens": row["output_tokens"],
                "output_text": row["output_text"],
                "task_success": native_successes[key],
            } for key, row in native_rows.items()],
        },
    }
    if args.candidate is not None:
        if not all(native_successes.values()):
            raise ValueError("native-success fixture is invalid; candidate must not be scored")
        if args.traffic is None or args.telemetry is None:
            parser.error("candidate analysis requires --traffic and --telemetry")
        candidate = load_manifest(args.candidate)
        candidate_rows = {str(row["id"]): row for row in candidate["sequences"]}
        if set(candidate_rows) != set(native_rows):
            raise ValueError("native/candidate task mismatch")
        sequences = []
        for identifier, native_row in native_rows.items():
            row = candidate_rows[identifier]
            if row["text"] != native_row["text"] or row["prompt_tokens"] != native_row["prompt_tokens"]:
                raise ValueError(f"native/candidate prompt mismatch {identifier}")
            native_tokens = [int(value) for value in native_row["output_tokens"]]
            candidate_tokens = [int(value) for value in row["output_tokens"]]
            divergence = first_divergence(native_tokens, candidate_tokens)
            sequences.append({
                "id": identifier,
                "native_text": native_row["output_text"],
                "candidate_text": row["output_text"],
                "native_tokens": native_tokens,
                "candidate_tokens": candidate_tokens,
                "first_generated_token_divergence": divergence,
                "candidate_task_success": task_success(identifier, str(row["output_text"])),
                "terminal": aligned_terminal(
                    logits_path(args.native, native_row),
                    logits_path(args.candidate, row),
                    int(row["prompt_token_count"]), divergence,
                ),
            })
        traffic = traffic_totals(args.traffic)
        process = {}
        if args.traffic_process and args.traffic_process.is_file():
            with args.traffic_process.open(newline="") as source:
                process = next(csv.DictReader(source, delimiter="\t"))
        telemetry = json.loads(args.telemetry.read_text())
        model_positions = sum(int(row["prompt_token_count"]) + max(0, len(row["output_tokens"]) - 1)
                              for row in candidate_rows.values())
        exact_routes = traffic["calibrated_routes"] + traffic["exact_fallback_routes"]
        result["provenance"]["candidate_manifest_sha256"] = sha256(
            args.candidate / "suite-manifest.json"
        )
        if args.budget_table:
            result["provenance"]["budget_table"] = str(args.budget_table)
            result["provenance"]["budget_table_sha256"] = sha256(args.budget_table)
        for name, path in (
            ("calibration_manifest", args.calibration_manifest),
            ("sidecar_manifest", args.sidecar_manifest),
        ):
            if path:
                if not path.is_file():
                    raise ValueError(f"missing {name}: {path}")
                result["provenance"][name] = str(path)
                result["provenance"][f"{name}_sha256"] = sha256(path)
        terminal_rows = sum(row["terminal"]["rows"] for row in sequences)
        terminal_top1 = sum(row["terminal"]["top1_agreement"] for row in sequences)
        terminal_kl = np.concatenate([
            np.asarray(row["terminal"].pop("_kl_values"), dtype=np.float64)
            for row in sequences
        ])
        result["candidate"] = {
            "native_successes_retained": sum(row["candidate_task_success"] for row in sequences),
            "native_success_denominator": len(sequences),
            "retention_fraction": sum(row["candidate_task_success"] for row in sequences) / len(sequences),
            "token_exact": sum(row["native_tokens"] == row["candidate_tokens"] for row in sequences),
            "aggregate_terminal": {
                "rows": terminal_rows,
                "mean_kl": float(terminal_kl.mean()),
                "median_kl": float(np.median(terminal_kl)),
                "p95_kl": float(np.quantile(terminal_kl, 0.95)),
                "maximum_kl": float(terminal_kl.max()),
                "top1_agreement": terminal_top1,
                "top1_denominator": terminal_rows,
            },
            "sequences": sequences,
        }
        result["bytes"] = {
            "model_positions": model_positions,
            "logical_provider_bytes": traffic["total_provider_bytes"],
            "logical_provider_gib_per_position": traffic["total_provider_bytes"] / model_positions / (1 << 30),
            "selected_sidecar_bytes": traffic["selected_sidecar_bytes"],
            "exact_fallback_bytes": traffic["exact_fallback_bytes"],
            "exact_fallback_route_fraction": traffic["exact_fallback_routes"] / exact_routes,
            "explicit_provider_read_bytes": int(process["explicit_provider_read_bytes"])
                if process.get("explicit_provider_read_bytes") else None,
            "process_read_bytes": int(process["process_read_bytes"])
                if process.get("process_read_bytes") else None,
            "device_read_bytes": telemetry["disk"]["read_bytes"],
            "process_read_to_logical_ratio": (
                int(process["process_read_bytes"]) / traffic["total_provider_bytes"]
                if process.get("process_read_bytes") else None
            ),
            "exact_fallback_byte_fraction": (
                traffic["exact_fallback_bytes"] / traffic["total_provider_bytes"]
            ),
        }
        result["runtime"] = {
            "elapsed_seconds": telemetry["elapsed_seconds"],
            "peak_ram_kib": telemetry["process"]["peak_rss_kib"],
            "peak_vram_mib": telemetry["graphics"]["peak_memory_mib"],
            "gpu_energy_joules": telemetry["graphics"]["integrated_energy_joules"],
        }
        result["warnings"] = [
            "Primary quality is retained native-success tasks, not exact token match.",
            "KL is scored only while generated history remains aligned.",
            "This six-task suite is a small decision gate, not broad quality certification.",
        ] + args.warning
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
