#!/usr/bin/env python3
"""Score an H22 adaptive free-generation run against the archived exact run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

from compare_kimi_logits import load_trace, log_softmax


SIDECAR_V1 = struct.Struct("<8s8I5Q")
SIDECAR_V2 = struct.Struct("<8s8I8Q")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    value = json.loads((path / "suite-manifest.json").read_text())
    if value.get("schema") != "kimi-k3-h16-suite-v1":
        raise ValueError(f"unsupported manifest {path}")
    return value


def trace_path(directory: Path, row: dict[str, object]) -> Path:
    registered = Path(str(row["output_logits"]))
    local = directory / registered.name
    return local if local.is_file() else registered


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, pair in enumerate(zip(left, right)):
        if pair[0] != pair[1]:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def task_success(identifier: str, text: str) -> bool:
    lowered = " ".join(text.lower().split())
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", lowered)
    if identifier == "science-photosynthesis":
        return any(word in lowered for word in ("plant", "plants")) and any(
            word in lowered for word in ("light", "sunlight", "solar")
        )
    if identifier == "math-multiply":
        return re.search(r"(?<!\d)703(?!\d)", lowered) is not None
    if identifier == "code-sum":
        return re.search(r"(?<!\d)10(?!\d)", lowered) is not None
    if identifier == "fact-capital":
        return "tokyo" in lowered
    if identifier == "logic-raven":
        return "bird" in lowered
    if identifier == "translation-italian":
        return "buongiorno" in lowered or "buon giorno" in lowered
    if identifier == "word-synonym":
        return any(word in words for word in (
            "adaptable", "durable", "elastic", "flexible", "hardy", "persistent",
            "robust", "strong", "tenacious", "tough",
        ))
    if identifier == "writing-moonlight":
        return len(words) == 5
    if identifier == "science-ice":
        return "water" in lowered and (
            "less dense" in lowered or "density" in lowered
        )
    if identifier == "math-power":
        return re.search(r"(?<!\d)1024(?!\d)", lowered) is not None
    if identifier == "computer-queue":
        return "queue" in lowered or "fifo" in lowered
    if identifier == "grammar-apples":
        return (
            "she doesn't like apples" in lowered
            or "she does not like apples" in lowered
        )
    raise ValueError(f"unregistered H22 task {identifier}")


def aligned_terminal(
    native_path: Path,
    candidate_path: Path,
    prompt_tokens: int,
    generated_divergence: int | None,
) -> dict[str, object]:
    native_header, native, _ = load_trace(native_path)
    candidate_header, candidate, _ = load_trace(candidate_path)
    if native_header["vocabulary"] != candidate_header["vocabulary"]:
        raise ValueError("vocabulary mismatch")
    aligned = min(native.shape[0], candidate.shape[0])
    if generated_divergence is not None:
        aligned = min(aligned, prompt_tokens + generated_divergence)
    if aligned <= 0:
        raise ValueError("no aligned terminal rows")
    native_logp = log_softmax(native[:aligned].astype(np.float64))
    candidate_logp = log_softmax(candidate[:aligned].astype(np.float64))
    probability = np.exp(native_logp)
    kl = np.maximum(
        np.sum(probability * (native_logp - candidate_logp), axis=1), 0.0
    )
    native_top = native[:aligned].argmax(axis=1)
    candidate_top = candidate[:aligned].argmax(axis=1)
    return {
        "aligned_rows": aligned,
        "mean_kl": float(kl.mean()),
        "median_kl": float(np.median(kl)),
        "maximum_kl": float(kl.max()),
        "decision_row_kl": float(kl[prompt_tokens - 1]),
        "top1_agreement": int((native_top == candidate_top).sum()),
        "top1_denominator": aligned,
    }


def read_budget_table(path: Path) -> list[int]:
    result = [0] * 92
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        layer, budget = map(int, line.split())
        if layer < 1 or layer > 92 or result[layer - 1] != 0:
            raise ValueError("invalid layer budget table")
        result[layer - 1] = budget
    if any(value == 0 for value in result):
        raise ValueError("incomplete layer budget table")
    return result


def sidecar_record_bytes(path: Path) -> int:
    with path.open("rb") as source:
        prefix = source.read(SIDECAR_V1.size)
        values = SIDECAR_V1.unpack(prefix)
        if values[0] != b"K3SLB001" or values[1] not in (1, 2):
            raise ValueError(f"bad sidecar {path}")
        if values[1] == 2:
            source.seek(0)
            values = SIDECAR_V2.unpack(source.read(SIDECAR_V2.size))
    return int(values[13])


def traffic(path: Path) -> dict[str, int]:
    header, *lines = path.read_text().splitlines()
    fields = header.split("\t")
    totals = {name: 0 for name in fields if name != "model_layer"}
    for line in lines:
        values = line.split("\t")
        row = dict(zip(fields, values))
        for key in totals:
            totals[key] += int(row[key])
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--budget-table", type=Path, required=True)
    parser.add_argument("--traffic", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    native_manifest = load_manifest(args.native)
    candidate_manifest = load_manifest(args.candidate)
    native_by_id = {str(row["id"]): row for row in native_manifest["sequences"]}
    results = []
    total_positions = 0
    for candidate in candidate_manifest["sequences"]:
        identifier = str(candidate["id"])
        native = native_by_id.get(identifier)
        if native is None or native["prompt_tokens"] != candidate["prompt_tokens"]:
            raise ValueError(f"native prompt mismatch {identifier}")
        native_tokens = [int(value) for value in native["output_tokens"]]
        candidate_tokens = [int(value) for value in candidate["output_tokens"]]
        divergence = first_divergence(native_tokens, candidate_tokens)
        prompt_count = int(candidate["prompt_token_count"])
        total_positions += prompt_count + max(0, len(candidate_tokens) - 1)
        native_text = str(native["output_text"])
        candidate_text = str(candidate["output_text"])
        results.append({
            "id": identifier,
            "prompt": candidate["text"],
            "native_text": native_text,
            "candidate_text": candidate_text,
            "native_tokens": native_tokens,
            "candidate_tokens": candidate_tokens,
            "first_generated_token_divergence": divergence,
            "token_exact": native_tokens == candidate_tokens,
            "native_task_success": task_success(identifier, native_text),
            "candidate_task_success": task_success(identifier, candidate_text),
            "terminal": aligned_terminal(
                trace_path(args.native, native),
                trace_path(args.candidate, candidate),
                prompt_count,
                divergence,
            ),
        })

    budgets = read_budget_table(args.budget_table)
    provider = traffic(args.traffic)
    exact_layer_bytes = 0
    exact_baseline_bytes = 0
    for layer, budget in enumerate(budgets, 1):
        record = sidecar_record_bytes(
            args.sidecars / f"kimi_layer{layer:02d}_natural_slabs.k3slab"
        )
        exact_baseline_bytes += total_positions * 16 * record
        if budget == 192:
            exact_layer_bytes += total_positions * 16 * record
    logical_bytes = provider["total_provider_bytes"] + exact_layer_bytes
    telemetry = json.loads(args.telemetry.read_text())
    result = {
        "schema": "kimi-k3-h22-adaptive-quality-v1",
        "status": "MEASURED",
        "provenance": {
            "native_manifest_sha256": sha256(args.native / "suite-manifest.json"),
            "candidate_manifest_sha256": sha256(args.candidate / "suite-manifest.json"),
            "budget_table": str(args.budget_table),
            "budget_table_sha256": sha256(args.budget_table),
        },
        "allocation": {
            "budget_counts": {
                str(budget): budgets.count(budget) for budget in sorted(set(budgets))
            },
            "evaluated_model_positions": total_positions,
            "partial_provider_bytes": provider["total_provider_bytes"],
            "native_exact_layer_bytes": exact_layer_bytes,
            "total_logical_routed_bytes": logical_bytes,
            "exact_baseline_routed_bytes": exact_baseline_bytes,
            "logical_fraction": logical_bytes / exact_baseline_bytes,
        },
        "runtime": {
            "elapsed_seconds": telemetry["elapsed_seconds"],
            "disk_read_bytes": telemetry["disk"]["read_bytes"],
            "peak_ram_kib": telemetry["process"]["peak_rss_kib"],
            "peak_vram_mib": telemetry["graphics"]["peak_memory_mib"],
            "gpu_energy_joules": telemetry["graphics"]["integrated_energy_joules"],
        },
        "task_success": {
            "count": sum(int(row["candidate_task_success"]) for row in results),
            "denominator": len(results),
        },
        "token_exact_count": sum(int(row["token_exact"]) for row in results),
        "sequences": results,
        "warnings": [
            "KL is reported only through the last row conditioned on a shared generated history.",
            "Task success uses prompt-specific deterministic heuristics preregistered before this candidate run.",
            "This small frozen suite is decision evidence, not broad quality certification.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "task_success": result["task_success"],
        "token_exact_count": result["token_exact_count"],
        "logical_fraction": result["allocation"]["logical_fraction"],
        "sequences": [{
            "id": row["id"],
            "candidate_text": row["candidate_text"],
            "first_divergence": row["first_generated_token_divergence"],
            "terminal": row["terminal"],
        } for row in results],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
