#!/usr/bin/env python3
"""Verify P28 logits and compute its integrated decision gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


STAGE = re.compile(r"\[kimi-k3-stage\] position=(\d+).*?total_ms=([0-9.]+)")
P28 = re.compile(
    r"\[kimi-k3-p28\] launches=(\d+) hits=(\d+) misses=(\d+) "
    r"oracle-read-ns=(\d+) demand-wait-ns=(\d+) physical-bytes=(\d+) "
    r"wasted-bytes=(\d+) extra-pinned-bytes=(\d+)"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def decode_seconds(stderr: str) -> tuple[int, float]:
    rows = [(int(position), float(ms)) for position, ms in STAGE.findall(stderr)]
    decode = [ms for position, ms in rows if position > 0]
    if not decode:
        raise ValueError("no decoded P28 stage rows")
    return len(decode), sum(decode) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_logits = args.reference_dir / "logits.f32"
    candidate_logits = args.candidate_dir / "logits.f32"
    reference_sha = digest(reference_logits)
    candidate_sha = digest(candidate_logits)
    candidate_stderr = (args.candidate_dir / "stderr.log").read_text()
    reference_stderr = (args.reference_dir / "stderr.log").read_text()
    transitions, candidate_s = decode_seconds(candidate_stderr)
    reference_transitions, reference_s = decode_seconds(reference_stderr)
    if transitions != reference_transitions:
        raise ValueError("reference/candidate transition count mismatch")
    match = P28.search(candidate_stderr)
    if not match:
        raise ValueError("P28 accounting footer missing")
    launches, hits, misses, read_ns, wait_ns, physical, wasted, pinned = (
        int(value) for value in match.groups()
    )
    gain = reference_s / candidate_s - 1.0
    bit_equal = reference_sha == candidate_sha
    verdict = "GO" if bit_equal and gain >= 0.25 else "NO_GO"
    result = {
        "schema": "k3-p28-integrated-oracle-v1",
        "verdict": verdict,
        "semantic_gate": {
            "bit_equal": bit_equal,
            "reference_logits_sha256": reference_sha,
            "candidate_logits_sha256": candidate_sha,
        },
        "performance": {
            "transitions": transitions,
            "reference_decode_seconds": reference_s,
            "candidate_decode_seconds": candidate_s,
            "reference_transitions_per_second": transitions / reference_s,
            "candidate_transitions_per_second": transitions / candidate_s,
            "throughput_gain_fraction": gain,
            "gate_fraction": 0.25,
        },
        "oracle": {
            "launches": launches,
            "hits": hits,
            "misses": misses,
            "read_seconds": read_ns / 1e9,
            "demand_wait_seconds": wait_ns / 1e9,
            "physical_bytes": physical,
            "wasted_bytes": wasted,
            "extra_pinned_bytes": pinned,
        },
        "predictor_research_earned": verdict == "GO",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
