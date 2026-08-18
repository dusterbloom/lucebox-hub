#!/usr/bin/env python3
"""Register the matched P30 bounded host-cache runtime gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


P20 = re.compile(
    r"\[kimi-k3-p20\].*?explicit-provider-reads=(?P<provider>\d+)"
    r".*?direct-physical-bytes=(?P<direct>\d+)"
    r".*?direct-io-ns=(?P<io_ns>\d+)"
)
P30 = re.compile(
    r"\[kimi-k3-p30\].*?capacity-bytes=(?P<capacity>\d+)"
    r".*?resident-bytes=(?P<resident>\d+)"
    r".*?entries=(?P<entries>\d+)"
    r".*?hits=(?P<hits>\d+)"
    r".*?misses=(?P<misses>\d+)"
    r".*?hit-bytes=(?P<hit_bytes>\d+)"
    r".*?inserted-bytes=(?P<inserted>\d+)"
    r".*?evicted-bytes=(?P<evicted>\d+)"
    r".*?sequence-resets=(?P<resets>\d+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(root: Path, requested_mib: int) -> dict[str, object]:
    manifest_path = root / "suite" / "suite-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["sequences"]) != 1:
        raise ValueError(f"expected one sequence: {manifest_path}")
    row = manifest["sequences"][0]
    stderr = (root / "stderr.log").read_text()
    p20_match = P20.search(stderr)
    if not p20_match:
        raise ValueError(f"missing P20 footer: {root}")
    p20 = {key: int(value) for key, value in p20_match.groupdict().items()}
    p30_match = P30.search(stderr)
    cache = ({key: int(value) for key, value in p30_match.groupdict().items()}
             if p30_match else None)
    if requested_mib == 0 and cache is not None:
        raise ValueError("zero-cache control unexpectedly emitted P30 stats")
    if requested_mib and (cache is None or cache["capacity"] != requested_mib << 20):
        raise ValueError(f"cache capacity mismatch: {root}")
    telemetry = json.loads((root / "telemetry.json").read_text())
    logits_path = Path(row["output_logits"])
    if not logits_path.is_file():
        logits_path = root / "suite" / logits_path.name
    transitions = max(0, len(row["output_tokens"]) - 1)
    return {
        "requested_cache_mib": requested_mib,
        "root": str(root),
        "manifest_sha256": sha256(manifest_path),
        "telemetry_sha256": sha256(root / "telemetry.json"),
        "logits_sha256": sha256(logits_path),
        "output_tokens": row["output_tokens"],
        "prefill_seconds": row["prefill_seconds"],
        "decode_seconds": row["decode_seconds"],
        "decode_transitions": transitions,
        "decode_transition_rate": transitions / row["decode_seconds"],
        "provider_physical_bytes": p20["provider"],
        "direct_physical_bytes": p20["direct"],
        "direct_io_seconds": p20["io_ns"] / 1e9,
        "disk_read_bytes": telemetry["disk"]["read_bytes"],
        "peak_rss_kib": telemetry["process"]["peak_rss_kib"],
        "peak_anon_rss_kib": telemetry["process"]["peak_rss_anon_kib"],
        "peak_swap_kib": telemetry["process"]["peak_swap_kib"],
        "cache": cache,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="CACHE_MIB:ROOT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for value in args.run:
        cache, root = value.split(":", 1)
        rows.append(load(Path(root), int(cache)))
    rows.sort(key=lambda row: row["requested_cache_mib"])
    if not rows or rows[0]["requested_cache_mib"] != 0:
        raise ValueError("a zero-cache control is required")
    base = rows[0]
    for row in rows:
        row["logits_bit_equal_to_control"] = row["logits_sha256"] == base["logits_sha256"]
        row["tokens_equal_to_control"] = row["output_tokens"] == base["output_tokens"]
        row["decode_speedup"] = base["decode_seconds"] / row["decode_seconds"]
        row["prefill_speedup"] = base["prefill_seconds"] / row["prefill_seconds"]
        row["provider_byte_reduction_fraction"] = (
            1 - row["provider_physical_bytes"] / base["provider_physical_bytes"]
        )
    result = {
        "schema": "kimi-k3-p30-host-read-cache-v1",
        "status": "MEASURED",
        "verdict": "NO_GO_AS_RUNTIME_DEFAULT",
        "runs": rows,
        "interpretation": {
            "measured": "The 8 GiB cache removed 56.88% of explicit provider reads but improved decode only 1.42% and slowed prefill 7.40%.",
            "semantic_gate": "All cache capacities produced byte-identical full logits and identical generated tokens.",
            "simulation_disposition": "The trace hit-rate prediction is validated; treating cache hits as free and storage as the integrated bottleneck is falsified.",
            "decision": "Do not make this host-copy LRU the runtime default or run the broad suite.",
            "open": "A GPU-resident, direct-destination cache could avoid the host lookup/copy and H2D costs, but must first earn a stage-level ceiling.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
