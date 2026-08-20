#!/usr/bin/env python3
"""Build an exact compact-expert geometry schedule from a K3 P20 trace.

The P56 trace records every logical gate/up/down slab read, including cache
hits whose physical-read charge is zero.  This tool reconstructs the natural
slab ID from the immutable sidecar record address, validates the complete
component-major compact layout, and emits two small benchmark inputs:

* ``k3_compact_specs.tsv`` contains the complete numerical/byte contract for
  each expert geometry in the trace.
* ``k3_compact_jobs.tsv`` contains the canonical layer-position job order,
  prefix depth, natural mask, compact residency order, and 12-entry sparse-K
  map for every prefill compact job.

No model bytes are read.  Natural IDs are accepted only when the trace itself
uniquely determines the sidecar payload origin for every layer file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SLAB_COUNT = 12
SLAB_SIZE = 256
INPUT_DIM = 3584
INTERMEDIATE_DIM = SLAB_COUNT * SLAB_SIZE
OUTPUT_DIM = 3584
MAP_BYTES = SLAB_COUNT * 4
REGIONS = ("gate", "up", "down")
TRACE_COLUMNS = {
    "base_pos", "token_index", "model_layer", "expert_id", "region",
    "qtype", "prefix_depth", "exact_fallback", "file_path", "file_offset",
    "logical_length", "destination_offset",
}
SPEC_FIELDS = (
    "input_dim", "intermediate_dim", "output_dim", "gate_type", "up_type",
    "down_type", "fused_gate_up", "activation", "situ_beta",
    "situ_linear_beta", "gate_scale", "up_scale", "down_scale",
    "gate_slab_bytes", "up_slab_bytes", "down_slab_bytes",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_lengths(path: Path) -> tuple[list[int], dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("draft_path"):
        raise ValueError("compact schedule requires speculative decoding off")
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or not rows:
        raise ValueError("suite manifest has no sequences")
    lengths: list[int] = []
    for row in rows:
        count = row.get("prompt_token_count") if isinstance(row, dict) else None
        if not isinstance(count, int) or count <= 0:
            raise ValueError("suite manifest has an invalid prompt length")
        lengths.append(count)
    return lengths, manifest


def rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not TRACE_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(TRACE_COLUMNS.difference(reader.fieldnames or []))
            raise ValueError(f"I/O trace is missing required columns: {missing}")
        for number, row in enumerate(reader, 2):
            yield number, row


@dataclass
class SidecarGeometry:
    qtypes: dict[str, str] = field(default_factory=dict)
    lengths: dict[str, int] = field(default_factory=dict)
    pending_gate_rows: list[tuple[int, int, int]] = field(default_factory=list)
    payload_candidates: set[int] | None = None
    payload_offset: int | None = None

    @property
    def complete(self) -> bool:
        return all(region in self.lengths for region in REGIONS)

    @property
    def record_bytes(self) -> int:
        if not self.complete:
            raise ValueError("sidecar component geometry is incomplete")
        return sum(self.lengths[region] for region in REGIONS)

    def observe_component(
        self, region: str, qtype: str, length: int, row_number: int
    ) -> None:
        if region in self.qtypes and self.qtypes[region] != qtype:
            raise ValueError(
                f"sidecar qtype changes at trace row {row_number}: "
                f"{region} {self.qtypes[region]} -> {qtype}")
        if region in self.lengths and self.lengths[region] != length:
            raise ValueError(
                f"sidecar component length changes at trace row {row_number}: "
                f"{region} {self.lengths[region]} -> {length}")
        self.qtypes[region] = qtype
        self.lengths[region] = length

    def observe_gate(self, expert: int, offset: int, row_number: int) -> None:
        if not self.complete:
            self.pending_gate_rows.append((expert, offset, row_number))
            return
        self._intersect_payload(expert, offset, row_number)

    def drain_pending(self) -> None:
        if not self.complete:
            return
        pending, self.pending_gate_rows = self.pending_gate_rows, []
        for expert, offset, row_number in pending:
            self._intersect_payload(expert, offset, row_number)

    def _intersect_payload(self, expert: int, offset: int, row_number: int) -> None:
        record = self.record_bytes
        candidates = {
            offset - (expert * SLAB_COUNT + natural) * record
            for natural in range(SLAB_COUNT)
        }
        if self.payload_candidates is None:
            self.payload_candidates = candidates
        else:
            self.payload_candidates.intersection_update(candidates)
        if not self.payload_candidates:
            raise ValueError(
                f"no consistent sidecar payload origin at trace row {row_number}")

    def resolve(self, path: str) -> None:
        self.drain_pending()
        if not self.complete:
            raise ValueError(f"sidecar {path} lacks a gate/up/down geometry")
        if not self.payload_candidates or len(self.payload_candidates) != 1:
            candidates = 0 if self.payload_candidates is None else len(self.payload_candidates)
            raise ValueError(
                f"sidecar {path} natural IDs are ambiguous ({candidates} payload origins)")
        self.payload_offset = next(iter(self.payload_candidates))
        if self.payload_offset < 0:
            raise ValueError(f"sidecar {path} has a negative payload origin")

    def natural(self, expert: int, component_offset: int, region: str) -> int:
        assert self.payload_offset is not None
        prefix = 0
        if region == "up":
            prefix = self.lengths["gate"]
        elif region == "down":
            prefix = self.lengths["gate"] + self.lengths["up"]
        record_start = component_offset - prefix
        delta = record_start - self.payload_offset
        if delta < 0 or delta % self.record_bytes:
            raise ValueError("component offset is outside the inferred sidecar records")
        natural = delta // self.record_bytes - expert * SLAB_COUNT
        if natural < 0 or natural >= SLAB_COUNT:
            raise ValueError("component offset resolves to an invalid natural slab")
        return natural

    def spec_key(self) -> tuple:
        return (
            INPUT_DIM, INTERMEDIATE_DIM, OUTPUT_DIM,
            self.qtypes["gate"], self.qtypes["up"], self.qtypes["down"],
            False, "situ", 4.0, 25.0, 1.0, 1.0, 1.0,
            self.lengths["gate"], self.lengths["up"], self.lengths["down"],
        )


def discover_geometries(trace: Path) -> dict[str, SidecarGeometry]:
    geometries: dict[str, SidecarGeometry] = {}
    for number, row in rows(trace):
        region = row["region"]
        if region not in REGIONS:
            continue
        try:
            length = int(row["logical_length"])
            expert = int(row["expert_id"])
            offset = int(row["file_offset"])
        except ValueError as error:
            raise ValueError(f"malformed trace row {number}: {error}") from error
        if length <= 0 or expert < 0 or offset < 0:
            raise ValueError(f"invalid component address at trace row {number}")
        path = row["file_path"]
        geometry = geometries.setdefault(path, SidecarGeometry())
        geometry.observe_component(region, row["qtype"], length, number)
        geometry.drain_pending()
        if region == "gate":
            geometry.observe_gate(expert, offset, number)
    if not geometries:
        raise ValueError("trace contains no compact gate/up/down rows")
    for path, geometry in geometries.items():
        geometry.resolve(path)
    return geometries


@dataclass
class Job:
    sequence: int
    position: int
    base: int
    token: int
    layer: int
    expert: int
    depth: int
    path: str
    slots: dict[int, dict[str, int]] = field(default_factory=dict)
    exact_fallback: bool = False

    def add(
        self, region: str, natural: int, slot: int, exact: bool,
        row_number: int,
    ) -> None:
        if slot < 0 or slot >= self.depth:
            raise ValueError(f"compact slot is out of range at trace row {row_number}")
        values = self.slots.setdefault(slot, {})
        if region in values:
            raise ValueError(f"duplicate {region} component at trace row {row_number}")
        values[region] = natural
        self.exact_fallback = self.exact_fallback or exact

    def finish(self) -> tuple[list[int], list[int], int]:
        if sorted(self.slots) != list(range(self.depth)):
            raise ValueError("compact job does not contain every prefix slot")
        naturals: list[int] = []
        for slot in range(self.depth):
            components = self.slots[slot]
            if set(components) != set(REGIONS):
                raise ValueError("compact slot does not contain gate/up/down")
            if len(set(components.values())) != 1:
                raise ValueError("compact slot components disagree on natural ID")
            naturals.append(components["gate"])
        if len(set(naturals)) != len(naturals):
            raise ValueError("compact job contains duplicate natural slabs")
        mapping = [-1] * SLAB_COUNT
        mask = 0
        for slot, natural in enumerate(naturals):
            mapping[natural] = slot
            mask |= 1 << natural
        return naturals, mapping, mask


def component_slot(
    geometry: SidecarGeometry, depth: int, region: str,
    destination: int, row_number: int,
) -> int:
    base = 32
    if region == "up":
        base += depth * geometry.lengths["gate"]
    elif region == "down":
        base += depth * (
            geometry.lengths["gate"] + geometry.lengths["up"])
    relative = destination - base
    width = geometry.lengths[region]
    if relative < 0 or relative % width:
        raise ValueError(f"invalid component-major destination at trace row {row_number}")
    return relative // width


def write_specs(
    output: Path, specs: list[tuple], spec_ids: dict[tuple, str]
) -> None:
    fields = ["spec_id", *SPEC_FIELDS]
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for spec in specs:
            writer.writerow([spec_ids[spec], *spec])


def build_jobs(
    trace: Path, lengths: list[int], geometries: dict[str, SidecarGeometry],
    spec_ids: dict[tuple, str], output: Path,
) -> dict:
    fields = [
        "group_id", "sequence", "position", "base_pos", "token_index",
        "model_layer", "expert_id", "job_rank", "spec_id", "prefix_depth",
        "route_kind", "natural_mask", "natural_ids", "natural_to_compact",
        "weight_bytes", "map_bytes",
    ]
    job_count = 0
    group_count = 0
    weight_bytes = 0
    unique_positions: set[tuple[int, int]] = set()
    depth_counts: Counter[int] = Counter()
    spec_counts: Counter[str] = Counter()
    mask_counts: Counter[int] = Counter()
    exact_job_count = 0
    current_job: Job | None = None
    group_key: tuple[int, int, int] | None = None
    group_jobs: list[tuple[Job, list[int], list[int], int]] = []
    sequence = 0
    last_base: int | None = None

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)

        def flush_job() -> None:
            nonlocal current_job, group_key, group_jobs
            if current_job is None:
                return
            finished = (*current_job.finish(),)
            key = (current_job.sequence, current_job.position, current_job.layer)
            if group_key is not None and key != group_key:
                flush_group()
            group_key = key
            group_jobs.append((current_job, *finished))
            current_job = None

        def flush_group() -> None:
            nonlocal group_count, job_count, weight_bytes, exact_job_count
            nonlocal group_jobs, group_key
            if not group_jobs:
                return
            group_jobs.sort(key=lambda item: item[0].expert)
            for rank, (job, naturals, mapping, mask) in enumerate(group_jobs):
                geometry = geometries[job.path]
                spec_id = spec_ids[geometry.spec_key()]
                job_weight_bytes = job.depth * sum(geometry.lengths.values())
                writer.writerow([
                    group_count, job.sequence, job.position, job.base, job.token,
                    job.layer, job.expert, rank, spec_id, job.depth,
                    "sidecar-exact" if job.exact_fallback else "calibrated-compact",
                    mask,
                    ",".join(map(str, naturals)),
                    ",".join(map(str, mapping)), job_weight_bytes, MAP_BYTES,
                ])
                job_count += 1
                weight_bytes += job_weight_bytes
                depth_counts[job.depth] += 1
                spec_counts[spec_id] += 1
                mask_counts[mask] += 1
                if job.exact_fallback:
                    exact_job_count += 1
                unique_positions.add((job.sequence, job.position))
            group_count += 1
            group_jobs = []
            group_key = None

        for number, row in rows(trace):
            try:
                base = int(row["base_pos"])
                token = int(row["token_index"])
            except ValueError as error:
                raise ValueError(f"malformed trace row {number}: {error}") from error
            if last_base is not None and base < last_base:
                flush_job()
                sequence += 1
            last_base = base
            region = row["region"]
            if region not in REGIONS:
                continue
            position = base + token
            if sequence >= len(lengths) or position >= lengths[sequence]:
                continue
            try:
                layer = int(row["model_layer"])
                expert = int(row["expert_id"])
                depth = int(row["prefix_depth"])
                destination = int(row["destination_offset"])
                offset = int(row["file_offset"])
                exact = int(row["exact_fallback"]) != 0
            except ValueError as error:
                raise ValueError(f"malformed trace row {number}: {error}") from error
            if depth <= 0 or depth > SLAB_COUNT:
                raise ValueError(f"invalid compact depth at trace row {number}")
            path = row["file_path"]
            geometry = geometries.get(path)
            if geometry is None:
                raise ValueError(f"unknown sidecar at trace row {number}")
            key = (sequence, position, token, layer, expert)
            current_key = None if current_job is None else (
                current_job.sequence, current_job.position, current_job.token,
                current_job.layer, current_job.expert)
            if current_key != key:
                flush_job()
                current_job = Job(
                    sequence, position, base, token, layer, expert, depth, path)
            assert current_job is not None
            if current_job.depth != depth or current_job.path != path:
                raise ValueError(f"compact job geometry changes at trace row {number}")
            natural = geometry.natural(expert, offset, region)
            # Exact sidecar routes are still P41 compact executions, but their
            # trace records the pre-pack SparseSlabPayload destinations rather
            # than the final component-major image. The production reader
            # constructs these slabs in ascending natural order.
            slot = natural if exact else component_slot(
                geometry, depth, region, destination, number)
            current_job.add(region, natural, slot, exact, number)
        flush_job()
        flush_group()

    return {
        "positions": len(unique_positions),
        "layer_groups": group_count,
        "jobs": job_count,
        "weight_bytes": weight_bytes,
        "map_bytes": job_count * MAP_BYTES,
        "sidecar_exact_jobs": exact_job_count,
        "depth_histogram": {str(key): depth_counts[key] for key in sorted(depth_counts)},
        "spec_job_counts": {key: spec_counts[key] for key in sorted(spec_counts)},
        "mask_variants": len(mask_counts),
        "most_common_masks": {
            str(mask): count for mask, count in sorted(
                mask_counts.items(), key=lambda item: (-item[1], item[0]))[:16]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--summary-only", action="store_true",
        help="validate/count the schedule but send TSV output to /dev/null")
    parser.add_argument("--expect-jobs", type=int)
    parser.add_argument("--expect-positions", type=int)
    args = parser.parse_args()
    if not args.summary_only and args.output_dir is None:
        parser.error("--output-dir is required unless --summary-only is used")
    lengths, manifest = prompt_lengths(args.manifest)
    geometries = discover_geometries(args.trace)
    specs = sorted({geometry.spec_key() for geometry in geometries.values()})
    spec_ids = {spec: f"spec-{index}" for index, spec in enumerate(specs)}
    if args.summary_only:
        specs_path = jobs_path = Path("/dev/null")
    else:
        assert args.output_dir is not None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        specs_path = args.output_dir / "k3_compact_specs.tsv"
        jobs_path = args.output_dir / "k3_compact_jobs.tsv"
    write_specs(specs_path, specs, spec_ids)
    totals = build_jobs(
        args.trace, lengths, geometries, spec_ids, jobs_path)
    if args.expect_jobs is not None and totals["jobs"] != args.expect_jobs:
        raise ValueError(
            f"compact job count {totals['jobs']} != expected {args.expect_jobs}")
    if args.expect_positions is not None and totals["positions"] != args.expect_positions:
        raise ValueError(
            f"prefill position count {totals['positions']} != expected "
            f"{args.expect_positions}")
    report = {
        "schema": "k3-p56-compact-job-schedule-v1",
        "status": "EXACT_TRACE_DERIVED_BENCHMARK_SCHEDULE",
        "scope": "NON_SPECULATIVE_PREFILL_COMPACT_JOBS",
        "provenance": {
            "trace": str(args.trace),
            "trace_sha256": sha256(args.trace),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "repository_commit": manifest.get("environment", {}).get(
                "KIMI_H16_REPOSITORY_COMMIT"),
        },
        "fixed_k3_contract": {
            "input_dim": INPUT_DIM,
            "intermediate_dim": INTERMEDIATE_DIM,
            "output_dim": OUTPUT_DIM,
            "slab_count": SLAB_COUNT,
            "slab_size": SLAB_SIZE,
            "activation": "situ",
            "situ_beta": 4.0,
            "situ_linear_beta": 25.0,
            "scales": {"gate": 1.0, "up": 1.0, "down": 1.0},
        },
        "sidecars": len(geometries),
        "specs": len(specs),
        "spec_table": [
            {
                "spec_id": spec_ids[spec],
                **dict(zip(SPEC_FIELDS, spec, strict=True)),
                "jobs": totals["spec_job_counts"][spec_ids[spec]],
            }
            for spec in specs
        ],
        "totals": totals,
        "outputs": {
            "specs": None if args.summary_only else str(specs_path),
            "specs_sha256": None if args.summary_only else sha256(specs_path),
            "jobs": None if args.summary_only else str(jobs_path),
            "jobs_sha256": None if args.summary_only else sha256(jobs_path),
        },
        "release_benchmark_contract": {
            "build_type": "Release",
            "schedule_order": "sequence-position-layer then ascending expert_id",
            "persistent_graph_key": "spec_id,prefix_depth",
            "payload_residency": "device-resident synthetic bytes; zero storage and payload H2D",
            "map_residency": "the emitted 12-entry natural_to_compact map",
            "submission": "one graph enqueue per job and one synchronization per layer_group",
            "publication_modes": ["compute-only", "compute-plus-output-D2D"],
            "required_counters": [
                "jobs", "layer_groups", "graph_enqueues", "synchronizations",
                "payload_h2d_bytes", "storage_bytes", "elapsed_ns",
            ],
            "validity": [
                "jobs and groups equal the schedule", "payload_h2d_bytes=0",
                "storage_bytes=0", "both warm execution orders",
                "gfx1151 and gfx1201 reported separately",
            ],
            "roofline_gate_positions_per_second": {"minimum": 20.0, "preferred": 24.0},
        },
    }
    if not args.summary_only:
        assert args.output_dir is not None
        report_path = args.output_dir / "k3_compact_schedule.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
