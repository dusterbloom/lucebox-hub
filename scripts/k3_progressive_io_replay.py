#!/usr/bin/env python3
"""Replay or simulate P20/P27 sidecar ranges independently of model compute.

The first implementation deliberately offers only honest buffered-pread modes.
Pinned/O_DIRECT/io_uring modes are added only after the request geometry is
measured; this tool never labels ordinary Python buffers as pinned.  The
bounded LRU mode is a metadata-only simulator whose keys bind the model and
sidecar provenance digests plus the exact aligned P27 source range.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


GIB = 1024**3


@dataclass(frozen=True)
class Request:
    path: str
    offset: int
    length: int
    layer: int


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    bytes: int
    sha256: str
    layer: int
    kind: str


@dataclass(frozen=True)
class CacheRequest:
    model_identity: str
    sidecar_identity: str
    path: str
    offset: int
    length: int
    layer: int
    sequence: int = 0
    base_pos: int = 0
    kind: str = "slab-page"

    @property
    def key(self) -> tuple[str, str, int, int]:
        # Do not let equal offsets from another model/sidecar alias, and do not
        # treat overlapping or adjacent ranges as hits.  An integrated cache
        # must return the exact P27 aligned source bytes selected by the trace.
        return (
            self.model_identity,
            self.sidecar_identity,
            self.offset,
            self.length,
        )


@dataclass(frozen=True)
class CacheTrace:
    requests: list[CacheRequest]
    sequence_count: int
    input_rows: int
    selected_component_rows: int
    selected_logical_bytes: int
    mean_tail_rows: int
    mean_tail_logical_bytes: int
    exact_fallback_rows: int
    exact_fallback_bytes: int


def proc_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    return 0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def load_requests(path: Path, include_means: bool) -> tuple[list[Request], int]:
    requests: list[Request] = []
    unresolved_fallback_bytes = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            region = row["region"]
            length = int(row["logical_length"])
            if region == "native-exact-expert":
                unresolved_fallback_bytes += length
                continue
            if region == "slab-mean" and not include_means:
                continue
            if region not in {"gate", "up", "down", "slab-mean"}:
                continue
            requests.append(Request(
                row["file_path"], int(row["file_offset"]), length,
                int(row["model_layer"])))
    return requests, unresolved_fallback_bytes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_cache_identities(
        path: Path) -> tuple[
            str, dict[str, ArtifactIdentity], dict[str, ArtifactIdentity]]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read cache identity manifest {path}: {error}")
    if manifest.get("schema") != "kimi-k3-all-layer-calibrated-slab-runtime-v2":
        raise ValueError("cache identity manifest has an incompatible schema")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("cache identity manifest has no layers")

    model_identities: set[str] = set()
    sidecars: dict[str, ArtifactIdentity] = {}
    means: dict[str, ArtifactIdentity] = {}
    for item in layers:
        if not isinstance(item, dict):
            raise ValueError("cache identity manifest contains a malformed layer")
        provenance = item.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("cache identity manifest layer has no provenance")
        model_layer = item.get("model_layer")
        model_identity = provenance.get("model_checksum_registry_sha256")
        sidecar_path = provenance.get("sidecar")
        sidecar_bytes = provenance.get("sidecar_bytes")
        sidecar_identity = provenance.get("sidecar_sha256")
        mean_path = item.get("output")
        mean_bytes = item.get("output_bytes")
        mean_identity = item.get("output_sha256")
        if (not isinstance(model_layer, int) or model_layer <= 0 or
                not valid_sha256(model_identity) or
                not isinstance(sidecar_path, str) or not sidecar_path or
                not isinstance(sidecar_bytes, int) or sidecar_bytes <= 0 or
                not valid_sha256(sidecar_identity)):
            raise ValueError(
                "cache identity manifest has incomplete model/sidecar identity")
        model_identities.add(model_identity.lower())
        canonical = str(Path(sidecar_path).resolve(strict=False))
        identity = ArtifactIdentity(
            canonical, sidecar_bytes, sidecar_identity.lower(), model_layer,
            "slab-page")
        previous = sidecars.get(canonical)
        if previous is not None and previous != identity:
            raise ValueError("cache identity manifest aliases a sidecar path")
        sidecars[canonical] = identity
        if (not isinstance(mean_path, str) or not mean_path or
                not isinstance(mean_bytes, int) or mean_bytes <= 0 or
                not valid_sha256(mean_identity)):
            raise ValueError(
                "cache identity manifest has incomplete mean-tail identity")
        canonical_mean = str(Path(mean_path).resolve(strict=False))
        mean = ArtifactIdentity(
            canonical_mean, mean_bytes, mean_identity.lower(), model_layer,
            "mean-tail")
        previous_mean = means.get(canonical_mean)
        if previous_mean is not None and previous_mean != mean:
            raise ValueError("cache identity manifest aliases a mean-tail path")
        means[canonical_mean] = mean
    if len(model_identities) != 1:
        raise ValueError(
            "cache identity manifest must bind one model checksum registry")
    return next(iter(model_identities)), sidecars, means


def load_cache_trace(
        path: Path, model_identity: str,
        sidecars: dict[str, ArtifactIdentity],
        means: dict[str, ArtifactIdentity], collect_requests: bool = True,
        request_observer: Callable[[CacheRequest], None] | None = None) -> CacheTrace:
    requests: list[CacheRequest] = []
    input_rows = 0
    selected_component_rows = 0
    selected_logical_bytes = 0
    mean_tail_rows = 0
    mean_tail_logical_bytes = 0
    exact_fallback_rows = 0
    exact_fallback_bytes = 0
    sequence = 0
    last_base_pos: int | None = None
    emitted_requests = 0

    def emit(request: CacheRequest) -> None:
        nonlocal emitted_requests
        emitted_requests += 1
        if collect_requests:
            requests.append(request)
        if request_observer is not None:
            request_observer(request)

    try:
        handle = path.open(newline="")
    except OSError as error:
        raise ValueError(f"cannot read cache trace {path}: {error}")
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "request_id", "model_layer", "region", "exact_fallback",
            "file_path", "file_offset", "logical_length", "aligned_offset",
            "aligned_length", "explicit_read_bytes",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("cache trace is missing required P27 columns")
        for row in reader:
            input_rows += 1
            try:
                region = row["region"]
                logical_length = int(row["logical_length"])
                explicit_read_bytes = int(row["explicit_read_bytes"])
                exact_fallback = int(row["exact_fallback"])
                layer = int(row["model_layer"])
                base_pos = int(row["base_pos"])
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"cache trace row {input_rows} is malformed: {error}")
            if logical_length < 0 or explicit_read_bytes < 0:
                raise ValueError(
                    f"cache trace row {input_rows} has negative byte counts")
            if last_base_pos is not None and base_pos < last_base_pos:
                sequence += 1
            last_base_pos = base_pos
            if region == "native-exact-expert" or exact_fallback:
                exact_fallback_rows += 1
                exact_fallback_bytes += logical_length
                continue
            if region == "slab-mean":
                mean_tail_rows += 1
                mean_tail_logical_bytes += logical_length
                if explicit_read_bytes == 0:
                    continue
                try:
                    offset = int(row["aligned_offset"])
                    length = int(row["aligned_length"])
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"cache trace row {input_rows} has an invalid mean-tail range: {error}")
                if offset < 0 or length <= 0 or explicit_read_bytes != length:
                    raise ValueError(
                        f"cache trace row {input_rows} is not an exact mean-tail read")
                canonical = str(Path(row["file_path"]).resolve(strict=False))
                identity = means.get(canonical)
                if identity is None:
                    raise ValueError(
                        f"cache trace row {input_rows} has an unregistered mean-tail artifact")
                if layer != identity.layer:
                    raise ValueError(
                        f"cache trace row {input_rows} disagrees with mean-tail layer")
                if offset + length > identity.bytes:
                    raise ValueError(
                        f"cache trace row {input_rows} lies outside its mean-tail artifact")
                emit(CacheRequest(
                    model_identity=model_identity,
                    sidecar_identity=identity.sha256,
                    path=identity.path,
                    offset=offset,
                    length=length,
                    layer=layer,
                    sequence=sequence,
                    base_pos=base_pos,
                    kind=identity.kind,
                ))
                continue
            if region not in {"gate", "up", "down"}:
                continue

            selected_component_rows += 1
            selected_logical_bytes += logical_length
            if explicit_read_bytes == 0:
                # P27 emits one physical aligned read on the gate trace row;
                # its up/down rows describe slices of those same source bytes.
                continue
            try:
                offset = int(row["aligned_offset"])
                length = int(row["aligned_length"])
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"cache trace row {input_rows} has an invalid range: {error}")
            if offset < 0 or length <= 0 or explicit_read_bytes != length:
                raise ValueError(
                    f"cache trace row {input_rows} is not an exact P27 aligned read")
            canonical = str(Path(row["file_path"]).resolve(strict=False))
            identity = sidecars.get(canonical)
            if identity is None:
                raise ValueError(
                    f"cache trace row {input_rows} has an unregistered sidecar")
            if layer != identity.layer:
                raise ValueError(
                    f"cache trace row {input_rows} disagrees with sidecar layer")
            if offset + length > identity.bytes:
                raise ValueError(
                    f"cache trace row {input_rows} lies outside its sidecar")
            emit(CacheRequest(
                model_identity=model_identity,
                sidecar_identity=identity.sha256,
                path=identity.path,
                offset=offset,
                length=length,
                layer=layer,
                sequence=sequence,
                base_pos=base_pos,
                kind=identity.kind,
            ))

    if emitted_requests == 0:
        raise ValueError("cache trace contains no exact P27 slab-page reads")
    return CacheTrace(
        requests=requests,
        sequence_count=sequence + 1,
        input_rows=input_rows,
        selected_component_rows=selected_component_rows,
        selected_logical_bytes=selected_logical_bytes,
        mean_tail_rows=mean_tail_rows,
        mean_tail_logical_bytes=mean_tail_logical_bytes,
        exact_fallback_rows=exact_fallback_rows,
        exact_fallback_bytes=exact_fallback_bytes,
    )


class BoundedLruSimulation:
    def __init__(self, capacity_bytes: int, cache_scope: str = "slab-only",
                 phase: str = "all", reset_policy: str = "none",
                 prompt_lengths: list[int] | None = None):
        if capacity_bytes <= 0:
            raise ValueError("cache capacity must be positive")
        if cache_scope not in {"slab-only", "unified"}:
            raise ValueError("cache scope must be slab-only or unified")
        if phase not in {"all", "decode"}:
            raise ValueError("cache phase must be all or decode")
        if reset_policy not in {"none", "sequence"}:
            raise ValueError("cache reset policy must be none or sequence")
        if phase == "decode" and not prompt_lengths:
            raise ValueError("decode simulation requires prompt lengths")
        self.capacity_bytes = capacity_bytes
        self.cache_scope = cache_scope
        self.phase = phase
        self.reset_policy = reset_policy
        self.prompt_lengths = prompt_lengths
        self.resident: OrderedDict[
            tuple[str, str, int, int], int] = OrderedDict()
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.hit_bytes = 0
        self.miss_bytes = 0
        self.oversized_requests = 0
        self.warmup_requests = 0
        self.warmup_request_bytes = 0
        self.warmup_hits = 0
        self.warmup_hit_bytes = 0
        self.warmup_misses = 0
        self.warmup_miss_bytes = 0
        self.sequence_resets = 0
        self.previous_sequence: int | None = None

    def observe(self, request: CacheRequest) -> None:
        if self.cache_scope == "slab-only" and request.kind != "slab-page":
            return
        if request.sequence != self.previous_sequence:
            if (self.previous_sequence is not None and
                    self.reset_policy == "sequence"):
                self.resident.clear()
                self.resident_bytes = 0
                self.sequence_resets += 1
            self.previous_sequence = request.sequence
        measured = self.phase == "all"
        if self.phase == "decode":
            assert self.prompt_lengths is not None
            if request.sequence >= len(self.prompt_lengths):
                raise ValueError(
                    "cache trace has more sequences than its manifest")
            measured = (
                request.base_pos >= self.prompt_lengths[request.sequence])
        key = request.key
        cached = self.resident.get(key)
        if cached is not None:
            if cached != request.length:
                raise AssertionError(
                    "cache key length disagrees with resident entry")
            if measured:
                self.hits += 1
                self.hit_bytes += request.length
            else:
                self.warmup_requests += 1
                self.warmup_request_bytes += request.length
                self.warmup_hits += 1
                self.warmup_hit_bytes += request.length
            self.resident.move_to_end(key)
            return
        if measured:
            self.misses += 1
            self.miss_bytes += request.length
        else:
            self.warmup_requests += 1
            self.warmup_request_bytes += request.length
            self.warmup_misses += 1
            self.warmup_miss_bytes += request.length
        if request.length > self.capacity_bytes:
            self.oversized_requests += 1
            return
        while (self.resident and
               self.resident_bytes + request.length > self.capacity_bytes):
            _, evicted_bytes = self.resident.popitem(last=False)
            self.resident_bytes -= evicted_bytes
            self.evictions += 1
        self.resident[key] = request.length
        self.resident_bytes += request.length
        self.peak_resident_bytes = max(
            self.peak_resident_bytes, self.resident_bytes)

    def result(self) -> dict[str, object]:
        total_bytes = self.hit_bytes + self.miss_bytes
        total_requests = self.hits + self.misses
        return {
            "capacity_bytes": self.capacity_bytes,
            "capacity_gib": self.capacity_bytes / GIB,
            "cache_scope": self.cache_scope,
            "measured_phase": self.phase,
            "prefill_warms_decode": self.phase == "decode",
            "reset_policy": self.reset_policy,
            "sequence_resets": self.sequence_resets,
            "requests": total_requests,
            "request_bytes": total_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "request_hit_fraction": (
                self.hits / total_requests if total_requests else 0.0),
            "hit_bytes": self.hit_bytes,
            "miss_bytes": self.miss_bytes,
            "byte_hit_fraction": (
                self.hit_bytes / total_bytes if total_bytes else 0.0),
            "avoided_physical_read_bytes": self.hit_bytes,
            "required_physical_read_bytes": self.miss_bytes,
            "warmup_requests": self.warmup_requests,
            "warmup_request_bytes": self.warmup_request_bytes,
            "warmup_hits": self.warmup_hits,
            "warmup_hit_bytes": self.warmup_hit_bytes,
            "warmup_misses": self.warmup_misses,
            "warmup_miss_bytes": self.warmup_miss_bytes,
            "total_required_physical_read_bytes": (
                self.miss_bytes + self.warmup_miss_bytes),
            "evictions": self.evictions,
            "oversized_requests": self.oversized_requests,
            "resident_entries_at_end": len(self.resident),
            "resident_bytes_at_end": self.resident_bytes,
            "peak_resident_bytes": self.peak_resident_bytes,
            "capacity_respected": (
                self.peak_resident_bytes <= self.capacity_bytes),
        }


def simulate_bounded_lru(
        requests: list[CacheRequest], capacity_bytes: int,
        cache_scope: str = "slab-only", phase: str = "all",
        reset_policy: str = "none",
        prompt_lengths: list[int] | None = None) -> dict[str, object]:
    simulation = BoundedLruSimulation(
        capacity_bytes, cache_scope, phase, reset_policy, prompt_lengths)
    for request in requests:
        simulation.observe(request)
    return simulation.result()


def load_sequence_manifest(path: Path) -> tuple[list[str], list[int]]:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read sequence manifest {path}: {error}")
    if manifest.get("schema") != "kimi-k3-h16-suite-v1":
        raise ValueError("sequence manifest has an incompatible schema")
    rows = manifest.get("sequences")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sequence manifest has no sequences")
    identifiers: list[str] = []
    prompt_lengths: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("sequence manifest contains a malformed row")
        identifier = row.get("id")
        prompt_length = row.get("prompt_token_count")
        if (not isinstance(identifier, str) or not identifier or
                not isinstance(prompt_length, int) or prompt_length <= 0):
            raise ValueError("sequence manifest has invalid prompt metadata")
        identifiers.append(identifier)
        prompt_lengths.append(prompt_length)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sequence manifest identifiers are not unique")
    return identifiers, prompt_lengths


def simulate_cache_trace(
        trace_path: Path, identity_manifest: Path,
        capacities_gib: list[int], cache_scopes: list[str] | None = None,
        phase: str = "all", reset_policy: str = "none",
        sequence_manifest: Path | None = None) -> dict[str, object]:
    if not capacities_gib or any(value not in {2, 4, 8, 16}
                                 for value in capacities_gib):
        raise ValueError(
            "cache capacities must be selected from 2, 4, 8, or 16 GiB")
    if len(set(capacities_gib)) != len(capacities_gib):
        raise ValueError("cache capacities must be unique")
    scopes = cache_scopes or ["slab-only"]
    if (not scopes or len(scopes) != len(set(scopes)) or
            any(scope not in {"slab-only", "unified"} for scope in scopes)):
        raise ValueError("cache scopes must be unique slab-only/unified values")
    if phase not in {"all", "decode"}:
        raise ValueError("cache phase must be all or decode")
    if reset_policy not in {"none", "sequence"}:
        raise ValueError("cache reset policy must be none or sequence")
    if phase == "decode" and sequence_manifest is None:
        raise ValueError("decode simulation requires a sequence manifest")
    model_identity, sidecars, means = load_cache_identities(identity_manifest)
    sequence_ids: list[str] = []
    prompt_lengths: list[int] | None = None
    if sequence_manifest is not None:
        sequence_ids, prompt_lengths = load_sequence_manifest(sequence_manifest)
    simulations = [
        BoundedLruSimulation(
            capacity * GIB, scope, phase, reset_policy, prompt_lengths)
        for scope in scopes for capacity in capacities_gib
    ]
    unique_requests: dict[tuple[str, str, int, int], int] = {}
    prompt_local_unique_requests: dict[
        tuple[int, str, str, int, int], int] = {}
    slab_request_count = 0
    request_bytes = 0
    mean_request_count = 0
    mean_request_bytes = 0

    def observe(request: CacheRequest) -> None:
        nonlocal slab_request_count, request_bytes
        nonlocal mean_request_count, mean_request_bytes
        if request.kind == "slab-page":
            slab_request_count += 1
            request_bytes += request.length
            unique_requests.setdefault(request.key, request.length)
            prompt_local_unique_requests.setdefault(
                (request.sequence, *request.key), request.length)
        elif request.kind == "mean-tail":
            mean_request_count += 1
            mean_request_bytes += request.length
        for simulation in simulations:
            simulation.observe(request)

    trace = load_cache_trace(
        trace_path, model_identity, sidecars, means,
        collect_requests=False, request_observer=observe)
    if prompt_lengths is not None and len(prompt_lengths) != trace.sequence_count:
        raise ValueError(
            "cache trace sequence resets disagree with sequence manifest")
    unique_bytes = sum(unique_requests.values())
    prompt_local_unique_bytes = sum(prompt_local_unique_requests.values())
    repeat_instances = slab_request_count - len(unique_requests)
    repeated_bytes = request_bytes - unique_bytes
    prompt_local_repeat_instances = (
        slab_request_count - len(prompt_local_unique_requests))
    prompt_local_repeated_bytes = request_bytes - prompt_local_unique_bytes
    policies = [simulation.result() for simulation in simulations]
    return {
        "schema": "k3-progressive-slab-page-cache-simulation-v2",
        "backend": "bounded-lru-simulation",
        "runtime_mutation": False,
        "default_behavior_changed": False,
        "capacity_accounting": (
            "exact cached source bytes; simulator metadata is excluded"),
        "cache_scopes": scopes,
        "cache_key": [
            "model_checksum_registry_sha256", "source_artifact_sha256",
            "aligned_offset", "aligned_length",
        ],
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "identity_manifest": str(identity_manifest),
        "identity_manifest_sha256": sha256(identity_manifest),
        "model_checksum_registry_sha256": model_identity,
        "registered_sidecars": len(sidecars),
        "registered_mean_tail_artifacts": len(means),
        "measured_phase": phase,
        "prefill_warms_decode": phase == "decode",
        "reset_policy": reset_policy,
        "sequence_manifest": (
            str(sequence_manifest) if sequence_manifest is not None else None),
        "sequence_manifest_sha256": (
            sha256(sequence_manifest) if sequence_manifest is not None else None),
        "sequence_count": trace.sequence_count,
        "sequence_ids": sequence_ids,
        "accounting": {
            "input_rows": trace.input_rows,
            "selected_component_rows": trace.selected_component_rows,
            "selected_logical_bytes": trace.selected_logical_bytes,
            "slab_page_requests": slab_request_count,
            "slab_page_request_bytes": request_bytes,
            "global_unique_slab_pages": len(unique_requests),
            "global_unique_slab_page_bytes": unique_bytes,
            "global_repeated_slab_page_instances": repeat_instances,
            "global_unlimited_avoidable_bytes": repeated_bytes,
            "global_unlimited_byte_hit_fraction": (
                repeated_bytes / request_bytes if request_bytes else 0.0),
            "prompt_local_unique_slab_pages": len(
                prompt_local_unique_requests),
            "prompt_local_unique_slab_page_bytes": (
                prompt_local_unique_bytes),
            "prompt_local_repeated_slab_page_instances": (
                prompt_local_repeat_instances),
            "prompt_local_unlimited_avoidable_bytes": (
                prompt_local_repeated_bytes),
            "prompt_local_unlimited_byte_hit_fraction": (
                prompt_local_repeated_bytes / request_bytes
                if request_bytes else 0.0),
            "mean_tail_rows": trace.mean_tail_rows,
            "mean_tail_logical_bytes": trace.mean_tail_logical_bytes,
            "mean_tail_cacheable_requests": mean_request_count,
            "mean_tail_cacheable_request_bytes": mean_request_bytes,
            "exact_fallback_rows_uncached": trace.exact_fallback_rows,
            "exact_fallback_bytes_uncached": trace.exact_fallback_bytes,
        },
        "audit_fingerprint": "whole trace SHA-256 above",
        "policies": policies,
        "semantic_boundary": (
            "This simulator changes no selected bytes, fallback decisions, "
            "mean-tail reads, arithmetic, or accumulation order. A hit can only "
            "replace one byte-identical aligned source read."
        ),
    }


def coalesce_adjacent(requests: list[Request]) -> list[Request]:
    if not requests:
        return []
    result = [requests[0]]
    for item in requests[1:]:
        previous = result[-1]
        if (item.path == previous.path and item.layer == previous.layer and
                item.offset == previous.offset + previous.length):
            result[-1] = Request(
                previous.path, previous.offset,
                previous.length + item.length, previous.layer)
        else:
            result.append(item)
    return result


def drop_cache(paths: set[str]) -> None:
    if not hasattr(os, "posix_fadvise"):
        return
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def replay(requests: list[Request], queue_depth: int) -> dict[str, object]:
    descriptors = {path: os.open(path, os.O_RDONLY) for path in
                   sorted({item.path for item in requests})}
    latencies_ms: list[float] = []

    def read_one(item: Request) -> tuple[int, float]:
        started = time.perf_counter_ns()
        payload = os.pread(descriptors[item.path], item.length, item.offset)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if len(payload) != item.length:
            raise OSError(
                f"short read {item.path}@{item.offset}: "
                f"{len(payload)} != {item.length}")
        return len(payload), elapsed_ms

    before = proc_read_bytes()
    started = time.perf_counter()
    submitted = 0
    try:
        if queue_depth <= 1:
            for item in requests:
                read_bytes, latency = read_one(item)
                submitted += read_bytes
                latencies_ms.append(latency)
        else:
            with ThreadPoolExecutor(max_workers=queue_depth) as pool:
                for read_bytes, latency in pool.map(read_one, requests):
                    submitted += read_bytes
                    latencies_ms.append(latency)
    finally:
        for fd in descriptors.values():
            os.close(fd)
    elapsed = time.perf_counter() - started
    physical = max(0, proc_read_bytes() - before)
    return {
        "submitted_bytes": submitted,
        "os_physical_bytes": physical,
        "elapsed_seconds": elapsed,
        "submitted_gib_s": submitted / GIB / elapsed if elapsed else 0.0,
        "os_physical_gib_s": physical / GIB / elapsed if elapsed else 0.0,
        "request_latency_ms": {
            "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--mode", choices=(
                            "current", "batched-pread", "bounded-lru-sim"),
                        default="current")
    parser.add_argument("--queue-depth", type=int, default=1)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--include-means", action="store_true")
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--cache-gib", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--cache-scope", nargs="+", default=["slab-only"],
                        choices=("slab-only", "unified"))
    parser.add_argument("--phase", choices=("all", "decode"), default="all")
    parser.add_argument("--reset-policy", choices=("none", "sequence"),
                        default="none")
    parser.add_argument("--sequence-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queue_depth <= 0:
        parser.error("--queue-depth must be positive")

    if args.mode == "bounded-lru-sim":
        if args.identity_manifest is None:
            parser.error("--identity-manifest is required for bounded-lru-sim")
        if args.cold or args.include_means or args.queue_depth != 1:
            parser.error(
                "bounded-lru-sim does not accept replay-only I/O options")
        try:
            result = simulate_cache_trace(
                args.trace, args.identity_manifest, args.cache_gib,
                cache_scopes=args.cache_scope, phase=args.phase,
                reset_policy=args.reset_policy,
                sequence_manifest=args.sequence_manifest)
        except ValueError as error:
            parser.error(str(error))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    if (args.identity_manifest is not None or args.cache_gib != [2, 4, 8] or
            args.cache_scope != ["slab-only"] or args.phase != "all" or
            args.reset_policy != "none" or args.sequence_manifest is not None):
        parser.error("cache options require --mode bounded-lru-sim")

    raw, unresolved = load_requests(args.trace, args.include_means)
    requests = coalesce_adjacent(raw) if args.mode == "batched-pread" else raw
    if args.cold:
        drop_cache({item.path for item in requests})
    result = replay(requests, args.queue_depth)
    logical = sum(item.length for item in requests)
    result.update({
        "schema": "k3-progressive-io-replay-v1",
        "backend": args.mode,
        "destination_mode": "ordinary-host-buffer",
        "cold_cache_requested": args.cold,
        "queue_depth": args.queue_depth,
        "input_requests": len(raw),
        "submitted_requests": len(requests),
        "logical_bytes": logical,
        "unresolved_exact_fallback_bytes": unresolved,
        "physical_over_logical": (
            result["os_physical_bytes"] / logical if logical else None),
        "note": "exact fallback model-shard ranges are not replayed by this sidecar-only baseline",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
