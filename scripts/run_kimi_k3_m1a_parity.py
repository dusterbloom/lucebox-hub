#!/usr/bin/env python3
"""Capture the 12-prompt M1a production HTTP/SSE parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "kimi_k3_m1a_parity_capture_v1"
REFERENCE_SHA256 = "124b6b119ce28418967d0f541b485f88f63077b01ae78dccbcf88a922ed211bb"
TASK_IDS = (
    "fact-capital", "fact-science", "code-sum", "code-function",
    "reasoning-marble", "reasoning-rate", "grammar-apples",
    "grammar-agreement", "translation-italian", "translation-spanish",
    "extract-code", "extract-decoys",
)
LAUNCH_SCHEMA = "kimi_k3_m1a_launch_manifest_v1"
CHAT_TEMPLATE_SHA256 = "05bb501f8ac31fa6b0bf04803b5ada49abf9cdd51c3c90a4719b739df0000722"
REQUIRED_ENVIRONMENT = {
    "GGML_BATCH_PEER_COPIES": "1",
    "DFLASH_MOE_NVME_DIRECT": "on",
    "DFLASH_MOE_NVME_DEVICE_CACHE_MB": "8192",
    "DFLASH_MOE_TP_GPU": "0",
    "DFLASH_MOE_PRIMARY_SHARE_PER_MILLE": "930",
    "DFLASH_KIMI_LAYER1_PROVIDER": "all-layers-calibrated96",
    "DFLASH_KIMI_SIDECAR_AUTHORITATIVE": "1",
    "DFLASH_KIMI_P20_PHYSICAL_LAYOUT": "scratch",
    "DFLASH_KIMI_P20_IO_BACKEND": "direct-pread",
    "DFLASH_KIMI_P23_PERSISTENT_SCRATCH": "1",
    "DFLASH_KIMI_P25_COMPACT_UPLOAD": "1",
    "DFLASH_KIMI_P26_PINNED_COMPACT": "1",
    "DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT": "1",
    "DFLASH_KIMI_P30_HOST_CACHE_MB": "16384",
    "DFLASH_KIMI_P41_COMPACT_EXECUTOR": "1",
    "DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN": "1",
    "DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE": "1",
    "DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP": "1",
    "DFLASH_KIMI_P52_PERSISTENT_ROUTED_JOIN": "0",
    "DFLASH_KIMI_P53_DEVICE_HIDDEN_CHAIN": "0",
    "DFLASH_SERVER_COMMITTED_TOKEN_TRACE": "1",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_bytes(path: Path, value: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(partial, path)
    partial.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def load_references(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or sha256(path) != REFERENCE_SHA256:
        raise ValueError("frozen P55 reference hash mismatch")
    value = json.loads(path.read_text())
    if value.get("schema") != "kimi-k3-h16-suite-v1" or \
            value.get("provider") != "all-layers-calibrated96" or \
            value.get("chat_template") != "gguf-jinja-thinking-off" or \
            value.get("thinking_enabled") is not False or \
            value.get("max_context") != 256 or value.get("n_gen") != 24 or \
            value.get("paired") is not False:
        raise ValueError("frozen P55 reference provenance mismatch")
    sequences = value.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("frozen P55 sequences absent")
    by_id = {row.get("id"): row for row in sequences if isinstance(row, dict)}
    if set(TASK_IDS) - set(by_id):
        raise ValueError("required P55 parity rows absent")
    result = []
    for task_id in TASK_IDS:
        row = by_id[task_id]
        if not isinstance(row.get("text"), str) or not row["text"] or \
                not all(isinstance(item, int) and item >= 0
                        for item in row.get("prompt_tokens", [])) or \
                not all(isinstance(item, int) and item >= 0
                        for item in row.get("output_tokens", [])):
            raise ValueError(f"malformed P55 row {task_id}")
        result.append({"id": row["id"], "prompt": row["text"],
                       "prompt_tokens": row["prompt_tokens"],
                       "output_tokens": row["output_tokens"]})
    return result


def validate_launch(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("registered launch manifest absent")
    value = json.loads(path.read_text())
    repo = value.get("repo")
    model = value.get("model", {})
    executable = value.get("binary", {})
    cmake = value.get("cmake_cache", {})
    template = value.get("chat_template", {})
    contract = value.get("g2_contract", {})
    environment = value.get("environment", {})
    aux = value.get("aux_manifest", {})
    sidecar = value.get("sidecar_manifest", {})
    policy = value.get("h23_policy", {})
    card_path = Path(str(repo)) / "share/model_cards/kimi-k3.json"
    card = {"path": str(card_path), "sha256":
            value.get("source_hashes", {}).get("share/model_cards/kimi-k3.json"),
            "source": str(card_path),
            "upstream_source": "https://huggingface.co/moonshotai/Kimi-K3"}
    if value.get("schema") != LAUNCH_SCHEMA or \
            contract.get("model_name") != "kimi-k3" or \
            not isinstance(value.get("bind"), str) or \
            not isinstance(card.get("path"), str) or not card["path"] or \
            not isinstance(card.get("sha256"), str) or len(card["sha256"]) != 64 or \
            not isinstance(model.get("frozen_p55_manifest"), str) or \
            not isinstance(model.get("frozen_p55_manifest_sha256"), str) or \
            model.get("shard_count") != 14 or \
            not isinstance(model.get("shards"), list) or len(model["shards"]) != 14 or \
            model.get("total_bytes") != 585690490336 or \
            not isinstance(executable.get("path"), str) or \
            not isinstance(executable.get("sha256"), str) or \
            not isinstance(cmake.get("path"), str) or \
            not isinstance(cmake.get("sha256"), str) or \
            not isinstance(template.get("path"), str) or \
            template.get("sha256") != CHAT_TEMPLATE_SHA256 or \
            contract.get("max_context") != 256 or \
            contract.get("prefix_cache_slots") != 0 or \
            contract.get("prefill_cache_slots") != 0 or \
            contract.get("disk_prefix_cache") != "off" or \
            value.get("server_argv", [None])[0] != executable.get("path") or \
            "--target-device" not in value.get("server_argv", []) or \
            "hip:1" not in value.get("server_argv", []) or \
            "--moe-storage" not in value.get("server_argv", []) or \
            "ssd" not in value.get("server_argv", []) or \
            any(environment.get(key) != expected
                for key, expected in REQUIRED_ENVIRONMENT.items()) or \
            environment.get("DFLASH_KIMI_CALIBRATED96_AUX_DIR") != \
                str(Path(aux.get("path", "")).parent) or \
            environment.get("DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR") != \
                str(Path(sidecar.get("path", "")).parent) or \
            environment.get("DFLASH_KIMI_H22_LAYER_BUDGETS") != policy.get("path") or \
            policy.get("sha256") != \
                "b73d203e9ae7d4f382baf3f30cf0381387596edda7a6fb16c6cf5c88626ad97a":
        raise ValueError("registered launch contract mismatch")
    for identity in (card, executable, cmake, template, aux, sidecar, policy):
        for key, item in identity.items():
            if key.endswith("sha256") and (not isinstance(item, str) or len(item) != 64):
                raise ValueError("malformed registered SHA-256")
    value["base_url"] = "http://" + value["bind"]
    value["model_name"] = contract["model_name"]
    value["model_card"] = card
    value["checkpoint"] = model
    value["executable"] = executable
    value["cmake"] = cmake
    value["effective_environment"] = environment
    return value


def observed_response_id(raw: bytes, stream: bool) -> str:
    values: list[dict[str, Any]] = []
    if stream:
        done = 0
        for line in raw.decode("utf-8").splitlines():
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data: "):
                raise ValueError("malformed SSE response")
            payload = line[6:]
            if payload == "[DONE]":
                done += 1
                continue
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("malformed SSE JSON event")
            values.append(value)
        if done != 1 or not values:
            raise ValueError("SSE response must contain events and one DONE")
    else:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("malformed JSON response")
        values = [value]
    ids = {value.get("id") for value in values}
    if len(ids) != 1:
        raise ValueError("response has inconsistent IDs")
    response_id = next(iter(ids))
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("response ID absent")
    return response_id


def request(url: str, body: bytes | None, timeout: float) -> tuple[int, str, bytes]:
    headers = {"Accept": "application/json"}
    method = "GET"
    if body is not None:
        headers["Content-Type"] = "application/json"
        method = "POST"
    call = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--server-stderr", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.timeout_seconds > 3600:
        parser.error("timeout must be in (0,3600]")
    if args.artifact_dir.exists() or args.manifest.exists() or \
            args.manifest.with_name(args.manifest.name + ".partial").exists():
        parser.error("artifact directory and manifest must be fresh")
    if not args.server_stderr.is_file() or args.server_stderr.is_symlink():
        parser.error("server stderr must be a regular non-symlink")
    try:
        references = load_references(args.reference)
        launch = validate_launch(args.launch_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.artifact_dir.mkdir(parents=False)
    base = launch["base_url"].rstrip("/")
    endpoints: list[dict[str, Any]] = []
    for name, route in (("health", "/health"), ("models", "/v1/models"),
                        ("props", "/props")):
        status, content_type, raw = request(base + route, None, args.timeout_seconds)
        path = args.artifact_dir / f"{name}.response"
        atomic_bytes(path, raw)
        endpoints.append({"name": name, "route": route, "status": status,
                          "content_type": content_type,
                          "path": str(path.resolve()), "sha256": sha256(path)})
    requests: list[dict[str, Any]] = []
    ordinal = 0
    for row in references:
        for stream in (False, True):
            body_value = {
                "model": "kimi-k3",
                "messages": [{"role": "user", "content": row["prompt"]}],
                "max_tokens": 24, "temperature": 0, "seed": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": stream,
            }
            body = (json.dumps(body_value, sort_keys=True,
                               separators=(",", ":")) + "\n").encode()
            stem = f"{ordinal:02d}-{row['id']}-{'sse' if stream else 'json'}"
            body_path = args.artifact_dir / f"{stem}.request.json"
            response_path = args.artifact_dir / f"{stem}.response"
            atomic_bytes(body_path, body)
            status, content_type, raw = request(
                base + "/v1/chat/completions", body, args.timeout_seconds)
            atomic_bytes(response_path, raw)
            if status != 200:
                raise ValueError(f"generation request {ordinal} returned HTTP {status}")
            response_id = observed_response_id(raw, stream)
            requests.append({
                "ordinal": ordinal, "task_id": row["id"], "stream": stream,
                "status": status, "content_type": content_type,
                "request": str(body_path.resolve()),
                "request_sha256": sha256(body_path),
                "response": str(response_path.resolve()),
                "response_sha256": sha256(response_path),
                "response_id": response_id,
            })
            ordinal += 1
    response_ids = [item["response_id"] for item in requests]
    expected_count = len(TASK_IDS) * 2
    if len(response_ids) != expected_count or \
            len(set(response_ids)) != expected_count:
        raise ValueError("distinct server response IDs required for every arm")
    stderr_snapshot = args.artifact_dir / "server-stderr.snapshot.log"
    atomic_bytes(stderr_snapshot, args.server_stderr.read_bytes())
    manifest = {
        "schema": SCHEMA, "scope": "prospective_m1a_g2_only",
        "reference": str(args.reference.resolve()),
        "reference_sha256": REFERENCE_SHA256,
        "launch_manifest": str(args.launch_manifest.resolve()),
        "launch_manifest_sha256": sha256(args.launch_manifest),
        "server_stderr": str(stderr_snapshot.resolve()),
        "server_stderr_sha256": sha256(stderr_snapshot),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "endpoints": endpoints, "requests": requests,
        "generation_request_count": len(requests),
        "expected_response_ids": response_ids,
        "request_order": [[row["id"], mode]
                          for row in references for mode in (False, True)],
        "request_contract": {"model": "kimi-k3", "max_tokens": 24,
                             "temperature": 0, "seed": 0,
                             "chat_template_kwargs": {
                                 "enable_thinking": False}},
        "analysis_performed": False,
    }
    atomic_json(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
