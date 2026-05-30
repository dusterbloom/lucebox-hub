#!/usr/bin/env python3
"""
Convert the z-lab DFlash draft (safetensors, bf16) to a GGUF that
llama.cpp can load.

Uses llama.cpp's own gguf-py (deps/llama.cpp/gguf-py) — no hand-rolled
binary writer. The library handles header layout, alignment, BF16
storage, and tensor info offsets correctly.

DFlash draft is a 5-layer Qwen-style transformer with two extra
model-level singletons specific to the spec-decode block-diffusion
algorithm:
  - `fc.weight`           [hidden, 5*hidden]  — fuses 5 captured target
                                                 hidden states into the
                                                 draft's input
  - `hidden_norm.weight`  [hidden]            — RMSNorm applied right after
                                                 the fc projection

These are stored under the `dflash.` prefix so llama.cpp can fetch them
via a custom arch loader without colliding with any upstream tensor
name.

Usage:
  PYTHONPATH=../../dflash_ggml/deps/llama.cpp/gguf-py python convert_dflash_to_gguf.py \
    models/draft/model.safetensors \
    qwen3.5-27b-dflash-draft.gguf
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Use llama.cpp's own GGUF writer — adds bf16 / metadata / alignment
# correctness without any hand-rolled code.
import gguf

# ──────────────────────────────────────────────────────────────────────
# DFlash 27B draft architecture constants
# ──────────────────────────────────────────────────────────────────────

ARCH                = "qwen35-dflash-draft"
HIDDEN              = 5120
N_LAYER             = 5
N_HEAD              = 32          # query heads
N_HEAD_KV           = 8
HEAD_DIM            = 128
INTERMEDIATE        = 17408
VOCAB               = 248320
N_TARGET_LAYERS     = 5            # fc projects 5*hidden -> hidden
ROPE_THETA          = 1_000_000.0
RMS_EPS             = 1e-6
MASK_TOKEN_ID       = 248070
BLOCK_SIZE          = 16
CTX_LEN             = 32768


# ──────────────────────────────────────────────────────────────────────
# Tensor name mapping  —  DFlash safetensors -> llama.cpp GGUF
# ──────────────────────────────────────────────────────────────────────

def map_name(name: str) -> str | None:
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3: return None
        i = int(parts[1])
        rest = parts[2]
        layer_map = {
            "input_layernorm.weight":          f"blk.{i}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
            "self_attn.q_proj.weight":         f"blk.{i}.attn_q.weight",
            "self_attn.k_proj.weight":         f"blk.{i}.attn_k.weight",
            "self_attn.v_proj.weight":         f"blk.{i}.attn_v.weight",
            "self_attn.o_proj.weight":         f"blk.{i}.attn_output.weight",
            "self_attn.q_norm.weight":         f"blk.{i}.attn_q_norm.weight",
            "self_attn.k_norm.weight":         f"blk.{i}.attn_k_norm.weight",
            "mlp.gate_proj.weight":            f"blk.{i}.ffn_gate.weight",
            "mlp.up_proj.weight":              f"blk.{i}.ffn_up.weight",
            "mlp.down_proj.weight":            f"blk.{i}.ffn_down.weight",
        }
        return layer_map.get(rest)
    return None


# ──────────────────────────────────────────────────────────────────────
# safetensors reader  —  header parse + raw byte slice
# ──────────────────────────────────────────────────────────────────────

def load_safetensors_header(path: Path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        return header_size, json.loads(header_json)


def read_tensor_bytes(path: Path, header_size: int, info: dict) -> bytes:
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_size + start)
        return f.read(end - start)


def bytes_to_np(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        # Convert BF16 -> F16 on the host. Several ggml-cuda ops (mul,
        # binbcast) only accept F32 / F16 inputs, and llama.cpp's
        # build_norm path multiplies normalised activations by the norm
        # weight tensor. Storing the draft as F16 throughout sidesteps
        # the unsupported BF16 path entirely. Quality impact ~0 for
        # weight tensors (BF16 -> F16 keeps 10/8 mantissa bits anyway
        # after the implicit cast).
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        # bf16 = sign(1) + exp(8) + mantissa(7); reinterpret as f32 by
        # putting it in the high half, then narrow to f16.
        u32 = (u16.astype(np.uint32) << 16)
        f32 = u32.view("<f4").reshape(shape)
        return f32.astype("<f2")
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    raise ValueError(f"unsupported safetensors dtype {dtype}")


SAFETENSORS_DTYPE_TO_GGUF = {
    "F32":  gguf.GGMLQuantizationType.F32,
    "F16":  gguf.GGMLQuantizationType.F16,
    # BF16 in safetensors -> we narrow to F16 in bytes_to_np above.
    "BF16": gguf.GGMLQuantizationType.F16,
}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """Load model config.json and return a flat dict of relevant constants."""
    with open(config_path) as f:
        cfg = json.load(f)
    dc = cfg.get("dflash_config", {})
    target_ids = dc.get("target_layer_ids", [])
    return {
        "HIDDEN":           cfg["hidden_size"],
        "N_LAYER":          cfg["num_hidden_layers"],
        "N_HEAD":           cfg["num_attention_heads"],
        "N_HEAD_KV":        cfg["num_key_value_heads"],
        "HEAD_DIM":         cfg["head_dim"],
        "INTERMEDIATE":     cfg["intermediate_size"],
        "VOCAB":            cfg.get("vocab_size", VOCAB),
        "ROPE_THETA":       float(cfg.get("rope_theta", ROPE_THETA)),
        "RMS_EPS":          cfg.get("rms_norm_eps", RMS_EPS),
        "BLOCK_SIZE":       cfg.get("block_size", BLOCK_SIZE),
        "MASK_TOKEN_ID":    dc.get("mask_token_id", MASK_TOKEN_ID),
        "N_TARGET_LAYERS":  len(target_ids) if target_ids else N_TARGET_LAYERS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", type=Path)
    ap.add_argument("out_gguf",     type=Path)
    ap.add_argument("--config",     type=Path, default=None,
                    help="Path to config.json (inferred from safetensors dir if omitted)")
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] safetensors not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)

    # Resolve config: explicit > sibling of safetensors > module-level defaults
    config_path = args.config
    if config_path is None:
        candidate = args.safetensors.parent / "config.json"
        if candidate.exists():
            config_path = candidate
            print(f"[info] auto-detected config: {config_path}")

    # Override module-level constants from config when available
    _HIDDEN       = HIDDEN
    _N_LAYER      = N_LAYER
    _N_HEAD       = N_HEAD
    _N_HEAD_KV    = N_HEAD_KV
    _HEAD_DIM     = HEAD_DIM
    _INTERMEDIATE = INTERMEDIATE
    _VOCAB        = VOCAB
    _ROPE_THETA   = ROPE_THETA
    _RMS_EPS      = RMS_EPS
    _BLOCK_SIZE   = BLOCK_SIZE
    _MASK_TOKEN_ID    = MASK_TOKEN_ID
    _N_TARGET_LAYERS  = N_TARGET_LAYERS

    if config_path is not None:
        cfg = load_config(config_path)
        _HIDDEN          = cfg["HIDDEN"]
        _N_LAYER         = cfg["N_LAYER"]
        _N_HEAD          = cfg["N_HEAD"]
        _N_HEAD_KV       = cfg["N_HEAD_KV"]
        _HEAD_DIM        = cfg["HEAD_DIM"]
        _INTERMEDIATE    = cfg["INTERMEDIATE"]
        _VOCAB           = cfg["VOCAB"]
        _ROPE_THETA      = cfg["ROPE_THETA"]
        _RMS_EPS         = cfg["RMS_EPS"]
        _BLOCK_SIZE      = cfg["BLOCK_SIZE"]
        _MASK_TOKEN_ID   = cfg["MASK_TOKEN_ID"]
        _N_TARGET_LAYERS = cfg["N_TARGET_LAYERS"]
        print(f"[info] config: hidden={_HIDDEN} n_layer={_N_LAYER} n_head={_N_HEAD} "
              f"n_head_kv={_N_HEAD_KV} head_dim={_HEAD_DIM} intermediate={_INTERMEDIATE} "
              f"n_target_layers={_N_TARGET_LAYERS} rope_theta={_ROPE_THETA}")
    else:
        print("[info] no config.json found, using module-level defaults (27B draft)")

    print(f"[info] reading safetensors header from {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)

    # Architecture metadata
    writer.add_string("general.name", "Qwen3.5-DFlash-Draft")
    writer.add_uint32(f"{ARCH}.context_length",          CTX_LEN)
    writer.add_uint32(f"{ARCH}.embedding_length",        _HIDDEN)
    writer.add_uint32(f"{ARCH}.block_count",             _N_LAYER)
    writer.add_uint32(f"{ARCH}.feed_forward_length",     _INTERMEDIATE)
    writer.add_uint32(f"{ARCH}.attention.head_count",    _N_HEAD)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", _N_HEAD_KV)
    # llama.cpp uses key_length / value_length to override the default
    # n_embd_head = n_embd / n_head heuristic (DFlash has n_embd=5120
    # but head_dim=128 so n_head*head_dim=4096 != n_embd).
    writer.add_uint32(f"{ARCH}.attention.key_length",    _HEAD_DIM)
    writer.add_uint32(f"{ARCH}.attention.value_length",  _HEAD_DIM)
    writer.add_uint32(f"{ARCH}.vocab_size",              _VOCAB)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", _RMS_EPS)
    writer.add_float32(f"{ARCH}.rope.freq_base",         _ROPE_THETA)

    # DFlash-specific hyperparameters
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", _N_TARGET_LAYERS)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      _BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   _MASK_TOKEN_ID)
    # feat_dim_per_capture: dims the target captures per capture layer.
    # fc.weight input = n_target_layers * hidden, so per-capture = hidden.
    writer.add_uint32(f"{ARCH}.dflash.feat_dim_per_capture", _HIDDEN)

    # Walk + add tensors. Sort: dflash.* singletons first, then output_*,
    # then per-layer in numeric order — keeps the on-disk layout stable.
    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__":
            continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}")
            continue
        dtype = SAFETENSORS_DTYPE_TO_GGUF.get(info["dtype"])
        if dtype is None:
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}", file=sys.stderr)
            sys.exit(1)
        pending.append((gguf_name, info["dtype"], info["shape"], info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."):     return (0, n)
        if n.startswith("output_"):     return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    for gguf_name, st_dtype, shape, info in pending:
        raw = read_tensor_bytes(args.safetensors, header_size, info)
        arr = bytes_to_np(raw, st_dtype, shape)
        raw_dtype = SAFETENSORS_DTYPE_TO_GGUF[st_dtype]
        # Norm weights and the dflash hidden_norm singleton must be F32:
        # the ggml-cuda mul path that build_norm emits asserts on
        # src1's element size alignment (binbcast.cu nb10 % sizeof) and
        # the F32 path is the safest cross-quant fallback.
        is_norm = (
            gguf_name.endswith("_norm.weight") or
            gguf_name == "output_norm.weight" or
            gguf_name == "dflash.hidden_norm.weight"
        )
        if is_norm:
            arr = arr.astype("<f4")
            raw_dtype = gguf.GGMLQuantizationType.F32
        writer.add_tensor(gguf_name, arr, raw_dtype=raw_dtype)
        print(f"[tensor] {gguf_name:50s} {st_dtype:4s}->{raw_dtype.name:4s} {tuple(shape)}")

    print(f"[info] writing {args.out_gguf}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[done] wrote {args.out_gguf}")


if __name__ == "__main__":
    main()
