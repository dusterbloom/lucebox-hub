#!/usr/bin/env python3
"""
Adapted converter: z-lab/Qwen3-Coder-30B-A3B-DFlash safetensors -> GGUF.
Forked from dflash/scripts/convert_dflash_to_gguf.py with constants
updated for the 30B-A3B draft (target = SR2AM-v1.0-30B / Qwen3-30B-A3B).

Source config: https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash/blob/main/config.json
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np
import gguf

# ── Constants for 30B-A3B DFlash draft (target: Qwen3-30B-A3B / SR2AM-30B) ──
ARCH                = "qwen35-dflash-draft"
HIDDEN              = 2048           # was 5120 (27B)
N_LAYER             = 8              # was 5; config: num_hidden_layers=8
N_HEAD              = 32
N_HEAD_KV           = 4              # was 8
HEAD_DIM            = 128
INTERMEDIATE        = 6144           # was 17408
VOCAB               = 151936         # was 248320 (Qwen3 tokenizer)
N_TARGET_LAYERS     = 5              # capture count (5 ids: [1,12,23,34,45])
ROPE_THETA          = 10_000_000.0   # was 1e6
RMS_EPS             = 1e-6
MASK_TOKEN_ID       = 151669         # was 248070
BLOCK_SIZE          = 16
CTX_LEN             = 262144         # was 32768; config: max_position_embeddings


def map_name(name: str):
    if name == "fc.weight":          return "dflash.fc.weight"
    if name == "hidden_norm.weight": return "dflash.hidden_norm.weight"
    if name == "norm.weight":        return "output_norm.weight"
    if name.startswith("layers."):
        parts = name.split(".", 2)
        if len(parts) < 3: return None
        i = int(parts[1]); rest = parts[2]
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


def load_safetensors_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        return header_size, json.loads(header_json)


def read_tensor_bytes(path, header_size, info):
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_size + start)
        return f.read(end - start)


def bytes_to_np(raw, dtype, shape):
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
        u32 = (u16.astype(np.uint32) << 16)
        f32 = u32.view("<f4").reshape(shape)
        return f32.astype("<f2")
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    raise ValueError(f"unsupported dtype {dtype}")


SAFETENSORS_DTYPE_TO_GGUF = {
    "F32":  gguf.GGMLQuantizationType.F32,
    "F16":  gguf.GGMLQuantizationType.F16,
    "BF16": gguf.GGMLQuantizationType.F16,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("safetensors", type=Path)
    ap.add_argument("out_gguf",    type=Path)
    args = ap.parse_args()

    if not args.safetensors.exists():
        print(f"[error] not found: {args.safetensors}", file=sys.stderr); sys.exit(1)

    print(f"[info] reading {args.safetensors}")
    header_size, header = load_safetensors_header(args.safetensors)
    n_entries = sum(1 for k in header if k != "__metadata__")
    print(f"[info]   {n_entries} tensor entries")

    writer = gguf.GGUFWriter(args.out_gguf, ARCH)
    writer.add_string("general.name", "Qwen3-Coder-30B-A3B-DFlash-Draft")
    writer.add_uint32(f"{ARCH}.context_length",          CTX_LEN)
    writer.add_uint32(f"{ARCH}.embedding_length",        HIDDEN)
    writer.add_uint32(f"{ARCH}.block_count",             N_LAYER)
    writer.add_uint32(f"{ARCH}.feed_forward_length",     INTERMEDIATE)
    writer.add_uint32(f"{ARCH}.attention.head_count",    N_HEAD)
    writer.add_uint32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    writer.add_uint32(f"{ARCH}.attention.key_length",    HEAD_DIM)
    writer.add_uint32(f"{ARCH}.attention.value_length",  HEAD_DIM)
    writer.add_uint32(f"{ARCH}.vocab_size",              VOCAB)
    writer.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS)
    writer.add_float32(f"{ARCH}.rope.freq_base",         ROPE_THETA)
    writer.add_uint32(f"{ARCH}.dflash.n_target_layers", N_TARGET_LAYERS)
    writer.add_uint32(f"{ARCH}.dflash.block_size",      BLOCK_SIZE)
    writer.add_uint32(f"{ARCH}.dflash.mask_token_id",   MASK_TOKEN_ID)

    pending = []
    for st_name, info in header.items():
        if st_name == "__metadata__": continue
        gguf_name = map_name(st_name)
        if gguf_name is None:
            print(f"[warn] skipping unmapped: {st_name}"); continue
        dtype = SAFETENSORS_DTYPE_TO_GGUF.get(info["dtype"])
        if dtype is None:
            print(f"[error] unsupported dtype {info['dtype']} for {st_name}", file=sys.stderr); sys.exit(1)
        pending.append((gguf_name, info["dtype"], info["shape"], info))

    def sort_key(t):
        n = t[0]
        if n.startswith("dflash."): return (0, n)
        if n.startswith("output_"): return (1, n)
        if n.startswith("blk."):
            i = int(n.split(".")[1])
            return (2, i, n)
        return (3, n)
    pending.sort(key=sort_key)

    for gguf_name, st_dtype, shape, info in pending:
        raw = read_tensor_bytes(args.safetensors, header_size, info)
        arr = bytes_to_np(raw, st_dtype, shape)
        raw_dtype = SAFETENSORS_DTYPE_TO_GGUF[st_dtype]
        is_norm = (gguf_name.endswith("_norm.weight") or
                   gguf_name == "output_norm.weight" or
                   gguf_name == "dflash.hidden_norm.weight")
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
    print(f"[done] {args.out_gguf}")


if __name__ == "__main__":
    main()
