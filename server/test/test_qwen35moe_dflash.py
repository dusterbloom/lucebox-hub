#!/usr/bin/env python3
"""Opt-in tests for the local qwen35moe DFlash drafter artifact.

These tests validate private model files, so pytest skips the module unless
QWEN35MOE_DFLASH_TEST=1 is set.
"""
import json
import os
import struct
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = Path("/home/peppi/models/qwen3.6-35b-a3b-dflash")
MODEL_DIR = Path(os.environ.get("QWEN35MOE_DFLASH_DIR", str(DEFAULT_MODEL_DIR)))
SAFETENSORS = Path(os.environ.get("QWEN35MOE_DFLASH_SAFETENSORS", str(MODEL_DIR / "model.safetensors")))
GGUF_OUT = Path(os.environ.get("QWEN35MOE_DFLASH_GGUF", str(MODEL_DIR / "qwen3.6-35b-a3b-dflash-f16.gguf")))
CONFIG_JSON = Path(os.environ.get("QWEN35MOE_DFLASH_CONFIG", str(MODEL_DIR / "config.json")))

RUN_LOCAL_ARTIFACT_TESTS = os.environ.get("QWEN35MOE_DFLASH_TEST") == "1"


def require_local_artifacts():
    if RUN_LOCAL_ARTIFACT_TESTS:
        return
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("requires private qwen35moe DFlash artifacts; set QWEN35MOE_DFLASH_TEST=1")
    raise RuntimeError("set QWEN35MOE_DFLASH_TEST=1 to run local artifact tests")


def read_gguf_kv(path):
    """Parse GGUF key-value metadata.

    GGUF value type enum (from llama.cpp/gguf-py GGUFValueType):
      0=UINT8  1=INT8   2=UINT16  3=INT16  4=UINT32  5=INT32
      6=FLOAT32  7=BOOL  8=STRING  9=ARRAY  10=UINT64  11=INT64  12=FLOAT64
    """
    # elem_size: bytes per element for scalar types (string/array handled specially)
    ELEM_FMT = {
        0: ("<B", 1),   # UINT8
        1: ("<b", 1),   # INT8
        2: ("<H", 2),   # UINT16
        3: ("<h", 2),   # INT16
        4: ("<I", 4),   # UINT32
        5: ("<i", 4),   # INT32
        6: ("<f", 4),   # FLOAT32
        7: ("<B", 1),   # BOOL (stored as uint8)
        10: ("<Q", 8),  # UINT64
        11: ("<q", 8),  # INT64
        12: ("<d", 8),  # FLOAT64
    }

    def read_string(f):
        slen = struct.unpack("<Q", f.read(8))[0]
        return f.read(slen).decode("utf-8")

    def read_scalar(f, vtype):
        if vtype == 8:   # STRING
            return read_string(f)
        if vtype == 7:   # BOOL
            return bool(struct.unpack("<B", f.read(1))[0])
        fmt, sz = ELEM_FMT[vtype]
        return struct.unpack(fmt, f.read(sz))[0]

    def skip_array(f):
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_n    = struct.unpack("<Q", f.read(8))[0]
        if arr_type == 8:  # string array
            for _ in range(arr_n):
                read_string(f)
        elif arr_type == 9:  # nested array (unusual)
            raise ValueError("nested GGUF arrays not supported")
        else:
            _, sz = ELEM_FMT[arr_type]
            f.read(arr_n * sz)
        return f"<array[{arr_type}] n={arr_n}>"

    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"not a GGUF: {magic}"
        _version   = struct.unpack("<I", f.read(4))[0]
        _n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv       = struct.unpack("<Q", f.read(8))[0]
        kv = {}
        for _ in range(n_kv):
            key   = read_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype == 9:   # ARRAY
                val = skip_array(f)
            else:
                val = read_scalar(f, vtype)
            kv[key] = val
    return kv


def expected_context_length():
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON) as f:
            cfg = json.load(f)
        return int(cfg.get("max_position_embeddings", 32768))
    return 32768


def test_gguf_exists():
    require_local_artifacts()
    assert GGUF_OUT.exists(), f"GGUF not yet created at {GGUF_OUT}"


def test_gguf_arch_metadata():
    require_local_artifacts()
    kv = read_gguf_kv(GGUF_OUT)
    arch = kv.get("general.architecture", "")
    A = f"{arch}." if arch else ""
    expected_ctx = expected_context_length()
    assert kv.get(f"{A}context_length", 0) == expected_ctx, \
        f"context_length must be {expected_ctx}, got {kv.get(f'{A}context_length', 'MISSING')}"
    assert kv.get(f"{A}embedding_length", 0) == 2048, \
        f"embedding_length must be 2048 (draft hidden), got {kv.get(f'{A}embedding_length', 'MISSING')}"
    assert kv.get(f"{A}block_count", 0) == 8, \
        f"block_count must be 8, got {kv.get(f'{A}block_count', 'MISSING')}"
    assert kv.get(f"{A}attention.head_count", 0) == 32, \
        f"head_count must be 32, got {kv.get(f'{A}attention.head_count', 'MISSING')}"
    assert kv.get(f"{A}attention.head_count_kv", 0) == 4, \
        f"head_count_kv must be 4, got {kv.get(f'{A}attention.head_count_kv', 'MISSING')}"
    assert kv.get(f"{A}dflash.n_target_layers", 0) == 5, \
        f"n_target_layers must be 5 (capture count), got {kv.get(f'{A}dflash.n_target_layers', 'MISSING')}"
    assert kv.get(f"{A}dflash.feat_dim_per_capture", 0) == 2048, \
        f"feat_dim_per_capture must be 2048, got {kv.get(f'{A}dflash.feat_dim_per_capture', 'MISSING')}"


def test_config_context_length():
    require_local_artifacts()
    assert CONFIG_JSON.exists(), f"config.json not found at {CONFIG_JSON}"
    assert expected_context_length() == 262144, \
        f"expected qwen35moe draft context 262144, got {expected_context_length()}"


def test_safetensors_fc_shape():
    require_local_artifacts()
    with open(SAFETENSORS, "rb") as f:
        sz  = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(sz))
    fc = hdr.get("fc.weight", {})
    assert fc.get("shape") == [2048, 10240], \
        f"fc.weight shape wrong: {fc.get('shape')}"


if __name__ == "__main__":
    if not RUN_LOCAL_ARTIFACT_TESTS:
        print("  SKIP local artifact tests (set QWEN35MOE_DFLASH_TEST=1)")
        sys.exit(0)
    failures = 0
    tests = [
        test_gguf_exists,
        test_gguf_arch_metadata,
        test_config_context_length,
        test_safetensors_fc_shape,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failures += 1
    sys.exit(failures)
