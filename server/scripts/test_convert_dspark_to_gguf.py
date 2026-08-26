from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import convert_dspark_to_gguf as converter
import gguf


def _to_bf16_bytes(values: np.ndarray) -> bytes:
    words = values.astype("<f4").view("<u4")
    return (words >> 16).astype("<u2").tobytes()


def _write_safetensors(
    path: Path,
    tensors: dict[str, np.ndarray],
    dtypes: dict[str, str] | None = None,
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    payload = bytearray()
    for name, values in tensors.items():
        dtype = (dtypes or {}).get(name, "BF16")
        if dtype == "BF16":
            raw = _to_bf16_bytes(values)
        elif dtype == "F16":
            raw = values.astype("<f2").tobytes()
        else:
            raise ValueError(f"unsupported fixture dtype: {dtype}")
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(values.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _fixture_tensors(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def values(*shape: int) -> np.ndarray:
        return rng.normal(0, 0.1, shape).astype("<f4")

    return {
        "confidence_head.proj.bias": values(1),
        "confidence_head.proj.weight": values(1, 64),
        "fc.weight": values(32, 32),
        "hidden_norm.weight": values(32),
        "layers.0.input_layernorm.weight": values(32),
        "layers.0.mlp.down_proj.weight": values(32, 64),
        "layers.0.mlp.gate_proj.weight": values(64, 32),
        "layers.0.mlp.up_proj.weight": values(64, 32),
        "layers.0.post_attention_layernorm.weight": values(32),
        "layers.0.self_attn.k_norm.weight": values(8),
        "layers.0.self_attn.k_proj.weight": values(16, 32),
        "layers.0.self_attn.o_proj.weight": values(32, 32),
        "layers.0.self_attn.q_norm.weight": values(8),
        "layers.0.self_attn.q_proj.weight": values(32, 32),
        "layers.0.self_attn.v_proj.weight": values(16, 32),
        "markov_head.markov_w1.weight": values(64, 32),
        "markov_head.markov_w2.weight": values(64, 32),
        "norm.weight": values(32),
    }


def _fixture_config() -> dict[str, object]:
    return {
        "architectures": ["DSparkDraftModel"],
        "attention_bias": False,
        "block_size": 2,
        "bos_token_id": 1,
        "confidence_head_with_markov": True,
        "dflash_config": {"mask_token_id": 63, "target_layer_ids": [1]},
        "enable_confidence_head": True,
        "eos_token_id": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "hidden_size": 32,
        "intermediate_size": 64,
        "layer_types": ["full_attention"],
        "markov_head_type": "vanilla",
        "markov_rank": 32,
        "max_position_embeddings": 4096,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "num_target_layers": 3,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "rope_type": "yarn",
        },
        "tie_word_embeddings": False,
        "vocab_size": 64,
    }


def _field_value(reader: gguf.GGUFReader, name: str):
    field = reader.fields[name]
    part = field.parts[field.data[0]]
    if field.types[0] == gguf.GGUFValueType.STRING:
        return bytes(part).decode("utf-8")
    return part.tolist()[0]


class ConvertDSparkToGGUFTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "draft"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(json.dumps(_fixture_config()))
        _write_safetensors(self.model_dir / "model.safetensors", _fixture_tensors())

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_converts_q8_with_complete_metadata_and_report(self) -> None:
        output = self.root / "draft-q8_0.gguf"
        report_path = self.root / "conversion-report.json"
        source_hash = hashlib.sha256(
            (self.model_dir / "model.safetensors").read_bytes()
        ).hexdigest()

        report = converter.convert_model(
            converter.ConversionOptions(
                model_dir=self.model_dir,
                output=output,
                report=report_path,
                source_repo="example/draft",
                source_revision="a" * 40,
                target_repo="example/target",
                expected_sha256=source_hash,
                sample_elements=256,
            )
        )

        self.assertTrue(output.is_file())
        self.assertTrue(report_path.is_file())
        self.assertEqual(report["source"]["tensor_count"], 18)
        self.assertEqual(
            report["quantization"]["tensor_counts"],
            {"Q8_0": 10, "F32": 8, "BF16": 0},
        )
        self.assertLess(report["quantization"]["sampled_relative_rmse_max"], 0.01)

        reader = gguf.GGUFReader(output)
        self.assertEqual(_field_value(reader, "general.architecture"), "dflash-draft")
        self.assertEqual(_field_value(reader, "dflash-draft.dflash.n_target_layers"), 1)
        self.assertEqual(_field_value(reader, "dflash-draft.dflash.target.block_count"), 3)
        self.assertEqual(_field_value(reader, "dflash-draft.dflash.dspark.markov_rank"), 32)
        self.assertEqual(_field_value(reader, "general.source.sha256"), source_hash)

        tensors = {tensor.name: tensor for tensor in reader.tensors}
        self.assertEqual(len(tensors), 18)
        self.assertEqual(
            tensors["dflash.fc.weight"].tensor_type,
            gguf.GGMLQuantizationType.Q8_0,
        )
        self.assertEqual(
            tensors["dflash.dspark.confidence.weight"].tensor_type,
            gguf.GGMLQuantizationType.F32,
        )

    def test_unknown_tensor_fails_without_partial_output(self) -> None:
        tensors = _fixture_tensors()
        tensors["unexpected.weight"] = np.ones((32, 32), dtype="<f4")
        _write_safetensors(self.model_dir / "model.safetensors", tensors)
        output = self.root / "should-not-exist.gguf"

        with self.assertRaisesRegex(converter.ConversionError, "unmapped source tensor"):
            converter.convert_model(
                converter.ConversionOptions(model_dir=self.model_dir, output=output)
            )

        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("*.partial.*")), [])

    def test_hash_mismatch_fails_before_output(self) -> None:
        output = self.root / "should-not-exist.gguf"
        with self.assertRaisesRegex(converter.ConversionError, "SHA256 mismatch"):
            converter.convert_model(
                converter.ConversionOptions(
                    model_dir=self.model_dir,
                    output=output,
                    expected_sha256="0" * 64,
                )
            )
        self.assertFalse(output.exists())

    def test_converts_speculators_nested_config_with_checkpoint_vocabulary(self) -> None:
        config = _fixture_config()
        transformer_keys = {
            "attention_bias",
            "bos_token_id",
            "eos_token_id",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "layer_types",
            "max_position_embeddings",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "rms_norm_eps",
            "rope_parameters",
            "sliding_window",
            "tie_word_embeddings",
            "vocab_size",
        }
        nested = {key: config.pop(key) for key in tuple(config) if key in transformer_keys}
        config.pop("num_target_layers")
        config["aux_hidden_state_layer_ids"] = [2]
        config["mask_token_id"] = config.pop("dflash_config")["mask_token_id"]
        config["transformer_layer_config"] = nested
        (self.model_dir / "config.json").write_text(json.dumps(config))

        tensors = _fixture_tensors()
        rng = np.random.default_rng(11)
        tensors["embed_tokens.weight"] = rng.normal(0, 0.1, (64, 32)).astype("<f4")
        tensors["lm_head.weight"] = rng.normal(0, 0.1, (64, 32)).astype("<f4")
        _write_safetensors(self.model_dir / "model.safetensors", tensors)

        output = self.root / "nested-q8_0.gguf"
        report = converter.convert_model(
            converter.ConversionOptions(
                model_dir=self.model_dir,
                output=output,
                sample_elements=256,
            )
        )

        self.assertEqual(report["target"]["block_count"], 3)
        self.assertEqual(
            report["source"]["checkpoint_vocabulary_tensors_retained"],
            ["embed_tokens.weight", "lm_head.weight"],
        )
        reader = gguf.GGUFReader(output)
        self.assertEqual(len(reader.tensors), 20)
        capture_field = reader.fields["dflash-draft.dflash.target_layer_ids"]
        capture_values = capture_field.parts[capture_field.data[0]].tolist()
        self.assertEqual(capture_values, [1])
        tensors_by_name = {t.name: t for t in reader.tensors}
        self.assertEqual(
            tensors_by_name["token_embd.weight"].tensor_type,
            gguf.GGMLQuantizationType.BF16,
        )
        self.assertEqual(
            tensors_by_name["output.weight"].tensor_type,
            gguf.GGMLQuantizationType.BF16,
        )
        restored_head = gguf.quants.dequantize(
            tensors_by_name["output.weight"].data,
            gguf.GGMLQuantizationType.BF16,
        ).reshape(tensors["lm_head.weight"].shape)
        source_header, source_entries = converter.load_safetensors_header(
            self.model_dir / "model.safetensors"
        )
        source_head = converter._read_tensor(
            self.model_dir / "model.safetensors",
            source_header,
            source_entries["lm_head.weight"],
        )
        expected_head = gguf.quants.dequantize(
            gguf.quants.quantize(source_head, gguf.GGMLQuantizationType.BF16),
            gguf.GGMLQuantizationType.BF16,
        ).reshape(source_head.shape)
        np.testing.assert_array_equal(restored_head, expected_head)

    def test_rejects_incomplete_checkpoint_vocabulary_pair(self) -> None:
        tensors = _fixture_tensors()
        tensors["lm_head.weight"] = np.zeros((64, 32), dtype="<f4")
        _write_safetensors(self.model_dir / "model.safetensors", tensors)
        _, entries = converter.load_safetensors_header(self.model_dir / "model.safetensors")
        with self.assertRaisesRegex(converter.ConversionError, "must contain both"):
            converter.validate_tensor_contract(
                entries, converter.load_model_spec(_fixture_config())
            )

    def test_rejects_non_bf16_checkpoint_vocabulary(self) -> None:
        tensors = _fixture_tensors()
        tensors["embed_tokens.weight"] = np.zeros((64, 32), dtype="<f4")
        tensors["lm_head.weight"] = np.zeros((64, 32), dtype="<f4")
        _write_safetensors(
            self.model_dir / "model.safetensors",
            tensors,
            {"lm_head.weight": "F16"},
        )
        header_size, entries = converter.load_safetensors_header(
            self.model_dir / "model.safetensors"
        )
        self.assertGreater(header_size, 0)
        with self.assertRaisesRegex(converter.ConversionError, "must be BF16"):
            converter.validate_tensor_contract(
                entries, converter.load_model_spec(_fixture_config())
            )

    def test_rejects_unsupported_dspark_sampling_contract(self) -> None:
        config = _fixture_config()
        config["sample_from_anchor"] = False
        with self.assertRaisesRegex(converter.ConversionError, "sample_from_anchor"):
            converter.load_model_spec(config)

        config = _fixture_config()
        config["sliding_window_non_causal"] = True
        with self.assertRaisesRegex(converter.ConversionError, "non_causal"):
            converter.load_model_spec(config)

    def test_rejects_malformed_or_empty_explicit_dflash_config(self) -> None:
        config = _fixture_config()
        config["dflash_config"] = []
        with self.assertRaisesRegex(converter.ConversionError, "must be an object"):
            converter.load_model_spec(config)

        config = _fixture_config()
        config["dflash_config"] = {"mask_token_id": 63, "target_layer_ids": []}
        config["aux_hidden_state_layer_ids"] = [2]
        with self.assertRaisesRegex(converter.ConversionError, "non-empty int array"):
            converter.load_model_spec(config)


if __name__ == "__main__":
    unittest.main()
