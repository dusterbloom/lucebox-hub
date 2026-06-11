from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from artifact import write_encoder_artifact
from auto_plan import (
    AUTO_CALIBRATION_SCHEMA,
    AUTO_PLAN_SCHEMA,
    LsaAutoInputs,
    RuntimeObservation,
    adapt_lsa_plan,
    estimate_fit,
    load_calibration_summary,
    load_encoder_summary,
    plan_lsa_auto,
)
from dataset import DatasetMetadata
from model import CompactQwen35Encoder


class AutoPlanTest(unittest.TestCase):
    def _encoder(self, root: Path) -> Path:
        path = root / "encoder"
        model = CompactQwen35Encoder(hidden_size=16, rank=4, kv_heads=2, head_dim=4)
        metadata = DatasetMetadata(
            model_fingerprint="unit-model",
            hidden_size=16,
            kv_heads=2,
            head_dim=4,
        )
        write_encoder_artifact(path, model, metadata)
        return path

    def _calibration(
        self,
        root: Path,
        *,
        recall_10: float = 0.82,
        recall_20: float = 0.93,
        random_20: float = 0.50,
        local_keep: float = 0.03,
        parity: bool = True,
    ) -> Path:
        path = root / "calibration.json"
        payload = {
            "schema": AUTO_CALIBRATION_SCHEMA,
            "max_trained_context_tokens": 131072,
            "metrics": {
                "learned_recall@0.100": recall_10,
                "learned_recall@0.200": recall_20,
                "random_recall@0.200": random_20,
                "recent_recall@0.200": 0.55,
                "local_only_keep": local_keep,
            },
            "gates": {
                "all_chunks_parity": parity,
                "host_cache_validated": False,
            },
            "recommended": {"k": 96},
        }
        path.write_text(json.dumps(payload))
        return path

    def test_missing_encoder_fails_closed(self) -> None:
        plan = plan_lsa_auto(
            LsaAutoInputs(max_context_tokens=65536),
            encoder=None,
            calibration=None,
        )
        self.assertEqual(plan["schema"], AUTO_PLAN_SCHEMA)
        self.assertFalse(plan["enabled"])
        self.assertEqual(plan["mode"], "disabled")
        self.assertIn("missing trained encoder", plan["reasons"][0])

    def test_valid_calibration_selects_oracle_until_host_cache_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = load_encoder_summary(self._encoder(root))
            calibration = load_calibration_summary(self._calibration(root))
            plan = plan_lsa_auto(
                LsaAutoInputs(
                    max_context_tokens=65536,
                    kv_heads=2,
                    head_dim=4,
                    full_attention_layers=16,
                    agentic=True,
                ),
                encoder=encoder,
                calibration=calibration,
            )
            self.assertTrue(plan["enabled"])
            self.assertFalse(plan["runtime_ready"])
            self.assertEqual(plan["mode"], "oracle")
            self.assertEqual(plan["policy"], "topk-stratified")
            self.assertEqual(plan["k"], 96)
            self.assertIn("host-cache", " ".join(plan["warnings"]))

    def test_quality_gate_failure_stays_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = load_encoder_summary(self._encoder(root))
            calibration = load_calibration_summary(
                self._calibration(root, recall_20=0.70)
            )
            plan = plan_lsa_auto(
                LsaAutoInputs(max_context_tokens=65536, kv_heads=2, head_dim=4),
                encoder=encoder,
                calibration=calibration,
            )
            self.assertFalse(plan["enabled"])
            self.assertEqual(plan["mode"], "oracle")
            self.assertIn("quality gates", plan["reasons"][0])

    def test_context_beyond_calibration_stays_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = load_encoder_summary(self._encoder(root))
            calibration = load_calibration_summary(self._calibration(root))
            plan = plan_lsa_auto(
                LsaAutoInputs(max_context_tokens=262144, kv_heads=2, head_dim=4),
                encoder=encoder,
                calibration=calibration,
            )
            self.assertFalse(plan["enabled"])
            self.assertIn("exceeds trained", plan["reasons"][0])

    def test_memory_fit_caps_k(self) -> None:
        inputs = LsaAutoInputs(
            max_context_tokens=131072,
            available_vram_gib=0.1,
            weights_gib=0.0,
            runtime_overhead_gib=0.0,
            draft_gib=0.0,
            cache_type="f16",
            kv_heads=4,
            head_dim=256,
            full_attention_layers=16,
            max_arena_chunks=256,
            min_arena_chunks=1,
        )
        fit = estimate_fit(inputs)
        self.assertLess(fit["fit_by_memory_chunks"], 256)

    def test_adaptation_shrinks_no_context_queries(self) -> None:
        base = {
            "enabled": True,
            "k": 96,
            "interval": 64,
            "policy": "topk-capped",
            "max_arena_chunks": 256,
            "inputs": {"min_arena_chunks": 32, "agentic": False},
        }
        plan = adapt_lsa_plan(
            base,
            RuntimeObservation(no_context_probability=0.90),
        )
        self.assertEqual(plan["k"], 32)
        self.assertEqual(plan["interval"], 128)
        self.assertEqual(plan["policy"], "local-first")
        self.assertTrue(plan["adaptive"]["applied"])

    def test_adaptation_widens_uncertain_long_memory_queries(self) -> None:
        base = {
            "enabled": True,
            "k": 96,
            "interval": 128,
            "policy": "topk-capped",
            "max_arena_chunks": 256,
            "inputs": {"min_arena_chunks": 32, "agentic": True},
        }
        plan = adapt_lsa_plan(
            base,
            RuntimeObservation(
                score_entropy=0.90,
                top_score_margin=0.03,
                selected_churn=0.80,
                long_memory_hint=True,
            ),
        )
        self.assertEqual(plan["k"], 192)
        self.assertEqual(plan["interval"], 64)
        self.assertEqual(plan["policy"], "topk-stratified")
        self.assertTrue(plan["adaptive"]["applied"])

    def test_adaptation_does_not_enable_disabled_plan(self) -> None:
        base = {
            "enabled": False,
            "k": 0,
            "interval": 64,
            "policy": "none",
            "max_arena_chunks": 256,
            "inputs": {"min_arena_chunks": 32},
        }
        plan = adapt_lsa_plan(base, RuntimeObservation(long_memory_hint=True))
        self.assertFalse(plan["enabled"])
        self.assertEqual(plan["k"], 0)
        self.assertFalse(plan["adaptive"]["applied"])


if __name__ == "__main__":
    unittest.main()
