"""Deterministic guardrails for the boundary-correct Phase 3 pipeline."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "phase3_uplift_pipeline.py"
SPEC = importlib.util.spec_from_file_location("phase3_uplift_pipeline", MODULE_PATH)
phase3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = phase3
SPEC.loader.exec_module(phase3)


class Phase3GuardrailTests(unittest.TestCase):
    def test_forbidden_feature_guard(self) -> None:
        canonical = [f"f{i}" for i in range(12)]
        phase3.check_model_matrix(canonical, "visit")
        with self.assertRaises(AssertionError):
            phase3.check_model_matrix(canonical + ["exposure"], "visit")

    def test_x_learner_uses_taught_single_pseudo_outcome(self) -> None:
        treatment = np.array([1, 0, 1, 0], dtype=np.uint8)
        outcome = np.array([1, 1, 0, 0], dtype=np.uint8)
        mu0_hat = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        mu1_hat = np.array([0.7, 0.8, 0.6, 0.4], dtype=np.float32)
        actual = phase3.construct_x_pseudo_outcome(
            treatment, outcome, mu0_hat, mu1_hat
        )
        expected = np.array([0.8, -0.2, -0.4, 0.4], dtype=np.float32)
        np.testing.assert_allclose(actual, expected)

    def test_prediction_artifact_uses_null_for_test_outcome(self) -> None:
        data = phase3.PartitionData(
            source_row_id=np.arange(4, dtype=np.uint64),
            features=np.zeros((4, 12), dtype=np.float32),
            treatment=np.array([0, 1, 0, 1], dtype=np.uint8),
            outcome=None,
            split="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.parquet"
            phase3.write_prediction_artifact(
                output,
                data,
                "visit",
                "s_learner",
                "config-hash",
                "source-hash",
                np.full(4, 0.01, dtype=np.float32),
            )
            table = pq.read_table(output)
            self.assertEqual(table.num_rows, 4)
            self.assertEqual(table["observed_outcome"].null_count, 4)
            self.assertEqual(table["uplift_hat"].null_count, 0)
            self.assertEqual(
                table.schema.names,
                [
                    "source_row_id",
                    "split",
                    "outcome_name",
                    "model_name",
                    "treatment",
                    "observed_outcome",
                    "uplift_hat",
                    "model_config_hash",
                    "source_file_sha256",
                ],
            )

    def test_phase3_scope_contains_only_three_learners(self) -> None:
        self.assertEqual(
            phase3.LEARNERS,
            ("s_learner", "t_learner", "x_learner"),
        )


if __name__ == "__main__":
    unittest.main()
