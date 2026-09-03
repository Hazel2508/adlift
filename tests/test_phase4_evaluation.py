"""Synthetic tests for the Phase 4 evaluation contract."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


from adlift import evaluation as phase4


def synthetic_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = 20_000
    source_row_id = np.arange(rows, dtype=np.uint64)
    treatment = (source_row_id % 2).astype(np.uint8)
    high_uplift = source_row_id < rows // 2
    outcome = np.zeros(rows, dtype=np.uint8)
    # All groups have a 10% control rate. Treatment is 30% in the high-uplift
    # half and 10% in the low-uplift half.
    outcome[(treatment == 0) & (source_row_id % 20 < 2)] = 1
    outcome[(treatment == 1) & high_uplift & (source_row_id % 20 < 6)] = 1
    outcome[(treatment == 1) & ~high_uplift & (source_row_id % 20 < 2)] = 1
    score = np.where(high_uplift, 0.20, 0.0).astype(np.float32)
    return source_row_id, treatment, outcome, score


class Phase4EvaluationTests(unittest.TestCase):
    def test_segment_effect_baseline_recovers_group_heterogeneity(self) -> None:
        groups = np.repeat(np.array([0, 1], dtype=np.int16), 10_000)
        treatment = (np.arange(len(groups)) % 2).astype(np.uint8)
        outcome = np.zeros(len(groups), dtype=np.uint8)
        outcome[(groups == 0) & (treatment == 1) & (np.arange(len(groups)) % 10 < 4)] = 1
        outcome[(groups == 0) & (treatment == 0) & (np.arange(len(groups)) % 10 < 2)] = 1
        outcome[(groups == 1) & (np.arange(len(groups)) % 10 < 2)] = 1
        effects, overall = phase4.observed_effect_by_group(groups, treatment, outcome, 2)
        self.assertGreater(effects[0], effects[1])
        self.assertGreater(overall, 0.0)

    def test_higher_scores_create_positive_qini(self) -> None:
        source_row_id, treatment, outcome, score = synthetic_fixture()
        curve, _ = phase4.cumulative_curve_table(
            "visit",
            "s_learner",
            score,
            source_row_id,
            treatment,
            outcome,
            20,
            0.95,
        )
        _, qini, _ = phase4.curve_metrics(curve)
        self.assertGreater(qini, 0.0)

    def test_top_100_percent_equals_n_times_observed_ate(self) -> None:
        source_row_id, treatment, outcome, score = synthetic_fixture()
        curve, _ = phase4.cumulative_curve_table(
            "visit",
            "t_learner",
            score,
            source_row_id,
            treatment,
            outcome,
            20,
            0.95,
        )
        counts = phase4.cell_counts(np.zeros(len(outcome), dtype=np.int16), treatment, outcome, 1)[
            0
        ]
        _, _, _, _, observed_ate, _, _ = phase4.difference_in_rates(counts, 0.95)
        self.assertAlmostEqual(
            float(curve.iloc[-1]["cumulative_gain"]),
            len(outcome) * observed_ate,
            places=10,
        )

    def test_group_one_contains_highest_scores_with_deterministic_ties(self) -> None:
        ids = np.array([5, 2, 9, 1], dtype=np.uint64)
        scores = np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float32)
        groups = phase4.rank_groups(scores, ids, 2)
        self.assertEqual(groups[1], 0)
        self.assertEqual(groups[0], 0)
        self.assertEqual(groups[3], 1)
        self.assertEqual(groups[2], 1)

    def test_top_fraction_mask_selects_exact_requested_count(self) -> None:
        ids = np.arange(100, dtype=np.uint64)
        scores = np.repeat(np.array([1.0, 0.0]), 50)
        mask = phase4.top_fraction_mask(scores, ids, 0.30, 17)
        self.assertEqual(int(mask.sum()), 30)
        self.assertTrue((scores[mask] == 1.0).all())

    def test_linear_calibration_records_unchanged_ranking(self) -> None:
        table = pd.DataFrame(
            {
                "outcome_name": ["visit"] * 4,
                "model_name": ["s_learner"] * 4,
                "score_group": [1, 2, 3, 4],
                "group_rows": [1000] * 4,
                "mean_predicted_score": [0.4, 0.3, 0.2, 0.1],
                "observed_uplift": [0.20, 0.15, 0.10, 0.05],
            }
        )
        calibrated, parameters = phase4.calibration_table("visit", "s_learner", table)
        self.assertFalse(parameters["ranking_changed"])
        self.assertGreater(parameters["slope"], 0)
        self.assertTrue(calibrated["calibrated_predicted_uplift"].is_monotonic_decreasing)

    def test_paired_bootstrap_returns_model_differences(self) -> None:
        source_row_id, treatment, outcome, score = synthetic_fixture()
        model_scores = {
            "s_learner": score,
            "t_learner": score * 0.9,
            "x_learner": score * 1.1,
            "response_propensity": score[::-1].copy(),
        }
        curves = []
        model_groups = {}
        for model_name, values in model_scores.items():
            curve, groups = phase4.cumulative_curve_table(
                "visit",
                model_name,
                values,
                source_row_id,
                treatment,
                outcome,
                20,
                0.95,
            )
            curves.append(curve)
            model_groups[model_name] = groups
        cumulative = pd.concat(curves, ignore_index=True)
        metrics = []
        for model_name, table in cumulative.groupby("model_name"):
            auuc, qini, endpoint = phase4.curve_metrics(table)
            metrics.append(
                {
                    "outcome_name": "visit",
                    "model_name": model_name,
                    "auuc": auuc,
                    "qini": qini,
                    "endpoint_gain": endpoint,
                }
            )
        policy = cumulative[cumulative["target_fraction"].isin([0.10, 0.20, 0.30])].copy()
        config = {
            "cumulative_step": 0.05,
            "confidence_level": 0.95,
            "bootstrap_seed": 7,
            "bootstrap_replicates": 20,
            "main_decision_rates": [0.10, 0.20, 0.30],
            "response_model": "response_propensity",
            "uplift_models": ["s_learner", "t_learner", "x_learner"],
        }
        uncertainty = phase4.bootstrap_uncertainty(
            config,
            "visit",
            model_groups,
            treatment,
            outcome,
            pd.DataFrame(metrics),
            policy,
        )
        self.assertIn("qini_minus_response", set(uncertainty["metric"]))
        self.assertIn("qini_minus_s_learner", set(uncertainty["metric"]))
        self.assertTrue((uncertainty["bootstrap_replicates"] == 20).all())


if __name__ == "__main__":
    unittest.main()
