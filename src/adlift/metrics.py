"""Ranking, treatment-control contrasts, and paired fixed-ranking bootstrap."""

from __future__ import annotations
import math
from statistics import NormalDist
from typing import Any
import numpy as np
import pandas as pd


def observed_effect_by_group(
    groups: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, float]:
    """Estimate one treatment effect per pre-treatment segment."""
    counts = cell_counts(groups, treatment, outcome, group_count)
    overall = difference_in_rates(counts.sum(axis=0), 0.95)[4]
    effects = np.full(group_count, overall, dtype=np.float64)
    for group in range(group_count):
        effect = difference_in_rates(counts[group], 0.95)[4]
        if np.isfinite(effect):
            effects[group] = effect
    return effects, float(overall)


def deterministic_tie_key(source_row_id: np.ndarray, seed: int) -> np.ndarray:
    """Create a reproducible ID hash so tied scores are not ordered by source row layout."""
    value = source_row_id.astype(np.uint64, copy=True) + np.uint64(seed)
    value += np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def rank_groups(
    scores: np.ndarray,
    source_row_id: np.ndarray,
    group_count: int,
    tie_break_seed: int = 0,
) -> np.ndarray:
    """Return zero-based groups; group zero contains the highest scores."""
    tie_key = deterministic_tie_key(source_row_id, tie_break_seed)
    order = np.lexsort((tie_key, -scores))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(len(order), dtype=np.int64)
    groups = np.minimum((ranks * group_count) // len(order), group_count - 1)
    return groups.astype(np.int16)


def cell_counts(
    groups: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    group_count: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    state = treatment.astype(np.int64) * 2 + outcome.astype(np.int64)
    index = groups.astype(np.int64) * 4 + state
    return np.bincount(index, weights=weights, minlength=group_count * 4).reshape(group_count, 2, 2)


def difference_in_rates(
    counts: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float, float, float, float, float]:
    n0 = float(counts[0].sum())
    n1 = float(counts[1].sum())
    if n0 <= 0 or n1 <= 0:
        return n0, n1, math.nan, math.nan, math.nan, math.nan, math.nan
    rate0 = float(counts[0, 1] / n0)
    rate1 = float(counts[1, 1] / n1)
    uplift = rate1 - rate0
    variance = rate1 * (1.0 - rate1) / n1 + rate0 * (1.0 - rate0) / n0
    standard_error = math.sqrt(max(variance, 0.0))
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return (
        n0,
        n1,
        rate0,
        rate1,
        uplift,
        uplift - z_value * standard_error,
        uplift + z_value * standard_error,
    )


def group_effect_table(
    outcome_name: str,
    model_name: str,
    scores: np.ndarray,
    source_row_id: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    group_count: int,
    confidence_level: float,
    tie_break_seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    groups = rank_groups(scores, source_row_id, group_count, tie_break_seed)
    counts = cell_counts(groups, treatment, outcome, group_count)
    score_sum = np.bincount(groups, weights=scores, minlength=group_count)
    score_n = np.bincount(groups, minlength=group_count)
    rows = []
    for group in range(group_count):
        n0, n1, rate0, rate1, uplift, ci_lower, ci_upper = difference_in_rates(
            counts[group], confidence_level
        )
        rows.append(
            {
                "outcome_name": outcome_name,
                "model_name": model_name,
                "score_group": group + 1,
                "population_start_pct": 100.0 * group / group_count,
                "population_end_pct": 100.0 * (group + 1) / group_count,
                "group_rows": int(score_n[group]),
                "treatment_rows": int(n1),
                "control_rows": int(n0),
                "treatment_rate": rate1,
                "control_rate": rate0,
                "observed_uplift": uplift,
                "mean_predicted_score": float(score_sum[group] / score_n[group]),
                "incremental_outcomes": float(score_n[group] * uplift),
                "uplift_ci_lower": ci_lower,
                "uplift_ci_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows), groups


def curve_from_counts(counts: np.ndarray, confidence_level: float) -> pd.DataFrame:
    cumulative = np.cumsum(counts, axis=0)
    total_rows = float(counts.sum())
    rows = []
    for index, group_counts in enumerate(cumulative):
        n0, n1, rate0, rate1, uplift, ci_lower, ci_upper = difference_in_rates(
            group_counts, confidence_level
        )
        selected_rows = n0 + n1
        rows.append(
            {
                "target_fraction": (index + 1) / len(counts),
                "selected_population_fraction": selected_rows / total_rows,
                "selected_rows": selected_rows,
                "treatment_rows": n1,
                "control_rows": n0,
                "treatment_rate": rate1,
                "control_rate": rate0,
                "cumulative_uplift": uplift,
                "cumulative_gain": selected_rows * uplift,
                "uplift_ci_lower": ci_lower,
                "uplift_ci_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows)


def cumulative_curve_table(
    outcome_name: str,
    model_name: str,
    scores: np.ndarray,
    source_row_id: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    curve_group_count: int,
    confidence_level: float,
    tie_break_seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    groups = rank_groups(scores, source_row_id, curve_group_count, tie_break_seed)
    counts = cell_counts(groups, treatment, outcome, curve_group_count)
    curve = curve_from_counts(counts, confidence_level)
    curve.insert(0, "model_name", model_name)
    curve.insert(0, "outcome_name", outcome_name)
    return curve, groups


def curve_metrics(curve: pd.DataFrame) -> tuple[float, float, float]:
    x = np.concatenate(([0.0], curve["target_fraction"].to_numpy(dtype=float)))
    y = np.concatenate(([0.0], curve["cumulative_gain"].to_numpy(dtype=float)))
    auuc = float(np.trapezoid(y, x))
    endpoint = float(y[-1])
    qini = auuc - 0.5 * endpoint
    return auuc, qini, endpoint


def calibration_table(
    outcome_name: str,
    model_name: str,
    group_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = group_table.dropna(subset=["mean_predicted_score", "observed_uplift"]).copy()
    x = valid["mean_predicted_score"].to_numpy(dtype=float)
    y = valid["observed_uplift"].to_numpy(dtype=float)
    weights = np.sqrt(valid["group_rows"].to_numpy(dtype=float))
    if len(valid) < 2 or np.allclose(x, x[0]):
        slope, intercept = math.nan, math.nan
        valid["calibrated_predicted_uplift"] = math.nan
    else:
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        valid["calibrated_predicted_uplift"] = intercept + slope * x
    valid = valid[
        [
            "outcome_name",
            "model_name",
            "score_group",
            "group_rows",
            "mean_predicted_score",
            "observed_uplift",
            "calibrated_predicted_uplift",
        ]
    ]
    parameters = {
        "outcome_name": outcome_name,
        "model_name": model_name,
        "intercept": float(intercept),
        "slope": float(slope),
        "ranking_changed": False,
    }
    return valid, parameters


def bootstrap_uncertainty(
    config: dict[str, Any],
    outcome_name: str,
    model_groups: dict[str, np.ndarray],
    treatment: np.ndarray,
    outcome: np.ndarray,
    point_metrics: pd.DataFrame,
    point_policy: pd.DataFrame,
) -> pd.DataFrame:
    model_names = list(model_groups)
    group_count = int(round(1.0 / float(config["cumulative_step"])))
    key = np.zeros(len(treatment), dtype=np.int64)
    for model_name in model_names:
        key = key * group_count + model_groups[model_name].astype(np.int64)
    key = key * 4 + treatment.astype(np.int64) * 2 + outcome.astype(np.int64)
    occupied, empirical_counts = np.unique(key, return_counts=True)
    probabilities = empirical_counts / empirical_counts.sum()
    state = occupied % 4
    remainder = occupied // 4
    occupied_groups = np.empty((len(occupied), len(model_names)), dtype=np.int16)
    for model_index in range(len(model_names) - 1, -1, -1):
        occupied_groups[:, model_index] = remainder % group_count
        remainder //= group_count

    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    replicates = int(config["bootstrap_replicates"])
    raw_rows: list[dict[str, Any]] = []
    main_rates = [float(value) for value in config["main_decision_rates"]]
    response_name = config["response_model"]
    for replicate in range(replicates):
        draw = rng.multinomial(len(treatment), probabilities)
        replicate_values: dict[str, dict[str, float]] = {}
        for model_index, model_name in enumerate(model_names):
            index = occupied_groups[:, model_index].astype(np.int64) * 4 + state
            counts = np.bincount(index, weights=draw, minlength=group_count * 4).reshape(
                group_count, 2, 2
            )
            curve = curve_from_counts(counts, float(config["confidence_level"]))
            auuc, qini, _ = curve_metrics(curve)
            values = {"auuc": auuc, "qini": qini}
            for rate in main_rates:
                position = int(round(rate * group_count)) - 1
                values[f"gain_at_{int(rate * 100)}pct"] = float(
                    curve.iloc[position]["cumulative_gain"]
                )
            replicate_values[model_name] = values
            for metric, value in values.items():
                raw_rows.append(
                    {
                        "replicate": replicate,
                        "model_name": model_name,
                        "metric": metric,
                        "value": value,
                    }
                )
        response_values = replicate_values[response_name]
        for model_name in model_names:
            if model_name == response_name:
                continue
            for metric, value in replicate_values[model_name].items():
                raw_rows.append(
                    {
                        "replicate": replicate,
                        "model_name": model_name,
                        "metric": f"{metric}_minus_response",
                        "value": value - response_values[metric],
                    }
                )
        uplift_order = list(config["uplift_models"])
        for later_index in range(1, len(uplift_order)):
            later_model = uplift_order[later_index]
            for earlier_model in uplift_order[:later_index]:
                for metric in ("auuc", "qini"):
                    raw_rows.append(
                        {
                            "replicate": replicate,
                            "model_name": later_model,
                            "metric": f"{metric}_minus_{earlier_model}",
                            "value": (
                                replicate_values[later_model][metric]
                                - replicate_values[earlier_model][metric]
                            ),
                        }
                    )

    raw = pd.DataFrame(raw_rows)
    alpha = (1.0 - float(config["confidence_level"])) / 2.0
    point_map: dict[tuple[str, str], float] = {}
    for row in point_metrics.itertuples(index=False):
        point_map[(row.model_name, "auuc")] = float(row.auuc)
        point_map[(row.model_name, "qini")] = float(row.qini)
    for row in point_policy.itertuples(index=False):
        rate = int(round(float(row.target_fraction) * 100))
        if float(row.target_fraction) in [float(x) for x in config["main_decision_rates"]]:
            point_map[(row.model_name, f"gain_at_{rate}pct")] = float(row.cumulative_gain)
    response_metrics = point_metrics.set_index("model_name")
    response_policy = point_policy[point_policy["model_name"] == response_name].set_index(
        "target_fraction"
    )
    for model_name in model_names:
        if model_name == response_name:
            continue
        point_map[(model_name, "auuc_minus_response")] = float(
            response_metrics.loc[model_name, "auuc"] - response_metrics.loc[response_name, "auuc"]
        )
        point_map[(model_name, "qini_minus_response")] = float(
            response_metrics.loc[model_name, "qini"] - response_metrics.loc[response_name, "qini"]
        )
        model_policy = point_policy[point_policy["model_name"] == model_name].set_index(
            "target_fraction"
        )
        for rate in [float(x) for x in config["main_decision_rates"]]:
            label = f"gain_at_{int(rate * 100)}pct_minus_response"
            point_map[(model_name, label)] = float(
                model_policy.loc[rate, "cumulative_gain"]
                - response_policy.loc[rate, "cumulative_gain"]
            )
    uplift_order = list(config["uplift_models"])
    for later_index in range(1, len(uplift_order)):
        later_model = uplift_order[later_index]
        for earlier_model in uplift_order[:later_index]:
            for metric in ("auuc", "qini"):
                point_map[(later_model, f"{metric}_minus_{earlier_model}")] = float(
                    response_metrics.loc[later_model, metric]
                    - response_metrics.loc[earlier_model, metric]
                )

    rows = []
    for (model_name, metric), values in raw.groupby(["model_name", "metric"])["value"]:
        rows.append(
            {
                "outcome_name": outcome_name,
                "model_name": model_name,
                "metric": metric,
                "estimate": point_map.get((model_name, metric), math.nan),
                "ci_lower": float(values.quantile(alpha)),
                "ci_upper": float(values.quantile(1.0 - alpha)),
                "bootstrap_replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def top_fraction_mask(
    scores: np.ndarray,
    source_row_id: np.ndarray,
    fraction: float,
    tie_break_seed: int,
) -> np.ndarray:
    tie_key = deterministic_tie_key(source_row_id, tie_break_seed)
    order = np.lexsort((tie_key, -scores))
    selected = max(1, int(round(len(order) * fraction)))
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:selected]] = True
    return mask


def selected_policy_effect(
    mask: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    confidence_level: float,
) -> dict[str, float]:
    counts = cell_counts(
        np.zeros(int(mask.sum()), dtype=np.int16),
        treatment[mask],
        outcome[mask],
        1,
    )[0]
    n0, n1, rate0, rate1, uplift, lower, upper = difference_in_rates(counts, confidence_level)
    selected_rows = int(mask.sum())
    return {
        "selected_rows": selected_rows,
        "control_rows": n0,
        "treatment_rows": n1,
        "control_rate": rate0,
        "treatment_rate": rate1,
        "uplift": uplift,
        "gain": selected_rows * uplift,
        "gain_ci_lower": selected_rows * lower,
        "gain_ci_upper": selected_rows * upper,
    }
