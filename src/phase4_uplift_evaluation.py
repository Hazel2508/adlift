"""Phase 4 uplift evaluation and targeting strategy.

The workflow has two explicit stages:

``prepare``
    Validate the frozen Phase 3 handoff, freeze the evaluation specification,
    fit the response-propensity baseline on train/validation, and score test
    features without reading test outcomes.

``evaluate``
    Join the frozen scores to test outcomes and evaluate ranking quality using
    score groups, cumulative gain, AUUC, Qini, uplift calibration, paired
    bootstrap uncertainty, and Top 10/20/30 targeting comparisons.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "phase4_evaluation.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import phase3_uplift_pipeline as p3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name, help_text in (
        ("prepare", "Freeze evaluation rules and prepare response baselines."),
        ("evaluate", "Evaluate frozen rankings on test outcomes."),
        ("all", "Run prepare followed by evaluate."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument(
            "--outcome",
            choices=["visit", "conversion", "both"],
            default="both",
        )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unique_scalar(column: pa.ChunkedArray, label: str) -> Any:
    values = pc.unique(column.combine_chunks()).to_pylist()
    if len(values) != 1:
        raise AssertionError(f"{label} must contain exactly one value; found {values[:5]}.")
    return values[0]


def phase4_spec_hash(config: dict[str, Any]) -> str:
    frozen_keys = {
        "outcomes": config["outcomes"],
        "uplift_models": config["uplift_models"],
        "constant_model": config["constant_model"],
        "segment_model": config["segment_model"],
        "response_model": config["response_model"],
        "segment_baseline": config["segment_baseline"],
        "stability": config["stability"],
        "score_groups": config["score_groups"],
        "cumulative_step": config["cumulative_step"],
        "report_rates": config["report_rates"],
        "main_decision_rates": config["main_decision_rates"],
        "confidence_level": config["confidence_level"],
        "bootstrap_replicates": config["bootstrap_replicates"],
        "bootstrap_seed": config["bootstrap_seed"],
        "tie_break_seed": config["tie_break_seed"],
        "analysis_status": config["analysis_status"],
        "evaluation_rule_version": config["evaluation_rule_version"],
    }
    return canonical_hash(frozen_keys)


def validate_prediction_file(
    path: Path,
    outcome: str,
    model_name: str,
    expected_source_hash: str,
    expected_run_hash: str,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pq.read_table(
        path,
        columns=[
            "source_row_id",
            "split",
            "outcome_name",
            "model_name",
            "observed_outcome",
            "uplift_hat",
            "model_config_hash",
            "source_file_sha256",
        ],
    )
    if unique_scalar(table["split"], "split") != "test":
        raise AssertionError(f"{path} is not a test prediction artifact.")
    if unique_scalar(table["outcome_name"], "outcome_name") != outcome:
        raise AssertionError(f"Outcome mismatch in {path}.")
    if unique_scalar(table["model_name"], "model_name") != model_name:
        raise AssertionError(f"Model mismatch in {path}.")
    if table["observed_outcome"].null_count != table.num_rows:
        raise AssertionError(f"Phase 3 test outcomes leaked into {path}.")
    scores = table["uplift_hat"].combine_chunks().to_numpy(zero_copy_only=False)
    if not np.isfinite(scores).all():
        raise AssertionError(f"Non-finite uplift scores in {path}.")
    source_hash = unique_scalar(table["source_file_sha256"], "source_file_sha256")
    if source_hash != expected_source_hash:
        raise AssertionError(f"Source hash mismatch in {path}.")
    artifact_hash = unique_scalar(table["model_config_hash"], "model_config_hash")
    expected_artifact_hash = canonical_hash(
        {"run_config_hash": expected_run_hash, "model_name": model_name}
    )
    if artifact_hash != expected_artifact_hash:
        raise AssertionError(f"Model artifact hash mismatch in {path}.")
    ids = table["source_row_id"].combine_chunks().to_numpy(zero_copy_only=False)
    if len(np.unique(ids)) != len(ids):
        raise AssertionError(f"Duplicate source_row_id values in {path}.")
    return ids


def validate_phase3_handoff(config: dict[str, Any], outcomes: list[str]) -> dict[str, Any]:
    phase3_config = read_json(resolve_path(config["phase3_config"]))
    manifest = read_json(resolve_path(phase3_config["input"]["manifest_file"]))
    expected_source_hash = manifest["raw_file_sha256"]
    reference_ids: np.ndarray | None = None
    summary: dict[str, Any] = {}

    for outcome in outcomes:
        model_directory = PROJECT_ROOT / "data" / "processed" / "phase3" / "models" / outcome
        metadata = read_json(model_directory / "metadata.json")
        frozen = read_json(model_directory / "FROZEN.json")
        if set(metadata.get("lineage_spec", {}).get("scope", [])) != set(config["uplift_models"]):
            raise AssertionError(f"{outcome} does not use the boundary-correct S/T/X scope.")
        if frozen["model_config_hash"] != metadata["model_config_hash"]:
            raise AssertionError(f"{outcome} frozen and metadata hashes disagree.")
        if frozen["source_file_sha256"] != expected_source_hash:
            raise AssertionError(f"{outcome} frozen source hash mismatch.")
        if frozen.get("test_outcomes_used") is not False:
            raise AssertionError(f"{outcome} freeze marker indicates test-outcome use.")

        outcome_ids: np.ndarray | None = None
        for model_name in config["uplift_models"]:
            path = (
                PROJECT_ROOT
                / "data"
                / "processed"
                / "phase3"
                / "predictions"
                / "test"
                / outcome
                / f"{model_name}.parquet"
            )
            ids = validate_prediction_file(
                path,
                outcome,
                model_name,
                expected_source_hash,
                metadata["model_config_hash"],
            )
            if outcome_ids is None:
                outcome_ids = ids
            elif not np.array_equal(outcome_ids, ids):
                raise AssertionError(f"Phase 3 {outcome} models do not score identical rows/order.")
        assert outcome_ids is not None
        if reference_ids is None:
            reference_ids = outcome_ids
        elif not np.array_equal(reference_ids, outcome_ids):
            raise AssertionError("Visit and conversion do not score identical test users/order.")
        summary[outcome] = {
            "rows": int(len(outcome_ids)),
            "phase3_run_hash": metadata["model_config_hash"],
            "source_file_sha256": expected_source_hash,
        }
    return summary


def fit_response_baseline(config: dict[str, Any], outcome: str) -> dict[str, Any]:
    phase3_config = read_json(resolve_path(config["phase3_config"]))
    phase3_file = resolve_path(config["phase3_input"])
    manifest = read_json(resolve_path(phase3_config["input"]["manifest_file"]))
    feature_names = phase3_config["features"]
    p3.check_model_matrix(feature_names, outcome)
    parameters = p3.lightgbm_parameters(phase3_config, outcome, "binary")
    response_hash = canonical_hash(
        {
            "outcome": outcome,
            "features": feature_names,
            "parameters": parameters,
            "source_file_sha256": manifest["raw_file_sha256"],
        }
    )
    output_root = resolve_path(config["output_root"])
    model_directory = output_root / "baselines" / outcome
    metadata_path = model_directory / "metadata.json"
    prediction_path = model_directory / "response_propensity.parquet"
    if metadata_path.exists() and prediction_path.exists():
        existing = read_json(metadata_path)
        if (
            existing.get("model_config_hash") == response_hash
            and existing.get("source_file_sha256") == manifest["raw_file_sha256"]
            and existing.get("test_outcomes_used") is False
            and pq.ParquetFile(prediction_path).metadata.num_rows == existing.get("rows")
        ):
            return {
                "rows": int(existing["rows"]),
                "model_config_hash": response_hash,
                "reused": True,
            }
    train = p3.load_partition(phase3_file, feature_names, "train", outcome)
    validation = p3.load_partition(phase3_file, feature_names, "validation", outcome)
    test = p3.load_partition(phase3_file, feature_names, "test", outcome=None)
    if test.outcome is not None:
        raise AssertionError("Response-baseline preparation must not read test outcomes.")

    lgb = p3.require_lightgbm()
    model = p3.fit_binary_model(
        lgb,
        train.features,
        train.outcome,
        validation.features,
        validation.outcome,
        parameters,
        feature_names,
    )
    score = model.predict(test.features, num_iteration=p3.prediction_iterations(model)).astype(
        np.float32
    )
    if not np.isfinite(score).all():
        raise AssertionError(f"Non-finite {outcome} response scores.")

    model_directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_directory / "response_propensity.txt"))
    table = pa.table(
        {
            "source_row_id": pa.array(test.source_row_id, type=pa.uint64()),
            "outcome_name": p3.constant_dictionary(outcome, len(test.source_row_id)),
            "model_name": p3.constant_dictionary(
                config["response_model"], len(test.source_row_id)
            ),
            "response_score": pa.array(score, type=pa.float32()),
            "model_config_hash": p3.constant_dictionary(response_hash, len(score)),
            "source_file_sha256": p3.constant_dictionary(
                manifest["raw_file_sha256"], len(score)
            ),
        }
    )
    pq.write_table(table, prediction_path, compression="zstd", row_group_size=100_000)
    write_json(
        model_directory / "metadata.json",
        {
            "outcome": outcome,
            "model_config_hash": response_hash,
            "source_file_sha256": manifest["raw_file_sha256"],
            "test_outcomes_used": False,
            "rows": int(len(score)),
        },
    )
    del train, validation, test, model, score
    gc.collect()
    return {"rows": table.num_rows, "model_config_hash": response_hash}


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


def quantile_edges(values: np.ndarray, requested_bins: int) -> np.ndarray:
    probabilities = np.arange(1, requested_bins, dtype=float) / requested_bins
    if len(probabilities) == 0:
        return np.empty(0, dtype=np.float32)
    edges = np.unique(np.quantile(values.astype(np.float64), probabilities))
    return edges.astype(np.float32)


def assign_segments(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def write_baseline_scores(
    path: Path,
    source_row_id: np.ndarray,
    outcome: str,
    model_name: str,
    scores: np.ndarray,
    model_hash: str,
    source_hash: str,
) -> None:
    table = pa.table(
        {
            "source_row_id": pa.array(source_row_id, type=pa.uint64()),
            "outcome_name": p3.constant_dictionary(outcome, len(source_row_id)),
            "model_name": p3.constant_dictionary(model_name, len(source_row_id)),
            "uplift_hat": pa.array(np.asarray(scores, dtype=np.float32), type=pa.float32()),
            "model_config_hash": p3.constant_dictionary(model_hash, len(source_row_id)),
            "source_file_sha256": p3.constant_dictionary(source_hash, len(source_row_id)),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", row_group_size=100_000)


def fit_structural_baselines(config: dict[str, Any], outcome: str) -> dict[str, Any]:
    """Fit constant-effect and one-feature segment baselines without test outcomes."""
    phase3_config = read_json(resolve_path(config["phase3_config"]))
    manifest = read_json(resolve_path(phase3_config["input"]["manifest_file"]))
    source_hash = manifest["raw_file_sha256"]
    baseline_spec = {
        "outcome": outcome,
        "constant_model": config["constant_model"],
        "segment_model": config["segment_model"],
        "segment_baseline": config["segment_baseline"],
        "tie_break_seed": config["tie_break_seed"],
        "source_file_sha256": source_hash,
    }
    model_hash = canonical_hash(baseline_spec)
    model_directory = resolve_path(config["output_root"]) / "baselines" / outcome
    metadata_path = model_directory / "structural_metadata.json"
    constant_path = model_directory / f"{config['constant_model']}.parquet"
    segment_path = model_directory / f"{config['segment_model']}.parquet"
    if metadata_path.exists() and constant_path.exists() and segment_path.exists():
        existing = read_json(metadata_path)
        expected_rows = existing.get("rows")
        if (
            existing.get("model_config_hash") == model_hash
            and existing.get("source_file_sha256") == source_hash
            and existing.get("test_outcomes_used") is False
            and pq.ParquetFile(constant_path).metadata.num_rows == expected_rows
            and pq.ParquetFile(segment_path).metadata.num_rows == expected_rows
        ):
            return {**existing, "reused": True}

    features = list(config["segment_baseline"]["candidate_features"])
    phase3_file = resolve_path(config["phase3_input"])
    train = p3.load_partition(phase3_file, features, "train", outcome)
    validation = p3.load_partition(phase3_file, features, "validation", outcome)
    test = p3.load_partition(phase3_file, features, "test", outcome=None)
    if test.outcome is not None:
        raise AssertionError("Structural-baseline preparation must not read test outcomes.")
    assert train.outcome is not None and validation.outcome is not None

    candidate_rows: list[dict[str, Any]] = []
    candidate_objects: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    selection_groups = int(config["segment_baseline"]["selection_curve_groups"])
    for feature_index, feature_name in enumerate(features):
        train_values = train.features[:, feature_index]
        validation_values = validation.features[:, feature_index]
        for requested_bins in config["segment_baseline"]["candidate_bins"]:
            edges = quantile_edges(train_values, int(requested_bins))
            actual_bins = len(edges) + 1
            train_groups = assign_segments(train_values, edges)
            train_effects, train_ate = observed_effect_by_group(
                train_groups, train.treatment, train.outcome, actual_bins
            )
            validation_scores = train_effects[assign_segments(validation_values, edges)]
            curve, _ = cumulative_curve_table(
                outcome,
                config["segment_model"],
                validation_scores,
                validation.source_row_id,
                validation.treatment,
                validation.outcome,
                selection_groups,
                float(config["confidence_level"]),
                int(config["tie_break_seed"]),
            )
            auuc, qini, _ = curve_metrics(curve)
            key = (feature_name, int(requested_bins))
            candidate_objects[key] = (edges, train_effects)
            candidate_rows.append(
                {
                    "outcome_name": outcome,
                    "feature": feature_name,
                    "requested_bins": int(requested_bins),
                    "actual_bins": actual_bins,
                    "train_ate": train_ate,
                    "validation_auuc": auuc,
                    "validation_qini": qini,
                }
            )
    selection = pd.DataFrame(candidate_rows).sort_values(
        ["validation_qini", "actual_bins", "feature"],
        ascending=[False, True, True],
    )
    best = selection.iloc[0]
    best_key = (str(best["feature"]), int(best["requested_bins"]))
    edges, _ = candidate_objects[best_key]
    feature_index = features.index(best_key[0])
    combined_values = np.concatenate(
        (train.features[:, feature_index], validation.features[:, feature_index])
    )
    combined_treatment = np.concatenate((train.treatment, validation.treatment))
    combined_outcome = np.concatenate((train.outcome, validation.outcome))
    combined_groups = assign_segments(combined_values, edges)
    segment_effects, constant_ate = observed_effect_by_group(
        combined_groups, combined_treatment, combined_outcome, len(edges) + 1
    )
    test_groups = assign_segments(test.features[:, feature_index], edges)
    segment_scores = segment_effects[test_groups]
    constant_scores = np.full(len(test.source_row_id), constant_ate, dtype=np.float32)
    write_baseline_scores(
        constant_path,
        test.source_row_id,
        outcome,
        config["constant_model"],
        constant_scores,
        model_hash,
        source_hash,
    )
    write_baseline_scores(
        segment_path,
        test.source_row_id,
        outcome,
        config["segment_model"],
        segment_scores,
        model_hash,
        source_hash,
    )
    table_directory = resolve_path(config["table_directory"])
    table_directory.mkdir(parents=True, exist_ok=True)
    selection.to_csv(table_directory / f"{outcome}_segment_baseline_selection.csv", index=False)
    metadata = {
        "outcome": outcome,
        "model_config_hash": model_hash,
        "source_file_sha256": source_hash,
        "test_outcomes_used": False,
        "rows": int(len(test.source_row_id)),
        "constant_ate_train_validation": constant_ate,
        "selected_feature": best_key[0],
        "requested_bins": best_key[1],
        "actual_bins": int(len(edges) + 1),
        "bin_edges": edges.tolist(),
        "segment_effects": segment_effects.tolist(),
        "validation_qini": float(best["validation_qini"]),
    }
    write_json(metadata_path, metadata)
    del train, validation, test, combined_values, combined_treatment, combined_outcome
    gc.collect()
    return metadata


def phase3_stability(config: dict[str, Any], outcome: str) -> dict[str, Any]:
    """Retrain the current Phase 3 implementation at alternate seeds on train/validation."""
    phase3_config = read_json(resolve_path(config["phase3_config"]))
    model_metadata = read_json(
        PROJECT_ROOT / "data" / "processed" / "phase3" / "models" / outcome / "metadata.json"
    )
    stability_spec = {
        "outcome": outcome,
        "reference_run_hash": model_metadata["model_config_hash"],
        "comparison_seeds": config["stability"]["comparison_seeds"],
        "top_fraction": config["stability"]["top_fraction"],
        "phase3_scope": config["uplift_models"],
    }
    stability_hash = canonical_hash(stability_spec)
    output_root = resolve_path(config["output_root"]) / "stability" / outcome
    metadata_path = output_root / "metadata.json"
    table_path = resolve_path(config["table_directory"]) / f"{outcome}_phase3_stability.csv"
    if metadata_path.exists() and table_path.exists():
        existing = read_json(metadata_path)
        if existing.get("stability_spec_hash") == stability_hash:
            return {**existing, "reused": True}

    validation_path = resolve_path(config["phase3_input"])
    train = p3.load_partition(validation_path, phase3_config["features"], "train", outcome)
    validation = p3.load_partition(
        validation_path, phase3_config["features"], "validation", outcome
    )
    reference_scores: dict[str, np.ndarray] = {}
    for model_name in config["uplift_models"]:
        path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "phase3"
            / "predictions"
            / "validation"
            / outcome
            / f"{model_name}.parquet"
        )
        table = pq.read_table(path, columns=["source_row_id", "uplift_hat", "model_config_hash"])
        ids = table["source_row_id"].combine_chunks().to_numpy(zero_copy_only=False)
        if not np.array_equal(ids, validation.source_row_id):
            raise AssertionError(f"Validation IDs changed for {outcome} {model_name} stability.")
        expected_hash = canonical_hash(
            {"run_config_hash": model_metadata["model_config_hash"], "model_name": model_name}
        )
        if unique_scalar(table["model_config_hash"], "model_config_hash") != expected_hash:
            raise AssertionError(f"Reference validation hash changed for {outcome} {model_name}.")
        reference_scores[model_name] = table["uplift_hat"].combine_chunks().to_numpy(
            zero_copy_only=False
        )

    from scipy.stats import rankdata

    rows: list[dict[str, Any]] = []
    group_count = int(round(1.0 / float(config["stability"]["top_fraction"])))
    for comparison_seed in config["stability"]["comparison_seeds"]:
        comparison_config = copy.deepcopy(phase3_config)
        comparison_config["seed"] = int(comparison_seed)
        result = p3.train_models(comparison_config, outcome, train, validation)
        for model_name in config["uplift_models"]:
            reference = reference_scores[model_name]
            comparison = result["predictions"][model_name]
            reference_rank = rankdata(reference, method="average")
            comparison_rank = rankdata(comparison, method="average")
            correlation = float(np.corrcoef(reference_rank, comparison_rank)[0, 1])
            reference_top = rank_groups(
                reference,
                validation.source_row_id,
                group_count,
                int(config["tie_break_seed"]),
            ) == 0
            comparison_top = rank_groups(
                comparison,
                validation.source_row_id,
                group_count,
                int(config["tie_break_seed"]),
            ) == 0
            overlap = float(np.logical_and(reference_top, comparison_top).sum() / reference_top.sum())
            rows.append(
                {
                    "outcome_name": outcome,
                    "model_name": model_name,
                    "reference_seed": int(phase3_config["seed"]),
                    "comparison_seed": int(comparison_seed),
                    "spearman_rank_correlation": correlation,
                    "top_fraction": float(config["stability"]["top_fraction"]),
                    "top_set_overlap": overlap,
                    "reference_mean_uplift": float(reference.mean()),
                    "comparison_mean_uplift": float(comparison.mean()),
                    "comparison_runtime_seconds": float(result["runtime_seconds"]),
                }
            )
            del reference_rank, comparison_rank
        del result
        gc.collect()
    stability = pd.DataFrame(rows)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    stability.to_csv(table_path, index=False)
    metadata = {
        "outcome": outcome,
        "stability_spec_hash": stability_hash,
        "reference_run_hash": model_metadata["model_config_hash"],
        "comparison_seeds": config["stability"]["comparison_seeds"],
        "rows": int(len(stability)),
        "test_outcomes_used": False,
    }
    write_json(metadata_path, metadata)
    del train, validation, reference_scores
    gc.collect()
    return metadata


def prepare(config: dict[str, Any], outcomes: list[str]) -> None:
    handoff = validate_phase3_handoff(config, outcomes)
    response = {}
    structural = {}
    stability = {}
    for outcome in outcomes:
        response[outcome] = fit_response_baseline(config, outcome)
        print(f"{outcome}: response-propensity baseline prepared without test outcomes.")
        structural[outcome] = fit_structural_baselines(config, outcome)
        print(f"{outcome}: constant and segment baselines prepared without test outcomes.")
        stability[outcome] = phase3_stability(config, outcome)
        print(f"{outcome}: current Phase 3 stability evidence prepared without test outcomes.")
    lock = {
        "status": "evaluation_spec_frozen_before_test_outcomes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_spec_hash": phase4_spec_hash(config),
        "test_outcomes_used": False,
        "analysis_status": config["analysis_status"],
        "handoff": handoff,
        "response_baselines": response,
        "structural_baselines": structural,
        "phase3_stability": stability,
    }
    output_root = resolve_path(config["output_root"])
    completion_marker = output_root / "TEST_EVALUATED.json"
    if completion_marker.exists():
        completion_marker.unlink()
    write_json(output_root / "EVALUATION_SPEC.json", lock)
    print("Phase 4 evaluation specification frozen.")


def align_values(
    source_ids: np.ndarray,
    values: np.ndarray,
    target_ids: np.ndarray,
    label: str,
) -> np.ndarray:
    if np.array_equal(source_ids, target_ids):
        return np.asarray(values)
    source_order = np.argsort(source_ids)
    target_order = np.argsort(target_ids)
    if not np.array_equal(source_ids[source_order], target_ids[target_order]):
        raise AssertionError(f"{label} does not contain the same source_row_id values.")
    aligned = np.empty_like(values)
    aligned[target_order] = values[source_order]
    return aligned


def load_test_outcomes(config: dict[str, Any], outcome: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = (
        pl.scan_parquet(resolve_path(config["phase3_input"]))
        .filter(pl.col("data_partition") == "test")
        .select(
            pl.col("source_row_id").cast(pl.UInt64),
            pl.col("treatment").cast(pl.UInt8),
            pl.col(outcome).cast(pl.UInt8),
        )
        .collect(engine="streaming")
    )
    ids = frame["source_row_id"].to_numpy().astype(np.uint64, copy=False)
    if len(np.unique(ids)) != len(ids):
        raise AssertionError("Duplicate source_row_id values in the test outcomes.")
    return (
        ids,
        frame["treatment"].to_numpy().astype(np.uint8, copy=False),
        frame[outcome].to_numpy().astype(np.uint8, copy=False),
    )


def load_scores(config: dict[str, Any], outcome: str, target_ids: np.ndarray) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for model_name in config["uplift_models"]:
        path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "phase3"
            / "predictions"
            / "test"
            / outcome
            / f"{model_name}.parquet"
        )
        table = pq.read_table(path, columns=["source_row_id", "uplift_hat"])
        ids = table["source_row_id"].combine_chunks().to_numpy(zero_copy_only=False)
        values = table["uplift_hat"].combine_chunks().to_numpy(zero_copy_only=False)
        scores[model_name] = align_values(ids, values, target_ids, model_name)

    response_path = (
        resolve_path(config["output_root"])
        / "baselines"
        / outcome
        / "response_propensity.parquet"
    )
    table = pq.read_table(response_path, columns=["source_row_id", "response_score"])
    ids = table["source_row_id"].combine_chunks().to_numpy(zero_copy_only=False)
    values = table["response_score"].combine_chunks().to_numpy(zero_copy_only=False)
    scores[config["response_model"]] = align_values(ids, values, target_ids, config["response_model"])
    for model_name in (config["constant_model"], config["segment_model"]):
        path = (
            resolve_path(config["output_root"])
            / "baselines"
            / outcome
            / f"{model_name}.parquet"
        )
        table = pq.read_table(path, columns=["source_row_id", "uplift_hat"])
        ids = table["source_row_id"].combine_chunks().to_numpy(zero_copy_only=False)
        values = table["uplift_hat"].combine_chunks().to_numpy(zero_copy_only=False)
        scores[model_name] = align_values(ids, values, target_ids, model_name)
    for model_name, value in scores.items():
        if not np.isfinite(value).all():
            raise AssertionError(f"Non-finite evaluation scores for {model_name}.")
    return scores


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
    return np.bincount(index, weights=weights, minlength=group_count * 4).reshape(
        group_count, 2, 2
    )


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


def phase2_ate(config: dict[str, Any], outcome_name: str) -> float:
    table = pd.read_csv(resolve_path(config["phase2_effects"]))
    row = table.loc[table["outcome"] == outcome_name]
    if len(row) != 1:
        raise AssertionError(f"Phase 2 ATE row missing or duplicated for {outcome_name}.")
    return float(row.iloc[0]["absolute_lift"])


def save_figures(
    config: dict[str, Any],
    outcome_name: str,
    group_effects: pd.DataFrame,
    cumulative: pd.DataFrame,
    calibration: pd.DataFrame,
    policy: pd.DataFrame,
) -> None:
    figure_directory = resolve_path(config["figure_directory"])
    figure_directory.mkdir(parents=True, exist_ok=True)
    model_order = config["uplift_models"] + [
        config["constant_model"],
        config["segment_model"],
        config["response_model"],
    ]

    figure, axis = plt.subplots(figsize=(8, 5))
    endpoint = float(cumulative["endpoint_gain"].iloc[0])
    for model_name in model_order:
        table = cumulative[cumulative["model_name"] == model_name]
        axis.plot(table["target_fraction"] * 100, table["cumulative_gain"], label=model_name)
    axis.plot([0, 100], [0, endpoint], linestyle="--", color="#667085", label="random_uniform")
    axis.set_title(f"{outcome_name.title()} cumulative gain")
    axis.set_xlabel("Targeted population (%)")
    axis.set_ylabel("Estimated incremental outcomes")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_cumulative_gain.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name in model_order:
        table = cumulative[cumulative["model_name"] == model_name]
        random_gain = table["target_fraction"] * endpoint
        axis.plot(
            table["target_fraction"] * 100,
            table["cumulative_gain"] - random_gain,
            label=model_name,
        )
    axis.axhline(0, linestyle="--", color="#667085")
    axis.set_title(f"{outcome_name.title()} Qini curves")
    axis.set_xlabel("Targeted population (%)")
    axis.set_ylabel("Gain above random")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_qini_curve.png", dpi=180)
    plt.close(figure)

    columns = 2
    rows = math.ceil(len(model_order) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(11, 3.7 * rows), sharex=False)
    for axis, model_name in zip(np.asarray(axes).flat, model_order):
        table = group_effects[group_effects["model_name"] == model_name]
        x = table["score_group"]
        y = table["observed_uplift"]
        lower = y - table["uplift_ci_lower"]
        upper = table["uplift_ci_upper"] - y
        axis.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=2)
        axis.axhline(0, color="#667085", linewidth=0.8)
        axis.set_title(model_name)
        axis.set_xlabel("Score group (1 = highest)")
        axis.set_ylabel("Observed uplift")
    for axis in list(np.asarray(axes).flat)[len(model_order):]:
        axis.set_visible(False)
    figure.suptitle(f"{outcome_name.title()} observed uplift by score group")
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_group_uplift.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name in config["uplift_models"]:
        table = calibration[calibration["model_name"] == model_name]
        axis.scatter(table["mean_predicted_score"], table["observed_uplift"], label=model_name)
        ordered = table.sort_values("mean_predicted_score")
        axis.plot(
            ordered["mean_predicted_score"],
            ordered["calibrated_predicted_uplift"],
        )
    axis.set_title(f"{outcome_name.title()} uplift calibration")
    axis.set_xlabel("Mean predicted uplift")
    axis.set_ylabel("Observed treatment-control difference")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_uplift_calibration.png", dpi=180)
    plt.close(figure)

    main = policy[
        policy["target_fraction"].isin([float(x) for x in config["main_decision_rates"]])
    ].copy()
    pivot = main.pivot(index="model_name", columns="target_fraction", values="cumulative_gain")
    pivot = pivot.reindex(model_order)
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title(f"{outcome_name.title()} targeting strategies")
    axis.set_xlabel("Ranking strategy")
    axis.set_ylabel("Estimated incremental outcomes")
    axis.legend(title="Target fraction", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_targeting_comparison.png", dpi=180)
    plt.close(figure)


def add_policy_comparisons(
    config: dict[str, Any], policy: pd.DataFrame, uncertainty: pd.DataFrame
) -> pd.DataFrame:
    policy = policy.copy()
    response_name = config["response_model"]
    response_gain = policy[policy["model_name"] == response_name].set_index(
        "target_fraction"
    )["cumulative_gain"]
    policy["gain_above_response"] = [
        float(row.cumulative_gain - response_gain.loc[row.target_fraction])
        for row in policy.itertuples(index=False)
    ]
    policy["incremental_gain_ci_lower"] = math.nan
    policy["incremental_gain_ci_upper"] = math.nan
    policy["gain_above_response_ci_lower"] = math.nan
    policy["gain_above_response_ci_upper"] = math.nan
    main_rates = {round(float(value), 10) for value in config["main_decision_rates"]}
    lookup = uncertainty.set_index(["model_name", "metric"])
    for index, row in policy.iterrows():
        rate = round(float(row["target_fraction"]), 10)
        if rate not in main_rates:
            continue
        percent = int(round(rate * 100))
        gain_key = (row["model_name"], f"gain_at_{percent}pct")
        if gain_key in lookup.index:
            interval = lookup.loc[gain_key]
            policy.loc[index, "incremental_gain_ci_lower"] = float(interval["ci_lower"])
            policy.loc[index, "incremental_gain_ci_upper"] = float(interval["ci_upper"])
        if row["model_name"] == response_name:
            policy.loc[index, "gain_above_response_ci_lower"] = 0.0
            policy.loc[index, "gain_above_response_ci_upper"] = 0.0
        else:
            difference_key = (
                row["model_name"],
                f"gain_at_{percent}pct_minus_response",
            )
            if difference_key in lookup.index:
                interval = lookup.loc[difference_key]
                policy.loc[index, "gain_above_response_ci_lower"] = float(
                    interval["ci_lower"]
                )
                policy.loc[index, "gain_above_response_ci_upper"] = float(
                    interval["ci_upper"]
                )
    policy["analysis_status"] = config["analysis_status"]
    return policy


def evaluate_outcome(config: dict[str, Any], outcome_name: str) -> dict[str, Any]:
    source_row_id, treatment, outcome = load_test_outcomes(config, outcome_name)
    scores = load_scores(config, outcome_name, source_row_id)
    curve_group_count = int(round(1.0 / float(config["cumulative_step"])))
    display_group_count = int(config["score_groups"][outcome_name])
    confidence_level = float(config["confidence_level"])
    tie_break_seed = int(config["tie_break_seed"])
    group_tables = []
    curve_tables = []
    curve_groups: dict[str, np.ndarray] = {}
    calibration_tables = []
    calibration_parameters = []

    for model_name, score in scores.items():
        groups, _ = group_effect_table(
            outcome_name,
            model_name,
            score,
            source_row_id,
            treatment,
            outcome,
            display_group_count,
            confidence_level,
            tie_break_seed,
        )
        group_tables.append(groups)
        curve, group_ids = cumulative_curve_table(
            outcome_name,
            model_name,
            score,
            source_row_id,
            treatment,
            outcome,
            curve_group_count,
            confidence_level,
            tie_break_seed,
        )
        curve_tables.append(curve)
        curve_groups[model_name] = group_ids
        if model_name in config["uplift_models"]:
            table, parameters = calibration_table(outcome_name, model_name, groups)
            calibration_tables.append(table)
            calibration_parameters.append(parameters)

    group_effects = pd.concat(group_tables, ignore_index=True)
    cumulative = pd.concat(curve_tables, ignore_index=True)
    endpoint_values = cumulative.groupby("model_name")["cumulative_gain"].last()
    if float(endpoint_values.max() - endpoint_values.min()) > 1e-8:
        raise AssertionError("All rankings must share the same Top 100% endpoint.")
    endpoint = float(endpoint_values.iloc[0])
    cumulative["random_gain"] = cumulative["target_fraction"] * endpoint
    cumulative["gain_above_random"] = cumulative["cumulative_gain"] - cumulative["random_gain"]
    cumulative["endpoint_gain"] = endpoint

    metric_rows = []
    for model_name, table in cumulative.groupby("model_name", sort=False):
        auuc, qini, endpoint_value = curve_metrics(table)
        metric_rows.append(
            {
                "outcome_name": outcome_name,
                "model_name": model_name,
                "auuc": auuc,
                "qini": qini,
                "endpoint_gain": endpoint_value,
            }
        )
    metric_rows.append(
        {
            "outcome_name": outcome_name,
            "model_name": "random_uniform",
            "auuc": 0.5 * endpoint,
            "qini": 0.0,
            "endpoint_gain": endpoint,
        }
    )
    metrics = pd.DataFrame(metric_rows)

    report_rates = [float(value) for value in config["report_rates"]]
    policy = cumulative[
        cumulative["target_fraction"].round(10).isin([round(x, 10) for x in report_rates])
    ].copy()
    calibration = pd.concat(calibration_tables, ignore_index=True)
    calibration_parameter_table = pd.DataFrame(calibration_parameters)
    bootstrap = bootstrap_uncertainty(
        config,
        outcome_name,
        curve_groups,
        treatment,
        outcome,
        metrics[metrics["model_name"] != "random_uniform"],
        policy,
    )
    policy = add_policy_comparisons(config, policy, bootstrap)

    overall_counts = cell_counts(
        np.zeros(len(outcome), dtype=np.int16), treatment, outcome, 1
    )[0]
    _, _, _, _, test_ate, _, _ = difference_in_rates(overall_counts, confidence_level)
    phase2_value = phase2_ate(config, outcome_name)
    ate_check = pd.DataFrame(
        [
            {
                "outcome_name": outcome_name,
                "test_rows": len(outcome),
                "test_ate": test_ate,
                "phase2_ate": phase2_value,
                "test_minus_phase2_ate": test_ate - phase2_value,
                "endpoint_gain": endpoint,
                "expected_endpoint_gain": len(outcome) * test_ate,
            }
        ]
    )
    if not math.isclose(endpoint, len(outcome) * test_ate, rel_tol=1e-10, abs_tol=1e-6):
        raise AssertionError("Top 100% gain does not equal N_test times observed test ATE.")

    targeting_references = pd.DataFrame(
        [
            {
                "outcome_name": outcome_name,
                "strategy": "target_none",
                "target_fraction": 0.0,
                "selected_rows": 0,
                "incremental_outcomes": 0.0,
            },
            {
                "outcome_name": outcome_name,
                "strategy": "target_all",
                "target_fraction": 1.0,
                "selected_rows": len(outcome),
                "incremental_outcomes": endpoint,
            },
            {
                "outcome_name": outcome_name,
                "strategy": "random_targeting",
                "target_fraction": math.nan,
                "selected_rows": math.nan,
                "incremental_outcomes": math.nan,
            },
        ]
    )
    targeting_references["analysis_status"] = config["analysis_status"]

    table_directory = resolve_path(config["table_directory"])
    table_directory.mkdir(parents=True, exist_ok=True)
    group_effects.to_csv(table_directory / f"{outcome_name}_group_effects.csv", index=False)
    cumulative.to_csv(table_directory / f"{outcome_name}_cumulative_gain.csv", index=False)
    metrics.to_csv(table_directory / f"{outcome_name}_model_metrics.csv", index=False)
    policy.to_csv(table_directory / f"{outcome_name}_policy_summary.csv", index=False)
    calibration.to_csv(table_directory / f"{outcome_name}_calibration_groups.csv", index=False)
    calibration_parameter_table.to_csv(
        table_directory / f"{outcome_name}_calibration_parameters.csv", index=False
    )
    bootstrap.to_csv(table_directory / f"{outcome_name}_uncertainty.csv", index=False)
    ate_check.to_csv(table_directory / f"{outcome_name}_ate_sanity.csv", index=False)
    targeting_references.to_csv(
        table_directory / f"{outcome_name}_targeting_references.csv", index=False
    )
    save_figures(config, outcome_name, group_effects, cumulative, calibration, policy)
    print(f"{outcome_name}: Phase 4 evaluation complete.")
    return {
        "source_row_id": source_row_id,
        "treatment": treatment,
        "outcome": outcome,
        "scores": scores,
        "metrics": metrics,
        "policy": policy,
        "uncertainty": bootstrap,
        "cumulative": cumulative,
    }


def select_final_models(
    config: dict[str, Any], results: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    """Apply one transparent rule to choose an uplift model and deployment ranking."""
    table_directory = resolve_path(config["table_directory"])
    recommendation_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for outcome_name, result in results.items():
        metrics = result["metrics"].set_index("model_name")
        uncertainty = result["uncertainty"].set_index(["model_name", "metric"])
        stability = pd.read_csv(table_directory / f"{outcome_name}_phase3_stability.csv")
        stability_summary = stability.groupby("model_name").agg(
            mean_spearman_rank_correlation=("spearman_rank_correlation", "mean"),
            minimum_spearman_rank_correlation=("spearman_rank_correlation", "min"),
            mean_top_set_overlap=("top_set_overlap", "mean"),
            minimum_top_set_overlap=("top_set_overlap", "min"),
        )
        for complexity_rank, model_name in enumerate(config["uplift_models"], start=1):
            difference = uncertainty.loc[(model_name, "qini_minus_response")]
            stability_row = stability_summary.loc[model_name]
            comparison_rows.append(
                {
                    "outcome_name": outcome_name,
                    "model_name": model_name,
                    "complexity_rank": complexity_rank,
                    "auuc": float(metrics.loc[model_name, "auuc"]),
                    "qini": float(metrics.loc[model_name, "qini"]),
                    "qini_minus_response": float(difference["estimate"]),
                    "qini_minus_response_ci_lower": float(difference["ci_lower"]),
                    "qini_minus_response_ci_upper": float(difference["ci_upper"]),
                    **{key: float(value) for key, value in stability_row.items()},
                    "analysis_status": config["analysis_status"],
                }
            )
        outcome_comparison = pd.DataFrame(
            [row for row in comparison_rows if row["outcome_name"] == outcome_name]
        )
        # Start with the point-Qini leader, then choose the simplest learner whose
        # paired difference from that leader is not statistically distinguishable.
        point_leader = outcome_comparison.sort_values(
            ["qini", "complexity_rank"], ascending=[False, True]
        ).iloc[0]
        selected = point_leader
        simplicity_rule_applied = False
        leader_index = list(config["uplift_models"]).index(str(point_leader["model_name"]))
        for simpler_model in config["uplift_models"][:leader_index]:
            difference_key = (
                str(point_leader["model_name"]),
                f"qini_minus_{simpler_model}",
            )
            if difference_key not in uncertainty.index:
                continue
            paired_difference = uncertainty.loc[difference_key]
            if float(paired_difference["ci_lower"]) <= 0.0:
                selected = outcome_comparison[
                    outcome_comparison["model_name"] == simpler_model
                ].iloc[0]
                simplicity_rule_applied = True
                break
        beats_response = bool(selected["qini_minus_response_ci_lower"] > 0)
        deployment_model = (
            str(selected["model_name"]) if beats_response else config["response_model"]
        )
        main_policy = result["policy"][
            (result["policy"]["model_name"] == deployment_model)
            & result["policy"]["target_fraction"].isin(
                [float(value) for value in config["main_decision_rates"]]
            )
        ].copy()
        if main_policy["incremental_gain_ci_lower"].notna().any():
            chosen_policy = main_policy.sort_values(
                ["incremental_gain_ci_lower", "cumulative_gain"], ascending=False
            ).iloc[0]
        else:
            chosen_policy = main_policy.sort_values("cumulative_gain", ascending=False).iloc[0]
        recommendation_rows.append(
            {
                "outcome_name": outcome_name,
                "point_qini_leader": str(point_leader["model_name"]),
                "recommended_uplift_model": str(selected["model_name"]),
                "simplicity_rule_applied": simplicity_rule_applied,
                "deployment_ranking": deployment_model,
                "uplift_model_beats_response_at_95pct": beats_response,
                "recommended_target_fraction": float(chosen_policy["target_fraction"]),
                "selected_users": int(chosen_policy["selected_rows"]),
                "selected_group_uplift": float(chosen_policy["cumulative_uplift"]),
                "estimated_incremental_outcomes": float(chosen_policy["cumulative_gain"]),
                "incremental_outcomes_ci_lower": float(
                    chosen_policy["incremental_gain_ci_lower"]
                ),
                "incremental_outcomes_ci_upper": float(
                    chosen_policy["incremental_gain_ci_upper"]
                ),
                "uplift_model_qini": float(selected["qini"]),
                "qini_minus_response_ci_lower": float(
                    selected["qini_minus_response_ci_lower"]
                ),
                "qini_minus_response_ci_upper": float(
                    selected["qini_minus_response_ci_upper"]
                ),
                "mean_stability_rank_correlation": float(
                    selected["mean_spearman_rank_correlation"]
                ),
                "mean_stability_top_set_overlap": float(
                    selected["mean_top_set_overlap"]
                ),
                "selection_rule": (
                    "start from highest uplift-model Qini; choose the simplest learner not "
                    "significantly worse in paired Qini; deploy uplift ranking only when its "
                    "paired Qini-minus-response 95% CI is above zero; choose 10/20/30 rate "
                    "with highest bootstrap lower bound for incremental outcomes"
                ),
                "analysis_status": config["analysis_status"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    recommendations = pd.DataFrame(recommendation_rows)
    comparison.to_csv(table_directory / "final_model_comparison.csv", index=False)
    recommendations.to_csv(table_directory / "final_model_recommendations.csv", index=False)
    return recommendations


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
    n0, n1, rate0, rate1, uplift, lower, upper = difference_in_rates(
        counts, confidence_level
    )
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


def compare_visit_conversion_policies(
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    visit = results["visit"]
    conversion = results["conversion"]
    if not np.array_equal(visit["source_row_id"], conversion["source_row_id"]):
        raise AssertionError("Visit and conversion policy comparison requires identical users/order.")
    if not np.array_equal(visit["treatment"], conversion["treatment"]):
        raise AssertionError("Treatment labels differ across outcome evaluations.")
    visit_model = str(
        recommendations.loc[
            recommendations["outcome_name"] == "visit", "recommended_uplift_model"
        ].iloc[0]
    )
    conversion_model = str(
        recommendations.loc[
            recommendations["outcome_name"] == "conversion", "recommended_uplift_model"
        ].iloc[0]
    )
    visit_score = visit["scores"][visit_model]
    conversion_score = conversion["scores"][conversion_model]
    from scipy.stats import spearmanr

    rank_correlation = float(spearmanr(visit_score, conversion_score).statistic)
    rows: list[dict[str, Any]] = []
    ids = visit["source_row_id"]
    seed = int(config["tie_break_seed"])
    confidence = float(config["confidence_level"])
    for rate in [float(value) for value in config["main_decision_rates"]]:
        visit_mask = top_fraction_mask(visit_score, ids, rate, seed)
        conversion_mask = top_fraction_mask(conversion_score, ids, rate, seed)
        intersection = int(np.logical_and(visit_mask, conversion_mask).sum())
        conversion_under_visit = selected_policy_effect(
            visit_mask, conversion["treatment"], conversion["outcome"], confidence
        )
        conversion_under_own = selected_policy_effect(
            conversion_mask, conversion["treatment"], conversion["outcome"], confidence
        )
        visit_under_conversion = selected_policy_effect(
            conversion_mask, visit["treatment"], visit["outcome"], confidence
        )
        visit_under_own = selected_policy_effect(
            visit_mask, visit["treatment"], visit["outcome"], confidence
        )
        row: dict[str, Any] = {
            "visit_model": visit_model,
            "conversion_model": conversion_model,
            "target_fraction": rate,
            "score_spearman_rank_correlation": rank_correlation,
            "shared_selected_users": intersection,
            "shared_as_pct_of_each_targeted_set": 100.0 * intersection / visit_mask.sum(),
            "shared_as_pct_of_test_population": 100.0 * intersection / len(ids),
            "analysis_status": config["analysis_status"],
        }
        for prefix, values in (
            ("conversion_gain_using_visit_policy", conversion_under_visit),
            ("conversion_gain_using_conversion_policy", conversion_under_own),
            ("visit_gain_using_conversion_policy", visit_under_conversion),
            ("visit_gain_using_visit_policy", visit_under_own),
        ):
            row[prefix] = values["gain"]
            row[f"{prefix}_ci_lower"] = values["gain_ci_lower"]
            row[f"{prefix}_ci_upper"] = values["gain_ci_upper"]
            row[f"{prefix}_uplift"] = values["uplift"]
        rows.append(row)
    comparison = pd.DataFrame(rows)
    table_directory = resolve_path(config["table_directory"])
    comparison.to_csv(table_directory / "visit_conversion_policy_comparison.csv", index=False)

    figure_directory = resolve_path(config["figure_directory"])
    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.bar(
        comparison["target_fraction"] * 100,
        comparison["shared_as_pct_of_each_targeted_set"],
        width=6,
        color="#3568A8",
        edgecolor="#1F2937",
    )
    axis.set_title("Visit versus conversion policy overlap")
    axis.set_xlabel("Targeted population in each policy (%)")
    axis.set_ylabel("Users selected by both (% of targeted set)")
    axis.set_ylim(0, 100)
    for x, y in zip(
        comparison["target_fraction"] * 100,
        comparison["shared_as_pct_of_each_targeted_set"],
    ):
        axis.text(x, y + 2, f"{y:.1f}%", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(figure_directory / "visit_conversion_policy_overlap.png", dpi=180)
    plt.close(figure)
    return comparison


def evaluate(config: dict[str, Any], outcomes: list[str]) -> None:
    lock_path = resolve_path(config["output_root"]) / "EVALUATION_SPEC.json"
    if not lock_path.exists():
        raise FileNotFoundError("Run Phase 4 prepare before joining test outcomes.")
    lock = read_json(lock_path)
    if lock["evaluation_spec_hash"] != phase4_spec_hash(config):
        raise AssertionError("Phase 4 evaluation config changed after the specification was frozen.")
    current_handoff = validate_phase3_handoff(config, outcomes)
    for outcome in outcomes:
        if current_handoff[outcome] != lock["handoff"][outcome]:
            raise AssertionError(f"The frozen Phase 3 handoff changed for {outcome}.")
        response_metadata = read_json(
            resolve_path(config["output_root"])
            / "baselines"
            / outcome
            / "metadata.json"
        )
        if response_metadata["model_config_hash"] != lock["response_baselines"][outcome][
            "model_config_hash"
        ]:
            raise AssertionError(f"The frozen response baseline changed for {outcome}.")
        if response_metadata.get("test_outcomes_used") is not False:
            raise AssertionError(f"The response baseline used test outcomes for {outcome}.")
        structural_metadata = read_json(
            resolve_path(config["output_root"])
            / "baselines"
            / outcome
            / "structural_metadata.json"
        )
        if structural_metadata["model_config_hash"] != lock["structural_baselines"][outcome][
            "model_config_hash"
        ]:
            raise AssertionError(f"The frozen structural baselines changed for {outcome}.")
        if structural_metadata.get("test_outcomes_used") is not False:
            raise AssertionError(f"A structural baseline used test outcomes for {outcome}.")
        stability_metadata = read_json(
            resolve_path(config["output_root"]) / "stability" / outcome / "metadata.json"
        )
        if stability_metadata["stability_spec_hash"] != lock["phase3_stability"][outcome][
            "stability_spec_hash"
        ]:
            raise AssertionError(f"The frozen Phase 3 stability evidence changed for {outcome}.")
        if stability_metadata.get("test_outcomes_used") is not False:
            raise AssertionError(f"Phase 3 stability used test outcomes for {outcome}.")
    results = {outcome: evaluate_outcome(config, outcome) for outcome in outcomes}
    recommendations = select_final_models(config, results)
    if set(outcomes) == {"visit", "conversion"}:
        compare_visit_conversion_policies(config, results, recommendations)
    write_json(
        resolve_path(config["output_root"]) / "TEST_EVALUATED.json",
        {
            "status": "test_evaluation_complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_spec_hash": lock["evaluation_spec_hash"],
            "outcomes": outcomes,
            "analysis_status": config["analysis_status"],
            "final_recommendations": recommendations.to_dict(orient="records"),
        },
    )


def main() -> None:
    args = parse_args()
    config = read_json(resolve_path(args.config))
    outcomes = config["outcomes"] if args.outcome == "both" else [args.outcome]
    if args.stage == "prepare":
        prepare(config, outcomes)
    elif args.stage == "evaluate":
        evaluate(config, outcomes)
    elif args.stage == "all":
        prepare(config, outcomes)
        evaluate(config, outcomes)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
