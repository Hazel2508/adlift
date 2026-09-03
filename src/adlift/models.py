"""Train and score the Phase 3 S-, T-, and X-Learners.

Phase 3 fits models with train data, uses validation data for hyperparameter
selection/early stopping, freezes the chosen specification, and then scores
test features without reading test outcomes. Ranking evaluation, ATE checks,
AUUC, Qini, calibration, and targeting decisions belong to Phase 4.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


from .paths import PROJECT_ROOT

LEARNERS = ("s_learner", "t_learner", "x_learner")


@dataclass
class PartitionData:
    source_row_id: np.ndarray
    features: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray | None
    split: str


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Install requirements-phase3.txt; on macOS LightGBM also requires libomp."
        ) from error
    return lgb


def load_partition(
    parquet_path: Path,
    feature_names: list[str],
    split: str,
    outcome: str | None,
    sample_fraction: float = 1.0,
) -> PartitionData:
    """Load only columns permitted for the requested Phase 3 operation."""
    columns: list[pl.Expr] = [pl.col("source_row_id").cast(pl.UInt64)]
    columns.extend(pl.col(name).cast(pl.Float32) for name in feature_names)
    columns.append(pl.col("treatment").cast(pl.UInt8))
    if outcome is not None:
        columns.append(pl.col(outcome).cast(pl.UInt8))
    lazy = pl.scan_parquet(parquet_path).filter(pl.col("data_partition") == split)
    if sample_fraction < 1.0:
        denominator = 10_000
        numerator = max(1, round(sample_fraction * denominator))
        lazy = lazy.filter((pl.col("source_row_id") % denominator) < numerator)
    frame = lazy.select(columns).collect(engine="streaming")
    if frame.height == 0:
        raise ValueError(f"No rows loaded for split={split}, fraction={sample_fraction}.")
    return PartitionData(
        source_row_id=frame["source_row_id"].to_numpy().astype(np.uint64, copy=False),
        features=np.ascontiguousarray(frame.select(feature_names).to_numpy(), dtype=np.float32),
        treatment=frame["treatment"].to_numpy().astype(np.uint8, copy=False),
        outcome=(frame[outcome].to_numpy().astype(np.uint8, copy=False) if outcome else None),
        split=split,
    )


def check_model_matrix(feature_names: list[str], outcome: str) -> None:
    forbidden = {
        "exposure",
        "source_row_id",
        "data_partition",
        "split",
        "split_stratum",
        "crossfit_fold",
        "visit" if outcome == "conversion" else "conversion",
    }
    overlap = forbidden.intersection(feature_names)
    if overlap:
        raise AssertionError(f"Forbidden model inputs detected: {sorted(overlap)}")
    if feature_names != [f"f{i}" for i in range(12)]:
        raise AssertionError("Canonical model inputs must be exactly f0 through f11.")


def lightgbm_parameters(config: dict[str, Any], outcome: str, kind: str) -> dict[str, Any]:
    parameters = dict(config["lightgbm"]["common"])
    parameters.update(config["lightgbm"][f"{outcome}_{kind}"])
    seed = int(config["seed"])
    parameters.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
        }
    )
    return parameters


def fit_binary_model(
    lgb: Any,
    train_features: np.ndarray,
    train_outcome: np.ndarray,
    validation_features: np.ndarray,
    validation_outcome: np.ndarray,
    parameters: dict[str, Any],
    feature_names: list[str],
) -> Any:
    params = dict(parameters)
    rounds = int(params.pop("num_boost_round"))
    patience = int(params.pop("early_stopping_rounds", 0))
    train_set = lgb.Dataset(train_features, label=train_outcome, feature_name=feature_names)
    validation_set = lgb.Dataset(
        validation_features,
        label=validation_outcome,
        reference=train_set,
        feature_name=feature_names,
    )
    callbacks = [lgb.early_stopping(patience, verbose=False)] if patience else []
    return lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
    )


def fit_effect_model(
    lgb: Any,
    features: np.ndarray,
    pseudo_outcome: np.ndarray,
    parameters: dict[str, Any],
    feature_names: list[str],
) -> Any:
    params = dict(parameters)
    rounds = int(params.pop("num_boost_round"))
    params.pop("early_stopping_rounds", None)
    dataset = lgb.Dataset(features, label=pseudo_outcome, feature_name=feature_names)
    return lgb.train(params, dataset, num_boost_round=rounds)


def prediction_iterations(model: Any) -> int:
    best = int(getattr(model, "best_iteration", 0) or 0)
    return best if best > 0 else int(model.current_iteration())


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return {
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "roc_auc": float(roc_auc_score(y_true, clipped)),
        "pr_auc": float(average_precision_score(y_true, clipped)),
    }


def construct_x_pseudo_outcome(
    treatment: np.ndarray,
    outcome: np.ndarray,
    mu0_hat: np.ndarray,
    mu1_hat: np.ndarray,
) -> np.ndarray:
    """Construct the single X-Learner target D for the pooled X-style variant."""
    treated = treatment == 1
    control = ~treated
    pseudo_outcome = np.empty(len(treatment), dtype=np.float32)
    pseudo_outcome[treated] = outcome[treated] - mu0_hat[treated]
    pseudo_outcome[control] = mu1_hat[control] - outcome[control]
    return pseudo_outcome


def train_models(
    config: dict[str, Any], outcome: str, train: PartitionData, validation: PartitionData
) -> dict[str, Any]:
    """Fit only S/T/X and produce validation uplift scores."""
    lgb = require_lightgbm()
    assert train.outcome is not None and validation.outcome is not None
    feature_names = config["features"]
    binary_params = lightgbm_parameters(config, outcome, "binary")
    effect_params = lightgbm_parameters(config, outcome, "effect")
    start = time.perf_counter()

    # S-Learner: one response model with treatment included.
    s_train = np.column_stack((train.features, train.treatment.astype(np.float32)))
    s_validation_factual = np.column_stack(
        (validation.features, validation.treatment.astype(np.float32))
    )
    s_model = fit_binary_model(
        lgb,
        s_train,
        train.outcome,
        s_validation_factual,
        validation.outcome,
        binary_params,
        feature_names + ["treatment"],
    )
    s_validation_0 = np.column_stack(
        (validation.features, np.zeros(len(validation.treatment), dtype=np.float32))
    )
    s_validation_1 = np.column_stack(
        (validation.features, np.ones(len(validation.treatment), dtype=np.float32))
    )
    s_mu0 = s_model.predict(s_validation_0, num_iteration=prediction_iterations(s_model)).astype(
        np.float32
    )
    s_mu1 = s_model.predict(s_validation_1, num_iteration=prediction_iterations(s_model)).astype(
        np.float32
    )

    # T-Learner: separate response models for treatment and control.
    control_train = train.treatment == 0
    treated_train = train.treatment == 1
    control_validation = validation.treatment == 0
    treated_validation = validation.treatment == 1
    t_mu0_model = fit_binary_model(
        lgb,
        train.features[control_train],
        train.outcome[control_train],
        validation.features[control_validation],
        validation.outcome[control_validation],
        binary_params,
        feature_names,
    )
    t_mu1_model = fit_binary_model(
        lgb,
        train.features[treated_train],
        train.outcome[treated_train],
        validation.features[treated_validation],
        validation.outcome[treated_validation],
        binary_params,
        feature_names,
    )
    t_mu0 = t_mu0_model.predict(
        validation.features, num_iteration=prediction_iterations(t_mu0_model)
    ).astype(np.float32)
    t_mu1 = t_mu1_model.predict(
        validation.features, num_iteration=prediction_iterations(t_mu1_model)
    ).astype(np.float32)

    # Pooled X-style learner: create one D dataset and train one g(X).
    train_mu0 = t_mu0_model.predict(
        train.features, num_iteration=prediction_iterations(t_mu0_model)
    ).astype(np.float32)
    train_mu1 = t_mu1_model.predict(
        train.features, num_iteration=prediction_iterations(t_mu1_model)
    ).astype(np.float32)
    pseudo_outcome = construct_x_pseudo_outcome(
        train.treatment, train.outcome, train_mu0, train_mu1
    )
    x_model = fit_effect_model(lgb, train.features, pseudo_outcome, effect_params, feature_names)
    x_uplift = x_model.predict(validation.features).astype(np.float32)

    predictions = {
        "s_learner": (s_mu1 - s_mu0).astype(np.float32),
        "t_learner": (t_mu1 - t_mu0).astype(np.float32),
        "x_learner": x_uplift,
    }
    for learner, uplift in predictions.items():
        if not np.isfinite(uplift).all():
            raise AssertionError(f"Non-finite validation predictions for {learner}.")

    diagnostics: list[dict[str, Any]] = []
    s_factual = np.where(validation.treatment == 1, s_mu1, s_mu0)
    t_factual = np.where(validation.treatment == 1, t_mu1, t_mu0)
    for learner, factual in (("s_learner", s_factual), ("t_learner", t_factual)):
        for arm in (0, 1):
            mask = validation.treatment == arm
            row: dict[str, Any] = {"model_name": learner, "arm": arm}
            row.update(binary_metrics(validation.outcome[mask], factual[mask]))
            diagnostics.append(row)
    for learner, uplift in predictions.items():
        diagnostics.append(
            {
                "model_name": learner,
                "arm": "uplift_score",
                "score_mean": float(uplift.mean()),
                "score_standard_deviation": float(uplift.std()),
                "score_minimum": float(uplift.min()),
                "score_maximum": float(uplift.max()),
            }
        )
    gc.collect()
    return {
        "models": {
            "s_learner": s_model,
            "t_mu0": t_mu0_model,
            "t_mu1": t_mu1_model,
            "x_learner": x_model,
        },
        "predictions": predictions,
        "diagnostics": pd.DataFrame(diagnostics),
        "runtime_seconds": time.perf_counter() - start,
    }


def constant_dictionary(value: str, length: int) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(
        pa.array(np.zeros(length, dtype=np.int8)), pa.array([value])
    )


def write_prediction_artifact(
    output_path: Path,
    data: PartitionData,
    outcome_name: str,
    model_name: str,
    config_hash: str,
    source_sha256: str,
    uplift: np.ndarray,
) -> None:
    length = len(data.source_row_id)
    observed = (
        pa.array(data.outcome, type=pa.uint8())
        if data.outcome is not None
        else pa.nulls(length, type=pa.uint8())
    )
    table = pa.table(
        {
            "source_row_id": pa.array(data.source_row_id, type=pa.uint64()),
            "split": constant_dictionary(data.split, length),
            "outcome_name": constant_dictionary(outcome_name, length),
            "model_name": constant_dictionary(model_name, length),
            "treatment": pa.array(data.treatment, type=pa.uint8()),
            "observed_outcome": observed,
            "uplift_hat": pa.array(np.asarray(uplift, dtype=np.float32), type=pa.float32()),
            "model_config_hash": constant_dictionary(config_hash, length),
            "source_file_sha256": constant_dictionary(source_sha256, length),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd", row_group_size=100_000)


def save_booster(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def clear_generated_directory(path: Path, suffixes: tuple[str, ...]) -> None:
    """Remove generated files only in the exact outcome directory being rebuilt."""
    if path.exists():
        for item in path.iterdir():
            if item.is_file() and item.suffix in suffixes:
                item.unlink()


def save_development(
    config: dict[str, Any],
    outcome: str,
    mode: str,
    validation: PartitionData,
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[Path, str]:
    artifact_root = resolve_path("data/processed/phase3")
    if mode == "smoke":
        artifact_root = artifact_root / "smoke"
    model_directory = artifact_root / "models" / outcome
    prediction_directory = artifact_root / "predictions" / "validation" / outcome
    clear_generated_directory(model_directory, (".txt", ".json"))
    clear_generated_directory(prediction_directory, (".parquet",))
    lineage_spec = {
        "phase": 3,
        "scope": list(LEARNERS),
        "outcome": outcome,
        "mode": mode,
        "seed": config["seed"],
        "features": config["features"],
        "lightgbm": config["lightgbm"],
        "source_file_sha256": manifest["raw_file_sha256"],
    }
    model_config_hash = canonical_hash(lineage_spec)
    save_booster(result["models"]["s_learner"], model_directory / "s_learner.txt")
    save_booster(result["models"]["t_mu0"], model_directory / "t_mu0.txt")
    save_booster(result["models"]["t_mu1"], model_directory / "t_mu1.txt")
    save_booster(result["models"]["x_learner"], model_directory / "x_learner.txt")
    write_json(
        model_directory / "metadata.json",
        {
            "model_config_hash": model_config_hash,
            "lineage_spec": lineage_spec,
            "runtime_seconds": result["runtime_seconds"],
            "test_outcomes_used": False,
        },
    )
    source_sha = manifest["raw_file_sha256"]
    for model_name in LEARNERS:
        artifact_hash = canonical_hash(
            {"run_config_hash": model_config_hash, "model_name": model_name}
        )
        write_prediction_artifact(
            prediction_directory / f"{model_name}.parquet",
            validation,
            outcome,
            model_name,
            artifact_hash,
            source_sha,
            result["predictions"][model_name],
        )
    report_directory = PROJECT_ROOT / "results" / "tables" / "phase3"
    report_directory.mkdir(parents=True, exist_ok=True)
    result["diagnostics"].to_csv(
        report_directory / f"{mode}_{outcome}_model_diagnostics.csv", index=False
    )
    return model_directory, model_config_hash


def freeze_specification(
    outcome: str, model_directory: Path, model_config_hash: str, manifest: dict[str, Any]
) -> None:
    write_json(
        model_directory / "FROZEN.json",
        {
            "outcome": outcome,
            "model_config_hash": model_config_hash,
            "source_file_sha256": manifest["raw_file_sha256"],
            "test_outcomes_used": False,
            "status": "frozen_after_validation",
        },
    )


def run_development(config: dict[str, Any], outcomes: list[str], mode: str, freeze: bool) -> None:
    if freeze and mode != "full":
        raise ValueError("Only a full-data specification may be frozen.")
    phase3_file = resolve_path(config["input"]["phase3_file"])
    manifest_path = resolve_path(config["input"]["manifest_file"])
    if not phase3_file.exists() or not manifest_path.exists():
        raise FileNotFoundError("Run scripts/prepare_data.py before model development.")
    manifest = read_json(manifest_path)
    for outcome in outcomes:
        check_model_matrix(config["features"], outcome)
        fraction = 1.0 if mode == "full" else float(config["smoke_fraction"][outcome])
        train = load_partition(phase3_file, config["features"], "train", outcome, fraction)
        validation = load_partition(
            phase3_file, config["features"], "validation", outcome, fraction
        )
        result = train_models(config, outcome, train, validation)
        model_directory, model_config_hash = save_development(
            config, outcome, mode, validation, result, manifest
        )
        if freeze:
            freeze_specification(outcome, model_directory, model_config_hash, manifest)
        print(
            f"{outcome}: {mode} complete in {result['runtime_seconds'] / 60:.1f} minutes; config {model_config_hash[:12]}"
        )
        del train, validation, result
        gc.collect()


def load_boosters(model_directory: Path) -> dict[str, Any]:
    lgb = require_lightgbm()
    return {
        "s_learner": lgb.Booster(model_file=str(model_directory / "s_learner.txt")),
        "t_mu0": lgb.Booster(model_file=str(model_directory / "t_mu0.txt")),
        "t_mu1": lgb.Booster(model_file=str(model_directory / "t_mu1.txt")),
        "x_learner": lgb.Booster(model_file=str(model_directory / "x_learner.txt")),
    }


def score_models(data: PartitionData, models: dict[str, Any]) -> dict[str, np.ndarray]:
    count = len(data.treatment)
    s0 = np.column_stack((data.features, np.zeros(count, dtype=np.float32)))
    s1 = np.column_stack((data.features, np.ones(count, dtype=np.float32)))
    s_mu0 = models["s_learner"].predict(s0).astype(np.float32)
    s_mu1 = models["s_learner"].predict(s1).astype(np.float32)
    t_mu0 = models["t_mu0"].predict(data.features).astype(np.float32)
    t_mu1 = models["t_mu1"].predict(data.features).astype(np.float32)
    predictions = {
        "s_learner": (s_mu1 - s_mu0).astype(np.float32),
        "t_learner": (t_mu1 - t_mu0).astype(np.float32),
        "x_learner": models["x_learner"].predict(data.features).astype(np.float32),
    }
    for model_name, uplift in predictions.items():
        if not np.isfinite(uplift).all():
            raise AssertionError(f"Non-finite test uplift for {model_name}.")
    return predictions


def run_test_scoring(config: dict[str, Any], outcomes: list[str]) -> None:
    phase3_file = resolve_path(config["input"]["phase3_file"])
    manifest = read_json(resolve_path(config["input"]["manifest_file"]))
    source_sha = manifest["raw_file_sha256"]
    for outcome in outcomes:
        model_directory = resolve_path(f"data/processed/phase3/models/{outcome}")
        frozen_path = model_directory / "FROZEN.json"
        if not frozen_path.exists():
            raise FileNotFoundError(
                f"No frozen specification for {outcome}. Run full development with --freeze."
            )
        frozen = read_json(frozen_path)
        metadata = read_json(model_directory / "metadata.json")
        if set(metadata.get("lineage_spec", {}).get("scope", [])) != set(LEARNERS):
            raise RuntimeError(
                f"The frozen {outcome} artifacts use the legacy Phase 3 scope. "
                "Run full development with --freeze before test scoring."
            )
        if frozen["model_config_hash"] != metadata["model_config_hash"]:
            raise AssertionError("Frozen marker and model metadata hashes disagree.")
        if frozen["source_file_sha256"] != source_sha:
            raise AssertionError("Frozen model and current source lineage disagree.")
        test = load_partition(phase3_file, config["features"], "test", outcome=None)
        if test.outcome is not None:
            raise AssertionError("Test outcomes must not be loaded during Phase 3 scoring.")
        models = load_boosters(model_directory)
        predictions = score_models(test, models)
        prediction_directory = resolve_path(f"data/processed/phase3/predictions/test/{outcome}")
        clear_generated_directory(prediction_directory, (".parquet",))
        for model_name in LEARNERS:
            artifact_hash = canonical_hash(
                {"run_config_hash": metadata["model_config_hash"], "model_name": model_name}
            )
            write_prediction_artifact(
                prediction_directory / f"{model_name}.parquet",
                test,
                outcome,
                model_name,
                artifact_hash,
                source_sha,
                predictions[model_name],
            )
        print(f"{outcome}: test features scored; test outcomes were not read.")
        del test, models, predictions
        gc.collect()
