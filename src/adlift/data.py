"""Prepare and audit the released Criteo v2.1 benchmark without dropping records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .paths import PROJECT_ROOT

FEATURES = [f"f{i}" for i in range(12)]
BINARY = ["treatment", "conversion", "visit", "exposure"]


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def partition_expression(seed: str = "adlift-v1") -> str:
    fields = ", ".join(f"CAST({f} AS VARCHAR)" for f in FEATURES)
    bucket = f"md5_number_lower(concat_ws('|', '{seed}', {fields})) % 100"
    return (
        f"CASE WHEN {bucket} < 60 THEN 'train' WHEN {bucket} < 80 THEN 'validation' ELSE 'test' END"
    )


def prepare_dataset(root: Path = PROJECT_ROOT) -> None:
    """Create missing artifacts, audit the data, and retain existing immutable splits."""
    config = json.loads((root / "config/models.json").read_text())
    inputs = config["input"]
    raw = root / inputs["raw_file"]
    base = root / "data/processed/criteo-uplift-v2.1.parquet"
    partitioned = root / inputs["existing_partitioned_file"]
    model_input = root / inputs["phase3_file"]
    manifest_path = root / inputs["manifest_file"]
    tables = root / "results/tables/eda"
    tables.mkdir(parents=True, exist_ok=True)
    base.parent.mkdir(parents=True, exist_ok=True)
    model_input.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '4GB'")
        con.execute("SET preserve_insertion_order = true")
        con.execute(f"SET temp_directory = '{sql_path(root / 'data/processed/duckdb_tmp')}'")
        if not base.exists():
            columns = [f"CAST({f} AS DOUBLE) AS {f}" for f in FEATURES]
            columns += [f"CAST({f} AS UTINYINT) AS {f}" for f in BINARY]
            con.execute(
                f"COPY (SELECT {', '.join(columns)} FROM read_csv_auto('{sql_path(raw)}', header=true, compression='gzip')) TO '{sql_path(base)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )
        source = f"read_parquet('{sql_path(base)}')"
        missing = " OR ".join(f"{f} IS NULL" for f in FEATURES + BINARY)
        invalid = " OR ".join(f"{f} NOT IN (0,1)" for f in BINARY)
        nonfinite = " OR ".join(f"NOT isfinite({f})" for f in FEATURES)
        quality = con.execute(
            f"SELECT COUNT(*) AS rows, COUNT_IF({missing}) AS missing_rows, COUNT_IF({invalid}) AS invalid_binary_rows, COUNT_IF({nonfinite}) AS nonfinite_feature_rows, COUNT_IF(conversion=1 AND visit=0) AS conversion_without_visit, COUNT_IF(treatment=0 AND exposure=1) AS exposed_controls FROM {source}"
        ).df()
        assert int(quality.iloc[0]["rows"]) == int(inputs["expected_rows"])
        assert (quality.iloc[0].drop("rows") == 0).all(), quality.to_string()
        quality.to_csv(tables / "quality_checks.csv", index=False)
        allocation = con.execute(
            f"SELECT treatment, COUNT(*) AS rows, SUM(visit) AS visit_positive, SUM(conversion) AS conversion_positive, SUM(exposure) AS exposed FROM {source} GROUP BY treatment ORDER BY treatment"
        ).df()
        allocation.to_csv(tables / "arm_summary.csv", index=False)
        balances = []
        for f in FEATURES:
            means, vars_ = [], []
            for arm in [0, 1]:
                mean, var = con.execute(
                    f"SELECT AVG({f}), VAR_SAMP({f}) FROM {source} WHERE treatment={arm}"
                ).fetchone()
                means.append(mean)
                vars_.append(var)
            balances.append((f, *means, (means[1] - means[0]) / np.sqrt(sum(vars_) / 2)))
        balance = pd.DataFrame(
            balances, columns=["feature", "control_mean", "treatment_mean", "smd"]
        )
        balance["absolute_smd"] = balance.smd.abs()
        balance.to_csv(tables / "feature_balance.csv", index=False)
        duplicate = con.execute(
            f"SELECT COUNT(*) AS groups, SUM(n) AS rows_in_groups, SUM(n-1) AS extra_identical_records FROM (SELECT COUNT(*) AS n FROM {source} GROUP BY {', '.join(FEATURES + BINARY)} HAVING COUNT(*)>1)"
        ).df()
        duplicate.to_csv(tables / "duplicate_summary.csv", index=False)
        rule = partition_expression(inputs["partition_seed"])
        if not partitioned.exists():
            con.execute(
                f"COPY (SELECT *, {rule} AS data_partition FROM {source}) TO '{sql_path(partitioned)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )
        split_source = f"read_parquet('{sql_path(partitioned)}')"
        mismatches = con.execute(
            f"SELECT COUNT_IF(data_partition <> ({rule})) FROM {split_source}"
        ).fetchone()[0]
        assert mismatches == 0, "Stored split does not match feature-only partition rule."
        if not model_input.exists():
            # The old crossfit_fold column is unnecessary: the frozen pooled learner never reads it.
            cols = ", ".join(FEATURES + ["treatment", "conversion", "visit", "data_partition"])
            con.execute(
                f"COPY (SELECT CAST(file_row_number AS UBIGINT) AS source_row_id, {cols} FROM read_parquet('{sql_path(partitioned)}', file_row_number=true)) TO '{sql_path(model_input)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )
        summary = con.execute(
            f"SELECT data_partition,treatment,COUNT(*) AS rows,SUM(visit) AS visit_positive,SUM(conversion) AS conversion_positive FROM read_parquet('{sql_path(model_input)}') GROUP BY 1,2 ORDER BY 1,2"
        ).df()
        original = con.execute(
            f"SELECT data_partition,treatment,COUNT(*) AS rows,SUM(visit) AS visit_positive,SUM(conversion) AS conversion_positive FROM {split_source} GROUP BY 1,2 ORDER BY 1,2"
        ).df()
        pd.testing.assert_frame_equal(summary, original)
        ids = con.execute(
            f"SELECT COUNT(*),COUNT(DISTINCT source_row_id),MIN(source_row_id),MAX(source_row_id) FROM read_parquet('{sql_path(model_input)}')"
        ).fetchone()
        assert ids == (
            int(inputs["expected_rows"]),
            int(inputs["expected_rows"]),
            0,
            int(inputs["expected_rows"]) - 1,
        )
        summary.to_csv(tables / "partition_summary.csv", index=False)
    if not manifest_path.exists():
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_file": str(raw),
            "raw_file_sha256": sha256_file(raw),
            "existing_partitioned_file": str(partitioned),
            "existing_partitioned_sha256": sha256_file(partitioned),
            "phase3_file": str(model_input),
            "phase3_file_sha256": sha256_file(model_input),
            "row_count": int(inputs["expected_rows"]),
            "partition_seed": inputs["partition_seed"],
            "reconciliation_mismatches": 0,
            "partition_summary": summary.to_dict("records"),
            "source_row_id": {"usage": "technical join key only; prohibited as a model feature"},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    import matplotlib.pyplot as plt

    figures = root / "results/figures/eda"
    figures.mkdir(parents=True, exist_ok=True)
    b = balance.sort_values("absolute_smd")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(b.absolute_smd, b.feature)
    ax.axvline(0.05, color="gray", linestyle="--")
    ax.set(
        xlabel="Absolute standardized mean difference",
        title="Released-sample covariate balance",
        xlim=(0, max(0.055, b.absolute_smd.max() * 1.1)),
    )
    fig.tight_layout()
    fig.savefig(figures / "feature_balance.png", dpi=160)
    plt.close(fig)
    print("Data audit passed; original rows retained and feature-only split verified.")
