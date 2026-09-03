"""Prepare a leakage-safe, lineage-tracked input file for Phase 3.

The existing Phase 1 split is preserved exactly. ``source_row_id`` is a
technical join key derived from the validated Phase 1 Parquet physical row
number and is never a model input.
``crossfit_fold`` is a deterministic feature-group fold used only by the
X-Learner to create out-of-fold pseudo-outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "phase3_models.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing Phase 3 input after all checks pass.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    input_config = config["input"]

    raw_file = resolve_project_path(input_config["raw_file"])
    existing_file = resolve_project_path(input_config["existing_partitioned_file"])
    output_file = resolve_project_path(input_config["phase3_file"])
    manifest_file = resolve_project_path(input_config["manifest_file"])
    temporary_file = output_file.with_suffix(".tmp.parquet")
    expected_rows = int(input_config["expected_rows"])

    for required in (raw_file, existing_file):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")
    if output_file.exists() and not args.force:
        raise FileExistsError(
            f"Phase 3 input already exists: {output_file}. Use --force to rebuild it."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if temporary_file.exists():
        temporary_file.unlink()

    features = config["features"]
    feature_strings = ", ".join(
        f"CAST({feature} AS VARCHAR)" for feature in features
    )
    crossfit_seed = input_config["crossfit_seed"].replace("'", "''")
    fold_count = int(config["crossfit_folds"])

    connection = duckdb.connect()
    connection.execute("SET threads = 8")

    build_sql = f"""
        COPY (
            SELECT
                CAST(file_row_number AS UBIGINT) AS source_row_id,
                {', '.join(features)},
                treatment,
                conversion,
                visit,
                data_partition,
                CAST(
                    md5_number_lower(concat_ws('|', '{crossfit_seed}', {feature_strings})) % {fold_count}
                    AS UTINYINT
                ) AS crossfit_fold
            FROM read_parquet(
                '{sql_path(existing_file)}',
                file_row_number = true
            )
        ) TO '{sql_path(temporary_file)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    connection.execute(build_sql)

    checks = connection.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            COUNT(DISTINCT source_row_id) AS unique_ids,
            MIN(source_row_id) AS minimum_id,
            MAX(source_row_id) AS maximum_id,
            COUNT_IF(data_partition NOT IN ('train', 'validation', 'test')) AS invalid_partition,
            COUNT_IF(crossfit_fold >= {fold_count}) AS invalid_fold,
            COUNT_IF(treatment NOT IN (0, 1)) AS invalid_treatment,
            COUNT_IF(visit NOT IN (0, 1)) AS invalid_visit,
            COUNT_IF(conversion NOT IN (0, 1)) AS invalid_conversion,
            COUNT_IF(conversion = 1 AND visit = 0) AS conversion_without_visit
        FROM read_parquet('{sql_path(temporary_file)}')
        """
    ).fetchone()

    assert checks[0] == expected_rows
    assert checks[1] == expected_rows
    assert checks[2] == 0 and checks[3] == expected_rows - 1
    assert all(value == 0 for value in checks[4:])

    reconciliation = connection.execute(
        f"""
        WITH new_counts AS (
            SELECT data_partition, treatment, COUNT(*) AS n,
                   SUM(visit) AS visit_positive,
                   SUM(conversion) AS conversion_positive
            FROM read_parquet('{sql_path(temporary_file)}')
            GROUP BY 1, 2
        ), old_counts AS (
            SELECT data_partition, treatment, COUNT(*) AS n,
                   SUM(visit) AS visit_positive,
                   SUM(conversion) AS conversion_positive
            FROM read_parquet('{sql_path(existing_file)}')
            GROUP BY 1, 2
        )
        SELECT COUNT(*)
        FROM new_counts AS n
        FULL OUTER JOIN old_counts AS o USING (data_partition, treatment)
        WHERE n.n IS DISTINCT FROM o.n
           OR n.visit_positive IS DISTINCT FROM o.visit_positive
           OR n.conversion_positive IS DISTINCT FROM o.conversion_positive
        """
    ).fetchone()[0]
    assert reconciliation == 0, "New Phase 3 input does not reconcile to Phase 1."

    summary_rows = connection.execute(
        f"""
        SELECT data_partition, treatment, COUNT(*) AS rows,
               SUM(visit) AS visit_positive,
               SUM(conversion) AS conversion_positive
        FROM read_parquet('{sql_path(temporary_file)}')
        GROUP BY 1, 2
        ORDER BY CASE data_partition WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END,
                 treatment
        """
    ).fetchall()
    connection.close()

    os.replace(temporary_file, output_file)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(raw_file),
        "raw_file_sha256": sha256_file(raw_file),
        "existing_partitioned_file": str(existing_file),
        "existing_partitioned_sha256": sha256_file(existing_file),
        "phase3_file": str(output_file),
        "phase3_file_sha256": sha256_file(output_file),
        "row_count": expected_rows,
        "source_row_id": {
            "minimum": 0,
            "maximum": expected_rows - 1,
            "unique": expected_rows,
            "usage": "technical join key only; prohibited as a model feature",
        },
        "partition_seed": input_config["partition_seed"],
        "crossfit_seed": input_config["crossfit_seed"],
        "crossfit_folds": fold_count,
        "reconciliation_mismatches": reconciliation,
        "partition_summary": [
            {
                "data_partition": row[0],
                "treatment": int(row[1]),
                "rows": int(row[2]),
                "visit_positive": int(row[3]),
                "conversion_positive": int(row[4]),
            }
            for row in summary_rows
        ],
    }
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Phase 3 input: {output_file}")
    print(f"Manifest: {manifest_file}")
    print(f"Rows: {expected_rows:,}")
    print("Reconciliation passed; the frozen Phase 1 split was preserved.")


if __name__ == "__main__":
    main()
