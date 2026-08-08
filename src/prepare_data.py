from pathlib import Path
import time

import duckdb


# Locate project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "criteo-uplift-v2.1.csv.gz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "criteo-uplift-v2.1.parquet"
)


def sql_path(path: Path) -> str:
    """Escape a local path so it can safely be used in DuckDB SQL."""
    return str(path).replace("'", "''")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("Converting compressed CSV to Parquet...")

    start_time = time.time()
    connection = duckdb.connect()

#csv to parquet 
    connection.execute(
        f"""
        COPY (
            SELECT
                CAST(f0 AS DOUBLE) AS f0, 
                CAST(f1 AS DOUBLE) AS f1,
                CAST(f2 AS DOUBLE) AS f2,
                CAST(f3 AS DOUBLE) AS f3,
                CAST(f4 AS DOUBLE) AS f4,
                CAST(f5 AS DOUBLE) AS f5,
                CAST(f6 AS DOUBLE) AS f6,
                CAST(f7 AS DOUBLE) AS f7,
                CAST(f8 AS DOUBLE) AS f8,
                CAST(f9 AS DOUBLE) AS f9,
                CAST(f10 AS DOUBLE) AS f10,
                CAST(f11 AS DOUBLE) AS f11,
                CAST(treatment AS UTINYINT) AS treatment,
                CAST(conversion AS UTINYINT) AS conversion,
                CAST(visit AS UTINYINT) AS visit,
                CAST(exposure AS UTINYINT) AS exposure
            FROM read_csv_auto(
                '{sql_path(INPUT_FILE)}',
                header = true,
                compression = 'gzip'
            )
        )
        TO '{sql_path(OUTPUT_FILE)}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,  
            ROW_GROUP_SIZE 100000
        )
        """
    )
#Approximately 100,000 rows are stored in each Parquet row group, enabling efficient chunk-based querying and selective data reads.

    row_count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_path(OUTPUT_FILE)}')
        """
    ).fetchone()[0]

    connection.close()

    elapsed = time.time() - start_time
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print("Conversion complete.")
    print(f"Rows: {row_count:,}")
    print(f"Parquet size: {size_mb:,.1f} MB")
    print(f"Elapsed time: {elapsed:,.1f} seconds")

    expected_rows = 13_979_592

    if row_count != expected_rows:
        raise ValueError(
            f"Unexpected row count: {row_count:,}; "
            f"expected {expected_rows:,}"
        )

    print("Row-count validation passed.")


if __name__ == "__main__":
    main()