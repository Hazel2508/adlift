"""Build the reader-facing, boundary-correct Phase 3 notebook."""

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebook" / "05_individual_uplift_modeling.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown(
        r"""
# Phase 3 — Individual Uplift Modeling

## TL;DR

Phase 3 trains S-, T-, and X-Learners for `visit` and `conversion`, then produces one continuous
uplift score per user. Train fits model parameters, validation supports tuning and early stopping,
and frozen models score test features without reading test outcomes.

Phase 3 does not calculate ATE, observed uplift by ranked group, cumulative gain, AUUC, Qini,
uplift calibration, a final model, or a targeting cutoff. Those tasks begin in Phase 4.
"""
    ),
    markdown(
        r"""
## 1. Scope and formulas

For a feature profile $x$:

$$
\tau(x)=P(Y=1\mid X=x,T=1)-P(Y=1\mid X=x,T=0)
$$

| Learner | Phase 3 implementation |
| :--- | :--- |
| S-Learner | Fit one model on $(X,T)$ and calculate $\hat\mu(X,1)-\hat\mu(X,0)$. |
| T-Learner | Fit separate treatment/control outcome models and calculate $\hat\mu_1(X)-\hat\mu_0(X)$. |
| X-Learner | Build $D=Y-\hat\mu_0(X)$ for treatment, $D=\hat\mu_1(X)-Y$ for control, combine the rows, and fit one $g(X)\rightarrow D$. |

Only `f0`–`f11` may enter the model. `exposure`, outcomes from the other task, identifiers, split
fields, and test outcomes are prohibited.
"""
    ),
    markdown("## 2. Setup"),
    code(
        r'''
from pathlib import Path
import json

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import Markdown, display


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "config").exists():
            return candidate
    raise FileNotFoundError("Could not locate the AdLift project root.")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
CONFIG_PATH = PROJECT_ROOT / "config" / "phase3_models.json"
PHASE3_INPUT = PROJECT_ROOT / "data" / "processed" / "phase3" / "criteo-uplift-v2.1-phase3.parquet"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "phase3" / "manifest.json"
TABLE_DIRECTORY = PROJECT_ROOT / "reports" / "tables" / "phase3"
EXPECTED_MODELS = {"s_learner", "t_learner", "x_learner"}

config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
print(f"Project root: {PROJECT_ROOT}")
print(f"Input ready: {PHASE3_INPUT.exists()}")
'''
    ),
    markdown("## 3. Data boundary"),
    code(
        r'''
manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
connection = duckdb.connect()
input_path = str(PHASE3_INPUT).replace("'", "''")
schema = connection.execute(
    f"DESCRIBE SELECT * FROM read_parquet('{input_path}')"
).df()
expected_columns = {
    "source_row_id", "treatment", "visit", "conversion",
    "data_partition", "crossfit_fold", *config["features"]
}
assert set(schema["column_name"]) == expected_columns
assert "exposure" not in expected_columns

# Only train and validation are summarized here. Test outcomes are deliberately not queried.
support = connection.execute(
    f"""
    SELECT data_partition, treatment, COUNT(*) AS rows,
           SUM(visit) AS visit_positive,
           SUM(conversion) AS conversion_positive
    FROM read_parquet('{input_path}')
    WHERE data_partition IN ('train', 'validation')
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
).df()
connection.close()
display(support)
print(f"Source SHA-256: {manifest['raw_file_sha256']}")
print("Boundary check passed: test outcomes were not queried.")
'''
    ),
    markdown("## 4. Available Phase 3 run"),
    code(
        r'''
def compatible_run_root(outcome: str):
    candidates = [
        ("full", PROJECT_ROOT / "data" / "processed" / "phase3" / "models" / outcome),
        ("smoke", PROJECT_ROOT / "data" / "processed" / "phase3" / "smoke" / "models" / outcome),
    ]
    for mode, model_dir in candidates:
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            scope = set(metadata.get("lineage_spec", {}).get("scope", []))
            if scope == EXPECTED_MODELS:
                return mode, model_dir, metadata
    return None, None, None


run_rows = []
for outcome in config["outcomes"]:
    mode, model_dir, metadata = compatible_run_root(outcome)
    run_rows.append({
        "outcome": outcome,
        "compatible_run": mode or "none",
        "runtime_seconds": None if metadata is None else metadata["runtime_seconds"],
        "test_outcomes_used": None if metadata is None else metadata["test_outcomes_used"],
    })
display(pd.DataFrame(run_rows))
'''
    ),
    markdown("## 5. Validation training diagnostics"),
    code(
        r'''
tables = []
for outcome in config["outcomes"]:
    mode, _, _ = compatible_run_root(outcome)
    if mode:
        path = TABLE_DIRECTORY / f"{mode}_{outcome}_model_diagnostics.csv"
        if path.exists():
            tables.append(pd.read_csv(path).assign(outcome=outcome, run_mode=mode))
if tables:
    diagnostics = pd.concat(tables, ignore_index=True)
    display(diagnostics.round(6))
    if set(diagnostics["run_mode"]) == {"smoke"}:
        display(Markdown("**Smoke outputs verify engineering only; they are not final model results.**"))
else:
    display(Markdown("No compatible Phase 3 diagnostics exist yet. Run the smoke tests below."))
'''
    ),
    markdown("## 6. Prediction artifact QA"),
    code(
        r'''
qa_rows = []
for outcome in config["outcomes"]:
    mode, _, _ = compatible_run_root(outcome)
    if not mode:
        qa_rows.append({"outcome": outcome, "mode": "none", "status": "missing"})
        continue
    root = PROJECT_ROOT / "data" / "processed" / "phase3"
    if mode == "smoke":
        root = root / "smoke"
    prediction_dir = root / "predictions" / "validation" / outcome
    files = sorted(prediction_dir.glob("*.parquet"))
    names = {path.stem for path in files}
    schemas_ok = all(
        pq.read_schema(path).names == [
            "source_row_id", "split", "outcome_name", "model_name", "treatment",
            "observed_outcome", "uplift_hat", "model_config_hash", "source_file_sha256"
        ]
        for path in files
    )
    finite = all(
        pd.Series(pq.read_table(path, columns=["uplift_hat"])["uplift_hat"].to_numpy()).notna().all()
        for path in files
    )
    qa_rows.append({
        "outcome": outcome,
        "mode": mode,
        "models_exact": names == EXPECTED_MODELS,
        "schema_ok": schemas_ok,
        "uplift_present": finite,
        "files": ", ".join(sorted(names)),
    })
display(pd.DataFrame(qa_rows))
'''
    ),
    markdown(
        r"""
## 7. Reproducible commands

```bash
# Engineering smoke tests
.venv/bin/python src/phase3_uplift_pipeline.py develop --outcome visit --mode smoke
.venv/bin/python src/phase3_uplift_pipeline.py develop --outcome conversion --mode smoke

# Full training and freeze, only after smoke passes
.venv/bin/python src/phase3_uplift_pipeline.py develop --outcome both --mode full --freeze

# Test-feature scoring without test outcomes
.venv/bin/python src/phase3_uplift_pipeline.py score-test --outcome both
```
"""
    ),
    markdown(
        r"""
## 8. Phase 4 handoff

Phase 4 will join the frozen test scores to the randomized test outcomes by `source_row_id`. It will
then perform 5% ranked grouping, observed treatment-control differences, cumulative gain,
AUUC/Qini, uplift-score calibration, uncertainty analysis, model comparison, and cutoff selection.

No conclusion about which learner is best is made in this notebook.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
