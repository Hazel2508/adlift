# Reproducibility and tests

## Two kinds of reproduction

The checked-in result tables describe the frozen August 25 analysis. A September 3 audit replayed saved test predictions and recovered every cumulative curve point, score-group effect, AUUC/Qini value, and the original 200-draw paired intervals. Raw and modeling-file hashes were verified against the historical manifest. A separate 2,000-draw sensitivity run is retained under `reports/audit/`.

`uv.lock` pins the environment verified during the repository restructuring. It does not recreate an unrecorded August training environment. New training can produce numerically different predictions across library versions or hardware. The existing model lineage files and test scores are local data artifacts, not downloadable repository contents.

With local frozen artifacts present:

```bash
uv run python scripts/evaluate_models.py evaluate
uv run python scripts/audit_evaluation.py
uv run python scripts/audit_evaluation.py --sensitivity
```

For a fresh download, `scripts/main.py` starts from the v2.1 gzip and creates the intermediate files, fitted models, frozen predictions, and result tables. It runs outside a notebook. During restructuring, the raw-to-model-input route was rebuilt in an isolated directory; every feature, outcome, split and technical ID matched the historical modeling input across all 13,979,592 rows. A visit smoke-training run, the full-data average-effect script, and the saved-prediction evaluation also executed successfully. Full retraining of both historical outcomes was not repeated during this audit.

## Why these tests exist

Tests protect mistakes that would change the causal conclusion:

| Contract | Failure it prevents |
|---|---|
| Feature allowlist | Exposure, outcome, split or row-ID leakage |
| Pseudo-outcome signs | Reversing treated/control effects |
| Null outcomes in test predictions | Accidental use of held-out labels during scoring |
| Ranking direction and deterministic ties | Selecting low-score records or source-order treatment blocks |
| Positive Qini on a known heterogeneous fixture | Incorrect gain/area calculation |
| 100% endpoint equals N times ATE | Wrong denominator or inconsistent model populations |
| Exact selected count | Incorrect targeting fraction |
| Paired bootstrap differences | Treating correlated strategy estimates as independent |
| Group effect recovery | Baseline segment aggregation errors |
| Diagnostic calibration | Accidentally implying that a diagnostic reranks users |

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src/adlift scripts tests
```

The original 11 tests pass in the refactored package. The test count is not a coverage percentage or evidence of independent policy validation.

## Notebook role

The three notebooks are executed reader-facing walkthroughs of saved results. They explain the data, average effects and targeting comparison. They are not pipeline dependencies. They can be read on GitHub without downloading the dataset.

## Platform and resources

The verified environment is Python 3.12 on macOS. Linux uses the same Python entry points but was not exercised in this audit. The data audit sets a 4 GB DuckDB memory limit and may spill into ignored local storage. Training uses the configured LightGBM thread count and materializes feature matrices; full-data execution needs more resources than the quick smoke check.
