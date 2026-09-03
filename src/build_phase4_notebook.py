"""Build the reader-facing Phase 4 uplift-evaluation notebook."""

from pathlib import Path

import nbformat as nbf
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebook" / "06_uplift_model_evaluation.ipynb"


def result_summary() -> str:
    table_directory = PROJECT_ROOT / "reports" / "tables" / "phase4"
    recommendation_path = table_directory / "final_model_recommendations.csv"
    if recommendation_path.exists():
        recommendations = pd.read_csv(recommendation_path)
        return "\n".join(
            f"- **{row.outcome_name.title()}**: recommended uplift model is "
            f"`{row.recommended_uplift_model}`; deployment ranking is "
            f"`{row.deployment_ranking}` at Top {row.recommended_target_fraction:.0%}, "
            f"with {row.estimated_incremental_outcomes:,.1f} estimated incremental outcomes."
            for row in recommendations.itertuples(index=False)
        )
    bullets = []
    for outcome in ("visit", "conversion"):
        metric_path = table_directory / f"{outcome}_model_metrics.csv"
        uncertainty_path = table_directory / f"{outcome}_uncertainty.csv"
        ate_path = table_directory / f"{outcome}_ate_sanity.csv"
        if not (metric_path.exists() and uncertainty_path.exists() and ate_path.exists()):
            continue
        metrics = pd.read_csv(metric_path)
        uplift = metrics[metrics["model_name"].isin(["s_learner", "t_learner", "x_learner"])]
        best = uplift.sort_values("qini", ascending=False).iloc[0]
        uncertainty = pd.read_csv(uncertainty_path)
        difference = uncertainty[
            (uncertainty["model_name"] == best["model_name"])
            & (uncertainty["metric"] == "qini_minus_response")
        ].iloc[0]
        ate = pd.read_csv(ate_path).iloc[0]
        bullets.append(
            f"- **{outcome.title()}**: test ATE is {ate['test_ate']:.4%}. "
            f"The strongest uplift learner by Qini is `{best['model_name']}` "
            f"({best['qini']:,.1f}); its paired Qini difference versus response targeting is "
            f"{difference['estimate']:,.1f} "
            f"(95% CI {difference['ci_lower']:,.1f} to {difference['ci_upper']:,.1f})."
        )
    if not bullets:
        return "Formal Phase 4 results have not been generated yet."
    return "\n".join(bullets)


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    markdown(
        f"""
# Phase 4 — Uplift Evaluation and Targeting Strategy

## TL;DR

{result_summary()}

The test set was opened only after the model candidates, targeting rates, metrics, confidence
level, and bootstrap rules were frozen. Model choice still considers Top 10/20/30 gains and nearby
targeting rates rather than a single Qini point estimate.
"""
    ),
    markdown(
        r"""
## 1. Evaluation boundary

- Phase 3 models are frozen; they are not retrained or tuned here.
- Test outcomes are first opened in the `evaluate` stage.
- Users are ranked by continuous score; negative uplift scores are retained.
- Cumulative gain is evaluated every 5% from Top 5% through Top 100%.
- Visit group diagnostics use 20 groups; conversion uses 10 because conversion is rarer.
- Top 10%, 20%, and 30% are the main policy decision points.
- Calibration is a linear rescaling for interpretation and does not change ranking.
- Constant-ATE, segment-uplift, response-propensity, random, target-none, and target-all references
  are included.
- Phase 3 stability is recomputed with the current S/T/X implementation at an alternate seed.
- Visit-versus-conversion ranking correlation, policy overlap, and cross-policy gains are included.
- The added completion analyses are labeled exploratory because they were requested after the
  initial test review; Phase 3 models were not retuned.
"""
    ),
    markdown("## 2. Setup and saved artifacts"),
    code(
        r'''
from pathlib import Path
import json

import pandas as pd
from IPython.display import Image, Markdown, display


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "config").exists():
            return candidate
    raise FileNotFoundError("Could not locate the AdLift project root.")


PROJECT_ROOT = find_project_root(Path.cwd().resolve())
CONFIG_PATH = PROJECT_ROOT / "config" / "phase4_evaluation.json"
TABLE_DIRECTORY = PROJECT_ROOT / "reports" / "tables" / "phase4"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures" / "phase4"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "phase4"
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

print(f"Project root: {PROJECT_ROOT}")
print(f"Evaluation spec frozen: {(OUTPUT_ROOT / 'EVALUATION_SPEC.json').exists()}")
print(f"Test evaluation complete: {(OUTPUT_ROOT / 'TEST_EVALUATED.json').exists()}")
'''
    ),
    markdown("## 3. Handoff and test-set protection"),
    code(
        r'''
lock_path = OUTPUT_ROOT / "EVALUATION_SPEC.json"
if lock_path.exists():
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    display(pd.DataFrame([
        {
            "outcome": outcome,
            "test_rows": details["rows"],
            "phase3_run_hash": details["phase3_run_hash"],
            "test_outcomes_used_during_prepare": lock["test_outcomes_used"],
            "response_baseline_rows": lock["response_baselines"][outcome]["rows"],
        }
        for outcome, details in lock["handoff"].items()
    ]))
else:
    display(Markdown("Run the Phase 4 `prepare` command before evaluation."))
'''
    ),
    markdown("## 4. Baseline definitions and validation-only segment selection"),
    code(
        r'''
baseline_rows = []
segment_selection = []
for outcome in config["outcomes"]:
    metadata_path = OUTPUT_ROOT / "baselines" / outcome / "structural_metadata.json"
    selection_path = TABLE_DIRECTORY / f"{outcome}_segment_baseline_selection.csv"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        baseline_rows.append({
            "outcome": outcome,
            "constant_ate_train_validation": metadata["constant_ate_train_validation"],
            "selected_segment_feature": metadata["selected_feature"],
            "segment_bins": metadata["actual_bins"],
            "validation_qini": metadata["validation_qini"],
            "test_outcomes_used": metadata["test_outcomes_used"],
        })
    if selection_path.exists():
        candidates = pd.read_csv(selection_path)
        segment_selection.append(candidates.head(8))
if baseline_rows:
    display(pd.DataFrame(baseline_rows).round(6))
if segment_selection:
    display(pd.concat(segment_selection, ignore_index=True).round(4))
'''
    ),
    markdown("## 5. Top 100% ATE sanity check"),
    code(
        r'''
ate_tables = []
for outcome in config["outcomes"]:
    path = TABLE_DIRECTORY / f"{outcome}_ate_sanity.csv"
    if path.exists():
        ate_tables.append(pd.read_csv(path))
if ate_tables:
    ate_sanity = pd.concat(ate_tables, ignore_index=True)
    display(ate_sanity.round(8))
    assert (
        (ate_sanity["endpoint_gain"] - ate_sanity["expected_endpoint_gain"]).abs()
        < 1e-6
    ).all()
else:
    display(Markdown("ATE sanity results will appear after the formal evaluation."))
'''
    ),
    markdown("## 6. Observed uplift by score group"),
    code(
        r'''
group_tables = []
for outcome in config["outcomes"]:
    path = TABLE_DIRECTORY / f"{outcome}_group_effects.csv"
    if path.exists():
        group_tables.append(pd.read_csv(path))
if group_tables:
    group_effects = pd.concat(group_tables, ignore_index=True)
    display(group_effects.head(24).round(6))
    for outcome in config["outcomes"]:
        figure_path = FIGURE_DIRECTORY / f"{outcome}_group_uplift.png"
        if figure_path.exists():
            display(Image(filename=str(figure_path)))
else:
    display(Markdown("Group-level uplift results are not available yet."))
'''
    ),
    markdown(
        r"""
## 7. Cumulative gain, AUUC, and Qini

At targeting fraction $q$, cumulative gain is $G(q)=n(q)(\bar{Y}_1(q)-\bar{Y}_0(q))$.
This project calculates AUUC with trapezoidal integration over the saved 5% grid, including the
origin: $\mathrm{AUUC}=\int_0^1 G(q)dq$. The random reference is $qG(1)$, so
$\mathrm{Qini}=\mathrm{AUUC}-\tfrac{1}{2}G(1)$. Values are unnormalized counts of incremental
outcomes, not library-normalized scores.
"""
    ),
    code(
        r'''
metric_tables = []
for outcome in config["outcomes"]:
    path = TABLE_DIRECTORY / f"{outcome}_model_metrics.csv"
    if path.exists():
        metric_tables.append(pd.read_csv(path))
if metric_tables:
    metrics = pd.concat(metric_tables, ignore_index=True)
    display(metrics.sort_values(["outcome_name", "qini"], ascending=[True, False]).round(4))
    for outcome in config["outcomes"]:
        for suffix in ("cumulative_gain", "qini_curve"):
            figure_path = FIGURE_DIRECTORY / f"{outcome}_{suffix}.png"
            if figure_path.exists():
                display(Image(filename=str(figure_path)))
else:
    display(Markdown("Cumulative-gain metrics are not available yet."))
'''
    ),
    markdown("## 8. Uplift score calibration"),
    code(
        r'''
parameter_tables = []
for outcome in config["outcomes"]:
    path = TABLE_DIRECTORY / f"{outcome}_calibration_parameters.csv"
    if path.exists():
        parameter_tables.append(pd.read_csv(path))
if parameter_tables:
    calibration_parameters = pd.concat(parameter_tables, ignore_index=True)
    display(calibration_parameters.round(6))
    assert (~calibration_parameters["ranking_changed"]).all()
    for outcome in config["outcomes"]:
        figure_path = FIGURE_DIRECTORY / f"{outcome}_uplift_calibration.png"
        if figure_path.exists():
            display(Image(filename=str(figure_path)))
else:
    display(Markdown("Calibration results are not available yet."))
'''
    ),
    markdown("## 9. Paired-bootstrap uncertainty"),
    code(
        r'''
uncertainty_tables = []
for outcome in config["outcomes"]:
    path = TABLE_DIRECTORY / f"{outcome}_uncertainty.csv"
    if path.exists():
        uncertainty_tables.append(pd.read_csv(path))
if uncertainty_tables:
    uncertainty = pd.concat(uncertainty_tables, ignore_index=True)
    key_metrics = uncertainty[
        uncertainty["metric"].isin(
            ["qini", "qini_minus_response", "gain_at_10pct", "gain_at_20pct", "gain_at_30pct"]
        )
    ]
    display(key_metrics.round(4))
else:
    display(Markdown("Uncertainty intervals are not available yet."))
'''
    ),
    markdown("## 10. Top 10%, 20%, and 30% strategy comparison"),
    code(
        r'''
policy_tables = []
for outcome in config["outcomes"]:
    path = TABLE_DIRECTORY / f"{outcome}_policy_summary.csv"
    if path.exists():
        table = pd.read_csv(path)
        policy_tables.append(table[table["target_fraction"].isin(config["main_decision_rates"])])
if policy_tables:
    policy = pd.concat(policy_tables, ignore_index=True)
    display(
        policy[
            [
                "outcome_name", "model_name", "target_fraction", "selected_rows",
                "cumulative_uplift", "cumulative_gain", "gain_above_random"
            ]
        ].round(5)
    )
    for outcome in config["outcomes"]:
        figure_path = FIGURE_DIRECTORY / f"{outcome}_targeting_comparison.png"
        if figure_path.exists():
            display(Image(filename=str(figure_path)))
else:
    display(Markdown("Targeting-strategy results are not available yet."))
'''
    ),
    markdown("## 11. Phase 3 stability across model runs"),
    code(
        r'''
stability_tables = []
for outcome in config["outcomes"]:
    path = TABLE_DIRECTORY / f"{outcome}_phase3_stability.csv"
    if path.exists():
        stability_tables.append(pd.read_csv(path))
if stability_tables:
    stability = pd.concat(stability_tables, ignore_index=True)
    display(stability.round(5))
else:
    display(Markdown("Current-implementation stability results are not available yet."))
'''
    ),
    markdown("## 12. Final model and targeting recommendations"),
    code(
        r'''
recommendation_path = TABLE_DIRECTORY / "final_model_recommendations.csv"
comparison_path = TABLE_DIRECTORY / "final_model_comparison.csv"
if recommendation_path.exists() and comparison_path.exists():
    model_comparison = pd.read_csv(comparison_path)
    recommendations = pd.read_csv(recommendation_path)
    display(model_comparison.round(4))
    display(recommendations.round(4))
else:
    display(Markdown("Final recommendations will appear after both outcomes are evaluated."))
'''
    ),
    markdown(
        r"""
The selection rule is fixed in code: start with the uplift learner having the highest Qini, then
choose the simplest learner that is not significantly worse under the paired Qini comparison.
Deploy that uplift ranking only if its paired Qini difference versus response targeting has a 95%
confidence interval above zero. Otherwise retain response targeting for deployment. Among Top
10%, 20%, and 30%, choose the rate with the largest bootstrap lower confidence bound for
incremental outcomes.
"""
    ),
    markdown("## 13. Visit-versus-conversion targeting"),
    code(
        r'''
cross_path = TABLE_DIRECTORY / "visit_conversion_policy_comparison.csv"
overlap_figure = FIGURE_DIRECTORY / "visit_conversion_policy_overlap.png"
if cross_path.exists():
    cross_policy = pd.read_csv(cross_path)
    display(cross_policy.round(4))
if overlap_figure.exists():
    display(Image(filename=str(overlap_figure)))
'''
    ),
    markdown(
        r"""
## 14. Reproducible commands

```bash
# Freeze rules and prepare response baselines without test outcomes
.venv/bin/python src/phase4_uplift_evaluation.py prepare --outcome both

# Open test outcomes and run the frozen evaluation
.venv/bin/python src/phase4_uplift_evaluation.py evaluate --outcome both

# Rebuild and execute this reader-facing notebook
.venv/bin/python src/build_phase4_notebook.py
.venv/bin/jupyter nbconvert --execute --to notebook --inplace \
  notebook/06_uplift_model_evaluation.ipynb
```
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
