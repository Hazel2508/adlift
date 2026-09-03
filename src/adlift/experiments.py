"""Average assignment effects, uncertainty, and released-sample diagnostics."""

from pathlib import Path
import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from .paths import PROJECT_ROOT

EXPECTED_ROW_COUNT = 13979592
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
Z_CRITICAL = 1.96
BOOTSTRAP_SEED = 20260809
INITIAL_BOOTSTRAP_REPLICATES = 10000
ESCALATED_BOOTSTRAP_REPLICATES = 50000
CONTROL_COLOR = "#98A2B3"
TREATMENT_COLOR = "#1F5A94"
REFERENCE_COLOR = "#475467"
SECONDARY_COLOR = "#D97706"


def calculate_effect_row(
    outcome: str, control_n: int, treatment_n: int, control_positive: int, treatment_positive: int
) -> dict:
    """Calculate Phase 2 effect measures and analytical intervals."""
    control_rate = control_positive / control_n
    treatment_rate = treatment_positive / treatment_n
    absolute_lift = treatment_rate - control_rate
    relative_lift = absolute_lift / control_rate
    absolute_se = np.sqrt(
        treatment_rate * (1 - treatment_rate) / treatment_n
        + control_rate * (1 - control_rate) / control_n
    )
    absolute_ci_lower = absolute_lift - Z_CRITICAL * absolute_se
    absolute_ci_upper = absolute_lift + Z_CRITICAL * absolute_se
    log_risk_ratio = np.log(treatment_rate / control_rate)
    log_risk_ratio_se = np.sqrt(
        1 / treatment_positive - 1 / treatment_n + 1 / control_positive - 1 / control_n
    )
    relative_ci_lower = np.exp(log_risk_ratio - Z_CRITICAL * log_risk_ratio_se) - 1
    relative_ci_upper = np.exp(log_risk_ratio + Z_CRITICAL * log_risk_ratio_se) - 1
    total_n = control_n + treatment_n
    return {
        "outcome": outcome,
        "control_n": control_n,
        "treatment_n": treatment_n,
        "control_positive": control_positive,
        "treatment_positive": treatment_positive,
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_lift": absolute_lift,
        "absolute_ci_lower": absolute_ci_lower,
        "absolute_ci_upper": absolute_ci_upper,
        "relative_lift": relative_lift,
        "relative_ci_lower": relative_ci_lower,
        "relative_ci_upper": relative_ci_upper,
        "incremental_treated": treatment_n * absolute_lift,
        "incremental_all": total_n * absolute_lift,
    }


def run_binary_bootstrap(effect_row: pd.Series, replicates: int, seed: int) -> pd.DataFrame:
    """Run an efficient arm-stratified bootstrap for one binary outcome."""
    random_generator = np.random.default_rng(seed)
    treatment_positive = random_generator.binomial(
        int(effect_row["treatment_n"]), float(effect_row["treatment_rate"]), size=replicates
    )
    control_positive = random_generator.binomial(
        int(effect_row["control_n"]), float(effect_row["control_rate"]), size=replicates
    )
    treatment_rate = treatment_positive / int(effect_row["treatment_n"])
    control_rate = control_positive / int(effect_row["control_n"])
    absolute_lift = treatment_rate - control_rate
    relative_lift = np.divide(
        absolute_lift,
        control_rate,
        out=np.full(replicates, np.nan, dtype=float),
        where=control_rate > 0,
    )
    return pd.DataFrame({"absolute_lift": absolute_lift, "relative_lift": relative_lift})


def percentile_interval(values: pd.Series) -> np.ndarray:
    """Return the 95% percentile interval after removing undefined values."""
    clean_values = values.dropna().to_numpy()
    return np.quantile(clean_values, [0.025, 0.975])


def build_bootstrap_intervals(samples_by_outcome: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for outcome, samples in samples_by_outcome.items():
        for measure in ["absolute_lift", "relative_lift"]:
            lower, upper = percentile_interval(samples[measure])
            rows.append(
                {
                    "outcome": outcome,
                    "measure": measure,
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "bootstrap_interval_width": upper - lower,
                    "undefined_replicates": int(samples[measure].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def stability_check(
    samples_by_outcome: dict[str, pd.DataFrame], full_interval_table: pd.DataFrame
) -> pd.DataFrame:
    """Compare the first and second halves against the full interval width."""
    rows = []
    for outcome, samples in samples_by_outcome.items():
        split_index = len(samples) // 2
        for measure in ["absolute_lift", "relative_lift"]:
            first_ci = percentile_interval(samples.iloc[:split_index][measure])
            second_ci = percentile_interval(samples.iloc[split_index:][measure])
            full_row = full_interval_table.loc[
                (full_interval_table["outcome"] == outcome)
                & (full_interval_table["measure"] == measure)
            ].iloc[0]
            full_width = float(full_row["bootstrap_interval_width"])
            lower_difference_ratio = abs(first_ci[0] - second_ci[0]) / full_width
            upper_difference_ratio = abs(first_ci[1] - second_ci[1]) / full_width
            passed = max(lower_difference_ratio, upper_difference_ratio) <= 0.02
            rows.append(
                {
                    "outcome": outcome,
                    "measure": measure,
                    "replicates": len(samples),
                    "first_half_lower": first_ci[0],
                    "second_half_lower": second_ci[0],
                    "lower_difference_as_width": lower_difference_ratio,
                    "first_half_upper": first_ci[1],
                    "second_half_upper": second_ci[1],
                    "upper_difference_as_width": upper_difference_ratio,
                    "stability_pass": passed,
                }
            )
    return pd.DataFrame(rows)


def two_sample_proportion_z_test(
    treatment_positive: int, treatment_n: int, control_positive: int, control_n: int
) -> tuple[float, float, float, str]:
    """Return the pooled two-sided z statistic and stable p-value reporting."""
    treatment_rate = treatment_positive / treatment_n
    control_rate = control_positive / control_n
    pooled_rate = (treatment_positive + control_positive) / (treatment_n + control_n)
    null_standard_error = np.sqrt(
        pooled_rate * (1 - pooled_rate) * (1 / treatment_n + 1 / control_n)
    )
    z_statistic = (treatment_rate - control_rate) / null_standard_error
    log_p_value = np.log(2) + norm.logsf(abs(z_statistic))
    log10_p_value = log_p_value / np.log(10)
    p_value = np.exp(log_p_value) if log_p_value > np.log(np.finfo(float).tiny) else 0.0
    p_value_display = f"{p_value:.3e}" if p_value > 0 else f"<1e{int(np.floor(log10_p_value)) + 1}"
    return (z_statistic, p_value, log10_p_value, p_value_display)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm step-down family-wise error adjustment."""
    p_array = p_values.to_numpy(dtype=float)
    order = np.argsort(p_array)
    adjusted_sorted = np.empty(len(p_array), dtype=float)
    running_maximum = 0.0
    for rank, original_index in enumerate(order):
        multiplier = len(p_array) - rank
        candidate = min(1.0, multiplier * p_array[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[rank] = running_maximum
    adjusted = np.empty(len(p_array), dtype=float)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=p_values.index)


def run_incrementality(root: Path = PROJECT_ROOT) -> None:
    PARQUET_PATH = root / "data/processed/criteo-uplift-v2.1.parquet"
    FIGURE_DIRECTORY = root / "results/figures/phase2"
    TABLE_DIRECTORY = root / "results/tables"
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PARQUET_SQL_PATH = str(PARQUET_PATH).replace("'", "''")
    connection = duckdb.connect()
    schema_table = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{PARQUET_SQL_PATH}')"
    ).df()
    expected_columns = {
        *[f"f{i}" for i in range(12)],
        "treatment",
        "conversion",
        "visit",
        "exposure",
    }
    input_checks = connection.execute(
        f"\n    SELECT\n        COUNT(*) AS row_count,\n        COUNT_IF(treatment IS NULL OR exposure IS NULL OR visit IS NULL OR conversion IS NULL)\n            AS rows_with_null_experiment_fields,\n        COUNT_IF(treatment NOT IN (0, 1)) AS invalid_treatment_rows,\n        COUNT_IF(exposure NOT IN (0, 1)) AS invalid_exposure_rows,\n        COUNT_IF(visit NOT IN (0, 1)) AS invalid_visit_rows,\n        COUNT_IF(conversion NOT IN (0, 1)) AS invalid_conversion_rows,\n        COUNT_IF(treatment = 0 AND exposure = 1) AS exposed_control_rows,\n        COUNT_IF(conversion = 1 AND visit = 0) AS conversion_without_visit_rows\n    FROM read_parquet('{PARQUET_SQL_PATH}')\n    "
    ).df()
    observed_columns = set(schema_table["column_name"])
    check_row = input_checks.iloc[0]
    assert observed_columns == expected_columns
    assert int(check_row["row_count"]) == EXPECTED_ROW_COUNT
    assert int(check_row["rows_with_null_experiment_fields"]) == 0
    assert int(check_row["invalid_treatment_rows"]) == 0
    assert int(check_row["invalid_exposure_rows"]) == 0
    assert int(check_row["invalid_visit_rows"]) == 0
    assert int(check_row["invalid_conversion_rows"]) == 0
    assert int(check_row["exposed_control_rows"]) == 0
    assert int(check_row["conversion_without_visit_rows"]) == 0
    print("Input validation passed.")
    arm_summary = connection.execute(
        f"\n    SELECT\n        treatment,\n        COUNT(*) AS sample_size,\n        SUM(exposure) AS exposure_positive,\n        AVG(exposure) AS exposure_rate,\n        SUM(visit) AS visit_positive,\n        AVG(visit) AS visit_rate,\n        SUM(conversion) AS conversion_positive,\n        AVG(conversion) AS conversion_rate\n    FROM read_parquet('{PARQUET_SQL_PATH}')\n    GROUP BY treatment\n    ORDER BY treatment\n    "
    ).df()
    assert set(arm_summary["treatment"]) == {0, 1}
    assert int(arm_summary["sample_size"].sum()) == EXPECTED_ROW_COUNT
    arm_summary["arm"] = arm_summary["treatment"].map({0: "Control", 1: "Treatment"})
    arm_summary = arm_summary[
        [
            "arm",
            "treatment",
            "sample_size",
            "exposure_positive",
            "exposure_rate",
            "visit_positive",
            "visit_rate",
            "conversion_positive",
            "conversion_rate",
        ]
    ]
    control_row = arm_summary.loc[arm_summary["treatment"] == 0].iloc[0]
    treatment_row = arm_summary.loc[arm_summary["treatment"] == 1].iloc[0]
    effect_rows = []
    for outcome in ["visit", "conversion"]:
        effect_rows.append(
            calculate_effect_row(
                outcome=outcome,
                control_n=int(control_row["sample_size"]),
                treatment_n=int(treatment_row["sample_size"]),
                control_positive=int(control_row[f"{outcome}_positive"]),
                treatment_positive=int(treatment_row[f"{outcome}_positive"]),
            )
        )
    effect_results = pd.DataFrame(effect_rows)
    assert (effect_results[["control_rate", "treatment_rate"]] > 0).all().all()
    assert (effect_results[["control_rate", "treatment_rate"]] < 1).all().all()
    assert (effect_results["absolute_lift"] > 0).all()
    initial_bootstrap_samples = {}
    for outcome_index, effect_row in effect_results.set_index("outcome").iterrows():
        seed_offset = 0 if outcome_index == "visit" else 1
        initial_bootstrap_samples[outcome_index] = run_binary_bootstrap(
            effect_row, replicates=INITIAL_BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED + seed_offset
        )
    initial_bootstrap_intervals = build_bootstrap_intervals(initial_bootstrap_samples)
    initial_stability = stability_check(initial_bootstrap_samples, initial_bootstrap_intervals)
    requires_escalation = not initial_stability["stability_pass"].all()
    if requires_escalation:
        print(
            "At least one 10,000-replicate endpoint comparison exceeded the 2% project threshold. Escalating all final intervals to 50,000 replicates."
        )
        final_replicates = ESCALATED_BOOTSTRAP_REPLICATES
        final_bootstrap_samples = {}
        for outcome_index, effect_row in effect_results.set_index("outcome").iterrows():
            seed_offset = 0 if outcome_index == "visit" else 1
            final_bootstrap_samples[outcome_index] = run_binary_bootstrap(
                effect_row, replicates=final_replicates, seed=BOOTSTRAP_SEED + seed_offset
            )
    else:
        print("All 10,000-replicate stability comparisons passed.")
        final_replicates = INITIAL_BOOTSTRAP_REPLICATES
        final_bootstrap_samples = initial_bootstrap_samples
    final_bootstrap_intervals = build_bootstrap_intervals(final_bootstrap_samples)
    final_stability = stability_check(final_bootstrap_samples, final_bootstrap_intervals)
    assert final_stability["stability_pass"].all(), (
        "Final bootstrap stability did not meet the project threshold."
    )
    analytical_interval_rows = []
    for _, row in effect_results.iterrows():
        analytical_interval_rows.extend(
            [
                {
                    "outcome": row["outcome"],
                    "measure": "absolute_lift",
                    "analytical_ci_lower": row["absolute_ci_lower"],
                    "analytical_ci_upper": row["absolute_ci_upper"],
                },
                {
                    "outcome": row["outcome"],
                    "measure": "relative_lift",
                    "analytical_ci_lower": row["relative_ci_lower"],
                    "analytical_ci_upper": row["relative_ci_upper"],
                },
            ]
        )
    analytical_intervals = pd.DataFrame(analytical_interval_rows)
    interval_agreement = analytical_intervals.merge(
        final_bootstrap_intervals, on=["outcome", "measure"], validate="one_to_one"
    )
    interval_agreement["analytical_interval_width"] = (
        interval_agreement["analytical_ci_upper"] - interval_agreement["analytical_ci_lower"]
    )
    interval_agreement["lower_endpoint_difference_ratio"] = (
        interval_agreement["analytical_ci_lower"] - interval_agreement["bootstrap_ci_lower"]
    ).abs() / interval_agreement["analytical_interval_width"]
    interval_agreement["upper_endpoint_difference_ratio"] = (
        interval_agreement["analytical_ci_upper"] - interval_agreement["bootstrap_ci_upper"]
    ).abs() / interval_agreement["analytical_interval_width"]
    interval_agreement["width_difference_ratio"] = (
        interval_agreement["bootstrap_interval_width"]
        - interval_agreement["analytical_interval_width"]
    ).abs() / interval_agreement["analytical_interval_width"]
    interval_agreement["agreement_pass"] = (
        interval_agreement[
            [
                "lower_endpoint_difference_ratio",
                "upper_endpoint_difference_ratio",
                "width_difference_ratio",
            ]
        ].max(axis=1)
        <= 0.1
    )
    assert interval_agreement["agreement_pass"].all(), (
        "Analytical and bootstrap intervals require implementation review."
    )
    test_rows = []
    for _, row in effect_results.iterrows():
        z_statistic, raw_p_value, raw_log10_p_value, raw_p_value_display = (
            two_sample_proportion_z_test(
                treatment_positive=int(row["treatment_positive"]),
                treatment_n=int(row["treatment_n"]),
                control_positive=int(row["control_positive"]),
                control_n=int(row["control_n"]),
            )
        )
        test_rows.append(
            {
                "outcome": row["outcome"],
                "z_statistic": z_statistic,
                "raw_p_value": raw_p_value,
                "raw_log10_p_value": raw_log10_p_value,
                "raw_p_value_display": raw_p_value_display,
            }
        )
    hypothesis_tests = pd.DataFrame(test_rows)
    hypothesis_tests["holm_adjusted_p_value"] = holm_adjust(hypothesis_tests["raw_p_value"])
    hypothesis_tests["holm_p_value_display"] = hypothesis_tests.apply(
        lambda row: (
            f"{row['holm_adjusted_p_value']:.3e}"
            if row["holm_adjusted_p_value"] > 0
            else row["raw_p_value_display"]
        ),
        axis=1,
    )
    hypothesis_tests["reject_at_0_05"] = hypothesis_tests["holm_adjusted_p_value"] < ALPHA
    assert hypothesis_tests["reject_at_0_05"].all()
    exposure_delivery = connection.execute(
        f"\n    SELECT\n        CASE\n            WHEN treatment = 0 THEN 'Control'\n            WHEN treatment = 1 AND exposure = 0 THEN 'Treatment assigned, unexposed'\n            WHEN treatment = 1 AND exposure = 1 THEN 'Treatment assigned, exposed'\n        END AS delivery_group,\n        COUNT(*) AS sample_size,\n        AVG(exposure) AS exposure_rate,\n        AVG(visit) AS visit_rate,\n        AVG(conversion) AS conversion_rate\n    FROM read_parquet('{PARQUET_SQL_PATH}')\n    GROUP BY 1\n    ORDER BY CASE delivery_group\n        WHEN 'Control' THEN 1\n        WHEN 'Treatment assigned, unexposed' THEN 2\n        WHEN 'Treatment assigned, exposed' THEN 3\n    END\n    "
    ).df()
    exposure_first_stage = float(treatment_row["exposure_rate"] - control_row["exposure_rate"])
    complier_effects = effect_results[["outcome", "absolute_lift"]].copy()
    complier_effects["exposure_first_stage"] = exposure_first_stage
    complier_effects["wald_complier_ratio"] = (
        complier_effects["absolute_lift"] / complier_effects["exposure_first_stage"]
    )
    rate_plot_rows = []
    for _, row in effect_results.iterrows():
        for arm in ["Control", "Treatment"]:
            rate = row["control_rate"] if arm == "Control" else row["treatment_rate"]
            n = row["control_n"] if arm == "Control" else row["treatment_n"]
            standard_error = np.sqrt(rate * (1 - rate) / n)
            rate_plot_rows.append(
                {
                    "outcome": row["outcome"],
                    "arm": arm,
                    "rate": rate,
                    "ci_lower": max(0, rate - Z_CRITICAL * standard_error),
                    "ci_upper": min(1, rate + Z_CRITICAL * standard_error),
                }
            )
    rate_plot_data = pd.DataFrame(rate_plot_rows)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, outcome in zip(axes, ["visit", "conversion"]):
        subset = rate_plot_data.loc[rate_plot_data["outcome"] == outcome].copy()
        x_positions = np.arange(len(subset))
        colors = [CONTROL_COLOR, TREATMENT_COLOR]
        lower_errors = subset["rate"] - subset["ci_lower"]
        upper_errors = subset["ci_upper"] - subset["rate"]
        axis.errorbar(
            x_positions,
            subset["rate"],
            yerr=np.vstack([lower_errors, upper_errors]),
            fmt="none",
            ecolor=REFERENCE_COLOR,
            capsize=5,
            linewidth=1.5,
            zorder=2,
        )
        axis.scatter(x_positions, subset["rate"], s=90, c=colors, edgecolor="#344054", zorder=3)
        for x_position, rate in zip(x_positions, subset["rate"]):
            axis.annotate(
                f"{rate:.4%}",
                (x_position, rate),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        axis.set_xticks(x_positions, subset["arm"])
        axis.set_ylabel("Outcome rate")
        axis.set_title(f"{outcome.title()} rate", loc="left", fontweight="bold")
        axis.grid(axis="y", color="#EAECF0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.2%}"))
    figure.suptitle(
        "Outcome Rates by Randomized Assignment", x=0.08, ha="left", fontsize=15, fontweight="bold"
    )
    figure.text(
        0.08,
        0.91,
        "Points show arm rates; error bars show 95% normal-approximation confidence intervals.",
        color="#475467",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.88])
    arm_rates_path = FIGURE_DIRECTORY / "arm_rates.png"
    figure.savefig(arm_rates_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {arm_rates_path}")
    absolute_plot = effect_results.sort_values("absolute_lift").copy()
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    y_positions = np.arange(len(absolute_plot))
    axis.errorbar(
        absolute_plot["absolute_lift"] * 100,
        y_positions,
        xerr=np.vstack(
            [
                (absolute_plot["absolute_lift"] - absolute_plot["absolute_ci_lower"]) * 100,
                (absolute_plot["absolute_ci_upper"] - absolute_plot["absolute_lift"]) * 100,
            ]
        ),
        fmt="o",
        color=TREATMENT_COLOR,
        ecolor=REFERENCE_COLOR,
        capsize=5,
        markersize=8,
    )
    axis.axvline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2)
    axis.set_yticks(y_positions, absolute_plot["outcome"].str.title())
    axis.set_xlabel("Treatment − control difference (percentage points)")
    axis.grid(axis="x", color="#EAECF0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    for _, row in absolute_plot.iterrows():
        y_position = absolute_plot.index.get_loc(row.name)
        axis.annotate(
            f"{row['absolute_lift'] * 100:.4f} pp",
            (row["absolute_ci_upper"] * 100, y_position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    figure.suptitle("Absolute Lift", x=0.12, y=0.98, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.12,
        0.91,
        "Dots show ATE estimates; error bars show analytical 95% confidence intervals.",
        color="#475467",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.84])
    absolute_lift_path = FIGURE_DIRECTORY / "absolute_lift.png"
    figure.savefig(absolute_lift_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {absolute_lift_path}")
    relative_bootstrap = final_bootstrap_intervals.loc[
        final_bootstrap_intervals["measure"] == "relative_lift",
        ["outcome", "bootstrap_ci_lower", "bootstrap_ci_upper"],
    ]
    relative_plot = effect_results.merge(relative_bootstrap, on="outcome", validate="one_to_one")
    relative_plot = relative_plot.sort_values("relative_lift")
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    y_positions = np.arange(len(relative_plot))
    axis.errorbar(
        relative_plot["relative_lift"] * 100,
        y_positions,
        xerr=np.vstack(
            [
                (relative_plot["relative_lift"] - relative_plot["bootstrap_ci_lower"]) * 100,
                (relative_plot["bootstrap_ci_upper"] - relative_plot["relative_lift"]) * 100,
            ]
        ),
        fmt="o",
        color=SECONDARY_COLOR,
        ecolor=REFERENCE_COLOR,
        capsize=5,
        markersize=8,
    )
    axis.axvline(0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2)
    axis.set_yticks(y_positions, relative_plot["outcome"].str.title())
    axis.set_xlabel("Relative lift versus control baseline")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
    axis.grid(axis="x", color="#EAECF0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    for y_position, (_, row) in enumerate(relative_plot.iterrows()):
        axis.annotate(
            f"{row['relative_lift']:.2%} (baseline {row['control_rate']:.4%})",
            (row["bootstrap_ci_upper"] * 100, y_position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
        )
    figure.suptitle("Relative Lift", x=0.12, y=0.98, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.12,
        0.91,
        f"Final {final_replicates:,}-replicate bootstrap intervals; control baselines appear beside each estimate.",
        color="#475467",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.84])
    relative_lift_path = FIGURE_DIRECTORY / "relative_lift.png"
    figure.savefig(relative_lift_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {relative_lift_path}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (_, row) in zip(axes, effect_results.iterrows()):
        labels = ["Observed treatment arm", "Treat-all released sample"]
        values = [row["incremental_treated"], row["incremental_all"]]
        bars = axis.barh(
            labels, values, color=[TREATMENT_COLOR, SECONDARY_COLOR], edgecolor="#344054"
        )
        axis.set_title(row["outcome"].title(), loc="left", fontweight="bold")
        axis.set_xlabel("Estimated incremental outcomes")
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
        axis.grid(axis="x", color="#EAECF0", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.invert_yaxis()
        axis.bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=5)
        axis.set_xlim(0, max(values) * 1.18)
    figure.suptitle("Incremental Outcome Scale", x=0.08, ha="left", fontsize=15, fontweight="bold")
    figure.text(
        0.08,
        0.91,
        "Treat-all values are scenario extrapolations to the released sample, not capacity or revenue forecasts.",
        color="#475467",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.88])
    incremental_path = FIGURE_DIRECTORY / "incremental_outcomes.png"
    figure.savefig(incremental_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {incremental_path}")
    delivery_counts = pd.DataFrame(
        {
            "stage": ["Treatment assigned", "Exposed", "Visited", "Converted"],
            "count": [
                int(treatment_row["sample_size"]),
                int(treatment_row["exposure_positive"]),
                int(treatment_row["visit_positive"]),
                int(treatment_row["conversion_positive"]),
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.barh(
        delivery_counts["stage"],
        delivery_counts["count"],
        color=[TREATMENT_COLOR, "#4C78A8", SECONDARY_COLOR, "#E9A23B"],
        edgecolor="#344054",
    )
    axis.invert_yaxis()
    axis.set_xlabel("Number of treatment-assigned records")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value / 1000000:.0f}M"))
    axis.grid(axis="x", color="#EAECF0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.bar_label(bars, labels=[f"{value:,.0f}" for value in delivery_counts["count"]], padding=5)
    axis.set_xlim(0, delivery_counts["count"].max() * 1.14)
    figure.suptitle(
        "Experiment Delivery Stages", x=0.18, y=0.98, ha="left", fontsize=15, fontweight="bold"
    )
    figure.text(
        0.18,
        0.91,
        "Descriptive stage counts; visit is not restricted to exposed records and the stages are not a causal funnel.",
        color="#475467",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.84])
    delivery_funnel_path = FIGURE_DIRECTORY / "experiment_delivery_funnel.png"
    figure.savefig(delivery_funnel_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {delivery_funnel_path}")
    absolute_bootstrap = final_bootstrap_intervals.loc[
        final_bootstrap_intervals["measure"] == "absolute_lift",
        ["outcome", "bootstrap_ci_lower", "bootstrap_ci_upper"],
    ].rename(
        columns={
            "bootstrap_ci_lower": "absolute_bootstrap_ci_lower",
            "bootstrap_ci_upper": "absolute_bootstrap_ci_upper",
        }
    )
    relative_bootstrap = final_bootstrap_intervals.loc[
        final_bootstrap_intervals["measure"] == "relative_lift",
        ["outcome", "bootstrap_ci_lower", "bootstrap_ci_upper"],
    ].rename(
        columns={
            "bootstrap_ci_lower": "relative_bootstrap_ci_lower",
            "bootstrap_ci_upper": "relative_bootstrap_ci_upper",
        }
    )
    final_results = (
        effect_results.merge(absolute_bootstrap, on="outcome", validate="one_to_one")
        .merge(relative_bootstrap, on="outcome", validate="one_to_one")
        .merge(hypothesis_tests, on="outcome", validate="one_to_one")
    )
    final_columns = [
        "outcome",
        "control_n",
        "treatment_n",
        "control_positive",
        "treatment_positive",
        "control_rate",
        "treatment_rate",
        "absolute_lift",
        "absolute_ci_lower",
        "absolute_ci_upper",
        "absolute_bootstrap_ci_lower",
        "absolute_bootstrap_ci_upper",
        "relative_lift",
        "relative_ci_lower",
        "relative_ci_upper",
        "relative_bootstrap_ci_lower",
        "relative_bootstrap_ci_upper",
        "incremental_treated",
        "incremental_all",
        "z_statistic",
        "raw_p_value",
        "raw_log10_p_value",
        "raw_p_value_display",
        "holm_adjusted_p_value",
        "holm_p_value_display",
        "reject_at_0_05",
    ]
    final_results = final_results[final_columns]
    results_table_path = TABLE_DIRECTORY / "phase2_effect_summary.csv"
    exposure_table_path = TABLE_DIRECTORY / "phase2_exposure_appendix.csv"
    bootstrap_qa_path = TABLE_DIRECTORY / "phase2_bootstrap_qa.csv"
    final_results.to_csv(results_table_path, index=False)
    exposure_delivery.to_csv(exposure_table_path, index=False)
    interval_agreement.to_csv(bootstrap_qa_path, index=False)
    print(f"Saved: {results_table_path}")
    print(f"Saved: {exposure_table_path}")
    print(f"Saved: {bootstrap_qa_path}")
    required_figure_paths = [
        arm_rates_path,
        absolute_lift_path,
        relative_lift_path,
        incremental_path,
        delivery_funnel_path,
    ]
    required_table_paths = [results_table_path, exposure_table_path, bootstrap_qa_path]
    assert all((path.exists() and path.stat().st_size > 0 for path in required_figure_paths))
    assert all((path.exists() and path.stat().st_size > 0 for path in required_table_paths))
    assert len(final_results) == 2
    assert final_stability["stability_pass"].all()
    assert interval_agreement["agreement_pass"].all()
    assert hypothesis_tests["reject_at_0_05"].all()
    connection.close()
    print(f"Final bootstrap replicates per outcome: {final_replicates:,}")
    print("All required figures and tables were created.")
    print("All Phase 2 analytical and reproducibility checks passed.")
    print("Database connection closed.")
