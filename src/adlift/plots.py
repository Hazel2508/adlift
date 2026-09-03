"""Figures generated from evaluated result tables."""

from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .paths import resolve_path


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
        axis.plot(
            np.r_[0, table["target_fraction"].to_numpy() * 100],
            np.r_[0, table["cumulative_gain"].to_numpy()],
            label={
                "x_learner": "X-style pooled",
                "s_learner": "S-Learner",
                "t_learner": "T-Learner",
                "response_propensity": "Response",
                "constant_ate": "Constant effect",
                "segment_uplift": "Segment",
            }.get(model_name, model_name),
        )
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
            np.r_[0, table["target_fraction"].to_numpy() * 100],
            np.r_[0, (table["cumulative_gain"] - random_gain).to_numpy()],
            label={
                "x_learner": "X-style pooled",
                "s_learner": "S-Learner",
                "t_learner": "T-Learner",
                "response_propensity": "Response",
                "constant_ate": "Constant effect",
                "segment_uplift": "Segment",
            }.get(model_name, model_name),
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
        axis.set_title("X-style pooled" if model_name == "x_learner" else model_name)
        axis.set_xlabel("Score group (1 = highest)")
        axis.set_ylabel("Observed uplift")
    for axis in list(np.asarray(axes).flat)[len(model_order) :]:
        axis.set_visible(False)
    figure.suptitle(f"{outcome_name.title()} observed uplift by score group")
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_group_uplift.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    for model_name in config["uplift_models"]:
        table = calibration[calibration["model_name"] == model_name]
        axis.scatter(
            table["mean_predicted_score"],
            table["observed_uplift"],
            label={
                "x_learner": "X-style pooled",
                "s_learner": "S-Learner",
                "t_learner": "T-Learner",
                "response_propensity": "Response",
                "constant_ate": "Constant effect",
                "segment_uplift": "Segment",
            }.get(model_name, model_name),
        )
        ordered = table.sort_values("mean_predicted_score")
        axis.plot(
            ordered["mean_predicted_score"],
            ordered["calibrated_predicted_uplift"],
        )
    axis.set_title(f"{outcome_name.title()} uplift calibration diagnostic")
    axis.set_xlabel("Mean predicted uplift")
    axis.set_ylabel("Observed treatment-control difference")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_uplift_calibration.png", dpi=180)
    plt.close(figure)

    rates = [float(x) for x in config["main_decision_rates"]]
    names = {
        "s_learner": "S-Learner",
        "t_learner": "T-Learner",
        "x_learner": "X-style pooled",
        "response_propensity": "Response",
        "constant_ate": "Constant effect",
        "segment_uplift": "Segment",
    }
    figure, axes = plt.subplots(1, len(rates), figsize=(12, 4.8), sharey=True, sharex=True)
    for axis, rate in zip(np.atleast_1d(axes), rates):
        table = policy[policy["target_fraction"] == rate].set_index("model_name").loc[model_order]
        values = table["cumulative_gain"].to_numpy()
        errors = np.vstack(
            [
                values - table["incremental_gain_ci_lower"].to_numpy(),
                table["incremental_gain_ci_upper"].to_numpy() - values,
            ]
        )
        colors = [
            "#2563a6"
            if m == "s_learner"
            else "#dd9335"
            if m == "response_propensity"
            else "#bcc7d5"
            for m in model_order
        ]
        axis.barh(np.arange(len(model_order)), values, color=colors, xerr=errors, capsize=3)
        axis.set_yticks(np.arange(len(model_order)), [names[m] for m in model_order])
        axis.set_title(f"Top {rate:.0%}")
        axis.set_xlabel("Estimated incremental outcomes")
        axis.set_xlim(
            0,
            float(policy[policy["target_fraction"].isin(rates)]["incremental_gain_ci_upper"].max())
            * 1.06,
        )
    axes[0].invert_yaxis()
    figure.suptitle(f"{outcome_name.title()} targeting — 95% fixed-ranking intervals")
    figure.tight_layout()
    figure.savefig(figure_directory / f"{outcome_name}_targeting_comparison.png", dpi=180)
    plt.close(figure)
