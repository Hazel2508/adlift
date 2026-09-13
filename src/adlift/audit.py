"""Independent cumulative-sum checks against the historical ranking tables."""

import math
import numpy as np
import pandas as pd
from . import evaluation as ev
from .artifacts import read_json, write_json
from . import metrics as ranking_metrics
from .paths import PROJECT_ROOT


def audit_saved_evaluation() -> dict:
    config = read_json(PROJECT_ROOT / "config/evaluation.json")
    summary = {
        "source": "data/processed/phase3",
        "spec_hash": ev.phase4_spec_hash(config),
        "handoff": ev.validate_phase3_handoff(config, config["outcomes"]),
        "checks": [],
    }
    checks = summary["checks"]
    for outcome in config["outcomes"]:
        ids, t, y = ev.load_test_outcomes(config, outcome)
        scores = ev.load_scores(config, outcome, ids)
        assert set(np.unique(t)) == {0, 1} and set(np.unique(y)) == {0, 1}
        path = PROJECT_ROOT / "results/tables/phase4"
        curves = pd.read_csv(path / f"{outcome}_cumulative_gain.csv")
        groups = pd.read_csv(path / f"{outcome}_group_effects.csv")
        metrics = pd.read_csv(path / f"{outcome}_model_metrics.csv")
        policy = pd.read_csv(path / f"{outcome}_policy_summary.csv")
        tie = ranking_metrics.deterministic_tie_key(ids, config["tie_break_seed"])
        curve_groups = {}
        for model, s in scores.items():
            order = np.lexsort((tie, -s))
            st = t[order]
            sy = y[order]
            c1 = np.cumsum(st, dtype=np.int64)
            c0 = np.arange(1, len(t) + 1) - c1
            y1 = np.cumsum(sy * st, dtype=np.int64)
            y0 = np.cumsum(sy * (1 - st), dtype=np.int64)
            ns = np.array([math.ceil(len(t) * k / 20) for k in range(1, 21)])
            j = ns - 1
            gain = ns * (y1[j] / c1[j] - y0[j] / c0[j])
            saved = curves[curves.model_name == model].sort_values("target_fraction")
            np.testing.assert_allclose(saved.selected_rows, ns, rtol=0, atol=0)
            np.testing.assert_allclose(saved.cumulative_gain, gain, rtol=1e-12, atol=1e-8)
            auuc = np.trapezoid(np.r_[0, gain], np.arange(21) / 20)
            qini = auuc - gain[-1] / 2
            m = metrics[metrics.model_name == model].iloc[0]
            np.testing.assert_allclose([m.auuc, m.qini], [auuc, qini], rtol=1e-12, atol=1e-8)
            bounds = [math.ceil(len(t) * k / 10) for k in range(11)]
            for k in range(10):
                tt = st[bounds[k] : bounds[k + 1]]
                yy = sy[bounds[k] : bounds[k + 1]]
                g = groups[(groups.model_name == model) & (groups.score_group == k + 1)].iloc[0]
                assert g.treatment_rows == int(tt.sum()) and g.control_rows == int((1 - tt).sum())
                np.testing.assert_allclose(
                    g.observed_uplift, yy[tt == 1].mean() - yy[tt == 0].mean(), atol=1e-14
                )
            rank_group = np.empty(len(ids), dtype=np.int16)
            rank_group[order] = np.minimum(np.arange(len(ids)) * 20 // len(ids), 19)
            curve_groups[model] = rank_group
            checks.append(
                {
                    "outcome": outcome,
                    "model": model,
                    "rows": len(ids),
                    "curve_points_verified": 20,
                    "deciles_verified": 10,
                    "qini": float(qini),
                    "max_gain_error": float(np.max(np.abs(saved.cumulative_gain - gain))),
                }
            )
        boot = ranking_metrics.bootstrap_uncertainty(
            config,
            outcome,
            curve_groups,
            t,
            y,
            metrics[metrics.model_name != "random_uniform"],
            policy,
        )
        old = pd.read_csv(path / f"{outcome}_uncertainty.csv")
        keys = ["model_name", "metric"]
        a = boot.sort_values(keys).reset_index(drop=True)
        b = old.sort_values(keys).reset_index(drop=True)
        assert a[keys].equals(b[keys])
        np.testing.assert_allclose(
            a[["estimate", "ci_lower", "ci_upper"]],
            b[["estimate", "ci_lower", "ci_upper"]],
            rtol=1e-10,
            atol=1e-8,
        )
        summary[outcome] = {
            "rows": len(ids),
            "treatment_rows": int(t.sum()),
            "control_rows": int((1 - t).sum()),
            "ate": float(y[t == 1].mean() - y[t == 0].mean()),
            "uncertainty_rows_verified": len(boot),
        }
        print(outcome, summary[outcome], flush=True)
    (PROJECT_ROOT / "reports/audit").mkdir(parents=True, exist_ok=True)
    write_json(PROJECT_ROOT / "reports/audit/reconciliation.json", summary)
    print("All row-level reconciliation checks passed.", flush=True)
    return summary


def bootstrap_sensitivity(replicates: int = 2000) -> None:
    config = read_json(PROJECT_ROOT / "config/evaluation.json")
    config["bootstrap_replicates"] = replicates
    config["main_decision_rates"] = [0.05, 0.1, 0.2, 0.3]
    directory = PROJECT_ROOT / "results/tables/phase4"
    output = PROJECT_ROOT / "reports/audit"
    output.mkdir(parents=True, exist_ok=True)
    for outcome in config["outcomes"]:
        ids, treatment, observed = ev.load_test_outcomes(config, outcome)
        scores = ev.load_scores(config, outcome, ids)
        groups = {
            name: ranking_metrics.rank_groups(score, ids, 20, config["tie_break_seed"])
            for name, score in scores.items()
        }
        metrics = pd.read_csv(directory / f"{outcome}_model_metrics.csv")
        policy = pd.read_csv(directory / f"{outcome}_policy_summary.csv")
        result = ranking_metrics.bootstrap_uncertainty(
            config,
            outcome,
            groups,
            treatment,
            observed,
            metrics[metrics.model_name != "random_uniform"],
            policy,
        )
        result.to_csv(output / f"{outcome}_bootstrap_{replicates}.csv", index=False)
        print(f"{outcome}: saved {replicates}-draw exploratory sensitivity check.")
