# Phase 4 final audit

Audit date: 2026-09-03. Original analysis: 2026-08-25.

**Assessment: Share with caveats.** The saved ranking metrics and policy estimates reconcile to the frozen test predictions. This is an exploratory benchmark analysis, not independent confirmation of a deployment policy. The numerical snapshot is preserved; subsequent repository restructuring changes presentation and execution paths.

## Evidence and calculation checks

- Raw compressed input SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc` (recomputed).
- Modeling input SHA-256: `ea3fecb183bd50af05363e6917ae97f43a83b79e4da95a04094e0a92fbaec2c2` (recomputed).
- Evaluation specification SHA-256: `6af08cd7a03f16316acf0404af5d6d089c2cf79f216e00c375b70d466aa01503`.
- Test population: 2,795,749 records; treatment 2,376,531; control 419,218. IDs are unique and identical across both outcomes and all three uplift predictions. Prediction outcomes are null; score and source/config metadata checks pass.
- Independently sorted predictions and recomputed 240 cumulative curve points, 120 decile effects and 12 AUUC/Qini pairs. Maximum discrepancies are floating-point rounding only. All six strategies have the same 100% endpoint for each outcome.
- Replayed all 200 paired bootstrap replicates: all 122 uncertainty rows match the saved tables. This verifies implementation reproducibility; it does not establish post-selection coverage.
- Existing analytical guardrails: 11 tests passed before restructuring.
- Table-level evidence is in the accompanying reconciliation JSON and bootstrap sensitivity CSVs. Row-level source data and model artifacts remain local.

## Material findings

1. **High — Selection after test inspection.** `analysis_status` explicitly records additions after the first test review. Learner choice, baseline gating and targeting fraction use this same test set. The saved `evaluation_spec_frozen_before_test_outcomes` status only describes the amended run's execution order; it cannot restore an untouched test set. All portfolio results must be labeled exploratory. A prospective frozen policy and fresh sample/online experiment are needed for confirmation.
2. **High — 30% is a constrained candidate, not an economic optimum.** The rule maximizes the lower confidence bound of total gains among 10%, 20%, 30%. There are no cost, revenue or per-person value fields. This does not identify profit, ROI, optimal spend, or the best targeting fraction over the entire population.
3. **Medium — Global and local comparisons differ.** Visit S-Learner beats response on global Qini. Its Top 20% paired difference interval includes zero. Conversion response has stronger global Qini, while its Top 30% difference from S-Learner is unresolved. A claim that one model wins at every budget is unsupported.
4. **Medium — X-Learner naming.** `x_learner` is a pooled single-pseudo-outcome regression: one function is fit to D=Y-mu0(X) for treatment and D=mu1(X)-Y for control. It does not implement the canonical two effect regressors with propensity blending. Keep the historical artifact ID but label it “X-style pooled learner” in the public narrative. No cross-fitting is used in the current model fit; an old `crossfit_fold` data column is unused.
5. **Medium — Conditional uncertainty.** Bootstrap draws preserve paired records across rankings but keep the fitted models and ranking bins fixed. They exclude training uncertainty, policy selection, multiple-comparison correction and unknown campaign/user clustering. Only one additional training seed was compared; this is limited stability evidence.
6. **Medium — Benchmark scope.** Criteo non-uniformly subsampled the release. Estimates describe the released benchmark under its analysis assumptions; they do not recover a named advertiser's original incrementality. No timestamps or campaign identifiers support temporal/campaign validation.
7. **Low — Calibration is diagnostic.** Regression is fitted to held-out decile estimates and not applied to ranking. It is not an independently validated calibrated individual effect model. A non-significant learner difference is also not proof of equivalence.
8. **Execution blocker — Stale preparation script.** `prepare_phase3_data.py` reads `config['crossfit_folds']`, absent from the current config, while model training no longer uses cross-fitting. Repair this route during restructuring and validate the raw-to-model-input path. The first “data quality audit” notebook only opens the database and does not actually perform the advertised audit; add real script-backed checks.

## Presentation review

Reviewed the original targeting and Qini figures. The targeting bars omit available intervals, the curves omit their zero origin, the overlap title does not specify that it compares the two S-Learner policies, and calibration titles omit their diagnostic nature. Add those distinctions to the regenerated figures. Keep full strategy tables for inspection and use a compact targeting comparison in the report.

## Evidence that remains unavailable

No new untouched test population, online policy experiment, advertiser cost/revenue, campaign identifiers, or original historical package lock is available. Replaying saved scores is not the same as retraining all historical models bit-for-bit. Do not describe these checks as completed.

## Sources

[Criteo dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/); [Künzel et al., canonical X-Learner](https://arxiv.org/abs/1706.03461). Instructor requirements were recovered from the user's final-session transcript in “项目收尾与展示整理”; teaching materials are retained outside the publication tree.

## 2,000-replicate sensitivity check

A separate exploratory rerun increases paired bootstrap draws to 2,000 and adds the 5% threshold. Models, scores and historical 200-draw results are unchanged.

| Outcome | S-Learner minus response | Difference | 95% interval |
|---|---|---:|---:|
| Visit | Qini | 644.54 | [413.21, 883.47] |
| Visit | Top 5% gain | 5,399.75 | [4,400.01, 6,452.40] |
| Visit | Top 10% gain | 3,805.23 | [2,803.61, 4,828.77] |
| Visit | Top 20% gain | 565.71 | [-104.23, 1,281.01] |
| Visit | Top 30% gain | 520.57 | [45.79, 1,029.27] |
| Conversion | Qini | -9.99 | [-18.78, -0.99] |
| Conversion | Top 5% gain | -114.07 | [-185.90, -36.80] |
| Conversion | Top 10% gain | -88.56 | [-147.20, -31.16] |
| Conversion | Top 20% gain | -14.52 | [-56.41, 28.05] |
| Conversion | Top 30% gain | 4.34 | [-16.31, 26.74] |

The nominal signs persist with more draws. Visit has its largest advantage at small targeting fractions; conversion does not establish a benefit over response ranking. These remain pointwise exploratory intervals, especially the marginal Visit 30% comparison.


## Restructuring closure

The preparation blocker is resolved in `src/adlift/data.py`. A fresh raw-data build matched all historical feature, outcome, partition and source-ID values across 13,979,592 records. The original incomplete data-audit notebook was replaced by a walkthrough of executed quality, balance, duplicate and partition checks. All three companion notebooks execute top-to-bottom. Generated targeting plots now show fixed-ranking intervals; curves include the zero origin; the overlap title identifies S-Learner; calibration is labeled diagnostic. Eleven analytical tests and the package lint check pass. The visit training route also passed a smoke run. The complete historical models were not retrained.

Reproduce the numerical audit with `uv run python scripts/audit_evaluation.py`; add `--sensitivity` for the separate 2,000-draw extension. See [reproducibility details](../../docs/reproducibility.md).
