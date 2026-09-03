# Project overview

AdLift asks two related questions on the public Criteo uplift benchmark: did randomized advertising eligibility change visits or conversions on average, and can a learned ranking concentrate those incremental outcomes within a limited targeting budget?

The analysis retains all 13,979,592 released records. It audits schema and observable balance, estimates randomized assignment effects, trains S-, T-, and pooled X-style learners, freezes test scores, and compares them with constant, one-feature segment, random, and response-propensity baselines. Evaluation uses score groups, cumulative gain, AUUC/Qini, fixed-ranking bootstrap uncertainty, and budget-specific policy estimates.

The project is a public benchmark analysis. Criteo reports that the release was anonymized and non-uniformly subsampled. It contains no campaign, date, cost, revenue, or identifiable user fields, so the analysis cannot estimate campaign ROI, seasonality, profit, or original advertiser incrementality.

The repository is organized by purpose: `scripts/` defines execution, `src/adlift/` provides reusable functions, `notebooks/` explains saved outputs, `results/` contains generated evidence, `reports/` contains the final narrative and audit, and `tests/` protects analytical invariants.
