# AdLift

**Advertising incrementality and uplift targeting on 13.98M released randomized experiment records**

AdLift separates *people likely to act* from *people whose behavior advertising is likely to change*. It estimates average randomized advertising effects, trains three heterogeneous-effect rankings, and tests whether causal targeting improves on conventional response propensity.

> **Portfolio status:** complete exploratory benchmark analysis. The numerical audit passed, with limitations recorded below. It is not a production policy or an advertiser case study.

## Key results

| Decision | Result |
|---|---|
| Did advertising increase visits? | +1.034 percentage points (95% CI 1.006–1.063), +27.1% relative |
| Did advertising increase conversions? | +0.115 percentage points (95% CI 0.108–0.122), +59.4% relative |
| Best supported visit ranking | S-Learner; Qini +645 vs response (2,000-draw 95% CI 413–883) |
| Visit value at a 10% budget | 17,919 estimated incremental visits; +3,805 vs response (95% CI 2,804–4,829) |
| Best supported conversion ranking | Response propensity; uplift learners did not improve global Qini |

![Visit uplift targeting](results/figures/phase4/visit_targeting_comparison.png)

The original rule selected 30% from the limited 10/20/30 candidate set by total incremental-outcome lower bound. Without cost or revenue data, **30% is not a profit-optimal budget**. See the [final report](reports/final_report.md) and [audit](reports/audit/phase4_final_audit.md).

## Methodology

1. Audit 13,979,592 public Criteo records and observable treatment balance.
2. Estimate intent-to-treat visit and conversion effects.
3. Split by a seeded hash of pre-treatment features: 60% train, 20% validation, 20% test.
4. Fit LightGBM S-, T-, and pooled X-style learners without test outcomes.
5. Compare cumulative gain, Qini, calibration diagnostics, response/structural baselines, and fixed-ranking paired bootstrap intervals.

The historical artifact called `x_learner` is a pooled single-pseudo-outcome variant; it is labeled precisely in [modeling choices](docs/modeling.md).

## Repository structure

```text
adlift/
├── config/              # model and evaluation parameters
├── docs/                # project, modeling, and evaluation rationale
├── notebooks/           # three reader-facing, executed walkthroughs
├── reports/             # final report and immutable audit evidence
├── results/             # generated tables and figures
├── scripts/             # process entry points, including main.py
├── src/adlift/          # reusable data, experiment, model, metric, and plot modules
└── tests/               # analytical invariants and leakage guardrails
```

## Getting started

Verified on macOS; Linux is documented but untested. Requirements: Python 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Hazel2508/adlift.git
cd adlift
uv sync
mkdir -p data/raw
```

Download the [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and save the v2.1 gzip file as:

```text
data/raw/criteo-uplift-v2.1.csv.gz
```

The repository does not distribute the dataset or trained model artifacts. Criteo publishes it under CC BY-NC-SA 4.0 and describes the release as anonymized and non-uniformly subsampled.

## Running the project

Run the complete process without opening a notebook:

```bash
uv run python scripts/main.py
```

Or run one stage:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/estimate_incrementality.py
uv run python scripts/train_models.py
uv run python scripts/evaluate_models.py all
```

A full model run processes 13.98M rows and can take substantial memory and time. For a quick engineering check:

```bash
uv run python scripts/train_models.py --outcome visit --smoke
uv run python -m unittest discover -s tests -v
```

## Explore the analysis

- [Data and experiment audit](notebooks/01_explore_data.ipynb)
- [Average incrementality](notebooks/02_estimate_incrementality.ipynb)
- [Uplift targeting evaluation](notebooks/03_evaluate_targeting.ipynb)

Detailed rationale: [overview](docs/project_overview.md), [modeling](docs/modeling.md), [evaluation](docs/evaluation.md), and [reproducibility & tests](docs/reproducibility.md).

## Known limitations

Evaluation rules were expanded after an initial test review, so targeting findings are exploratory. The bootstrap fixes fitted rankings and does not capture training, selection, multiplicity, or unknown cluster uncertainty. The public data lack campaign, time, cost, revenue, and interpretable feature labels. A prospectively frozen policy and fresh or online randomized evaluation are required before deployment.

## Data attribution

Diemert, E.; Betlei, A.; Renaudin, C.; Amini, M.-R. (2018), *A Large Scale Benchmark for Uplift Modeling*, AdKDD & TargetAd Workshop, KDD 2018. Dataset: [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
