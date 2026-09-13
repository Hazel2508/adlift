# Experimental design

## Questions and estimands

AdLift separates two decisions that require different estimands:

1. **Average incrementality:** does randomized advertising eligibility change visit or
   conversion rates across the released population?
2. **Targeting incrementality:** can a score rank released records so that a limited targeting
   budget concentrates more incremental outcomes than random or response-propensity targeting?

For outcome \(Y\), let \(Y(1)\) and \(Y(0)\) denote the potential outcomes under treatment and
control assignment. The primary average estimand is the intent-to-treat effect:

$$
\operatorname{ATE}=E[Y(1)-Y(0)].
$$

The released randomized assignment is stored in `treatment`. The observed `exposure` field is a
post-assignment delivery measure, so it is not used to define the primary causal groups. Comparing
exposed and unexposed records would condition on a post-treatment event and would no longer retain
the randomized treatment-control comparison. Exposure is therefore reported only as a descriptive
delivery appendix.

The project reports two average effect scales:

$$
\text{absolute lift}=\hat p_1-\hat p_0,
\qquad
\text{relative lift}=\frac{\hat p_1-\hat p_0}{\hat p_0},
$$

where \(\hat p_1\) and \(\hat p_0\) are the observed outcome rates in treatment and control.
No CUPED or regression-adjusted ATE is reported. The release contains no validated pre-period
outcome, campaign, or time field for such an adjustment, and the prespecified project estimate is
the unadjusted randomized difference in means.

## Data, population, and outcomes

The analysis uses all 13,979,592 records in the public Criteo uplift v2.1 release. Each row contains
twelve anonymized pre-treatment features (`f0`–`f11`), randomized `treatment`, post-assignment
`exposure`, and two binary outcomes:

- `visit`: a website visit;
- `conversion`: the advertiser-defined conversion, which is nested within visit in this release.

The unit of analysis is a released benchmark record. Because the data are anonymized and
non-uniformly subsampled, a row is not interpreted as a known advertiser, campaign, or identifiable
customer. Results describe the released benchmark population under its documented construction.

The data preparation check retains every row and verifies the expected schema, finite features,
binary indicators, outcome nesting, and the absence of exposed controls. It also reports observable
feature balance by treatment assignment. The largest absolute standardized mean difference among
the twelve released features is 0.0488. See the saved [quality checks](../results/tables/eda/quality_checks.csv),
[balance table](../results/tables/eda/feature_balance.csv), and
[partition summary](../results/tables/eda/partition_summary.csv).

## Train, validation, and test separation

A seeded hash of the twelve pre-treatment features assigns records to approximately 60% train,
20% validation, and 20% test partitions. Identical feature vectors receive the same hash and cannot
cross partitions. The stored partition is checked against the feature-only rule whenever the data
preparation pipeline runs.

| Partition | Treatment rows | Control rows | Allowed use |
|---|---:|---:|---|
| Train | 7,130,304 | 1,258,910 | Fit model parameters |
| Validation | 2,375,820 | 418,809 | Early stopping and constrained model development |
| Test | 2,376,531 | 419,218 | Score features, then evaluate frozen rankings |

Only `f0`–`f11` may enter a model matrix. `exposure`, outcomes, treatment-derived values, row IDs,
and partition labels are excluded as predictive features. `treatment` is included only where the
learner definition requires the intervention indicator.

During model development, test features may be scored after the specification is frozen, but test
outcomes are written as null in the Phase 3 prediction artifacts. Phase 4 joins the outcomes by the
technical `source_row_id` only after verifying model lineage, score completeness, uniqueness, and
common test populations. This boundary prevents held-out outcomes from influencing model fitting or
score creation.

## Models and comparison strategies

Separate Visit and Conversion models are trained with LightGBM:

- **S-Learner:** one response function \(\hat\mu(x,t)\), scored at both treatment states;
- **T-Learner:** separate \(\hat\mu_1(x)\) and \(\hat\mu_0(x)\) response functions;
- **pooled X-style learner:** one effect model fitted to pooled imputed-effect pseudo-outcomes.

The historical artifact named `x_learner` is a pooled X-style variant, not the canonical
two-effect-regressor, propensity-blended X-Learner. Model comparisons also include random,
constant-ATE, validation-selected segment-uplift, and response-propensity baselines. The response
baseline ranks records by predicted outcome probability and answers whether causal ranking adds
value beyond conventional high-response targeting.

## Group-level uplift evaluation

An individual treatment effect cannot be observed because each record appears under only one
assignment. Phase 4 therefore ranks records using scores created without test outcomes and estimates
treatment-control differences within selected ranked groups.

For the top-scoring fraction \(q\), let \(S(q)\) denote the selected test records. The observed
selected-group uplift and cumulative gain are:

$$
\widehat u(q)=\bar Y_1(q)-\bar Y_0(q),
\qquad
\widehat G(q)=n(q)\widehat u(q).
$$

The evaluation reports 5%, 10%, 20%, 30%, 40%, 50%, 75%, and 100% targeting fractions. The 10%,
20%, and 30% fractions are the historical main decision points; 5% was added in a later exploratory
sensitivity check. At 100%, every valid ranking must reach the same endpoint,
\(N_{test}\widehat{ATE}_{test}\). This identity checks treatment direction, denominators, and model
population consistency.

AUUC integrates the cumulative-gain curve. This project defines Qini as:

$$
\operatorname{Qini}=\operatorname{AUUC}-\tfrac{1}{2}G(1),
$$

the area above the random-targeting triangle. Higher Qini means the ranking concentrates more
incremental outcomes earlier, but model selection also considers paired uncertainty and performance
at the decision fractions.

## Uncertainty and decision rule

Average treatment effects use unpooled analytical 95% confidence intervals, treatment-stratified
binary-outcome bootstrap intervals, and two-sided proportion tests with Holm adjustment across the
two outcomes.

Phase 4 uses a paired bootstrap that resamples joint cells formed from all fixed model-ranking bins,
treatment, and outcome. This compares rankings on the same resampled test population. The saved
historical analysis uses 200 draws; a retained sensitivity audit uses 2,000 draws and includes the
5% comparison.

The historical selection rule:

1. starts with the uplift learner having the highest point Qini;
2. prefers a simpler learner when its paired Qini difference from the leader is unresolved;
3. deploys uplift ranking only when its Qini-minus-response 95% interval is above zero; and
4. chooses among 10%, 20%, and 30% using the largest lower confidence bound for total incremental
   outcomes.

This last step selects among three candidate coverage levels by incremental outcome volume. It does
not estimate an economically optimal threshold because the release has no cost, revenue, impression
price, or per-outcome value.

## Interpretation limits

The assignment contrast supports causal interpretation for the released randomized benchmark,
subject to consistency, no interference between released records, correct outcome measurement, and
the integrity of the assignment field. It does not identify the effect for a named advertiser's
original traffic or prove that a targeting policy will reproduce the result prospectively.

The evaluation rule was expanded after initial test inspection. Bootstrap intervals condition on
the fitted models and fixed ranking bins and do not include training, model-selection, threshold-
selection, multiplicity, or unknown campaign/customer clustering uncertainty. All targeting
conclusions are therefore exploratory. Deployment requires a prospectively frozen policy evaluated
once on fresh data or in an online randomized policy experiment.

Implementation details are in [modeling choices](modeling.md), metric definitions are in
[evaluation choices](evaluation.md), and the numerical limitations are recorded in the
[evaluation audit](../reports/audit/evaluation_audit.md).
