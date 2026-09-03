# AdLift: Advertising Incrementality and Uplift Targeting

## Executive summary

Randomized advertising eligibility increased visits and conversions in Criteo's released uplift benchmark. Treatment raised visits from 3.820% to 4.854% (absolute lift 1.034 percentage points; 95% CI 1.006–1.063) and conversions from 0.194% to 0.309% (absolute lift 0.115 percentage points; 95% CI 0.108–0.122).

Individual targeting results differ by objective. For visits, the S-Learner has Qini 10,550 and exceeds response targeting by 645 (2,000-draw fixed-ranking 95% interval 413–883). At a 10% targeting budget, it estimates 17,919 incremental visits, 3,805 more than response ranking (95% interval 2,804–4,829). For conversions, response propensity remains the supported choice: S-Learner's Qini is 9.99 lower (95% interval -18.78 to -0.99), and its disadvantage is clearest at 5–10% targeting.

These are exploratory benchmark findings. The evaluation specification was expanded after an initial look at test results. A future deployment decision needs a prospectively frozen rule and fresh sample or online experiment.

## Business question

Response targeting asks who is likely to act. Uplift targeting asks whose behavior advertising is likely to change. AdLift tests whether this causal ranking can allocate a limited advertising budget more effectively than random or high-response targeting.

## Data and experimental setup

The analysis uses 13,979,592 rows from the public Criteo uplift release, including twelve anonymized features, treatment assignment, exposure, visit, and conversion. The release combines randomized incrementality tests. Treatment is the causal assignment; exposure is a post-assignment delivery measure and does not define the primary groups.

All released rows are retained. The modeling split uses a seeded feature-vector hash so identical feature profiles cannot cross train, validation, and test. The final test population contains 2,795,749 rows: 2,376,531 treatment and 419,218 control.

## Average incrementality

| Outcome | Control | Treatment | Absolute lift | Relative lift | Incremental outcomes among treated |
|---|---:|---:|---:|---:|---:|
| Visit | 3.820% | 4.854% | 1.034 pp | 27.07% | 122,895 |
| Conversion | 0.194% | 0.309% | 0.115 pp | 59.45% | 13,687 |

Both assignment effects are statistically precise in the released sample. Incremental counts are model-based aggregate estimates, since no individual's untreated and treated outcomes are jointly observable.

![Average assignment effects](../results/figures/phase2/absolute_lift.png)

## Uplift models and evaluation

The study compares an S-Learner, T-Learner, and pooled X-style learner using LightGBM. It also evaluates constant-effect, validation-selected segment, response-propensity, and random baselines. Test outcomes remain hidden in score artifacts and are joined only for evaluation.

Models are judged by cumulative gain and Qini rather than ordinary prediction metrics. The fixed-ranking paired bootstrap preserves comparisons among rankings but does not include training or model-selection uncertainty.

## Targeting results

### Visits

![Visit Qini curves](../results/figures/phase4/visit_qini_curve.png)

X-style pooling has the highest point Qini, but its paired difference from the simpler S-Learner is unresolved, so the historical rule chooses S-Learner. S-Learner's global Qini exceeds response targeting. Its clearest advantage occurs at tighter budgets: 5,400 additional visits over response at 5%, and 3,805 at 10%. The 20% difference includes zero; the 30% difference is positive but marginal in the 2,000-draw sensitivity check.

![Visit targeting comparison](../results/figures/phase4/visit_targeting_comparison.png)

### Conversions

![Conversion Qini curves](../results/figures/phase4/conversion_qini_curve.png)

S-Learner leads the uplift learners but does not beat response propensity. It performs worse than response ranking at 5% and 10%, and differences at 20% and 30% are unresolved. Response propensity is therefore the supported conversion ranking under this exploratory evaluation.

![Conversion targeting comparison](../results/figures/phase4/conversion_targeting_comparison.png)

### Cross-objective policies

Visit and conversion S-Learner scores have Spearman rank correlation 0.888. At 30%, the policies share 84.7% of selected records. This is high agreement, but the policy differences can still matter for rare conversions; objective-specific evaluation should be retained.

## Recommendation

If the operational goal is visits and the budget targets a small fraction of the population, advance the S-Learner ranking to prospective validation. If the goal is conversions, retain response propensity as the benchmark challenger. Do not claim that 30% is an optimal spend level: it is only the volume-maximizing result among three tested candidate fractions under no cost or revenue model.

Before deployment, freeze one learner, threshold, primary metric, and uncertainty procedure; evaluate once on untouched data or in an online randomized policy test; and add cost or value per action to choose an economically meaningful threshold.

## Limitations

- Criteo non-uniformly subsampled and anonymized the public release; results do not recover an advertiser's original incrementality.
- No campaign, timestamp, cost, revenue, or identifiable user fields support campaign, temporal, profit, or cluster-aware analysis.
- The historical `x_learner` is a pooled X-style variant rather than the canonical two-regressor, propensity-blended X-Learner.
- Only one alternate training seed was saved; model stability evidence is limited.
- Bootstrap intervals condition on trained models and fixed ranking bins.
- Evaluation rules were expanded after initial test inspection, so confirmation requires new data.

The complete numerical and methodological audit is in [Phase 4 final audit](audit/phase4_final_audit.md).
