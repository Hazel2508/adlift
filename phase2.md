# Phase 2 — A/B Testing and Average Incrementality

## Phase objective

Phase 2 estimates the average causal effect of enabling Criteo advertising on:

1. website visits; and
2. advertiser-defined conversions.

The analysis uses randomized `treatment`, not observed `exposure`. Its main output is an
absolute and relative lift estimate for each outcome, accompanied by uncertainty, statistical
tests, and a business-scale interpretation.

This phase answers whether advertising works **on average**. It does not yet answer which users
should be targeted. Heterogeneous treatment effects begin in Phase 3.

## Required inputs

- Phase 1 analysis-ready Parquet dataset

## Step 1 — Analysis plan

| Decision | Selected specification |
| :--- | :--- |
| Treatment | Randomized `treatment` |
| Outcomes | `visit` and `conversion`, analyzed separately |
| Effect measures | Absolute lift (ATE) and relative lift |
| Confidence level | 95% |
| Hypothesis test | Two-sided two-sample z-test for proportions |
| Significance level | 0.05 |
| P-value | Report raw and Holm-adjusted p-values for `visit` and `conversion` |
| Bootstrap | Treatment-stratified binary-outcome bootstrap |

## Step 2 — Calculate effect sizes

For each outcome, first calculate the treatment and control outcome counts and rates. Use those
rates to calculate the following effect measures.

### 2.1 Absolute lift (ATE)

Absolute lift measures how much the average outcome rate changes when advertising is enabled. For
these binary outcomes, absolute lift is the estimated **average treatment effect (ATE)**:

$$
\widehat{\mathrm{ATE}} = \hat{p}_1-\hat{p}_0
$$

### 2.2 Relative lift

Relative lift is useful for communicating change relative to a low baseline, especially for
conversion.

$$
\text{Relative Lift} = \frac{\hat{p}_1-\hat{p}_0}{\hat{p}_0}
$$

### 2.3 Incremental outcomes

The estimated incremental outcomes among treatment-assigned users measure how many additional
visits or conversions occurred in the treatment group compared with the number expected if those
users had experienced the control outcome rate:

$$
\widehat{N}_{\text{incremental, treated}} = N_1(\hat{p}_1-\hat{p}_0)
$$

The estimated incremental outcomes under a treat-all policy measure how many additional visits or
conversions would be expected if the entire released population were assigned to treatment rather
than control:

$$
\widehat{N}_{\text{incremental, all}} = N(\hat{p}_1-\hat{p}_0)
$$

The first quantity describes the scale of incremental outcomes within the experiment's treatment
group. The second extrapolates the same estimated effect to the full released population.

## Step 3 — Confidence intervals

### 3.1 Analytical confidence interval for absolute lift

Use the unpooled standard error:

$$
SE(\widehat{\mathrm{ATE}}) = \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{N_1} + \frac{\hat{p}_0(1-\hat{p}_0)}{N_0}}
$$

The 95% confidence interval is:

$$
\widehat{\mathrm{ATE}} \pm 1.96 \times SE(\widehat{\mathrm{ATE}})
$$

The sample and event counts are large enough for this approximation for both outcomes.

### 3.2 Analytical confidence interval for relative lift

Relative lift is:

$$
\widehat{L}_{\text{relative}} = \frac{\hat{p}_1}{\hat{p}_0}-1
$$

Construct its analytical confidence interval on the log scale. Let $y_1$ and $y_0$ be the numbers
of outcome-positive users in treatment and control:

$$
\hat{\theta} = \log\left(\frac{\hat{p}_1}{\hat{p}_0}\right)
$$

$$
SE(\hat{\theta}) =
\sqrt{
\frac{1}{y_1}-\frac{1}{N_1}
+
\frac{1}{y_0}-\frac{1}{N_0}
}
$$

Transform the log-scale endpoints back to relative lift:

$$
CI_{\text{relative lift}} =
\left[
\exp\left(\hat{\theta}-1.96SE(\hat{\theta})\right)-1,\;
\exp\left(\hat{\theta}+1.96SE(\hat{\theta})\right)-1
\right]
$$

The log transformation is used because relative lift is a ratio-based measure and its sampling
distribution is not necessarily symmetric.

### 3.3 Bootstrap confidence intervals

Run the bootstrap separately for `visit` and `conversion`. Use 10,000 bootstrap replicates and
a frozen random seed. For each replicate $b$, sample $N_1$ outcome values with replacement from the
treatment group and $N_0$ outcome values with replacement from the control group.

Because the outcomes are binary, the same resampling can be performed more efficiently by drawing
the number of positive outcomes:

$$
Y_1^{*(b)} \sim \text{Binomial}(N_1,\hat{p}_1)
$$

$$
Y_0^{*(b)} \sim \text{Binomial}(N_0,\hat{p}_0)
$$

Then calculate:

$$
\hat{p}_1^{*(b)} = \frac{Y_1^{*(b)}}{N_1},
\qquad
\hat{p}_0^{*(b)} = \frac{Y_0^{*(b)}}{N_0}
$$

For each replicate:

$$
\widehat{\mathrm{ATE}}^{*(b)} = \hat{p}_1^{*(b)}-\hat{p}_0^{*(b)}
$$

$$

\widehat{L}_{\text{relative}}^{*(b)} = \frac{\hat{p}_1^{*(b)}-\hat{p}_0^{*(b)}}{\hat{p}_0^{*(b)}}
$$

After all 10,000 replicates, sort the bootstrap estimates for each measure and use the 2.5th and
97.5th percentiles as the lower and upper endpoints.

This produces a 95% percentile-bootstrap confidence interval for both absolute lift and relative
lift. If a bootstrap replicate has a zero control outcome rate, relative lift is undefined; this
is not expected for either project outcome given the observed control event counts.

### 3.4 Check bootstrap stability and analytical agreement

First verify that 10,000 replicates are sufficient. Compare the intervals produced by the first
5,000 and second 5,000 replicates. Corresponding endpoints should differ by no more than 2% of the
full 10,000-replicate interval width. If they differ more, increase the bootstrap to 50,000
replicates.

Then compare the analytical and bootstrap intervals separately for absolute and relative visit
lift and absolute and relative conversion lift. Treat the two methods as reasonably close when the absolute difference between corresponding
analytical and bootstrap endpoints is no greater than 10% of the analytical interval width, and
the two interval widths differ by no more than 10%. These are project QA thresholds for identifying implementation problems, not statistical
requirements that the two methods must satisfy. If the intervals do not meet these criteria, verify the treatment labels, denominators, formulas,
and bootstrap sampling procedure. If the discrepancy remains after increasing the number of
replicates, report both intervals and document the discrepancy.

### Step 4 — Perform hypothesis tests

For each outcome:

$$
H_0: p_1-p_0=0
$$

$$
H_A: p_1-p_0\neq0
$$

Use a two-sided two-sample z-test for proportions. For a two-group binary outcome, its squared test
statistic is equivalent to the Pearson chi-square statistic from the corresponding two-by-two
table.

Report:

- test statistic;
- raw p-value;
- Holm-adjusted p-value across `visit` and `conversion`;
- absolute lift with its analytical and bootstrap confidence intervals; and
- relative lift with its analytical and bootstrap confidence intervals.

### Step 5 — Exposure analysis as a descriptive appendix

First, calculate the difference in exposure rates between the randomized groups:

$$
\hat{p}(\text{exposure}=1 \mid T=1) - \hat{p}(\text{exposure}=1 \mid T=0)
$$

This measures how much assignment to treatment increased the probability of receiving an
advertisement. It describes treatment delivery, not the effect of advertising on visits or
conversions.

Create a descriptive table for:

- control users;
- treatment-assigned but unexposed users; and
- treatment-assigned exposed users.

Show exposure, visit, and conversion rates, but state prominently:

> The exposed-versus-unexposed contrast is observational and is not an estimate of the causal
> effect of seeing an advertisement.

Users with an impression necessarily had an eligible auction opportunity and may have been
selected by Criteo's bidding system. Their much higher response rates combine treatment effects
with selection.

#### Optional advanced extension: complier effect

Estimate a Wald instrumental-variable ratio:

$$
\frac{E[Y\mid T=1]-E[Y\mid T=0]}{E[D\mid T=1]-E[D\mid T=0]}
$$

where $D$ is `exposure`.

### Step 6 — Build the required visual evidence

#### Figure 1 — Arm rates

For visits and conversions, plot treatment and control rates with 95% confidence intervals. Use
separate panels or scales because conversion is much rarer.

#### Figure 2 — Absolute lift

Plot the treatment-control percentage-point difference and confidence interval for each outcome.
Include a vertical zero reference line.

#### Figure 3 — Relative lift

Plot relative lift with bootstrap confidence intervals. Label the control baseline rate so the
relative effect is not viewed without context.

#### Figure 4 — Incremental outcome scale

Show estimated incremental visits and conversions for:

- the observed treatment-arm size; and
- the treat-all released-population scenario.

#### Figure 5 — Experiment delivery funnel

Show counts from treatment assignment to exposure, visit, and conversion. The funnel is
descriptive; do not imply that each later event is caused only by exposure.

### Step 7 — Interpret the two outcomes together

The final narrative to answer:

1. Did enabling advertising increase visits?
2. Did enabling advertising increase conversions?
3. What was the absolute and relative effect for each?
4. How many incremental outcomes does that represent at the observed scale?
5. Is the conversion conclusion less precise because the event is rare?
6. Does visit lift translate proportionally into conversion lift?
7. What remains unknown because experiment, campaign, cost, and revenue fields are absent?

A positive average effect does not mean every user benefits. It establishes that heterogeneous
uplift modeling is worth attempting.

Because the released benchmark was anonymized and subsampled, describe these as effects in the
released experimental sample. Do not present them as the original incrementality of a specific
advertiser or campaign.
