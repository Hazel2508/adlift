# Evaluation choices

Ordinary response AUC cannot select an uplift ranking because individual treatment effects are unobserved. AdLift instead estimates randomized treatment-control differences within ranked subsets.

For targeting fraction `q`, cumulative gain is the number of selected released records multiplied by their observed treatment-minus-control outcome rate. AUUC integrates this curve; Qini subtracts the random-targeting triangle. All rankings must meet at the same 100% endpoint, which equals test-set size times the observed test assignment effect.

The paired bootstrap resamples joint cells defined by every model's fixed ranking bin, treatment, and outcome. This keeps comparisons paired and makes the historical tables reproducible. It conditions on fitted models and fixed bins; it omits training, threshold-selection, multiplicity, and possible hidden campaign/user clustering uncertainty.

The historical selection rule chooses the point-Qini leading uplift learner, then prefers a simpler learner when the paired difference is unresolved. It deploys uplift ranking only if the global Qini difference from response has a 95% interval above zero. Finally, it chooses among 10%, 20%, and 30% using the largest lower confidence bound for total incremental outcomes. That last choice rewards total volume and cannot determine economic optimality without costs or values.

The rule set was expanded after an initial test review. Results are exploratory and require a prospectively frozen policy plus fresh data or an online experiment before deployment. Full audit evidence is in [`reports/audit/phase4_final_audit.md`](../reports/audit/phase4_final_audit.md).
