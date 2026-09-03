# Phase 4 QA notes

## 2026-08-25 — deterministic tie handling correction

The first QA evaluation used raw `source_row_id` order to break identical score ties. Some
X-Learner score blocks were large, and source row layout was associated with treatment ordering.
This produced score groups containing treatment rows but no control rows, so their observed uplift
was undefined.

That QA run was invalidated. The frozen evaluation specification now uses a deterministic hash of
`source_row_id` with `tie_break_seed = 20260825`. The hash does not use treatment or outcome and
prevents raw source order from affecting tied-score groups. Response-baseline models were reused
unchanged because they were fitted before test outcomes were opened. The formal evaluation and all
200 paired-bootstrap replicates were rerun under the corrected frozen specification.

Final QA confirms that every displayed score group contains both randomized arms and that all
observed-uplift estimates and confidence intervals are finite.

## 2026-08-25 — complete phase4.md amendment

After the initial test review, the requested implementation boundary changed from the classroom
core to every explicit item in `phase4.md`. The following additions are therefore labeled
`exploratory_completion_requested_after_initial_test_review` in the frozen specification and
saved outputs:

- constant average-effect and validation-selected segment-uplift baselines;
- current-implementation Phase 3 stability at comparison seed `20260817`;
- explicit target-none, target-all, random, response, and uplift strategy comparisons;
- a deterministic final model and Top 10/20/30 targeting rule; and
- visit-versus-conversion rank correlation, selected-user overlap, and cross-policy gains.

The constant and segment baselines were fitted on train plus validation data, and segment choices
were made using validation Qini. The current Phase 3 models were not changed or retuned. Stability
training used train and validation only. The expanded evaluation specification hash is
`6af08cd7a03f16316acf0404af5d6d089c2cf79f216e00c375b70d466aa01503`.

Both outcome score-group diagnostics use deciles, while cumulative-gain curves retain the frozen
5% grid. Paired bootstrap comparisons between S/T/X were added so the final choice can implement
the documented rule to prefer the simpler model when Qini differences are uncertain. For visit,
X-Learner has the highest point Qini, but its paired difference from S-Learner includes zero, so
S-Learner is the final recommendation.

The completed evaluation contains six scored strategies for each outcome. Every score group has
both randomized arms, all Top 10/20/30 bootstrap intervals are populated, and every strategy has
the same Top 100% endpoint. Eleven automated tests pass.
