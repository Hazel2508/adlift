# Modeling choices

## Inputs and split

Only the twelve anonymized pre-treatment features `f0`–`f11` enter the models. Treatment, exposure, outcomes, split labels, and the technical row ID are prohibited as features. A seeded hash of the feature vector assigns 60% train, 20% validation, and 20% test. Identical feature vectors therefore cannot cross partitions.

## Learners

- **S-Learner:** one LightGBM response model with treatment as an input; uplift is the difference between predictions with treatment set to one and zero.
- **T-Learner:** separate LightGBM response models for treatment and control; uplift is their prediction difference.
- **Pooled X-style learner:** treatment records receive `Y - mu0(X)` and control records receive `mu1(X) - Y`; one LightGBM regressor learns the pooled pseudo-outcome.

The last method preserves the historical artifact name `x_learner` for lineage. It is not the canonical X-Learner, which fits two conditional effect regressors and combines them with propensity weights. The old `crossfit_fold` data column was never used by this implementation and has been removed from the preparation route.

Hyperparameters live in `config/models.json`. Validation supports early stopping and diagnostics. Test outcome values are null in the Phase 3 prediction artifacts and are first joined during evaluation.

## Baselines

Response propensity predicts the outcome from pre-treatment features without treatment. It represents conventional high-response targeting. A constant average effect and a validation-selected one-feature segment ranking provide structural references. Random targeting is the cumulative-gain diagonal.
