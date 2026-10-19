# FraudGuard RiskOps — Feature Contract v1

## Dataset
PaySim synthetic financial transactions.

## Approved Model Features
- amount: float64, finite, nonnegative
- log_amount: float64, calculated as log1p(amount)
- transaction_type: categorical string

## Training Target
- is_fraud: binary integer (0 or 1)

## Audit Metadata
- source_step: simulation hour, never passed to the model

## Excluded From Model Inputs
- isFraud / is_fraud as input
- isFlaggedFraud
- oldbalanceOrg
- newbalanceOrig
- oldbalanceDest
- newbalanceDest
- nameOrig
- nameDest
- source_step
- simulation_day

## Chronological Split
- Training: steps 1–504
- Validation: steps 505–624
- Test: steps 625–743

## Preprocessing Rules
- Fit categorical encoders only on training data
- Fit numerical scalers only on training data
- Never fit transformations using validation or test data
- Keep the final test set out of hyperparameter tuning
- Reuse the same feature transformations during inference

## Production Limitation
PaySim amounts use simulated mobile-money units.
The application API currently uses synthetic USD.
These domains must be reconciled and validated before
claiming meaningful deployment performance.

## Model Version
Initial feature contract: v1.0
