# FraudGuard RiskOps — EDA Findings

## Dataset
- Source: PaySim synthetic financial transactions
- Version: v1
- Transactions: 6,362,620
- Fraud cases: 8,213
- Fraud rate: 0.1291%

## Data Quality
- Missing values:
- Invalid amounts:
- Schema validation:
- Manifest reconciliation:

## Transaction Type Analysis
- Highest fraud-rate transaction type:
- Types containing no labeled fraud:
- Possible synthetic-data limitations:

## Amount Analysis
- Amount band with highest fraud rate:
- Amount band with highest fraud count:
- Implications for feature engineering:

## Temporal Analysis
- Fraud rate over simulation days:
- Transaction volume changes:
- Implications for chronological validation:

## ML Recommendations
- Use PR-AUC, precision, recall, and recall at review capacity
- Build chronologically separated datasets
- Exclude leakage-prone balance fields
- Fit preprocessing only on training data
- Keep the final test dataset untouched during tuning
- Verify point-in-time availability of every feature

## Limitations
The source data is synthetic and does not establish
real-world fraud detection performance.