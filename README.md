# Customer Churn Prediction & Explainability

End-to-end machine learning pipeline to predict and explain
customer churn for a telecom company, using XGBoost and SHAP values
to generate actionable business insights.

## Results

| Model               | AUC-ROC   | F1        | Recall    |
| ------------------- | --------- | --------- | --------- |
| Logistic Regression | 0.841     | 0.613     | 0.781     |
| Random Forest       | 0.828     | 0.552     | 0.492     |
| **XGBoost (tuned)** | **0.84+** | **0.638** | **0.767** |

Optimal decision threshold: **0.546**

## Key Findings

- **Contract type** is the strongest churn predictor — month-to-month
  customers churn at 43% vs 3% on two-year contracts
- **Tenure** is the second strongest driver — new customers (<6 months)
  are the highest-risk segment regardless of other factors
- **Service add-ons** (OnlineSecurity, TechSupport) act as retention
  anchors — customers without them churn at ~2x the rate
- **Fiber optic + electronic check** compound churn risk independently
  of contract type

## Business Recommendations

1. Offer contract upgrade incentives to month-to-month customers
   within the first 90 days — contract + tenure together contribute
   the largest SHAP values by a wide margin
2. Bundle OnlineSecurity and TechSupport into onboarding —
   add-on adoption significantly reduces long-term churn
3. Review pricing strategy for high monthly charge segments —
   price sensitivity compounds churn risk without contract lock-in

## Project Structure

```
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_prediction.ipynb
└── README.md
```

## Stack

Python · Pandas · Scikit-learn · XGBoost · SHAP ·
Matplotlib · Seaborn · Jupyter Notebook

## Dataset

[Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
