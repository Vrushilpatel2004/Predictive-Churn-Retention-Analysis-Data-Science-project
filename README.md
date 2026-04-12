# Predictive Churn and Retention Analysis with A/B Testing

This project builds an end-to-end churn analytics workflow:
- Explore and clean telecom customer data
- Train and compare churn prediction models
- Identify high-risk customers
- Simulate a retention intervention via A/B testing
- Evaluate statistical significance and business impact

## Project Structure
- `data/`
	- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: raw dataset
	- `cleaned_churn_data.csv`: cleaned dataset used in modeling
- `notebooks/`
	- `weekone.ipynb`: EDA and preprocessing
	- `week2.ipynb`: feature engineering and model building
	- `week3.ipynb`: A/B testing and retention simulation
- `models/`
	- `week2_best_model.joblib`: best model artifact from Week 2
- `reports/`
	- `week2_model_comparison.csv`
	- `week3_ab_test_summary.csv`
	- `week3_high_risk_simulation.csv`
	- `figures/week2_top_feature_importances.png`

## Setup
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Execution Order
Run notebooks in this order:
1. `notebooks/weekone.ipynb`
2. `notebooks/week2.ipynb`
3. `notebooks/week3.ipynb`

## Workflow Summary
### Week 1: EDA and Preprocessing
- Load Telco dataset
- Perform basic data quality checks
- Handle missing values and type conversion
- Create exploratory visualizations
- Export cleaned dataset for downstream notebooks

### Week 2: Churn Modeling
- Build preprocessing pipelines
- Train baseline models:
	- Logistic Regression
	- Random Forest
	- Gradient Boosting
	- XGBoost (if installed)
- Tune Random Forest with cross-validation
- Evaluate with classification metrics, confusion matrix, ROC
- Save best model and model comparison report

### Week 3: A/B Testing and Retention Simulation
- Score customer churn risk using the trained model
- Select high-risk segment
- Assign customers to control/treatment groups
- Simulate retention outcomes under intervention
- Evaluate uplift significance using two-proportion z-test
- Export summary and simulated experiment data

## Key Outputs
- Predictive churn model (`models/week2_best_model.joblib`)
- Feature importance and model comparison reports
- A/B test significance metrics (lift, p-value, confidence interval)
- Actionable retention recommendation based on experiment outcome

## Business Interpretation
If treatment retention is significantly higher than control (`p < 0.05`),
the intervention should be considered for rollout to high-risk customers,
with ongoing monitoring of conversion, cost, and long-term retention impact.