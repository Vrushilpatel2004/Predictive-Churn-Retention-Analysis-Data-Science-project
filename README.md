# Predictive Churn and Retention Analysis with A/B Testing

A recruiter-ready churn analytics project that takes raw telecom customer data from exploration to model training and retention simulation. The workflow identifies which customers are likely to churn, explains the strongest churn drivers, and tests whether a targeted retention offer produces measurable business value.

## Problem Statement
Telecom churn is expensive because losing a customer is usually more costly than targeting them with a retention offer. This project answers three practical questions:

1. Which customers are most likely to churn?
2. Which factors drive churn risk?
3. Does a targeted retention campaign improve outcomes enough to justify rollout?

## Pipeline Flow
1. Week 1 cleans and explores the raw dataset, then exports a consistent modeling file.
2. Week 2 builds preprocessing pipelines, trains several churn models, tunes the best performer, and saves model metadata.
3. Week 3 scores churn risk, selects a high-risk segment, simulates an A/B retention test, and converts the result into a business recommendation.

## Results
- Best saved model: Tuned Random Forest
- Test set performance: accuracy 0.7587, precision 0.5312, recall 0.7727, F1 0.6296, ROC-AUC 0.8417
- Close benchmark: Logistic Regression ROC-AUC 0.8421
- Week 3 experiment output: statistically significant uplift with p-value 0.0020
- Top churn drivers: month-to-month contract, tenure, total charges, two-year contract, online security

## Project Structure
- [README.md](/Users/vrushil/Desktop/Temple%20University/Principle%20of%20Data%20Science/Predictive-Churn-Retention-Analysis-Data-Science-project/README.md)
- `data/`
  - `WA_Fn-UseC_-Telco-Customer-Churn.csv`: raw dataset
  - `cleaned_churn_data.csv`: cleaned dataset used in modeling
- `notebooks/`
  - `weekone.ipynb`: data cleaning and exploratory analysis
  - `week2.ipynb`: preprocessing, model training, and model comparison
  - `week3.ipynb`: A/B testing and retention simulation
- `models/`
  - `week2_best_model.joblib`: saved tuned model artifact
  - `model_info.json`: model metadata and threshold settings
- `src/`
  - `preprocessing.py`: shared preprocessing utilities
  - `model.py`: model interpretation and report generation helpers
  - `ab_testing.py`: thresholding, uplift, and campaign economics helpers
- `reports/`
  - `week2_model_comparison.csv`: model benchmark table
  - `model_interpretation.md`: business-facing explanation of the final model
  - `week3_ab_test_summary.csv`: A/B test summary metrics
  - `week3_high_risk_simulation.csv`: customer-level simulation output
  - `week3_business_recommendation.csv`: rollout recommendation and economics
  - `figures/week2_top_feature_importances.png`: model feature importance chart

## How To Run
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run the notebooks in this order:

```text
notebooks/weekone.ipynb
notebooks/week2.ipynb
notebooks/week3.ipynb
```

4. Run each notebook top to bottom.

Week 1 produces the cleaned dataset, Week 2 trains and saves the model, and Week 3 reads that model to generate the A/B test outputs and recommendation.

## What Each Week Does
### Week 1: Data Cleaning and Exploration
- Loads the raw telecom churn data
- Checks for missing values and data type issues
- Engineers a few basic fields
- Produces exploratory plots
- Exports the cleaned dataset for downstream notebooks

### Week 2: Model Training and Selection
- Builds a reusable preprocessing pipeline
- Compares baseline models
- Tunes Random Forest with cross-validation
- Saves the best model and metrics
- Writes a model interpretation report for business review

### Week 3: Retention Simulation
- Scores churn probability for all customers
- Identifies a high-risk segment
- Simulates control and treatment groups
- Tests whether the intervention is statistically significant
- Estimates the campaign recommendation and business impact

## Why The Helper Files Exist
- `src/preprocessing.py` keeps preprocessing logic consistent across notebooks.
- `src/model.py` turns trained model outputs into readable explanations and reports.
- `src/ab_testing.py` centralizes Week 3 thresholding, uplift, and campaign economics.

These files are imported by the notebooks and are not meant to be run directly.

## Business Takeaway
The strongest churn signals are contract type and tenure, which suggests the highest-value retention effort should focus on short-tenure, month-to-month customers first. The Week 3 A/B test output indicates the intervention can produce a statistically significant retention lift, so the campaign is worth considering for rollout with cost monitoring.