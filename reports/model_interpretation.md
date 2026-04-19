# Churn Model Interpretation

## Model Comparison Justification

TunedRandomForest was kept even though Logistic Regression has a very similar ROC-AUC, because churn is a recall-sensitive problem. The tuned forest improves recall by 21.39%, which means fewer churners are missed at the intervention stage.

ROC-AUC is effectively tied here as well (0.8417 vs 0.8421, gap -0.0004), so the selection is driven by decision quality after thresholding rather than ranking alone.

Random Forest also captures non-linear feature interactions and mixed effects across contract type, tenure, and service add-ons, which makes it a better fit for churn behavior than a purely linear boundary.

Precision is lower by 12.60% compared with Logistic Regression, but that trade-off is acceptable because the retention program is cheaper than losing a customer and the business priority is to catch likely churners early.

## Top Churn Drivers

- **cat__Contract_Month-to-month** (0.134): Month-to-month customers are the least locked in, so they are more likely to leave quickly.
- **num__tenure** (0.124): Short tenure is a strong churn signal because newer customers have had less time to become sticky.
- **num__TotalCharges** (0.092): Total charges proxy customer lifetime value and account maturity; low values often align with newer accounts.
- **cat__Contract_Two year** (0.071): Two-year contracts reduce churn by creating a stronger commitment and higher switching cost.
- **cat__OnlineSecurity_No** (0.071): Customers without online security are less bundled into the product and can be easier to lose.

## Business Takeaway

The strongest churn drivers are mostly contract structure and customer tenure variables. That means retention action should prioritize short-tenure, month-to-month customers before expanding to lower-risk segments.
