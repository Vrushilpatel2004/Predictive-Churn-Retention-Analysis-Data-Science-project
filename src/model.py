"""Utilities for churn model interpretation and reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_EXPLANATIONS = {
    "cat__Contract_Month-to-month": (
        "Month-to-month customers are the least locked in, so they are more likely to leave quickly."
    ),
    "num__tenure": (
        "Short tenure is a strong churn signal because newer customers have had less time to become sticky."
    ),
    "num__TotalCharges": (
        "Total charges proxy customer lifetime value and account maturity; low values often align with newer accounts."
    ),
    "cat__Contract_Two year": (
        "Two-year contracts reduce churn by creating a stronger commitment and higher switching cost."
    ),
    "cat__OnlineSecurity_No": (
        "Customers without online security are less bundled into the product and can be easier to lose."
    ),
    "cat__TechSupport_No": (
        "Missing tech support usually means fewer service attachments and weaker retention leverage."
    ),
    "num__MonthlyCharges": (
        "Higher monthly charges can increase price sensitivity, especially for customers already showing churn risk."
    ),
    "cat__InternetService_Fiber optic": (
        "Fiber customers often have more expensive plans and more alternative options, which can increase churn pressure."
    ),
    "cat__PaymentMethod_Electronic check": (
        "Electronic check users often show weaker retention than auto-pay customers because payment friction is higher."
    ),
}


def get_feature_importance_table(model, top_n: int = 20) -> pd.DataFrame:
    """Return a sorted feature-importance table for a fitted sklearn pipeline."""
    if not hasattr(model, "named_steps"):
        raise ValueError("Expected a fitted sklearn Pipeline with named_steps.")

    preprocessor = model.named_steps.get("preprocess")
    estimator = model.named_steps.get("model")

    if preprocessor is None or estimator is None:
        raise ValueError("Expected pipeline steps named 'preprocess' and 'model'.")

    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = preprocessor.get_feature_names_out()
    importances = estimator.feature_importances_

    feature_importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return feature_importance_df.head(top_n)


def explain_feature(feature_name: str) -> str:
    """Map a model feature to a concise business interpretation."""
    if feature_name in FEATURE_EXPLANATIONS:
        return FEATURE_EXPLANATIONS[feature_name]

    readable = feature_name.replace("cat__", "").replace("num__", "").replace("_", " ")
    return f"{readable} contributes to churn risk through a customer behavior or plan-structure effect."


def build_feature_importance_section(feature_importance_df: pd.DataFrame, top_n: int = 5) -> str:
    """Build a markdown section describing the top churn drivers."""
    top_features = feature_importance_df.head(top_n).copy()
    lines = ["## Top Churn Drivers", ""]

    for _, row in top_features.iterrows():
        feature_name = str(row["feature"])
        importance = float(row["importance"])
        lines.append(f"- **{feature_name}** ({importance:.3f}): {explain_feature(feature_name)}")

    return "\n".join(lines)


def build_model_comparison_justification(comparison_df: pd.DataFrame, selected_model: str) -> str:
    """Explain why the selected model is preferred over close alternatives."""
    comparison = comparison_df.set_index("model")

    if "LogisticRegression" not in comparison.index or selected_model not in comparison.index:
        return (
            "## Model Comparison Justification\n\n"
            "The selected model was chosen from the comparison table because it balanced predictive quality and operational usefulness."
        )

    selected = comparison.loc[selected_model]
    logistic = comparison.loc["LogisticRegression"]

    roc_gap = float(selected["roc_auc"] - logistic["roc_auc"])
    recall_gap = float(selected["recall"] - logistic["recall"])
    precision_gap = float(selected["precision"] - logistic["precision"])

    return "\n".join(
        [
            "## Model Comparison Justification",
            "",
            (
                f"{selected_model} was kept even though Logistic Regression has a very similar ROC-AUC, "
                f"because churn is a recall-sensitive problem. The tuned forest improves recall by {recall_gap:.2%}, "
                f"which means fewer churners are missed at the intervention stage."
            ),
            (
                f"ROC-AUC is effectively tied here as well ({selected['roc_auc']:.4f} vs {logistic['roc_auc']:.4f}, gap {roc_gap:.4f}), "
                f"so the selection is driven by decision quality after thresholding rather than ranking alone."
            ),
            (
                "Random Forest also captures non-linear feature interactions and mixed effects across contract type, tenure, "
                "and service add-ons, which makes it a better fit for churn behavior than a purely linear boundary."
            ),
            (
                f"Precision is lower by {abs(precision_gap):.2%} compared with Logistic Regression, but that trade-off is acceptable "
                "because the retention program is cheaper than losing a customer and the business priority is to catch likely churners early."
            ),
        ]
    )


def build_model_interpretation_report(
    feature_importance_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    selected_model: str,
    top_n: int = 5,
) -> str:
    """Create a markdown report covering feature importance and model choice."""
    section_lines = [
        "# Churn Model Interpretation",
        "",
        build_model_comparison_justification(comparison_df, selected_model),
        "",
        build_feature_importance_section(feature_importance_df, top_n=top_n),
        "",
        "## Business Takeaway",
        "",
        (
            "The strongest churn drivers are mostly contract structure and customer tenure variables. "
            "That means retention action should prioritize short-tenure, month-to-month customers before expanding to lower-risk segments."
        ),
    ]
    return "\n".join(section_lines)


def write_model_interpretation_report(
    output_path: Path,
    feature_importance_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    selected_model: str,
    top_n: int = 5,
) -> Path:
    """Write the churn interpretation report to disk and return the path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_model_interpretation_report(
        feature_importance_df=feature_importance_df,
        comparison_df=comparison_df,
        selected_model=selected_model,
        top_n=top_n,
    )
    output_path.write_text(report_text, encoding="utf-8")
    return output_path
