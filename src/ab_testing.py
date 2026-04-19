"""Utilities for retention simulation and campaign economics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def derive_risk_threshold(
    churn_probabilities,
    metadata_threshold: float | None = None,
    fallback_quantile: float = 0.75,
    lower_bound: float = 0.55,
    upper_bound: float = 0.90,
) -> dict:
    """Return a configurable risk threshold using metadata or probability quantiles."""
    series = pd.Series(churn_probabilities).dropna()
    if series.empty:
        raise ValueError("Churn probability series is empty.")

    quantile_value = float(series.quantile(fallback_quantile))
    median_value = float(series.quantile(0.50))
    p75_value = float(series.quantile(0.75))
    p90_value = float(series.quantile(0.90))

    if metadata_threshold is not None and np.isfinite(metadata_threshold):
        threshold = float(metadata_threshold)
        source = "Week 2 metadata optimal threshold"
    else:
        threshold = float(np.clip(quantile_value, lower_bound, upper_bound))
        source = f"{fallback_quantile:.0%} churn-probability quantile"

    return {
        "threshold": threshold,
        "source": source,
        "fallback_quantile": fallback_quantile,
        "quantile_value": quantile_value,
        "median": median_value,
        "p75": p75_value,
        "p90": p90_value,
    }


def derive_campaign_uplift(
    churn_probabilities,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
    min_uplift: float = 0.02,
    max_uplift: float = 0.10,
) -> dict:
    """Derive a simulated uplift from the spread of the churn-probability distribution."""
    series = pd.Series(churn_probabilities).dropna()
    if series.empty:
        raise ValueError("Churn probability series is empty.")

    lower_value = float(series.quantile(low_quantile))
    upper_value = float(series.quantile(high_quantile))
    spread = max(upper_value - lower_value, 0.0)
    uplift = float(np.clip(spread / 2.0, min_uplift, max_uplift))

    return {
        "uplift": uplift,
        "source": f"scaled from the {high_quantile:.0%}-{low_quantile:.0%} churn-probability spread",
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "low_value": lower_value,
        "high_value": upper_value,
        "spread": spread,
    }


def estimate_campaign_economics(
    lift: float,
    treatment_n: int,
    offer_cost: float,
    customer_value: float,
) -> dict:
    """Estimate incremental value, net profit, and ROI for a retention campaign."""
    incremental_retained = max(0.0, float(lift)) * int(treatment_n)
    incremental_value = incremental_retained * float(customer_value)
    total_offer_cost = int(treatment_n) * float(offer_cost)
    estimated_net_profit = incremental_value - total_offer_cost
    roi = estimated_net_profit / total_offer_cost if total_offer_cost > 0 else np.nan

    return {
        "incremental_retained": incremental_retained,
        "incremental_value": incremental_value,
        "total_offer_cost": total_offer_cost,
        "estimated_net_profit": estimated_net_profit,
        "roi": roi,
    }


def build_distribution_basis_note(risk_threshold_info: dict, uplift_info: dict) -> str:
    """Create a short note explaining the data-driven threshold and uplift settings."""
    return (
        f"Risk threshold source: {risk_threshold_info['source']} (q50={risk_threshold_info['median']:.2f}, "
        f"q75={risk_threshold_info['p75']:.2f}, q90={risk_threshold_info['p90']:.2f}); "
        f"uplift source: {uplift_info['source']} (q{uplift_info['high_quantile']:.0%}={uplift_info['high_value']:.2f}, "
        f"q{uplift_info['low_quantile']:.0%}={uplift_info['low_value']:.2f})."
    )
