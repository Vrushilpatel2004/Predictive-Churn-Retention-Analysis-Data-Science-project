"""Reusable preprocessing utilities for churn modeling."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _to_object_dtype(x):
    """Cast incoming categorical block to object dtype for sklearn imputers."""
    return x.astype(object)


def infer_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical feature columns from a DataFrame."""
    # Keep bool columns in the categorical path (after casting to object),
    # because sklearn SimpleImputer does not accept raw bool dtype.
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number], exclude=["bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build a reusable sklearn preprocessing ColumnTransformer."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "to_object",
                FunctionTransformer(_to_object_dtype, validate=False, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
