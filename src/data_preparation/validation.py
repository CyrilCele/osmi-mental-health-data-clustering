"""Study-design and preparation validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DOMAIN_FEATURES,
    EXPECTED_COHORT_SIZE,
    MODEL_FEATURES,
)
from .feature_engineering import validate_encoded_integer_types
from .transformations import validate_distance_matrix


def validate_cohort(
    cohort: pd.DataFrame,
    *,
    expected_size: int = EXPECTED_COHORT_SIZE,
) -> None:
    if len(cohort) != expected_size:
        raise ValueError(
            f"Cohort contains {len(cohort)} respondents; "
            f"expected {expected_size}."
        )

    if "source_row_id" not in cohort:
        raise KeyError("source_row_id is required for lineage.")

    if cohort["source_row_id"].duplicated().any():
        raise ValueError("Duplicate source-row identifiers detected.")


def validate_feature_matrix(
    feature_df: pd.DataFrame,
) -> None:
    if list(feature_df.columns) != MODEL_FEATURES:
        raise ValueError(
            "Feature matrix does not match the approved 15-feature schema."
        )

    if feature_df.isna().any().any():
        missing = feature_df.isna().sum()
        raise ValueError(
            "Primary feature matrix is incomplete:\n"
            f"{missing[missing.gt(0)].to_string()}"
        )

    for feature in MODEL_FEATURES:
        if feature_df[feature].nunique(dropna=True) < 2:
            raise ValueError(
                f"Feature has fewer than two observed categories: {feature}"
            )


def validate_domain_weights() -> None:
    domain_weights = {
        domain: 1.0 / len(DOMAIN_FEATURES)
        for domain in DOMAIN_FEATURES
    }

    if not np.isclose(sum(domain_weights.values()), 1.0):
        raise ValueError("Domain weights do not sum to one.")

    if not np.allclose(
        list(domain_weights.values()),
        np.repeat(1.0 / 3.0, 3),
    ):
        raise ValueError("The three domains must have equal total weight.")


def validate_integer_codes(
    engineered_df: pd.DataFrame,
) -> None:
    validate_encoded_integer_types(engineered_df)


def validate_distance(
    distance: np.ndarray,
    *,
    expected_size: int = EXPECTED_COHORT_SIZE,
) -> None:
    if distance.shape != (expected_size, expected_size):
        raise ValueError(
            f"Unexpected distance shape: {distance.shape}"
        )

    validate_distance_matrix(distance)


def validate_row_lineage(
    source_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> None:
    source_ids = set(source_df["source_row_id"])
    cohort_ids = set(cohort_df["source_row_id"])

    if not cohort_ids.issubset(source_ids):
        raise ValueError(
            "Cohort contains source-row IDs absent from the source data."
        )


def run_preparation_validation(
    source_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    engineered_df: pd.DataFrame,
    distance: np.ndarray,
) -> dict:
    """Run the complete preparation validation suite."""
    validate_cohort(cohort_df)
    validate_feature_matrix(feature_df)
    validate_domain_weights()
    validate_integer_codes(engineered_df)
    validate_distance(distance)
    validate_row_lineage(source_df, cohort_df)

    return {
        "status": "passed",
        "source_rows": int(len(source_df)),
        "cohort_rows": int(len(cohort_df)),
        "model_features": int(len(MODEL_FEATURES)),
        "domain_count": int(len(DOMAIN_FEATURES)),
        "distance_shape": list(distance.shape),
        "distance_min": float(distance.min()),
        "distance_max": float(distance.max()),
        "distance_mean": float(distance.mean()),
        "distance_symmetric": bool(
            np.allclose(distance, distance.T)
        ),
        "distance_diagonal_zero": bool(
            np.allclose(np.diag(distance), 0.0)
        ),
    }
