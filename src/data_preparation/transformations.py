"""Approved domain-balanced Gower-style dissimilarity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DOMAIN_FEATURES,
    MEDICAL_LEAVE_ORDER,
    MEDICAL_LEAVE_UNKNOWN,
    MODEL_FEATURES,
)


def _nominal_distance(left, right) -> float:
    """Simple matching distance for nominal responses."""
    if pd.isna(left) or pd.isna(right):
        raise ValueError(
            "Missing values are not allowed in the primary feature matrix."
        )

    return 0.0 if left == right else 1.0


def _ordinal_distance(
    left,
    right,
    *,
    n_levels: int,
) -> float:
    """Normalised absolute distance between ordinal categories."""
    if pd.isna(left) or pd.isna(right):
        raise ValueError(
            "Missing values are not allowed in the primary feature matrix."
        )

    return abs(int(left) - int(right)) / (n_levels - 1)


def _medical_leave_distance(left, right) -> float:
    """
    Medical-leave dissimilarity.

    The five substantive categories are ordinal.
    'I don't know' is a distinct state:
      - unknown versus unknown = 0
      - unknown versus a substantive response = 1
    """
    left_unknown = left == MEDICAL_LEAVE_UNKNOWN
    right_unknown = right == MEDICAL_LEAVE_UNKNOWN

    if left_unknown or right_unknown:
        return 0.0 if left_unknown and right_unknown else 1.0

    order = {
        response: position
        for position, response in enumerate(MEDICAL_LEAVE_ORDER)
    }

    if left not in order or right not in order:
        raise ValueError(
            f"Unexpected medical-leave values: {left!r}, {right!r}"
        )

    return abs(order[left] - order[right]) / (
        len(MEDICAL_LEAVE_ORDER) - 1
    )


def _feature_distance(feature, left, right) -> float:
    """Select the correct dissimilarity for a feature."""
    if feature == "medical_leave_ease":
        return _medical_leave_distance(left, right)

    ordinal_map = {
        "mh_disclosure_negative_consequences": {
            "No": 0, "Maybe": 1, "Yes": 2
        },
        "physical_disclosure_negative_consequences": {
            "No": 0, "Maybe": 1, "Yes": 2
        },
        "coworker_discussion_comfort": {
            "No": 0, "Maybe": 1, "Yes": 2
        },
        "supervisor_discussion_comfort": {
            "No": 0, "Maybe": 1, "Yes": 2
        },
        "mh_as_serious_as_physical": {
            "No": 0, "Maybe": 1, "Yes": 2
        },
        "interview_disclosure_physical": {
            "No": 0, "Maybe": 1, "Yes": 2
        },
        "interview_disclosure_mental": {
            "No": 0, "Maybe": 1, "Yes": 2
        },
    }

    if feature in ordinal_map:
        mapping = ordinal_map[feature]

        if left not in mapping or right not in mapping:
            raise ValueError(
                f"Unexpected ordinal values for {feature}: "
                f"{left!r}, {right!r}"
            )

        return _ordinal_distance(
            mapping[left],
            mapping[right],
            n_levels=len(mapping),
        )

    return _nominal_distance(left, right)


def feature_distance_matrix(
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Calculate one respondent-by-respondent matrix per feature."""
    missing = sorted(set(MODEL_FEATURES) - set(df.columns))

    if missing:
        raise KeyError(f"Missing model features: {missing}")

    matrix = {}

    for feature in MODEL_FEATURES:
        values = df[feature].to_numpy(dtype=object)
        n = len(values)

        distance = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                value = _feature_distance(
                    feature,
                    values[i],
                    values[j],
                )
                distance[i, j] = value
                distance[j, i] = value

        matrix[feature] = distance

    return matrix


def domain_distance_matrices(
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Calculate mean dissimilarity within each approved domain."""
    feature_matrices = feature_distance_matrix(df)

    return {
        domain: np.mean(
            [
                feature_matrices[feature]
                for feature in features
            ],
            axis=0,
        )
        for domain, features in DOMAIN_FEATURES.items()
    }


def gower_distance_matrix(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Construct the domain-balanced dissimilarity matrix.

    Each domain contributes one-third to the final distance,
    regardless of the number of features within that domain.
    """
    missing = sorted(set(MODEL_FEATURES) - set(df.columns))

    if missing:
        raise KeyError(f"Missing model features: {missing}")

    model_df = df.loc[:, MODEL_FEATURES]

    if model_df.isna().any().any():
        raise ValueError(
            "Primary features contain missing values. "
            "The approved design does not permit silent imputation."
        )

    domain_matrices = domain_distance_matrices(model_df)

    final_matrix = np.mean(
        list(domain_matrices.values()),
        axis=0,
    )

    np.fill_diagonal(final_matrix, 0.0)

    return final_matrix


def validate_distance_matrix(
    distance: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> None:
    """Check the mathematical properties of the dissimilarity matrix."""
    if distance.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    if distance.shape[0] != distance.shape[1]:
        raise ValueError("Distance matrix must be square.")

    if not np.isfinite(distance).all():
        raise ValueError("Distance matrix contains non-finite values.")

    if np.any(distance < -tolerance):
        raise ValueError("Distance matrix contains negative values.")

    if not np.allclose(
        distance,
        distance.T,
        atol=tolerance,
    ):
        raise ValueError("Distance matrix is not symmetric.")

    if not np.allclose(
        np.diag(distance),
        0.0,
        atol=tolerance,
    ):
        raise ValueError("Distance matrix diagonal is not zero.")

    if np.any(distance > 1.0 + tolerance):
        raise ValueError("Dissimilarities must lie in [0, 1].")
