"""Validation checks for prepared clustering inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_row_lineage(
    reference: pd.DataFrame,
    *matrices: pd.DataFrame,
) -> None:
    """Require identical respondent indexes and ordering."""
    reference_index = reference.index

    if not reference_index.is_unique:
        raise ValueError(
            "Reference respondent index is not unique."
        )

    for matrix in matrices:
        if not matrix.index.equals(reference_index):
            raise ValueError(
                "Respondent alignment failed: a prepared matrix has "
                "a different index or respondent order."
            )


def validate_categorical_matrix(
    matrix: pd.DataFrame,
    matrix_name: str,
) -> None:
    if matrix.empty:
        raise ValueError(f"{matrix_name} is empty.")

    if matrix.columns.duplicated().any():
        raise ValueError(
            f"{matrix_name} contains duplicate column names."
        )

    if matrix.isna().any().any():
        raise ValueError(
            f"{matrix_name} contains missing values."
        )

    if matrix.shape[1] < 2:
        raise ValueError(
            f"{matrix_name} has fewer than two features."
        )

    constant_columns = [
        column
        for column in matrix.columns
        if matrix[column].nunique(dropna=False) < 2
    ]

    if constant_columns:
        raise ValueError(
            f"{matrix_name} contains constant columns: "
            f"{constant_columns}"
        )


def validate_numeric_matrix(
    matrix: pd.DataFrame,
    matrix_name: str,
) -> None:
    if matrix.empty:
        raise ValueError(f"{matrix_name} is empty.")

    values = matrix.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{matrix_name} contains NaN or infinite values."
        )

    constant_columns = [
        column
        for column in matrix.columns
        if matrix[column].nunique(dropna=False) < 2
    ]

    if constant_columns:
        raise ValueError(
            f"{matrix_name} contains constant columns: "
            f"{constant_columns}"
        )


def validate_gower_matrix(
    distance_matrix: np.ndarray,
    tolerance: float = 1e-10,
) -> None:
    matrix = np.asarray(distance_matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Gower distance matrix must be square."
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Gower distance matrix contains non-finite values."
        )

    if not np.allclose(
        matrix, matrix.T, atol=tolerance
    ):
        raise ValueError(
            "Gower distance matrix is not symmetric."
        )

    if not np.allclose(
        np.diag(matrix), 0.0, atol=tolerance
    ):
        raise ValueError(
            "Gower distance matrix diagonal is not zero."
        )

    if (
        matrix.min() < -tolerance
        or matrix.max() > 1.0 + tolerance
    ):
        raise ValueError(
            "Gower dissimilarities must be in [0, 1]."
        )


def validate_model_input_bundle(
    kmodes_matrix: pd.DataFrame,
    mca_input: pd.DataFrame,
    mca_coordinates: pd.DataFrame,
    gower_input: pd.DataFrame,
    gower_distances: np.ndarray,
    umap_embedding: pd.DataFrame,
) -> dict[str, object]:
    validate_categorical_matrix(
        kmodes_matrix, "K-Modes matrix"
    )
    validate_categorical_matrix(
        mca_input, "MCA input"
    )
    validate_numeric_matrix(
        mca_coordinates, "MCA coordinates"
    )
    validate_gower_matrix(gower_distances)
    validate_numeric_matrix(
        umap_embedding, "UMAP embedding"
    )

    validate_row_lineage(
        kmodes_matrix,
        mca_input,
        mca_coordinates,
        gower_input,
        umap_embedding,
    )

    n_rows = len(kmodes_matrix)

    if gower_distances.shape != (n_rows, n_rows):
        raise ValueError(
            "Gower matrix dimensions do not match respondent count."
        )

    if gower_input.empty:
        raise ValueError("Gower input is empty.")

    return {
        "status": "passed",
        "respondent_count": n_rows,
        "kmodes_features": kmodes_matrix.shape[1],
        "mca_input_features": mca_input.shape[1],
        "mca_dimensions": mca_coordinates.shape[1],
        "gower_input_features": gower_input.shape[1],
        "gower_matrix_shape": list(gower_distances.shape),
        "umap_dimensions": umap_embedding.shape[1],
    }
