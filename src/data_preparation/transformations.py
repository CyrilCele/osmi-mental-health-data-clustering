"""Build algorityhm-specific representations for clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances


def prepare_kmodes_matrix(
    categorical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a categorical-only matrix for K-Modes.

    Missing values are represented by an explicit category rather than
    imputed as No. this preserves the distinction between a negative answer
    and an unanswered response.
    """
    result = categorical_df.copy()

    for column in result.columns:
        result[column] = (
            result[column]
            .astype("string")
            .fillna("__MISSING__")
        )

    return result.astype(object)


def prepare_mca_input(
    categorical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return raw categorical columns for prince.MCA.

    Do not one-hot encode this DataFrame before passing it to prince.MCA:
    MCA performs its own indicator-table construction internally.
    """
    result = categorical_df.copy()

    for column in result.columns:
        result[column] = (
            result[column]
            .astype("string")
            .fillna("__MISSING__")
        )

    return result


def fit_mca_coordinates(
    mca_input: pd.DataFrame,
    n_components: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, object]:
    """
    Fit MCA abd return respondent coordinates.

    prince is imported inside the function si the package can be imported
    before the optional dependency is installed.
    """
    try:
        import prince
    except ImportError as exc:
        raise ImportError(
            "MCA requires the 'prince' package. Install it in the project "
            "environment before running the preparation notebook."
        ) from exc

    n_components = min(
        n_components,
        max(1, mca_input.shape[1]),
        max(1, mca_input.shape[0] - 1),
    )

    model = prince.MCA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine="sklearn",
        random_state=random_state,
    )

    coordinates = model.fit_transform(mca_input)
    coordinates = pd.DataFrame(
        coordinates,
        index=mca_input.index,
        columns=[f"mca_dim_{i + 1}" for i in range(coordinates.shape[1])],
    )

    return coordinates, model


def prepare_gower_input(
    df: pd.DataFrame,
    categorical_features: list[str],
    ordinal_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Construct a mixed-data input table and its variable-type metadata.

    The initial feature set is categorical. Ordinal/numeric variables are
    added only when explicitly supplied. Missing values remain missing so
    that the Gower calculation can exclude unavailable feature-pairs.
    """
    ordinal_features = ordinal_features or []
    numeric_features = numeric_features or []

    features = list(dict.fromkeys(
        categorical_features + ordinal_features + numeric_features
    ))
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Gower input columns missing from dataset: {missing}"
        )

    result = df.loc[:, features].copy()
    types = {}

    for column in categorical_features:
        result[column] = result[column].astype("string")
        types[column] = "categorical"

    for column in ordinal_features:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        types[column] = "ordinal"

    for column in numeric_features:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        types[column] = "numeric"

    return result, types


def gower_distance_matrix(
    data: pd.DataFrame,
    variable_types: dict[str, str],
) -> np.ndarray:
    """
    Compute Gower dissimilarities with pairwise missing-value exclusion.

    Categorical dissimilarity: 0 if equal, 1 if different.
    Numeric/ordinal dissimilarity: absolute difference divided by observed range.
    Pairwise feature weights are equal. Missing values are omitted from the
    numerator and denominator for that respondent pair.

    This implementation treats binary variables as nominal categorical
    variables. It does not assume asymmetric-binary semantics.
    """
    columns = list(data.columns)
    n = len(data)
    result = np.zeros((n, n), dtype=float)

    ranges = {}
    for column in columns:
        if variable_types[column] in {"numeric", "ordinal"}:
            values = pd.to_numeric(data[column], errors="coerce")
            value_range = values.max() - values.min()
            ranges[column] = float(value_range) if pd.notna(
                value_range) else 0.0

    for i in range(n):
        for j in range(i + 1, n):
            distances = []

            for column in columns:
                left = data.iloc[i][column]
                right = data.iloc[j][column]

                if pd.isna(left) or pd.isna(right):
                    continue

                variable_type = variable_types[column]

                if variable_type == "categorical":
                    distance = float(str(left) != str(right))
                else:
                    value_range = ranges[column]
                    if value_range == 0:
                        distance = 0.0
                    else:
                        distance = abs(
                            float(left) - float(right)) / value_range

                distances.append(distance)

            pair_distance = float(np.mean(distances)) if distances else np.nan
            result[i, j] = pair_distance
            result[j, i] = pair_distance

    if np.isnan(result).any():
        raise ValueError(
            "At least one respondent pair has no jointly observed features. "
            "Resolve the affected rows/features before calculating Gower distances."
        )

    np.fill_diagonal(result, 0.0)
    return result


def prepare_umap_input(
    mca_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and return the continuous MCA coordinates used by UMAP."""
    values = mca_coordinates.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("MCA coordinates contain NaN or infinite values.")

    return mca_coordinates.copy()


def fit_umap_embedding(
    mca_coordinates: pd.DataFrame,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int = 42,
) -> tuple[pd.DataFrame, object]:
    """
    Fit UMAP to MCA coordinates and return a respondent-aligned embedding.

    The two-dimensional embedding is intended for density-based clustering
    and visual inspection. UMAP parameters must be held fixed when comparing
    HDBSCAN runs so that changes in clustering are not confounded by changes
    in the embedding.
    """
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP requires 'umap-learn'. Install it in the project environment."
        ) from exc

    values = mca_coordinates.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("MCA coordinates contain NaN or infinite values.")

    if len(mca_coordinates) <= n_neighbors:
        raise ValueError(
            f"UMAP n_neighbors={n_neighbors} must be smaller that the "
            f"number of respondents ({len(mca_coordinates)})."
        )

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
    )

    embedding_values = reducer.fit_transform(values)

    embedding = pd.DataFrame(
        embedding_values,
        index=mca_coordinates.index,
        columns=[f"umap_dim_{i + 1}" for i in range(n_components)],
    )

    return embedding, reducer
