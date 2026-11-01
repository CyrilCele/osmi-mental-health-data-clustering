"""Algorithm-specific representations for categorical and mixed-data clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd


MISSING_CATEGORY = "__MISSING__"


def _fill_categorical_missing(
    df: pd.DataFrame,
    missing_category: str = MISSING_CATEGORY,
) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        result[column] = (
            result[column]
            .astype("string")
            .fillna(missing_category)
            .astype(object)
        )

    return result


def prepare_kmodes_matrix(
    categorical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare categorical input for K-Modes.

    Missing responses receive an explicit token, not a negative answer.
    If a missing value is known to mean not applicable, that state must be
    represented separately before this function is called.
    """
    if categorical_df.empty:
        raise ValueError("K-Modes input cannot be empty.")

    return _fill_categorical_missing(categorical_df)


def prepare_mca_input(
    categorical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw categorical variables for prince.MCA.

    Do not one-hot encode this input manually; prince.MCA builds its own
    indicator matrix.
    """
    if categorical_df.empty:
        raise ValueError("MCA input cannot be empty.")

    return _fill_categorical_missing(categorical_df)


def fit_mca_coordinates(
    mca_input: pd.DataFrame,
    n_components: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, object]:
    try:
        import prince
    except ImportError as exc:
        raise ImportError(
            "MCA requires the 'prince' package."
        ) from exc

    if n_components < 2:
        raise ValueError("n_components must be at least 2.")

    if len(mca_input) < 3:
        raise ValueError("MCA requires at least three observations.")

    if mca_input.isna().any().any():
        raise ValueError("MCA input still contains missing values.")

    model = prince.MCA(
        n_components=min(n_components, len(mca_input) - 1),
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
        columns=[
            f"mca_dim_{i + 1}"
            for i in range(coordinates.shape[1])
        ],
    )

    return coordinates, model


def prepare_gower_input(
    df: pd.DataFrame,
    categorical_features: list[str],
    ordinal_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    ordinal_features = ordinal_features or []
    numeric_features = numeric_features or []

    all_features = (
        categorical_features + ordinal_features + numeric_features
    )

    if len(all_features) != len(set(all_features)):
        raise ValueError(
            "A feature has been assigned to more than one Gower type."
        )

    missing = [
        column for column in all_features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Gower input columns are missing: {missing}"
        )

    if not all_features:
        raise ValueError("At least one Gower feature is required.")

    result = df.loc[:, all_features].copy()
    variable_types = {}

    for column in categorical_features:
        result[column] = result[column].astype("string")
        variable_types[column] = "categorical"

    for column in ordinal_features:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )
        variable_types[column] = "ordinal"

    for column in numeric_features:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )
        variable_types[column] = "numeric"

    return result, variable_types


def gower_distance_matrix(
    data: pd.DataFrame,
    variable_types: dict[str, str],
) -> np.ndarray:
    """
    Compute a Gower dissimilarity matrix with pairwise missing exclusion.

    Numeric and ordinal variables use range-normalised absolute differences.
    Categorical variables use simple matching dissimilarity.
    Each feature receives equal weight.

    Pairs with no jointly observed features are rejected rather than assigned
    an invented distance.
    """
    if data.empty:
        raise ValueError("Gower input cannot be empty.")

    if set(data.columns) != set(variable_types):
        raise ValueError(
            "variable_types must define every Gower input column exactly."
        )

    allowed_types = {"categorical", "ordinal", "numeric"}

    invalid_types = {
        column: variable_type
        for column, variable_type in variable_types.items()
        if variable_type not in allowed_types
    }

    if invalid_types:
        raise ValueError(
            f"Unsupported Gower variable types: {invalid_types}"
        )

    n_rows = len(data)
    columns = list(data.columns)

    numeric_ranges = {}

    for column in columns:
        if variable_types[column] in {"numeric", "ordinal"}:
            numeric_values = pd.to_numeric(
                data[column], errors="coerce"
            )
            observed_range = (
                numeric_values.max() - numeric_values.min()
            )

            numeric_ranges[column] = (
                float(observed_range)
                if pd.notna(observed_range)
                else 0.0
            )

    values = data.to_dict(orient="list")
    distances = np.zeros((n_rows, n_rows), dtype=float)

    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            total = 0.0
            comparable_features = 0

            for column in columns:
                left = values[column][i]
                right = values[column][j]

                if pd.isna(left) or pd.isna(right):
                    continue

                variable_type = variable_types[column]

                if variable_type == "categorical":
                    dissimilarity = float(str(left) != str(right))
                else:
                    value_range = numeric_ranges[column]

                    if value_range == 0:
                        dissimilarity = 0.0
                    else:
                        dissimilarity = (
                            abs(float(left) - float(right))
                            / value_range
                        )

                total += dissimilarity
                comparable_features += 1

            if comparable_features == 0:
                raise ValueError(
                    "Respondent pair "
                    f"({i}, {j}) has no jointly observed Gower features. "
                    "Investigate missingness or the selected feature set."
                )

            pair_distance = total / comparable_features
            distances[i, j] = pair_distance
            distances[j, i] = pair_distance

    np.fill_diagonal(distances, 0.0)

    return distances


def prepare_umap_input(
    mca_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    values = mca_coordinates.to_numpy(dtype=float)

    if values.size == 0:
        raise ValueError("MCA coordinate matrix is empty.")

    if not np.isfinite(values).all():
        raise ValueError(
            "MCA coordinates contain NaN or infinite values."
        )

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
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP requires the 'umap-learn' package."
        ) from exc

    values = mca_coordinates.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "MCA coordinates contain NaN or infinite values."
        )

    if not 2 <= n_neighbors < len(mca_coordinates):
        raise ValueError(
            "n_neighbors must be at least 2 and smaller than "
            "the number of observations."
        )

    if n_components < 2:
        raise ValueError("n_components must be at least 2.")

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
        columns=[
            f"umap_dim_{i + 1}"
            for i in range(n_components)
        ],
    )

    return embedding, reducer


def apply_structural_applicability(
    df: pd.DataFrame,
    *,
    applicability_masks: dict[str, pd.Series],
    not_applicable_token: str = "__NOT_APPLICABLE__",
) -> pd.DataFrame:
    """
    Mark only confirmed structurally inapplicable missing responses.

    Each mask must be explicitly derived from the survey-routing decisions.
    Missing values outside the supplied masks remain missing.
    """
    result = df.copy()

    for column, mask in applicability_masks.items():
        if column not in result.columns:
            raise KeyError(
                f"Applicability field {column!r} is not in the DataFrame."
            )

        if not mask.index.equals(result.index):
            raise ValueError(
                f"Applicability mask for {column!r} is not row-aligned."
            )

        if mask.isna().any():
            raise ValueError(
                f"Applicability mask for {column!r} contains missing values."
            )

        mask = mask.astype(bool)
        replace_mask = mask & result[column].isna()

        result.loc[replace_mask, column] = not_applicable_token

    return result
