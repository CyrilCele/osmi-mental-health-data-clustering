"""Input loading and representation transformations for clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import umap

from prince import MCA
from sklearn.preprocessing import OneHotEncoder

from src.data_preparation.config import (
    DOMAIN_FEATURES,
    MODEL_FEATURES,
)
from src.data_preparation.transformations import validate_distance_matrix

from . import config


@dataclass(frozen=True)
class ModelingDataset:
    """Validated modelling inputs and respondent lineage."""

    features: pd.DataFrame
    distances: np.ndarray
    row_mapping: pd.DataFrame

    @property
    def n_samples(self) -> int:
        return len(self.features)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]


@dataclass
class MCAResult:
    """Fitted MCA model and its respondent coordinates."""

    estimator: MCA
    coordinates: pd.DataFrame
    eigenvalues: np.ndarray
    explained_inertia: np.ndarray


@dataclass
class UMAPResult:
    """Fitted UMAP model and resulting embedding."""

    estimator: umap.UMAP
    coordinates: np.ndarray
    parameters: dict


class ModelingDataLoader:
    """Loads and validates the exact outputs of Data Preparation."""

    def __init__(self) -> None:
        self.feature_path = config.FEATURE_MATRIX_PATH
        self.distance_path = config.DISTANCE_MATRIX_PATH
        self.mapping_path = config.ROW_MAPPING_PATH

    def load(self) -> ModelingDataset:
        self._validate_files()

        features = pd.read_csv(self.feature_path)
        distances = np.load(
            self.distance_path,
            allow_pickle=False,
        )
        mapping = pd.read_csv(self.mapping_path)

        self._validate_features(features)
        self._validate_mapping(mapping, len(features))
        self._validate_distances(distances, len(features))

        return ModelingDataset(
            features=features,
            distances=distances.astype(np.float64, copy=False),
            row_mapping=mapping,
        )

    def _validate_files(self) -> None:
        required = (
            self.feature_path,
            self.distance_path,
            self.mapping_path,
        )

        missing = [path for path in required if not path.is_file()]

        if missing:
            raise FileNotFoundError(
                "Missing Data Preparation artefacts:\n"
                + "\n".join(str(path) for path in missing)
            )

    @staticmethod
    def _validate_features(features: pd.DataFrame) -> None:
        if features.empty:
            raise ValueError("The model feature matrix is empty.")

        if list(features.columns) != list(MODEL_FEATURES):
            raise ValueError(
                "Feature columns do not match the approved "
                "MODEL_FEATURES definition and order."
            )

        if features.shape != (
            config.EXPECTED_N_RESPONDENTS,
            config.EXPECTED_N_FEATURES,
        ):
            raise ValueError(
                f"Expected "
                f"{config.EXPECTED_N_RESPONDENTS} respondents and "
                f"{config.EXPECTED_N_FEATURES} features; "
                f"received {features.shape}."
            )

        if features.isna().any().any():
            raise ValueError(
                "Primary model features contain missing values. "
                "Resolve this in Data Preparation."
            )

        if features.columns.duplicated().any():
            raise ValueError("Duplicate feature names detected.")

        # The preparation output is integer encoded. Preserve its values
        # while presenting them as categorical values to MCA and K-Modes.
        if not all(
            pd.api.types.is_integer_dtype(dtype)
            for dtype in features.dtypes
        ):
            raise TypeError(
                "The prepared feature matrix must contain integer-coded "
                "categorical features."
            )

    @staticmethod
    def _validate_mapping(
        mapping: pd.DataFrame,
        n_samples: int,
    ) -> None:
        required = {
            config.ROW_POSITION_COLUMN,
            config.SOURCE_ROW_COLUMN,
        }

        if not required.issubset(mapping.columns):
            raise KeyError(
                f"Row mapping must contain {sorted(required)}."
            )

        if len(mapping) != n_samples:
            raise ValueError(
                "Feature matrix and row mapping have different lengths."
            )

        expected_positions = np.arange(n_samples)

        observed_positions = mapping[
            config.ROW_POSITION_COLUMN
        ].to_numpy()

        if not np.array_equal(observed_positions, expected_positions):
            raise ValueError(
                "Row mapping positions are not consecutive and ordered."
            )

        if mapping[config.SOURCE_ROW_COLUMN].duplicated().any():
            raise ValueError("Duplicate source respondent IDs detected.")

    @staticmethod
    def _validate_distances(
        distances: np.ndarray,
        n_samples: int,
    ) -> None:
        if distances.shape != (n_samples, n_samples):
            raise ValueError(
                "Distance matrix dimensions do not match "
                "the feature matrix."
            )

        validate_distance_matrix(
            distances,
            tolerance=config.MATRIX_TOLERANCE,
        )


class MCATransformer:
    """Transforms the categorical feature matrix into MCA coordinates."""

    def __init__(
        self,
        n_components: int = config.MCA_COMPONENTS,
        random_state: int = config.SEED,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.model: MCA | None = None
        self.result: MCAResult | None = None

    def fit_transform(self, features: pd.DataFrame) -> MCAResult:
        if features.isna().any().any():
            raise ValueError("MCA input contains missing values.")

        categorical = features.astype("string")

        maximum_components = min(
            categorical.shape[0] - 1,
            categorical.shape[1],
        )

        components = min(self.n_components, maximum_components)

        if components < 2:
            raise ValueError("Insufficient dimensions for MCA.")

        self.model = MCA(
            n_components=components,
            n_iter=config.MCA_ITERATIONS,
            copy=True,
            check_input=True,
            random_state=self.random_state,
            engine="sklearn",
        )

        self.model.fit(categorical)

        coordinates = self.model.row_coordinates(categorical)
        coordinates = pd.DataFrame(
            coordinates.to_numpy(dtype=np.float64),
            index=features.index,
            columns=[
                f"mca_{index + 1}"
                for index in range(coordinates.shape[1])
            ],
        )

        eigenvalues = np.asarray(
            self.model.eigenvalues_,
            dtype=np.float64,
        )

        total_inertia = float(self.model.total_inertia_)

        explained = (
            eigenvalues / total_inertia
            if total_inertia > 0
            else np.full_like(eigenvalues, np.nan)
        )

        self.result = MCAResult(
            estimator=self.model,
            coordinates=coordinates,
            eigenvalues=eigenvalues,
            explained_inertia=explained,
        )

        return self.result


class UMAPTransformer:
    """Constructs a low-dimensional embedding from precomputed distances."""

    def __init__(
        self,
        n_components: int = config.UMAP_COMPONENTS,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        random_state: int = config.UMAP_RANDOM_STATE,
    ) -> None:
        self.parameters = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": "precomputed",
            "random_state": random_state,
        }
        self.model: umap.UMAP | None = None

    def fit_transform(self, distances: np.ndarray) -> UMAPResult:
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError("UMAP requires a square precomputed matrix.")

        if self.parameters["n_neighbors"] >= len(distances):
            raise ValueError("UMAP n_neighbors must be below sample count.")

        self.model = umap.UMAP(
            n_components=self.parameters["n_components"],
            n_neighbors=self.parameters["n_neighbors"],
            min_dist=self.parameters["min_dist"],
            metric="precomputed",
            random_state=self.parameters["random_state"],
            transform_seed=self.parameters["random_state"],
            low_memory=True,
            n_jobs=1,
        )

        coordinates = self.model.fit_transform(distances)

        if not np.isfinite(coordinates).all():
            raise ValueError("UMAP generated non-finite coordinates.")

        return UMAPResult(
            estimator=self.model,
            coordinates=coordinates,
            parameters=self.parameters.copy(),
        )
