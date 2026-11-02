"""Estimator classes for the four predeclared clustering algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import hdbscan
import numpy as np
import pandas as pd

from kmodes.kmodes import KModes
from sklearn.cluster import AgglomerativeClustering, KMeans

from . import config
from .transformations import MCAResult, UMAPResult


@dataclass
class ClusteringResult:
    """Standardised result container for every candidate model."""

    algorithm: str
    parameters: dict[str, Any]
    labels: np.ndarray
    estimator: Any
    runtime_seconds: float
    objective: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels.tolist()) - {-1})

    @property
    def noise_count(self) -> int:
        return int(np.count_nonzero(self.labels == -1))

    @property
    def noise_fraction(self) -> float:
        return self.noise_count / len(self.labels)


class BaseClusteringEstimator:
    """Shared validation and result-construction behaviour."""

    algorithm: str

    @staticmethod
    def validate_cluster_count(k: int, n_samples: int) -> None:
        if not 2 <= k < n_samples:
            raise ValueError(
                f"Cluster count must be between 2 and {n_samples - 1}."
            )

    @staticmethod
    def validate_labels(
        labels: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int64)

        if labels.ndim != 1 or len(labels) != n_samples:
            raise ValueError("Estimator returned invalid labels.")

        return labels


class KModesEstimator(BaseClusteringEstimator):
    """K-Modes with repeated independent random initialisations."""

    algorithm = "k_modes"

    def __init__(
        self,
        n_init: int = config.KMODES_N_INIT,
        max_iter: int = config.KMODES_MAX_ITER,
        seeds: tuple[int, ...] = config.KMODES_SEEDS,
    ) -> None:
        self.n_init = n_init
        self.max_iter = max_iter
        self.seeds = seeds
        self.results: list[ClusteringResult] = []

    def fit(self, features: pd.DataFrame, k: int) -> ClusteringResult:
        self.validate_cluster_count(k, len(features))

        categorical = features.astype(str).to_numpy()
        runs: list[ClusteringResult] = []

        for seed in self.seeds:
            start = perf_counter()

            estimator = KModes(
                n_clusters=k,
                init="Huang",
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=seed,
                n_jobs=-1,
                verbose=0,
            )

            labels = estimator.fit_predict(categorical)
            runtime = perf_counter() - start

            runs.append(
                ClusteringResult(
                    algorithm=self.algorithm,
                    parameters={
                        "n_clusters": k,
                        "seed": seed,
                        "n_init": self.n_init,
                        "max_iter": self.max_iter,
                    },
                    labels=self.validate_labels(labels, len(features)),
                    estimator=estimator,
                    runtime_seconds=runtime,
                    objective=float(estimator.cost_),
                )
            )

        # Select the minimum-cost run as the representative candidate.
        # Retain all runs for NMI stability evaluation.
        self.results = runs
        best = min(runs, key=lambda result: result.objective)

        best.diagnostics["repeated_runs"] = runs
        best.diagnostics["cost_mean"] = float(
            np.mean([result.objective for result in runs])
        )
        best.diagnostics["cost_std"] = float(
            np.std([result.objective for result in runs], ddof=1)
        )

        return best


class KMeansEstimator(BaseClusteringEstimator):
    """K-Means fitted to MCA coordinates."""

    algorithm = "k_means"

    def __init__(
        self,
        n_init: int = config.KMEANS_N_INIT,
        max_iter: int = config.KMEANS_MAX_ITER,
        tolerance: float = config.KMEANS_TOL,
        random_state: int = config.SEED,
    ) -> None:
        self.n_init = n_init
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

    def fit(self, mca: MCAResult, k: int) -> ClusteringResult:
        coordinates = mca.coordinates.to_numpy(dtype=np.float64)
        self.validate_cluster_count(k, len(coordinates))

        start = perf_counter()

        estimator = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tolerance,
            random_state=self.random_state,
            algorithm="lloyd",
        )

        labels = estimator.fit_predict(coordinates)
        runtime = perf_counter() - start

        return ClusteringResult(
            algorithm=self.algorithm,
            parameters={
                "n_clusters": k,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "tol": self.tolerance,
                "random_state": self.random_state,
                "mca_components": coordinates.shape[1],
            },
            labels=self.validate_labels(labels, len(coordinates)),
            estimator=estimator,
            runtime_seconds=runtime,
            objective=float(estimator.inertia_),
        )


class AgglomerativeEstimator(BaseClusteringEstimator):
    """Agglomerative clustering over the precomputed Gower-style matrix."""

    algorithm = "agglomerative"

    def fit(
        self,
        distances: np.ndarray,
        k: int,
        linkage: str = "average",
    ) -> ClusteringResult:
        self.validate_cluster_count(k, len(distances))

        if linkage not in config.AGGLOMERATIVE_LINKAGES:
            raise ValueError(f"Unsupported linkage: {linkage}")

        start = perf_counter()

        estimator = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage=linkage,
            compute_full_tree=True,
            compute_distances=True,
        )

        labels = estimator.fit_predict(distances)
        runtime = perf_counter() - start

        return ClusteringResult(
            algorithm=self.algorithm,
            parameters={
                "n_clusters": k,
                "linkage": linkage,
                "metric": "precomputed",
            },
            labels=self.validate_labels(labels, len(distances)),
            estimator=estimator,
            runtime_seconds=runtime,
        )


class HDBSCANEstimator(BaseClusteringEstimator):
    """HDBSCAN fitted to UMAP coordinates."""

    algorithm = "hdbscan"

    def fit(
        self,
        embedding: UMAPResult,
        min_cluster_size: int,
        min_samples: int | None,
        selection_method: str = "eom",
    ) -> ClusteringResult:
        coordinates = embedding.coordinates

        if min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2.")

        if selection_method not in {"eom", "leaf"}:
            raise ValueError("Invalid cluster selection method.")

        start = perf_counter()

        estimator = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=selection_method,
            prediction_data=True,
            gen_min_span_tree=True,
            core_dist_n_jobs=-1,
        )

        labels = estimator.fit_predict(coordinates)
        runtime = perf_counter() - start

        persistence = np.asarray(
            estimator.cluster_persistence_,
            dtype=np.float64,
        )

        diagnostics = {
            "cluster_persistence": persistence.tolist(),
            "mean_cluster_persistence": (
                float(np.mean(persistence))
                if len(persistence)
                else np.nan
            ),
            "membership_probabilities": (
                estimator.probabilities_.astype(float).tolist()
            ),
            "outlier_scores": (
                estimator.outlier_scores_.astype(float).tolist()
            ),
            "umap_parameters": embedding.parameters,
        }

        return ClusteringResult(
            algorithm=self.algorithm,
            parameters={
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": selection_method,
                "metric": "euclidean",
                "umap_parameters": embedding.parameters,
            },
            labels=self.validate_labels(labels, len(coordinates)),
            estimator=estimator,
            runtime_seconds=runtime,
            diagnostics=diagnostics,
        )
