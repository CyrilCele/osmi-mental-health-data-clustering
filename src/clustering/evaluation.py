"""Algorithm-specific and comparative clustering evaluation."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import hdbscan
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from . import config
from .estimators import ClusteringResult


def cluster_counts(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)

    return {
        int(label): int(count)
        for label, count in zip(unique, counts)
    }


def calculate_nmi_stability(
    repeated_results: Iterable[ClusteringResult],
) -> dict[str, float]:
    """Calculate pairwise NMI across independent K-Modes runs."""

    results = list(repeated_results)

    if len(results) < 2:
        return {
            "nmi_mean": np.nan,
            "nmi_std": np.nan,
            "nmi_min": np.nan,
        }

    scores = [
        normalized_mutual_info_score(
            left.labels,
            right.labels,
            average_method="arithmetic",
        )
        for left, right in combinations(results, 2)
    ]

    return {
        "nmi_mean": float(np.mean(scores)),
        "nmi_std": float(np.std(scores, ddof=1)),
        "nmi_min": float(np.min(scores)),
    }


def calculate_kmeans_metrics(
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Geometric validation in the MCA coordinate space."""

    labels = np.asarray(labels)
    unique = np.unique(labels)

    if len(unique) < 2 or len(unique) >= len(labels):
        return {
            "silhouette_mca": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    return {
        "silhouette_mca": float(
            silhouette_score(coordinates, labels)
        ),
        "davies_bouldin": float(
            davies_bouldin_score(coordinates, labels)
        ),
        "calinski_harabasz": float(
            calinski_harabasz_score(coordinates, labels)
        ),
    }


def calculate_gower_silhouette(
    distances: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Silhouette using the prepared precomputed dissimilarities."""

    unique = np.unique(labels)

    if len(unique) < 2 or len(unique) >= len(labels):
        return np.nan

    return float(
        silhouette_score(
            distances,
            labels,
            metric="precomputed",
        )
    )


def calculate_cophenetic_correlation(
    distances: np.ndarray,
    linkage_method: str,
) -> tuple[float, np.ndarray]:
    """Calculate CPCC and the corresponding hierarchical linkage tree."""

    condensed = squareform(distances, checks=True)

    tree = linkage(
        condensed,
        method=linkage_method,
    )

    coefficient, cophenetic_distances = cophenet(
        tree,
        condensed,
    )

    return float(coefficient), tree


def calculate_hubert_gamma(
    distances: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Hubert's Gamma correlation between original dissimilarities
    and binary same-cluster/different-cluster relationships.

    The sign convention is defined so that larger values indicate
    larger original distances between observations in different clusters.
    """

    condensed = squareform(distances, checks=True)
    upper_i, upper_j = np.triu_indices(len(labels), k=1)

    same_cluster = (
        labels[upper_i] == labels[upper_j]
    ).astype(np.float64)

    different_cluster = 1.0 - same_cluster

    if np.std(condensed) == 0 or np.std(different_cluster) == 0:
        return np.nan

    # Correlate original distance with different-cluster indicator.
    return float(
        pearsonr(condensed, different_cluster).statistic
    )


def calculate_dbcv(
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Density-Based Clustering Validation in the UMAP space."""

    if len(set(labels) - {-1}) < 2:
        return np.nan

    try:
        return float(
            hdbscan.validity.validity_index(
                coordinates,
                labels,
                metric="euclidean",
            )
        )
    except (ValueError, FloatingPointError):
        return np.nan


def evaluate_kmodes(
    result: ClusteringResult,
) -> dict:
    repeated = result.diagnostics["repeated_runs"]
    stability = calculate_nmi_stability(repeated)

    return {
        "algorithm": "k_modes",
        "parameters": result.parameters,
        "cost": result.objective,
        "cost_mean": result.diagnostics["cost_mean"],
        "cost_std": result.diagnostics["cost_std"],
        **stability,
        "n_clusters": result.n_clusters,
        "cluster_sizes": cluster_counts(result.labels),
        "runtime_seconds": result.runtime_seconds,
    }


def evaluate_kmeans(
    result: ClusteringResult,
    coordinates: np.ndarray,
) -> dict:
    metrics = calculate_kmeans_metrics(
        coordinates,
        result.labels,
    )

    return {
        "algorithm": "k_means",
        "parameters": result.parameters,
        "inertia": result.objective,
        **metrics,
        "n_clusters": result.n_clusters,
        "cluster_sizes": cluster_counts(result.labels),
        "runtime_seconds": result.runtime_seconds,
    }


def evaluate_agglomerative(
    result: ClusteringResult,
    distances: np.ndarray,
) -> dict:
    linkage_method = result.parameters["linkage"]

    cpcc, _ = calculate_cophenetic_correlation(
        distances,
        linkage_method,
    )

    return {
        "algorithm": "agglomerative",
        "parameters": result.parameters,
        "cophenetic_correlation": cpcc,
        "silhouette_gower": calculate_gower_silhouette(
            distances,
            result.labels,
        ),
        "hubert_gamma": calculate_hubert_gamma(
            distances,
            result.labels,
        ),
        "n_clusters": result.n_clusters,
        "cluster_sizes": cluster_counts(result.labels),
        "runtime_seconds": result.runtime_seconds,
    }


def evaluate_hdbscan(
    result: ClusteringResult,
    coordinates: np.ndarray,
) -> dict:
    return {
        "algorithm": "hdbscan",
        "parameters": result.parameters,
        "dbcv": calculate_dbcv(coordinates, result.labels),
        "n_clusters": result.n_clusters,
        "noise_count": result.noise_count,
        "noise_percentage": result.noise_fraction * 100,
        "coverage_percentage": (
            1.0 - result.noise_fraction
        ) * 100,
        "cluster_sizes": cluster_counts(result.labels),
        "cluster_persistence": result.diagnostics[
            "cluster_persistence"
        ],
        "mean_cluster_persistence": result.diagnostics[
            "mean_cluster_persistence"
        ],
        "runtime_seconds": result.runtime_seconds,
    }


class ClusteringEvaluator:
    """Dispatches evaluation to the correct algorithm-specific metrics."""

    def evaluate(
        self,
        result: ClusteringResult,
        *,
        distances: np.ndarray,
        mca_coordinates: np.ndarray | None = None,
        umap_coordinates: np.ndarray | None = None,
    ) -> dict:
        if result.algorithm == "k_modes":
            return evaluate_kmodes(result)

        if result.algorithm == "k_means":
            if mca_coordinates is None:
                raise ValueError("MCA coordinates are required.")
            return evaluate_kmeans(result, mca_coordinates)

        if result.algorithm == "agglomerative":
            return evaluate_agglomerative(result, distances)

        if result.algorithm == "hdbscan":
            if umap_coordinates is None:
                raise ValueError("UMAP coordinates are required.")
            return evaluate_hdbscan(result, umap_coordinates)

        raise ValueError(f"Unknown algorithm: {result.algorithm}")
