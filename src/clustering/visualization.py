"""Visual diagnostics for the four clustering approaches."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram

from sklearn.metrics import silhouette_samples

from . import config
from .estimators import ClusteringResult


class ClusteringVisualizer:
    """Creates and persists reproducible clustering diagnostic figures."""

    def __init__(
        self,
        output_dir: Path = config.FIGURES_DIR,
        dpi: int = 300,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        sns.set_theme(
            style="whitegrid",
            context="notebook",
        )

    def _save(self, figure, filename: str) -> Path:
        path = self.output_dir / filename
        figure.savefig(
            path,
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(figure)
        return path

    def plot_kmodes_cost(
        self,
        results: list[ClusteringResult],
    ) -> Path:
        frame = pd.DataFrame([
            {
                "k": result.parameters["n_clusters"],
                "cost": result.objective,
            }
            for result in results
        ])

        summary = (
            frame.groupby("k", as_index=False)
            .agg(
                mean_cost=("cost", "mean"),
                std_cost=("cost", "std"),
                minimum_cost=("cost", "min"),
            )
        )

        figure, ax = plt.subplots(figsize=(9, 5))

        ax.errorbar(
            summary["k"],
            summary["mean_cost"],
            yerr=summary["std_cost"],
            marker="o",
            capsize=4,
            label="Mean cost ± standard deviation",
        )

        ax.plot(
            summary["k"],
            summary["minimum_cost"],
            marker="s",
            linestyle="--",
            label="Minimum cost",
        )

        ax.set(
            title="K-Modes cost across candidate cluster counts",
            xlabel="Number of clusters (k)",
            ylabel="Categorical mismatch cost",
        )
        ax.legend()

        return self._save(figure, "kmodes_cost_elbow.png")

    def plot_mca_eigenvalues(
        self,
        eigenvalues: np.ndarray,
        explained_inertia: np.ndarray,
    ) -> Path:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(13, 5),
        )

        components = np.arange(1, len(eigenvalues) + 1)

        axes[0].bar(components, eigenvalues)
        axes[0].set(
            title="MCA eigenvalues",
            xlabel="Component",
            ylabel="Eigenvalue",
        )

        axes[1].plot(
            components,
            np.cumsum(explained_inertia) * 100,
            marker="o",
        )
        axes[1].set(
            title="Cumulative explained inertia",
            xlabel="Number of components",
            ylabel="Cumulative inertia (%)",
        )

        return self._save(figure, "mca_inertia.png")

    def plot_silhouette(
        self,
        coordinates: np.ndarray,
        labels: np.ndarray,
        filename: str,
        title: str,
    ) -> Path:
        scores = silhouette_samples(
            coordinates,
            labels,
        )

        figure, ax = plt.subplots(figsize=(10, 7))
        y_lower = 10

        for cluster in np.unique(labels):
            values = np.sort(scores[labels == cluster])
            y_upper = y_lower + len(values)

            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                values,
                alpha=0.7,
            )
            ax.text(-0.05, y_lower + len(values) / 2, str(cluster))
            y_lower = y_upper + 10

        ax.axvline(
            np.mean(scores),
            linestyle="--",
            label=f"Mean silhouette = {np.mean(scores):.3f}",
        )

        ax.set(
            title=title,
            xlabel="Silhouette coefficient",
            ylabel="Observations grouped by cluster",
        )
        ax.legend()

        return self._save(figure, filename)

    def plot_dendrogram(
        self,
        tree: np.ndarray,
        linkage_method: str,
    ) -> Path:
        figure, ax = plt.subplots(figsize=(15, 7))

        dendrogram(
            tree,
            truncate_mode="lastp",
            p=config.DENDROGRAM_TRUNCATE_COUNT,
            show_leaf_counts=True,
            leaf_rotation=90,
            ax=ax,
        )

        ax.set(
            title=f"Agglomerative dendrogram — {linkage_method}",
            xlabel="Merged respondent groups",
            ylabel="Dissimilarity",
        )

        return self._save(
            figure,
            f"agglomerative_dendrogram_{linkage_method}.png",
        )

    def plot_umap_clusters(
        self,
        coordinates: np.ndarray,
        labels: np.ndarray,
        filename: str,
        title: str,
    ) -> Path:
        if coordinates.shape[1] < 2:
            raise ValueError("At least two UMAP dimensions are required.")

        frame = pd.DataFrame({
            "UMAP 1": coordinates[:, 0],
            "UMAP 2": coordinates[:, 1],
            "Cluster": labels.astype(str),
        })

        figure, ax = plt.subplots(figsize=(10, 8))

        sns.scatterplot(
            data=frame,
            x="UMAP 1",
            y="UMAP 2",
            hue="Cluster",
            palette="tab10",
            alpha=0.75,
            s=35,
            linewidth=0,
            ax=ax,
        )

        ax.set_title(title)
        ax.legend(
            title="Cluster",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        return self._save(figure, filename)

    def plot_hdbscan_noise(
        self,
        results: list[ClusteringResult],
    ) -> Path:
        frame = pd.DataFrame([
            {
                "min_cluster_size": result.parameters["min_cluster_size"],
                "min_samples": result.parameters["min_samples"],
                "selection_method": result.parameters[
                    "cluster_selection_method"
                ],
                "noise_percentage": result.noise_fraction * 100,
                "n_clusters": result.n_clusters,
            }
            for result in results
        ])

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5),
        )

        sns.barplot(
            data=frame,
            x="min_cluster_size",
            y="noise_percentage",
            hue="selection_method",
            ax=axes[0],
        )
        axes[0].set_title("HDBSCAN noise percentage")

        sns.barplot(
            data=frame,
            x="min_cluster_size",
            y="n_clusters",
            hue="selection_method",
            ax=axes[1],
        )
        axes[1].set_title("HDBSCAN discovered cluster count")

        figure.tight_layout()

        return self._save(
            figure,
            "hdbscan_parameter_diagnostics.png",
        )

    def plot_metric_comparison(
        self,
        metrics: pd.DataFrame,
        metric: str,
        title: str,
        filename: str,
    ) -> Path:
        if metric not in metrics:
            raise KeyError(f"Metric not found: {metric}")

        frame = metrics.dropna(subset=[metric]).copy()

        figure, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=frame,
            x="candidate",
            y=metric,
            hue="algorithm",
            ax=ax,
        )

        ax.set(
            title=title,
            xlabel="Candidate model",
            ylabel=metric,
        )
        ax.tick_params(axis="x", rotation=75)

        return self._save(figure, filename)
