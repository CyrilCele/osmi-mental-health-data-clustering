"""Persistence of modelling results and reproducibility metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .estimators import ClusteringResult
from .transformations import ModelingDataset


def json_safe(value: Any) -> Any:
    """Convert common scientific Python objects into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
            if key != "repeated_runs"
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not np.isfinite(value):
        return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


class ClusteringResultsStore:
    """Writes candidate labels, metrics and experiment metadata."""

    def __init__(
        self,
        output_dir: Path = config.RESULTS_DIR,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_candidate_labels(
        self,
        dataset: ModelingDataset,
        results: list[ClusteringResult],
    ) -> Path:
        labels = dataset.row_mapping.copy()

        for index, result in enumerate(results):
            labels[f"candidate_{index}_{result.algorithm}"] = (
                result.labels
            )

        path = self.output_dir / "candidate_cluster_labels.csv"
        labels.to_csv(path, index=False, encoding="utf-8")

        return path

    def save_metrics(
        self,
        metrics: pd.DataFrame,
    ) -> Path:
        path = self.output_dir / "candidate_model_metrics.csv"
        metrics.to_csv(path, index=False, encoding="utf-8")
        return path

    def save_metadata(
        self,
        dataset: ModelingDataset,
        results: list[ClusteringResult],
    ) -> Path:
        metadata = {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "random_state": config.SEED,
            "n_respondents": dataset.n_samples,
            "n_features": dataset.n_features,
            "features": list(dataset.features.columns),
            "algorithms": sorted({
                result.algorithm for result in results
            }),
            "candidate_models": [
                {
                    "algorithm": result.algorithm,
                    "parameters": result.parameters,
                    "runtime_seconds": result.runtime_seconds,
                    "objective": result.objective,
                    "n_clusters": result.n_clusters,
                    "noise_count": result.noise_count,
                }
                for result in results
            ],
        }

        path = self.output_dir / "modeling_metadata.json"

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                json_safe(metadata),
                file,
                indent=4,
                ensure_ascii=False,
            )

        return path

    def save_mca_coordinates(
        self,
        coordinates: pd.DataFrame,
    ) -> Path:
        path = self.output_dir / "mca_coordinates.csv"
        coordinates.to_csv(path, index=False)
        return path

    def save_umap_coordinates(
        self,
        coordinates: np.ndarray,
        parameters: dict,
        filename: str,
    ) -> Path:
        frame = pd.DataFrame(
            coordinates,
            columns=[
                f"umap_{index + 1}"
                for index in range(coordinates.shape[1])
            ],
        )

        path = self.output_dir / filename
        frame.to_csv(path, index=False)

        metadata_path = path.with_suffix(".json")
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                json_safe(parameters),
                file,
                indent=4,
            )

        return path
