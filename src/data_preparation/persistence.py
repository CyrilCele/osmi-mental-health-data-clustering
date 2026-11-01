"""Persistence of reproducible preparation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_output_directory(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_dataframe(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
    *,
    index: bool = False,
) -> Path:
    output_path = ensure_output_directory(output_dir) / filename

    if output_path.suffix.lower() != ".csv":
        raise ValueError("DataFrame output filename must end in .csv.")

    df.to_csv(
        output_path,
        index=index,
        encoding="utf-8",
    )

    return output_path


def save_numpy(
    array: np.ndarray,
    output_dir: str | Path,
    filename: str,
) -> Path:
    output_path = ensure_output_directory(output_dir) / filename

    if output_path.suffix.lower() != ".npy":
        raise ValueError("NumPy output filename must end in .npy.")

    np.save(output_path, array)

    return output_path


def save_json(
    payload: dict,
    output_dir: str | Path,
    filename: str,
) -> Path:
    output_path = ensure_output_directory(output_dir) / filename

    if output_path.suffix.lower() != ".json":
        raise ValueError("JSON output filename must end in .json.")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return output_path


def build_respondent_row_mapping(
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Map distance-matrix row positions to source respondents."""
    if "source_row_id" not in cohort:
        raise KeyError("source_row_id is required.")

    return pd.DataFrame({
        "distance_matrix_row": np.arange(
            len(cohort),
            dtype=np.int64,
        ),
        "source_row_id": cohort["source_row_id"].to_numpy(
            dtype=np.int64
        ),
    })


def build_feature_metadata(
    domain_features: dict[str, list[str]],
) -> pd.DataFrame:
    """Build feature/domain and weighting metadata."""
    records = []
    number_of_domains = len(domain_features)

    for domain, features in domain_features.items():
        for feature in features:
            records.append({
                "feature": feature,
                "domain": domain,
                "domain_feature_count": len(features),
                "domain_weight": 1.0 / number_of_domains,
                "within_domain_weight": 1.0 / len(features),
                "overall_weight": (
                    1.0 / number_of_domains / len(features)
                ),
            })

    return pd.DataFrame(records)


def save_preparation_outputs(
    *,
    output_dir: str | Path,
    cohort: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    profile_descriptors: pd.DataFrame,
    missingness_audit: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    row_mapping: pd.DataFrame,
    distance_matrix: np.ndarray,
    validation_report: dict,
    source_metadata: dict,
) -> dict[str, str]:
    """Persist all primary preparation artefacts."""
    output_dir = ensure_output_directory(output_dir)

    outputs = {
        "eligible_cohort": save_dataframe(
            cohort,
            output_dir,
            "eligible_cohort.csv",
        ),
        "model_features": save_dataframe(
            feature_matrix,
            output_dir,
            "model_feature_matrix.csv",
        ),
        "profile_descriptors": save_dataframe(
            profile_descriptors,
            output_dir,
            "profile_descriptors.csv",
        ),
        "missingness_audit": save_dataframe(
            missingness_audit,
            output_dir,
            "missingness_audit.csv",
        ),
        "feature_metadata": save_dataframe(
            feature_metadata,
            output_dir,
            "feature_metadata.csv",
        ),
        "respondent_row_mapping": save_dataframe(
            row_mapping,
            output_dir,
            "respondent_row_mapping.csv",
        ),
        "distance_matrix": save_numpy(
            distance_matrix,
            output_dir,
            "domain_balanced_dissimilarity.npy",
        ),
        "validation_report": save_json(
            validation_report,
            output_dir,
            "validation_report.json",
        ),
        "source_metadata": save_json(
            source_metadata,
            output_dir,
            "source_metadata.json",
        ),
    }

    return {
        name: str(path.resolve())
        for name, path in outputs.items()
    }
