"""Persist cleaned data, model inputs, audits, and validation results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_output_directory(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataframe(
    df: pd.DataFrame,
    directory: Path,
    filename: str,
    *,
    index: bool = False,
) -> Path:
    ensure_output_directory(directory)
    path = directory / filename
    df.to_csv(path, index=index, encoding="utf-8")
    return path


def save_numpy(
    array: np.ndarray,
    directory: Path,
    filename: str,
) -> Path:
    ensure_output_directory(directory)
    path = directory / filename
    np.save(path, np.asarray(array))
    return path


def save_json(
    payload: dict,
    directory: Path,
    filename: str,
) -> Path:
    ensure_output_directory(directory)
    path = directory / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return path


def save_preparation_outputs(
    directory: Path,
    *,
    cleaned_data: pd.DataFrame,
    kmodes_matrix: pd.DataFrame,
    mca_input: pd.DataFrame,
    mca_coordinates: pd.DataFrame,
    gower_input: pd.DataFrame,
    gower_distances: np.ndarray,
    umap_embedding: pd.DataFrame,
    missingness_audit: pd.DataFrame,
    applicability_register: pd.DataFrame,
    exclusion_register: pd.DataFrame,
    feature_decisions: pd.DataFrame,
    type_audit: pd.DataFrame,
    multiselect_audit: pd.DataFrame,
    validation_report: dict,
    row_mapping: pd.DataFrame,
    feature_metadata: pd.DataFrame,
) -> dict[str, str]:
    ensure_output_directory(directory)

    if len(gower_distances) != len(gower_input):
        raise ValueError(
            "Gower distance matrix row count does not match Gower input."
        )

    if gower_distances.shape != (
        len(gower_input),
        len(gower_input),
    ):
        raise ValueError(
            "Gower distance matrix must have one row and column "
            "per Gower input respondent."
        )

    outputs = {
        "cleaned_data": (
            cleaned_data,
            "cleaned_survey.csv",
        ),
        "kmodes_matrix": (
            kmodes_matrix,
            "kmodes_matrix.csv",
        ),
        "mca_input": (
            mca_input,
            "mca_input.csv",
        ),
        "mca_coordinates": (
            mca_coordinates,
            "kmeans_mca_coordinates.csv",
        ),
        "gower_input": (
            gower_input,
            "gower_input.csv",
        ),
        "missingness_audit": (
            missingness_audit,
            "missingness_audit.csv",
        ),
        "applicability_register": (
            applicability_register,
            "applicability_register.csv",
        ),
        "exclusion_register": (
            exclusion_register,
            "exclusion_register.csv",
        ),
        "feature_decisions": (
            feature_decisions,
            "feature_decision_log.csv",
        ),
        "type_audit": (
            type_audit,
            "type_harmonisation_audit.csv",
        ),
        "multiselect_audit": (
            multiselect_audit,
            "multiselect_audit.csv",
        ),
        "row_mapping": (
            row_mapping,
            "respondent_row_mapping.csv",
        ),
        "feature_metadata": (
            feature_metadata,
            "feature_metadata.csv",
        ),
    }

    saved = {}

    for key, (dataframe, filename) in outputs.items():
        saved[key] = str(
            save_dataframe(dataframe, directory, filename)
        )

    saved["gower_distances"] = str(
        save_numpy(
            gower_distances,
            directory,
            "gower_distance_matrix.npy",
        )
    )

    saved["umap_embedding"] = str(
        save_dataframe(
            umap_embedding,
            directory,
            "umap_embedding.csv",
        )
    )

    saved["validation_report"] = str(
        save_json(
            validation_report,
            directory,
            "preparation_validation.json",
        )
    )

    return saved


def build_respondent_row_mapping(
    source_row_ids: pd.Series,
    index: pd.Index,
) -> pd.DataFrame:
    if len(source_row_ids) != len(index):
        raise ValueError(
            "Source row IDs and prepared index have different lengths."
        )

    if not index.is_unique:
        raise ValueError(
            "Prepared respondent index must be unique."
        )

    return pd.DataFrame(
        {
            "prepared_row": range(len(index)),
            "source_row_id": source_row_ids.to_numpy(),
        }
    )


def build_feature_metadata(
    feature_decisions: pd.DataFrame,
    cleaned_data: pd.DataFrame,
) -> pd.DataFrame:
    metadata = feature_decisions.copy()

    unknown_columns = (
        set(metadata["column"]) - set(cleaned_data.columns)
    )

    if unknown_columns:
        raise ValueError(
            "Feature decision register references columns that are "
            f"not in cleaned_data: {sorted(unknown_columns)}"
        )

    metadata["pandas_dtype"] = metadata["column"].map(
        lambda column: str(cleaned_data[column].dtype)
    )

    metadata["missing_count"] = metadata["column"].map(
        lambda column: int(cleaned_data[column].isna().sum())
    )

    metadata["distinct_non_missing"] = metadata["column"].map(
        lambda column: int(
            cleaned_data[column].nunique(dropna=True)
        )
    )

    return metadata
