"""Canonicalise survey response types without discarding response meaning."""

from __future__ import annotations

import re
import pandas as pd


YES_VALUES = {
    "yes", "1", "1.0", "true", "Yes", "YES", "True"
}

NO_VALUES = {
    "no", "0", "0.0", "false", "True", "No", "NO"
}

# Only columns explicitly known to be binary are harmonised to nullable Int8.
BINARY_COLUMN_NAMES = {
    "self_employed",
    "tech_employer",
    "tech_role",
    "self_employed_medical_coverage",
    "previous_employers",
    "medical_professional_diagnosis",
    "sought_treatment",
}


def canonicalise_response(value):
    """Normalise whitespace and common text artefacts, preserving semantics."""
    if pd.isna(value):
        return pd.NA

    text = str(value).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return pd.NA

    return text


def canonicalise_yes_no(value):
    """Convert an unambiguous yes/no value to 1/0; preserve all other states."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().casefold()

    if text in YES_VALUES:
        return 1
    if text in NO_VALUES:
        return 0

    return canonicalise_response(value)


def harmonise_binary_columns(
    df: pd.DataFrame,
    columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert known binary fields to nullable Int8.

    Unexpected non-binary labels are not silently coerced. They are reported
    and left as strings so that a schema/data issue cannot become false data.
    """
    result = df.copy()
    columns = columns or BINARY_COLUMN_NAMES
    audit = []

    for column in sorted(columns.intersection(result.columns)):
        original = result[column]
        converted = original.map(canonicalise_yes_no)

        unexpected = sorted(
            {
                str(value)
                for value in converted.dropna().unique()
                if value not in (0, 1)
            }
        )

        if unexpected:
            audit.append(
                {
                    "column": column,
                    "encoding": "not_converted",
                    "unexpected_values": " | ".join(unexpected),
                    "reason": "Unexpected non-binary response(s) found.",
                }
            )
            result[column] = original.map(
                canonicalise_response).astype("string")
        else:
            result[column] = converted.astype("Int8")
            audit.append(
                {
                    "column": column,
                    "encoding": "nullable_Int8",
                    "unexpected_values": "",
                    "reason": "All observed responses are unambiguous yes/no values.",
                }
            )

    return result, pd.DataFrame(audit)


def harmonise_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise object/string columns while leaving numeric columns numeric."""
    result = df.copy()

    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = (
            result[column]
            .map(canonicalise_response)
            .astype("string")
        )

    return result


def build_type_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Describe each column's stored type and observed categorical cardinality."""
    records = []

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            inferred_type = "numeric"
        else:
            inferred_type = "categorical"

        records.append(
            {
                "column": column,
                "pandas_dtype": str(series.dtype),
                "inferred_type": inferred_type,
                "distinct_non_missing": int(series.nunique(dropna=True)),
                "contains_missing": bool(series.isna().any()),
            }
        )

    return pd.DataFrame(records)
