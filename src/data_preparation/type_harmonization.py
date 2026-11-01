"""Response harmonisation without loss of survey meaning."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


def canonicalise_response(value):
    """
    Normalise formatting while preserving the semantic response.

    This function deliberately does not collapse Maybe, I don't know,
    Not applicable, or other distinct survey answers.
    """
    if pd.isna(value):
        return pd.NA

    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else pd.NA


def canonicalise_yes_no(value):
    """
    Standardise binary response spelling.

    Non-binary responses are retained unchanged.
    """
    value = canonicalise_response(value)

    if pd.isna(value):
        return pd.NA

    key = str(value).casefold()

    mapping = {
        "yes": "Yes",
        "y": "Yes",
        "no": "No",
        "n": "No",
    }

    return mapping.get(key, value)


def harmonise_categorical_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    binary_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Harmonise selected response columns."""
    result = df.copy()
    binary_columns = set(binary_columns or [])

    missing = sorted(set(columns) - set(result.columns))

    if missing:
        raise KeyError(f"Columns not found: {missing}")

    for column in columns:
        if column in binary_columns:
            result[column] = (
                result[column]
                .map(canonicalise_yes_no)
                .astype("string")
            )
        else:
            result[column] = (
                result[column]
                .map(canonicalise_response)
                .astype("string")
            )

    return result


def build_type_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Describe the stored dtype and observed values of each column."""
    records = []

    for column in df.columns:
        series = df[column]

        records.append({
            "column": column,
            "dtype": str(series.dtype),
            "non_missing": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "distinct_non_missing": int(series.nunique(dropna=True)),
        })

    return pd.DataFrame(records)
