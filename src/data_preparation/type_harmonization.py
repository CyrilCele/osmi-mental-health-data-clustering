from __future__ import annotations

import re

import pandas as pd


YES_VALUES = {
    "yes",
    "true",
    "1",
    "1.0",
}

NO_VALUES = {
    "no",
    "false",
    "0",
    "0.0",
}


def canonicalise_response(value):
    """Standardise whitespace while preserving meaningful response text."""
    if pd.isna(value):
        return pd.NA

    text = str(value).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else pd.NA


def canonicalise_yes_no(value):
    """
    Standardise only unambiguous yes/no values.

    Maybe, unsure, unknown, and not-applicable responses remain distinct.
    """
    response = canonicalise_response(value)

    if pd.isna(response):
        return pd.NA

    key = str(response).casefold()

    if key in YES_VALUES:
        return "Yes"

    if key in NO_VALUES:
        return "No"

    return response


def harmonise_categorical_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardise exact yes/no variants in text columns.

    This does not impute missing values or collapse other response categories.
    """
    result = df.copy()

    for column in result.select_dtypes(
        include=["object", "string"]
    ).columns:
        result[column] = (
            result[column]
            .map(canonicalise_yes_no)
            .astype("string")
        )

    return result


def build_type_metadata(df: pd.DataFrame) -> pd.DataFrame:
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
                "distinct_non_missing": int(
                    series.nunique(dropna=True)
                ),
                "contains_missing": bool(series.isna().any()),
            }
        )

    return pd.DataFrame(records)
