from __future__ import annotations

import pandas as pd


MULTISELECT_COLUMNS = {
    "current_diagnosed_conditions",
    "suspected_conditions",
    "professionally_diagnosed_conditions",
    "work_position",
}

COMPANY_SIZE_ORDER = [
    "1-5",
    "6-25",
    "26-100",
    "100-500",
    "500-1000",
    "More than 1000",
]

REMOTE_WORK_ORDER = [
    "Never",
    "Sometimes",
    "Always",
]


def split_multiselect_value(value) -> tuple[str, ...] | None:
    """Split a pipe-separated response; preserve missing responses as None."""
    if pd.isna(value):
        return None

    selections = [
        part.strip()
        for part in str(value).split("|")
        if part.strip()
    ]

    return tuple(dict.fromkeys(selections))


def parse_multiselect_columns(
    df: pd.DataFrame,
    columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    audit = []

    selected_columns = (
        MULTISELECT_COLUMNS if columns is None else columns
    )

    for column in sorted(selected_columns.intersection(result.columns)):
        parsed_column = f"{column}_items"

        result[parsed_column] = result[column].map(
            split_multiselect_value
        )

        observed = result[parsed_column].dropna()
        all_items = [
            item
            for selections in observed
            for item in selections
        ]

        audit.append(
            {
                "column": column,
                "parsed_column": parsed_column,
                "rows_with_selections": int(observed.map(bool).sum()),
                "rows_missing": int(result[parsed_column].isna().sum()),
                "unique_selection_labels": len(set(all_items)),
                "total_individual_selections": len(all_items),
            }
        )

    return result, pd.DataFrame(audit)


def encode_multiselect(
    series: pd.Series,
    prefix: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create multi-hot columns and return a missingness mask.

    Missing source responses remain missing in the indicators. They are not
    silently converted into evidence that the respondent selected nothing.
    """
    parsed = series.map(split_multiselect_value)
    observed = parsed.dropna()

    labels = sorted({
        label
        for selections in observed
        for label in selections
    })

    encoded = pd.DataFrame(
        index=series.index,
        columns=[f"{prefix}__{label}" for label in labels],
        dtype="Int8",
    )

    encoded.loc[:, :] = 0

    for index, selections in parsed.items():
        if selections is None:
            encoded.loc[index, :] = pd.NA
        else:
            for label in selections:
                encoded.at[index, f"{prefix}__{label}"] = 1

    return encoded, parsed.isna().rename(f"{prefix}_source_missing")


def encode_work_position(
    df: pd.DataFrame,
    column: str = "work_position",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if column not in df.columns:
        return (
            pd.DataFrame(index=df.index),
            pd.DataFrame(columns=["encoded_column",
                         "source_column", "encoding"]),
        )

    encoded, missing_mask = encode_multiselect(
        df[column],
        prefix="position",
    )

    audit = pd.DataFrame(
        {
            "encoded_column": encoded.columns,
            "source_column": column,
            "encoding": "multi-hot",
        }
    )

    audit["source_missing_count"] = int(missing_mask.sum())

    return encoded, audit


def add_ordinal_company_size(
    df: pd.DataFrame,
    column: str = "company_size",
) -> pd.DataFrame:
    result = df.copy()

    if column not in result.columns:
        return result

    mapping = {
        label: rank
        for rank, label in enumerate(COMPANY_SIZE_ORDER, start=1)
    }

    result["company_size_ordinal"] = (
        result[column].map(mapping).astype("Int8")
    )

    return result


def add_remote_work_ordinal(
    df: pd.DataFrame,
    column: str = "remote_work",
) -> pd.DataFrame:
    result = df.copy()

    if column not in result.columns:
        return result

    mapping = {
        label: rank
        for rank, label in enumerate(REMOTE_WORK_ORDER, start=1)
    }

    result["remote_work_ordinal"] = (
        result[column].map(mapping).astype("Int8")
    )

    return result
