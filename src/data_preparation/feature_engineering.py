"""Feature engineering for survey multi-select and ordinal variables."""

from __future__ import annotations

import re
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer


MULTISELECT_COLUMNS = {
    "work_position",
    "diagnosed_conditions_current",
    "believed_conditions_if_maybe",
    "diagnosed_conditions_professional",
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


def split_multiselect_value(value) -> tuple[str, ...]:
    """Split pipe delimited selections, trim labels, and remove duplicates."""
    if pd.isna(value):
        return value

    parts = [
        re.sub(r"\s+", " ", part).strip()
        for part in str(value).split("|")
    ]
    parts = [part for part in parts if part]

    # Preserve original order while removing duplicate selections.
    return tuple(dict.fromkeys(parts))


def parse_multiselect_columns(
        df: pd.DataFrame,
        columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create tuple-valued parsed columns, retaining the original response column.

    The original string response is retained for traceability. The parsed
    representation is stored in a companion column ending in '_items'.
    """
    result = df.copy()
    audit = []

    for column in sorted(
        (columns or MULTISELECT_COLUMNS).intersection(result.columns)
    ):
        parsed_column = f"{column}_items"
        result[parsed_column] = result[column].map(split_multiselect_value)

        all_items = [
            item
            for selections in result[parsed_column]
            for item in selections
        ]

        audit.append(
            {
                "column": column,
                "parsed_column": parsed_column,
                "rows_with_selections": int(
                    result[parsed_column].map(bool).sum()
                ),
                "unique_selection_labels": len(set(all_items)),
                "total_individual_selections": len(all_items),
            }
        )

    return result, pd.DataFrame(audit)


def encode_work_position(
    df: pd.DataFrame,
    column: str = "work_position",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-hot encode work positions.

    Each position becomes its own 0/1 feature. This is intended for descriptive
    analysis and is not automatically added to the main clustering feature set.
    """
    if column not in df.columns:
        return pd.DataFrame(index=df.index), pd.DataFrame()

    parsed = df[column].map(split_multiselect_value)
    encoder = MultiLabelBinarizer()
    values = encoder.fit_transform(parsed)

    encoded = pd.DataFrame(
        values,
        index=df.index,
        columns=[f"position__{label}" for label in encoder.classes_],
        dtype="int8",
    )

    audit = pd.DataFrame(
        {
            "encoded_column": encoded.columns,
            "source_column": column,
            "encoding": "multi-hot",
        }
    )

    return encoded, audit


def add_ordinal_company_size(
    df: pd.DataFrame,
    column: str = "company_size",
) -> pd.DataFrame:
    """Add an ordered company-size representation; keep N/A as missing."""
    result = df.copy()

    if column not in result.columns:
        return result

    mapping = {
        label: rank for rank, label in enumerate(
            COMPANY_SIZE_ORDER, start=1
        )
    }

    result["company_size_ordinal"] = (
        result[column]
        .map(mapping)
        .astype("Int8")
    )

    # 'Not applicable' and unknown values remain missing, never become zero.
    return result


def add_remote_work_ordinal(
    df: pd.DataFrame,
    column: str = "remote_work",
):
    """Add a 1-3 ordinal representation for Never, Sometimes, and Always."""
    result = df.copy()

    if column not in result.columns:
        return result

    mapping = {
        label: rank for rank, label in enumerate(
            REMOTE_WORK_ORDER, start=1
        )
    }
    result["remote_work_ordinal"] = result[column].map(mapping).astype("Int8")

    return result
