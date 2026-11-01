"""Cohort-aware missingness and applicability audits."""

from __future__ import annotations

import pandas as pd

from .config import MODEL_FEATURES, SUPPLEMENTARY_FEATURES


def build_missingness_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate column-level missingness."""
    count = len(df)

    records = []

    for column in df.columns:
        missing = int(df[column].isna().sum())

        records.append({
            "feature": column,
            "n_rows": count,
            "n_missing": missing,
            "n_observed": count - missing,
            "missing_percent": (
                100.0 * missing / count if count else 0.0
            ),
            "distinct_observed": int(df[column].nunique(dropna=True)),
        })

    return (
        pd.DataFrame(records)
        .sort_values(
            ["missing_percent", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_applicability_register(
    df: pd.DataFrame,
    *,
    cohort_column: str | None = None,
) -> pd.DataFrame:
    """
    Record applicability and completeness for the primary features.

    Primary model features are expected to be observed in the eligible
    cohort. Missing values are not silently imputed.
    """
    records = []

    for feature in MODEL_FEATURES:
        if feature not in df:
            records.append({
                "feature": feature,
                "applicability": "missing_column",
                "observed": 0,
                "missing": len(df),
                "complete": False,
            })
            continue

        missing = int(df[feature].isna().sum())

        records.append({
            "feature": feature,
            "applicability": "primary_cohort_feature",
            "observed": int(df[feature].notna().sum()),
            "missing": missing,
            "complete": missing == 0,
        })

    for feature in SUPPLEMENTARY_FEATURES:
        if feature not in df:
            continue

        records.append({
            "feature": feature,
            "applicability": "supplementary_or_conditional",
            "observed": int(df[feature].notna().sum()),
            "missing": int(df[feature].isna().sum()),
            "complete": bool(df[feature].notna().all()),
        })

    return pd.DataFrame(records)


def build_exclusion_register(
    df: pd.DataFrame,
    *,
    exclusion_reason: str = "Outside approved model feature specification",
) -> pd.DataFrame:
    """Register columns not selected as primary model features."""
    records = []

    for column in df.columns:
        if column in MODEL_FEATURES:
            continue

        records.append({
            "column": column,
            "decision": "excluded_from_primary_distance",
            "reason": exclusion_reason,
        })

    return pd.DataFrame(records)


def assert_primary_features_complete(df: pd.DataFrame) -> None:
    """Fail explicitly if any primary feature is incomplete."""
    missing_columns = sorted(set(MODEL_FEATURES) - set(df.columns))

    if missing_columns:
        raise KeyError(
            f"Primary feature columns are absent: {missing_columns}"
        )

    missing_counts = df[MODEL_FEATURES].isna().sum()
    incomplete = missing_counts[missing_counts.gt(0)]

    if not incomplete.empty:
        raise ValueError(
            "Primary model features contain missing responses. "
            "The approved cohort is expected to have complete primary "
            "features. No automatic imputation is permitted.\n"
            f"{incomplete.to_string()}"
        )
