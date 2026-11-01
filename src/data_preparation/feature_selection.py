"""Fixed feature selection for the approved OSMI workplace study."""

import pandas as pd

from .config import (
    DOMAIN_FEATURES,
    MODEL_FEATURES,
    PROFILE_DESCRIPTORS,
    SENSITIVE_EXCLUSIONS,
    SUPPLEMENTARY_FEATURES,
)


def build_feature_decision_register(
    df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    model_set = set(MODEL_FEATURES)
    profile_set = set(PROFILE_DESCRIPTORS)
    supplementary_set = set(SUPPLEMENTARY_FEATURES)
    sensitive_set = set(SENSITIVE_EXCLUSIONS)

    for column in df.columns:
        if column in model_set:
            domain = next(
                name
                for name, features in DOMAIN_FEATURES.items()
                if column in features
            )
            decision = "primary_model_feature"
            reason = f"Predeclared feature in domain: {domain}."

        elif column in profile_set:
            domain = None
            decision = "profile_descriptor"
            reason = "Retained for descriptive profiling; not a model input."

        elif column in supplementary_set:
            domain = None
            decision = "supplementary_analysis"
            reason = (
                "Analysed separately using its interpretable response "
                "denominator; excluded from primary distance matrix."
            )

        elif column in sensitive_set:
            domain = None
            decision = "sensitive_exclusion"
            reason = (
                "Personal mental-health history or condition information; "
                "excluded from clustering and target definition."
            )

        elif column in {"source_row_id", "source_index"}:
            domain = None
            decision = "identifier_only"
            reason = "Traceability only; never a feature."

        else:
            domain = None
            decision = "excluded_or_unassigned"
            reason = (
                "Not part of the predeclared model feature set. "
                "Retained in source-derived data for audit where appropriate."
            )

        records.append({
            "column": column,
            "decision": decision,
            "domain": domain,
            "reason": reason,
        })

    return pd.DataFrame(records)


def select_model_features(
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(set(MODEL_FEATURES) - set(cohort.columns))

    if missing:
        raise KeyError(
            f"Required model features are absent: {missing}"
        )

    selected = cohort.loc[:, MODEL_FEATURES].copy()

    if selected.shape[1] != 15:
        raise AssertionError("The model feature matrix must have 15 columns.")

    return selected


def build_profile_descriptors(
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    available = [
        column
        for column in PROFILE_DESCRIPTORS
        if column in cohort.columns
    ]

    descriptors = cohort.loc[:, available].copy()

    if "age_clean" in cohort.columns:
        descriptors["age_clean"] = cohort["age_clean"]

    if "age_implausible_flag" in cohort.columns:
        descriptors["age_implausible_flag"] = (
            cohort["age_implausible_flag"]
        )

    return descriptors


def validate_feature_exclusions(
    model_features: pd.DataFrame,
) -> None:
    forbidden = (
        set(PROFILE_DESCRIPTORS)
        | set(SENSITIVE_EXCLUSIONS)
        | set(SUPPLEMENTARY_FEATURES)
        | {"source_row_id", "source_index", "age_clean"}
    )

    violations = sorted(set(model_features.columns) & forbidden)

    if violations:
        raise ValueError(
            f"Forbidden variables entered model features: {violations}"
        )

    if list(model_features.columns) != MODEL_FEATURES:
        raise ValueError(
            "Model feature columns do not exactly match the approved "
            "15-feature order."
        )
