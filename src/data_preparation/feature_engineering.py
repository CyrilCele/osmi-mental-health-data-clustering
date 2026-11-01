"""Project-specific feature engineering and integer encoding."""

from __future__ import annotations

import pandas as pd

from .config import (
    DOMAIN_FEATURES,
    MEDICAL_LEAVE_ORDER,
    MEDICAL_LEAVE_UNKNOWN,
    MODEL_FEATURES,
    ORDINAL_RESPONSE_MAPS,
)


def validate_feature_encoding_specification() -> None:
    """Validate that encoding rules refer only to approved model features."""

    ordinal_features = set(ORDINAL_RESPONSE_MAPS)
    model_features = set(MODEL_FEATURES)

    invalid_mappings = ordinal_features - model_features

    if invalid_mappings:
        raise ValueError(
            "Ordinal mappings reference variables outside the approved "
            f"model feature set: {sorted(invalid_mappings)}"
        )

    if "medical_leave_ease" not in model_features:
        raise ValueError(
            "Medical leave ease must be included in the approved "
            "model feature set."
        )

    if len(MODEL_FEATURES) != 15:
        raise ValueError(
            "The approved model specification must contain exactly "
            "15 features."
        )


def encode_ordinal_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode approved ordinal model features as nullable Int8 columns.

    Nominal model features are deliberately not integer-encoded.
    Missing responses remain missing.
    """
    validate_feature_encoding_specification()

    result = df.copy()
    audit_records = []

    for feature in MODEL_FEATURES:
        if feature not in result.columns:
            raise KeyError(
                f"Approved model feature is missing: {feature}"
            )

        if feature == "medical_leave_ease":
            continue

        if feature not in ORDINAL_RESPONSE_MAPS:
            continue

        mapping = ORDINAL_RESPONSE_MAPS[feature]

        observed = set(result[feature].dropna().unique())
        unexpected = observed - set(mapping)

        if unexpected:
            raise ValueError(
                f"Unmapped categories in {feature}: "
                f"{sorted(unexpected)}"
            )

        encoded_name = f"{feature}_ordinal"

        result[encoded_name] = (
            result[feature]
            .map(mapping)
            .astype("Int8")
        )

        audit_records.append({
            "feature": feature,
            "encoded_feature": encoded_name,
            "feature_role": "ordinal_model_feature",
            "mapping": mapping,
            "dtype": str(result[encoded_name].dtype),
        })

    return result, pd.DataFrame(audit_records)


def encode_medical_leave(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode medical leave ease while preserving 'I don't know' separately.

    The five substantive categories are ordered. 'I don't know' is not
    assigned an ordinal rank and receives its own indicator.
    """
    validate_feature_encoding_specification()

    result = df.copy()
    feature = "medical_leave_ease"

    if feature not in MODEL_FEATURES:
        raise ValueError(
            "Medical leave ease is not an approved model feature."
        )

    if feature not in result.columns:
        raise KeyError(f"Missing feature: {feature}")

    mapping = {
        response: code
        for code, response in enumerate(MEDICAL_LEAVE_ORDER)
    }

    allowed = set(mapping) | {MEDICAL_LEAVE_UNKNOWN}

    observed = set(result[feature].dropna().unique())
    unexpected = observed - allowed

    if unexpected:
        raise ValueError(
            "Unexpected medical-leave responses: "
            f"{sorted(unexpected)}"
        )

    result["medical_leave_ease_ordinal"] = (
        result[feature]
        .map(mapping)
        .astype("Int8")
    )

    result["medical_leave_ease_unknown"] = (
        result[feature]
        .eq(MEDICAL_LEAVE_UNKNOWN)
        .astype("boolean")
    )

    audit = pd.DataFrame({
        "response": list(mapping) + [MEDICAL_LEAVE_UNKNOWN],
        "ordinal_code": list(mapping.values()) + [pd.NA],
        "unknown_indicator": [False] * len(mapping) + [True],
    })

    return result, audit


def build_domain_metadata() -> pd.DataFrame:
    """Return the approved feature-to-domain weighting register."""
    records = []

    for domain, features in DOMAIN_FEATURES.items():
        for feature in features:
            records.append({
                "feature": feature,
                "domain": domain,
                "domain_feature_count": len(features),
                "domain_weight": 1.0 / len(DOMAIN_FEATURES),
                "within_domain_feature_weight": 1.0 / len(features),
                "overall_feature_weight": (
                    1.0 / len(DOMAIN_FEATURES) / len(features)
                ),
            })

    metadata = pd.DataFrame(records)

    if metadata["feature"].nunique() != len(MODEL_FEATURES):
        raise ValueError(
            "Domain metadata does not cover the exact approved "
            "15-feature specification."
        )

    if set(metadata["feature"]) != set(MODEL_FEATURES):
        raise ValueError(
            "Domain metadata and MODEL_FEATURES do not agree."
        )

    return metadata


def validate_encoded_integer_types(df: pd.DataFrame) -> None:
    """Ensure generated ordinal encodings use integer dtypes."""
    encoded_columns = [
        f"{feature}_ordinal"
        for feature in MODEL_FEATURES
        if feature != "medical_leave_ease"
        and feature in ORDINAL_RESPONSE_MAPS
    ]

    encoded_columns.append("medical_leave_ease_ordinal")

    invalid = {
        column: str(df[column].dtype)
        for column in encoded_columns
        if column not in df.columns
        or not pd.api.types.is_integer_dtype(df[column].dtype)
    }

    if invalid:
        raise TypeError(
            "Ordinal encodings must exist and use integer dtypes: "
            f"{invalid}"
        )


def engineer_preparation_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply only the encoding rules defined for approved model features."""
    validate_feature_encoding_specification()

    result, ordinal_audit = encode_ordinal_features(df)
    result, leave_audit = encode_medical_leave(result)

    validate_encoded_integer_types(result)

    leave_record = pd.DataFrame([{
        "feature": "medical_leave_ease",
        "encoded_feature": "medical_leave_ease_ordinal",
        "feature_role": "ordinal_model_feature",
        "mapping": {
            response: code
            for code, response in enumerate(MEDICAL_LEAVE_ORDER)
        },
        "dtype": str(result["medical_leave_ease_ordinal"].dtype),
    }])

    audit = pd.concat(
        [ordinal_audit, leave_record],
        ignore_index=True,
    )

    return result, audit
