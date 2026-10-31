"""Explicit feature decisions for the four clustering approaches."""

from __future__ import annotations

import pandas as pd


# These are the shared population-level features. The names correspond to
# the snake_case mapping established in schema.py.
#
# Sensitive demographic/geographic identifiers, free text, and highly
# conditional follow-ups are not included in this feature set.
CORE_CATEGORICAL_FEATURES = [
    "self_employed",
    "company_size",
    "tech_employer",
    "mental_health_benefits",
    "coverage_options_awareness",
    "employer_formal_discussion",
    "employer_mental_health_resources",
    "employer_anonymity_protection",
    "mental_health_leave_ease",
    "employer_discussion_negative_consequences",
    "physical_health_discussion_negative_consequences",
    "coworker_discussion_comfort",
    "supervisor_discussion_comfort",
    "employer_takes_mental_health_seriously",
    "coworker_negative_consequences_observed",
    "previous_employers",
    "previous_employer_benefits",
    "previous_employer_options_awareness",
    "previous_employer_formal_discussion",
    "previous_employer_resources",
    "previous_employer_anonymity_protection",
    "previous_employer_discussion_negative_consequences",
    "previous_employer_physical_health_discussion_negative_consequences",
    "previous_coworker_discussion_willingness",
    "previous_supervisor_discussion_willingness",
    "previous_employer_takes_mental_health_seriously",
    "previous_employer_coworker_negative_consequences",
    "family_history_mental_illness",
    "past_mental_health_disorder",
    "current_mental_health_disorder",
    "medical_professional_diagnosis",
    "sought_treatment",
    "work_interference_effective_treatment",
    "work_interference_ineffective_treatment",
    "remote_work",
]

# Explicitly excluded from population-wide clustering. These remain in the
# cleaned dataset and can still be used for descriptive or subgroup analysis.
EXCLUSION_REASONS = {
    "gender": "Sensitive demographic variable; retained for descriptive profiling.",
    "residence_country": "High-cardinality geographic identifier; retained for profiling.",
    "work_country": "High-cardinality geographic identifier; retained for profiling.",
    "residence_state": "US-specific geographic field with structural missingness.",
    "work_state": "US-specific geographic field with structural missingness.",
    "age": (
        "Demographic variable; kept for descriptive profiling and not used in "
        "the primary categorical clustering space."
    ),
    "tech_role": "Conditional role question; not comparable across all respondents.",
    "work_time_affected_pct": "Conditional follow-up with very high missingness.",
    "client_disclosure_negative_impact": "Only applicable after client disclosure.",
    "observed_disclosure_influence": "Conditional follow-up with limited applicability.",
    "why_or_why_not_physical_health": "Free text; high cardinality and qualitative in nature.",
    "why_or_why_not_mental_health": "Free text; high cardinality and qualitative in nature.",
    "work_position": "Multi-select role description; retained for profiling, not core clustering.",
    "diagnosed_conditions_current": "Sensitive multi-select clinical detail; excluded from clustering.",
    "believed_conditions_if_maybe": "Sensitive multi-select self-reported clinical detail.",
    "diagnosed_conditions_professional": "Sensitive multi-select clinical detail; excluded from clustering.",
    "self_employed_medical_coverage": (
        "Self-employment-specific item; not comparable across the full population."
    ),
    "self_employed_local_resources": (
        "Self-employment-specific item; not comparable across the full population."
    ),
    "client_disclosure": "Self-employment-specific disclosure item.",
    "coworker_disclosure": "Self-employment-specific disclosure item.",
    "productivity_affected": (
        "Self-employment-specific/conditional response; exclude from common matrix."
    ),
}


def build_feature_decision_register(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create an auditable include/exclude decision for every cleaned column."""
    core = set(CORE_CATEGORICAL_FEATURES)
    exclusions = EXCLUSION_REASONS
    records = []

    for column in df.columns:
        if column in core:
            decision = "include"
            reason = (
                "Population-level categorical workplace, employment, "
                "mental-health history or work-context feature."
            )
        elif column in exclusions:
            decision = "exclude"
            reason = exclusions[column]
        elif column.endswith("_items"):
            decision = "retain_for_profiling"
            reason = "Parsed multi-select representation; not part of the primary matrix."
        elif column.endswith("_ordinal"):
            decision = "retain_for_alternative_representation"
            reason = "Derived ordinal representation; use only in mixed-data/Gower design."
        elif column in {"source_row_id"}:
            decision = "identifier_only"
            reason = "Row lineage key; never a clustering feature."
        else:
            decision = "review_required"
            reason = (
                "No explicit inclusion decision is defined for this field. "
                "It is not automatically included in any model matrix."
            )

        records.append(
            {
                "column": column,
                "decision": decision,
                "reason": reason,
                "present_in_dataset": True,
            }
        )

    return pd.DataFrame(records)


def select_core_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Select shared categorical features that actually exist in the dataset.

    Missing expected fields are returned separately. The notebook must treat
    missing expected fields as a schema mismatch rather than silently ignoring
    them.
    """
    selected = [c for c in CORE_CATEGORICAL_FEATURES if c in df.columns]
    missing = [c for c in CORE_CATEGORICAL_FEATURES if c not in df.columns]

    return df.loc[:, selected].copy(), selected, missing


def assert_expected_features_present(missing: list[str]) -> None:
    """Stop preparation if the expected feature schema is incomplete."""
    if missing:
        raise ValueError(
            "The cleaned dataset is missing expected clustering features: "
            + ", ".join(missing)
            + ". Check schema.py column mappings and source headers."
        )
