from __future__ import annotations

import pandas as pd


# Candidates supported by the Data Understanding notebook.
# Conditional variables are deliberately not included here.
CORE_CATEGORICAL_FEATURES = [
    "self_employed",
    "tech_employer",
    "employer_mental_health_benefits",
    "employer_benefits_awareness",
    "employer_mental_health_discussion",
    "employer_mental_health_resources",
    "employer_confidentiality",
    "medical_leave_ease",
    "employer_mental_health_disclosure_consequences",
    "employer_physical_health_disclosure_consequences",
    "coworker_disclosure_comfort",
    "supervisor_disclosure_comfort",
    "employer_mental_health_parity",
    "observed_workplace_disclosure_consequences",
    "observed_unsupportive_workplace_response",
    "family_mental_illness_history",
    "past_mental_health_disorder",
    "current_mental_health_disorder",
    "professional_mental_health_diagnosis",
    "mental_health_treatment_history",
    "work_interference_effective_treatment",
    "work_interference_ineffective_treatment",
    "remote_work",
]


# These are legitimate candidates but require an explicit treatment of
# applicability or a separate representation before inclusion.
CONDITIONAL_FEATURE_CANDIDATES = {
    "company_size": (
        "Only meaningful for respondents with an employer. Preserve "
        "non-applicability for self-employed respondents."
    ),
    "previous_employer_mental_health_benefits": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_employer_benefits_awareness": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_employer_mental_health_discussion": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_employer_mental_health_resources": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_employer_confidentiality": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_employer_mental_health_disclosure_consequences": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_employer_physical_health_disclosure_consequences": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_coworker_disclosure_comfort": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_supervisor_disclosure_comfort": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_employer_mental_health_parity": (
        "Only applicable to respondents with previous employers."
    ),
    "previous_workplace_disclosure_consequences": (
        "Only applicable to respondents with previous employers."
    ),
}


EXCLUSION_REASONS = {
    "gender": (
        "Sensitive free-text demographic variable; retain for descriptive profiling."
    ),
    "age": (
        "Retain for cluster description; not part of the primary categorical matrix."
    ),
    "residence_country": (
        "High-cardinality geographic variable; retain for profiling."
    ),
    "work_country": (
        "High-cardinality geographic variable; retain for profiling."
    ),
    "residence_state": (
        "US-specific geographic field with structural missingness."
    ),
    "work_state": (
        "US-specific geographic field with structural missingness."
    ),
    "tech_role": (
        "Conditional technology-role question; exclude from full-population matrix."
    ),
    "medical_coverage_includes_mental_health": (
        "Self-employment-specific; retain for subgroup analysis."
    ),
    "mental_health_resources_awareness": (
        "Self-employment-specific; retain for subgroup analysis."
    ),
    "client_disclosure": (
        "Self-employment-specific; retain for subgroup analysis."
    ),
    "client_disclosure_impact": (
        "Conditional follow-up after client disclosure."
    ),
    "coworker_disclosure": (
        "Self-employment-specific; retain for subgroup analysis."
    ),
    "coworker_disclosure_impact": (
        "Conditional follow-up after coworker disclosure."
    ),
    "productivity_affected": (
        "Self-employment-specific; retain for subgroup analysis, "
        "exclude from full-population clustering."
    ),
    "work_time_affected_pct": (
        "Highly sparse conditional follow-up to productivity impact."
    ),
    "observed_disclosure_influence": (
        "Limited applicability and substantial missingness."
    ),
    "interview_physical_health_reason": (
        "High-cardinality free text; qualitative analysis only."
    ),
    "interview_mental_health_reason": (
        "High-cardinality free text; qualitative analysis only."
    ),
    "current_diagnosed_conditions": (
        "Sensitive multi-select condition description; not a raw clustering feature."
    ),
    "suspected_conditions": (
        "Sensitive multi-select suspected condition description."
    ),
    "professionally_diagnosed_conditions": (
        "Sensitive multi-select diagnosis description."
    ),
    "work_position": (
        "Multi-select role description; original unsplit field excluded."
    ),
}


def build_feature_decision_register(
    df: pd.DataFrame,
) -> pd.DataFrame:
    core = set(CORE_CATEGORICAL_FEATURES)
    conditional = CONDITIONAL_FEATURE_CANDIDATES

    records = []

    for column in df.columns:
        if column in core:
            decision = "candidate_for_full_population_matrix"
            reason = (
                "Candidate identified in Data Understanding; subject to "
                "coverage, distribution, and redundancy checks."
            )
        elif column in conditional:
            decision = "conditional_candidate"
            reason = conditional[column]
        elif column in EXCLUSION_REASONS:
            decision = "exclude_from_full_population_matrix"
            reason = EXCLUSION_REASONS[column]
        elif column.endswith("_items") or column.endswith("_selections"):
            decision = "retain_for_separate_analysis"
            reason = (
                "Parsed multi-select representation; not automatically "
                "included in the primary clustering matrix."
            )
        elif column.endswith("_ordinal"):
            decision = "alternative_representation"
            reason = (
                "Derived ordinal representation; inclusion requires an "
                "explicit model-specific decision."
            )
        elif column in {"source_row_id", "source_index"}:
            decision = "identifier_only"
            reason = "Row lineage identifier; never a clustering feature."
        else:
            decision = "review_required"
            reason = (
                "No explicit inclusion decision is defined. "
                "Do not automatically include this field."
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
    selected = [
        column
        for column in CORE_CATEGORICAL_FEATURES
        if column in df.columns
    ]

    missing = [
        column
        for column in CORE_CATEGORICAL_FEATURES
        if column not in df.columns
    ]

    return df.loc[:, selected].copy(), selected, missing


def assert_expected_features_present(missing: list[str]) -> None:
    if missing:
        raise ValueError(
            "Expected clustering candidate columns are missing: "
            + ", ".join(missing)
            + ". Check schema.py and the source-header mapping."
        )
