from __future__ import annotations

import pandas as pd


STRUCTURALLY_CONDITIONAL_FIELDS = {
    "self_employment_fields": [
        "medical_coverage_includes_mental_health",
        "mental_health_resources_awareness",
        "client_disclosure",
        "client_disclosure_impact",
        "coworker_disclosure",
        "coworker_disclosure_impact",
        "productivity_affected",
        "work_time_affected_pct",
    ],
    "previous_employer_fields": [
        "previous_employer_mental_health_benefits",
        "previous_employer_benefits_awareness",
        "previous_employer_mental_health_discussion",
        "previous_employer_mental_health_resources",
        "previous_employer_confidentiality",
        "previous_employer_mental_health_disclosure_consequences",
        "previous_employer_physical_health_disclosure_consequences",
        "previous_coworker_disclosure_comfort",
        "previous_supervisor_disclosure_comfort",
        "previous_employer_mental_health_parity",
        "previous_workplace_disclosure_consequences",
    ],
    "us_geography_fields": [
        "residence_state",
        "work_state",
    ],
    "technology_role_fields": [
        "tech_role",
    ],
}


FULL_POPULATION_EXCLUSIONS = {
    "gender": (
        "Sensitive free-text demographic variable; retain for descriptive profiling."
    ),
    "age": (
        "Retain for cluster description; exclude from the primary categorical matrix."
    ),
    "residence_country": (
        "High-cardinality geographic variable; retain for geographic profiling."
    ),
    "work_country": (
        "High-cardinality geographic variable; retain for geographic profiling."
    ),
    "residence_state": (
        "US-specific geographic field with structural missingness."
    ),
    "work_state": (
        "US-specific geographic field with structural missingness."
    ),
    "tech_role": (
        "Conditional technology-role question; do not infer missing values."
    ),
    "medical_coverage_includes_mental_health": (
        "Self-employment-specific question; restricted subgroup coverage."
    ),
    "mental_health_resources_awareness": (
        "Self-employment-specific question; restricted subgroup coverage."
    ),
    "client_disclosure": (
        "Self-employment-specific disclosure question."
    ),
    "client_disclosure_impact": (
        "Conditional follow-up after client disclosure."
    ),
    "coworker_disclosure": (
        "Self-employment-specific disclosure question."
    ),
    "coworker_disclosure_impact": (
        "Conditional follow-up after coworker disclosure."
    ),
    "productivity_affected": (
        "Self-employment-specific question; retain for subgroup analysis, "
        "exclude from full-population clustering."
    ),
    "work_time_affected_pct": (
        "Conditional follow-up to productivity impact; high missingness."
    ),
    "observed_disclosure_influence": (
        "Limited applicability and substantial missingness."
    ),
    "interview_physical_health_reason": (
        "High-cardinality free text; retain for qualitative analysis."
    ),
    "interview_mental_health_reason": (
        "High-cardinality free text; retain for qualitative analysis."
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
        "Multi-select role description; transform for separate analysis, "
        "not as the original unsplit field."
    ),
}


def build_missingness_audit(df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(df)
    records = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        records.append(
            {
                "column": column,
                "row_count": n_rows,
                "missing_count": missing_count,
                "missing_pct": (
                    round(missing_count / n_rows * 100, 2)
                    if n_rows else 0.0
                ),
                "observed_count": int(df[column].notna().sum()),
                "distinct_non_missing": int(
                    df[column].nunique(dropna=True)
                ),
                "dtype": str(df[column].dtype),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["missing_pct", "column"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_applicability_register(
    df: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for group, columns in STRUCTURALLY_CONDITIONAL_FIELDS.items():
        for column in columns:
            if column not in df.columns:
                continue

            missing_count = int(df[column].isna().sum())

            records.append(
                {
                    "column": column,
                    "applicability_group": group,
                    "present_in_dataset": True,
                    "missing_count": missing_count,
                    "observed_count": int(df[column].notna().sum()),
                    "missing_pct": round(
                        df[column].isna().mean() * 100, 2
                    ),
                    "handling": (
                        "Preserve missingness; do not impute or interpret "
                        "structural missingness as a negative response."
                    ),
                }
            )

    return pd.DataFrame(records)


def build_exclusion_register(
    df: pd.DataFrame,
    additional_exclusions: dict[str, str] | None = None,
) -> pd.DataFrame:
    exclusions = dict(FULL_POPULATION_EXCLUSIONS)

    if additional_exclusions:
        exclusions.update(additional_exclusions)

    records = []

    for column, reason in exclusions.items():
        if column in df.columns:
            records.append(
                {
                    "column": column,
                    "excluded_from_full_population_clustering": True,
                    "reason": reason,
                    "retained_in_cleaned_dataset": True,
                }
            )

    return pd.DataFrame(records)
