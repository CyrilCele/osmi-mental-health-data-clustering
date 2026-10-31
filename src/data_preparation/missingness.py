"""Missingness profiling and applicability classification."""

from __future__ import annotations

import pandas as pd


# These fields are only meaningful for respondents who meet the stated
# applicability condition. Their missing values must not be treated as No.
STRUCTURALLY_CONDITIONAL_FIELDS = {
    "current_employer_fields": [
        "company_size",
        "tech_employer",
        "tech_role",
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
    ],
    "self_employment_fields": [
        "self_employed_medical_coverage",
        "self_employed_local_resources",
        "client_disclosure",
        "client_disclosure_negative_impact",
        "coworker_disclosure",
        "coworker_disclosure_negative_impact",
        "productivity_affected",
        "work_time_affected_pct",
    ],
    "previous_employer_fields": [
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
    ],
}

# High-missingness or conditional fields excluded from the full-population
# clustering feature space, but retained in the cleaned analytical dataset.
FULL_POPULATION_EXCLUSIONS = {
    "work_time_affected_pct": (
        "Conditional follow-up to productivity impact; very high missingness."
    ),
    "client_disclosure_negative_impact": (
        "Only applicable to respondents who disclosed to clients."
    ),
    "observed_disclosure_influence": (
        "Conditional workplace follow-up; not consistently applicable."
    ),
    "tech_role": (
        "Conditional on employment context and whether the employer is a tech organisation."
    ),
    "residence_state": (
        "US-specific geographic field; structurally unavailable for most non-US respondents."
    ),
    "work_state": (
        "US-specific geographic field; structurally unavailable for most non-US respondents."
    ),
    "why_or_why_not_physical_health": (
        "High-cardinality free text; retained for qualitative analysis."
    ),
    "why_or_why_not_mental_health": (
        "High-cardinality free text; retained for qualitative analysis."
    ),
}


def build_missingness_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missingness counts, rates and observed cardinality."""
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
                    round(missing_count / n_rows * 100, 2) if n_rows else 0.0
                ),
                "observed_count": int(df[column].notna().sum()),
                "distinct_non_missing": int(df[column].nunique(dropna=True)),
                "dtype": str(df[column].dtype),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(["missing_pct", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_applicability_register(df: pd.DataFrame) -> pd.DataFrame:
    """Document known conditional fields and whether each exists in the data."""
    records = []

    for group, columns in STRUCTURALLY_CONDITIONAL_FIELDS.items():
        for column in columns:
            if column in df.columns:
                records.append(
                    {
                        "column": column,
                        "applicability_group": group,
                        "present_in_dataset": True,
                        "missing_count": int(df[column].isna().sum()),
                        "missing_pct": round(df[column].isna().mean() * 100, 2),
                        "handling": (
                            "Do not impute as No; interpret missingness in light "
                            "of the survey routing/applicability condition."
                        ),
                    }
                )

    return pd.DataFrame(records)


def build_exclusion_register(
    df: pd.DataFrame,
    additional_exclusions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return explicit reasons for excluding fields from population-wide clustering."""
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
