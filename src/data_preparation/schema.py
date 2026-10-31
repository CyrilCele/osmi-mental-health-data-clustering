import re
import unicodedata

import pandas as pd


# Explicit semantic names for cloumns used frequently in the pipeline.
# All other names are generated deterministically from their source labels.
COLUMN_OVERRIDES = {
    "index": "source_index",
    "Are you self-employed?": "self_employed",
    "How many employees does your company or organization have?": "company_size",
    "Is your employer primarily a tech company/organization?": "tech_employer",
    "Is your primary role within your company related to tech/IT?": "tech_role",
    "Does your employer provide mental health benefits as part of healthcare "
    "coverage?": "employer_mental_health_benefits",
    "Do you know the options for mental health care available under your employer-"
    "provided coverage?": "employer_benefits_awareness",
    "Has your employer ever formally discussed mental health (for example, as part "
    "of a wellness campaign or other official communication)?": "employer_mental_health_discussion",
    "Does your employer offer resources to learn more about mental health concerns "
    "and options for seeking help?": "employer_mental_health_resources",
    "Is your anonymity protected if you choose to take advantage of mental health "
    "or substance abuse treatment resources provided by your employer?": "employer_confidentiality",
    "If a mental health issue prompted you to request a medical leave from work, "
    "asking for that leave would be:": "medical_leave_ease",
    "Do you think that discussing a mental health disorder with your employer would "
    "have negative consequences?": "employer_mental_health_disclosure_consequences",
    "Do you think that discussing a physical health issue with your employer would "
    "have negative consequences?": "employer_physical_health_disclosure_consequences",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?":
    "coworker_disclosure_comfort",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?":
    "supervisor_disclosure_comfort",
    "Do you feel that your employer takes mental health as seriously as physical health?":
    "employer_mental_health_parity",
    "Have you heard of or observed negative consequences for co-workers who have been "
    "open about mental health issues in your workplace?": "observed_workplace_disclosure_consequences",
    "Do you have medical coverage (private insurance or state-provided) which includes "
    "treatment of  mental health issues?": "medical_coverage_includes_mental_health",
    "Do you know local or online resources to seek help for a mental health disorder?":
    "mental_health_resources_awareness",
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal "
    "this to clients or business contacts?": "client_disclosure",
    "If you have revealed a mental health issue to a client or business contact, do you "
    "believe this has impacted you negatively?": "client_disclosure_impact",
    "If you have been diagnosed or treated for a mental health disorder, do you ever reveal "
    "this to coworkers or employees?": "coworker_disclosure",
    "If you have revealed a mental health issue to a coworker or employee, do you believe "
    "this has impacted you negatively?": "coworker_disclosure_impact",
    "Do you believe your productivity is ever affected by a mental health issue?": "productivity_affected",
    "If yes, what percentage of your work time (time performing primary or secondary job "
    "functions) is affected by a mental health issue?": "work_time_affected_pct",
    "Do you have previous employers?": "has_previous_employers",
    "Have your previous employers provided mental health benefits?": "previous_employer_mental_health_benefits",
    "Were you aware of the options for mental health care provided by your previous "
    "employers?": "previous_employer_benefits_awareness",
    "Did your previous employers ever formally discuss mental health (as part of a "
    "wellness campaign or other official communication)?": "previous_employer_mental_health_discussion",
    "Did your previous employers provide resources to learn more about mental health "
    "issues and how to seek help?": "previous_employer_mental_health_resources",
    "Was your anonymity protected if you chose to take advantage of mental health or "
    "substance abuse treatment resources with previous employers?": "previous_employer_confidentiality",
    "Do you think that discussing a mental health disorder with previous employers "
    "would have negative consequences?": "previous_employer_mental_health_disclosure_consequences",
    "Do you think that discussing a physical health issue with previous employers "
    "would have negative consequences?": "previous_employer_physical_health_disclosure_consequences",
    "Would you have been willing to discuss a mental health issue with your "
    "previous co-workers?": "previous_coworker_disclosure_comfort",
    "Would you have been willing to discuss a mental health issue with your "
    "direct supervisor(s)?": "previous_supervisor_disclosure_comfort",
    "Did you feel that your previous employers took mental health as seriously "
    "as physical health?": "previous_employer_mental_health_parity",
    "Did you hear of or observe negative consequences for co-workers with mental "
    "health issues in your previous workplaces?": "previous_workplace_disclosure_consequences",
    "Would you be willing to bring up a physical health issue with a potential "
    "employer in an interview?": "interview_physical_health_disclosure",
    "Why or why not?": "interview_physical_health_reason",
    "Would you bring up a mental health issue with a potential employer in an "
    "interview?": "interview_mental_health_disclosure",
    "Why or why not?.1": "interview_mental_health_reason",
    "Do you feel that being identified as a person with a mental health issue "
    "would hurt your career?": "perceived_career_impact",
    "Do you think that team members/co-workers would view you more negatively "
    "if they knew you suffered from a mental health issue?": "anticipated_coworker_stigma",
    "How willing would you be to share with friends and family that you have a "
    "mental illness?": "friends_family_disclosure_willingness",
    "Have you observed or experienced an unsupportive or badly handled response "
    "to a mental health issue in your current or previous workplace?":
    "observed_unsupportive_workplace_response",
    "Have your observations of how another individual who discussed a mental health "
    "disorder made you less likely to reveal a mental health issue yourself in your "
    "current workplace?": "observed_disclosure_influence",
    "Do you have a family history of mental illness?": "family_mental_illness_history",
    "Have you had a mental health disorder in the past?": "past_mental_health_disorder",
    "Do you currently have a mental health disorder?": "current_mental_health_disorder",
    "If yes, what condition(s) have you been diagnosed with?": "current_diagnosed_conditions",
    "If maybe, what condition(s) do you believe you have?": "suspected_conditions",
    "Have you been diagnosed with a mental health condition by a medical professional?":
    "professional_mental_health_diagnosis",
    "If so, what condition(s) were you diagnosed with?": "professionally_diagnosed_conditions",
    "Have you ever sought treatment for a mental health issue from a mental health professional?":
    "mental_health_treatment_history",
    "If you have a mental health issue, do you feel that it interferes with your "
    "work when being treated effectively?": "work_interference_effective_treatment",
    "If you have a mental health issue, do you feel that it interferes with your work "
    "when NOT being treated effectively?": "work_interference_ineffective_treatment",
    "What is your age?": "age",
    "What is your gender?": "gender",
    "What country do you live in?": "residence_country",
    "What US state or territory do you live in?": "residence_state",
    "What country do you work in?": "work_country",
    "What US state or territory do you work in?": "work_state",
    "Which of the following best describes your work position?": "work_position",
    "Do you work remotely?": "remote_work",
}


def to_snake_case(value: str) -> str:
    """Convert an arbitrary source label into a deterministic snake_case name."""

    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.replace("&", " and ")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()

    return value or "unnamed_column"


def build_column_mapping(columns) -> dict[str, str]:
    """Build a complete, collision-checked source-to-canonical mapping."""
    mapping = {}

    for source_name in columns:
        canonical = COLUMN_OVERRIDES.get(
            source_name,
            to_snake_case(source_name),
        )

        if canonical in mapping.values():
            raise ValueError(
                f"Canonical column-name collision detected: {canonical!r}"
            )

        mapping[source_name] = canonical

    return mapping


def rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = build_column_mapping(df.columns)
    renamed = df.rename(columns=mapping).copy()

    audit = pd.DataFrame({
        "original_name": list(mapping.keys()),
        "canonical_name": list(mapping.values()),
    })

    if renamed.columns.duplicated().any():
        raise ValueError("Duplicate canonical column names detected.")

    return renamed, audit
