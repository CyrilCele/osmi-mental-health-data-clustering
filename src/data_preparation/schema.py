"""Canonical source-column mapping for the OSMI 2016 survey."""

import re
import unicodedata

import pandas as pd

from .config import (
    SELF_EMPLOYED_SOURCE,
    TECH_EMPLOYER_SOURCE,
    TECH_ROLE_SOURCE,
)


COLUMN_OVERRIDES = {
    SELF_EMPLOYED_SOURCE: "self_employed",
    "How many employees does your company or organization have?": "company_size",
    TECH_EMPLOYER_SOURCE: "tech_employer",
    TECH_ROLE_SOURCE: "tech_role",

    "Does your employer provide mental health benefits as part of healthcare coverage?":
        "mental_health_benefits",

    "Do you know the options for mental health care available under your employer-provided coverage?":
        "employer_benefits_awareness",

    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?":
        "formal_mental_health_communication",

    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?":
        "mental_health_resources",

    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?":
        "anonymity_protected",

    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:":
        "medical_leave_ease",

    "Do you think that discussing a mental health disorder with your employer would have negative consequences?":
        "mh_disclosure_negative_consequences",

    "Do you think that discussing a physical health issue with your employer would have negative consequences?":
        "physical_disclosure_negative_consequences",

    "Would you feel comfortable discussing a mental health disorder with your coworkers?":
        "coworker_discussion_comfort",

    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?":
        "supervisor_discussion_comfort",

    "Do you feel that your employer takes mental health as seriously as physical health?":
        "mh_as_serious_as_physical",

    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?":
        "observed_negative_consequences",

    "Would you be willing to bring up a physical health issue with a potential employer in an interview?":
        "interview_disclosure_physical",

    "Would you bring up a mental health issue with a potential employer in an interview?":
        "interview_disclosure_mental",

    "Do you feel that being identified as a person with a mental health issue would hurt your career?":
        "perceived_career_harm",

    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?":
        "expected_coworker_stigma",

    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?":
        "observed_unsupportive_workplace_response",

    "What is your age?": "age",
    "What is your gender?": "gender",
    "What country do you live in?": "residence_country",
    "What US state or territory do you live in?": "residence_state",
    "What country do you work in?": "work_country",
    "What US state or territory do you work in?": "work_state",
    "Which of the following best describes your work position?": "work_position",
    "Do you work remotely?": "remote_work",

    "Do you have medical coverage (private insurance or state-provided) which includes treatment of mental health issues?":
        "medical_coverage_includes_mental_health",

    "Do you have a family history of mental illness?":
        "family_mental_illness_history",

    "Have you had a mental health disorder in the past?":
        "past_mental_health_disorder",

    "Do you currently have a mental health disorder?":
        "current_mental_health_disorder",

    "Have you been diagnosed with a mental health condition by a medical professional?":
        "professional_mental_health_diagnosis",

    "Have you ever sought treatment for a mental health issue from a mental health professional?":
        "mental_health_treatment_history",

    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?":
        "work_interference_effective_treatment",

    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?":
        "work_interference_ineffective_treatment",
}


def normalize_header(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value).strip()


NORMALIZED_OVERRIDES = {
    normalize_header(source): canonical
    for source, canonical in COLUMN_OVERRIDES.items()
}


def to_snake_case(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.replace("&", " and ")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value or "unnamed_column"


def build_column_mapping(columns) -> dict[str, str]:
    mapping = {}

    for source_name in columns:
        normalized = normalize_header(source_name)
        canonical = NORMALIZED_OVERRIDES.get(
            normalized,
            to_snake_case(normalized),
        )

        if canonical in mapping.values():
            raise ValueError(
                f"Canonical column-name collision: {canonical!r}. "
                "Inspect source headers and mapping overrides."
            )

        mapping[source_name] = canonical

    return mapping


def rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = build_column_mapping(df.columns)
    renamed = df.rename(columns=mapping).copy()

    audit = pd.DataFrame({
        "original_name": list(mapping),
        "canonical_name": list(mapping.values()),
    })

    if renamed.columns.duplicated().any():
        raise ValueError("Duplicate canonical column names detected.")

    return renamed, audit


def validate_model_schema(df: pd.DataFrame) -> None:
    from .config import MODEL_FEATURES

    missing = sorted(set(MODEL_FEATURES) - set(df.columns))

    if missing:
        raise KeyError(
            "The source schema does not provide all approved model "
            f"features. Missing canonical columns: {missing}"
        )
