from pathlib import Path

# Safely resolve PROJECT_ROOT whether imported from a script or Jupyter notebook
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
else:
    # Fallback for Jupyter Notebooks (assumes notebook is run from a subfolder like notebooks/)
    # Adjust .parent count if your notebook is nested deeper or shallower
    PROJECT_ROOT = Path.cwd().resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPARATION_DIR = PROCESSED_DIR / "data_preparation"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SOURCE_FILENAME = "mental-heath-in-tech-2016_20161114.csv"
SOURCE_PATH = RAW_DIR / SOURCE_FILENAME

RANDOM_STATE = 42
EXPECTED_RAW_SHAPE = (1433, 63)
EXPECTED_COHORT_SIZE = 1131

# Approved descriptive-age quality rule.
MIN_PLAUSIBLE_AGE = 18
MAX_PLAUSIBLE_AGE = 75

# Source values are preserved; cleaned age is a separate descriptor.
AGE_RAW_COLUMN = "age"
AGE_CLEAN_COLUMN = "age_clean"

# Exact source questions defining the primary cohort.
SELF_EMPLOYED_SOURCE = "Are you self-employed?"
TECH_EMPLOYER_SOURCE = (
    "Is your employer primarily a tech company/organization?"
)
TECH_ROLE_SOURCE = (
    "Is your primary role within your company related to tech/IT?"
)

# Approved 15-feature study design.
DOMAIN_FEATURES = {
    "employer_support_clarity": [
        "mental_health_benefits",
        "formal_mental_health_communication",
        "mental_health_resources",
        "anonymity_protected",
    ],
    "psychological_safety_consequences": [
        "medical_leave_ease",
        "mh_disclosure_negative_consequences",
        "physical_disclosure_negative_consequences",
        "coworker_discussion_comfort",
        "supervisor_discussion_comfort",
        "mh_as_serious_as_physical",
        "observed_negative_consequences",
    ],
    "anticipated_stigma_disclosure": [
        "interview_disclosure_physical",
        "interview_disclosure_mental",
        "perceived_career_harm",
        "expected_coworker_stigma",
    ],
}

MODEL_FEATURES = [
    feature
    for features in DOMAIN_FEATURES.values()
    for feature in features
]

DOMAIN_WEIGHTS = {
    domain: 1.0 / len(DOMAIN_FEATURES)
    for domain in DOMAIN_FEATURES
}

assert len(MODEL_FEATURES) == 15
assert len(set(MODEL_FEATURES)) == 15
assert [len(features) for features in DOMAIN_FEATURES.values()] == [4, 7, 4]
assert abs(sum(DOMAIN_WEIGHTS.values()) - 1.0) < 1e-12

# Ordinal mappings are intentionally restricted to semantically ordered items.
ORDINAL_RESPONSE_MAPS = {
    "mh_disclosure_negative_consequences": {
        "No": 0,
        "Maybe": 1,
        "Yes": 2,
    },
    "physical_disclosure_negative_consequences": {
        "No": 0,
        "Maybe": 1,
        "Yes": 2,
    },
    "coworker_discussion_comfort": {
        "No": 0,
        "Maybe": 1,
        "Yes": 2,
    },
    "supervisor_discussion_comfort": {
        "No": 0,
        "Maybe": 1,
        "Yes": 2,
    },
    "mh_as_serious_as_physical": {
        "No": 0,
        "Maybe": 1,
        "Yes": 2,
    },
    "interview_disclosure_physical": {
        "No": 0,
        "Maybe": 1,
        "Yes": 2,
    },
    "interview_disclosure_mental": {
        "No": 0,
        "Maybe": 1,
        "Yes": 2,
    },
}

MEDICAL_LEAVE_ORDER = [
    "Very difficult",
    "Somewhat difficult",
    "Neither easy nor difficult",
    "Somewhat easy",
    "Very easy",
]

MEDICAL_LEAVE_UNKNOWN = "I don't know"

# These items are nominal: their response categories do not form one
# defensible monotonic scale.
NOMINAL_MODEL_FEATURES = [
    "mental_health_benefits",
    "formal_mental_health_communication",
    "mental_health_resources",
    "anonymity_protected",
    "observed_negative_consequences",
    "perceived_career_harm",
    "expected_coworker_stigma",
]

# Supplementary fields are deliberately outside the 15-feature matrix.
SUPPLEMENTARY_FEATURES = [
    "employer_benefits_awareness",
    "observed_unsupportive_workplace_response",
]

# Sensitive personal-health fields are never model features.
SENSITIVE_EXCLUSIONS = [
    "family_mental_illness_history",
    "past_mental_health_disorder",
    "current_mental_health_disorder",
    "professional_mental_health_diagnosis",
    "mental_health_treatment_history",
    "current_diagnosed_conditions",
    "suspected_conditions",
    "professionally_diagnosed_conditions",
    "work_interference_effective_treatment",
    "work_interference_ineffective_treatment",
]

# Contextual descriptors are kept separate from the feature matrix.
PROFILE_DESCRIPTORS = [
    "company_size",
    "tech_employer",
    "tech_role",
    "remote_work",
    "age",
    "gender",
    "residence_country",
    "residence_state",
    "work_country",
    "work_state",
    "work_position",
    "employer_benefits_awareness",
    "observed_unsupportive_workplace_response",
]

# The raw source is immutable. All processing operates on copies.
