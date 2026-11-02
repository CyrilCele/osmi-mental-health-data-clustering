"""OSMI 2016 project-specific data preparation package."""

from .cleaning import (
    clean_age,
    clean_countries,
    clean_gender,
    normalize_string_columns,
    normalize_text_value,
)
from .config import (
    DOMAIN_FEATURES,
    DOMAIN_WEIGHTS,
    EXPECTED_COHORT_SIZE,
    MODEL_FEATURES,
    PREPARATION_DIR,
)
from .feature_engineering import (
    build_domain_metadata,
    encode_medical_leave,
    encode_ordinal_features,
    engineer_preparation_features,
    validate_encoded_integer_types,
)
from .feature_selection import (
    build_feature_decision_register,
    build_profile_descriptors,
    select_model_features,
    validate_feature_exclusions,
)
from .ingestion import (
    load_source,
    sha256_file,
)
from .missingness import (
    assert_primary_features_complete,
    build_applicability_register,
    build_exclusion_register,
    build_missingness_audit,
)
from .persistence import (
    build_feature_metadata,
    build_respondent_row_mapping,
    ensure_output_directory,
    save_dataframe,
    save_json,
    save_numpy,
    save_preparation_outputs,
)
from .schema import (
    rename_columns,
    validate_model_schema,
)
from .transformations import (
    domain_distance_matrices,
    feature_distance_matrix,
    gower_distance_matrix,
    validate_distance_matrix,
)
from .type_harmonization import (
    build_type_metadata,
    canonicalise_response,
    canonicalise_yes_no,
    harmonise_categorical_columns,
)
from .validation import (
    run_preparation_validation,
    validate_cohort,
    validate_distance,
    validate_domain_weights,
    validate_feature_matrix,
    validate_integer_codes,
    validate_row_lineage,
)

__all__ = [
    "DOMAIN_FEATURES",
    "DOMAIN_WEIGHTS",
    "EXPECTED_COHORT_SIZE",
    "MODEL_FEATURES",
    "PREPARATION_DIR",
    "assert_primary_features_complete",
    "build_applicability_register",
    "build_domain_metadata",
    "build_exclusion_register",
    "build_feature_decision_register",
    "build_feature_metadata",
    "build_missingness_audit",
    "build_profile_descriptors",
    "build_respondent_row_mapping",
    "build_type_metadata",
    "canonicalise_response",
    "canonicalise_yes_no",
    "clean_age",
    "clean_countries",
    "clean_gender",
    "domain_distance_matrices",
    "encode_medical_leave",
    "encode_ordinal_features",
    "engineer_preparation_features",
    "ensure_output_directory",
    "feature_distance_matrix",
    "gower_distance_matrix",
    "harmonise_categorical_columns",
    "load_source",
    "normalize_string_columns",
    "normalize_text_value",
    "rename_columns",
    "run_preparation_validation",
    "save_dataframe",
    "save_json",
    "save_numpy",
    "save_preparation_outputs",
    "select_model_features",
    "sha256_file",
    "validate_cohort",
    "validate_distance",
    "validate_distance_matrix",
    "validate_domain_weights",
    "validate_encoded_integer_types",
    "validate_feature_exclusions",
    "validate_feature_matrix",
    "validate_integer_codes",
    "validate_model_schema",
    "validate_row_lineage",
]
