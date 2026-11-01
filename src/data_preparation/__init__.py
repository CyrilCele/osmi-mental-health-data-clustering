"""OSMI 2016 project-specific data preparation package."""

from .config import (
    DOMAIN_FEATURES,
    DOMAIN_WEIGHTS,
    MODEL_FEATURES,
    EXPECTED_COHORT_SIZE,
)

from .ingestion import (
    load_source,
    sha256_file,
)

from .schema import (
    rename_columns,
    validate_model_schema,
)

from .cleaning import (
    normalize_text_value,
    normalize_string_columns,
    clean_age,
    clean_gender,
    clean_countries,
)

from .type_harmonization import (
    canonicalise_response,
    canonicalise_yes_no,
    harmonise_categorical_columns,
    build_type_metadata,
)

from .missingness import (
    build_missingness_audit,
    build_applicability_register,
    build_exclusion_register,
    assert_primary_features_complete,
)

from .feature_selection import (
    build_feature_decision_register,
    select_model_features,
    build_profile_descriptors,
    validate_feature_exclusions,
)

from .feature_engineering import (
    encode_ordinal_features,
    encode_medical_leave,
    build_domain_metadata,
    engineer_preparation_features,
    validate_encoded_integer_types,
)

from .transformations import (
    feature_distance_matrix,
    domain_distance_matrices,
    gower_distance_matrix,
    validate_distance_matrix,
)

from .validation import (
    validate_cohort,
    validate_feature_matrix,
    validate_domain_weights,
    validate_integer_codes,
    validate_distance,
    validate_row_lineage,
    run_preparation_validation,
)

from .persistence import (
    ensure_output_directory,
    save_dataframe,
    save_numpy,
    save_json,
    build_respondent_row_mapping,
    build_feature_metadata,
    save_preparation_outputs,
)

__all__ = [
    "DOMAIN_FEATURES",
    "DOMAIN_WEIGHTS",
    "MODEL_FEATURES",
    "EXPECTED_COHORT_SIZE",
    "load_source",
    "sha256_file",
    "rename_columns",
    "validate_model_schema",
    "normalize_text_value",
    "normalize_string_columns",
    "clean_age",
    "clean_gender",
    "clean_countries",
    "canonicalise_response",
    "canonicalise_yes_no",
    "harmonise_categorical_columns",
    "build_type_metadata",
    "build_missingness_audit",
    "build_applicability_register",
    "build_exclusion_register",
    "assert_primary_features_complete",
    "build_feature_decision_register",
    "select_model_features",
    "build_profile_descriptors",
    "validate_feature_exclusions",
    "encode_ordinal_features",
    "encode_medical_leave",
    "build_domain_metadata",
    "engineer_preparation_features",
    "validate_encoded_integer_types",
    "feature_distance_matrix",
    "domain_distance_matrices",
    "gower_distance_matrix",
    "validate_distance_matrix",
    "validate_cohort",
    "validate_feature_matrix",
    "validate_domain_weights",
    "validate_integer_codes",
    "validate_distance",
    "validate_row_lineage",
    "run_preparation_validation",
    "ensure_output_directory",
    "save_dataframe",
    "save_numpy",
    "save_json",
    "build_respondent_row_mapping",
    "build_feature_metadata",
    "save_preparation_outputs",
]
