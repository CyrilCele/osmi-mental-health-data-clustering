"""Conservative, auditable cleaning for the OSMI 2016 survey."""

import re
import unicodedata

import pandas as pd

from .config import (
    MIN_PLAUSIBLE_AGE,
    MAX_PLAUSIBLE_AGE,
)


def normalize_text_value(value):
    """Normalise formatting without changing response meaning."""
    if pd.isna(value):
        return pd.NA

    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else pd.NA


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in result.select_dtypes(
        include=["object", "string"]
    ).columns:
        result[column] = (
            result[column]
            .map(normalize_text_value)
            .astype("string")
        )

    return result


def clean_age(
    df: pd.DataFrame,
    source_column: str = "age",
    cleaned_column: str = "age_clean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preserve original age and create a separate cleaned descriptor.

    Approved rule: values below 13 or above 100 are implausible.
    No age imputation is performed because age is descriptive only.
    """
    result = df.copy()

    if source_column not in result:
        raise KeyError(f"Missing age source column: {source_column}")

    numeric_age = pd.to_numeric(
        result[source_column],
        errors="coerce",
    )

    non_numeric = result[source_column].notna() & numeric_age.isna()

    if non_numeric.any():
        values = result.loc[non_numeric, source_column].unique().tolist()
        raise ValueError(f"Unexpected non-numeric age values: {values}")

    implausible = (
        numeric_age.lt(MIN_PLAUSIBLE_AGE)
        | numeric_age.gt(MAX_PLAUSIBLE_AGE)
    )

    result[cleaned_column] = (
        numeric_age.mask(implausible).astype("Int64")
    )

    result[f"{source_column}_implausible_flag"] = (
        implausible.fillna(False).astype("bool")
    )

    audit = pd.DataFrame({
        "source_row_id": (
            result["source_row_id"]
            if "source_row_id" in result
            else result.index
        ),
        "age_original": result[source_column],
        "age_clean": result[cleaned_column],
        "implausible": implausible,
    })

    audit = audit.loc[implausible].reset_index(drop=True)

    return result, audit


# The following classification sets preserve the explicitly supplied
# gender classification decisions from the project.

male_values = {
    "male", "Male", "M", "m", "man", "Cis male", "Male.",
    "male 9:1 female, roughly", "Male (cis)", "Man", "Sex is male",
    "cis male", "Malr", "Dude", "mail", "M|", "Male/genderqueer",
    "male ", "Cis Male", "cisdude", "cis man", "MALE", "Male ",
    "Male (trans, FtM)",
    (
        "I'm a man why didn't you make this a drop down question. "
        "You should of asked sex? And I would of answered yes please. "
        "Seriously how much text can this take? "
    ),
}

female_values = {
    "Female", "female", "female ", "F", "Woman", "fm", "f",
    "Cis female ", "Transitioned, M2F", " Female", "Cis-woman",
    "AFAB", "Genderfluid (born female)", "Female ", "woman",
    "female/woman", "Cisgender Female", "fem",
    "female-bodied; no feelings about gender",
    "Female (props for making this a freeform field, though)",
    "I identify as female.", "Female assigned at birth ",
    "Female or Multi-Gender Femme", "mtf", "Transgender woman",
}

other_values = {
    "non-binary", "Agender", "Nonbinary", "Queer", "nb masculine",
    "human", "genderqueer woman", "genderqueer", "Unicorn",
    "Bigender", "Enby", "Other", "Other/Transfeminine", "Human",
    "Genderqueer", "Genderflux demi-girl", "Genderfluid", "Fluid",
    "Enby Bigender", "Androgynous", "none of your business",
}


def _gender_key(value):
    normalized = normalize_text_value(value)

    if pd.isna(normalized):
        return None

    return str(normalized).casefold()


def _build_gender_lookup():
    lookup = {}

    for label, values in (
        ("Male", male_values),
        ("Female", female_values),
        ("Other", other_values),
    ):
        for value in values:
            key = _gender_key(value)

            if key in lookup and lookup[key] != label:
                raise ValueError(
                    f"Conflicting gender classification: {value!r}"
                )

            lookup[key] = label

    return lookup


GENDER_LOOKUP = _build_gender_lookup()


def normalize_gender(value):
    key = _gender_key(value)

    if key is None:
        return pd.NA

    return GENDER_LOOKUP.get(key, "Other")


def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "gender" not in result:
        raise KeyError("Expected canonical column 'gender'.")

    if "gender_original" not in result:
        result["gender_original"] = result["gender"].copy()

    result["gender"] = (
        result["gender_original"]
        .map(normalize_gender)
        .astype("string")
    )

    return result


def clean_countries(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in (
        "residence_country",
        "work_country",
        "residence_state",
        "work_state",
    ):
        if column in result:
            result[column] = (
                result[column]
                .map(normalize_text_value)
                .astype("string")
            )

    return result
