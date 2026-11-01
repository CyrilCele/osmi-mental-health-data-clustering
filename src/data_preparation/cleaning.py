import re
import unicodedata

import numpy as np
import pandas as pd

from .config import INVALID_AGES


def normalize_text_value(value):
    """Clean formatting while preserving the response's semantic meaning."""
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()

    if "age" not in result.columns:
        raise KeyError("Expected canonical column 'age'.")

    original = result["age"].copy()
    numeric_age = pd.to_numeric(original, errors="coerce")

    non_numeric_mask = original.notna() & numeric_age.isna()

    if non_numeric_mask.any():
        unexpected = original.loc[non_numeric_mask].unique().tolist()
        raise ValueError(
            "Unexpected non-numeric age values found. "
            f"Inspect these before cleaning: {unexpected}"
        )

    invalid_mask = numeric_age.isin(INVALID_AGES)

    audit_rows = []

    for index in result.index[invalid_mask]:
        audit_rows.append(
            {
                "source_row_id": (
                    result.at[index, "source_row_id"]
                    if "source_row_id" in result.columns
                    else index
                ),
                "age_original": original.at[index],
                "decision": "Set to missing: identified invalid age.",
            }
        )

    result["age"] = numeric_age.mask(invalid_mask).astype("Float64")

    return result, pd.DataFrame(audit_rows)


# These sets are the user's authoritative classifications.
male_values = {
    "male",
    "Male",
    "M",
    "m",
    "man",
    "Cis male",
    "Male.",
    "male 9:1 female, roughly",
    "Male (cis)",
    "Man",
    "Sex is male",
    "cis male",
    "Malr",
    "Dude",
    "mail",
    "M|",
    "Male/genderqueer",
    "male ",
    "Cis Male",
    "cisdude",
    "cis man",
    "MALE",
    "Male ",
    "Male (trans, FtM)",
    (
        "I'm a man why didn't you make this a drop down question. "
        "You should of asked sex? And I would of answered yes please. "
        "Seriously how much text can this take? "
    )
}

female_values = {
    "Female",
    "female",
    "female ",
    "F",
    "Woman",
    "fm",
    "f",
    "Cis female ",
    "Transitioned, M2F",
    " Female",
    "Cis-woman",
    "AFAB",
    "Genderfluid (born female)",
    "Female ",
    "woman",
    "female/woman",
    "Cisgender Female",
    "fem",
    "female-bodied; no feelings about gender",
    "Female (props for making this a freeform field, though)",
    "I identify as female.",
    "Female assigned at birth ",
    "Female or Multi-Gender Femme",
    "mtf",
    "Transgender woman"
}

other_values = {
    "non-binary",
    "Agender",
    "Nonbinary",
    "Queer",
    "nb masculine",
    "human",
    "genderqueer woman",
    "genderqueer",
    "Unicorn",
    "Bigender",
    "Enby",
    "Other",
    "Other/Transfeminine",
    "Human",
    "Genderqueer",
    "Genderflux demi-girl",
    "Genderfluid",
    "Fluid",
    "Enby Bigender",
    "Androgynous",
    "none of your business"
}


def _gender_key(value) -> str | None:
    """Apply only case and whitespace normalisation for set matching."""
    normalized = normalize_text_value(value)

    if pd.isna(normalized):
        return None

    return str(normalized).casefold()


def _build_gender_lookup() -> dict[str, str]:
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
                    f"Conflicting gender classification for {value!r}: "
                    f"{lookup[key]!r} versus {label!r}."
                )

            lookup[key] = label

    return lookup


GENDER_LOOKUP = _build_gender_lookup()


def normalize_gender(value):
    """Map the user's exact response sets to Male, Female, or Other."""
    key = _gender_key(value)

    if key is None:
        return pd.NA

    # Any non-missing response not explicitly listed is Other.
    return GENDER_LOOKUP.get(key, "Other")


def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "gender" not in result.columns:
        raise KeyError("Expected canonical column 'gender'.")

    # Preserve the response before replacing the analytical field.
    if "gender_original" not in result.columns:
        result["gender_original"] = result["gender"].copy()

    result["gender"] = (
        result["gender_original"]
        .map(normalize_gender)
        .astype("string")
    )

    unexpected = set(result["gender"].dropna().unique()) - {
        "Male", "Female", "Other"
    }

    if unexpected:
        raise ValueError(
            f"Unexpected gender categories: {unexpected}"
        )

    return result


def normalize_country(value):
    """Standardise whitespace and Unicode; do not consolidate countries."""
    return normalize_text_value(value)


def clean_countries(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    for column in ("residence_country", "work_country"):
        if column in result.columns:
            result[column] = (
                result[column]
                .map(normalize_country)
                .astype("string")
            )

    return result


def parse_pipe_multiselect(value) -> tuple[str, ...] | pd._libs.missing.NAType:
    """Split genuine pipe-delimited responses, preserving missingness."""
    normalized = normalize_text_value(value)

    if pd.isna(normalized):
        return pd.NA

    parts = [
        normalize_text_value(part)
        for part in str(normalized).split("|")
    ]

    selections = [
        str(part)
        for part in parts
        if not pd.isna(part)
    ]

    return tuple(dict.fromkeys(selections))


def normalize_multiselect_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    columns = [
        "current_diagnosed_conditions",
        "suspected_conditions",
        "professionally_diagnosed_conditions",
        "work_position",
    ]

    for column in columns:
        if column in result.columns:
            result[f"{column}_selections"] = (
                result[column].map(parse_pipe_multiselect)
            )

    return result
