import re
import unicodedata

import numpy as np
import pandas as pd


def normalize_text_value(value):
    """Normalize formatting without changing the semantic response."""
    if pd.isna(value):
        return pd.NA

    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u200b", "").replace("\ufeff", "")
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else pd.NA


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in cleaned.select_dtypes(
        include=["object", "string"]
    ).columns:
        cleaned[column] = cleaned[column].map(
            normalize_text_value
        ).astype("string")

    return cleaned


def clean_age(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = df.copy()

    if "age" not in cleaned.columns:
        raise KeyError("Expected canonical column 'age'.")

    cleaned["age_raw"] = pd.to_numeric(cleaned["age"], errors="coerce")

    invalid_age_mask = (
        cleaned["age_raw"].notna()
        & ~cleaned["age_raw"].between(18, 75)
    )

    age_audit = pd.DataFrame({
        "source_row_id": cleaned.loc[invalid_age_mask, "source_row_id"],
        "age_original": cleaned.loc[invalid_age_mask, "age_raw"],
        "decision": "Set cleaned age to missing; do not infer replacement",
    })

    cleaned["age"] = cleaned["age_raw"].mask(
        invalid_age_mask).astype("Float64")
    cleaned = cleaned.drop(columns="age_raw")

    return cleaned, age_audit


def normalize_gender(value):
    """Map only unambiguous responses to Female, Male, or Other."""
    if pd.isna(value):
        return pd.NA

    raw = normalize_text_value(value)
    if pd.isna(raw):
        return pd.NA

    value = str(raw).strip()
    lower = value.casefold()

    male_values = {
        "male", "m", "man", "cis male", "cis-male", "cisgender male",
        "male.", "male (cis)", "cis male.", "cisgender man",
        "male (trans, ftm)", "ftm", "trans man", "transitioned, f2m",
        "transitioned, m2f", "malr", "dude", "mail", "male/genderqueer",
        "cis male", "cisdude", "cis man", "male 9:1 female, roughly",
        "i'm a man why didn't you make this a drop down question. you "
        "should of asked sex? and i would of answered yes please. seriously how much..."
    }

    female_values = {
        "female", "f", "woman", "fm", "cis female", "cis-female",
        "cis woman", "cis-woman", "cisgender female", "female.",
        "female assigned at birth", "afab", "transgender woman",
        "trans woman", "mtf", "transitioned, m2f", "female/woman",
        "fem", "female-bodied; no feelings about gender",
        "female (props for making this a freeform field, though)",
        "i identify as female.", "female or multi-gender femme",
        "genderfluid (born female)",
    }

    # The source contains an unambiguous malformed "M|" response.
    # Strip the trailing delimiter only for this exact known malformed value.
    if lower == "m|":
        return "Male"

    if lower in male_values:
        return "Male"

    if lower in female_values:
        return "Female"

    # All other self-descriptions and non-binary identities are preserved
    # in the aggregate descriptive category, not forced into a sex binary.
    return "Other"


def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    if "gender" not in cleaned.columns:
        raise KeyError("Expected canonical column 'gender'.")

    cleaned["gender"] = cleaned["gender"].map(
        normalize_gender).astype("string")

    allowed = {"Female", "Male", "Other"}
    unexpected = set(cleaned["gender"].dropna().unique()) - allowed

    if unexpected:
        raise ValueError(f"Unexpected gender categories: {unexpected}")

    return cleaned


def normalize_country(value):
    if pd.isna(value):
        return pd.NA

    value = normalize_text_value(value)
    if pd.isna(value):
        return pd.NA

    return str(value).strip()


def group_rare_countries(
    series: pd.Series,
    minimum_frequency: int = 3,
) -> pd.Series:
    """Group countries with fewer than minimum_frequency observations."""
    normalized = series.map(normalize_country).astype("string")
    counts = normalized.value_counts(dropna=True)
    rare = counts[counts < minimum_frequency].index

    return normalized.mask(normalized.isin(rare), "Other")


def clean_countries(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in ("residence_country", "work_country"):
        if column in cleaned.columns:
            cleaned[column] = group_rare_countries(cleaned[column], 3)

        return cleaned


def parse_pipe_multiselect(value):
    """Return unique, trimmed selections without splitting ordinary text."""
    if pd.isna(value):
        return tuple()

    parts = [
        normalize_text_value(part)
        for part in str(value).split("|")
    ]

    return tuple(dict.fromkeys(
        str(part) for part in parts if not pd.isna(part)
    ))


def normalize_multiselect_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    multi_select_columns = [
        "current_diagnosed_conditions",
        "suspected_conditions",
        "professionally_diagnosed_conditions",
        "work_position",
    ]

    for column in multi_select_columns:
        if column in cleaned.columns:
            cleaned[f"{column}_selections"] = (
                cleaned[column].map(parse_pipe_multiselect)
            )

    return cleaned
