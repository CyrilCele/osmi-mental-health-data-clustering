"""Source ingestion and provenance utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum of a file."""
    file_path = Path(path)

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_source(
    path: str | Path,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load the original CSV and return its provenance metadata."""
    source_path = Path(path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source dataset does not exist: {source_path}"
        )

    if source_path.suffix.lower() != ".csv":
        raise ValueError("The source dataset must be a CSV file.")

    df = pd.read_csv(
        source_path,
        low_memory=False,
        encoding="utf-8-sig",
    )

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            "Duplicate source headers were detected after CSV parsing: "
            f"{duplicates}"
        )

    if expected_shape is not None and df.shape != expected_shape:
        raise ValueError(
            f"Unexpected source shape {df.shape}; "
            f"expected {expected_shape}."
        )

    metadata = {
        "source_path": str(source_path.resolve()),
        "source_filename": source_path.name,
        "source_sha256": sha256_file(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_rows": int(df.shape[0]),
        "source_columns": int(df.shape[1]),
        "source_shape": tuple(df.shape),
    }

    return df, metadata
