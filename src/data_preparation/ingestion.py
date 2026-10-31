import hashlib
from pathlib import Path

import pandas as pd

from .config import EXPECTED_RAW_SHAPE, SOURCE_PATH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), "b"):
            digest.update(chunk)

    return digest.hexdigest()


def load_source(path: Path = SOURCE_PATH) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"OSMI source CSV not found: {path}. "
            "Place the original CSV in data/raw/."
        )

    raw = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8",
        quotechar='"',
        doublequote=True,
        skipinitialspace=False,
    )

    if raw.shape != EXPECTED_RAW_SHAPE:
        raise ValueError(
            f"Unexpected source shape {raw.shape}; "
            f"expected {EXPECTED_RAW_SHAPE}. "
            "Verify that the correct, unmodified OSMI 2016 file is loaded."
        )

    metadata = {
        "source_path": str(path),
        "source_filename": path.name,
        "source_sha256": sha256_file(path),
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_columns": int(raw.columns.duplicated().sum()),
    }

    return raw, metadata
