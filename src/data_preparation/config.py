from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROCESSED_DIR / "reports" / "figures"

SOURCE_FILENAME = "mental-heath-in-tech-2016_20161114.csv"
SOURCE_PATH = RAW_DIR / SOURCE_FILENAME

RANDOM_STATE = 42
EXPECTED_RAW_SHAPE = (1433, 63)

# Data Understanding identified these exact invalid observations.
INVALID_AGES = {3, 15, 17, 99, 323}

# The source data is immutable. All processing operates on copies.
