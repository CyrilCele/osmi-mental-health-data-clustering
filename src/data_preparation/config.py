from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROCESSED_DIR / "reports" / "figures"

SOURCE_FILENAME = "mental-heath-in-tech-2016_20161114.csv"
SOURCE_PATH = RAW_DIR / SOURCE_FILENAME

RANDOM_STATE = 42
EXPECTED_RAW_SHAPE = (1433, 63)

# Numerical validity boundaries are used for data-quality flags,
# not for deleting respondents
MIN_PLAUSIBLE_AGE = 18
MAX_PLAUSIBLE_AGE = 75

# Countries with fewer than this many respondents are grouped as Other.
COUNTRY_MIN_FREQUENCY = 3

# The soure data is immutable. All processing operates on copies.
