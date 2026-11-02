"""Central configuration for the OSMI clustering experiments."""

from pathlib import Path

from src.data_preparation.config import (
    DOMAIN_FEATURES,
    MODEL_FEATURES,
    RANDOM_STATE,
)

from src.data_preparation.config import PROCESSED_DIR as PREPARATION_OUTPUT_DIR


# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PREPARATION_OUTPUT_DIR

FEATURE_MATRIX_PATH = DATA_DIR / "model_feature_matrix.csv"
DISTANCE_MATRIX_PATH = DATA_DIR / "domain_balanced_dissimilarity.npy"
ROW_MAPPING_PATH = DATA_DIR / "eligible_cohort_row_mapping.csv"

# Supplementary data: never used for fitting.
PROFILE_DESCRIPTORS_PATH = DATA_DIR / "profile_descriptors.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures" / "clustering"
RESULTS_DIR = REPORTS_DIR / "clustering"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Preparation contract

EXPECTED_N_FEATURES = len(MODEL_FEATURES)
EXPECTED_N_RESPONDENTS = 1131

ROW_POSITION_COLUMN = "distance_matrix_row"
SOURCE_ROW_COLUMN = "source_row_id"

FEATURES = tuple(MODEL_FEATURES)
DOMAINS = DOMAIN_FEATURES


# Reproducibility

SEED = RANDOM_STATE

K_CANDIDATES = (2, 3, 4, 5, 6)

# K-Modes repeated initialisation.
KMODES_N_INIT = 20
KMODES_MAX_ITER = 300
KMODES_SEEDS = tuple(range(SEED, SEED + 20))

# K-Means.
KMEANS_N_INIT = 50
KMEANS_MAX_ITER = 500
KMEANS_TOL = 1e-4

# MCA.
MCA_COMPONENTS = 10
MCA_ITERATIONS = 10

# Agglomerative.
AGGLOMERATIVE_LINKAGES = ("average", "complete")

# HDBSCAN / UMAP.
UMAP_COMPONENTS = 5
UMAP_N_NEIGHBORS = (15, 30)
UMAP_MIN_DIST = (0.0, 0.1)
UMAP_METRIC = "precomputed"

HDBSCAN_MIN_CLUSTER_SIZE = (15, 25, 40)
HDBSCAN_MIN_SAMPLES = (5, 10, 15)
HDBSCAN_SELECTION_METHODS = ("eom", "leaf")

# HDBSCAN diagnostic threshold, not a universal acceptance criterion.
HDBSCAN_NOISE_DIAGNOSTIC_TARGET = 0.30

# Evaluation.
SILHOUETTE_SAMPLE_SIZE = 1000
SILHOUETTE_RANDOM_STATE = SEED

# Dendrogram.
DENDROGRAM_TRUNCATE_COUNT = 40

# UMAP and embedding reproducibility.
UMAP_RANDOM_STATE = SEED

# Numerical tolerance.
MATRIX_TOLERANCE = 1e-10