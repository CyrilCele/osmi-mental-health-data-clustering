# OSMI 2016 — Workplace Support and Disclosure Climate Clustering

## Project purpose

This project follows the CRISP-DM lifecycle to investigate whether the OSMI 2016 Mental Health in Tech survey contains defensible, aggregate profiles of workplace support, psychological safety, and disclosure climate.

The project is exploratory. It does not diagnose mental-health conditions, infer causality, rank individuals, or support employee-level decisions. The source is a historical, voluntary, self-reported survey; findings require independent replication before informing current organizational policy.

## Notebook sequence

The six notebooks cover all CRISP-DM phases. The final Deployment phase is a responsible communication and future-operationalization recommendation only; it does not implement a production system.

Run the notebooks in order from the repository root or open Jupyter from `notebooks/`:

1. `01_business_understanding.ipynb` — business objective, scope, decision log, ethical guardrails.
2. `02_data_understanding.ipynb` — source schema, distributions, data quality, conditional survey logic.
3. `03_data_preparation.ipynb` — eligible cohort, feature definition, ordinal handling, domain-balanced Gower-style dissimilarities, lineage and persisted outputs.
4. `04_clustering_models.ipynb` — PAM/K-medoids, HDBSCAN, average-linkage agglomerative clustering.
5. `05_evaluation.ipynb` — subsampling stability, cross-method agreement, internal diagnostics, and candidate screening.
6. `06_deployment_and_communication.ipynb` — aggregate profiles, visualization, responsible-use decision, and communication recommendation; no production deployment.

The business-understanding and data-understanding notebooks are preserved from the project materials. The preparation and modelling stages create fresh artifacts from the supplied raw CSV. We do not reuse a precomputed distance matrix unless it can be verified against the exact feature matrix and row mapping.

## Environment setup

Python 3.11 or 3.12 is recommended for broad scientific-package compatibility. From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch Jupyter with the project root as the working directory:

```bash
jupyter lab
```

For notebook cells that resolve paths relative to `notebooks/`, open the notebook from the `notebooks/` directory or let the path-resolution logic select the project root. The expected source file is `data/raw/mental-heath-in-tech-2016_20161114.csv`.

## Modelling design

- **Cohort:** non-self-employed respondents who either report a primarily technology employer or a technology/IT role; expected size is 1,131 for the supplied source.
- **Features:** 15 predeclared workplace-support and disclosure-climate survey items, organized into three equally weighted domains.
- **Exclusions:** personal diagnosis/history, treatment, condition descriptions, demographic identifiers, country/state, and free-text explanations are not used to form clusters.
- **Dissimilarity:** nominal responses use 0/1 mismatch; leave-request difficulty uses normalized ordinal rank distance, while “I don't know” remains a separate unordered response. Within-domain dissimilarities are averaged, then domain averages receive equal weight.
- **PAM:** uses the original precomputed dissimilarity matrix and actual respondent medoids.
- **HDBSCAN:** uses the same precomputed dissimilarities and reports cluster coverage/noise explicitly.
- **Agglomerative:** average linkage operates directly on the precomputed matrix. Ward linkage is not used because the matrix is not Euclidean feature space.
- **Evaluation:** silhouette, cluster-size diagnostics, HDBSCAN coverage/noise, subsampling ARI stability, cross-method agreement, medoid/cluster profiles, and 2D MDS visualization. MDS is visualization only and does not replace the fitting matrix.

## Generated artifacts

`03_data_preparation.ipynb` writes:

- `data/processed/model_feature_matrix.csv`
- `data/processed/domain_balanced_dissimilarity.npy`
- `data/processed/eligible_cohort_row_mapping.csv`
- `data/processed/preparation_metadata.json`

The feature-level data dictionary and cohort flow are saved at `reports/tables/feature_data_dictionary.csv` and `reports/tables/cohort_flow.csv`. The modelling and evaluation notebooks write candidate metrics, aligned cluster assignments, stability tables, aggregate profiles, and plots under `reports/`.

## Reproducibility and interpretation

The notebooks use a fixed seed where the method supports randomness. Candidate results are persisted separately from the source dataset. A candidate is not accepted merely because it has the largest silhouette. The evaluation notebook applies explicit size, stability, and separation gates; if no candidate passes, the correct outcome is to report that no sufficiently robust partition was found.

Cluster numbers are arbitrary labels, not ranks. All interpretations must remain aggregate, non-clinical, non-causal, and historically bounded. Do not use individual assignments for screening, diagnosis, discipline, hiring, promotion, or any other employment decision.

## Method references

- Hennig, C. & Liao, T. F. (2013). *How to find an appropriate clustering for mixed-type variables with application to socio-economic stratification.* Journal of the Royal Statistical Society: Series C, 62(3), 309–369.
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis.* Wiley.
- Schubert, E. & Rousseeuw, P. J. (2019/2021). FastPAM/PAM swap optimization literature; the FastPAM1 implementation used here is documented by the `kmedoids` package.
- Campello, R. J. G. B., Moulavi, D. & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates.* PAKDD.
- Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.* Journal of Computational and Applied Mathematics, 20, 53–65.
