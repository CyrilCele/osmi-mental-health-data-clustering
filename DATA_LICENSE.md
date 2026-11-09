# Data licence and attribution

## Source data

**OSMI Mental Health in Tech Survey 2016**, created by **Open Sourcing Mental Illness (OSMI)**.

- Source: <https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016>
- Licence: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**
  <https://creativecommons.org/licenses/by-sa/4.0/>

Suggested attribution: *Open Sourcing Mental Illness (2016). OSMI Mental Health in Tech Survey 2016 [Data set]. Kaggle. https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016. Licensed under CC BY-SA 4.0.*

OSMI does not endorse this project, and the project does not imply any endorsement by OSMI.

## What in this repository is covered

| Path | Status | Licence |
|---|---|---|
| `data/raw/` | Unmodified copy of the original survey file | CC BY-SA 4.0 (original licence retained) |
| `data/interim/` | Same observations with **standardised column names only** | CC BY-SA 4.0 |
| `data/processed/` | **Adaptations**: eligible-cohort selection, text normalisation, descriptive gender harmonisation, categorical/indicator representations, Gower dissimilarity matrix, association tables | CC BY-SA 4.0 |
| `reports/results/` | Tables **derived from** the data (cluster labels, profiles, scorecards) | CC BY-SA 4.0 |
| `reports/figures/` | Figures displaying the data or derived results | CC BY-SA 4.0 |

## Changes made to the original data

The changes are documented step by step in `notebooks/01_data_understanding.ipynb` and `notebooks/02_data_preparation.ipynb`. In summary:

1. Column names were standardised (no response values changed) to create the interim file.
2. The analysis cohort was restricted to non-self-employed respondents who report a technology employer or a technology/IT role (1,131 of 1,433 records).
3. Text formatting was normalised (whitespace, empty strings to missing values); a coarse descriptive gender harmonisation was added alongside the original response.
4. Model-ready representations were derived: categorical labels, a binary indicator matrix and a domain-balanced Gower dissimilarity matrix.
5. Notebooks 03 and 04 add cluster assignments, profiles and evaluation tables.

## Share-alike

Adaptations of the data contained in this repository are shared under the **same licence (CC BY-SA 4.0)**. Anyone reusing the data or derived datasets must give appropriate credit, indicate that changes were made, and distribute their own adaptations under CC BY-SA 4.0 or a compatible licence.

## Code

The analysis code is licensed separately under the MIT licence; see `LICENSE`.

## Privacy and responsible use

The analysis uses only the 15 predeclared workplace-climate items for clustering and joins a small set of contextual variables afterwards for description. Free-text responses and detailed condition fields are not used. Respondents should not be re-identified, and nothing in this repository should be used to attempt it. Cluster assignments describe aggregate response patterns and must not be used to make decisions about individuals.
