# 📊 OSMI Mental Health Data Clustering

An unsupervised machine learning project focusing on the segmentation and profile discovery of mental health patterns within the global tech community. This project utilizes data collected from the **Open Sourcing Mental Illness (OSMI)** annual Mental Health in Tech Surveys to identify distinct clusters of employees based on their workplace experiences, demographics, attitudes, and available support systems.

## 🎯 Project Objectives
* **Pattern Discovery:** Uncover latent segments of tech workers sharing common mental health experiences and professional challenges.
* **Workplace Profiling:** Analyze how employer behaviors (benefits awareness, wellness discussions, remote flexibility) group across different corporate sizes.
* **Anonymized Clustering:** Execute robust unsupervised pipelines (K-Means, Hierarchical Clustering) on high-dimensional category data using silhouette analysis for optimal cluster selection.

## 📁 Repository Structure
* `data/`: Placeholder directory for raw and processed OSMI survey data CSV files.
* `notebooks/`: Exploratory workflows, data understanding phase (`data_understanding.ipynb`), and experimental clustering structures.
* `src/`: Reusable production-grade source code packaging data imputation, encoding transformer pipelines, and evaluation metrics.

## 🛠️ Tech Stack & Environment
This project is built using modern Python isolation utilities:
* **Environment Management:** Powered natively by `uv` for fast, reproducible lockfiles.
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `ipykernel`.

---
*Disclaimer: This repository is intended strictly for data exploration, statistical analysis, and machine learning study. It does not provide medical diagnostic insight.*
