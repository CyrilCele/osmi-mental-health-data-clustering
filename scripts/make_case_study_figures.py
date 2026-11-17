"""Redraw the key project diagnostics at page size for the written case study.

Every figure is drawn from artefacts already produced by Notebooks 01-05 (no model is refitted here), so the
numbers in the case study are the numbers in the repository. Run from the repository root:

    python scripts/make_case_study_figures.py
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MOD = ROOT / "reports" / "results" / "data_modelling"
EVAL = ROOT / "reports" / "results" / "model_evaluation"
SUP = ROOT / "reports" / "results" / "supplementary_dimensionality"
OUT = ROOT / "reports" / "figures" / "case_study"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "savefig.dpi": 300,
    "figure.dpi": 120,
    "axes.titleweight": "bold",
})
FAM = {
    "K-Modes": "#4c72b0",
    "MCA + K-Means": "#dd8452",
    "Agglomerative": "#55a868",
    "HDBSCAN": "#c44e52"
}
# the two clusters of the recommended partition
C0, C1 = "#6a51a3", "#2a9d8f"
NAMES = {0: "Guarded climate (C0)", 1: "Open climate (C1)"}
CLUSTER_COL = {0: C0, 1: C1}


def save(fig, name):
    fig.savefig(
        OUT / name,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)
    print("wrote", name)


def nice(name: str) -> str:
    return name.replace("_", " ")


meta = json.loads(
    (PROC / "osmi_2016_preparation_metadata.json")
    .read_text(encoding="utf-8")
)
DOMAIN = {
    f: d
    for d, fs in meta["feature_domains"].items()
    for f in fs
}
grid = pd.read_csv(MOD / "osmi_2016_model_grid_search.csv")
mmeta = json.loads(
    (MOD / "osmi_2016_modelling_metadata.json")
    .read_text(encoding="utf-8")
)
MIN_SIZE = mmeta["admissibility"]["min_cluster_size"]


# --------------------------------------------------------------------------------------------- 1 pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, body, colour):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.25,rounding_size=1.2",
                fc=colour,
                ec="#333333",
                lw=0.8
            )
        )
        ax.text(
            x + w / 2,
            y + h - 2.2,
            title,
            ha="center",
            va="top",
            fontsize=7.4,
            fontweight="bold"
        )
        ax.text(
            x + w / 2,
            y + h - 6.4,
            body,
            ha="center",
            va="top",
            fontsize=6.3,
            linespacing=1.25
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=9, lw=1, color="#333333"
            )
        )

    w, h, y = 18, 24, 24
    box(
        1, y, w, h,
        "Raw survey",
        "OSMI 2016\n1,433 records\n63 variables\nfree text, nulls,\nmixed formats",
        "#e9ecef"
    )
    box(
        21, y, w, h,
        "01 Understanding",
        "schema and quality\nrouting-aware missing\nvalues, text, ages\ncohort and feature\nscope fixed",
        "#dbe9f6"
    )
    box(
        41, y, w, h,
        "02 Preparation",
        "1,131 respondents\n15 items, 3 domains\nK-Modes input,\nindicator matrix,\nGower matrix",
        "#dcefe3"
    )
    box(
        61, y, w, h,
        "03 Modelling",
        "K-Modes\nMCA + K-Means\nAgglomerative\nHDBSCAN\n(last two on Gower)",
        "#fbe8d3"
    )
    box(
        81, y, w - 1, h,
        "04 Evaluation",
        "profiles, stability,\nagreement, gates,\nscorecard,\nrecommendation",
        "#f3dcdc"
    )

    for x in (19.6, 39.6, 59.6, 79.6):
        arrow(x, y + h / 2, x + 1.6, y + h / 2)

    box(
        41, 2, 38, 15,
        "05 Supplementary analyses",
        "PCA vs MCA vs UMAP (trustworthiness)\nshort pulse-survey reconstruction",
        "#efe6f7"
    )
    box(
        81, 2, 18, 15,
        "HR concept",
        "cluster profiles,\nleverage points,\npulse survey",
        "#fff3c4"
    )
    arrow(88, y - 0.4, 88, 17.6)
    arrow(70, y - 0.4, 62, 17.6)
    arrow(79.6, 9.5, 81, 9.5)
    ax.text(
        1, 9,
        "Each notebook reads only the\nartefacts written by the previous\none; "
        "decisions are logged at the\nend of every stage.",
        fontsize=6.3,
        va="center",
        style="italic",
        color="#444444"
    )
    save(fig, "fig01_pipeline.png")


# --------------------------------------------------------------------------------------------- 2 data quality
def fig_data_quality():
    d = pd.read_csv(ROOT / "data" / "interim" /
                    "osmi_2016_renamed_interim.csv")
    miss = (
        d.isna().mean() * 100).sort_values(ascending=False
                                           ).head(12)[::-1]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.9, 3.0),
        gridspec_kw={"width_ratios": [1.25, 1]}
    )
    axes[0].barh(
        [nice(i) for i in miss.index],
        miss.values,
        color="#8da0cb"
    )

    for y, v in enumerate(miss.values):
        axes[0].text(
            v + 1,
            y,
            f"{v:.0f}%",
            va="center",
            fontsize=6.5
        )

    axes[0].set(
        xlim=(0, 105),
        xlabel="Missing values (% of 1,433 records)",
        title="(a) The twelve most incomplete variables"
    )
    pairs = [
        ("mh_benefits", "Employer MH benefits"),
        ("tech_company", "Tech employer flag"),
        ("medical_coverage_mh", "Coverage (self-employed)")
    ]
    x = np.arange(len(pairs))
    wd = 0.36

    for off, (val, lab, col) in zip(
        (-wd / 2, wd / 2),
        [
            (0, "Employed (n = 1,146)", "#4c72b0"),
            (1, "Self-employed (n = 287)", "#dd8452")
        ]
    ):
        sub = d[d["self_employed"] == val]
        axes[1].bar(
            x + off,
            [sub[p].isna().mean() * 100 for p, _ in pairs],
            wd, label=lab, color=col
        )
    axes[1].set_xticks(x, [textwrap.fill(l, 12) for _, l in pairs])
    axes[1].set(
        ylabel="Missing (%)",
        ylim=(0, 140),
        yticks=range(0, 101, 20),
        title="(b) Missingness follows survey routing"
    )
    axes[1].legend(
        loc="upper center",
        ncol=1,
        frameon=False,
        fontsize=6.5
    )
    save(fig, "fig02_data_quality.png")


# --------------------------------------------------------------------------------------------- 3 cohort funnel
def fig_cohort():
    d = pd.read_csv(ROOT / "data" / "interim" /
                    "osmi_2016_renamed_interim.csv")
    n_all = len(d)
    n_emp = int((d["self_employed"] == 0).sum())
    elig = (
        d["self_employed"] == 0) & ((d["tech_company"] == 1) | (d["tech_role"] == 1)
                                    )
    n_el = int(
        elig.sum())
    n_tc = int(((d["self_employed"] == 0) & (d["tech_company"] == 1)).sum()
               )
    n_role = n_el - n_tc

    assert (n_all, n_emp, n_el) == (1433, 1146, 1131)

    fig, ax = plt.subplots(figsize=(6.9, 2.2))
    labels = [
        "All survey records",
        "Not self-employed",
        "Eligible cohort (tech employer or tech role)"
    ]
    vals = [n_all, n_emp, n_el]
    ax.barh(
        labels[::-1],
        vals[::-1],
        color=["#4c72b0", "#8da0cb", "#c6d0e6"][::-1],
        height=0.6
    )

    for y, v in zip(range(3), vals[::-1]):
        ax.text(
            v + 12, y,
            f"{v:,}",
            va="center",
            fontsize=8,
            fontweight="bold"
        )

    ax.text(
        n_emp + 190, 1.0,
        f"−{n_all - n_emp} self-employed\n(employer questions not asked)",
        fontsize=6.8,
        va="center",
        color="#444444"
    )
    ax.text(
        n_el + 190, 0.0,
        f"−{n_emp - n_el} neither tech employer\nnor tech role",
        fontsize=6.8,
        va="center",
        color="#444444"
    )
    ax.set(
        xlim=(0, 2000),
        xlabel="Respondents",
        title=f"Cohort definition: {n_tc} at a tech employer, plus {n_role} in a tech role elsewhere"
    )
    ax.grid(axis="y", visible=False)
    save(fig, "fig03_cohort.png")


# --------------------------------------------------------------------------------------------- 4 association heatmap
def fig_cramers():
    m = pd.read_csv(PROC / "osmi_2016_cramers_v_matrix.csv", index_col=0)
    order = [f for fs in meta["feature_domains"].values() for f in fs]
    m = m.loc[order, order]
    fig, ax = plt.subplots(figsize=(6.6, 5.7))
    mask = np.triu(np.ones_like(m, dtype=bool), k=0)
    data = np.ma.array(m.to_numpy(), mask=mask)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=0.6)

    ax.set_xticks(
        range(len(order)),
        [nice(o) for o in order],
        rotation=60,
        ha="right",
        fontsize=6.3
    )
    ax.set_yticks(
        range(len(order)),
        [nice(o) for o in order],
        fontsize=6.3
    )

    for i in range(len(order)):
        for j in range(i):
            ax.text(
                j, i,
                f"{m.iat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white" if m.iat[i, j] > 0.35 else "black"
            )

    sizes = [len(v) for v in meta["feature_domains"].values()]
    pos = np.cumsum(sizes)[:-1] - 0.5

    for p in pos:
        ax.axhline(p, color="black", lw=0.8)
        ax.axvline(p, color="black", lw=0.8)

    ax.grid(False)
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.02,
        label="Bias-corrected Cramér's V"
    )
    ax.set_title(
        "Pairwise association between the 15 clustering items (domains separated by lines)"
    )
    save(fig, "fig04_cramers_v.png")


# --------------------------------------------------------------------------------------------- 5 dimensionality
def fig_dimensionality():
    v = pd.read_csv(SUP / "pca_vs_mca_variance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))
    axes[0].bar(
        v["dimension"],
        v["mca_raw_inertia_pct"],
        color="#9bb7d4"
    )
    thr = 100 / 37

    axes[0].axhline(
        thr,
        color="crimson",
        ls="--", lw=1,
        label=f"1/Q rule ({thr:.2f}% of inertia)"
    )
    axes[0].set(
        xlabel="MCA dimension",
        ylabel="Share of total inertia (%)",
        title="(a) MCA: dimensions above the average"
    )
    axes[0].legend()

    axes[1].plot(
        v["dimension"],
        v["pca_cumulative_pct"],
        marker="o",
        ms=3,
        label="PCA, explained variance"
    )
    axes[1].plot(
        v["dimension"],
        v["mca_raw_cumulative_pct"],
        marker="s",
        ms=3,
        label="MCA, raw inertia"
    )
    axes[1].plot(
        v["dimension"],
        v["mca_adjusted_cumulative_pct"],
        marker="^",
        ms=3,
        label="MCA, adjusted inertia"
    )
    axes[1].set(
        xlabel="Number of dimensions",
        ylabel="Cumulative (%)",
        ylim=(0, 102),
        title="(b) Cumulative variance retained"
    )
    axes[1].legend(loc="lower right")
    save(fig, "fig05_dimensionality.png")


# --------------------------------------------------------------------------------------------- 6 2-D views
def fig_two_d():
    e = pd.read_csv(SUP / "two_dimensional_embeddings.csv")
    tc = pd.read_csv(SUP / "trustworthiness_continuity.csv")
    tc = tc[tc["k"] == 15].set_index("view")
    rng = np.random.default_rng(42)
    views = ["PCA", "MCA", "Classical MDS (Gower)", "UMAP (Gower)"]
    letters = "abcd"

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.2))
    for ax, name, let in zip(axes.ravel(), views, letters):
        s = e[e["view"] == name]
        xy = s[["x", "y"]].to_numpy()
        xy = xy + rng.normal(
            scale=0.012, size=xy.shape
        ) * np.ptp(xy, axis=0)

        for c in (0, 1):
            ax.scatter(
                *xy[s["cluster_label"].to_numpy() == c].T,
                s=7, alpha=0.55,
                color=CLUSTER_COL[c], label=NAMES[c], lw=0
            )

        ax.set(
            xticks=[],
            yticks=[],
            title=f"({let}) {name}\nT(15) = {tc.loc[name, 'trustworthiness']:.3f}   C(15) = {tc.loc[name, 'continuity']:.3f}"
        )
        ax.grid(False)

    axes[0, 0].legend(markerscale=2.4, loc="lower left", frameon=True)
    save(fig, "fig06_two_dimensional_views.png")


def fig_trust():
    tc = pd.read_csv(SUP / "trustworthiness_continuity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.1), sharey=False)
    styles = {
        "PCA": "-",
        "MCA": "-",
        "Classical MDS (Gower)": "-",
        "UMAP (Gower)": "-"
    }

    for ax, col, title in zip(
        axes,
        ["trustworthiness", "continuity"],
        ["(a) Trustworthiness T(k)", "(b) Continuity C(k)"]
    ):
        for name, sub in tc.groupby("view", sort=False):
            ax.plot(
                sub["k"],
                sub[col],
                marker="o",
                ms=3,
                ls="--" if name.startswith("MCA (") else styles.get(name, "-"),
                label=name
            )

        ax.set(xlabel="Neighbourhood size k", title=title)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h, l,
        loc="upper center",
        ncol=3, fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0)
    )
    save(fig, "fig07_neighbourhood_preservation.png")


# --------------------------------------------------------------------------------------------- 8 K-Modes
def fig_kmodes():
    g = grid[grid["model_key"] == "K-Modes"]
    fig, axes = plt.subplots(
        2, 2,
        figsize=(6.9, 4.8),
        constrained_layout=True
    )
    k = g["k_requested"]

    axes[0, 0].plot(k, g["cost"] / 1000, marker="o", color=FAM["K-Modes"])
    axes[0, 0].axvline(
        mmeta["kmodes"]["cost_elbow_k"],
        color="crimson",
        ls="--", lw=1,
        label=f"elbow k = {mmeta['kmodes']['cost_elbow_k']}"
    )
    axes[0, 0].set(
        title="(a) Cost (thousands of mismatches)", xlabel="k"
    )
    axes[0, 0].legend()
    axes[0, 1].plot(k, g["mean_pairwise_ari"],
                    marker="o", label="Repeated-run ARI")
    axes[0, 1].plot(k, g["resampling_ari_mean"],
                    marker="s", label="80% resampling ARI")
    axes[0, 1].axhline(0.5, color="grey", ls=":", lw=1)
    axes[0, 1].set(title="(b) Stability", xlabel="k", ylim=(0, 1))
    axes[0, 1].legend()
    axes[1, 0].plot(k, g["gower_silhouette"], marker="o", color="#555555")
    axes[1, 0].set(title="(c) Gower silhouette (supplementary)", xlabel="k")
    axes[1, 1].plot(k, g["smallest_cluster_size"], marker="o", color="#555555")
    axes[1, 1].axhline(MIN_SIZE, color="crimson", ls="--",
                       lw=1, label=f"minimum = {MIN_SIZE}")
    axes[1, 1].set(title="(d) Smallest cluster (respondents)", xlabel="k")
    axes[1, 1].legend()
    save(fig, "fig08_kmodes.png")


def fig_kmeans():
    g = grid[grid["model_key"] == "MCA + K-Means"]
    k = g["k_requested"]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.6), constrained_layout=True)

    for ax, col, title in zip(
        axes.ravel(),
        ["mca_silhouette",
         "davies_bouldin",
         "calinski_harabasz",
         "mean_pairwise_ari"],
        [
            "(a) Silhouette in MCA space",
            "(b) Davies–Bouldin (lower is better)",
            "(c) Calinski–Harabasz",
            "(d) Repeated-start ARI"
        ]
    ):
        ax.plot(k, g[col], marker="o", color=FAM["MCA + K-Means"])
        ax.set(title=title, xlabel="k")

    axes[1, 1].axhline(0.5, color="grey", ls=":", lw=1)
    axes[1, 1].set_ylim(0, 1)
    save(fig, "fig09_mca_kmeans.png")


def fig_agglomerative():
    a = grid[grid["model_key"] == "Agglomerative"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))

    for link, sub in a.groupby("linkage"):
        sub = sub[sub["k_actual"] >= 2]
        lab = f"{link}" + (" (primary)" if link == "average" else "")

        axes[0].plot(
            sub["k_actual"],
            sub["gower_silhouette"],
            marker="o",
            ms=3.5,
            label=lab
        )
        axes[1].plot(
            sub["k_actual"],
            sub["smallest_cluster_size"],
            marker="o",
            ms=3.5,
            label=lab
        )

    axes[1].axhline(
        MIN_SIZE,
        color="crimson",
        ls="--",
        lw=1,
        label=f"minimum = {MIN_SIZE}"
    )
    axes[1].set_yscale("log")
    axes[0].set(
        title="(a) Gower silhouette",
        xlabel="Number of clusters"
    )
    axes[0].legend()
    axes[1].set(
        title="(b) Smallest cluster (log scale)",
        xlabel="Number of clusters"
    )
    axes[1].legend(loc="lower left")
    save(fig, "fig10_agglomerative.png")


def fig_hdbscan():
    h = grid[grid["model_key"] == "HDBSCAN"]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.0, 3.2),
        gridspec_kw={"width_ratios": [1.1, 1]},
        constrained_layout=True
    )

    for method, mk in (("eom", "o"), ("leaf", "^")):
        s = h[
            (
                h["cluster_selection_method"] == method) & (h["n_clusters"] >= 2
                                                            )
        ]
        sc = axes[0].scatter(
            s["coverage"],
            s["dbcv"],
            c=s["n_clusters"],
            cmap="viridis",
            marker=mk,
            s=28,
            ec="black",
            lw=0.3,
            label=method.upper()
        )

    axes[0].axvline(0.5, color="crimson", ls="--", lw=1)
    axes[0].axhline(0, color="grey", ls=":", lw=1)
    axes[0].text(
        0.505,
        axes[0].get_ylim()[1] * 0.85,
        "coverage gate",
        color="crimson",
        fontsize=6.5
    )
    axes[0].set(
        xlim=(0, 1),
        xlabel="Coverage (share of respondents clustered)",
        ylabel="DBCV",
        title="(a) DBCV against coverage"
    )
    axes[0].legend(title="Selection", loc="lower right")

    fig.colorbar(
        sc, ax=axes[0],
        fraction=0.05,
        pad=0.02,
        label="Clusters found (colour)"
    )

    pv = h.pivot_table(
        index="min_samples",
        columns=["cluster_selection_method", "min_cluster_size"],
        values="noise_share"
    )
    im = axes[1].imshow(
        pv.to_numpy(),
        cmap="magma_r",
        vmin=0, vmax=1,
        aspect="auto"
    )

    axes[1].set_xticks(
        range(pv.shape[1]),
        [f"{m[:1].upper()}{int(c)}" for m, c in pv.columns],
        fontsize=6
    )
    axes[1].set_yticks(range(pv.shape[0]), [int(i) for i in pv.index])
    axes[1].set(
        xlabel="E = EOM, L = leaf; number = min_cluster_size",
        ylabel="min_samples",
        title="(b) Noise share"
    )
    axes[1].grid(False)

    fig.colorbar(
        im,
        ax=axes[1],
        fraction=0.045,
        pad=0.02,
        label="Noise share"
    )
    save(fig, "fig11_hdbscan.png")


# --------------------------------------------------------------------------------------------- 12 stability
def fig_stability():
    r = pd.read_csv(EVAL / "osmi_2016_resampling_stability.csv")
    fams = [f for f in FAM if f in r["model_key"].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))

    for ax, col, title, gate in zip(
        axes,
        ["ari", "jaccard_min"],
        ["(a) Resampling ARI (30 subsamples)",
         "(b) Least stable cluster (Jaccard)"],
        [0.6, 0.6]
    ):
        bp = ax.boxplot(
            [r.loc[r["model_key"] == f, col] for f in fams],
            tick_labels=[textwrap.fill(f, 13) for f in fams],
            patch_artist=True, widths=0.55,
            medianprops={"color": "black"}
        )

        for patch, f in zip(bp["boxes"], fams):
            patch.set_facecolor(FAM[f])
            patch.set_alpha(0.85)

        ax.axhline(
            gate,
            color="crimson",
            ls="--", lw=1,
            label="Gate = 0.60"
        )
        ax.set(title=title, ylim=(0, 1.02))

    axes[0].legend(loc="lower left")
    save(fig, "fig12_stability.png")


# --------------------------------------------------------------------------------------------- 13 scorecard
def fig_scorecard():
    s = pd.read_csv(
        EVAL / "osmi_2016_model_scorecard.csv").set_index("model_key")
    s = s[s["representative_declared"]]
    crit = [
        ("resampling_ari_mean", "Resampling\nARI"),
        ("jaccard_worst_cluster", "Worst-cluster\nJaccard"),
        ("gower_silhouette", "Gower\nsilhouette"),
        ("profile_separation", "Profile\nseparation (V)"),
        ("coverage", "Coverage")
    ]

    fig, ax = plt.subplots(figsize=(6.9, 3.0))
    n = len(s)
    w = 0.26
    x = np.arange(len(crit))

    for i, (fam, row) in enumerate(s.iterrows()):
        vals = [row[c] for c, _ in crit]
        bars = ax.bar(
            x + (i - (n - 1) / 2) * w,
            vals, w,
            color=FAM[fam],
            label=f"{fam} ({int(row['gates_passed'])}/5 gates)"
        )

        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.015, f"{v:.2f}",
                ha="center",
                fontsize=6
            )

    ax.set_xticks(x, [l for _, l in crit])
    ax.set(
        ylim=(0, 1.32),
        ylabel="Value (higher is better)",
        title="Common-footing criteria for the family representatives"
    )
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=6.4)
    save(fig, "fig13_scorecard.png")


# --------------------------------------------------------------------------------------------- 14 profile contrast
def fig_profile():
    p = pd.read_csv(
        EVAL / "osmi_2016_cluster_profiles_long.csv"
    )
    p = p[p["model_key"] == "MCA + K-Means"]
    w = p.pivot_table(
        index=["feature", "category"],
        columns="cluster",
        values="share_in_cluster"
    ) * 100
    w["diff"] = (w[1] - w[0]).abs()
    w = w.sort_values(
        "diff", ascending=False
    ).head(16).sort_values("diff")
    labels = []

    for f, c in w.index:
        c = re.sub(
            r"^(Yes|No), (I think it would|I don't think it would|I think they would|I don't think they would)",
            r"\1 (\2)", c
        )
        labels.append(f"{nice(f)}: {c}")

    fig, ax = plt.subplots(figsize=(6.9, 5.3))
    y = np.arange(len(w))

    ax.hlines(y, w[0], w[1], color="#bbbbbb", lw=2, zorder=1)
    ax.scatter(w[0], y, s=34, color=C0, label=f"{NAMES[0]}, n = 607", zorder=2)
    ax.scatter(w[1], y, s=34, color=C1, label=f"{NAMES[1]}, n = 524", zorder=2)
    ax.set_yticks(y, [textwrap.fill(l, 46) for l in labels], fontsize=6.6)
    ax.set(
        xlim=(-3, 103), xlabel="Share of the cluster giving this response (%)",
        title="The sixteen responses that separate the two clusters most"
    )
    ax.legend(loc="lower right")
    save(fig, "fig14_profile_contrast.png")


# --------------------------------------------------------------------------------------------- 15 domains + context
def fig_domains_context():
    k = pd.read_csv(PROC / "osmi_2016_kmodes_input.csv", dtype="string")
    lab = pd.read_csv(
        EVAL / "osmi_2016_recommended_cluster_assignments.csv"
    ).sort_values("model_row_index")["cluster_label"].to_numpy()
    pat = r"don'?t know|do not know|not sure|unsure|maybe|neither|unknown"
    unc = pd.DataFrame(
        {
            f: k[f].str.contains(
                pat, case=False, regex=True, na=False
            )
            .astype("boolean")
            .fillna(False)
            .to_numpy(bool)
            for f in k.columns
        }
    ).astype(float)
    dom = pd.DataFrame(
        {
            d: unc[fs].mean(axis=1)
            .groupby(lab)
            .mean() * 100
            for d, fs in meta["feature_domains"].items()
        }
    )
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.2, 3.1),
        gridspec_kw={"width_ratios": [1.1, 1]},
        constrained_layout=True
    )
    x = np.arange(len(dom.columns))
    wd = 0.36

    for off, c in zip((-wd / 2, wd / 2), (0, 1)):
        axes[0].bar(
            x + off,
            dom.loc[c].to_numpy(),
            wd,
            color=CLUSTER_COL[c],
            label=NAMES[c]
        )

        for xi, v in zip(x + off, dom.loc[c]):
            axes[0].text(
                xi,
                v + 0.8,
                f"{v:.0f}%",
                ha="center",
                fontsize=6.5
            )

    axes[0].set_xticks(x, [textwrap.fill(nice(c), 14) for c in dom.columns])
    axes[0].set(
        ylabel="'Unsure' responses (%)",
        ylim=(0, 62),
        title="(a) Uncertain answers by domain"
    )
    axes[0].legend(loc="upper right")

    ctx = pd.read_csv(
        EVAL / "osmi_2016_recommended_context_profile.csv"
    ).sort_values("effect_size")
    ctx["variable"] = ctx["variable"].replace({"gender_profile": "gender"})
    cols = [
        "#9aa5b1"
        if m == "negligible"
        else "#e9a23b"
        for m in ctx["magnitude"]
    ]
    axes[1].barh([nice(v) for v in ctx["variable"]],
                 ctx["effect_size"], color=cols)

    for yv, (e, m) in enumerate(zip(ctx["effect_size"], ctx["metric"])):
        axes[1].text(e + 0.004, yv, f"{e:.2f}", va="center", fontsize=6.5)

    axes[1].axvline(0.10, color="grey", ls=":", lw=1)
    axes[1].set(
        xlim=(0, 0.23),
        xlabel="Effect size (V; epsilon-squared for age)",
        title="(b) Who is in each cluster?"
    )
    save(fig, "fig15_domains_context.png")


# --------------------------------------------------------------------------------------------- 16 pulse survey
def fig_pulse():
    p = pd.read_csv(SUP / "pulse_survey_reconstruction.csv")
    base = max(
        np.bincount(
            pd.read_csv(
                EVAL / "osmi_2016_recommended_cluster_assignments.csv"
            )["cluster_label"]
        )
    ) / 1131

    fig, ax = plt.subplots(figsize=(6.9, 2.9))
    ax.errorbar(
        p["items"],
        p["accuracy"],
        yerr=p["accuracy_sd"],
        marker="o",
        capsize=3,
        label="Accuracy (±1 SD across folds)"
    )
    ax.plot(
        p["items"],
        p["cohen_kappa"],
        marker="s",
        label="Cohen's kappa"
    )
    ax.axhline(
        base,
        color="grey",
        ls=":",
        label=f"Majority-class baseline ({base:.2f})"
    )
    ax.set_xticks(p["items"])
    ax.set(
        ylim=(0, 1.02),
        xlabel="Number of items asked (ranked by association with the partition)",
        ylabel="Reconstruction of the 15-item partition",
        title="How short can the pulse survey be?"
    )
    ax.legend(loc="lower right")
    save(fig, "fig16_pulse_survey.png")


if __name__ == "__main__":
    for f in (
        fig_pipeline,
        fig_data_quality,
        fig_cohort,
        fig_cramers,
        fig_dimensionality,
        fig_two_d, fig_trust,
        fig_kmodes,
        fig_kmeans,
        fig_agglomerative,
        fig_hdbscan,
        fig_stability,
        fig_scorecard,
        fig_profile,
        fig_domains_context,
        fig_pulse
    ):
        f()
