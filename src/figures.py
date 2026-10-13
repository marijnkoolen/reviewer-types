"""Generate report figures from the gold-set analysis results (§2-5). Static
PNGs for the markdown/HTML report -- palette and mark choices follow the
dataviz skill's method: one hue for magnitude (sequential), two fixed-order
categorical hues only where there's a real identity distinction (early/late
position), a blue<->red diverging pair for signed association measures
(PMI, z-scores), never a rainbow across many bars.

Usage: python src/figures.py
Writes PNGs to results/figures/
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import LABEL_TO_CODE, MAIN_LABELS, SPARSE_GOLD_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(exist_ok=True)

# --- palette (validated default, see dataviz skill references/palette.md) ---
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"   # sequential step 200, for sub-category bars
ORANGE = "#eb6834"
GRAY_MUTED = "#898781"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DIVERGING_NEG = "#2a78d6"   # blue = less than chance
DIVERGING_POS = "#e34948"   # red  = more than chance
DIVERGING_MID = "#f0efec"

# fixed-order 8-slot categorical palette (validated default, light mode)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "grid.color": GRID,
    "axes.grid": True,
    "axes.axisbelow": True,
    "font.size": 10,
})

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_red_diverging", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS]
)


def short(label):
    return LABEL_TO_CODE.get(label, label)


def fig_occurrence():
    df = pd.read_csv(RESULTS / "01_occurrence/gold_occurrence.csv").sort_values("prevalence")
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = [BLUE if lvl == "main" else BLUE_LIGHT for lvl in df["level"]]
    y = np.arange(len(df))
    ax.barh(y, df["prevalence"] * 100, color=colors, height=0.7, zorder=3)
    err_lo = (df["prevalence"] - df["prevalence_ci_lo"]) * 100
    err_hi = (df["prevalence_ci_hi"] - df["prevalence"]) * 100
    ax.errorbar(df["prevalence"] * 100, y, xerr=[err_lo, err_hi], fmt="none",
                ecolor=INK_SECONDARY, elinewidth=1, capsize=2, zorder=4)
    labels = [l + (" *sparse*" if s else "") for l, s in zip(df["label"], df["sparse_gold_flag"])]
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlabel("% of reviews containing >=1 sentence of this category")
    ax.set_title("Prevalence of each (sub)category across 1,152 gold reviews\n"
                  "(95% bootstrap CI; darker bars = main category, lighter = sub-category)",
                  fontsize=10.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    legend_handles = [Patch(facecolor=BLUE, label="main category"),
                       Patch(facecolor=BLUE_LIGHT, label="sub-category")]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "occurrence_prevalence.png", dpi=160)
    plt.close(fig)


NAN_COLOR = "#fff3d6"  # pale yellow -- deliberately off the blue-gray-red scale so
                        # "excluded pair / diagonal" never reads as "PMI about 0"


def _heatmap(matrix, labels, out_path, title, value_label, vlim=None):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    vmax = vlim if vlim else np.nanmax(np.abs(matrix))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = DIVERGING_CMAP.copy()
    cmap.set_bad(NAN_COLOR)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
    ax.set_xticks(range(n), [short(l) for l in labels], rotation=90, fontsize=7.5)
    ax.set_yticks(range(n), [short(l) for l in labels], fontsize=7.5)
    ax.set_title(title, fontsize=10.5, loc="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(value_label, fontsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_cooccurrence():
    sent = pd.read_csv(RESULTS / "02_cooccurrence/gold_cooccurrence_sentence.csv")
    labels = sorted(set(sent["label_a"]) | set(sent["label_b"]))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)

    pmi_mat = np.full((n, n), np.nan)
    for _, row in sent.iterrows():
        i, j = idx[row["label_a"]], idx[row["label_b"]]
        pmi_mat[i, j] = pmi_mat[j, i] = row["pmi"]
    _heatmap(pmi_mat, labels, FIG_DIR / "cooccurrence_sentence_pmi.png",
              "Sentence-level co-occurrence (PMI, parent-child pairs excluded)\n"
              "red = co-occur more than chance, blue = less, pale yellow = excluded/never",
              "PMI (log2)", vlim=3.5)

    rev = pd.read_csv(RESULTS / "02_cooccurrence/gold_cooccurrence_review.csv")
    z_mat = np.full((n, n), np.nan)
    for _, row in rev.iterrows():
        i, j = idx[row["label_a"]], idx[row["label_b"]]
        z_mat[i, j] = z_mat[j, i] = row["z_score"]
    _heatmap(z_mat, labels, FIG_DIR / "cooccurrence_review_zscore.png",
              "Review-level co-occurrence vs. length-stratified permutation null\n"
              "red = co-occur across a review more than length alone predicts, blue = less",
              "z-score", vlim=7)


def fig_contiguity():
    df = pd.read_csv(RESULTS / "03_contiguity/gold_contiguity_label_summary.csv").dropna(subset=["mean_clumpiness"])
    df = df.sort_values("mean_clumpiness")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    ax.barh(y, df["mean_clumpiness"], color=BLUE, height=0.7, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("mean clumpiness index (0 = maximally scattered, 1 = single contiguous block)")
    ax.set_title("Contiguity: do a category's sentences cluster together within a review?\n"
                  "All labels shown test significantly more contiguous than random placement (p<0.05)",
                  fontsize=10, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "contiguity_clumpiness.png", dpi=160)
    plt.close(fig)


def fig_order_position():
    df = pd.read_csv(RESULTS / "04_order/gold_order_position.csv").sort_values("mean_rel_position")
    color_map = {"early": BLUE, "late": ORANGE, "none": GRAY_MUTED}
    colors = df["direction"].map(color_map)
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(df))
    err_lo = df["mean_rel_position"] - df["ci_lo"]
    err_hi = df["ci_hi"] - df["mean_rel_position"]
    ax.errorbar(df["mean_rel_position"], y, xerr=[err_lo, err_hi], fmt="o",
                color=INK_SECONDARY, ecolor=INK_SECONDARY, elinewidth=1, capsize=2,
                markersize=0, zorder=3)
    ax.scatter(df["mean_rel_position"], y, c=colors, s=60, zorder=4, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean relative position in review (0 = first sentence, 1 = last)")
    ax.set_title("Where in the review each category tends to appear\n"
                  "(95% bootstrap CI; dashed line = review center / no positional preference)",
                  fontsize=10.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    legend_handles = [Patch(facecolor=BLUE, label="significantly early"),
                       Patch(facecolor=ORANGE, label="significantly late"),
                       Patch(facecolor=GRAY_MUTED, label="no significant preference")]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "order_position.png", dpi=160)
    plt.close(fig)


def fig_order_transitions():
    wide = pd.read_csv(RESULTS / "04_order/gold_order_transition_matrix.csv", index_col=0)
    labels = list(wide.index)
    counts = wide.to_numpy(dtype=float)
    total = counts.sum()
    from_marginal = counts.sum(axis=1) / total
    to_marginal = counts.sum(axis=0) / total
    n = len(labels)
    pmi = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            p_obs = counts[i, j] / total
            p_exp = from_marginal[i] * to_marginal[j]
            if p_obs > 0 and p_exp > 0:
                pmi[i, j] = np.log2(p_obs / p_exp)
    _heatmap(pmi, labels, FIG_DIR / "order_transition_pmi.png",
              "What kind of sentence tends to follow what (main categories only)\n"
              "row = category of sentence t, column = category of sentence t+1; "
              "red = more than base rates predict",
              "PMI (log2)", vlim=2.5)


def fig_typology_k_selection():
    df = pd.read_csv(RESULTS / "05_typology/gold_typology_k_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, clustering, color in zip(axes, ["compositional", "sequence"], [BLUE, ORANGE]):
        sub = df[df["clustering"] == clustering]
        ax.plot(sub["k"], sub["silhouette"], color=color, marker="o", markersize=4,
                linewidth=1.6, label="silhouette")
        ax.plot(sub["k"], sub["bootstrap_ari_stability"], color=color, marker="o",
                markersize=4, linewidth=1.6, linestyle="--", label="bootstrap stability (ARI)")
        ax.set_title(clustering, fontsize=10.5, loc="left")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="center right")
    axes[0].set_ylabel("score")
    fig.suptitle("Cluster count (k) selection: silhouette vs. bootstrap stability", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "typology_k_selection.png", dpi=160)
    plt.close(fig)


def fig_typology_cluster_profile():
    profile = pd.read_csv(RESULTS / "05_typology/gold_typology_cluster_profile.csv")
    main_used = [m for m in MAIN_LABELS if m != "None"]
    cols = [f"mean_prop__{l}" for l in main_used]
    matrix = profile[cols].to_numpy()
    n_clusters = len(profile)

    fig, ax = plt.subplots(figsize=(8, 1 + 0.55 * n_clusters))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(main_used)), main_used, rotation=30, ha="right", fontsize=9)
    ylabels = [f"cluster {c}  (n={n}, {p:.0f}%)" for c, n, p in
               zip(profile["cluster"], profile["n_reviews"], profile["pct_of_corpus"])]
    ax.set_yticks(range(n_clusters), ylabels, fontsize=9)
    for i in range(n_clusters):
        for j in range(len(main_used)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if val > matrix.max() * 0.6 else INK_PRIMARY)
    ax.set_title("Compositional cluster profile: mean sentence proportion per main category",
                  fontsize=10.5, loc="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("mean proportion of sentences")
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "typology_cluster_profile.png", dpi=160)
    plt.close(fig)


def fig_typology_pca_scatter():
    clusters = pd.read_csv(RESULTS / "05_typology/gold_typology_clusters.csv")
    features = pd.read_csv(RESULTS / "00_data_prep/gold_review_features.csv")
    main_used = [m for m in MAIN_LABELS if m != "None"]
    comp = features[[f"prop__{l}" for l in main_used]].copy()
    comp["prop__unlabeled"] = features["n_unlabeled_sentences"] / features["num_sentences"]
    X = StandardScaler().fit_transform(comp.to_numpy())
    coords = PCA(n_components=2, random_state=0).fit_transform(X)
    var_explained = PCA(n_components=2, random_state=0).fit(X).explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 7))
    n_clusters = clusters["cluster_composition"].nunique()
    for c in range(n_clusters):
        mask = clusters["cluster_composition"] == c
        color = CATEGORICAL[c % len(CATEGORICAL)]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, color=color, alpha=0.65,
                   linewidth=0, label=f"cluster {c}")
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.annotate(str(c), (cx, cy), fontsize=11, fontweight="bold", color="white",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor=color, edgecolor="white", linewidth=1.2))
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.0f}% of variance)")
    ax.set_title("Reviews in composition space, colored by compositional cluster\n"
                  "(numbered circles mark each cluster's centroid -- color is a secondary cue, not the only one)",
                  fontsize=10, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "typology_pca_scatter.png", dpi=160)
    plt.close(fig)


def _sequence_cluster_heatmap(csv_name, out_name, title, n_reviews_label="n"):
    profile = pd.read_csv(RESULTS / csv_name)
    main_used = [m for m in MAIN_LABELS if m != "None"]
    cols = [f"mean_seqpos__{l}" for l in main_used]
    matrix = profile[cols].to_numpy()
    n_clusters = len(profile)

    fig, ax = plt.subplots(figsize=(8, 1 + 0.55 * n_clusters))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = DIVERGING_CMAP.copy()
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(main_used)), main_used, rotation=30, ha="right", fontsize=9)
    ylabels = [f"seq cluster {c}  ({n_reviews_label}={n:,}, {p:.0f}%)" for c, n, p in
               zip(profile["cluster_sequence"], profile["n_reviews"], profile["pct_of_corpus"])]
    ax.set_yticks(range(n_clusters), ylabels, fontsize=9)
    for i in range(n_clusters):
        for j in range(len(main_used)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val - 0.5) > 0.3 else INK_PRIMARY)
    ax.set_title(title, fontsize=10, loc="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("mean rel. position\n(0=first, 1=last, 0.5=none)", fontsize=8)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_gold_sequence_cluster_profile():
    _sequence_cluster_heatmap(
        "11_sequence_cluster_profile/gold_typology_sequence_cluster_profile.csv", "gold_sequence_cluster_profile.png",
        "Sequence cluster profile, gold: mean relative position of each main category's own sentences")


def fig_silver_sequence_cluster_profile():
    _sequence_cluster_heatmap(
        "11_sequence_cluster_profile/silver_typology_sequence_cluster_profile.csv", "silver_sequence_cluster_profile.png",
        "Sequence cluster profile, silver: mean relative position of each main category's own sentences")


def fig_typology_crosstab():
    crosstab = pd.read_csv(RESULTS / "05_typology/gold_typology_crosstab.csv", index_col=0)
    counts = crosstab.to_numpy()
    row_pct = counts / counts.sum(axis=1, keepdims=True) * 100
    marginal_pct = counts.sum(axis=0) / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 6.5))
    im = ax.imshow(row_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(counts.shape[1]),
                  [f"seq {c}\n({marginal_pct[i]:.0f}% overall)" for i, c in enumerate(crosstab.columns)],
                  fontsize=8.5)
    ax.set_yticks(range(counts.shape[0]), [f"comp {c}" for c in crosstab.index], fontsize=9)
    for i in range(row_pct.shape[0]):
        for j in range(row_pct.shape[1]):
            val = row_pct[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if val > 60 else INK_PRIMARY)
    ax.set_title("Compositional vs. sequence-based clusters (row %)\n"
                  "each row sums to 100%; compare to each column's overall share below its label",
                  fontsize=10, loc="left")
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "typology_crosstab.png", dpi=160)
    plt.close(fig)


def _prevalence_heatmap(csv_name, group_col_prefix, out_name, title, group_order=None):
    df = pd.read_csv(RESULTS / csv_name)
    main_used = [m for m in MAIN_LABELS if m != "None"]
    df = df[df["label"].isin(main_used)].set_index("label").loc[main_used]
    prev_cols = [c for c in df.columns if c.startswith(group_col_prefix)]
    groups = [c[len(group_col_prefix):] for c in prev_cols]
    if group_order:
        order_idx = [groups.index(g) for g in group_order]
        prev_cols = [prev_cols[i] for i in order_idx]
        groups = group_order
    matrix = df[prev_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(1.3 * len(groups) + 2.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(main_used)), main_used, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if val > 0.6 else INK_PRIMARY)
    ax.set_title(title, fontsize=10.5, loc="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("prevalence")
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160)
    plt.close(fig)


def fig_genre_prevalence():
    order = ["Literary_fiction", "Literary_thriller", "Suspense", "Other fiction",
             "Children_fiction", "Fantasy_fiction", "Young_adult", "Historical_fiction",
             "Regional_fiction", "Romance"]
    _prevalence_heatmap("06_genre_platform/gold_genre_prevalence.csv", "prevalence__", "genre_prevalence_heatmap.png",
                         "Main-category prevalence by genre", group_order=order)


def fig_platform_prevalence():
    _prevalence_heatmap("06_genre_platform/gold_platform_prevalence.csv", "prevalence__", "platform_prevalence_heatmap.png",
                         "Main-category prevalence by platform")


def fig_platform_length_confound():
    df = pd.read_csv(RESULTS / "06_genre_platform/gold_platform_length_stratified_check.csv")
    terciles = ["short", "medium", "long"]
    platforms = sorted(df["source_id"].unique())
    x = np.arange(len(terciles))
    width = 0.8 / len(platforms)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, platform in enumerate(platforms):
        sub = df[df["source_id"] == platform].set_index("length_tercile").reindex(terciles)
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.bar(x + i * width - 0.4 + width / 2, sub["prevalence"], width=width,
               color=color, label=platform, zorder=3)
    ax.set_xticks(x, [f"{t}\n(n_sentences {b})" for t, b in
                       zip(terciles, ["1-4", "5-10", "11+"])])
    ax.set_ylabel("Content--Narrative prevalence")
    ax.set_ylim(0, 1.08)
    ax.set_title("Does the platform effect on synopsis-inclusion survive controlling for length?\n"
                  "(within each length tercile, platform differences shrink but don't fully disappear)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "platform_length_confound.png", dpi=160)
    plt.close(fig)


def fig_litfic_rating():
    df = pd.read_csv(RESULTS / "06_genre_platform/gold_litfic_rating_by_platform.csv")
    platforms = sorted(df["source_id"].unique())
    x = np.arange(len(platforms))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5))
    for is_lf, color, label in [(False, BLUE, "other genres"), (True, ORANGE, "Literary_fiction")]:
        sub = df[df["is_literary_fiction"] == is_lf].set_index("source_id").reindex(platforms)
        offset = -width / 2 if not is_lf else width / 2
        ax.bar(x + offset, sub["mean_rating"], width=width, color=color, label=label, zorder=3)
    ax.set_xticks(x, platforms)
    ax.set_ylabel("mean rating")
    ax.set_ylim(0, 5)
    ax.set_title("Literary_fiction rates lower than other genres, within every platform",
                  fontsize=10.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "litfic_rating_by_platform.png", dpi=160)
    plt.close(fig)


def _typology_group_heatmap(csv_name, out_name, title, row_label):
    table = pd.read_csv(RESULTS / csv_name, index_col=0)
    row_pct = table.to_numpy() / table.to_numpy().sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(table) + 1.5))
    im = ax.imshow(row_pct, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(table.shape[1]), [f"cluster {c}" for c in table.columns], fontsize=8.5)
    ax.set_yticks(range(table.shape[0]), table.index, fontsize=8.5)
    for i in range(row_pct.shape[0]):
        for j in range(row_pct.shape[1]):
            val = row_pct[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if val > 55 else INK_PRIMARY)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_ylabel(row_label)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160)
    plt.close(fig)


def fig_typology_by_genre():
    _typology_group_heatmap("06_genre_platform/gold_typology_by_genre.csv", "typology_by_genre.png",
                             "Compositional cluster membership by genre (row %)", "genre")


def fig_typology_by_platform():
    _typology_group_heatmap("06_genre_platform/gold_typology_by_platform.csv", "typology_by_platform.png",
                             "Compositional cluster membership by platform (row %)", "platform")


def fig_similarity_tests():
    tests = {
        "book\n(same work)": pd.read_csv(RESULTS / "07_book_author_similarity/gold_book_similarity_test.csv").iloc[0],
        "author\n(diff. work)": pd.read_csv(RESULTS / "07_book_author_similarity/gold_author_similarity_test.csv").iloc[0],
        "reviewer\n(excl. platform_1)": pd.read_csv(RESULTS / "07_book_author_similarity/gold_reviewer_similarity_test.csv").iloc[0],
    }
    labels = list(tests.keys())
    baseline = [tests[l]["baseline_mean_similarity"] for l in labels]
    observed = [tests[l]["observed_mean_similarity"] for l in labels]
    pvals = [tests[l]["p_value"] for l in labels]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(x - width / 2, baseline, width=width, color=GRAY_MUTED, label="random-pairs baseline", zorder=3)
    ax.bar(x + width / 2, observed, width=width, color=BLUE, label="same-group pairs (observed)", zorder=3)
    for i, p in enumerate(pvals):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        y = max(baseline[i], observed[i]) + 0.02
        ax.text(x[i], y, stars, ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.set_ylabel("mean cosine similarity")
    ax.set_ylim(0, 1.0)
    ax.set_title("Are same-book / same-author / same-reviewer review pairs more similar than random?\n"
                  "(* p<.05, ** p<.01, *** p<.001, permutation test; author is the one that isn't)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "similarity_tests_summary.png", dpi=160)
    plt.close(fig)


def fig_variance_partitioning():
    df = pd.read_csv(RESULTS / "07_book_author_similarity/gold_variance_partitioning.csv").sort_values("r2")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(df))
    ax.barh(y, df["r2"], color=BLUE, height=0.6, zorder=3)
    for i, (r2, p, k, n) in enumerate(zip(df["r2"], df["p_value"], df["k_groups"], df["n"])):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.text(r2 + 0.01, i, f"{stars}  (k={k}, n={n})", va="center", fontsize=8.5, color=INK_SECONDARY)
    ax.set_yticks(y, df["factor"])
    ax.set_xlabel("R² (one-way PERMANOVA, share of variance explained)")
    ax.set_xlim(0, df["r2"].max() * 1.35)
    ax.set_title("How much does each factor explain on its own?\n"
                  "(independent one-way tests, not a joint decomposition -- author's R² is confounded\n"
                  "with genre: 76% of multi-review authors write in only one genre, see text)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "variance_partitioning.png", dpi=160)
    plt.close(fig)


def fig_temporal_category_trends():
    df = pd.read_csv(RESULTS / "08_temporal/gold_temporal_category_trends.csv")
    df = df.sort_values("adjusted_year_coef")
    y = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_raw = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["raw_year_coef"]]
    colors_adj = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["adjusted_year_coef"]]
    ax.barh(y + width / 2, df["raw_year_coef"], height=width, color=colors_raw, alpha=0.45,
            label="raw (year alone)", zorder=3)
    ax.barh(y - width / 2, df["adjusted_year_coef"], height=width, color=colors_adj,
            label="adjusted (year + platform)", zorder=3)
    for i, q in enumerate(df["adjusted_q_bh"]):
        if q < 0.05:
            x = df["adjusted_year_coef"].iloc[i]
            ax.text(x + (0.004 if x > 0 else -0.004), y[i] - width / 2, "*",
                    ha="left" if x > 0 else "right", va="center", fontsize=11, fontweight="bold")
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("logistic regression coefficient per year (centered)")
    ax.set_title("Category prevalence trend over time: raw vs. platform-adjusted\n"
                  "(* survives platform adjustment at BH-adjusted q<0.05 -- faint bar = raw, solid = adjusted)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=GRAY_MUTED, alpha=0.45, label="raw (year alone)"),
                        Patch(facecolor=GRAY_MUTED, label="adjusted (year + platform)")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temporal_category_trends.png", dpi=160)
    plt.close(fig)


def _temporal_review_vs_sentence(csv_name, out_name, title_suffix):
    df = pd.read_csv(RESULTS / csv_name).sort_values("adjusted_year_coef")
    y = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors_rev = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["adjusted_year_coef"]]
    colors_sent = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["sentence_adjusted_year_coef"]]
    ax.barh(y + width / 2, df["adjusted_year_coef"], height=width, color=colors_rev, alpha=0.45,
            label="review-level (>=1 sentence)", zorder=3)
    ax.barh(y - width / 2, df["sentence_adjusted_year_coef"], height=width, color=colors_sent,
            label="sentence-level (no length confound)", zorder=3)
    for i in range(len(df)):
        if df["adjusted_q_bh"].iloc[i] < 0.05:
            x = df["adjusted_year_coef"].iloc[i]
            ax.text(x + (0.003 if x > 0 else -0.003), y[i] + width / 2, "*",
                    ha="left" if x > 0 else "right", va="center", fontsize=10, fontweight="bold")
        if df["sentence_adjusted_q_bh"].iloc[i] < 0.05:
            x = df["sentence_adjusted_year_coef"].iloc[i]
            ax.text(x + (0.003 if x > 0 else -0.003), y[i] - width / 2, "*",
                    ha="left" if x > 0 else "right", va="center", fontsize=10, fontweight="bold")
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("platform-adjusted logistic regression coefficient per year")
    ax.set_title(f"Does the review-level trend survive at the sentence level? {title_suffix}\n"
                  "(* q<0.05; faint = review-level, solid = sentence-level -- a review-level trend that\n"
                  "flips sign or loses its star at the sentence level was mostly a length artifact)",
                  fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=GRAY_MUTED, alpha=0.45, label="review-level (>=1 sentence)"),
                        Patch(facecolor=GRAY_MUTED, label="sentence-level (no length confound)")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, dpi=160)
    plt.close(fig)


def fig_gold_temporal_review_vs_sentence():
    _temporal_review_vs_sentence("08_temporal/gold_temporal_category_trends.csv",
                                  "gold_temporal_review_vs_sentence.png", "(gold)")


def fig_silver_temporal_review_vs_sentence():
    _temporal_review_vs_sentence("08_temporal/silver_temporal_category_trends.csv",
                                  "silver_temporal_review_vs_sentence.png", "(silver)")


def fig_period_prevalence():
    order = ["1999-2005", "2006-2010", "2011-2013", "2014-2016", "2017-2022"]
    _prevalence_heatmap("08_temporal/gold_period_prevalence.csv", "prevalence__", "period_prevalence_heatmap.png",
                         "Main-category prevalence by period", group_order=order)


def fig_typology_by_period():
    _typology_group_heatmap("08_temporal/gold_typology_by_period.csv", "typology_by_period.png",
                             "Compositional cluster membership by period (row %)", "period")


def fig_rating_length_over_time():
    df = pd.read_csv(RESULTS / "08_temporal/gold_temporal_rating_length.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["rating", "num_sentences"]
    y = np.arange(len(labels))
    raw = [df.loc[df.outcome == l, "raw_year_coef"].iloc[0] for l in labels]
    adj = [df.loc[df.outcome == l, "adjusted_year_coef"].iloc[0] for l in labels]
    raw_p = [df.loc[df.outcome == l, "raw_p_value"].iloc[0] for l in labels]
    adj_p = [df.loc[df.outcome == l, "adjusted_p_value"].iloc[0] for l in labels]
    width = 0.32
    ax.barh(y + width / 2, raw, height=width, color=BLUE, alpha=0.45, label="raw (year alone)", zorder=3)
    ax.barh(y - width / 2, adj, height=width, color=BLUE, label="adjusted (year + platform)", zorder=3)
    for i in range(len(labels)):
        for val, p, off in [(raw[i], raw_p[i], width / 2), (adj[i], adj_p[i], -width / 2)]:
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(val + (val / 20 if val >= 0 else -val / 20) + 0.02, y[i] + off, stars,
                    va="center", fontsize=8.5)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, ["mean rating\n(1-5 scale)", "review length\n(sentences)"], fontsize=9)
    ax.set_xlabel("regression coefficient per year (centered)")
    ax.set_title("Rating and review length over time, raw vs. platform-adjusted",
                  fontsize=10, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rating_length_over_time.png", dpi=160)
    plt.close(fig)


def fig_rating_correlations():
    df = pd.read_csv(RESULTS / "10_ratings/gold_rating_correlations.csv").sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["spearman_rho"]]
    ax.barh(y, df["spearman_rho"], color=colors, height=0.65, zorder=3)
    for i, q in enumerate(df["q_value_bh"]):
        if q < 0.05:
            x = df["spearman_rho"].iloc[i]
            ax.text(x + (0.01 if x > 0 else -0.01), i, "*", ha="left" if x > 0 else "right",
                    va="center", fontsize=11, fontweight="bold")
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("Spearman correlation with rating")
    ax.set_title("Which categories correlate with higher or lower ratings?\n(* q<0.05, BH-adjusted)",
                  fontsize=10.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rating_correlations.png", dpi=160)
    plt.close(fig)


def fig_rating_by_cluster():
    df = pd.read_csv(RESULTS / "10_ratings/gold_rating_by_cluster.csv").sort_values("mean_rating")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(df))
    se = df["std_rating"] / np.sqrt(df["n"])
    ax.barh(y, df["mean_rating"], xerr=1.96 * se, color=BLUE, height=0.65, zorder=3,
            error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1, capsize=3))
    ax.set_yticks(y, [f"cluster {c} (n={n})" for c, n in zip(df["cluster"], df["n"])], fontsize=9)
    ax.set_xlabel("mean rating (1-5 scale, 95% CI)")
    ax.set_xlim(1, 5)
    ax.set_title("Mean rating by typology cluster (Kruskal-Wallis p=1.3e-9)", fontsize=10.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rating_by_cluster.png", dpi=160)
    plt.close(fig)


def fig_rating_ordinal_model():
    df = pd.read_csv(RESULTS / "10_ratings/gold_rating_ordinal_model.csv").sort_values("coef")
    label_map = {
        "is_literary_fiction": "Literary_fiction (genre)",
        "length_std": "review length (standardized)",
        **{f"prop__{l}": l for l in MAIN_LABELS if l != "None"},
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))
    colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["coef"]]
    ax.barh(y, df["coef"], color=colors, height=0.65, zorder=3)
    for i, q in enumerate(df["q_value_bh"]):
        stars = "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""
        if stars:
            x = df["coef"].iloc[i]
            ax.text(x + (0.08 if x > 0 else -0.08), i, stars, ha="left" if x > 0 else "right",
                    va="center", fontsize=10, fontweight="bold")
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["predictor"].map(lambda p: label_map.get(p, p)), fontsize=9)
    ax.set_xlabel("ordinal logistic regression coefficient")
    ax.set_title("What predicts rating, holding the others constant?\n"
                  "(rating 1-5 ~ main-category proportions + length + Literary_fiction; * q<.05 ** q<.01 *** q<.001)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rating_ordinal_model.png", dpi=160)
    plt.close(fig)


"""--- Phase C (silver-set) figures below. Same palette/mark conventions,
reusing the generic helpers above where the CSV shape matches; bespoke
functions only where silver needs something gold's report didn't (the
gold-vs-silver replication comparisons). ---"""


def fig_silver_occurrence_comparison():
    df = pd.read_csv(RESULTS / "01_occurrence/silver_vs_gold_occurrence.csv").sort_values("sentence_prevalence_corrected")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(df))
    width = 0.26
    ax.barh(y - width, df["gold_sentence_prevalence"], height=width, color=GRAY_MUTED, label="gold (sentence-level)", zorder=3)
    ax.barh(y, df["sentence_prevalence_raw"], height=width, color=BLUE_LIGHT, label="silver, raw", zorder=3)
    ax.barh(y + width, df["sentence_prevalence_corrected"], height=width, color=BLUE, label="silver, Rogan-Gladen corrected", zorder=3)
    err_lo = df["sentence_prevalence_corrected"] - df["ci_lo"]
    err_hi = df["ci_hi"] - df["sentence_prevalence_corrected"]
    ax.errorbar(df["sentence_prevalence_corrected"], y + width, xerr=[err_lo, err_hi], fmt="none",
                ecolor=INK_SECONDARY, elinewidth=1, capsize=2, zorder=4)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("sentence-level prevalence")
    ax.set_title("Occurrence replication: gold vs. silver, raw vs. Rogan-Gladen corrected\n"
                  "(both silver bars are sentence-level, matching gold and the confusion matrix's own unit)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_occurrence_comparison.png", dpi=160)
    plt.close(fig)


def fig_silver_cooccurrence():
    sent = pd.read_csv(RESULTS / "02_cooccurrence/silver_cooccurrence_sentence.csv")
    labels = sorted(set(sent["label_a"]) | set(sent["label_b"]))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)

    pmi_mat = np.full((n, n), np.nan)
    for _, row in sent.iterrows():
        i, j = idx[row["label_a"]], idx[row["label_b"]]
        pmi_mat[i, j] = pmi_mat[j, i] = row["pmi"]
    _heatmap(pmi_mat, labels, FIG_DIR / "silver_cooccurrence_sentence_pmi.png",
              "Sentence-level co-occurrence, silver (raw), 5.2M sentences\n"
              "red = co-occur more than chance, blue = less, pale yellow = excluded/never",
              "PMI (log2)", vlim=3.5)

    rev = pd.read_csv(RESULTS / "02_cooccurrence/silver_cooccurrence_review.csv")
    z_mat = np.full((n, n), np.nan)
    for _, row in rev.iterrows():
        i, j = idx[row["label_a"]], idx[row["label_b"]]
        z_mat[i, j] = z_mat[j, i] = row["z_score"]
    _heatmap(z_mat, labels, FIG_DIR / "silver_cooccurrence_review_zscore.png",
              "Review-level co-occurrence vs. length-stratified permutation null, silver (raw)\n"
              "red = co-occur across a review more than length alone predicts, blue = less",
              "z-score", vlim=120)


def fig_silver_contiguity():
    df = pd.read_csv(RESULTS / "03_contiguity/silver_contiguity_label_summary.csv").dropna(subset=["mean_clumpiness"])
    df = df.sort_values("mean_clumpiness")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    ax.barh(y, df["mean_clumpiness"], color=BLUE, height=0.7, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("mean clumpiness index (0 = maximally scattered, 1 = single contiguous block)")
    ax.set_title("Contiguity, silver (raw), 466,976 reviews\n"
                  "All labels significantly more contiguous than random (n this large: p-values aren't informative, see text)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_contiguity_clumpiness.png", dpi=160)
    plt.close(fig)


def fig_silver_order_position():
    df = pd.read_csv(RESULTS / "04_order/silver_order_position.csv").sort_values("mean_rel_position")
    color_map = {"early": BLUE, "late": ORANGE, "none": GRAY_MUTED}
    colors = df["direction"].map(color_map)
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(df))
    err_lo = df["mean_rel_position"] - df["ci_lo"]
    err_hi = df["ci_hi"] - df["mean_rel_position"]
    ax.errorbar(df["mean_rel_position"], y, xerr=[err_lo, err_hi], fmt="o",
                color=INK_SECONDARY, ecolor=INK_SECONDARY, elinewidth=1, capsize=2,
                markersize=0, zorder=3)
    ax.scatter(df["mean_rel_position"], y, c=colors, s=60, zorder=4, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mean relative position in review (0 = first sentence, 1 = last)")
    ax.set_title("Where in the review each category tends to appear, silver (raw)\n"
                  "(95% CI via analytic normal approximation -- n too large for resampling to add anything)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    legend_handles = [Patch(facecolor=BLUE, label="significantly early"),
                       Patch(facecolor=ORANGE, label="significantly late"),
                       Patch(facecolor=GRAY_MUTED, label="no significant preference")]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_order_position.png", dpi=160)
    plt.close(fig)


def fig_silver_order_transitions():
    wide = pd.read_csv(RESULTS / "04_order/silver_order_transition_matrix.csv", index_col=0)
    labels = list(wide.index)
    counts = wide.to_numpy(dtype=float)
    total = counts.sum()
    from_marginal = counts.sum(axis=1) / total
    to_marginal = counts.sum(axis=0) / total
    n = len(labels)
    pmi = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            p_obs = counts[i, j] / total
            p_exp = from_marginal[i] * to_marginal[j]
            if p_obs > 0 and p_exp > 0:
                pmi[i, j] = np.log2(p_obs / p_exp)
    _heatmap(pmi, labels, FIG_DIR / "silver_order_transition_pmi.png",
              "What kind of sentence tends to follow what, silver (raw, main categories only)\n"
              "row = category of sentence t, column = category of sentence t+1; "
              "red = more than base rates predict",
              "PMI (log2)", vlim=2.0)


def fig_silver_typology_k_selection():
    df = pd.read_csv(RESULTS / "05_typology/silver_typology_k_selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, clustering, color in zip(axes, ["compositional", "sequence"], [BLUE, ORANGE]):
        sub = df[df["clustering"] == clustering]
        ax.plot(sub["k"], sub["silhouette"], color=color, marker="o", markersize=4,
                linewidth=1.6, label="silhouette")
        ax.plot(sub["k"], sub["bootstrap_ari_stability"], color=color, marker="o",
                markersize=4, linewidth=1.6, linestyle="--", label="bootstrap stability (ARI)")
        ax.set_title(clustering, fontsize=10.5, loc="left")
        ax.set_xlabel("k (number of clusters)")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="center right")
    axes[0].set_ylabel("score")
    fig.suptitle("Cluster count (k) selection, silver -- silhouette on a 10K subsample, "
                  "stability on 20K-row bootstraps", fontsize=10.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "silver_typology_k_selection.png", dpi=160)
    plt.close(fig)


def fig_silver_typology_cluster_profile():
    profile = pd.read_csv(RESULTS / "05_typology/silver_typology_cluster_profile.csv")
    main_used = [m for m in MAIN_LABELS if m != "None"]
    cols = [f"mean_prop__{l}" for l in main_used]
    matrix = profile[cols].to_numpy()
    n_clusters = len(profile)

    fig, ax = plt.subplots(figsize=(8, 1 + 0.55 * n_clusters))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(main_used)), main_used, rotation=30, ha="right", fontsize=9)
    ylabels = [f"cluster {c}  (n={n:,}, {p:.0f}%)" for c, n, p in
               zip(profile["cluster"], profile["n_reviews"], profile["pct_of_corpus"])]
    ax.set_yticks(range(n_clusters), ylabels, fontsize=9)
    for i in range(n_clusters):
        for j in range(len(main_used)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if val > matrix.max() * 0.6 else INK_PRIMARY)
    ax.set_title("Compositional cluster profile, silver (raw): mean sentence proportion per main category",
                  fontsize=10, loc="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("mean proportion of sentences")
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_typology_cluster_profile.png", dpi=160)
    plt.close(fig)


def fig_silver_typology_crosstab():
    crosstab = pd.read_csv(RESULTS / "05_typology/silver_typology_crosstab.csv", index_col=0)
    counts = crosstab.to_numpy()
    row_pct = counts / counts.sum(axis=1, keepdims=True) * 100
    marginal_pct = counts.sum(axis=0) / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6.5, 7))
    im = ax.imshow(row_pct, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(counts.shape[1]),
                  [f"seq {c}\n({marginal_pct[i]:.0f}%)" for i, c in enumerate(crosstab.columns)],
                  fontsize=8)
    ax.set_yticks(range(counts.shape[0]), [f"comp {c}" for c in crosstab.index], fontsize=9)
    for i in range(row_pct.shape[0]):
        for j in range(row_pct.shape[1]):
            val = row_pct[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if val > 60 else INK_PRIMARY)
    ax.set_title("Compositional vs. sequence-based clusters, silver (row %)", fontsize=10, loc="left")
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_typology_crosstab.png", dpi=160)
    plt.close(fig)


def fig_silver_genre_prevalence():
    order = ["Literary_fiction", "Literary_thriller", "Suspense", "Other fiction",
             "Children_fiction", "Fantasy_fiction", "Young_adult", "Historical_fiction",
             "Regional_fiction", "Romance"]
    _prevalence_heatmap("06_genre_platform/silver_genre_prevalence.csv", "prevalence__", "silver_genre_prevalence_heatmap.png",
                         "Main-category prevalence by genre, silver (raw)", group_order=order)


def fig_silver_platform_prevalence():
    _prevalence_heatmap("06_genre_platform/silver_platform_prevalence.csv", "prevalence__", "silver_platform_prevalence_heatmap.png",
                         "Main-category prevalence by platform, silver (raw, 8 platforms)")


def fig_silver_typology_by_genre():
    _typology_group_heatmap("06_genre_platform/silver_typology_by_genre.csv", "silver_typology_by_genre.png",
                             "Compositional cluster membership by genre, silver (row %)", "genre")


def fig_silver_typology_by_platform():
    _typology_group_heatmap("06_genre_platform/silver_typology_by_platform.csv", "silver_typology_by_platform.png",
                             "Compositional cluster membership by platform, silver (row %)", "platform")


def fig_silver_typology_by_period():
    _typology_group_heatmap("08_temporal/silver_typology_by_period.csv", "silver_typology_by_period.png",
                             "Compositional cluster membership by period, silver (row %)", "period")


def fig_silver_temporal_category_trends():
    df = pd.read_csv(RESULTS / "08_temporal/silver_temporal_category_trends.csv")
    df = df.sort_values("adjusted_year_coef")
    y = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_raw = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["raw_year_coef"]]
    colors_adj = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["adjusted_year_coef"]]
    ax.barh(y + width / 2, df["raw_year_coef"], height=width, color=colors_raw, alpha=0.45,
            label="raw (year alone)", zorder=3)
    ax.barh(y - width / 2, df["adjusted_year_coef"], height=width, color=colors_adj,
            label="adjusted (year + platform)", zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("logistic regression coefficient per year (centered)")
    ax.set_title("Category prevalence trend over time, silver: raw vs. platform-adjusted\n"
                  "(all coefficients shown are significant at q<0.05 -- n this large makes that a low bar, see text)",
                  fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=GRAY_MUTED, alpha=0.45, label="raw (year alone)"),
                        Patch(facecolor=GRAY_MUTED, label="adjusted (year + platform)")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_temporal_category_trends.png", dpi=160)
    plt.close(fig)


def fig_silver_rating_correlations():
    df = pd.read_csv(RESULTS / "10_ratings/silver_rating_correlations.csv").sort_values("spearman_rho")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["spearman_rho"]]
    ax.barh(y, df["spearman_rho"], color=colors, height=0.65, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["label"].map(short), fontsize=8.5)
    ax.set_xlabel("Spearman correlation with rating")
    ax.set_title("Which categories correlate with higher or lower ratings? Silver (raw)\n"
                  "(all shown are significant at q<0.05 given n~437K)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_rating_correlations.png", dpi=160)
    plt.close(fig)


def fig_silver_rating_by_cluster():
    df = pd.read_csv(RESULTS / "10_ratings/silver_rating_by_cluster.csv").sort_values("mean_rating")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    y = np.arange(len(df))
    se = df["std_rating"] / np.sqrt(df["n"])
    ax.barh(y, df["mean_rating"], xerr=1.96 * se, color=BLUE, height=0.65, zorder=3,
            error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1, capsize=3))
    ax.set_yticks(y, [f"cluster {c} (n={n:,})" for c, n in zip(df["cluster"], df["n"])], fontsize=9)
    ax.set_xlabel("mean rating (1-5 scale, 95% CI)")
    ax.set_xlim(1, 5)
    ax.set_title("Mean rating by typology cluster, silver (raw)", fontsize=10.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_rating_by_cluster.png", dpi=160)
    plt.close(fig)


def fig_gold_vs_silver_litfic_rating():
    gold = pd.read_csv(RESULTS / "06_genre_platform/gold_litfic_rating_by_platform.csv")
    silver = pd.read_csv(RESULTS / "06_genre_platform/silver_litfic_rating_by_platform.csv")
    gold_platforms = sorted(gold["source_id"].unique())
    silver_platforms = sorted(silver["source_id"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, df, platforms, title in [
        (axes[0], gold, gold_platforms, f"gold (n={int(gold['n'].sum())} reviews)"),
        (axes[1], silver, silver_platforms, f"silver (n={int(silver['n'].sum()):,} reviews)"),
    ]:
        x = np.arange(len(platforms))
        width = 0.32
        for is_lf, color, label in [(False, BLUE, "other genres"), (True, ORANGE, "Literary_fiction")]:
            sub = df[df["is_literary_fiction"] == is_lf].set_index("source_id").reindex(platforms)
            offset = -width / 2 if not is_lf else width / 2
            ax.bar(x + offset, sub["mean_rating"], width=width, color=color, label=label, zorder=3)
        ax.set_xticks(x, platforms, rotation=30, ha="right")
        ax.set_ylim(0, 5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean rating")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Literary_fiction's rating penalty: dramatic in gold, mostly gone in silver",
                  fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "gold_vs_silver_litfic_rating.png", dpi=160)
    plt.close(fig)


def fig_gold_vs_silver_ordinal_model():
    gold = pd.read_csv(RESULTS / "10_ratings/gold_rating_ordinal_model.csv")
    silver = pd.read_csv(RESULTS / "10_ratings/silver_rating_ordinal_model.csv")
    label_map = {
        "is_literary_fiction": "Literary_fiction (genre)",
        "length_std": "review length (standardized)",
        **{f"prop__{l}": l for l in MAIN_LABELS if l != "None"},
    }
    common = [p for p in gold["predictor"] if p in set(silver["predictor"])]
    gold_idx = gold.set_index("predictor").loc[common]
    silver_idx = silver.set_index("predictor").loc[common]
    order = gold_idx["coef"].sort_values().index
    y = np.arange(len(order))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(y + width / 2, gold_idx.loc[order, "coef"], height=width, color=GRAY_MUTED, label="gold", zorder=3)
    ax.barh(y - width / 2, silver_idx.loc[order, "coef"], height=width, color=BLUE, label="silver", zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, [label_map.get(p, p) for p in order], fontsize=9)
    ax.set_xlabel("ordinal logistic regression coefficient")
    ax.set_title("Ordinal rating model, gold vs. silver: Literary_fiction shrinks ~10x, Recommendations stays strong",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gold_vs_silver_ordinal_model.png", dpi=160)
    plt.close(fig)


def fig_gold_vs_silver_similarity():
    def _load(prefix):
        return {
            "book\n(same work)": pd.read_csv(RESULTS / "07_book_author_similarity" / f"{prefix}_book_similarity_test.csv").iloc[0],
            "author\n(diff. work)": pd.read_csv(RESULTS / "07_book_author_similarity" / f"{prefix}_author_similarity_test.csv").iloc[0],
            "reviewer\n(excl. platform_1)": pd.read_csv(RESULTS / "07_book_author_similarity" / f"{prefix}_reviewer_similarity_test.csv").iloc[0],
        }
    gold_tests, silver_tests = _load("gold"), _load("silver")
    labels = list(gold_tests.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, tests, title in [
        (axes[0], gold_tests, "gold (permutation test, n in the hundreds)"),
        (axes[1], silver_tests, "silver (cluster bootstrap + sampled baseline, n up to 461K)"),
    ]:
        baseline = [tests[l]["baseline_mean_similarity"] for l in labels]
        observed = [tests[l]["observed_mean_similarity"] for l in labels]
        pvals = [tests[l]["p_value"] for l in labels]
        x = np.arange(len(labels))
        width = 0.32
        ax.bar(x - width / 2, baseline, width=width, color=GRAY_MUTED, label="random-pairs baseline", zorder=3)
        ax.bar(x + width / 2, observed, width=width, color=BLUE, label="same-group pairs (observed)", zorder=3)
        for i, p in enumerate(pvals):
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            y = max(baseline[i], observed[i]) + 0.02
            ax.text(x[i], y, stars, ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x, labels, fontsize=9.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.6)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("mean cosine similarity")
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Book/author/reviewer similarity, gold vs. silver: author flips from n.s. to significant",
                  fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "gold_vs_silver_similarity.png", dpi=160)
    plt.close(fig)


def fig_length_share_correlation():
    gold = pd.read_csv(RESULTS / "09_length_dimension/gold_length_share_correlation.csv").set_index("label")
    silver = pd.read_csv(RESULTS / "09_length_dimension/silver_length_share_correlation.csv").set_index("label")
    common = [l for l in silver.index if l in gold.index]  # silver's TESTED_LABELS subset of gold's
    order = gold.loc[common, "spearman_rho"].sort_values().index
    y = np.arange(len(order))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y + width / 2, gold.loc[order, "spearman_rho"], height=width, color=GRAY_MUTED, label="gold", zorder=3)
    ax.barh(y - width / 2, silver.loc[order, "spearman_rho"], height=width, color=BLUE, label="silver", zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, [short(l) for l in order], fontsize=9)
    ax.set_xlabel("Spearman rho: category's share of a review vs. review length (num_sentences)")
    ax.set_title("Do long and short reviews allocate their sentences differently?\n"
                  "(negative = category's share shrinks in longer reviews, positive = grows)",
                  fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_share_correlation.png", dpi=160)
    plt.close(fig)


def fig_length_quartile_by_cluster():
    for source in ["gold", "silver"]:
        _typology_group_heatmap(f"09_length_dimension/{source}_length_quartile_by_composition_cluster.csv",
                                 f"{source}_length_quartile_by_composition_cluster.png",
                                 f"Compositional cluster membership by review-length quartile, {source} (row %)",
                                 "length quartile")


def fig_density_specificity_by_cluster():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for col, source in enumerate(["gold", "silver"]):
        df = pd.read_csv(RESULTS / "09_length_dimension" / f"{source}_density_specificity_by_composition_cluster.csv")
        df = df.sort_values("cluster_composition")
        x = np.arange(len(df))
        xticks = [f"cluster {c}" for c in df["cluster_composition"]]

        ax_d = axes[0, col]
        ax_d.bar(x, df["mean_density"], color=BLUE, width=0.6, zorder=3)
        ax_d.set_title(f"{source} (n={int(df['n_reviews'].sum()):,} reviews)", fontsize=10, loc="left")
        if col == 0:
            ax_d.set_ylabel("mean density\n(active labels/sentence)", fontsize=9)
        ax_d.spines[["top", "right"]].set_visible(False)
        ax_d.grid(axis="y", linewidth=0.6)
        ax_d.grid(axis="x", visible=False)

        ax_s = axes[1, col]
        ax_s.bar(x, df["mean_specificity"], color=ORANGE, width=0.6, zorder=3)
        ax_s.set_xticks(x, xticks, rotation=30, ha="right", fontsize=8.5)
        ax_s.set_ylim(0, 1)
        if col == 0:
            ax_s.set_ylabel("mean specificity\n(share of labeled sentences w/ a sub-label)", fontsize=9)
        ax_s.spines[["top", "right"]].set_visible(False)
        ax_s.grid(axis="y", linewidth=0.6)
        ax_s.grid(axis="x", visible=False)

    fig.suptitle("Sentence density and specificity by compositional cluster", fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "density_specificity_by_cluster.png", dpi=160)
    plt.close(fig)


def fig_reviewer_similarity_ranking():
    df = pd.read_csv(RESULTS / "13_reviewer_similarity_ranking/silver_reviewer_similarity_ranking.csv")
    comp_sig = df["comp_q_bh"] < 0.05
    seq_sig = df["seq_q_bh"] < 0.05

    groups = [
        (~comp_sig & ~seq_sig, GRAY_MUTED, "neither", 0.35),
        (comp_sig & ~seq_sig, BLUE, "compositional only", 0.55),
        (~comp_sig & seq_sig, ORANGE, "sequential only", 0.55),
        (comp_sig & seq_sig, CATEGORICAL[5], "both", 0.75),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for mask, color, label, alpha in groups:
        ax.scatter(df.loc[mask, "comp_mean_similarity"], df.loc[mask, "seq_mean_similarity"],
                   s=10, color=color, alpha=alpha, linewidth=0, label=f"{label} (n={mask.sum()})")
    ax.set_xlabel("mean pairwise compositional similarity (own reviews)")
    ax.set_ylabel("mean pairwise sequential similarity (own reviews)")
    corr = df[["comp_mean_similarity", "seq_mean_similarity"]].corr().iloc[0, 1]
    ax.set_title(f"Continuous method, same question as the modal-share version: does a reviewer's own\n"
                 f"reviews resemble each other in composition AND in sequence? correlation = {corr:.2f} --\n"
                 f"5,747 reviewers with a valid composition vector; z-tested per reviewer against item 5's baseline",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_reviewer_similarity_ranking.png", dpi=160)
    plt.close(fig)


def fig_prolific_vs_occasional_platform():
    df = pd.read_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_platform.csv", index_col=0)
    platforms = list(df.columns)
    x = np.arange(len(platforms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x - width / 2, df.loc["occasional"], width=width, color=GRAY_MUTED, label="occasional (1-2 reviews)", zorder=3)
    ax.bar(x + width / 2, df.loc["prolific"], width=width, color=BLUE, label="prolific (>=10 reviews)", zorder=3)
    ax.set_xticks(x, platforms, rotation=30, ha="right")
    ax.set_ylabel("% of group's reviews")
    ax.set_title("Where prolific vs. occasional reviewers' reviews come from\n"
                 "(Cramer's V=0.70 -- the strongest single association in this comparison)",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_prolific_vs_occasional_platform.png", dpi=160)
    plt.close(fig)


def fig_prolific_early_vs_late():
    df = pd.read_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_early_vs_late_paired_composition.csv").sort_values("mean_paired_diff")
    y = np.arange(len(df))
    colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in df["mean_paired_diff"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(y, df["mean_paired_diff"], color=colors, zorder=3)
    for i, q in enumerate(df["q_bh"]):
        if q < 0.05:
            v = df["mean_paired_diff"].iloc[i]
            ax.text(v + (0.001 if v > 0 else -0.001), y[i], "*", ha="left" if v > 0 else "right",
                    va="center", fontsize=11, fontweight="bold")
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, df["measure"].map(short), fontsize=8.5)
    ax.set_xlabel("mean paired difference (late reviews - early reviews), same reviewer")
    ax.set_title("Do a prolific reviewer's own reviews change from their first 3 to their most recent 3?\n"
                 "(* q<0.05, Wilcoxon signed-rank, n=5,735 reviewers -- entangled with the corpus-wide\n"
                 "temporal trend and length growth, see text)",
                 fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_prolific_early_vs_late.png", dpi=160)
    plt.close(fig)


def fig_fame_mediation():
    df = pd.read_csv(RESULTS / "17_fame_extensions/silver_fame_total_vs_direct_effect.csv")
    labels = df["label"].unique().tolist()
    y = np.arange(len(labels))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, covariate, title in [(axes[0], "book_fame", "book fame"), (axes[1], "author_fame", "author fame")]:
        sub = df[df["covariate"] == covariate].set_index("label").loc[labels]
        ax.barh(y + width / 2, sub["total_effect_mean"], height=width, color=GRAY_MUTED,
                label="total effect (no reviewer_experience)", zorder=3)
        ax.barh(y - width / 2, sub["direct_effect_mean"], height=width, color=BLUE,
                label="direct effect (item 9, w/ reviewer_experience)", zorder=3)
        ax.axvline(0, color=BASELINE, linewidth=1)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", linewidth=0.6)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("posterior mean coefficient")
    axes[1].set_xlabel("posterior mean coefficient")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Is fame's effect direct, or partly explained by who reviews?\n"
                 "(total = fame's raw relationship with coverage; direct = net of reviewer experience;\n"
                 "gap between the two = share mediated through reviewer selection, mostly small)",
                 fontsize=9.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(FIG_DIR / "silver_fame_mediation.png", dpi=160)
    plt.close(fig)


def fig_fame_by_genre():
    df = pd.read_csv(RESULTS / "17_fame_extensions/silver_fame_genre_fixed_effects.csv")
    sub = df[(df["label"] == "Content") & (df["covariate"].str.startswith("book_fame:nur_genre"))].copy()
    sub["genre"] = sub["covariate"].str.extract(r"\[(.*)\]")
    sub = sub.sort_values("mean")
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in sub["mean"]]
    err_lo = sub["mean"] - sub["hdi_3%"]
    err_hi = sub["hdi_97%"] - sub["mean"]
    ax.barh(y, sub["mean"], color=colors, zorder=3)
    ax.errorbar(sub["mean"], y, xerr=[err_lo, err_hi], fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1, capsize=3, zorder=4)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, sub["genre"], fontsize=9)
    ax.set_xlabel("book_fame x genre coefficient on Content share, 94% HDI")
    ax.set_title("Book fame's effect on synopsis (Content) share is genre-specific,\n"
                 "not a clean literary-vs-genre-fiction split",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_fame_by_genre_content.png", dpi=160)
    plt.close(fig)


def fig_author_brand_by_genre():
    repeat = pd.read_csv(RESULTS / "17_fame_extensions/silver_repeat_author_rate_by_genre.csv")
    conc = pd.read_csv(RESULTS / "17_fame_extensions/silver_fame_concentration_by_genre.csv")
    fe = pd.read_csv(RESULTS / "17_fame_extensions/silver_fame_genre_fixed_effects.csv")
    author_coef = fe[(fe["label"] == "Author") & (fe["covariate"].str.startswith("author_fame:nur_genre", na=False))].copy()
    author_coef["nur_genre"] = author_coef["covariate"].str.extract(r"\[(.*)\]")

    merged = conc.merge(author_coef[["nur_genre", "mean"]], on="nur_genre").rename(columns={"mean": "author_fame_coef"})
    merged = merged.merge(repeat.rename(columns={"dominant_genre": "nur_genre"})[["nur_genre", "n_reviewers"]],
                           on="nur_genre", how="left")

    highlight = {"Suspense", "Literary_thriller", "Romance", "Literary_fiction"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    repeat_sorted = repeat.sort_values("mean_observed_repeat_author_rate")
    y = np.arange(len(repeat_sorted))
    colors = [ORANGE if g in highlight else GRAY_MUTED for g in repeat_sorted["dominant_genre"]]
    ax.barh(y, repeat_sorted["mean_observed_repeat_author_rate"], color=colors, zorder=3, label="observed")
    ax.scatter(repeat_sorted["mean_null_repeat_author_rate"], y, color=INK_PRIMARY, s=18, zorder=4, label="chance baseline")
    ax.set_yticks(y, repeat_sorted["dominant_genre"], fontsize=8.5)
    ax.set_xlabel("mean repeat-author rate")
    ax.set_title("Repeat-author rate by dominant genre", fontsize=10, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    # Literary_thriller and Historical_fiction sit almost on top of each other -- stagger
    # label offsets by rank along x so close points don't render overlapping text.
    merged_sorted = merged.sort_values("mean_author_fame_concentration").reset_index(drop=True)
    offsets = [(4, 4), (4, -12)] * (len(merged_sorted) // 2 + 1)
    for i, row in merged_sorted.iterrows():
        color = ORANGE if row["nur_genre"] in highlight else GRAY_MUTED
        ax.scatter(row["mean_author_fame_concentration"], row["author_fame_coef"], color=color, s=60, zorder=3)
        ax.annotate(row["nur_genre"], (row["mean_author_fame_concentration"], row["author_fame_coef"]),
                    fontsize=7.5, xytext=offsets[i], textcoords="offset points")
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("mean author_fame_concentration")
    ax.set_ylabel("author_fame x genre coefficient on Author share")
    ax.set_title("Concentration vs. author-brand effect, by genre\n(Spearman rho=-0.21, p=0.59 -- suggestive for the\nhighlighted genres, not a clean rule across all 9)",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_author_brand_by_genre.png", dpi=160)
    plt.close(fig)


def fig_review_contiguity():
    """Two panels because the single-panel raw histogram was actively
    misleading (A1b, 2026-08-31): the raw score's spike at 1.0 is short
    reviews whose own chance baseline is already ~0.82, not evidence of
    organization. The right panel is what actually carries the finding."""
    df = pd.read_csv(RESULTS / "19_review_contiguity_precedence/gold_review_contiguity.csv")
    strat = pd.read_csv(RESULTS / "22_review_contiguity_scaled/gold_review_contiguity_by_length.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    bins = np.linspace(0, 1, 31)
    ax.hist(df["null_mean"], bins=bins, color=GRAY_MUTED, alpha=0.75,
            label=f"chance baseline (mean {df['null_mean'].mean():.2f})", zorder=2)
    ax.hist(df["review_clumpiness"], bins=bins, color=BLUE, alpha=0.75,
            label=f"observed (mean {df['review_clumpiness'].mean():.2f})", zorder=3)
    ax.set_xlabel("raw review-level clumpiness")
    ax.set_ylabel("reviews")
    ax.set_title("Raw score — not comparable across lengths\n"
                 "(the 1.0 spike is short reviews, whose own baseline is ~0.82)",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    x = np.arange(len(strat))
    width = 0.36
    ax.bar(x - width / 2, strat["mean_raw_clumpiness"], width=width, color=GRAY_MUTED,
           label="raw clumpiness", zorder=3)
    ax.bar(x - width / 2, strat["mean_chance_baseline"], width=width, color=BASELINE,
           label="chance baseline", zorder=4)
    ax.bar(x + width / 2, strat["mean_excess_over_chance"], width=width, color=BLUE,
           label="excess over chance", zorder=3)
    for i, v in enumerate(strat["pct_significant"]):
        ax.text(x[i] + width / 2, strat["mean_excess_over_chance"].iloc[i] + 0.015,
                f"{v:.0f}% sig", ha="center", fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x, [f"{q}\n(med. {int(m)} sent.)" for q, m in
                       zip(strat["length_quartile"], strat["median_sentences"])], fontsize=8.5)
    ax.set_ylabel("clumpiness")
    ax.set_title("Length-controlled — organization RISES with length\n"
                 "raw score is near-flat only because the baseline falls with it",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.suptitle("Review-level contiguity: why the raw score needs a length control",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "gold_review_contiguity.png", dpi=160)
    plt.close(fig)


def fig_reviewer_organization_typology():
    ksel = pd.read_csv(RESULTS / "26_reviewer_organization_typology/silver_reviewer_organization_k_selection.csv")
    prof = pd.read_csv(RESULTS / "26_reviewer_organization_typology/silver_reviewer_organization_profile.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    ax.plot(ksel["k"], ksel["silhouette"], "o-", color=BLUE, label="silhouette", zorder=3)
    ax.plot(ksel["k"], ksel["bootstrap_ari"], "s--", color=ORANGE,
            label="bootstrap stability (ARI)", zorder=3)
    ax.axvline(2, color=BASELINE, linewidth=1)
    ax.set_xlabel("number of reviewer clusters (k)")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.02)
    ax.set_title("The data supports one axis, not a rich typology\n"
                 "both criteria peak at k=2 and fall away steadily after",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    feats = ["excess_over_chance", "boundary_excess", "Content", "Reader_response",
             "Style", "Author", "Classification", "Other_works", "Recommendations"]
    nice = {"excess_over_chance": "contiguity excess", "boundary_excess": "boundary excess"}
    p0 = prof[prof["org_cluster"] == 0].iloc[0]
    y = np.arange(len(feats))
    vals = [p0[f] for f in feats]
    colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in vals]
    ax.barh(y, vals, color=colors, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, [nice.get(f, f"{short(f)} position") for f in feats], fontsize=8.5)
    ax.set_xlabel("cluster 0 mean, residualized (0 = corpus norm)")
    ax.set_title("Cluster 0 (n=1,595): more blocked, and a more pronounced arc\n"
                 "Content earlier, everything else later. Cluster 1 (n=2,388)\n"
                 "is its mirror image — a single organized/unorganized axis",
                 fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_reviewer_organization_typology.png", dpi=160)
    plt.close(fig)


def fig_contiguity_by_genre_platform():
    aov = pd.read_csv(RESULTS / "25_contiguity_by_genre_platform/silver_contiguity_anova.csv")
    prec = pd.read_csv(RESULTS / "25_contiguity_by_genre_platform/silver_precedence_by_group.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    term_lab = {"C(nur_genre)": "genre", "C(source_id)": "platform", "n_sentences": "review length"}
    dvs = [("excess_over_chance", "contiguity excess"), ("boundary_excess", "boundary excess")]
    terms = ["C(nur_genre)", "C(source_id)", "n_sentences"]
    x = np.arange(len(terms))
    width = 0.36
    for i, (dv, lab) in enumerate(dvs):
        vals = [float(aov[(aov["dv"] == dv) & (aov["term"] == t)]["partial_eta_sq"].iloc[0]) for t in terms]
        ax.bar(x + (i - 0.5) * width, vals, width=width,
               color=[BLUE, GRAY_MUTED][i], label=lab, zorder=3)
    ax.axhline(0.01, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(0.02, 0.0115, "1% of variance", color=ORANGE, fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(x, [term_lab[t] for t in terms], fontsize=9)
    ax.set_ylabel("partial eta-squared")
    ax.set_title("Genre and platform barely move contiguity\n"
                 "every factor explains under 2% of variance — platform beats genre,\n"
                 "as elsewhere in this report, but both are tiny",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    opening = {"Author", "Classification", "Content", "Other_works"}
    rows = []
    for f, sub in prec.groupby("factor"):
        piv = sub.pivot_table(index=["label_a", "label_b"], columns="group",
                              values="p_a_before_b").dropna(thresh=3)
        for (a, b), r in piv.iterrows():
            rows.append({"mean_p": r.mean(), "range_p": r.max() - r.min(),
                         "within_opening": a in opening and b in opening})
    t = pd.DataFrame(rows)
    for mask, color, lab in [(~t["within_opening"], BLUE, "cross-zone pairs (ordered)"),
                              (t["within_opening"], ORANGE, "within the opening zone (unordered)")]:
        ax.scatter(t.loc[mask, "mean_p"], t.loc[mask, "range_p"], s=34, color=color,
                   alpha=0.8, linewidth=0, label=lab, zorder=3)
    ax.axvline(0.5, color=BASELINE, linewidth=1)
    ax.set_xlabel("mean P(A before B) across groups")
    ax.set_ylabel("range of P across groups (instability)")
    ax.set_title("The arc is universal — what varies never had an order\n"
                 "pairs that shift direction across genres/platforms all sit at P≈0.5;\n"
                 "every genuinely ordered pair keeps its direction everywhere",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linewidth=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_contiguity_by_genre_platform.png", dpi=160)
    plt.close(fig)


def fig_contiguity_as_trait():
    icc = pd.read_csv(RESULTS / "24_contiguity_as_trait/silver_contiguity_trait_icc.csv")
    cons = pd.read_csv(RESULTS / "24_contiguity_as_trait/silver_contiguity_trait_consistency.csv")
    icc = icc.sort_values("icc_raw")
    y = np.arange(len(icc))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    ax = axes[0]
    # benchmarks in grey, the measures actually under test in blue
    is_test = icc["measure"].isin(["excess_over_chance", "boundary_excess", "review_clumpiness"])
    ax.barh(y + width / 2, icc["icc_raw"], height=width,
            color=[BLUE if t else GRAY_MUTED for t in is_test], alpha=0.45,
            label="raw ICC", zorder=3)
    ax.barh(y - width / 2, icc["icc_resid_length_genre_platform"], height=width,
            color=[BLUE if t else GRAY_MUTED for t in is_test],
            label="after removing length, genre, platform", zorder=3)
    err = np.vstack([icc["icc_raw"] - icc["icc_ci_lo"], icc["icc_ci_hi"] - icc["icc_raw"]])
    ax.errorbar(icc["icc_raw"], y + width / 2, xerr=err, fmt="none",
                ecolor=INK_SECONDARY, elinewidth=1, capsize=2, zorder=4)
    ax.set_yticks(y, icc["label"], fontsize=8.5)
    ax.set_xlabel("ICC — share of variance between reviewers rather than within")
    ax.set_title("Contiguity is a real reviewer trait, but a modest one\n"
                 "blue = measures under test, grey = benchmarks; it survives length\n"
                 "control almost intact, yet stays ~3x weaker than composition",
                 fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(facecolor=GRAY_MUTED, alpha=0.45, label="raw ICC"),
                        Patch(facecolor=GRAY_MUTED, label="after removing length, genre, platform")],
              frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    lab = {"excess_over_chance": "contiguity\nexcess", "boundary_excess": "boundary\nexcess",
           "n_sentences": "review\nlength"}
    c = cons.copy()
    c["name"] = c["measure"].map(lab)
    x = np.arange(len(c))
    ax.bar(x, c["ratio"], width=0.55,
           color=[BLUE if m != "n_sentences" else GRAY_MUTED for m in c["measure"]], zorder=3)
    ax.axhline(1.0, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(0.02, 1.03, "1.0 = no more variable than chance", color=ORANGE, fontsize=8,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(x, c["name"], fontsize=8.5)
    ax.set_ylabel("observed / null spread of per-reviewer variability")
    ax.set_title("Consistency is itself a trait\n"
                 "reviewers differ in how variable they are — and more so\n"
                 "for organization than for length",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_contiguity_as_trait.png", dpi=160)
    plt.close(fig)


def fig_gold_vs_silver_zones():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    prec = pd.read_csv(RESULTS / "23_zone_analysis_scaled/silver_pairwise_precedence.csv")
    mem = pd.read_csv(RESULTS / "21_zone_discovery/silver_zone_membership.csv")
    labels = mem["label"].tolist()
    idx = {l: i for i, l in enumerate(labels)}
    D = np.zeros((len(labels), len(labels)))
    for _, r in prec.iterrows():
        if r["label_a"] in idx and r["label_b"] in idx:
            i, j = idx[r["label_a"]], idx[r["label_b"]]
            D[i, j] = D[j, i] = 2 * abs(r["p_a_before_b"] - 0.5)
    Z = linkage(squareform(D, checks=False), method="average")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    dendrogram(Z, labels=[short(l) for l in labels], ax=ax, color_threshold=0.20,
               above_threshold_color=GRAY_MUTED, leaf_font_size=9)
    ax.axhline(0.20, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(0.02, 0.215, "cut = 3 zones (0 consistency errors)", color=ORANGE, fontsize=8.5,
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("precedence distance")
    ax.set_title("Silver: a perfect 3-zone partition\n"
                 "resolves the k=3 vs k=4 ambiguity gold could not — sty/rea group together",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    vals = {}
    for src in ["gold", "silver"]:
        m = pd.read_csv(RESULTS / "21_zone_discovery" / f"{src}_zone_membership.csv")
        zmap = dict(zip(m["label"], m["zone"]))
        row = []
        zone_measure_dir = {"gold": "20_contiguity_zones", "silver": "23_zone_analysis_scaled"}[src]
        for fname in ["pair_interleaving", "pair_zone_affinity"]:
            d = pd.read_csv(RESULTS / zone_measure_dir / f"{src}_{fname}.csv")
            same = d["label_a"].map(zmap) == d["label_b"].map(zmap)
            row += [float(d.loc[same, "mean_excess"].mean()),
                    float(d.loc[~same, "mean_excess"].mean())]
        vals[src] = row
    x = np.arange(4)
    width = 0.36
    ax.bar(x - width / 2, vals["gold"], width=width, color=GRAY_MUTED, label="gold", zorder=3)
    ax.bar(x + width / 2, vals["silver"], width=width, color=BLUE, label="silver", zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x, ["separation\nWITHIN zone", "separation\nACROSS zones",
                       "affinity\nWITHIN zone", "affinity\nACROSS zones"], fontsize=8)
    ax.set_ylabel("mean excess over chance")
    ax.set_title("Gold said position and contiguity were orthogonal;\n"
                 "silver refines that — within-zone pairs are somewhat less\n"
                 "separated, though still not merged (affinity stays negative)",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gold_vs_silver_zones.png", dpi=160)
    plt.close(fig)


def fig_gold_vs_silver_review_contiguity():
    from scipy import stats as _st
    g = pd.read_csv(RESULTS / "22_review_contiguity_scaled/gold_review_contiguity_scaled_by_length.csv")
    s = pd.read_csv(RESULTS / "22_review_contiguity_scaled/silver_review_contiguity_scaled_by_length.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    x = np.arange(len(g))
    width = 0.36
    ax.bar(x - width / 2, g["mean_excess_over_chance"], width=width, color=GRAY_MUTED,
           label=f"gold (n={int(g['n_reviews'].sum()):,})", zorder=3)
    ax.bar(x + width / 2, s["mean_excess_over_chance"], width=width, color=BLUE,
           label=f"silver (n={int(s['n_reviews'].sum()):,})", zorder=3)
    ax.set_xticks(x, [f"{q}\n(med. {int(m)} / {int(m2)} sent.)" for q, m, m2 in
                       zip(g["length_quartile"], g["median_sentences"], s["median_sentences"])],
                  fontsize=8)
    ax.set_ylabel("excess organization over chance")
    ax.set_title("Organization rises with review length — replicates at 400x scale\n"
                 "(smaller magnitude in silver, identical monotonic shape)",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    rows = []
    for src in ["gold", "silver"]:
        d = pd.read_csv(RESULTS / "22_review_contiguity_scaled" / f"{src}_review_contiguity_scaled.csv")
        d = d[d["eligible_a1"]]
        rows.append([_st.spearmanr(d["n_sentences"], d[c])[0]
                     for c in ["review_clumpiness", "null_mean", "excess_over_chance"]])
    labels = ["raw\nclumpiness", "chance\nbaseline", "EXCESS\n(the real pattern)"]
    x = np.arange(3)
    ax.bar(x - width / 2, rows[0], width=width, color=GRAY_MUTED, label="gold", zorder=3)
    ax.bar(x + width / 2, rows[1], width=width, color=BLUE, label="silver", zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("Spearman rho with review length")
    ax.set_title("Why the raw score misleads — same structure in both\n"
                 "raw looks flat/negative only because the baseline falls faster",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gold_vs_silver_review_contiguity.png", dpi=160)
    plt.close(fig)


def fig_zone_discovery():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    prec = pd.read_csv(RESULTS / "19_review_contiguity_precedence/gold_pairwise_precedence.csv")
    membership = pd.read_csv(RESULTS / "21_zone_discovery/gold_zone_membership.csv")
    labels = membership["label"].tolist()
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    D = np.zeros((n, n))
    for _, r in prec.iterrows():
        i, j = idx[r["label_a"]], idx[r["label_b"]]
        D[i, j] = D[j, i] = 2 * abs(r["p_a_before_b"] - 0.5)
    Z = linkage(squareform(D, checks=False), method="average")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax = axes[0]
    dendrogram(Z, labels=[short(l) for l in labels], ax=ax, color_threshold=0.10,
               above_threshold_color=GRAY_MUTED, leaf_font_size=9)
    ax.axhline(0.10, color=ORANGE, linestyle="--", linewidth=1.2)
    ax.text(0.02, 0.115, "cut = 4 zones", color=ORANGE, fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.text(0.02, 0.135, "sty/rea merge at 0.12 — the k=3 vs k=4 call is this thin",
            color=INK_SECONDARY, fontsize=7.5, transform=ax.get_yaxis_transform())
    ax.set_ylabel("precedence distance  =  2 x |P(A before B) - 0.5|")
    ax.set_title("Zones from the precedence structure\n"
                 "(0 = no reliable order between them, 1 = one always precedes)",
                 fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    inter = pd.read_csv(RESULTS / "20_contiguity_zones/gold_pair_interleaving.csv")
    zone = pd.read_csv(RESULTS / "20_contiguity_zones/gold_pair_zone_affinity.csv")
    zmap = dict(zip(membership["label"], membership["zone"]))
    groups = []
    for df, name in [(inter, "separation\n(A3)"), (zone, "zone affinity\n(B2)")]:
        d = df.copy()
        same = d["label_a"].map(zmap) == d["label_b"].map(zmap)
        groups.append((name, float(d.loc[same, "mean_excess"].mean()),
                       float(d.loc[~same, "mean_excess"].mean())))
    x = np.arange(len(groups))
    width = 0.35
    ax.bar(x - width / 2, [g[1] for g in groups], width=width, color=BLUE,
           label="within the same positional zone", zorder=3)
    ax.bar(x + width / 2, [g[2] for g in groups], width=width, color=GRAY_MUTED,
           label="across different zones", zorder=3)
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x, [g[0] for g in groups], fontsize=9)
    ax.set_ylabel("mean excess over chance")
    ax.set_title("Sharing a positional zone does NOT make categories merge\n"
                 "within-zone and cross-zone contiguity profiles are near-identical —\n"
                 "position and contiguity are essentially orthogonal here",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gold_zone_discovery.png", dpi=160)
    plt.close(fig)


def fig_contiguity_zones():
    review = pd.read_csv(RESULTS / "20_contiguity_zones/gold_review_boundary_interleaving.csv")
    a1 = pd.read_csv(RESULTS / "19_review_contiguity_precedence/gold_review_contiguity.csv")[["review_id", "excess_over_chance"]]
    zone = pd.read_csv(RESULTS / "20_contiguity_zones/gold_pair_zone_affinity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    m = review.merge(a1, on="review_id").dropna(subset=["boundary_excess", "excess_over_chance"])
    ax.scatter(m["excess_over_chance"], m["boundary_excess"], s=12, color=BLUE, alpha=0.45, linewidth=0)
    lim = [-0.35, 0.75]
    ax.plot(lim, lim, color=BASELINE, linewidth=1, zorder=1)
    ax.axhline(0, color=BASELINE, linewidth=0.8)
    ax.axvline(0, color=BASELINE, linewidth=0.8)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("A1: per-label clumpiness, excess over chance")
    ax.set_ylabel("A2: configuration-boundary score, excess over chance")
    rho = m["excess_over_chance"].corr(m["boundary_excess"], method="spearman")
    ax.set_title(f"A1 and A2 measure related but distinct things\n"
                 f"(rho={rho:+.2f}, n={len(m)}; a review can block each label well\n"
                 f"yet still change configuration often, via staggered boundaries)",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linewidth=0.6)

    ax = axes[1]
    z = zone.sort_values("mean_excess").reset_index(drop=True)
    y = np.arange(len(z))
    colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in z["mean_excess"]]
    ax.barh(y, z["mean_excess"], color=colors, zorder=3)
    for i, (v, q) in enumerate(zip(z["mean_excess"], z["q_bh"])):
        if q < 0.05:
            ax.text(v - 0.004 if v < 0 else v + 0.004, y[i], "*", ha="right" if v < 0 else "left",
                    va="center", fontsize=11, fontweight="bold")
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, [f"{short(a)} + {short(b)}" for a, b in zip(z["label_a"], z["label_b"])], fontsize=8)
    ax.set_xlabel("B2: zone affinity, excess over chance")
    ax.set_title("B2: categories do NOT merge into joint zones\n"
                 "no pair is above chance; 6 are significantly BELOW it (*q<0.05),\n"
                 "i.e. their blocks sit further apart than random ordering gives",
                 fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gold_contiguity_zones.png", dpi=160)
    plt.close(fig)


def fig_pairwise_precedence():
    prec = pd.read_csv(RESULTS / "19_review_contiguity_precedence/gold_pairwise_precedence.csv")
    order = pd.read_csv(RESULTS / "04_order/gold_order_position.csv")
    order = order[order["level"] == "main"].sort_values("mean_rel_position")["label"].tolist()
    order = [l for l in order if l in set(prec["label_a"]) | set(prec["label_b"])]
    n = len(order)
    idx = {l: i for i, l in enumerate(order)}

    mat = np.full((n, n), np.nan)
    sig = np.zeros((n, n), dtype=bool)
    for _, r in prec.iterrows():
        i, j = idx[r["label_a"]], idx[r["label_b"]]
        mat[i, j] = r["p_a_before_b"]
        mat[j, i] = 1 - r["p_a_before_b"]
        sig[i, j] = sig[j, i] = r["q_bh"] < 0.05

    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = DIVERGING_CMAP.copy()
    cmap.set_bad(NAN_COLOR)
    im = ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(vmin=0.1, vcenter=0.5, vmax=0.9), aspect="equal")
    ax.set_xticks(range(n), [short(l) for l in order], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n), [short(l) for l in order], fontsize=9)
    for i in range(n):
        for j in range(n):
            if i == j or np.isnan(mat[i, j]):
                continue
            txt = f"{mat[i, j]:.2f}" + ("" if sig[i, j] else " n.s.")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if abs(mat[i, j] - 0.5) > 0.28 else INK_PRIMARY)
    ax.set_title("P(row category precedes column category), within reviews containing both\n"
                 "red = row comes first, blue = column comes first, 'n.s.' = no reliable order (q>=0.05)\n"
                 "rows/cols ordered by marginal mean position; the 4 opening categories are\n"
                 "mutually unordered -- a zone, not a sequence",
                 fontsize=9, loc="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("P(row before column)", fontsize=9)
    ax.spines[:].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gold_pairwise_precedence.png", dpi=160)
    plt.close(fig)


def fig_content_subs_by_fame_genre():
    df = pd.read_csv(RESULTS / "18_content_spoiler_check/silver_content_subs_by_fame_genre.csv")
    sub = df[df["covariate"].str.startswith("book_fame:nur_genre", na=False)].copy()
    sub["genre"] = sub["covariate"].str.extract(r"\[(.*)\]")

    labels = ["Content--Narrative", "Content--Other", "Content--Quote"]
    # one consistent genre order across all 3 panels (sorted by the hypothesized driver,
    # Content--Narrative) -- sharey=True means the panels' tick labels are shared, so
    # sorting each panel independently would silently mismatch bars against labels
    genre_order = sub[sub["label"] == "Content--Narrative"].sort_values("mean")["genre"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, label in zip(axes, labels):
        s = sub[sub["label"] == label].set_index("genre").loc[genre_order].reset_index()
        y = np.arange(len(s))
        colors = [DIVERGING_POS if v > 0 else DIVERGING_NEG for v in s["mean"]]
        err_lo = s["mean"] - s["hdi_3%"]
        err_hi = s["hdi_97%"] - s["mean"]
        ax.barh(y, s["mean"], color=colors, zorder=3)
        ax.errorbar(s["mean"], y, xerr=[err_lo, err_hi], fmt="none", ecolor=INK_SECONDARY,
                    elinewidth=1, capsize=3, zorder=4)
        ax.axvline(0, color=BASELINE, linewidth=1)
        ax.set_yticks(y, s["genre"], fontsize=8.5)
        ax.set_title(short(label), fontsize=10, loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", linewidth=0.6)
        ax.grid(axis="y", visible=False)
    fig.suptitle("The Historical_fiction/Suspense divergence lives almost entirely in Content--Narrative\n"
                 "(plot/synopsis) -- flat-to-reversed in the other two Content sub-categories",
                 fontsize=10, x=0.02, ha="left")
    fig.supxlabel("book_fame x genre coefficient, 94% HDI", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.91])
    fig.savefig(FIG_DIR / "silver_content_subs_by_fame_genre.png", dpi=160)
    plt.close(fig)


def fig_bayesian_fame_coefficients():
    fe = pd.read_csv(RESULTS / "16_bayesian_coverage_model/silver_bayesian_coverage_fixed_effects.csv")
    labels = fe["label"].unique().tolist()
    y = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for offset, covariate, color, name in [(width / 2, "book_fame", BLUE, "book fame"),
                                            (-width / 2, "author_fame", ORANGE, "author fame")]:
        sub = fe[fe["covariate"] == covariate].set_index("label").loc[labels]
        err_lo = sub["mean"] - sub["hdi_3%"]
        err_hi = sub["hdi_97%"] - sub["mean"]
        ax.errorbar(sub["mean"], y + offset, xerr=[err_lo, err_hi], fmt="o", color=color,
                    markersize=5, capsize=3, label=name, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("posterior coefficient on log1p(prior review count), 94% HDI")
    ax.set_title("Book fame vs. author fame: separately estimable, often opposite-signed\n"
                 "(raw covariate correlation 0.37, but VIF~1.2 and posterior coefficient\n"
                 "correlation |r|<0.05 for every label -- not fighting each other)",
                 fontsize=9, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_bayesian_fame_coefficients.png", dpi=160)
    plt.close(fig)


def fig_imputed_correction_scatter():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cooc = pd.read_csv(RESULTS / "14_imputed_correction/silver_imputed_cooccurrence.csv").dropna(
        subset=["raw_subsample_pmi", "mean_imputed_pmi"])
    is_sparse = cooc["label_a"].isin(SPARSE_GOLD_LABELS) | cooc["label_b"].isin(SPARSE_GOLD_LABELS)
    ax = axes[0]
    ax.scatter(cooc.loc[~is_sparse, "raw_subsample_pmi"], cooc.loc[~is_sparse, "mean_imputed_pmi"],
               s=18, color=BLUE, alpha=0.7, label="ordinary label pair")
    ax.scatter(cooc.loc[is_sparse, "raw_subsample_pmi"], cooc.loc[is_sparse, "mean_imputed_pmi"],
               s=18, color=ORANGE, alpha=0.7, label="involves a sparse label")
    lim = [min(cooc["raw_subsample_pmi"].min(), cooc["mean_imputed_pmi"].min()) - 0.3,
           max(cooc["raw_subsample_pmi"].max(), cooc["mean_imputed_pmi"].max()) + 0.3]
    ax.plot(lim, lim, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("raw PMI (subsample)")
    ax.set_ylabel("corrected PMI (pooled, M=20)")
    ax.set_title("Co-occurrence (sentence PMI)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    contig = pd.read_csv(RESULTS / "14_imputed_correction/silver_imputed_contiguity.csv")
    ax = axes[1]
    ax.scatter(contig["raw_subsample_clumpiness"], contig["pooled_imputed_clumpiness"], s=18, color=BLUE, alpha=0.8)
    ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1, zorder=1)
    ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.02)
    ax.set_xlabel("raw clumpiness (subsample)")
    ax.set_ylabel("corrected clumpiness (pooled, M=20)")
    ax.set_title("Contiguity (mean clumpiness)", fontsize=10, loc="left")

    order = pd.read_csv(RESULTS / "14_imputed_correction/silver_imputed_order.csv")
    ax = axes[2]
    ax.scatter(order["raw_subsample_position"], order["pooled_imputed_position"], s=18, color=BLUE, alpha=0.8)
    lim = [order[["raw_subsample_position", "pooled_imputed_position"]].to_numpy().min() - 0.03,
           order[["raw_subsample_position", "pooled_imputed_position"]].to_numpy().max() + 0.03]
    ax.plot(lim, lim, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("raw mean rel. position (subsample)")
    ax.set_ylabel("corrected mean rel. position (pooled, M=20)")
    ax.set_title("Order (mean relative position)", fontsize=10, loc="left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(linewidth=0.6)
    fig.suptitle("Instance-level correction vs. raw, 10,000-review silver subsample "
                 "(points on the diagonal = correction changes nothing)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "silver_imputed_correction_scatter.png", dpi=160)
    plt.close(fig)


def fig_prolific_reviewer_tendency():
    df = pd.read_csv(RESULTS / "12_prolific_reviewer_tendency/silver_prolific_reviewer_tendency.csv")
    comp_sig = df["cluster_composition_q_bh"] < 0.05
    seq_sig = df["cluster_sequence_q_bh"] < 0.05

    groups = [
        (~comp_sig & ~seq_sig, GRAY_MUTED, "neither", 0.35),
        (comp_sig & ~seq_sig, BLUE, "compositional only", 0.55),
        (~comp_sig & seq_sig, ORANGE, "sequential only", 0.55),
        (comp_sig & seq_sig, CATEGORICAL[5], "both", 0.75),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for mask, color, label, alpha in groups:
        ax.scatter(df.loc[mask, "cluster_composition_share"], df.loc[mask, "cluster_sequence_share"],
                   s=10, color=color, alpha=alpha, linewidth=0, label=f"{label} (n={mask.sum()})")
    ax.set_xlabel("compositional modal-cluster share")
    ax.set_ylabel("sequential modal-cluster share")
    corr = df[["cluster_composition_share", "cluster_sequence_share"]].corr().iloc[0, 1]
    ax.set_title(f"Do reviewers with a strong compositional signature also have a strong sequential one?\n"
                 f"5,782 reviewers with >=10 reviews (excl. platform_1); correlation = {corr:.2f} --\n"
                 f"unlike the review-level composition/sequence independence in §5",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "silver_prolific_reviewer_tendency.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_occurrence()
    fig_cooccurrence()
    fig_contiguity()
    fig_order_position()
    fig_order_transitions()
    fig_typology_k_selection()
    fig_typology_cluster_profile()
    fig_typology_pca_scatter()
    fig_gold_sequence_cluster_profile()
    fig_typology_crosstab()
    fig_genre_prevalence()
    fig_platform_prevalence()
    fig_platform_length_confound()
    fig_litfic_rating()
    fig_typology_by_genre()
    fig_typology_by_platform()
    fig_similarity_tests()
    fig_variance_partitioning()
    fig_temporal_category_trends()
    fig_gold_temporal_review_vs_sentence()
    fig_period_prevalence()
    fig_typology_by_period()
    fig_rating_length_over_time()
    fig_rating_correlations()
    fig_rating_by_cluster()
    fig_rating_ordinal_model()
    fig_silver_occurrence_comparison()
    fig_silver_cooccurrence()
    fig_silver_contiguity()
    fig_silver_order_position()
    fig_silver_order_transitions()
    fig_silver_typology_k_selection()
    fig_silver_typology_cluster_profile()
    fig_silver_sequence_cluster_profile()
    fig_silver_typology_crosstab()
    fig_silver_genre_prevalence()
    fig_silver_platform_prevalence()
    fig_silver_typology_by_genre()
    fig_silver_typology_by_platform()
    fig_silver_typology_by_period()
    fig_silver_temporal_category_trends()
    fig_silver_temporal_review_vs_sentence()
    fig_silver_rating_correlations()
    fig_silver_rating_by_cluster()
    fig_gold_vs_silver_litfic_rating()
    fig_gold_vs_silver_ordinal_model()
    fig_gold_vs_silver_similarity()
    fig_length_share_correlation()
    fig_length_quartile_by_cluster()
    fig_density_specificity_by_cluster()
    fig_prolific_reviewer_tendency()
    fig_reviewer_similarity_ranking()
    fig_imputed_correction_scatter()
    fig_prolific_vs_occasional_platform()
    fig_prolific_early_vs_late()
    fig_bayesian_fame_coefficients()
    fig_fame_mediation()
    fig_fame_by_genre()
    fig_content_subs_by_fame_genre()
    fig_review_contiguity()
    fig_pairwise_precedence()
    fig_contiguity_zones()
    fig_zone_discovery()
    fig_gold_vs_silver_review_contiguity()
    fig_gold_vs_silver_zones()
    fig_contiguity_as_trait()
    fig_contiguity_by_genre_platform()
    fig_reviewer_organization_typology()
    fig_author_brand_by_genre()
    print(f"wrote figures to {FIG_DIR}")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(" ", p.name)
