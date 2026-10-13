"""Characterizes the sequence-based typology clusters (ANALYSIS_PLAN.md §6)
by themselves -- gold's and silver's reports so far only showed *that*
composition and sequence clusters are largely independent (the crosstab), not
*what* each sequence cluster actually looks like. This fills that gap.

Each sequence cluster is a group of reviews whose main categories tend to
occur in a similar relative order (independent of how much of each category
they contain -- that's what the compositional clustering captures instead).
The profile here reports, per sequence cluster: mean relative position of
each main category's own sentences (the "order signature" the clustering was
actually fit on), for comparison the mean *proportion* of each main category
(composition, so it's visible whether a sequence cluster corresponds to any
particular composition or genuinely cuts across them), and the usual
descriptives (size, rating, dominant genre/platform, and which compositional
cluster its members most often also belong to).

Doesn't refit anything -- reads the cluster assignments already saved by
05a_typology_clustering.py / 05c_typology_clustering.py and recomputes the
per-review positional features the same way those scripts did (cheap: a
groupby + fillna, not the expensive k-selection step), so this reruns in
seconds even for silver.

Positional feature is onset (first occurrence), not mean position, per
review per label, corpus onset-marginal-imputed where absent (2026-09-22,
matching 05a/05c after prototyping showed onset gives a cleaner, less
length-confounded signal -- see 05a's module docstring).

Usage: python src/11_sequence_cluster_profile.py [gold|silver]  (default: gold)
Writes results/{source}_typology_sequence_cluster_profile.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "gold"
assert SOURCE in ("gold", "silver")


def load(name, subdir):
    if SOURCE == "gold":
        return pd.read_csv(RESULTS / subdir / f"gold_{name}.csv")
    path_csv = RESULTS / subdir / f"silver_{name}.csv"
    path_parquet = RESULTS / subdir / f"silver_{name}.parquet"
    return pd.read_csv(path_csv) if path_csv.exists() else pd.read_parquet(path_parquet)


def compute_seqpos(sentence_table, review_ids, onset_marginal):
    out = pd.DataFrame({"review_id": review_ids})
    for label in MAIN_LABELS_USED:
        sub = sentence_table[sentence_table[label] == 1].groupby("review_id")["rel_position"].min()
        out[f"seqpos__{label}"] = out["review_id"].map(sub).fillna(onset_marginal.get(label, 0.5))
    return out


def main():
    clusters = load("typology_clusters", "05_typology")
    review_features = load("review_features", "00_data_prep")
    sentence_table = load("sentence_sequences", "00_data_prep")
    onset_marginal = load("order_onset", "04_order").set_index("label")["first_mean_position"]

    seqpos = compute_seqpos(sentence_table, clusters["review_id"].tolist(), onset_marginal)
    df = clusters.merge(seqpos, on="review_id").merge(
        review_features[["review_id"] + [f"prop__{l}" for l in MAIN_LABELS_USED]], on="review_id")

    n_total = len(df)
    rows = []
    for c, sub in df.groupby("cluster_sequence"):
        comp_mode = sub["cluster_composition"].value_counts()
        row = {
            "cluster_sequence": c, "n_reviews": len(sub),
            "pct_of_corpus": round(100 * len(sub) / n_total, 1),
            "mean_num_sentences": round(sub["num_sentences"].mean(), 1),
            "median_num_sentences": sub["num_sentences"].median(),
            "mean_rating": round(sub["rating"].dropna().astype(float).mean(), 2) if sub["rating"].notna().any() else "",
            "top_genre": sub["nur_genre"].value_counts().idxmax(),
            "top_genre_pct": round(100 * sub["nur_genre"].value_counts().iloc[0] / len(sub), 1),
            "top_platform": sub["source_id"].value_counts().idxmax(),
            "top_platform_pct": round(100 * sub["source_id"].value_counts().iloc[0] / len(sub), 1),
            "most_common_composition_cluster": int(comp_mode.idxmax()),
            "most_common_composition_cluster_pct": round(100 * comp_mode.iloc[0] / len(sub), 1),
        }
        for label in MAIN_LABELS_USED:
            row[f"mean_seqpos__{label}"] = round(sub[f"seqpos__{label}"].mean(), 4)
        for label in MAIN_LABELS_USED:
            row[f"mean_prop__{label}"] = round(sub[f"prop__{label}"].mean(), 4)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("cluster_sequence")
    out_path = RESULTS / "11_sequence_cluster_profile" / f"{SOURCE}_typology_sequence_cluster_profile.csv"
    out.to_csv(out_path, index=False)

    print(f"{SOURCE}: {n_total} reviews, {len(out)} sequence clusters")
    show_cols = ["cluster_sequence", "n_reviews", "pct_of_corpus", "mean_rating",
                 "top_genre", "top_genre_pct", "most_common_composition_cluster",
                 "most_common_composition_cluster_pct"] + [f"mean_seqpos__{l}" for l in MAIN_LABELS_USED]
    print(out[show_cols].to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
