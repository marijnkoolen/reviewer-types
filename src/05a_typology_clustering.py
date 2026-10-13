"""Review typology: are there distinct kinds of reviews based on what they
discuss? (ANALYSIS_PLAN.md §6, gold set only.)

Two independent clusterings, cross-checked against each other:

1. Compositional clustering -- KMeans on standardized main-category sentence
   proportions (Author, Classification, Content, Other_works, Reader_response,
   Recommendations, Style) plus the proportion of unlabeled sentences. This
   answers "what does the review spend its sentences on."

2. Sequence-based clustering -- KMeans on each main category's *onset*
   (first-occurrence) position *within reviews that contain it* (categories
   absent from a review are imputed with the corpus-wide onset marginal from
   §5, i.e. "no information" rather than an arbitrary 0). This answers
   "where in the review do things happen," independent of how much of each
   thing there is.

   Onset rather than mean position (chat, 2026-09-22, prototyped before
   adopting): a category that recurs many times in one review gets its mean
   position smoothed toward that review's local center almost by
   construction, which after standardization injects noise scaled up to the
   same weight as a genuine signal -- worst for exactly the categories most
   reviews discuss most (`Content`, `Reader_response`). Onset (always one
   value per review, however many times the category recurs) avoids that.
   Prototyped on gold before rollout: silhouette nearly doubles at k=2
   (0.138->0.257), the sequence/compositional independence finding sharpens
   (ARI 0.063->0.0085), and the length confound *drops* (eta^2 0.099->0.069)
   -- a clean improvement on every axis tested. Length-stratified imputation
   (matching the marginal to the review's own length quartile, the other
   §5 refinement) was tested alongside this and rejected: it leaks review
   length into the feature space through the imputed value itself and nearly
   doubled the confound (eta^2 up to 0.21) instead of reducing it, so
   imputation stays a single global constant.

k for each is chosen by silhouette score, corroborated with a bootstrap
stability check (resample reviews, refit, measure agreement with the
full-data clustering via adjusted Rand index -- a k that isn't stable under
resampling isn't a real typology, just a fit to noise).

Usage: python src/05a_typology_clustering.py
Writes results/05_typology/gold_typology_k_selection.csv, gold_typology_clusters.csv,
gold_typology_cluster_profile.csv, gold_typology_representative_reviews.csv,
gold_typology_crosstab.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GOLD_RAW = ROOT / "data" / "review_sentence_annotations.tsv"
REVIEW_FEATURES = RESULTS / "00_data_prep/gold_review_features.csv"
ONSET_TABLE = RESULTS / "04_order/gold_order_onset.csv"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
ALL_LABELS = MAIN_LABELS_USED + SUB_LABELS
K_RANGE = range(2, 9)
N_BOOTSTRAP = 50
RNG_SEED = 0


def select_k(X, label):
    rng = np.random.default_rng(RNG_SEED)
    n = X.shape[0]
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X)
        sil = silhouette_score(X, km.labels_)

        aris = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            km_boot = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X[idx])
            boot_labels_on_full = km_boot.predict(X)
            aris.append(adjusted_rand_score(km.labels_, boot_labels_on_full))
        stability = float(np.mean(aris))

        rows.append({"clustering": label, "k": k, "silhouette": round(sil, 4),
                      "bootstrap_ari_stability": round(stability, 4)})
    return pd.DataFrame(rows)


def pick_best_k(k_table):
    """Highest silhouette among k's with reasonable bootstrap stability (>=0.5);
    falls back to the highest-stability k if none clears that bar."""
    stable = k_table[k_table["bootstrap_ari_stability"] >= 0.5]
    pool = stable if len(stable) else k_table
    return int(pool.loc[pool["silhouette"].idxmax(), "k"])


def representative_reviews(df, X, km, id_col="review_id", n_per_cluster=3):
    sent_text = {}
    with open(GOLD_RAW, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (row["review_id"], row["sent_num"])
            sent_text[key] = row["sent_text"]

    rows = []
    for c in sorted(set(km.labels_)):
        mask = km.labels_ == c
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(X[mask] - centroid, axis=1)
        closest_idx = np.argsort(dists)[:n_per_cluster]
        sub = df.loc[mask].iloc[closest_idx]
        for _, row in sub.iterrows():
            snippet = sent_text.get((row[id_col], "1"), "")
            words = snippet.split()
            if len(words) > 15:
                snippet = " ".join(words[:15]) + " ..."
            rows.append({
                "cluster": c, "review_id": row[id_col], "nur_genre": row["nur_genre"],
                "source_id": row["source_id"], "review_num_words": row["review_num_words"],
                "first_sentence": snippet,
            })
    return pd.DataFrame(rows)


def cluster_profile(df, cluster_col, n_reviews):
    rows = []
    for c, sub in df.groupby(cluster_col):
        row = {
            "cluster": c, "n_reviews": len(sub), "pct_of_corpus": round(100 * len(sub) / n_reviews, 1),
            "mean_num_sentences": round(sub["num_sentences"].mean(), 1),
            "median_num_sentences": sub["num_sentences"].median(),
            "mean_num_words": round(sub["review_num_words"].mean(), 1),
            "median_num_words": sub["review_num_words"].median(),
            "mean_rating": round(sub["rating"].dropna().astype(float).mean(), 2) if sub["rating"].notna().any() else "",
            "n_with_rating": int(sub["rating"].notna().sum()),
            "top_genre": sub["nur_genre"].value_counts().idxmax(),
            "top_genre_pct": round(100 * sub["nur_genre"].value_counts().iloc[0] / len(sub), 1),
            "top_platform": sub["source_id"].value_counts().idxmax(),
            "top_platform_pct": round(100 * sub["source_id"].value_counts().iloc[0] / len(sub), 1),
        }
        for label in ALL_LABELS:
            row[f"mean_prop__{label}"] = round(sub[f"prop__{label}"].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")


def main():
    df = pd.read_csv(REVIEW_FEATURES)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    n = len(df)

    # --- 1. compositional clustering ---
    comp_cols = [f"prop__{label}" for label in MAIN_LABELS_USED]
    comp_features = df[comp_cols].copy()
    comp_features["prop__unlabeled"] = df["n_unlabeled_sentences"] / df["num_sentences"]
    X_comp = StandardScaler().fit_transform(comp_features.to_numpy())

    comp_k_table = select_k(X_comp, "compositional")
    k_comp = pick_best_k(comp_k_table)
    km_comp = KMeans(n_clusters=k_comp, n_init=10, random_state=RNG_SEED).fit(X_comp)
    df["cluster_composition"] = km_comp.labels_

    # --- 2. sequence-based clustering (cross-check) ---
    onset_marginal = pd.read_csv(ONSET_TABLE).set_index("label")["first_mean_position"]
    seq_pos_df = compute_review_level_positions(df["review_id"].tolist(), MAIN_LABELS_USED, onset_marginal)
    df = df.merge(seq_pos_df, on="review_id", how="left")
    X_seq = StandardScaler().fit_transform(df[[f"seqpos__{l}" for l in MAIN_LABELS_USED]].to_numpy())

    seq_k_table = select_k(X_seq, "sequence")
    k_seq = pick_best_k(seq_k_table)
    km_seq = KMeans(n_clusters=k_seq, n_init=10, random_state=RNG_SEED).fit(X_seq)
    df["cluster_sequence"] = km_seq.labels_

    (RESULTS / "05_typology").mkdir(exist_ok=True, parents=True)

    k_table = pd.concat([comp_k_table, seq_k_table], ignore_index=True)
    k_table.to_csv(RESULTS / "05_typology/gold_typology_k_selection.csv", index=False)

    cluster_cols = ["review_id", "source_id", "nur_genre", "work_id", "author_id",
                     "rating", "num_sentences", "review_num_words",
                     "cluster_composition", "cluster_sequence"]
    df[cluster_cols].to_csv(RESULTS / "05_typology/gold_typology_clusters.csv", index=False)

    profile = cluster_profile(df, "cluster_composition", n)
    profile.to_csv(RESULTS / "05_typology/gold_typology_cluster_profile.csv", index=False)

    reps = representative_reviews(df, X_comp, km_comp)
    reps.to_csv(RESULTS / "05_typology/gold_typology_representative_reviews.csv", index=False)

    crosstab = pd.crosstab(df["cluster_composition"], df["cluster_sequence"])
    crosstab.to_csv(RESULTS / "05_typology/gold_typology_crosstab.csv")
    ari = adjusted_rand_score(df["cluster_composition"], df["cluster_sequence"])

    print(f"chosen k: compositional={k_comp}, sequence={k_seq}")
    print()
    print("k selection:")
    print(k_table.to_string(index=False))
    print()
    print(f"compositional x sequence cross-tab (ARI = {ari:.3f}):")
    print(crosstab.to_string())
    print()
    print("cluster profile (compositional):")
    show_cols = ["cluster", "n_reviews", "pct_of_corpus", "mean_num_sentences", "median_num_sentences",
                 "mean_rating", "top_genre", "top_genre_pct"] + \
                [f"mean_prop__{l}" for l in MAIN_LABELS_USED]
    print(profile[show_cols].to_string(index=False))
    print()
    print(f"wrote results to {RESULTS} (gold_typology_*.csv)")


def compute_review_level_positions(review_ids, labels, onset_marginal):
    """For each review and main label, the position of that label's *first*
    sentence in that review (onset, not mean -- see module docstring); corpus
    onset marginal when the review doesn't contain the label at all (neutral
    imputation -- 'no information' rather than an arbitrary constant like 0
    or 0.5)."""
    seq_table = pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
    out = pd.DataFrame({"review_id": review_ids})
    for label in labels:
        sub = seq_table[seq_table[label] == 1].groupby("review_id")["rel_position"].min()
        out[f"seqpos__{label}"] = out["review_id"].map(sub).fillna(onset_marginal.get(label, 0.5))
    return out


if __name__ == "__main__":
    main()
