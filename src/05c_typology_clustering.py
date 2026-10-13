"""Review typology on the silver set (ANALYSIS_PLAN.md §6, Phase C), raw
(uncorrected) -- 466,976 reviews. Same two-clustering design as gold's
05a_typology_clustering.py (compositional + sequence-based, cross-checked),
restructured for scale:

- `silhouette_score` is O(n^2) by default -- infeasible at 467K rows (2x10^11
  pairs). Uses sklearn's `sample_size` parameter to compute it on a random
  10,000-row subsample instead, standard practice at this scale.
- Bootstrap stability resampled at n=n_reviews per draw in gold (fine at
  1,152). Here each draw subsamples a fixed 20,000 rows instead of the full
  467K before refitting -- still measures the same thing (does a
  different-but-comparable sample of the data produce a compatible
  clustering, judged by ARI against the full-data fit), just bounded cost.
- No sentence text available in the silver predictions file (unlike gold's
  raw annotation TSV), so representative reviews are listed by
  metadata only, no example sentence.
- Sequence-clustering feature (2026-09-22): onset (first-occurrence) position
  per label per review, not mean position -- see gold's module docstring for
  the full prototyping rationale (cleaner signal, lower length confound,
  length-stratified imputation tested and rejected).

Usage: python src/05c_typology_clustering.py
Writes results/05_typology/silver_typology_k_selection.csv, silver_typology_clusters.csv,
silver_typology_cluster_profile.csv, silver_typology_representative_reviews.csv,
silver_typology_crosstab.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, WEAK_MODEL_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
ALL_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
K_RANGE = range(2, 9)
N_BOOTSTRAP = 20
BOOTSTRAP_SAMPLE_SIZE = 20_000
SILHOUETTE_SAMPLE_SIZE = 10_000
RNG_SEED = 0


def select_k(X, label):
    rng = np.random.default_rng(RNG_SEED)
    n = X.shape[0]
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=RNG_SEED)

        aris = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=BOOTSTRAP_SAMPLE_SIZE)
            km_boot = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X[idx])
            boot_labels_on_full = km_boot.predict(X)
            aris.append(adjusted_rand_score(km.labels_, boot_labels_on_full))
        stability = float(np.mean(aris))

        rows.append({"clustering": label, "k": k, "silhouette": round(sil, 4),
                      "bootstrap_ari_stability": round(stability, 4)})
        print(f"  {label} k={k}: silhouette={sil:.4f}, stability={stability:.4f}")
    return pd.DataFrame(rows)


def pick_best_k(k_table):
    stable = k_table[k_table["bootstrap_ari_stability"] >= 0.5]
    pool = stable if len(stable) else k_table
    return int(pool.loc[pool["silhouette"].idxmax(), "k"])


def representative_reviews(df, X, km, id_col="review_id", n_per_cluster=5):
    rows = []
    for c in sorted(set(km.labels_)):
        mask = km.labels_ == c
        centroid = km.cluster_centers_[c]
        dists = np.linalg.norm(X[mask] - centroid, axis=1)
        closest_idx = np.argsort(dists)[:n_per_cluster]
        sub = df.loc[mask].iloc[closest_idx]
        for _, row in sub.iterrows():
            rows.append({
                "cluster": c, "review_id": row[id_col], "nur_genre": row["nur_genre"],
                "source_id": row["source_id"], "review_num_words": row["review_num_words"],
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


def compute_review_level_positions(review_ids, labels, onset_marginal):
    """Onset (first-occurrence), not mean, position per review per label --
    see gold's 05a_typology_clustering.py module docstring for why (prototyped
    on gold before rollout: cleaner signal, lower length confound, and
    length-stratified imputation rejected as counterproductive)."""
    seq_table = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    out = pd.DataFrame({"review_id": review_ids})
    for label in labels:
        sub = seq_table[seq_table[label] == 1].groupby("review_id")["rel_position"].min()
        out[f"seqpos__{label}"] = out["review_id"].map(sub).fillna(onset_marginal.get(label, 0.5))
    return out


def main():
    df = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    n = len(df)
    print(f"n reviews: {n}")

    # --- 1. compositional clustering ---
    comp_cols = [f"prop__{label}" for label in MAIN_LABELS_USED]
    comp_features = df[comp_cols].copy()
    # unlabeled proportion: (num_sentences - any-label-active count) isn't directly
    # stored for silver; approximate via 1 - max coverage isn't right either, so
    # use the fraction of sentences with NO main label active, computed once here
    sent = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    any_main = sent[MAIN_LABELS_USED].sum(axis=1) > 0
    unlabeled_per_review = (~any_main).groupby(sent["review_id"]).mean()
    comp_features["prop__unlabeled"] = df["review_id"].map(unlabeled_per_review).fillna(0.0).to_numpy()

    X_comp = StandardScaler().fit_transform(comp_features.to_numpy())

    print("selecting k (compositional)...")
    comp_k_table = select_k(X_comp, "compositional")
    k_comp = pick_best_k(comp_k_table)
    km_comp = KMeans(n_clusters=k_comp, n_init=10, random_state=RNG_SEED).fit(X_comp)
    df["cluster_composition"] = km_comp.labels_

    # --- 2. sequence-based clustering ---
    onset_marginal = pd.read_csv(RESULTS / "04_order/silver_order_onset.csv").set_index("label")["first_mean_position"]
    seq_pos_df = compute_review_level_positions(df["review_id"].tolist(), MAIN_LABELS_USED, onset_marginal)
    df = df.merge(seq_pos_df, on="review_id", how="left")
    X_seq = StandardScaler().fit_transform(df[[f"seqpos__{l}" for l in MAIN_LABELS_USED]].to_numpy())

    print("selecting k (sequence)...")
    seq_k_table = select_k(X_seq, "sequence")
    k_seq = pick_best_k(seq_k_table)
    km_seq = KMeans(n_clusters=k_seq, n_init=10, random_state=RNG_SEED).fit(X_seq)
    df["cluster_sequence"] = km_seq.labels_

    k_table = pd.concat([comp_k_table, seq_k_table], ignore_index=True)
    k_table.to_csv(RESULTS / "05_typology/silver_typology_k_selection.csv", index=False)

    cluster_cols = ["review_id", "source_id", "nur_genre", "work_id", "author_id",
                     "rating", "num_sentences", "review_num_words",
                     "cluster_composition", "cluster_sequence"]
    df[cluster_cols].to_csv(RESULTS / "05_typology/silver_typology_clusters.csv", index=False)

    profile = cluster_profile(df, "cluster_composition", n)
    profile.to_csv(RESULTS / "05_typology/silver_typology_cluster_profile.csv", index=False)

    reps = representative_reviews(df, X_comp, km_comp)
    reps.to_csv(RESULTS / "05_typology/silver_typology_representative_reviews.csv", index=False)

    crosstab = pd.crosstab(df["cluster_composition"], df["cluster_sequence"])
    crosstab.to_csv(RESULTS / "05_typology/silver_typology_crosstab.csv")
    ari = adjusted_rand_score(df["cluster_composition"], df["cluster_sequence"])

    print(f"\nchosen k: compositional={k_comp}, sequence={k_seq}")
    print("\nk selection:")
    print(k_table.to_string(index=False))
    print(f"\ncompositional x sequence cross-tab (ARI = {ari:.3f}):")
    print(crosstab.to_string())
    print("\ncluster profile (compositional):")
    show_cols = ["cluster", "n_reviews", "pct_of_corpus", "mean_num_sentences", "median_num_sentences",
                 "mean_rating", "top_genre", "top_genre_pct"] + \
                [f"mean_prop__{l}" for l in MAIN_LABELS_USED]
    print(profile[show_cols].to_string(index=False))
    print(f"\nwrote results to {RESULTS} (silver_typology_*.csv)")


if __name__ == "__main__":
    main()
