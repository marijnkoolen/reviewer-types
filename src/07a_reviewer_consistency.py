"""Reviewer consistency and variance partitioning (ANALYSIS_PLAN.md §9), gold
set only.

Do reviewers who write multiple reviews tend to write the same kind of
review? Same cosine-similarity-vs-permutation-null design as §8's book/author
tests. platform_1 is excluded here (and only here) since its `user_id` is a
placeholder with no real reviewer signal (ANALYSIS_PLAN.md §0).

Then: a comparative one-way PERMANOVA (McArdle & Anderson 2001, via
src/common/similarity.py) for reviewer, work, author, genre, and platform
separately, each on its own natural subset -- this is a simplification of
the plan's suggested joint multi-factor PERMANOVA (implementing a proper
joint/nested design by hand was more than this pass warranted), so R^2
values are independently estimated, not a variance decomposition that sums
to 100%, and aren't perfectly comparable given very different group counts
and sample sizes per factor -- but they do give a real, directly interpretable
answer to "how much of a review's category composition does each of these
explain on its own."

Usage: python src/07a_reviewer_consistency.py
Writes results/07_book_author_similarity/gold_reviewer_similarity_test.csv,
results/07_book_author_similarity/gold_reviewer_similarity_per_reviewer.csv,
results/07_book_author_similarity/gold_variance_partitioning.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS  # noqa: E402
from similarity import cosine_distance_matrix, pairwise_group_similarity_test, oneway_permanova  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = MAIN_LABELS_USED + SUB_LABELS
N_PERM = 999
RNG_SEED = 0


def load_matrix():
    df = pd.read_csv(RESULTS / "00_data_prep/gold_review_features.csv")
    X = df[[f"prop__{l}" for l in FEATURE_LABELS]].to_numpy()
    nonzero = X.sum(axis=1) > 0
    return df.loc[nonzero].reset_index(drop=True), X[nonzero]


def per_group_summary(df, X, group_col, min_size=2):
    counts = df[group_col].value_counts()
    groups = counts[counts >= min_size].index
    rows = []
    for g in groups:
        idx = df.index[df[group_col] == g].to_numpy()
        Xg = X[idx]
        D = cosine_distance_matrix(Xg)
        n = len(idx)
        sims = (1 - D)[np.triu_indices(n, k=1)]
        rows.append({group_col: g, "n_reviews": n, "mean_similarity": round(sims.mean(), 4)})
    return pd.DataFrame(rows).sort_values("mean_similarity", ascending=False)


def main():
    df, X = load_matrix()

    # --- reviewer consistency (platform_1 excluded) ---
    df_r = df[df["source_id"] != "platform_1"].reset_index(drop=True)
    X_r = X[(df["source_id"] != "platform_1").to_numpy()]
    user_counts = df_r["user_id"].value_counts()
    multi_user_ids = user_counts[user_counts >= 2].index
    rev_mask = df_r["user_id"].isin(multi_user_ids).to_numpy()
    df_rev = df_r.loc[rev_mask].reset_index(drop=True)
    X_rev = X_r[rev_mask]
    D_rev = cosine_distance_matrix(X_rev)

    reviewer_test = pairwise_group_similarity_test(
        D_rev, df_rev["user_id"].to_numpy(), n_perm=N_PERM, seed=RNG_SEED)
    pd.DataFrame([reviewer_test]).to_csv(RESULTS / "07_book_author_similarity/gold_reviewer_similarity_test.csv", index=False)

    reviewer_summary = per_group_summary(df_rev, X_rev, "user_id")
    reviewer_summary.to_csv(RESULTS / "07_book_author_similarity/gold_reviewer_similarity_per_reviewer.csv", index=False)

    print("=== reviewer similarity (same reviewer vs. random pairs, platform_1 excluded) ===")
    for k, v in reviewer_test.items():
        print(f"  {k}: {v}")
    print(f"\n{len(multi_user_ids)} reviewers with >=2 reviews, {len(df_rev)} reviews total")
    print("\nmost self-consistent reviewers (top 5):")
    print(reviewer_summary.head(5).to_string(index=False))

    # --- comparative variance partitioning ---
    D_full = cosine_distance_matrix(X)
    rows = []

    work_counts = df["work_id"].value_counts()
    book_mask = df["work_id"].isin(work_counts[work_counts >= 2].index).to_numpy()
    res = oneway_permanova(cosine_distance_matrix(X[book_mask]), df.loc[book_mask, "work_id"].to_numpy(),
                            n_perm=N_PERM, seed=RNG_SEED)
    rows.append({"factor": "work (book)", "subset": "reviews of works with >=2 reviews", **res})

    author_counts = df["author_id"].value_counts()
    auth_mask = df["author_id"].isin(author_counts[author_counts >= 2].index).to_numpy()
    res = oneway_permanova(cosine_distance_matrix(X[auth_mask]), df.loc[auth_mask, "author_id"].to_numpy(),
                            n_perm=N_PERM, seed=RNG_SEED)
    rows.append({"factor": "author", "subset": "reviews by authors with >=2 reviews", **res})

    res = oneway_permanova(D_rev, df_rev["user_id"].to_numpy(), n_perm=N_PERM, seed=RNG_SEED)
    rows.append({"factor": "reviewer", "subset": "reviews by reviewers with >=2 reviews, excl. platform_1", **res})

    res = oneway_permanova(D_full, df["nur_genre"].to_numpy(), n_perm=N_PERM, seed=RNG_SEED)
    rows.append({"factor": "genre", "subset": "all reviews", **res})

    res = oneway_permanova(D_full, df["source_id"].to_numpy(), n_perm=N_PERM, seed=RNG_SEED)
    rows.append({"factor": "platform", "subset": "all reviews", **res})

    variance_table = pd.DataFrame(rows).sort_values("r2", ascending=False)
    variance_table.to_csv(RESULTS / "07_book_author_similarity/gold_variance_partitioning.csv", index=False)

    print("\n=== comparative variance partitioning (one-way PERMANOVA per factor) ===")
    print(variance_table.to_string(index=False))
    print(f"\nwrote results to {RESULTS} (gold_reviewer_similarity_*, gold_variance_partitioning.csv)")


if __name__ == "__main__":
    main()
