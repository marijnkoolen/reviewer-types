"""Book and author similarity (ANALYSIS_PLAN.md §8), gold set only.

Are reviews of the same book more similar to each other than random review
pairs? Are reviews of different books by the same author more similar than
random pairs? Both measured as cosine similarity between each review's
20-dimension category-proportion vector (7 main + 13 sub, the same feature
set occurrence/co-occurrence use), tested against a label-permutation null
(shuffle which review belongs to which work/author, recompute, repeat).

platform_1 is kept in both analyses -- its reviews' work_id/author_id are
still meaningful even though its reviewer identity isn't (see ANALYSIS_PLAN.md
§0). Two reviews with an all-zero category vector (fully unlabeled) are
excluded throughout since cosine similarity is undefined for a zero vector.

Usage: python src/07a_book_author_similarity.py
Writes results/07_book_author_similarity/gold_book_similarity_test.csv, gold_book_similarity_per_work.csv,
results/07_book_author_similarity/gold_author_similarity_test.csv, gold_author_similarity_per_author.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS  # noqa: E402
from similarity import cosine_distance_matrix, pairwise_group_similarity_test  # noqa: E402

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
    n_dropped = (~nonzero).sum()
    print(f"dropping {n_dropped} review(s) with an all-zero category vector "
          f"(cosine similarity undefined)")
    return df.loc[nonzero].reset_index(drop=True), X[nonzero]


def per_group_summary(df, X, group_col, min_size=2, extra_cols=()):
    counts = df[group_col].value_counts()
    groups = counts[counts >= min_size].index
    rows = []
    for g in groups:
        idx = df.index[df[group_col] == g].to_numpy()
        Xg = X[idx]
        D = cosine_distance_matrix(Xg)
        n = len(idx)
        sims = (1 - D)[np.triu_indices(n, k=1)]
        row = {group_col: g, "n_reviews": n, "mean_similarity": round(sims.mean(), 4),
               "min_similarity": round(sims.min(), 4), "max_similarity": round(sims.max(), 4)}
        for c in extra_cols:
            row[c] = df.loc[idx[0], c]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_similarity", ascending=False)


def main():
    df, X = load_matrix()

    # --- books ---
    work_counts = df["work_id"].value_counts()
    multi_work_ids = work_counts[work_counts >= 2].index
    book_mask = df["work_id"].isin(multi_work_ids).to_numpy()
    df_book = df.loc[book_mask].reset_index(drop=True)
    X_book = X[book_mask]
    D_book = cosine_distance_matrix(X_book)

    book_test = pairwise_group_similarity_test(
        D_book, df_book["work_id"].to_numpy(), n_perm=N_PERM, seed=RNG_SEED)
    pd.DataFrame([book_test]).to_csv(RESULTS / "07_book_author_similarity/gold_book_similarity_test.csv", index=False)

    book_summary = per_group_summary(df_book, X_book, "work_id", extra_cols=["nur_genre"])
    book_summary.to_csv(RESULTS / "07_book_author_similarity/gold_book_similarity_per_work.csv", index=False)

    # --- authors (same author, different book) ---
    author_counts = df["author_id"].value_counts()
    multi_author_ids = author_counts[author_counts >= 2].index
    author_mask = df["author_id"].isin(multi_author_ids).to_numpy()
    df_auth = df.loc[author_mask].reset_index(drop=True)
    X_auth = X[author_mask]
    D_auth = cosine_distance_matrix(X_auth)

    work_ids = df_auth["work_id"].to_numpy()
    diff_work = work_ids[:, None] != work_ids[None, :]

    author_test = pairwise_group_similarity_test(
        D_auth, df_auth["author_id"].to_numpy(), pair_filter=diff_work, n_perm=N_PERM, seed=RNG_SEED)
    pd.DataFrame([author_test]).to_csv(RESULTS / "07_book_author_similarity/gold_author_similarity_test.csv", index=False)

    # per-author summary restricted to authors with >=2 DISTINCT works
    distinct_works = df.groupby("author_id")["work_id"].nunique()
    multi_book_authors = distinct_works[distinct_works >= 2].index
    df_auth_mw = df[df["author_id"].isin(multi_book_authors)].reset_index(drop=True)
    X_auth_mw = X[df["author_id"].isin(multi_book_authors).to_numpy()]
    author_summary = per_group_summary(df_auth_mw, X_auth_mw, "author_id")
    author_summary["n_distinct_works"] = author_summary["author_id"].map(distinct_works)
    author_summary.to_csv(RESULTS / "07_book_author_similarity/gold_author_similarity_per_author.csv", index=False)

    print("\n=== book similarity (same work vs. random pairs) ===")
    for k, v in book_test.items():
        print(f"  {k}: {v}")
    print(f"\n{len(multi_work_ids)} works with >=2 reviews, {len(df_book)} reviews total")
    print("\nmost internally-consistent books (top 5):")
    print(book_summary.head(5).to_string(index=False))
    print("\nmost internally-diverse books (bottom 5, n>=3):")
    print(book_summary[book_summary["n_reviews"] >= 3].tail(5).to_string(index=False))

    print("\n=== author similarity (same author, different book, vs. random pairs) ===")
    for k, v in author_test.items():
        print(f"  {k}: {v}")
    print(f"\n{len(multi_book_authors)} authors with >=2 distinct reviewed works, {len(df_auth_mw)} reviews total")
    print("\nmost internally-consistent authors (top 5):")
    print(author_summary.head(5).to_string(index=False))

    print(f"\nwrote results to {RESULTS} (gold_book_similarity_*, gold_author_similarity_*)")


if __name__ == "__main__":
    main()
