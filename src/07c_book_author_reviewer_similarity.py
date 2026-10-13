"""Book, author, and reviewer similarity on the silver set (ANALYSIS_PLAN.md
§8-9, Phase C item 5), raw (uncorrected) -- the one Phase A section with no
Phase C counterpart until now.

Sizing check (done first, per the plan) found the naive approach from
07a_book_author_similarity.py / 07a_reviewer_consistency.py doesn't transfer:
at gold scale "multi-review work/author/reviewer" was a small, convenient
subset (85 works, 122 authors, 51 reviewers). At silver scale it's most of
the corpus -- 37,212 works (455,908 reviews, 98% of the corpus), 14,126
authors (464,002 reviews, 99%), 30,288 reviewers excl. platform_1 (338,139
reviews, 72%) -- and some individual groups are enormous (the single most
prolific author has 54,889 reviews). A dense N x N distance matrix over that
is many GB before computing anything.

Uses the scalable machinery in src/common/similarity.py instead
(`scalable_group_similarity_test`): within-group similarity via block-diagonal
per-group matrices (capped at 100 members/group -- exact under the cap, an
unbiased random-subsample estimate over it, so one huge group can't blow up
compute), a cluster bootstrap (resample groups, not individual pairs, since
within-group pairs aren't independent of each other) for the CI, and the
random-pairs baseline from a large *sample* of pairs rather than an exhaustive
enumeration (infeasible: ~10^11 pairs in the full corpus). Validated against
the old dense method on gold's own book/author subsets before trusting it here
(chat, 2026-08-19) -- exact agreement on the point estimate, same substantive
conclusion (significant/not) on both tests, z-scores differ modestly since the
variance estimators differ (cluster bootstrap + baseline SE vs. label
permutation), not a discrepancy in what's being measured.

Comparative variance-partitioning (PERMANOVA, gold's §9) is NOT redone here --
`oneway_permanova` has the same dense-matrix problem and would need its own
scalable redesign (SS_total via sampling, SS_within via the capped per-group
approach already built here); scoped out as follow-up work, not attempted
this pass so as not to ship a rushed implementation of a statistic this
central.

Usage: python src/07c_book_author_reviewer_similarity.py
Writes results/07_book_author_similarity/silver_book_similarity_test.csv, silver_book_similarity_per_work.csv,
results/07_book_author_similarity/silver_author_similarity_test.csv, silver_author_similarity_per_author.csv,
results/07_book_author_similarity/silver_reviewer_similarity_test.csv, silver_reviewer_similarity_per_reviewer.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from similarity import scalable_group_similarity_test  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
CAP = 100
N_BOOT = 2000
N_BASELINE_SAMPLE = 1_000_000
RNG_SEED = 0


def load_matrix():
    df = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    X = df[[f"prop__{l}" for l in FEATURE_LABELS]].to_numpy()
    nonzero = X.sum(axis=1) > 0
    n_dropped = (~nonzero).sum()
    print(f"dropping {n_dropped} review(s) with an all-zero category vector")
    return df.loc[nonzero].reset_index(drop=True), X[nonzero]


def run_test(df, X, group_col, label, min_size=2, pair_filter_col=None, source_id_exclude=None):
    d = df
    x = X
    if source_id_exclude is not None:
        mask = (df["source_id"] != source_id_exclude).to_numpy()
        d, x = df.loc[mask].reset_index(drop=True), X[mask]

    counts = d[group_col].value_counts()
    multi_ids = counts[counts >= min_size].index
    mask2 = d[group_col].isin(multi_ids).to_numpy()
    d2, x2 = d.loc[mask2].reset_index(drop=True), x[mask2]

    print(f"\n=== {label} ===")
    print(f"{len(multi_ids)} {group_col} groups with >={min_size} reviews, {len(d2)} reviews total")
    test, per_group = scalable_group_similarity_test(
        d2, x2, group_col, min_size=min_size, cap=CAP, pair_filter_col=pair_filter_col,
        n_boot=N_BOOT, n_baseline_sample=N_BASELINE_SAMPLE, seed=RNG_SEED)
    for k, v in test.items():
        print(f"  {k}: {v}")
    return test, per_group.sort_values("mean_similarity", ascending=False)


def main():
    df, X = load_matrix()

    book_test, book_summary = run_test(df, X, "work_id", "book similarity (same work)")
    pd.DataFrame([book_test]).to_csv(RESULTS / "07_book_author_similarity/silver_book_similarity_test.csv", index=False)
    book_summary.to_csv(RESULTS / "07_book_author_similarity/silver_book_similarity_per_work.csv", index=False)

    author_test, author_summary = run_test(
        df, X, "author_id", "author similarity (same author, different book)", pair_filter_col="work_id")
    pd.DataFrame([author_test]).to_csv(RESULTS / "07_book_author_similarity/silver_author_similarity_test.csv", index=False)
    author_summary.to_csv(RESULTS / "07_book_author_similarity/silver_author_similarity_per_author.csv", index=False)

    reviewer_test, reviewer_summary = run_test(
        df, X, "user_id", "reviewer similarity (excl. platform_1)", source_id_exclude="platform_1")
    pd.DataFrame([reviewer_test]).to_csv(RESULTS / "07_book_author_similarity/silver_reviewer_similarity_test.csv", index=False)
    reviewer_summary.to_csv(RESULTS / "07_book_author_similarity/silver_reviewer_similarity_per_reviewer.csv", index=False)

    print(f"\nwrote results to {RESULTS} (silver_book_similarity_*, silver_author_similarity_*, "
          f"silver_reviewer_similarity_*)")


if __name__ == "__main__":
    main()
