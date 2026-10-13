"""Contiguity on the silver set (ANALYSIS_PLAN.md §4, Phase C), raw
(uncorrected) -- 466,976 reviews.

Same method as gold's 03a_contiguity.py (observed run-count vs. a permutation
null for the same (n, k), aggregated per label via Stouffer's method), but
restructured for scale: gold looped per-review in Python (fine at 1,152
reviews); at 467K reviews that loop doesn't finish in reasonable time. Here,
the observed run-count for every review x label is computed via a single
vectorized shift+groupby per label (16 total, not 467K x 16), and the
permutation null is still cached per unique (n, k) pair -- looping over
unique (n, k) combinations (typically a few hundred to a couple thousand,
however many distinct review-length x count combinations actually occur),
not over every review.

Caveat worth stating plainly: with hundreds of thousands of review x label
observations feeding each label's combined Stouffer z, statistical
significance stops being an interesting question at this scale -- almost
every label will come back "significant" even for a trivially small effect,
simply because n is enormous. The mean clumpiness index (an effect size, not
a significance test) is the number to actually read; z/p-values are reported
for completeness and consistency with the gold-set version, not because
they're informative here.

Usage: python src/03c_contiguity.py
Writes results/03_contiguity/silver_contiguity_label_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, WEAK_MODEL_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SUMMARY_OUT = RESULTS / "03_contiguity/silver_contiguity_label_summary.csv"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
N_NULL_DRAWS = 1000
RNG_SEED = 0


def observed_runs_matrix(sent):
    """(n_reviews x n_labels) matrix of observed run-counts, one label at a
    time via vectorized shift+groupby -- no per-review Python loop."""
    sent = sent.sort_values(["review_id", "sent_num"])
    grouped = sent.groupby("review_id", sort=False)
    runs = {}
    for label in FEATURE_LABELS:
        prev = grouped[label].shift(fill_value=0)
        starts = (sent[label] == 1) & (prev != 1)
        runs[label] = starts.groupby(sent["review_id"], sort=False).sum()
    return pd.DataFrame(runs)


def null_run_distribution(n, k, rng, cache):
    key = (n, k)
    if key not in cache:
        base = np.zeros(n, dtype=np.int64)
        base[:k] = 1
        rand_vals = rng.random((N_NULL_DRAWS, n))
        order = np.argsort(rand_vals, axis=1)
        perm = base[order]
        starts = perm[:, 0].copy()
        transitions = ((perm[:, 1:] == 1) & (perm[:, :-1] == 0)).sum(axis=1)
        cache[key] = starts + transitions
    return cache[key]


def main():
    sent = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    review = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet").set_index("review_id")

    print("computing observed run counts (vectorized, per label)...")
    runs = observed_runs_matrix(sent)

    counts = review[[f"count__{l}" for l in FEATURE_LABELS]].rename(columns=lambda c: c.replace("count__", ""))
    n_sentences = review["num_sentences"]

    # long-form detail: review_id, label, n, k, runs_observed (k>=1 only)
    detail_parts = []
    for label in FEATURE_LABELS:
        k = counts[label]
        mask = k > 0
        detail_parts.append(pd.DataFrame({
            "label": label, "n": n_sentences[mask].to_numpy(),
            "k": k[mask].to_numpy(), "runs_observed": runs.loc[mask.index[mask], label].to_numpy(),
        }))
    detail = pd.concat(detail_parts, ignore_index=True)
    print(f"  {len(detail)} review x label observations")

    testable = detail[(detail["k"] > 1) & (detail["k"] < detail["n"])].copy()
    print(f"  {len(testable)} testable (1 < k < n)")

    print("computing permutation nulls per unique (n, k)...")
    rng = np.random.default_rng(RNG_SEED)
    cache = {}
    unique_nk = testable[["n", "k"]].drop_duplicates()
    print(f"  {len(unique_nk)} unique (n, k) pairs")
    null_means = {}
    null_stds = {}
    for n, k in unique_nk.itertuples(index=False):
        dist = null_run_distribution(int(n), int(k), rng, cache)
        null_means[(n, k)] = dist.mean()
        null_stds[(n, k)] = dist.std()

    testable["null_mean"] = testable.set_index(["n", "k"]).index.map(null_means)
    testable["null_std"] = testable.set_index(["n", "k"]).index.map(null_stds)
    testable["clumpiness"] = 1 - (testable["runs_observed"] - 1) / (testable["k"] - 1)
    testable["z_score"] = (testable["runs_observed"] - testable["null_mean"]) / testable["null_std"]

    print("aggregating per label...")
    rows = []
    for label, g in testable.groupby("label"):
        z = g["z_score"].dropna()
        stouffer_z = z.sum() / np.sqrt(len(z)) if len(z) else float("nan")
        combined_p = 2 * stats.norm.sf(abs(stouffer_z)) if len(z) else float("nan")
        label_all = detail[detail["label"] == label]
        rows.append({
            "label": label,
            "n_reviews_present": len(label_all),
            "n_singleton": int((label_all["k"] == 1).sum()),
            "n_full_coverage": int(((label_all["k"] == label_all["n"]) & (label_all["k"] > 1)).sum()),
            "n_testable": len(g),
            "mean_clumpiness": round(g["clumpiness"].mean(), 4),
            "stouffer_z": round(stouffer_z, 2),
            "combined_p_value": combined_p,
        })
    summary = pd.DataFrame(rows).sort_values("mean_clumpiness", ascending=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"\nwrote {SUMMARY_OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
