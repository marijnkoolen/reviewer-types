"""Contiguity: for reviews with multiple sentences of a category, do those
sentences cluster together or scatter across the review? (ANALYSIS_PLAN.md §4,
gold set only.)

For each review x label with k sentences of that label among n sentences
total, the observed number of contiguous runs is compared to a permutation
null (randomly place k active positions among n slots, holding k fixed) --
the null distribution for a given (n, k) is identical for every review that
shares it, so it's simulated once and cached, not per review.

clumpiness index = 1 - (runs - 1) / (k - 1), on a review x label basis:
  1.0 = a single contiguous block (however large)
  0.0 = maximally scattered (each occurrence its own run)

Per-label aggregate significance uses Stouffer's method to combine the
per-review z-scores (independent reviews) into one z/p-value per label,
testing whether the label is, corpus-wide, more contiguous or more scattered
than random placement would produce.

Reviews where k==1 (trivially one run) or k==n (trivially one run, no
placement variability) are descriptive-only and excluded from the
significance test.

Usage: python src/03a_contiguity.py
Writes results/03_contiguity/gold_contiguity_detail.csv and results/03_contiguity/gold_contiguity_label_summary.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SENTENCE_TABLE = ROOT / "results" / "00_data_prep/gold_sentence_sequences.csv"
DETAIL_OUT = ROOT / "results" / "03_contiguity/gold_contiguity_detail.csv"
SUMMARY_OUT = ROOT / "results" / "03_contiguity/gold_contiguity_label_summary.csv"

FEATURE_LABELS = [m for m in MAIN_LABELS if m != "None"] + SUB_LABELS
N_NULL_DRAWS = 1000
RNG_SEED = 0

_null_cache = {}


def count_runs(vec):
    """Number of contiguous runs of 1s in a 1D 0/1 array."""
    if vec[0] == 1:
        runs = 1
    else:
        runs = 0
    runs += int(np.sum((vec[1:] == 1) & (vec[:-1] == 0)))
    return runs


def null_run_distribution(n, k, rng):
    key = (n, k)
    if key not in _null_cache:
        base = np.zeros(n, dtype=np.int64)
        base[:k] = 1
        rand_vals = rng.random((N_NULL_DRAWS, n))
        order = np.argsort(rand_vals, axis=1)
        perm = base[order]  # (N_NULL_DRAWS, n), each row a random placement of k ones
        starts = perm[:, 0].copy()
        transitions = ((perm[:, 1:] == 1) & (perm[:, :-1] == 0)).sum(axis=1)
        runs = starts + transitions
        _null_cache[key] = runs
    return _null_cache[key]


def main():
    df = pd.read_csv(SENTENCE_TABLE)
    rng = np.random.default_rng(RNG_SEED)

    detail_rows = []
    for rid, g in df.sort_values("sent_num").groupby("review_id"):
        n = int(g["num_sentences"].iloc[0])
        for label in FEATURE_LABELS:
            vec = g[label].to_numpy(dtype=np.int64)
            k = int(vec.sum())
            if k == 0:
                continue
            runs = count_runs(vec)
            clumpiness = 1.0 if k == 1 else round(1 - (runs - 1) / (k - 1), 4)
            testable = 1 < k < n
            z = p = np.nan
            if testable:
                null_runs = null_run_distribution(n, k, rng)
                null_mean, null_std = null_runs.mean(), null_runs.std()
                if null_std > 0:
                    z = (runs - null_mean) / null_std
                    dev_obs = abs(runs - null_mean)
                    dev_null = np.abs(null_runs - null_mean)
                    p = (1 + np.sum(dev_null >= dev_obs)) / (N_NULL_DRAWS + 1)
            detail_rows.append({
                "review_id": rid, "label": label, "n_sentences": n, "k": k,
                "runs_observed": runs, "clumpiness": clumpiness,
                "testable": testable, "z_score": z, "p_value": p,
            })

    detail = pd.DataFrame(detail_rows)
    DETAIL_OUT.parent.mkdir(exist_ok=True)
    detail.to_csv(DETAIL_OUT, index=False)

    summary_rows = []
    for label in FEATURE_LABELS:
        sub = detail[detail["label"] == label]
        testable = sub[sub["testable"]]
        n_present = len(sub)
        n_singleton = int((sub["k"] == 1).sum())
        n_full = int(((sub["k"] == sub["n_sentences"]) & (sub["k"] > 1)).sum())
        n_testable = len(testable)
        mean_clump = testable["clumpiness"].mean() if n_testable else float("nan")
        zs = testable["z_score"].dropna().to_numpy()
        if len(zs) > 0:
            stouffer_z = zs.sum() / np.sqrt(len(zs))
            combined_p = 2 * stats.norm.sf(abs(stouffer_z))
        else:
            stouffer_z, combined_p = float("nan"), float("nan")
        summary_rows.append({
            "label": label,
            "n_reviews_present": n_present,
            "n_singleton": n_singleton,
            "n_full_coverage": n_full,
            "n_testable": n_testable,
            "mean_clumpiness": round(mean_clump, 4) if n_testable else float("nan"),
            "stouffer_z": round(stouffer_z, 2) if n_testable else float("nan"),
            "combined_p_value": round(combined_p, 5) if n_testable else float("nan"),
            "interpretation": (
                "more contiguous than random" if n_testable and stouffer_z < -1.96 else
                "more scattered than random" if n_testable and stouffer_z > 1.96 else
                "not distinguishable from random" if n_testable else
                "too few testable reviews"
            ),
        })

    summary = pd.DataFrame(summary_rows).sort_values("mean_clumpiness", ascending=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print(f"wrote {DETAIL_OUT} ({len(detail)} review x label rows)")
    print(f"wrote {SUMMARY_OUT} ({len(summary)} labels)")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
