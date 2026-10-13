"""Occurrence: prevalence and multiplicity of each (sub)category (ANALYSIS_PLAN.md §2).

Prevalence: % of reviews containing >=1 sentence of a category, with a
bootstrap CI (resampling reviews, not sentences -- reviews are the
independent unit here).

Multiplicity: among reviews that contain the category, how many sentences
carry it -- both as a raw count and as a rate per 100 sentences (the
length-normalized version, since raw counts are confounded with review
length).

Runs on the gold set only. WEAK_MODEL_LABELS is a silver-only exclusion (see
ANALYSIS_PLAN.md §0b) and does not apply here -- gold labels are human
annotations, not model predictions.

Usage: python src/01a_occurrence.py
Writes results/01_occurrence/gold_occurrence.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REVIEW_FEATURES = ROOT / "results" / "00_data_prep/gold_review_features.csv"
OUT_PATH = ROOT / "results" / "01_occurrence/gold_occurrence.csv"

FEATURE_LABELS = [m for m in MAIN_LABELS if m != "None"] + SUB_LABELS
N_BOOTSTRAP = 2000
RNG_SEED = 0


def bootstrap_prevalence_ci(indicator, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(indicator)
    idx = rng.integers(0, n, size=(n_boot, n))
    resampled = indicator[idx]  # shape (n_boot, n)
    boot_prevalence = resampled.mean(axis=1)
    lo, hi = np.percentile(boot_prevalence, [2.5, 97.5])
    return lo, hi


def main():
    df = pd.read_csv(REVIEW_FEATURES)
    n_reviews = len(df)

    rows = []
    for label in FEATURE_LABELS:
        count_col = f"count__{label}"
        counts = df[count_col].to_numpy()
        indicator = (counts > 0).astype(float)

        prevalence = indicator.mean()
        ci_lo, ci_hi = bootstrap_prevalence_ci(indicator)

        with_label = df.loc[counts > 0]
        n_with_label = len(with_label)
        rate_per_100 = (with_label[count_col] / with_label["num_sentences"] * 100)

        rows.append({
            "label": label,
            "level": "main" if label in MAIN_LABELS else "sub",
            "n_reviews_total": n_reviews,
            "n_reviews_with_label": n_with_label,
            "prevalence": round(prevalence, 4),
            "prevalence_ci_lo": round(ci_lo, 4),
            "prevalence_ci_hi": round(ci_hi, 4),
            "multiplicity_mean": round(with_label[count_col].mean(), 2) if n_with_label else float("nan"),
            "multiplicity_median": with_label[count_col].median() if n_with_label else float("nan"),
            "rate_per100_mean": round(rate_per_100.mean(), 2) if n_with_label else float("nan"),
            "rate_per100_median": round(rate_per_100.median(), 2) if n_with_label else float("nan"),
            "sparse_gold_flag": label in SPARSE_GOLD_LABELS,
        })

    out = pd.DataFrame(rows).sort_values("prevalence", ascending=False)
    OUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out)} labels, {n_reviews} reviews, {N_BOOTSTRAP} bootstrap draws/label)")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
