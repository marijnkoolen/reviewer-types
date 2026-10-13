"""Order on the silver set (ANALYSIS_PLAN.md §5, Phase C), raw (uncorrected)
-- 466,973 reviews, main categories only (same scope reasoning as gold's
04a_order.py). Restructured for scale versus the gold version:

- Positional density: gold bootstraps the mean, fine at a few thousand
  positions per label. Several silver labels have hundreds of thousands to
  millions of positions -- at that size a bootstrap distribution of a mean
  is, by the central limit theorem, indistinguishable from the analytic
  normal approximation (mean ± z * SE), which is what's used here instead of
  resampling throughout, including the three refinements below.
- Transition structure: gold looped per-review in Python, building each
  review's local transition matrix and accumulating. At 467K reviews /
  ~4.7M within-review consecutive sentence pairs, that loop doesn't finish in
  reasonable time. Vectorized instead: build a "from" matrix (every sentence
  except each review's last) and a "to" matrix (the sentence immediately
  following each "from" row, using a same-review shift check to never cross
  a review boundary), then one matrix multiply (from.T @ to) gives the exact
  same transition-count matrix the nested loop would produce.

Same three refinements ported from 04a_order.py (chat, 2026-09-22), adapted to
analytic (non-bootstrap) form for scale:

1. Non-independence in the significance test. A label's positions within one
   review are correlated, not independent draws -- pooling them naively (as
   the plain SE = std/sqrt(n) above does) understates the CI for
   high-multiplicity labels. Gold fixes this with a cluster bootstrap over
   review_id; at silver's scale (up to ~460K reviews for one label) that
   bootstrap doesn't finish either, so the analytic equivalent is used
   instead: a cluster-robust ("sandwich") SE on the ratio-of-sums mean,
   Var = (G/(G-1)) * sum_g(y_g - n_g*mean)^2 / N^2 over review-groups g, the
   same sandwich estimator behind `vce(cluster)` for a clustered mean -- it
   converges to the same answer the cluster bootstrap would give, without
   resampling.
2. Tail mass and onset, not just the mean. A label that recurs many times
   per review has its own instances averaged toward the center of wherever
   they span, pulling the *pooled* mean toward 0.5 regardless of how
   early/late the label's usage actually is. Two statistics that aren't
   compressed this way: the fraction of a label's sentences in the
   first/last fifth of the review, and the position of each review's first
   and last sentence carrying the label (silver_order_onset.csv) -- one
   value per review regardless of recurrence.
3. Length stratification. Positional density is recomputed within
   review-length quartiles (silver_order_position_by_length.csv) as a
   robustness check on whether a label's early/late direction holds across
   review lengths or is driven by one length regime.

Usage: python src/04c_order.py
Writes results/04_order/silver_order_position.csv (pooled positional density,
cluster-robust CI, tail-mass columns), results/04_order/silver_order_onset.csv
(per-review first/last occurrence position), results/
silver_order_position_by_length.csv (positional density within review-length
quartiles), results/04_order/silver_order_transitions.csv, and results/
silver_order_transition_matrix.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, WEAK_MODEL_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
POSITION_OUT = RESULTS / "04_order/silver_order_position.csv"
ONSET_OUT = RESULTS / "04_order/silver_order_onset.csv"
LENGTH_STRATIFIED_OUT = RESULTS / "04_order/silver_order_position_by_length.csv"
TRANSITIONS_OUT = RESULTS / "04_order/silver_order_transitions.csv"
TRANSITION_MATRIX_OUT = RESULTS / "04_order/silver_order_transition_matrix.csv"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
Z_95 = 1.96
FRONT_BACK_THRESHOLD = 0.2  # "front fifth" / "back fifth" of the review
LENGTH_QUARTILE_LABELS = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]


def cluster_robust_mean_ci(group_sums, group_sizes, z=Z_95):
    """Analytic cluster-robust ('sandwich') CI on the ratio-of-sums mean --
    the closed-form limit of gold's cluster bootstrap, scalable to silver's
    review counts. Degenerates to a point CI when there's only one cluster
    (no between-review information)."""
    n_groups = len(group_sums)
    n_total = group_sizes.sum()
    mean_pos = group_sums.sum() / n_total
    if n_groups <= 1:
        return mean_pos, mean_pos, mean_pos
    residuals = group_sums - group_sizes * mean_pos
    var = (n_groups / (n_groups - 1)) * np.sum(residuals ** 2) / (n_total ** 2)
    se = np.sqrt(var)
    return mean_pos, mean_pos - z * se, mean_pos + z * se


def normal_mean_ci(values, z=Z_95):
    """Plain normal-approximation CI for statistics already one-value-per-
    review (e.g. onset position), where no within-review clustering applies."""
    n = len(values)
    mean_val = values.mean()
    if n <= 1:
        return mean_val, mean_val, mean_val
    se = values.std(ddof=1) / np.sqrt(n)
    return mean_val, mean_val - z * se, mean_val + z * se


def positional_density(df):
    multi_sentence = df[df["num_sentences"] > 1]
    rows = []
    for label in FEATURE_LABELS:
        active = multi_sentence.loc[multi_sentence[label] == 1]
        n = len(active)
        if n == 0:
            continue
        positions = active["rel_position"].to_numpy()
        grouped = active.groupby("review_id")["rel_position"].agg(["sum", "count"])
        mean_pos, ci_lo, ci_hi = cluster_robust_mean_ci(
            grouped["sum"].to_numpy(), grouped["count"].to_numpy()
        )
        median_pos = np.median(positions)
        deviates = not (ci_lo <= 0.5 <= ci_hi)
        rows.append({
            "label": label, "level": "main" if label in MAIN_LABELS else "sub",
            "n_sentences": n, "n_reviews_with_label": len(grouped),
            "mean_sentences_per_review": round(n / len(grouped), 2),
            "mean_rel_position": round(mean_pos, 4),
            "median_rel_position": round(median_pos, 4),
            "std_rel_position": round(positions.std(), 4),
            "frac_front_fifth": round((positions < FRONT_BACK_THRESHOLD).mean(), 4),
            "frac_back_fifth": round((positions > 1 - FRONT_BACK_THRESHOLD).mean(), 4),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "deviates_from_center": deviates,
            "direction": ("early" if mean_pos < 0.5 else "late") if deviates else "none",
        })
    out = pd.DataFrame(rows).sort_values("mean_rel_position")
    out.to_csv(POSITION_OUT, index=False)
    return out


def onset_analysis(df):
    """Per label, the position of each review's *first* and *last* sentence
    carrying that label -- one value per review regardless of recurrence, so
    unaffected by the multiplicity-driven compression toward 0.5 the pooled
    mean shows."""
    multi_sentence = df[df["num_sentences"] > 1]
    rows = []
    for label in FEATURE_LABELS:
        active = multi_sentence.loc[multi_sentence[label] == 1]
        if active.empty:
            continue
        span = active.groupby("review_id")["rel_position"].agg(["min", "max"])
        first = span["min"].to_numpy()
        last = span["max"].to_numpy()
        first_mean, first_ci_lo, first_ci_hi = normal_mean_ci(first)
        last_mean, last_ci_lo, last_ci_hi = normal_mean_ci(last)
        rows.append({
            "label": label, "level": "main" if label in MAIN_LABELS else "sub",
            "n_reviews": len(span),
            "first_mean_position": round(first_mean, 4),
            "first_median_position": round(np.median(first), 4),
            "first_ci_lo": round(first_ci_lo, 4), "first_ci_hi": round(first_ci_hi, 4),
            "last_mean_position": round(last_mean, 4),
            "last_median_position": round(np.median(last), 4),
            "last_ci_lo": round(last_ci_lo, 4), "last_ci_hi": round(last_ci_hi, 4),
        })
    out = pd.DataFrame(rows).sort_values("first_mean_position")
    out.to_csv(ONSET_OUT, index=False)
    return out


def positional_density_by_length(df):
    """Positional density recomputed within review-length quartiles, as a
    robustness check on whether a label's early/late direction holds across
    review lengths or is an artifact of pooling very different review
    lengths together."""
    multi_sentence = df[df["num_sentences"] > 1].copy()
    review_lengths = multi_sentence.groupby("review_id")["num_sentences"].first()
    quartile = pd.qcut(review_lengths, 4, labels=LENGTH_QUARTILE_LABELS, duplicates="drop")
    multi_sentence["length_quartile"] = multi_sentence["review_id"].map(quartile)

    bounds = review_lengths.groupby(quartile, observed=True).agg(["min", "max", "count"])

    rows = []
    for q in quartile.cat.categories:
        qdf = multi_sentence[multi_sentence["length_quartile"] == q]
        if qdf.empty:
            continue
        for label in FEATURE_LABELS:
            active = qdf.loc[qdf[label] == 1]
            n = len(active)
            if n == 0:
                continue
            grouped = active.groupby("review_id")["rel_position"].agg(["sum", "count"])
            mean_pos, ci_lo, ci_hi = cluster_robust_mean_ci(
                grouped["sum"].to_numpy(), grouped["count"].to_numpy()
            )
            deviates = not (ci_lo <= 0.5 <= ci_hi)
            rows.append({
                "length_quartile": q,
                "sentences_range": f"{bounds.loc[q, 'min']}-{bounds.loc[q, 'max']}",
                "label": label, "level": "main" if label in MAIN_LABELS else "sub",
                "n_sentences": n, "n_reviews_with_label": len(grouped),
                "mean_rel_position": round(mean_pos, 4),
                "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
                "deviates_from_center": deviates,
                "direction": ("early" if mean_pos < 0.5 else "late") if deviates else "none",
            })
    out = pd.DataFrame(rows)
    out.to_csv(LENGTH_STRATIFIED_OUT, index=False)
    return out, bounds


def transition_structure(df):
    sent_sorted = df.sort_values(["review_id", "sent_num"]).reset_index(drop=True)
    same_next = (sent_sorted["review_id"] == sent_sorted["review_id"].shift(-1)).to_numpy()

    labels_mat = sent_sorted[MAIN_LABELS_USED].to_numpy(dtype=np.int64)
    from_mat = labels_mat[same_next]
    to_mat = labels_mat[1:][same_next[:-1]]  # row immediately after each 'from' row

    counts = from_mat.T @ to_mat
    n_labels = len(MAIN_LABELS_USED)

    total = counts.sum()
    from_marginal = counts.sum(axis=1) / total
    to_marginal = counts.sum(axis=0) / total
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)

    rows = []
    for i, a in enumerate(MAIN_LABELS_USED):
        for j, b in enumerate(MAIN_LABELS_USED):
            observed_p = counts[i, j] / total if total else 0.0
            expected_p = from_marginal[i] * to_marginal[j]
            pmi = np.log2(observed_p / expected_p) if observed_p > 0 and expected_p > 0 else float("nan")
            rows.append({
                "label_from": a, "label_to": b, "count": int(counts[i, j]),
                "transition_prob": round(probs[i, j], 4), "pmi": round(pmi, 4) if not np.isnan(pmi) else "",
            })
    long_out = pd.DataFrame(rows).sort_values("pmi", ascending=False)
    long_out.to_csv(TRANSITIONS_OUT, index=False)

    wide = pd.DataFrame(counts, index=MAIN_LABELS_USED, columns=MAIN_LABELS_USED)
    wide.to_csv(TRANSITION_MATRIX_OUT)
    return long_out, wide


def main():
    df = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")

    pos = positional_density(df)
    print(f"wrote {POSITION_OUT} ({len(pos)} labels)")
    print(pos.to_string(index=False))

    onset = onset_analysis(df)
    print(f"\nwrote {ONSET_OUT} ({len(onset)} labels)")
    print(onset.to_string(index=False))

    by_length, bounds = positional_density_by_length(df)
    print(f"\nwrote {LENGTH_STRATIFIED_OUT} ({len(by_length)} label x length-quartile rows)")
    print("length quartile boundaries (num_sentences min-max, n reviews):")
    print(bounds.to_string())
    print("\nmain-category mean position by length quartile (direction flips flagged):")
    pivot = by_length[by_length["level"] == "main"].pivot(
        index="label", columns="length_quartile", values="mean_rel_position"
    )
    print(pivot.to_string())

    trans, wide = transition_structure(df)
    print(f"\nwrote {TRANSITIONS_OUT} ({len(trans)} label-pair transitions)")
    print(f"wrote {TRANSITION_MATRIX_OUT} ({wide.shape[0]}x{wide.shape[1]} matrix)")
    print("\nTransitions ranked by PMI:")
    print(trans.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
