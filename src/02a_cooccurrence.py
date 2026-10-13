"""Co-occurrence between (sub)categories (ANALYSIS_PLAN.md §3), gold set only.

Sentence-level: which categories are marked on the *same sentence*. Trivial
parent-child pairs (e.g. Content / Content--Narrative) are excluded since a
sub-category being active essentially implies its parent is too -- that's
definitional, not an interesting association. Reported via PMI (association
strength relative to chance) and Jaccard (overlap size).

Review-level: which categories appear *somewhere in the same review*, tested
against a null model that controls for the main confound (longer reviews have
more chances to touch every category, inflating co-occurrence). The null is a
length-stratified permutation: bin reviews by sentence count, then within each
bin shuffle each label's presence indicator independently across reviews --
this destroys real within-review association between labels while preserving
each label's prevalence at a given length, so any leftover length effect is
absorbed into the null rather than mistaken for a real association.

Sentence-level by length (chat, 2026-09-22): unlike review-level, the plain
sentence-level PMI above has never been length-conditioned -- it pools every
sentence from every review into one flat matrix regardless of which review
(and how long) it came from. That's a real gap, not a hypothetical one:
09_length_dimension.py's density measure (mean simultaneously-active labels
per sentence) already correlates negatively with review length (rho=-0.38),
so sentences from long and short reviews aren't drawn from the same
co-occurrence-propensity distribution, and the flat pool implicitly weights
toward long reviews (more sentences = more rows). Prototyped before adopting:
splitting by length quartile shows the full-corpus PMI correlates with each
quartile's own PMI at r=0.72 (shortest) rising to r=0.97 (longest) -- the
pooled estimate is disproportionately representative of long reviews' pairs,
exactly the mechanical-weighting concern. The strongest pairs don't flip sign
across quartiles, but several show a real magnitude gradient (stronger
association in longer reviews), not just noise.

Usage: python src/02a_cooccurrence.py
Writes results/02_cooccurrence/gold_cooccurrence_sentence.csv, results/02_cooccurrence/gold_cooccurrence_review.csv,
  results/02_cooccurrence/gold_cooccurrence_sentence_by_length.csv,
  results/02_cooccurrence/gold_cooccurrence_sentence_length_corr.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SUB_TO_MAIN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SENTENCE_TABLE = ROOT / "results" / "00_data_prep/gold_sentence_sequences.csv"
REVIEW_TABLE = ROOT / "results" / "00_data_prep/gold_review_features.csv"
SENT_OUT = ROOT / "results" / "02_cooccurrence/gold_cooccurrence_sentence.csv"
REVIEW_OUT = ROOT / "results" / "02_cooccurrence/gold_cooccurrence_review.csv"
SENT_BY_LENGTH_OUT = ROOT / "results" / "02_cooccurrence/gold_cooccurrence_sentence_by_length.csv"
SENT_LENGTH_CORR_OUT = ROOT / "results" / "02_cooccurrence/gold_cooccurrence_sentence_length_corr.csv"

FEATURE_LABELS = [m for m in MAIN_LABELS if m != "None"] + SUB_LABELS
N_LENGTH_BINS = 10
N_PERMUTATIONS = 500
RNG_SEED = 0
LENGTH_QUARTILE_LABELS = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]
POWERED_THRESHOLD = 20  # min n_both corpus-wide to count a pair as well-powered for stability checks

PARENT_CHILD_PAIRS = {frozenset((sub, main)) for sub, main in SUB_TO_MAIN.items()}


def pairwise_stats(X, labels):
    """X: (n_items, n_labels) binary matrix. Returns long-form co-occurrence table."""
    n = X.shape[0]
    joint = X.T @ X  # joint counts, diagonal = marginal counts
    marginal = np.diag(joint)
    rows = []
    for i, j in combinations(range(len(labels)), 2):
        a, b = labels[i], labels[j]
        if frozenset((a, b)) in PARENT_CHILD_PAIRS:
            continue
        n_both = int(joint[i, j])
        n_a, n_b = int(marginal[i]), int(marginal[j])
        p_a, p_b, p_ab = n_a / n, n_b / n, n_both / n
        pmi = np.log(p_ab / (p_a * p_b)) if n_both > 0 else float("nan")
        union = n_a + n_b - n_both
        jaccard = n_both / union if union > 0 else 0.0
        rows.append({
            "label_a": a, "label_b": b, "n_a": n_a, "n_b": n_b, "n_both": n_both,
            "jaccard": round(jaccard, 4), "pmi": round(pmi, 4) if n_both > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def sentence_level():
    df = pd.read_csv(SENTENCE_TABLE)
    X = df[FEATURE_LABELS].to_numpy(dtype=np.int64)
    out = pairwise_stats(X, FEATURE_LABELS)
    out = out.sort_values("pmi", ascending=False, na_position="last")
    out.to_csv(SENT_OUT, index=False)
    print(f"wrote {SENT_OUT} ({len(out)} non-trivial label pairs, {len(df)} sentences)")
    print()
    print("Top 10 by PMI (co-occur far more than chance):")
    print(out.head(10).to_string(index=False))
    print()
    print("Bottom 10 by PMI (co-occur far less than chance, among pairs that ever co-occur):")
    print(out[out["n_both"] > 0].tail(10).to_string(index=False))


def sentence_level_by_length():
    """Same PMI/Jaccard as sentence_level(), recomputed within each review-
    length quartile alongside the full-corpus value, to check whether the
    pooled estimate is stable across length or dominated by longer reviews
    (which contribute more sentences to the flat pool)."""
    df = pd.read_csv(SENTENCE_TABLE)
    review_lengths = df.groupby("review_id")["num_sentences"].first()
    quartile = pd.qcut(review_lengths, len(LENGTH_QUARTILE_LABELS),
                        labels=LENGTH_QUARTILE_LABELS, duplicates="drop")
    df["length_quartile"] = df["review_id"].map(quartile)

    X_full = df[FEATURE_LABELS].to_numpy(dtype=np.int64)
    full = pairwise_stats(X_full, FEATURE_LABELS).rename(
        columns={"n_both": "n_both_full", "jaccard": "jaccard_full", "pmi": "pmi_full"}
    ).set_index(["label_a", "label_b"])

    merged = full.copy()
    for q in LENGTH_QUARTILE_LABELS:
        sub = df[df["length_quartile"] == q]
        if sub.empty:
            continue
        X = sub[FEATURE_LABELS].to_numpy(dtype=np.int64)
        stats_q = pairwise_stats(X, FEATURE_LABELS).set_index(["label_a", "label_b"])
        merged[f"n_both_{q}"] = stats_q["n_both"]
        merged[f"pmi_{q}"] = stats_q["pmi"]
    merged = merged.reset_index()
    merged.to_csv(SENT_BY_LENGTH_OUT, index=False)

    pmi_cols = [f"pmi_{q}" for q in LENGTH_QUARTILE_LABELS]
    powered = merged[merged["n_both_full"] >= POWERED_THRESHOLD].copy()
    powered["sign_stable"] = powered[pmi_cols].apply(
        lambda row: (row.dropna() > 0).nunique() <= 1, axis=1
    )

    corr_rows = []
    for q in LENGTH_QUARTILE_LABELS:
        c = powered[["pmi_full", f"pmi_{q}"]].dropna()
        corr_rows.append({
            "length_quartile": q, "n_pairs": len(c),
            "corr_with_full_pmi": round(c["pmi_full"].corr(c[f"pmi_{q}"]), 4) if len(c) > 1 else float("nan"),
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(SENT_LENGTH_CORR_OUT, index=False)

    print()
    print(f"wrote {SENT_BY_LENGTH_OUT} ({len(merged)} pairs x {len(LENGTH_QUARTILE_LABELS)} length quartiles)")
    print(f"wrote {SENT_LENGTH_CORR_OUT}")
    print()
    print(f"Correlation of full-corpus PMI with each quartile's own PMI "
          f"(pairs with n_both >= {POWERED_THRESHOLD} corpus-wide, n={len(powered)}):")
    print(corr_df.to_string(index=False))
    n_flip = int((~powered["sign_stable"]).sum())
    print(f"\n{n_flip}/{len(powered)} well-powered pairs flip sign across at least one quartile.")
    print()
    print("Top 10 pairs by |pmi_full|, with per-quartile PMI (does magnitude/direction hold across length?):")
    top = merged.reindex(merged["pmi_full"].abs().sort_values(ascending=False).index).head(10)
    print(top[["label_a", "label_b", "pmi_full"] + pmi_cols].to_string(index=False))


def review_level():
    df = pd.read_csv(REVIEW_TABLE)
    presence = pd.DataFrame({
        label: (df[f"count__{label}"] > 0).astype(np.int64) for label in FEATURE_LABELS
    })
    Y = presence.to_numpy()
    n_reviews, n_labels = Y.shape

    # length-stratified bins (equal-frequency on num_sentences)
    bin_id = pd.qcut(df["num_sentences"], q=N_LENGTH_BINS, labels=False, duplicates="drop")
    bins = [np.where(bin_id.to_numpy() == b)[0] for b in sorted(bin_id.unique())]

    observed = Y.T @ Y

    rng = np.random.default_rng(RNG_SEED)
    null_joint = np.empty((N_PERMUTATIONS, n_labels, n_labels), dtype=np.int64)
    for p in range(N_PERMUTATIONS):
        Yp = Y.copy()
        for idx in bins:
            for col in range(n_labels):
                Yp[idx, col] = rng.permutation(Yp[idx, col])
        null_joint[p] = Yp.T @ Yp

    null_mean = null_joint.mean(axis=0)
    null_std = null_joint.std(axis=0)

    rows = []
    for i, j in combinations(range(n_labels), 2):
        a, b = FEATURE_LABELS[i], FEATURE_LABELS[j]
        if frozenset((a, b)) in PARENT_CHILD_PAIRS:
            continue
        obs = int(observed[i, j])
        nm, ns = null_mean[i, j], null_std[i, j]
        z = (obs - nm) / ns if ns > 0 else float("nan")
        null_dist = null_joint[:, i, j]
        dev_obs = abs(obs - nm)
        dev_null = np.abs(null_dist - nm)
        p_value = (1 + np.sum(dev_null >= dev_obs)) / (N_PERMUTATIONS + 1)
        rows.append({
            "label_a": a, "label_b": b,
            "n_reviews_a": int(observed[i, i]), "n_reviews_b": int(observed[j, j]),
            "n_both_observed": obs, "n_both_null_mean": round(nm, 1),
            "z_score": round(z, 2), "p_value": round(p_value, 4),
        })
    out = pd.DataFrame(rows).sort_values("z_score", ascending=False, na_position="last")
    out.to_csv(REVIEW_OUT, index=False)
    print()
    print(f"wrote {REVIEW_OUT} ({len(out)} non-trivial label pairs, {n_reviews} reviews, "
          f"{N_PERMUTATIONS} length-stratified permutations)")
    print()
    print("Top 10 by z-score (co-occur across a review far more than length alone predicts):")
    print(out.head(10).to_string(index=False))
    print()
    print("Bottom 10 by z-score (co-occur far less than length alone predicts):")
    print(out.tail(10).to_string(index=False))


if __name__ == "__main__":
    sentence_level()
    sentence_level_by_length()
    review_level()
