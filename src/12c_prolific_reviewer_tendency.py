"""Prolific-reviewer compositional/sequential tendency, part (a): modal
cluster share (ANALYSIS_PLAN.md TODO, extends §9). Silver-only -- gold's max
is 8 reviews/reviewer (excl. platform_1), not enough to say anything about an
individual's personal tendency; silver has thousands of reviewers with
enough of their own reviews to ask the question properly.

For each reviewer with >=10 reviews, what fraction of their reviews fall in
their single most-common compositional cluster (of 8, from 05c_typology_clustering.py)
and, separately, their most-common sequence cluster (of 7)? A reviewer sitting
in one cluster 80% of the time has a real personal signature; one spread
evenly across clusters doesn't.

"Unusually high" is judged against a null: if a reviewer's n reviews were
i.i.d. draws from the corpus-wide cluster distribution (not uniform -- the
real, unequal cluster sizes), what modal share would that produce by chance
alone? Simulated per unique n (not per reviewer -- the null only depends on
n and the corpus distribution, cached the same way as the contiguity and
order null models elsewhere in this project), since a modal share of, say,
40% is unremarkable for n=10 draws across unevenly-sized clusters but would
be a strong signal for n=200.

Multiple-comparisons correction (ANALYSIS_PLAN.md item 4, chat 2026-08-19):
5,782 reviewers x 2 dimensions = 11,564 tests run here with no correction
until now. Benjamini-Hochberg FDR applied separately within each of the two
families (compositional-share tests, sequential-share tests) -- not
Bonferroni/Holm (FWER is the wrong error rate for a screening/exploratory
question like this one; at ~5,782 tests Bonferroni's per-test threshold,
0.05/5782~=8.6e-6, would erase most real findings along with the false
ones), and not pooled across both families (they're two different
questions, same precedent as 06a_genre_platform.py's separate genre/platform
correction). N_SIM also bumped 2,000 -> 20,000: the permutation p-value
floor (1/(N_SIM+1)) was 0.0005 at 2,000 sims, coarse enough that many of the
most extreme reviewers were tied at the floor rather than actually
differentiated -- BH needs real resolution among the smallest p-values to
rank them correctly.

platform_1 excluded throughout (its user_id is a placeholder, not a real
reviewer identity -- ANALYSIS_PLAN.md §0).

Usage: python src/12c_prolific_reviewer_tendency.py
Writes results/12_prolific_reviewer_tendency/silver_prolific_reviewer_tendency.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from stats_utils import benjamini_hochberg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MIN_REVIEWS = 10
N_SIM = 20_000
RNG_SEED = 0

_null_cache = {}


def null_modal_share_dist(n, cluster_probs, rng):
    key = (n, tuple(np.round(cluster_probs, 6)))
    if key not in _null_cache:
        k = len(cluster_probs)
        draws = rng.choice(k, size=(N_SIM, n), p=cluster_probs)
        counts = (draws[:, :, None] == np.arange(k)).sum(axis=1)  # (N_SIM, k)
        _null_cache[key] = counts.max(axis=1) / n
    return _null_cache[key]


def modal_share_with_test(df, cluster_col, rng):
    """Per group (user_id), the modal cluster + share, plus a p-value against
    the corpus-distribution null for that group's own n."""
    corpus_probs = df[cluster_col].value_counts(normalize=True).sort_index()
    probs_array = corpus_probs.to_numpy()

    rows = []
    for user_id, sub in df.groupby("user_id"):
        n = len(sub)
        vc = sub[cluster_col].value_counts()
        modal_cluster = vc.idxmax()
        modal_share = vc.max() / n
        null_dist = null_modal_share_dist(n, probs_array, rng)
        p_value = (1 + np.sum(null_dist >= modal_share)) / (N_SIM + 1)
        z = (modal_share - null_dist.mean()) / null_dist.std() if null_dist.std() > 0 else float("nan")
        rows.append({
            "user_id": user_id, "n_reviews": n,
            f"modal_{cluster_col}": modal_cluster, f"{cluster_col}_share": round(modal_share, 4),
            f"{cluster_col}_null_mean": round(null_dist.mean(), 4),
            f"{cluster_col}_z": round(z, 2), f"{cluster_col}_p_value": p_value,
        })
    out = pd.DataFrame(rows)
    out[f"{cluster_col}_q_bh"] = benjamini_hochberg(out[f"{cluster_col}_p_value"].to_numpy())
    return out


def main():
    review = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")[["review_id", "user_id"]]
    clusters = pd.read_csv(RESULTS / "05_typology/silver_typology_clusters.csv")
    df = clusters.merge(review, on="review_id")
    df = df[df["source_id"] != "platform_1"]

    counts = df["user_id"].value_counts()
    prolific = counts[counts >= MIN_REVIEWS].index
    df = df[df["user_id"].isin(prolific)]
    print(f"{len(prolific)} reviewers with >={MIN_REVIEWS} reviews (excl. platform_1), {len(df)} reviews total")

    rng = np.random.default_rng(RNG_SEED)
    comp = modal_share_with_test(df, "cluster_composition", rng)
    seq = modal_share_with_test(df, "cluster_sequence", rng)
    out = comp.merge(seq, on=["user_id", "n_reviews"])

    genre_top = df.groupby("user_id")["nur_genre"].agg(lambda s: s.value_counts().idxmax())
    out["top_genre"] = out["user_id"].map(genre_top)

    out = out.sort_values("cluster_composition_share", ascending=False)
    out.to_csv(RESULTS / "12_prolific_reviewer_tendency/silver_prolific_reviewer_tendency.csv", index=False)

    n_sig_comp_raw = (out["cluster_composition_p_value"] < 0.05).sum()
    n_sig_seq_raw = (out["cluster_sequence_p_value"] < 0.05).sum()
    n_sig_comp = (out["cluster_composition_q_bh"] < 0.05).sum()
    n_sig_seq = (out["cluster_sequence_q_bh"] < 0.05).sum()
    n_sig_both = ((out["cluster_composition_q_bh"] < 0.05) & (out["cluster_sequence_q_bh"] < 0.05)).sum()
    corr = out[["cluster_composition_share", "cluster_sequence_share"]].corr().iloc[0, 1]

    print(f"\nreviewers with a significantly (raw p<0.05) high compositional modal share: {n_sig_comp_raw}/{len(out)}")
    print(f"reviewers with a significantly (raw p<0.05) high sequential modal share: {n_sig_seq_raw}/{len(out)}")
    print(f"\nafter BH-FDR correction (separate families):")
    print(f"reviewers with a significantly (q<0.05) high compositional modal share: {n_sig_comp}/{len(out)}")
    print(f"reviewers with a significantly (q<0.05) high sequential modal share: {n_sig_seq}/{len(out)}")
    print(f"reviewers significant on BOTH (q<0.05): {n_sig_both}/{len(out)}")
    print(f"correlation between compositional share and sequential share across reviewers: {corr:.3f}")

    print("\ntop 15 by compositional modal share (n>=10):")
    show = out[["user_id", "n_reviews", "modal_cluster_composition", "cluster_composition_share",
                "cluster_composition_q_bh", "modal_cluster_sequence", "cluster_sequence_share",
                "cluster_sequence_q_bh", "top_genre"]]
    print(show.head(15).to_string(index=False))

    print(f"\nwrote {RESULTS / 'silver_prolific_reviewer_tendency.csv'}")


if __name__ == "__main__":
    main()
