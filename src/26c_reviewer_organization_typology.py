"""Is there a reviewer typology based on contiguity and order?
(ANALYSIS_PLAN.md item 14, Q2 -- gated on Q1, which cleared it: the
length-residualized ICC was 0.102, not collapsing toward the ~0.05 that
would have meant clustering noise.)

Unit is the REVIEWER, not the review. Feature vector per reviewer:
  - mean contiguity excess (item 13 A1)
  - mean boundary excess (item 13 A2)
  - mean onset (first-occurrence) position of each of the 7 main labels (the
    reviewer's characteristic ordering) -- onset rather than mean position as
    of 2026-09-22, matching 05a/05c_typology_clustering.py after prototyping
    showed onset gives a cleaner, less length-confounded signal (see 05a's
    module docstring); the length-correlation figures below predate that
    switch and motivate the residualization step regardless of which
    construction they were measured on.

EVERY feature is residualized on length + genre + platform at the REVIEW
level before aggregating, and this is not a formality:
  - contiguity excess rises with length (rho=+0.19, item 13 A1b);
  - the positional features do too, some strongly -- Reader_response
    position correlates rho=+0.52 with review length, Style +0.25,
    Classification +0.20 (checked, not assumed);
  - and item 8 found Cramer's V=0.70 between prolificness and platform.
Without the control, a "reviewer organization typology" would substantially
be a length-and-platform typology wearing a different label. The confound
check below tests whether that leakage survives anyway.

Missing positions (a review with no sentences of label L) are imputed with
the corpus onset marginal for L, following the precedent already set by
11_sequence_cluster_profile.py and 13c_reviewer_similarity_ranking.py.

k is chosen by §5's protocol -- silhouette plus bootstrap stability (ARI
between refits on reviewer-resampled data) -- so the answer is comparable to
the compositional typology rather than using a different standard.

Usage: python src/26c_reviewer_organization_typology.py [silver]  (silver only:
  gold's most prolific reviewer has 4 contiguity-eligible reviews)
Writes results/26_reviewer_organization_typology/silver_reviewer_organization_typology.csv,
  results/26_reviewer_organization_typology/silver_reviewer_organization_k_selection.csv,
  results/26_reviewer_organization_typology/silver_reviewer_organization_profile.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS  # noqa: E402
from stats_utils import cramers_v  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SOURCE = "silver"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
MIN_REVIEWS = 10
K_RANGE = range(2, 9)
N_BOOT = 20
RNG_SEED = 0


def build_review_features():
    contig = pd.read_csv(RESULTS / "22_review_contiguity_scaled" / f"{SOURCE}_review_contiguity_scaled.csv")
    contig = contig[contig["eligible_a1"]]
    rf = pd.read_parquet(RESULTS / "00_data_prep" / f"{SOURCE}_review_features.parquet")
    meta = rf[["review_id", "user_id", "source_id", "nur_genre"]]

    sent = pd.read_parquet(RESULTS / "00_data_prep" / f"{SOURCE}_sentence_sequences.parquet")
    marginal = pd.read_csv(RESULTS / "04_order" / f"{SOURCE}_order_onset.csv")
    marginal = marginal[marginal["level"] == "main"].set_index("label")["first_mean_position"]
    pos = {}
    for label in MAIN_LABELS_USED:
        pos[label] = sent[sent[label] == 1].groupby("review_id")["rel_position"].min()
    pos = pd.DataFrame(pos)

    d = contig.merge(meta, on="review_id").merge(pos, left_on="review_id", right_index=True, how="left")
    d = d[d["source_id"] != "platform_1"]
    for label in MAIN_LABELS_USED:                     # corpus-marginal imputation, per 19a_/23_
        d[label] = d[label].fillna(marginal.get(label, 0.5))
    return d.reset_index(drop=True)


def residualize_all(d, feature_cols):
    """Remove length, genre and platform from every feature at review level."""
    out = d[["review_id", "user_id"]].copy()
    for col in feature_cols:
        # dropna then assign BY INDEX: boundary_excess is undefined when every
        # sentence has a distinct configuration (item 13 A2), so the fitted rows
        # are a subset and positional assignment would silently misalign them.
        sub = d[[col, "n_sentences", "nur_genre", "source_id"]].rename(columns={col: "_y"}).dropna()
        model = smf.ols("_y ~ n_sentences + C(nur_genre) + C(source_id)", data=sub).fit()
        out[col] = np.nan
        out.loc[sub.index, col] = model.resid.to_numpy()
    return out


def to_reviewer_level(resid):
    counts = resid["user_id"].value_counts()
    keep = counts[counts >= MIN_REVIEWS].index
    r = resid[resid["user_id"].isin(keep)]
    agg = r.groupby("user_id").mean(numeric_only=True)
    agg["n_reviews"] = r.groupby("user_id").size()
    return agg


def select_k(X, rng):
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(5000, len(X)), random_state=RNG_SEED)
        aris = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(X), size=len(X))
            kb = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X[idx])
            aris.append(adjusted_rand_score(km.labels_, kb.predict(X)))
        rows.append({"k": k, "silhouette": round(float(sil), 4),
                     "bootstrap_ari": round(float(np.mean(aris)), 4),
                     "bootstrap_ari_sd": round(float(np.std(aris)), 4)})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(RNG_SEED)
    d = build_review_features()
    feature_cols = ["excess_over_chance", "boundary_excess"] + MAIN_LABELS_USED
    print(f"{SOURCE}: {len(d):,} eligible reviews, {d['user_id'].nunique():,} reviewers")

    resid = residualize_all(d, feature_cols)
    rv = to_reviewer_level(resid)
    print(f"reviewer-level matrix: {len(rv):,} reviewers with >={MIN_REVIEWS} eligible reviews, "
          f"{len(feature_cols)} features (all residualized on length+genre+platform)")

    # A reviewer can still be missing a feature entirely (e.g. none of their
    # reviews is scorable for boundary excess). These are residuals, so 0 is
    # exactly "no deviation from the corpus norm" -- the honest fill.
    n_missing = int(rv[feature_cols].isna().any(axis=1).sum())
    if n_missing:
        print(f"  {n_missing} reviewer(s) missing >=1 feature entirely; filled with 0 (= corpus norm)")
    X = StandardScaler().fit_transform(rv[feature_cols].fillna(0.0).to_numpy())

    ksel = select_k(X, rng)
    ksel.to_csv(RESULTS / "26_reviewer_organization_typology" / f"{SOURCE}_reviewer_organization_k_selection.csv", index=False)
    print("\n=== k selection (silhouette + bootstrap stability, §5's protocol) ===")
    print(ksel.to_string(index=False))

    best = ksel.sort_values("silhouette", ascending=False).iloc[0]
    k = int(best["k"])
    print(f"\nbest silhouette at k={k} ({best['silhouette']:.4f}), "
          f"bootstrap ARI {best['bootstrap_ari']:.4f}")

    km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X)
    rv = rv.assign(org_cluster=km.labels_)

    # --- confound check: is this really a platform/genre typology? ---------
    meta = d.groupby("user_id").agg(
        dominant_platform=("source_id", lambda s: s.value_counts().idxmax()),
        dominant_genre=("nur_genre", lambda s: s.value_counts().idxmax()),
        mean_n_sentences=("n_sentences", "mean"))
    rv = rv.join(meta, how="left")
    print("\n=== confound check: does the typology just recover platform/genre/length? ===")
    for col in ["dominant_platform", "dominant_genre"]:
        tab = pd.crosstab(rv["org_cluster"], rv[col])
        chi2, p, v, _ = cramers_v(tab)
        print(f"  org_cluster x {col}: Cramer's V = {v:.4f}")
    grp = rv.groupby("org_cluster")["mean_n_sentences"]
    print(f"  mean review length by cluster: {grp.mean().round(1).to_dict()}")

    # --- is it a relabelling of the compositional signature? ---------------
    tend = pd.read_csv(RESULTS / "12_prolific_reviewer_tendency" / f"{SOURCE}_prolific_reviewer_tendency.csv")
    comp = tend.set_index("user_id")["modal_cluster_composition"]
    j = rv.join(comp, how="inner").dropna(subset=["modal_cluster_composition"])
    ari = adjusted_rand_score(j["modal_cluster_composition"].astype(int), j["org_cluster"])
    print(f"\n=== vs. the compositional signature (item 3) ===")
    print(f"  ARI(organization typology, modal compositional cluster) = {ari:.4f}  (n={len(j):,})")
    print("  ~0 means these are independent axes, not a relabelling")

    prof_cols = feature_cols + ["n_reviews", "mean_n_sentences"]
    profile = rv.groupby("org_cluster")[prof_cols].mean().round(4)
    profile.insert(0, "n_reviewers", rv["org_cluster"].value_counts().sort_index())
    profile.to_csv(RESULTS / "26_reviewer_organization_typology" / f"{SOURCE}_reviewer_organization_profile.csv")
    rv.reset_index().to_csv(RESULTS / "26_reviewer_organization_typology" / f"{SOURCE}_reviewer_organization_typology.csv", index=False)

    print("\n=== cluster profile (residualized means: + = above the corpus norm) ===")
    show = ["n_reviewers", "excess_over_chance", "boundary_excess"] + MAIN_LABELS_USED
    print(profile[show].to_string())

    print(f"\nwrote results to {RESULTS} ({SOURCE}_reviewer_organization_*.csv)")


if __name__ == "__main__":
    main()
