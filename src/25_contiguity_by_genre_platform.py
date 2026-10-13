"""Do contiguity and order vary by genre and platform? (ANALYSIS_PLAN.md
item 14, Q3.)

Two parts:

A. CONTIGUITY by genre/platform, NET OF LENGTH. An uncontrolled comparison
   would largely re-measure length: §6 established that platform explains
   review length strongly (platform_2 reviews run ~3x platform_5's), and
   item 13 A1b showed contiguity excess still rises with length (rho=+0.19)
   even though it already controls each review's own n and k. So the model
   is `excess ~ n_sentences + C(genre) + C(platform)`, reported as adjusted
   marginal means plus partial eta-squared per factor -- the eta-squared is
   what makes "does genre or platform matter more, and how much next to
   length" answerable rather than a list of coefficients. Robustness: the
   same group means computed within each length quartile, so a pattern that
   only exists in long reviews is visible as such (§6's precedent).

B. ORDER by genre/platform -- is the review "arc" universal? Rather than
   running full zone discovery separately per genre (10 incomparable zone
   structures), this computes within-review precedence P(A before B) per
   group for all 21 main-label pairs and asks how much that structure MOVES.
   Two summaries: the spread of P across groups per pair, and whether the
   two structural signatures found on the pooled corpus survive everywhere
   -- `Recommendations` last against everything, and the four-category
   opening group being internally unordered.

Groups with too few reviews are dropped rather than reported as noise; the
threshold is applied per (group, pair), so a genre can be estimable for
common pairs and not for rare ones.

Usage: python src/25_contiguity_by_genre_platform.py [gold|silver]  (default: silver)
Writes results/{source}_contiguity_by_genre_platform.csv,
  results/{source}_contiguity_anova.csv,
  results/{source}_contiguity_by_length_quartile_group.csv,
  results/{source}_precedence_by_group.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, WEAK_MODEL_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "silver"
assert SOURCE in ("gold", "silver")

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
if SOURCE == "silver":
    MAIN_LABELS_USED = [m for m in MAIN_LABELS_USED if m not in WEAK_MODEL_LABELS]
PAIRS = list(combinations(MAIN_LABELS_USED, 2))

MIN_REVIEWS_GROUP = 200      # a genre/platform must clear this to be reported at all
MIN_PAIR_REVIEWS = 100       # and this per (group, pair) for a precedence estimate
DV = [("excess_over_chance", "contiguity excess"), ("boundary_excess", "boundary excess")]


def load():
    contig = pd.read_csv(RESULTS / "22_review_contiguity_scaled" / f"{SOURCE}_review_contiguity_scaled.csv")
    if SOURCE == "gold":
        rf = pd.read_csv(RESULTS / "00_data_prep/gold_review_features.csv")
    else:
        rf = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    d = contig.merge(rf[["review_id", "source_id", "nur_genre"]], on="review_id")
    return d[d["source_id"] != "platform_1"].reset_index(drop=True)


def drop_small(d, col):
    counts = d[col].value_counts()
    return d[d[col].isin(counts[counts >= MIN_REVIEWS_GROUP].index)]


def anova_and_margins(d, dv):
    """OLS with length as covariate; partial eta-squared per factor, plus
    adjusted marginal means (group mean of residuals-after-length, re-centred
    on the grand mean -- interpretable on the original scale)."""
    sub = d[[dv, "n_sentences", "nur_genre", "source_id"]].dropna()
    sub = drop_small(drop_small(sub, "nur_genre"), "source_id").rename(columns={dv: "_y"})

    model = smf.ols("_y ~ n_sentences + C(nur_genre) + C(source_id)", data=sub).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_resid = aov.loc["Residual", "sum_sq"]
    aov = aov.drop(index="Residual")
    aov["partial_eta_sq"] = aov["sum_sq"] / (aov["sum_sq"] + ss_resid)
    aov = aov.reset_index().rename(columns={"index": "term"})
    aov.insert(0, "dv", dv)

    # length-adjusted group means: y minus the fitted length contribution
    adj = sub["_y"] - model.params["n_sentences"] * (sub["n_sentences"] - sub["n_sentences"].mean())
    sub = sub.assign(_adj=adj)
    rows = []
    for factor in ["nur_genre", "source_id"]:
        g = sub.groupby(factor)["_adj"]
        raw = sub.groupby(factor)["_y"]
        for name in g.groups:
            rows.append({"dv": dv, "factor": factor, "group": name,
                         "n_reviews": int(g.size()[name]),
                         "raw_mean": round(float(raw.mean()[name]), 4),
                         "length_adjusted_mean": round(float(g.mean()[name]), 4),
                         "mean_n_sentences": round(float(sub.groupby(factor)["n_sentences"].mean()[name]), 1)})
    return aov, pd.DataFrame(rows)


def by_length_quartile(d, dv):
    sub = d[[dv, "n_sentences", "nur_genre", "source_id"]].dropna()
    sub = drop_small(drop_small(sub, "nur_genre"), "source_id")
    sub = sub.assign(q=pd.qcut(sub["n_sentences"], 4,
                                labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"))
    out = []
    for factor in ["nur_genre", "source_id"]:
        t = sub.pivot_table(index=factor, columns="q", values=dv, aggfunc="mean", observed=True)
        t = t.round(4).reset_index().rename(columns={factor: "group"})
        t.insert(0, "factor", factor)
        t.insert(0, "dv", dv)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def label_positions(sent):
    """Per review, the mean relative position of each label's own sentences."""
    pos = {}
    for label in MAIN_LABELS_USED:
        active = sent[sent[label] == 1]
        pos[label] = active.groupby("review_id")["rel_position"].mean()
    return pd.DataFrame(pos)


def precedence_by_group(sent, meta):
    pos = label_positions(sent)
    pos = pos.join(meta.set_index("review_id")[["nur_genre", "source_id"]], how="inner")
    rows = []
    for factor in ["nur_genre", "source_id"]:
        counts = pos[factor].value_counts()
        for group in counts[counts >= MIN_REVIEWS_GROUP].index:
            sub = pos[pos[factor] == group]
            for a, b in PAIRS:
                both = sub[[a, b]].dropna()
                diff = both[b] - both[a]
                nz = diff[diff != 0]
                if len(nz) < MIN_PAIR_REVIEWS:
                    continue
                rows.append({"factor": factor, "group": group, "label_a": a, "label_b": b,
                             "n": len(nz), "p_a_before_b": round(float((nz > 0).mean()), 4)})
    return pd.DataFrame(rows)


def summarize_precedence(prec):
    out = []
    for factor, sub in prec.groupby("factor"):
        piv = sub.pivot_table(index=["label_a", "label_b"], columns="group", values="p_a_before_b")
        piv = piv.dropna(thresh=3)     # need >=3 groups to talk about spread
        s = pd.DataFrame({
            "factor": factor,
            "label_a": [i[0] for i in piv.index], "label_b": [i[1] for i in piv.index],
            "n_groups": piv.notna().sum(axis=1).to_numpy(),
            "mean_p": piv.mean(axis=1).round(4).to_numpy(),
            "min_p": piv.min(axis=1).round(4).to_numpy(),
            "max_p": piv.max(axis=1).round(4).to_numpy(),
            "sd_p": piv.std(axis=1).round(4).to_numpy(),
        })
        s["range_p"] = (s["max_p"] - s["min_p"]).round(4)
        # Does the direction hold in every group where the pair is estimable?
        # NaN-safe: a plain piv.gt(0.5).all() would count a missing group as
        # False and mark otherwise-stable pairs unstable, so absent groups are
        # explicitly excluded rather than treated as disagreement.
        above = (piv > 0.5) | piv.isna()
        below = (piv < 0.5) | piv.isna()
        s["direction_stable"] = (above.all(axis=1) | below.all(axis=1)).to_numpy()
        out.append(s)
    return pd.concat(out, ignore_index=True).sort_values(["factor", "sd_p"], ascending=[True, False])


def main():
    d = load()
    print(f"{SOURCE}: {len(d):,} reviews ({int(d['eligible_a1'].sum()):,} contiguity-eligible)")

    # --- A: contiguity by genre/platform, net of length --------------------
    el = d[d["eligible_a1"]]
    aovs, margins, quarts = [], [], []
    for dv, label in DV:
        a, m = anova_and_margins(el, dv)
        aovs.append(a); margins.append(m)
        quarts.append(by_length_quartile(el, dv))
    aov = pd.concat(aovs, ignore_index=True)
    marg = pd.concat(margins, ignore_index=True)
    quart = pd.concat(quarts, ignore_index=True)
    aov.to_csv(RESULTS / "25_contiguity_by_genre_platform" / f"{SOURCE}_contiguity_anova.csv", index=False)
    marg.to_csv(RESULTS / "25_contiguity_by_genre_platform" / f"{SOURCE}_contiguity_by_genre_platform.csv", index=False)
    quart.to_csv(RESULTS / "25_contiguity_by_genre_platform" / f"{SOURCE}_contiguity_by_length_quartile_group.csv", index=False)

    print("\n=== A: how much do genre/platform explain, next to length? (partial eta-sq) ===")
    print(aov[["dv", "term", "partial_eta_sq", "PR(>F)"]].to_string(index=False))

    for dv, label in DV:
        print(f"\n--- {label}: length-adjusted means by platform ---")
        s = marg[(marg["dv"] == dv) & (marg["factor"] == "source_id")].sort_values("length_adjusted_mean")
        print(s[["group", "n_reviews", "mean_n_sentences", "raw_mean", "length_adjusted_mean"]].to_string(index=False))
        print(f"--- {label}: length-adjusted means by genre ---")
        s = marg[(marg["dv"] == dv) & (marg["factor"] == "nur_genre")].sort_values("length_adjusted_mean")
        print(s[["group", "n_reviews", "mean_n_sentences", "raw_mean", "length_adjusted_mean"]].to_string(index=False))

    # --- B: is the arc universal? -----------------------------------------
    sent = (pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv") if SOURCE == "gold"
            else pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet"))
    prec = precedence_by_group(sent, d[["review_id", "nur_genre", "source_id"]])
    prec.to_csv(RESULTS / "25_contiguity_by_genre_platform" / f"{SOURCE}_precedence_by_group.csv", index=False)
    summ = summarize_precedence(prec)

    print("\n=== B: does the review arc hold across genres and platforms? ===")
    for factor in ["nur_genre", "source_id"]:
        s = summ[summ["factor"] == factor]
        stable = int(s["direction_stable"].sum())
        print(f"\n{factor}: {len(s)} pairs estimable in >=3 groups; "
              f"{stable}/{len(s)} keep the SAME direction in every group")
        print("  most variable pairs:")
        print(s.head(4)[["label_a", "label_b", "n_groups", "mean_p", "min_p", "max_p",
                          "range_p", "direction_stable"]].to_string(index=False))
        rec = s[(s["label_a"] == "Recommendations") | (s["label_b"] == "Recommendations")]
        if len(rec):
            print(f"  Recommendations pairs: {int(rec['direction_stable'].sum())}/{len(rec)} "
                  f"direction-stable (mean range {rec['range_p'].mean():.3f})")

    print(f"\nwrote results to {RESULTS} ({SOURCE}_contiguity_*.csv, {SOURCE}_precedence_by_group.csv)")


if __name__ == "__main__":
    main()
