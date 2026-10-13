"""Review-level contiguity at silver scale (ANALYSIS_PLAN.md item 13, parts
A1/A1b/A2 -- the whole-review measures). Gold's implementations (31_, 32_)
loop per review with a 1000-permutation array each, which does not finish on
467K reviews; this is the vectorized replacement.

KEY SIMPLIFICATION (2026-08-31): no permutation is needed at all for the
headline `excess over chance` measure. Both scores are LINEAR in an
underlying count, so the null EXPECTATION is available in closed form:

  A1  clumpiness = 1 - (runs-1)/(k-1), linear in runs, and for k ones placed
      uniformly among n slots  E[runs] = k(n-k+1)/n  (position 1 starts a run
      w.p. k/n; each later position starts one w.p. (k/n)((n-k)/(n-1))).

  A2  score = 1 - (B - (m-1))/((n-1) - (m-1)), linear in B, and for a random
      permutation of a multiset with configuration counts c_1..c_m,
      E[B] = (n-1) * (1 - sum_i c_i(c_i-1) / (n(n-1))), since each adjacent
      pair is a uniformly random ordered pair of distinct positions.

Validated against gold's simulated nulls (see validate_against_simulated):
mean |difference| ~0.003, correlation ~0.9995 -- the residual is Monte Carlo
noise in the 1000-permutation simulation, not error in the closed form.

What this does NOT provide: per-review z-scores and p-values, which need the
null VARIANCE and hence the joint permutation (labels are not independent --
31_'s docstring explains why the joint shuffle matters). That is a deliberate
scope choice, not an oversight: A1b established that at review level the
significance flag mostly tracks statistical power (short reviews cannot clear
it regardless of organization) while `excess over chance` is the effect size.
The gold pass has the significance testing; silver reports effect sizes.

Usage: python src/22_review_contiguity_scaled.py [gold|silver]  (default: silver)
Writes results/{source}_review_contiguity_scaled.csv,
  results/{source}_review_contiguity_scaled_by_length.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, WEAK_MODEL_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "silver"
assert SOURCE in ("gold", "silver")

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
if SOURCE == "silver":
    # NOTE: all four WEAK_MODEL_LABELS are SUB-labels, so this removes nothing
    # from the main-label set -- silver keeps all 7. Kept explicit so the
    # filter is visible rather than assumed.
    MAIN_LABELS_USED = [m for m in MAIN_LABELS_USED if m not in WEAK_MODEL_LABELS]

MIN_ELIGIBLE_LABELS = 2
MIN_SENTENCES = 3


def load_sentences():
    if SOURCE == "gold":
        return pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
    return pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")


def compute(sent):
    sent = sent.sort_values(["review_id", "sent_num"]).reset_index(drop=True)
    rid = sent["review_id"]
    grouped = sent.groupby("review_id", sort=False)

    n_by_review = grouped.size().rename("n_sentences")

    # --- A1: per review x label runs and counts, vectorized per label -------
    parts = []
    for label in MAIN_LABELS_USED:
        col = sent[label]
        prev = grouped[label].shift(fill_value=0)
        starts = (col == 1) & (prev != 1)
        k = col.groupby(rid, sort=False).sum()
        runs = starts.groupby(rid, sort=False).sum()
        parts.append(pd.DataFrame({"review_id": k.index, "label": label,
                                    "k": k.to_numpy(), "runs": runs.to_numpy()}))
    per_label = pd.concat(parts, ignore_index=True)
    per_label = per_label.merge(n_by_review.rename("n"), left_on="review_id", right_index=True)
    per_label = per_label[per_label["k"] >= 2].copy()

    per_label["clumpiness"] = 1.0 - (per_label["runs"] - 1) / (per_label["k"] - 1)
    e_runs = per_label["k"] * (per_label["n"] - per_label["k"] + 1) / per_label["n"]
    per_label["null_clumpiness"] = 1.0 - (e_runs - 1) / (per_label["k"] - 1)

    per_label["_wo"] = per_label["clumpiness"] * per_label["k"]
    per_label["_wn"] = per_label["null_clumpiness"] * per_label["k"]
    g = per_label.groupby("review_id")
    a1 = pd.DataFrame({
        "n_eligible_labels": g.size(),
        "total_eligible_sentences": g["k"].sum(),
        "review_clumpiness": g["_wo"].sum() / g["k"].sum(),
        "null_mean": g["_wn"].sum() / g["k"].sum(),
    })

    # --- A2: configuration boundaries --------------------------------------
    weights = (1 << np.arange(len(MAIN_LABELS_USED))).astype(np.int64)
    cfg = (sent[MAIN_LABELS_USED].to_numpy(dtype=np.int64) * weights).sum(axis=1)
    sent = sent.assign(_cfg=cfg)
    prev_cfg = sent.groupby("review_id", sort=False)["_cfg"].shift()
    changed = (sent["_cfg"] != prev_cfg) & prev_cfg.notna()
    B = changed.groupby(rid, sort=False).sum().rename("boundaries")
    m = sent.groupby("review_id", sort=False)["_cfg"].nunique().rename("n_configurations")
    cc = sent.groupby(["review_id", "_cfg"], sort=False).size().rename("c").reset_index()
    cc["cc1"] = cc["c"] * (cc["c"] - 1)
    sum_cc1 = cc.groupby("review_id")["cc1"].sum().rename("sum_c_c_minus_1")

    a2 = pd.concat([n_by_review, B, m, sum_cc1], axis=1)
    n_, m_ = a2["n_sentences"], a2["n_configurations"]
    # n >= MIN_SENTENCES applies to A2 as well as A1: a 2-sentence review has a
    # technically-defined boundary score but cannot meaningfully be "block
    # structured", and 32_'s gold pass excluded these, so keeping the filter
    # identical is what makes the silver numbers comparable to the published ones.
    denom = np.where(n_ >= MIN_SENTENCES, (n_ - 1) - (m_ - 1), 0)
    e_b = (n_ - 1) * (1.0 - a2["sum_c_c_minus_1"] / (n_ * (n_ - 1)))
    a2["boundary_score"] = np.where(denom > 0, 1.0 - (a2["boundaries"] - (m_ - 1)) / denom, np.nan)
    a2["boundary_null"] = np.where(denom > 0, 1.0 - (e_b - (m_ - 1)) / denom, np.nan)

    out = a2.join(a1, how="left")
    out["excess_over_chance"] = out["review_clumpiness"] - out["null_mean"]
    out["boundary_excess"] = out["boundary_score"] - out["boundary_null"]

    eligible = (out["n_eligible_labels"] >= MIN_ELIGIBLE_LABELS) & (out["n_sentences"] >= MIN_SENTENCES)
    out["eligible_a1"] = eligible.fillna(False)
    return out.reset_index()


def length_stratified(df, value_cols):
    d = df[df["eligible_a1"]].copy()
    d["length_quartile"] = pd.qcut(d["n_sentences"], 4,
                                    labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"],
                                    duplicates="drop")
    agg = {"n_reviews": ("review_id", "size"), "median_sentences": ("n_sentences", "median")}
    for c in value_cols:
        agg[f"mean_{c}"] = (c, "mean")
    return d.groupby("length_quartile", observed=True).agg(**agg).round(4).reset_index()


def validate_against_simulated(out):
    """Cross-check the closed-form nulls against gold's 1000-permutation
    simulated nulls -- the residual should be Monte Carlo noise only."""
    if SOURCE != "gold":
        return
    for path, sim_col, ana_col, name in [
        ("19_review_contiguity_precedence/gold_review_contiguity.csv", "null_mean", "null_mean", "A1"),
        ("20_contiguity_zones/gold_review_boundary_interleaving.csv", "boundary_null", "boundary_null", "A2"),
    ]:
        p = RESULTS / path
        if not p.exists():
            continue
        sim = pd.read_csv(p)[["review_id", sim_col]].rename(columns={sim_col: "simulated"})
        m = out[["review_id", ana_col]].rename(columns={ana_col: "analytic"}).merge(sim, on="review_id").dropna()
        if len(m):
            d = (m["simulated"] - m["analytic"]).abs()
            print(f"  {name} null: n={len(m)}  mean|diff|={d.mean():.6f}  max|diff|={d.max():.5f}  "
                  f"r={m['simulated'].corr(m['analytic']):.6f}")


def main():
    sent = load_sentences()
    print(f"{SOURCE}: {len(sent):,} sentences, {sent['review_id'].nunique():,} reviews, "
          f"{len(MAIN_LABELS_USED)} main labels")

    out = compute(sent)
    out.to_csv(RESULTS / "22_review_contiguity_scaled" / f"{SOURCE}_review_contiguity_scaled.csv", index=False)

    print("\n=== closed-form null validation (gold only) ===")
    validate_against_simulated(out)

    el = out[out["eligible_a1"]]
    print(f"\n=== A1: review-level contiguity ===")
    print(f"{len(el):,} eligible reviews (>={MIN_ELIGIBLE_LABELS} labels with k>=2, "
          f"n>={MIN_SENTENCES}) of {len(out):,}")
    print(f"mean clumpiness {el['review_clumpiness'].mean():.4f} vs null "
          f"{el['null_mean'].mean():.4f} (excess {el['excess_over_chance'].mean():+.4f})")

    a2ok = out["boundary_excess"].notna()
    print(f"\n=== A2: configuration-boundary score ===")
    print(f"{int(a2ok.sum()):,} scorable reviews; {int((~a2ok).sum()):,} undefined (m == n)")
    print(f"mean boundary score {out.loc[a2ok,'boundary_score'].mean():.4f} vs null "
          f"{out.loc[a2ok,'boundary_null'].mean():.4f} "
          f"(excess {out.loc[a2ok,'boundary_excess'].mean():+.4f})")

    both = el["boundary_excess"].notna()
    rho, p = stats.spearmanr(el.loc[both, "excess_over_chance"], el.loc[both, "boundary_excess"])
    print(f"\nA1 vs A2 excess: spearman rho={rho:+.3f} (p={p:.2g}, n={int(both.sum()):,})")

    strat = length_stratified(out, ["review_clumpiness", "null_mean", "excess_over_chance",
                                     "boundary_score", "boundary_null", "boundary_excess"])
    strat.to_csv(RESULTS / "22_review_contiguity_scaled" / f"{SOURCE}_review_contiguity_scaled_by_length.csv", index=False)
    print("\n=== A1b: by review-length quartile ===")
    print(strat.to_string(index=False))

    for col, label in [("review_clumpiness", "raw clumpiness"), ("null_mean", "chance baseline"),
                        ("excess_over_chance", "EXCESS")]:
        rho, p = stats.spearmanr(el["n_sentences"], el[col])
        print(f"spearman(n_sentences, {label:<16}) = {rho:+.3f} (p={p:.2g})")

    print(f"\nwrote results to {RESULTS} ({SOURCE}_review_contiguity_scaled*.csv)")


if __name__ == "__main__":
    main()
