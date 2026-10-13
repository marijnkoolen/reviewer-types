"""Small shared statistics helpers used across multiple analysis scripts."""
import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def cramers_v(table):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    r, c = table.shape
    v = np.sqrt((chi2 / n) / (min(r - 1, c - 1))) if min(r - 1, c - 1) > 0 else float("nan")
    min_expected = expected.min()
    return chi2, p, v, min_expected


def prevalence_by_group(df, group_col, tested_labels):
    """Category prevalence per group + chi-square test of independence
    (label present/absent x group), Cramer's V effect size, BH-adjusted q-value
    across the family of labels tested."""
    groups = sorted(df[group_col].unique())
    rows, pvals = [], []
    for label in tested_labels:
        present = df[f"count__{label}"] > 0
        table = pd.crosstab(df[group_col], present)
        if table.shape[1] < 2:
            continue
        chi2, p, v, min_exp = cramers_v(table)
        row = {"label": label, "chi2": round(chi2, 2), "p_value": p,
               "cramers_v": round(v, 4), "min_expected_cell": round(min_exp, 2)}
        for g in groups:
            g_sub = df[df[group_col] == g]
            row[f"prevalence__{g}"] = round((g_sub[f"count__{label}"] > 0).mean(), 4)
        rows.append(row)
        pvals.append(p)
    out = pd.DataFrame(rows)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values("cramers_v", ascending=False)


def typology_by_group(clusters, group_col, cluster_col="cluster_composition"):
    table = pd.crosstab(clusters[group_col], clusters[cluster_col])
    chi2, p, v, min_exp = cramers_v(table)
    print(f"typology x {group_col}: chi2={chi2:.2f}, p={p:.4g}, Cramer's V={v:.3f}, "
          f"min expected cell={min_exp:.2f}")
    return table, {"chi2": chi2, "p_value": p, "cramers_v": v, "min_expected_cell": min_exp}
