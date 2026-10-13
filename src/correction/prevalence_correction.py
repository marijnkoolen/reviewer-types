"""Rogan-Gladen misclassification correction for corpus-level prevalence
(ANALYSIS_PLAN.md §0b, method 1). For questions that only need a corpus- or
group-level rate (occurrence, genre/platform/temporal prevalence trends) --
not the per-sentence label set, which needs the instance-level method in
imputation.py instead.

    p_true = (p_observed + specificity - 1) / (sensitivity + specificity - 1)

Confidence intervals come from a bootstrap that resamples *both* uncertain
inputs: the observed population (what real prevalence uncertainty looks
like at this sample size) and the confusion matrix itself (Se/Sp are only
known from a 2,323-sentence test set, so they carry real sampling error too,
especially for the rarer labels) -- ignoring the second source would understate
the true correction uncertainty.
"""
import numpy as np


def rogan_gladen(p_observed, sensitivity, specificity):
    """Point correction. Returns (corrected, clipped, denom) where `clipped`
    flags that the raw estimate fell outside [0, 1] (denominator too small /
    noisy inputs -- a real failure mode when Se+Sp is close to 1) and was
    clamped, and `denom` = sensitivity + specificity - 1 (near zero = unstable)."""
    denom = sensitivity + specificity - 1
    if denom == 0 or np.isnan(denom):
        return float("nan"), True, denom
    raw = (p_observed + specificity - 1) / denom
    clipped = raw < 0 or raw > 1
    return float(np.clip(raw, 0.0, 1.0)), clipped, denom


def bootstrap_corrected_prevalence(observed_indicator, tp, fp, fn, tn, n_boot=2000, seed=0):
    """observed_indicator: 0/1 array, one per unit in the population being
    corrected (whether the *predicted* label is present on that unit).
    tp/fp/fn/tn: the label's confusion-matrix counts from the held-out test set,
    used to resample sensitivity and specificity alongside the observed rate.

    Bootstrap resamples are drawn via Binomial rather than materializing an
    (n_boot, n) index array: resampling n binary units with replacement and
    taking the mean is exactly Binomial(n, p_hat)/n by construction (each of
    the n resampled draws is an i.i.d. Bernoulli(p_hat) draw), so this is
    mathematically identical to explicit index resampling, not an
    approximation -- and it's the difference between this running instantly
    and needing multiple GB at silver-set scale (n ~ 467K)."""
    rng = np.random.default_rng(seed)
    n = len(observed_indicator)

    p_obs = observed_indicator.mean()
    p_obs_boot = rng.binomial(n, p_obs, size=n_boot) / n

    # sensitivity from the (tp+fn) true-positive test sentences, specificity
    # from the (tn+fp) true-negative ones
    n_pos, n_neg = tp + fn, tn + fp
    se_hat = tp / n_pos if n_pos else float("nan")
    sp_hat = tn / n_neg if n_neg else float("nan")
    sens_boot = rng.binomial(n_pos, se_hat, size=n_boot) / n_pos if n_pos else np.full(n_boot, np.nan)
    spec_boot = rng.binomial(n_neg, sp_hat, size=n_boot) / n_neg if n_neg else np.full(n_boot, np.nan)

    denom = sens_boot + spec_boot - 1
    corrected_boot = (p_obs_boot + spec_boot - 1) / denom
    n_unstable = int(np.sum((denom <= 0) | np.isnan(denom)))
    corrected_boot = np.clip(corrected_boot, 0.0, 1.0)
    corrected_boot = corrected_boot[~np.isnan(corrected_boot)]

    point, clipped, point_denom = rogan_gladen(p_obs, se_hat, sp_hat)

    return {
        "n": n, "p_observed": p_obs,
        "sensitivity": se_hat, "specificity": sp_hat,
        "p_corrected": point, "clipped_at_point_estimate": clipped, "denom": point_denom,
        "ci_lo": np.percentile(corrected_boot, 2.5) if len(corrected_boot) else float("nan"),
        "ci_hi": np.percentile(corrected_boot, 97.5) if len(corrected_boot) else float("nan"),
        "n_unstable_bootstrap_draws": n_unstable,
    }


def correct_labels(df, count_col_prefix, confusion_df, code_of_label, n_boot=2000, seed=0):
    """Apply the correction to every label in confusion_df for a review- or
    sentence-level dataframe with `{count_col_prefix}{label}` presence columns.
    code_of_label: dict mapping full label name -> short code (confusion_df's key)."""
    rows = []
    for label, code in code_of_label.items():
        cm = confusion_df.loc[confusion_df["code"] == code]
        if cm.empty:
            continue
        cm = cm.iloc[0]
        col = f"{count_col_prefix}{label}"
        if col not in df.columns:
            continue
        indicator = (df[col].to_numpy() > 0).astype(float)
        result = bootstrap_corrected_prevalence(
            indicator, int(cm["tp"]), int(cm["fp"]), int(cm["fn"]), int(cm["tn"]),
            n_boot=n_boot, seed=seed,
        )
        result["label"] = label
        result["code"] = code
        rows.append(result)
    import pandas as pd
    return pd.DataFrame(rows)[[
        "label", "code", "n", "p_observed", "sensitivity", "specificity",
        "p_corrected", "ci_lo", "ci_hi", "clipped_at_point_estimate", "n_unstable_bootstrap_draws",
    ]]
