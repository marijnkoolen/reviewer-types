"""Multiple imputation for instance-level correction (ANALYSIS_PLAN.md §0b,
method 2). Aggregate correction (prevalence_correction.py) only recovers a
corpus-level rate; co-occurrence, contiguity, order, and typology clustering
all need the actual per-sentence label *set*, which needs this instead.

Model: each observed (predicted) label is a noisy proxy for a latent true
label. Per label, independently (matching how the classifier actually makes
20 separate binary decisions -- see ANALYSIS_PLAN.md §0b on why a joint error
model isn't the right target here), Bayes' rule turns the label's
sensitivity/specificity and a prior P(true=1) into a posterior
P(true=1 | predicted) for every sentence. Multiple imputation draws M
plausible "corrected" label sets from that posterior rather than picking a
single hard-thresholded guess, so that any downstream statistic computed
across the M draws carries explicit model-error uncertainty when pooled with
Rubin's rules -- a single imputation would silently understate uncertainty.

The parent-implies-child hierarchy (a sub-category being true implies its
main category is too) is enforced *after* imputation, not built into the
per-label posteriors -- each label's error model stays independent and
simple, and the hierarchy is a cheap deterministic fix-up on top.
"""
import numpy as np
import pandas as pd
from scipy import stats


def posterior_true_given_predicted(predicted, prior, sensitivity, specificity):
    """predicted: 0/1 array. Returns P(true=1 | predicted) per element via Bayes' rule."""
    predicted = np.asarray(predicted)
    p_true_pos = sensitivity * prior / (sensitivity * prior + (1 - specificity) * (1 - prior))
    p_true_neg = (1 - sensitivity) * prior / ((1 - sensitivity) * prior + specificity * (1 - prior))
    return np.where(predicted == 1, p_true_pos, p_true_neg)


def multiple_impute(df, labels, code_of_label, confusion_df, priors, sub_to_main,
                     m=20, seed=0, pred_col_prefix="", conditional_confusion_df=None):
    """df: sentence-level frame with one 0/1 column per label (name = pred_col_prefix + label).
    priors: dict label -> P(true=1), typically the Rogan-Gladen corrected prevalence.
    sub_to_main: dict sub-label -> main-label, for the hierarchy constraint.
    Returns a list of `m` binary DataFrames (same shape as the input label columns),
    each one plausible corrected label set.

    Main labels are imputed first, from their own posterior alone (unconditional
    prior = their own R-G corrected prevalence). Each sub-label is then imputed
    CONDITIONAL on its main's draw: if main drew 0, sub is deterministically 0
    (a sub can't be true when its main isn't -- no uncertainty there, it's a
    hard logical fact, not something to model). If main drew 1, sub is drawn
    from its own posterior but recomputed with the correct CONDITIONAL prior
    P(sub=1 | main=1) = P(sub=1)/P(main=1) (from the priors dict; sub=1 implies
    main=1, so this is exact, not an approximation), instead of the
    unconditional prior -- since we've already conditioned on main=1 for this
    branch of the imputation, using the unconditional prior would be wrong.

    Two simpler designs were tried first and both validated badly biased (see
    results/gold_correction_validation_*.csv, docstring history in git): (1)
    draw every label independently then force main=1 whenever any child sub=1
    -- ORing k independently-drawn noisy sub-imputations compounds their
    false-positive rates, up to +0.17 absolute prevalence bias on main labels
    with several children. (2) fold children's posteriors into main's
    probability via noisy-OR before drawing -- mathematically identical to (1)
    in expectation, so no improvement. (3) draw independently, suppress the
    sub on conflict instead of forcing main up -- fixed the main labels but
    pushed large downward bias onto the subs instead (up to -0.14), since it
    zeroes out correctly-drawn true sub-positives whenever main's own draw
    happens to be a false negative. This conditional-prior approach validated
    well for most labels -- adding `conditional_confusion_df` (Se/Sp measured
    only on test sentences where the true main is 1, since a sub-label's
    classifier can behave differently on "main=1, this sub=0" sentences than
    on "main=0" ones, which the unconditional specificity folds together)
    tightened it further for most labels too.

    KNOWN REMAINING LIMITATION, documented rather than silently shipped: two
    of the most prevalent subs -- Content--Narrative (conditional prior ~0.86,
    i.e. most Content sentences ARE Narrative) and
    Reader_response--Evaluation_of_quality (~0.55) -- still show a residual
    ~-0.03 to -0.04 absolute bias in validation even with both fixes applied,
    specifically where the conditional prior is high. The most likely
    remaining cause: the synthetic corruption used to validate this pipeline
    (see validate_pipeline.py) applies one uniform error rate per label, while
    the real classifier's actual error rate demonstrably differs between the
    "easy" true-negative population (main=0) and the "hard" one (main=1,
    sub=0) -- so the conditional confusion matrix, though real and correctly
    measured, doesn't line up perfectly against a synthetic corruption process
    that can't reproduce that same heterogeneity. For these two labels
    specifically, prefer the aggregate Rogan-Gladen corrected prevalence
    (prevalence_correction.py) over the per-sentence imputed values for any
    silver-set claim that only needs a corpus-level rate -- the aggregate
    method has ~0 validated bias for every label, including these two; only
    the instance-level (per-sentence) values for these two carry this caveat.

    conditional_confusion_df: optional; per-sub Se/Sp conditional on main=1
    (falls back to the unconditional confusion_df for a sub if not provided
    or not present for that label, at the cost of the larger residual bias
    documented in validate_pipeline.py's history without this argument).
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    def label_posterior(label, prior, cm_source):
        code = code_of_label[label]
        cm = cm_source.loc[cm_source["code"] == code]
        col = pred_col_prefix + label
        if cm.empty or label not in priors or col not in df.columns:
            return None
        cm = cm.iloc[0]
        return posterior_true_given_predicted(df[col].to_numpy(), prior, cm["sensitivity"], cm["specificity"])

    main_labels = [l for l in labels if l not in sub_to_main]
    sub_labels = [l for l in labels if l in sub_to_main]

    main_posteriors = {l: label_posterior(l, priors[l], confusion_df) for l in main_labels if l in priors}
    main_posteriors = {l: p for l, p in main_posteriors.items() if p is not None}

    sub_posteriors = {}
    for sub in sub_labels:
        main = sub_to_main[sub]
        if sub not in priors or main not in priors or main not in main_posteriors:
            continue
        conditional_prior = float(np.clip(priors[sub] / priors[main], 1e-4, 1 - 1e-4))
        cm_source = confusion_df
        if conditional_confusion_df is not None and (conditional_confusion_df["code"] == code_of_label[sub]).any():
            cm_source = conditional_confusion_df
        p = label_posterior(sub, conditional_prior, cm_source)
        if p is not None:
            sub_posteriors[sub] = p

    main_order = list(main_posteriors.keys())
    main_matrix = np.column_stack([main_posteriors[l] for l in main_order])

    imputations = []
    for draw in range(m):
        main_draws = rng.random((n, len(main_order))) < main_matrix
        imputed = pd.DataFrame(main_draws.astype(int), columns=main_order, index=df.index)
        for sub, post in sub_posteriors.items():
            main = sub_to_main[sub]
            sub_draw = (rng.random(n) < post).astype(int)
            sub_draw[imputed[main].to_numpy() == 0] = 0  # hard constraint, not a coin flip
            imputed[sub] = sub_draw
        imputations.append(imputed)
    return imputations


def rubin_pool(estimates, variances):
    """Combine M point estimates + their within-imputation variances into one
    pooled estimate, total variance, and 95% CI (Rubin's rules)."""
    estimates = np.asarray(estimates, dtype=float)
    variances = np.asarray(variances, dtype=float)
    m = len(estimates)
    pooled = estimates.mean()
    within = variances.mean()
    between = estimates.var(ddof=1) if m > 1 else 0.0
    total = within + (1 + 1 / m) * between
    se = np.sqrt(total)
    if between > 0 and within > 0:
        r = (1 + 1 / m) * between / within
        df = (m - 1) * (1 + 1 / r) ** 2 if r > 0 else np.inf
    else:
        df = np.inf
    tcrit = stats.t.ppf(0.975, df) if np.isfinite(df) and df > 0 else 1.96
    return {
        "pooled_estimate": pooled, "within_var": within, "between_var": between,
        "total_var": total, "se": se, "ci_lo": pooled - tcrit * se, "ci_hi": pooled + tcrit * se,
        "df": df,
    }


def pool_prevalence(imputations, label):
    """Convenience wrapper: pool a prevalence (proportion) statistic for `label`
    across the M imputed label sets, using each imputation's own binomial
    variance p(1-p)/n as its within-imputation variance."""
    estimates, variances = [], []
    for imp in imputations:
        p = imp[label].mean()
        n = len(imp)
        estimates.append(p)
        variances.append(p * (1 - p) / n)
    return rubin_pool(estimates, variances)
