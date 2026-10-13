"""Validates the Phase B correction pipeline without touching the silver set
at all: corrupt the *known-true* gold sentence labels using each label's own
sensitivity/specificity (simulating what RobBERT's predictions would look
like if it had been run on this same gold data), then check whether the
Rogan-Gladen and multiple-imputation corrections recover the real gold
prevalence better than the naive (uncorrected) corrupted estimate does.

Two things get validated:
1. Aggregate correction (prevalence_correction.py) -- bias and 95% CI coverage
   across many repeated corruption draws, since corruption is stochastic.
2. Instance-level correction (imputation.py) -- same bias/coverage check for
   pooled (Rubin's rules) prevalence, run through the realistic full chain
   (the R-G corrected estimate feeds the imputation step's prior, exactly as
   Phase C will use it), plus a check that sentence-level co-occurrence
   structure (not just a corpus-wide rate) is better recovered by imputation
   than by the naive corrupted data.

Usage: python src/correction/validate_pipeline.py
Writes results/gold_correction_validation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SUB_TO_MAIN, LABEL_TO_CODE, WEAK_MODEL_LABELS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prevalence_correction import rogan_gladen, bootstrap_corrected_prevalence  # noqa: E402
from imputation import multiple_impute, pool_prevalence  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
ALL_LABELS = MAIN_LABELS_USED + SUB_LABELS
R_BIAS = 300         # repeats for the fast point-correction bias check
R_COVERAGE = 100     # repeats for the (slower, bootstrapped) CI coverage check
R_IMPUTATION = 50    # repeats for the full corrupt->correct->impute->pool chain
M_IMPUTATIONS = 20
SEED = 0


def load():
    sent = pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
    confusion = pd.read_csv(RESULTS / "correction/confusion_matrix.csv")
    confusion_conditional = pd.read_csv(RESULTS / "correction/confusion_matrix_conditional.csv")
    return sent, confusion, confusion_conditional


def cm_lookup(confusion):
    """label -> confusion-matrix row, keyed by full label name instead of code."""
    by_code = confusion.set_index("code")
    return {label: by_code.loc[code] for label, code in LABEL_TO_CODE.items() if code in by_code.index}


def corrupt(true, sensitivity, specificity, rng):
    true = np.asarray(true)
    observed = np.empty_like(true)
    pos = true == 1
    observed[pos] = rng.random(pos.sum()) < sensitivity
    observed[~pos] = rng.random((~pos).sum()) < (1 - specificity)
    return observed.astype(int)


def aggregate_validation(sent, confusion):
    rng = np.random.default_rng(SEED)
    cms = cm_lookup(confusion)
    rows = []
    for label in ALL_LABELS:
        if label not in cms:
            continue
        cm = cms[label]
        se, sp = cm["sensitivity"], cm["specificity"]
        tp, fp, fn, tn = int(cm["tp"]), int(cm["fp"]), int(cm["fn"]), int(cm["tn"])
        true = sent[label].to_numpy()
        p_true = true.mean()

        naive_biases, corrected_biases = [], []
        for _ in range(R_BIAS):
            observed = corrupt(true, se, sp, rng)
            p_obs = observed.mean()
            p_corr, _, _ = rogan_gladen(p_obs, se, sp)
            naive_biases.append(p_obs - p_true)
            corrected_biases.append(p_corr - p_true)

        n_covered = 0
        for _ in range(R_COVERAGE):
            observed = corrupt(true, se, sp, rng)
            result = bootstrap_corrected_prevalence(observed.astype(float), tp, fp, fn, tn,
                                                      n_boot=500, seed=rng.integers(1e9))
            if result["ci_lo"] <= p_true <= result["ci_hi"]:
                n_covered += 1

        rows.append({
            "label": label, "p_true": round(p_true, 4),
            "sensitivity": se, "specificity": sp,
            "mean_naive_bias": round(np.mean(naive_biases), 4),
            "mean_corrected_bias": round(np.mean(corrected_biases), 4),
            "bias_reduction_pct": round(100 * (1 - abs(np.mean(corrected_biases)) / abs(np.mean(naive_biases))), 1)
                                   if np.mean(naive_biases) != 0 else float("nan"),
            "ci_coverage_95pct_nominal": round(n_covered / R_COVERAGE, 3),
            "weak_model_label": label in WEAK_MODEL_LABELS,
        })
    return pd.DataFrame(rows).sort_values("mean_corrected_bias", key=abs)


def _corrupt_all_labels(sent, cms, rng):
    return pd.DataFrame({l: corrupt(sent[l].to_numpy(), cms[l]["sensitivity"], cms[l]["specificity"], rng)
                          for l in ALL_LABELS if l in cms})


def _rogan_gladen_priors(corrupted, cms):
    return {l: float(np.clip(rogan_gladen(corrupted[l].mean(), cms[l]["sensitivity"], cms[l]["specificity"])[0],
                              0.01, 0.99))
            for l in corrupted.columns}


def instance_level_validation(sent, confusion, confusion_conditional):
    rng = np.random.default_rng(SEED)
    cms = cm_lookup(confusion)
    code_of_label = {l: LABEL_TO_CODE[l] for l in ALL_LABELS if l in cms}
    true_prevalence = {l: sent[l].mean() for l in ALL_LABELS if l in cms}

    per_label = {l: {"pooled_biases": [], "naive_biases": [], "n_covered": 0} for l in code_of_label}

    for _ in range(R_IMPUTATION):
        corrupted = _corrupt_all_labels(sent, cms, rng)
        priors = _rogan_gladen_priors(corrupted, cms)
        imputations = multiple_impute(corrupted, list(code_of_label), code_of_label, confusion, priors,
                                       SUB_TO_MAIN, m=M_IMPUTATIONS, seed=rng.integers(1e9),
                                       conditional_confusion_df=confusion_conditional)
        for label in code_of_label:
            p_obs = corrupted[label].mean()
            pooled = pool_prevalence(imputations, label)
            p_true = true_prevalence[label]
            per_label[label]["pooled_biases"].append(pooled["pooled_estimate"] - p_true)
            per_label[label]["naive_biases"].append(p_obs - p_true)
            if pooled["ci_lo"] <= p_true <= pooled["ci_hi"]:
                per_label[label]["n_covered"] += 1

    rows = []
    for label, acc in per_label.items():
        rows.append({
            "label": label, "p_true": round(true_prevalence[label], 4),
            "mean_naive_bias": round(np.mean(acc["naive_biases"]), 4),
            "mean_pooled_imputed_bias": round(np.mean(acc["pooled_biases"]), 4),
            "ci_coverage_95pct_nominal": round(acc["n_covered"] / R_IMPUTATION, 3),
            "weak_model_label": label in WEAK_MODEL_LABELS,
        })
    return pd.DataFrame(rows).sort_values("mean_pooled_imputed_bias", key=abs)


def cooccurrence_recovery_demo(sent, confusion, confusion_conditional):
    """Illustrative, not exhaustive: does imputation recover sentence-level
    co-occurrence structure (not just a corpus-wide rate) better than the
    naive corrupted data does, for a few label pairs already known to matter
    from ANALYSIS_PLAN.md's gold-set co-occurrence findings?"""
    rng = np.random.default_rng(SEED)
    cms = cm_lookup(confusion)
    code_of_label = {l: LABEL_TO_CODE[l] for l in ALL_LABELS if l in cms}
    pairs = [("Style", "Reader_response--Evaluation_of_quality"),
             ("Content", "Other_works"),
             ("Content--Narrative", "Reader_response--Reading_Context")]

    corrupted = _corrupt_all_labels(sent, cms, rng)
    priors = _rogan_gladen_priors(corrupted, cms)
    imputations = multiple_impute(corrupted, list(code_of_label), code_of_label, confusion, priors,
                                   SUB_TO_MAIN, m=M_IMPUTATIONS, seed=rng.integers(1e9),
                                   conditional_confusion_df=confusion_conditional)

    def cooc_rate(df, a, b):
        return ((df[a] == 1) & (df[b] == 1)).mean()

    rows = []
    for a, b in pairs:
        true_rate = cooc_rate(sent, a, b)
        naive_rate = cooc_rate(corrupted, a, b)
        imputed_rate = np.mean([cooc_rate(imp, a, b) for imp in imputations])
        rows.append({
            "label_a": a, "label_b": b, "true_cooccurrence": round(true_rate, 4),
            "naive_corrupted_cooccurrence": round(naive_rate, 4),
            "naive_bias": round(naive_rate - true_rate, 4),
            "imputed_pooled_cooccurrence": round(imputed_rate, 4),
            "imputed_bias": round(imputed_rate - true_rate, 4),
        })
    return pd.DataFrame(rows)


def main():
    sent, confusion, confusion_conditional = load()

    print("=== aggregate correction validation (Rogan-Gladen) ===")
    agg = aggregate_validation(sent, confusion)
    agg.to_csv(RESULTS / "correction/gold_correction_validation_aggregate.csv", index=False)
    print(agg.to_string(index=False))
    print(f"\nmean |bias| naive: {agg['mean_naive_bias'].abs().mean():.4f}, "
          f"corrected: {agg['mean_corrected_bias'].abs().mean():.4f}")
    print(f"mean CI coverage (nominal 95%): {agg['ci_coverage_95pct_nominal'].mean():.3f}")
    weak = agg[agg["weak_model_label"]]
    strong = agg[~agg["weak_model_label"]]
    print(f"WEAK_MODEL_LABELS mean |corrected bias|: {weak['mean_corrected_bias'].abs().mean():.4f} "
          f"vs. other labels: {strong['mean_corrected_bias'].abs().mean():.4f}")

    print("\n=== instance-level correction validation (multiple imputation + Rubin's rules) ===")
    inst = instance_level_validation(sent, confusion, confusion_conditional)
    inst.to_csv(RESULTS / "correction/gold_correction_validation_instance.csv", index=False)
    print(inst.to_string(index=False))
    print(f"\nmean |bias| naive: {inst['mean_naive_bias'].abs().mean():.4f}, "
          f"pooled-imputed: {inst['mean_pooled_imputed_bias'].abs().mean():.4f}")
    print(f"mean CI coverage (nominal 95%): {inst['ci_coverage_95pct_nominal'].mean():.3f}")

    print("\n=== sentence-level co-occurrence recovery (illustrative pairs) ===")
    cooc = cooccurrence_recovery_demo(sent, confusion, confusion_conditional)
    cooc.to_csv(RESULTS / "correction/gold_correction_validation_cooccurrence.csv", index=False)
    print(cooc.to_string(index=False))

    print(f"\nwrote results to {RESULTS} (gold_correction_validation_*.csv)")


if __name__ == "__main__":
    main()
