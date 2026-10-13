"""Per-label confusion matrix from the held-out test set.

The model is trained as one binary present/not-present task per category
(MaChAmp multi-task), so there is no single joint predicted-class-vs-true-class
matrix across the 20 categories -- a 20x20 confusion matrix isn't meaningful
here. What *is* meaningful, and what the correction methods in the analysis
plan (Rogan-Gladen aggregate correction, latent-class multiple imputation) need,
is one 2x2 confusion matrix per label: TP/FP/FN/TN, sensitivity, specificity,
precision, F1.

Also computes a second, CONDITIONAL confusion matrix for every sub-label,
restricted to test sentences where the true main label is 1 -- imputation.py's
multiple imputation draws a sub-label's value conditional on its main already
having been drawn 1, and validation showed the unconditional Se/Sp don't
transfer to that conditional population cleanly (the classifier can behave
differently on "main=1, this sub=0" sentences than on "main=0" sentences,
which are folded together in the unconditional specificity). The conditional
matrix estimates Se/Sp on exactly the population the conditional imputation
step actually needs.

Usage: python src/correction/confusion_matrix.py
Writes results/correction/confusion_matrix.csv and results/correction/confusion_matrix_conditional.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.schema import ALL_CODES, SUB_TO_MAIN, LABEL_TO_CODE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "data" / "ground_truth-test_set.tsv"
PRED_PATH = ROOT / "data" / "predictions-test_set.tsv"
OUT_PATH = ROOT / "results" / "correction/confusion_matrix.csv"
OUT_COND_PATH = ROOT / "results" / "correction/confusion_matrix_conditional.csv"


def load(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return {(row["review_id"], row["sent_num"]): row for row in reader}


def confusion_row(code, gt, pred, keys):
    tp = fp = fn = tn = 0
    for key in keys:
        g = gt[key][code] == "1"
        p = pred[key][code] == "1"
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
        else:
            tn += 1
    n = len(keys)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec and sens and (prec + sens) else float("nan")
    return {
        "code": code, "n_test": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "prevalence_gt": round((tp + fn) / n, 4) if n else float("nan"),
        "prevalence_pred": round((tp + fp) / n, 4) if n else float("nan"),
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "precision": round(prec, 4), "f1": round(f1, 4),
    }


def main():
    gt = load(GT_PATH)
    pred = load(PRED_PATH)

    missing_from_pred = set(gt) - set(pred)
    if missing_from_pred:
        print(f"note: {len(missing_from_pred)} test-set sentences have no matching "
              f"prediction row, dropped: {sorted(missing_from_pred)}", file=sys.stderr)

    joined = sorted(set(gt) & set(pred))
    n = len(joined)

    rows = [confusion_row(code, gt, pred, joined) for code in ALL_CODES]
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_PATH} ({n} joined test-set sentences)")

    cond_rows = []
    for sub, main in SUB_TO_MAIN.items():
        main_code = LABEL_TO_CODE[main]
        keys_main1 = [k for k in joined if gt[k][main_code] == "1"]
        row = confusion_row(LABEL_TO_CODE[sub], gt, pred, keys_main1)
        row["main_label"] = main
        cond_rows.append(row)
    with open(OUT_COND_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cond_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cond_rows)
    print(f"wrote {OUT_COND_PATH} ({len(cond_rows)} sub-labels, conditional on their main=1)")


if __name__ == "__main__":
    main()
