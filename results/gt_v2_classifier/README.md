# Sentence classifier on the v2.4 ground truth (`data/gt_v2/labels.tsv`)

Scripts: `src/classifier/` (run from the project root with `/opt/homebrew/opt/python@3.11/bin/python3.11`; `venv/` has no torch).

| step | script | output |
|---|---|---|
| split (fixed, seed 20261114) | `gt_v2_data.py` | `split.tsv` |
| TF-IDF + one-vs-rest LR | `gt_v2_baseline.py` (~30 s) | `tfidf_lr/` |
| RobBERT fine-tune | `gt_v2_transformer.py` (76 min on MPS, 4 epochs; log `robbert_train_rerun.log`) | `robbert/` (model weights not in git) |
| vs old RobBERT on human gold | `gt_v2_compare_old.py [names]` | `compare_old/` |

Split: by review, stratified by `stratum`; 80/10/10 → train 3,056 / dev 382 / test 382 reviews
(22,778 / 3,521 / 2,815 sentences). The 30 reviews holding the 348 human-gold sentences are split `gold`,
never trained or tuned on.

Per model dir: `summary.json` (dev-tuned thresholds + micro/macro F1 per eval set), `per_label.tsv`
(per label × eval set: support, n_pred, P, R, F1), `predictions.tsv.gz` (probabilities + predicted labels for all
29,462 sentences). Eval sets: `test`, `test_original`, `test_random`, `test_enriched`, `dev`, `gold_human`
(model vs your gold), `gold_human__v24_labels` (the v2.4 labels vs your gold = ceiling).

`smoke/` is a 30-step GysBERT pipeline test, not a result.

## Results (micro / macro F1, dev-tuned thresholds)

| eval set | sentences | TF-IDF + LR | RobBERT |
|---|---|---|---|
| test | 2,815 | 0.72 / 0.63 | 0.80 / 0.73 |
| test, random stratum | 596 | 0.67 / 0.60 | 0.77 / 0.69 |
| your gold (`gold_human`) | 348 | 0.69 / 0.56 | 0.77 / 0.69 |
| ceiling: v2.4 labels vs your gold | 348 | 0.97 / 0.97 | |

Per label: `*/per_label.tsv`. RobBERT's weakest labels on test are REF 0.49, SCX 0.52 and STR 0.63; QUO rose from 0.36 (TF-IDF) to 0.76.
`robbert_crash_2ep/` (not in git) is the run that crashed on 2026-11-14 after 2 epochs.
