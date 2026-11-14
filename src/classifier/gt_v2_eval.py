"""Shared evaluation for the gt_v2 sentence classifiers: threshold tuning on dev, per-label metrics."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from gt_v2_data import LABELS, OUT, to_matrix


def tune_thresholds(Y, P, grid=np.arange(0.05, 0.96, 0.025)):
    """Per-label threshold that maximises F1 on dev (falls back to 0.5 when the label is absent)."""
    th = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        if Y[:, j].sum() == 0:
            continue
        f = [f1_score(Y[:, j], P[:, j] >= t, zero_division=0) for t in grid]
        th[j] = grid[int(np.argmax(f))]
    return th


def metrics(Y, Yhat, labels=LABELS):
    p, r, f, s = precision_recall_fscore_support(Y, Yhat, zero_division=0)
    rows = pd.DataFrame({'label': labels, 'support': s, 'n_pred': Yhat.sum(0), 'precision': p, 'recall': r, 'f1': f})
    have = s > 0
    summ = {
        'n_sentences': int(len(Y)),
        'micro_f1': float(f1_score(Y, Yhat, average='micro', zero_division=0)),
        'macro_f1': float(f[have].mean()),           # over labels present in this set
        'samples_f1': float(f1_score(Y, Yhat, average='samples', zero_division=0)),
        'exact_match': float((Y == Yhat).all(1).mean()),
    }
    return rows, summ


def evaluate_all(name, d, P, th):
    """d: sentence table (gt_v2_data.load()) aligned with probabilities P. Writes per-label and summary files."""
    out = f'{OUT}/{name}'
    os.makedirs(out, exist_ok=True)
    Yhat = (P >= th).astype(int)
    Y = to_matrix(d.labels)
    sets = {
        'test': d.split == 'test',
        'test_original': (d.split == 'test') & (d.stratum == 'original'),
        'test_random': (d.split == 'test') & (d.stratum == 'random'),
        'test_enriched': (d.split == 'test') & d.stratum.isin(['MET', 'RCP', 'REF', 'SCX']),
        'dev': d.split == 'dev',
    }
    summary, per_label = {}, []
    for k, m in sets.items():
        m = m.to_numpy()
        rows, s = metrics(Y[m], Yhat[m])
        summary[k] = s
        per_label.append(rows.assign(eval_set=k))
    g = (d.split == 'gold').to_numpy()
    Yg = to_matrix(d.gold[g])
    rows, s = metrics(Yg, Yhat[g])
    summary['gold_human'] = s
    per_label.append(rows.assign(eval_set='gold_human'))
    rows, s = metrics(Yg, Y[g])  # ceiling: the v2.4 ground truth itself against the human gold
    summary['gold_human__v24_labels'] = s
    per_label.append(rows.assign(eval_set='gold_human__v24_labels'))
    pl = pd.concat(per_label)
    pl.to_csv(f'{out}/per_label.tsv', sep='\t', index=False, float_format='%.4f')
    with open(f'{out}/summary.json', 'w') as fh:
        json.dump({'thresholds': dict(zip(LABELS, map(float, th))), 'sets': summary}, fh, indent=2)
    pred = d[['review_id', 'sent_id', 'split', 'stratum']].copy()
    for j, l in enumerate(LABELS):
        pred[f'p_{l}'] = P[:, j].round(4)
    pred['pred'] = [','.join(l for l, v in zip(LABELS, row) if v) for row in Yhat]
    pred.to_csv(f'{out}/predictions.tsv.gz', sep='\t', index=False)
    return summary, pl
