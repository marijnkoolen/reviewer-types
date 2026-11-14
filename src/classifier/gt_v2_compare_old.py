"""Compare new classifiers with the old RobBERT predictions on the user's human-gold sentences.

Old predictions: data/predctions-robbert-sentences-all.tsv.gz (old label scheme, binary, no MET), joined on
review_id + sentence number. Reviews whose old sentence count differs from the gold segmentation are dropped
(1 review), so every system is scored on the same sentences. MET is left out of the comparison.
26 of the 30 gold reviews are in the old model's test set (data/ground_truth-test_set.tsv); the other 4 may have
been in its training data, so results are also given on the old-test-set reviews only ('*_oldtest').

Run: python src/classifier/gt_v2_compare_old.py [model_name ...]   (default: every results dir with predictions)
"""
import glob
import os
import sys

import pandas as pd

from gt_v2_data import LABELS, OLD, OUT, read_gold, to_matrix
from gt_v2_eval import metrics

CMP = [l for l in LABELS if l in OLD]


def main(names):
    g = read_gold()
    g['sent_num'] = g.sent_id.str[-3:].astype(int)
    old = pd.read_csv('data/predctions-robbert-sentences-all.tsv.gz', sep='\t')
    old = old[old.review_id.isin(g.review_id)]
    n_gold = g.groupby('review_id').size()
    n_old = old.groupby('review_id').num_sentences.first()
    keep = n_gold.index[n_gold.reindex(n_old.index).eq(n_old).reindex(n_gold.index, fill_value=False)]
    g = g[g.review_id.isin(keep)].merge(old, on=['review_id', 'sent_num'], how='left')
    gi = [LABELS.index(l) for l in CMP]
    Yg = to_matrix(g.gold)[:, gi]
    systems = {'old_robbert': g[[OLD[l] for l in CMP]].to_numpy().astype(int)}
    for n in names:
        p = pd.read_csv(f'{OUT}/{n}/predictions.tsv.gz', sep='\t', keep_default_na=False).set_index('sent_id')
        systems[n] = to_matrix(p.pred.reindex(g.sent_id))[:, gi]
    old_test = set(pd.read_csv('data/ground_truth-test_set.tsv', sep='\t', usecols=['review_id']).review_id)
    m = g.review_id.isin(old_test).to_numpy()
    rows, summ = [], []
    for n, Yhat in systems.items():
        for suffix, mask in [('', slice(None)), ('_oldtest', m)]:
            r, s = metrics(Yg[mask], Yhat[mask], CMP)
            rows.append(r.assign(system=n + suffix)); summ.append({'system': n + suffix, **s})
    rows = pd.concat(rows)
    summ = pd.DataFrame(summ)
    os.makedirs(f'{OUT}/compare_old', exist_ok=True)
    rows.to_csv(f'{OUT}/compare_old/per_label.tsv', sep='\t', index=False, float_format='%.4f')
    summ.to_csv(f'{OUT}/compare_old/summary.tsv', sep='\t', index=False, float_format='%.4f')
    print(f'{len(g)} gold sentences in {g.review_id.nunique()} reviews; labels: {len(CMP)} (no MET)')
    print(summ.round(3).to_string(index=False))
    print(rows.pivot(index='label', columns='system', values='f1').reindex(CMP).round(3).to_string())


if __name__ == '__main__':
    names = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p))
                                   for p in glob.glob(f'{OUT}/*/predictions.tsv.gz') if '/smoke/' not in p)
    main(names)
