"""Data for the sentence classifier trained on the v2.4 ground truth (data/gt_v2/labels.tsv).

- Split by REVIEW (never by sentence), stratified by `stratum` (original, random, MET, RCP, REF, SCX),
  so every stratum is in train, dev and test.
- The 30 reviews that hold the user's 348 human-gold sentences (data/adjudication/gold_batches1-3.tsv)
  are kept out of train/dev/test entirely and form a separate evaluation set ('gold').
- Model input uses only the context the labellers had: the review text plus the header from
  src/annotate/review_meta.py (genre, rating, review year). No outside knowledge.

Run once to write the split:  python src/classifier/gt_v2_data.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'annotate'))
from review_meta import review_headers  # noqa: E402

LABELS = ['AUT', 'CLS', 'OTW', 'REC', 'NAR', 'COO', 'QUO', 'THM', 'EVA', 'FEE', 'IDI', 'CTX', 'RCP', 'REF',
          'SCX', 'STR', 'STY', 'MET']
GT = 'data/gt_v2/labels.tsv'
GOLD = 'data/adjudication/gold_batches1-3.tsv'
OUT = 'results/gt_v2_classifier'
SPLIT = f'{OUT}/split.tsv'
SEED = 20261114
FRAC = {'train': 0.8, 'dev': 0.1, 'test': 0.1}

# old RobBERT output columns (data/predctions-robbert-sentences-all.tsv.gz) -> v2.4 codes (no MET in the old scheme)
OLD = {'AUT': 'aut', 'CLS': 'cla', 'OTW': 'oth', 'REC': 'rec', 'NAR': 'con_nar', 'COO': 'con_oth', 'QUO': 'con_quo',
       'THM': 'con_the', 'EVA': 'rea_eva', 'FEE': 'rea_fee', 'IDI': 'rea_ide', 'CTX': 'rea_rea', 'RCP': 'rea_rec',
       'REF': 'rea_ref', 'SCX': 'sty_con', 'STR': 'sty_str', 'STY': 'sty_sty'}


def read_gt():
    return pd.read_csv(GT, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False)


def read_gold():
    return pd.read_csv(GOLD, sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False)


def to_matrix(label_strings, sep=','):
    Y = np.zeros((len(label_strings), len(LABELS)), dtype=np.int8)
    idx = {l: i for i, l in enumerate(LABELS)}
    for r, s in enumerate(label_strings):
        for l in s.replace(',', ' ').split():
            Y[r, idx[l]] = 1
    return Y


def make_split():
    d = read_gt()
    gold_reviews = set(read_gold().review_id)
    rev = d.drop_duplicates('review_id')[['review_id', 'set', 'stratum']].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    rev['split'] = ''
    rev.loc[rev.review_id.isin(gold_reviews), 'split'] = 'gold'
    for _, grp in rev[rev.split == ''].groupby('stratum'):
        ids = grp.index.to_numpy()
        rng.shuffle(ids)
        n_tr = int(round(FRAC['train'] * len(ids)))
        n_dev = int(round(FRAC['dev'] * len(ids)))
        rev.loc[ids[:n_tr], 'split'] = 'train'
        rev.loc[ids[n_tr:n_tr + n_dev], 'split'] = 'dev'
        rev.loc[ids[n_tr + n_dev:], 'split'] = 'test'
    os.makedirs(OUT, exist_ok=True)
    rev.to_csv(SPLIT, sep='\t', index=False)
    return rev


def load():
    """Sentence table with split, header, position and gold labels (for the gold reviews)."""
    d = read_gt()
    split = pd.read_csv(SPLIT, sep='\t', keep_default_na=False)
    d = d.merge(split[['review_id', 'split']], on='review_id', how='left')
    hdr = review_headers(d.review_id.unique())
    d['header'] = d.review_id.map(hdr).fillna('Genre: unknown · Rating: unknown · Review year: unknown')
    d['pos'] = d.groupby('review_id').cumcount()
    d['n_sent'] = d.groupby('review_id').sent_id.transform('size')
    g = read_gold().set_index('sent_id').gold
    d['gold'] = d.sent_id.map(g).fillna('')
    return d


if __name__ == '__main__':
    if os.path.exists(SPLIT):
        sys.exit(f'{SPLIT} exists; the split is fixed. Delete it deliberately to redraw.')
    rev = make_split()
    print(pd.crosstab(rev.stratum, rev.split, margins=True))
    d = load()
    print(pd.crosstab(d.stratum, d.split, margins=True))
