"""Combine the agent-labelled sets into one ground-truth file (codebook v2.4).

usage: python src/annotate/build_ground_truth.py
input : data/full_run/labels_v24.tsv   (1,250 originally annotated reviews; v2.3 run + v2.4 re-check)
        data/phase3/majority_labels.tsv (2,600 Phase 3 reviews, labelled under v2.4)
        data/phase3/sample.tsv          (stratum and inclusion probability per Phase 3 review)
output: data/gt_v2/labels.tsv  columns review_id sent_id sent_text labels status set stratum inclusion_prob
        (tab-separated, no quoting; read with quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False)
set: 'original' (the 1,250 reviews with three human annotations as well) or 'phase3'.
stratum/inclusion_prob: Phase 3 sampling design (random, RCP, REF, SCX, MET); for 'original' stratum = 'original'.
Prevalence estimates for the corpus should use only set=original or stratum=random (or weight by 1/inclusion_prob).
"""
import csv, os
import pandas as pd

rd = lambda f: pd.read_csv(f, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False)
a = rd('data/full_run/labels_v24.tsv')[['review_id', 'sent_id', 'sent_text', 'labels', 'status']]
a['set'], a['stratum'], a['inclusion_prob'] = 'original', 'original', ''
b = rd('data/phase3/majority_labels.tsv')[['review_id', 'sent_id', 'sent_text', 'labels', 'status']]
s = pd.read_csv('data/phase3/sample.tsv', sep='\t').set_index('review_id')
b['set'] = 'phase3'
b['stratum'] = b.review_id.map(s.stratum)
b['inclusion_prob'] = b.review_id.map(s.inclusion_prob).map(lambda x: f'{x:.6g}')
g = pd.concat([a, b], ignore_index=True)
assert not g.sent_id.duplicated().any()
os.makedirs('data/gt_v2', exist_ok=True)
g.to_csv('data/gt_v2/labels.tsv', sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

CODES = 'AUT CLS OTW REC NAR COO QUO THM EVA FEE IDI CTX RCP REF SCX STR STY MET'.split()
ex = g.assign(l=g.labels.str.split(',')).explode('l')
t = ex[ex.l != ''].groupby(['l', 'set']).size().unstack(fill_value=0).reindex(CODES)
t['total'] = t.sum(axis=1)
print(f"{g.review_id.nunique()} reviews, {len(g)} sentences ({(g.set == 'original').sum()} original, {(g.set == 'phase3').sum()} phase3)")
print(t.to_string())
