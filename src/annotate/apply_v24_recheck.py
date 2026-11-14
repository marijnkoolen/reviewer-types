"""Apply the v2.4 re-check (UA-016) to the full-run labels.

usage: python src/annotate/apply_v24_recheck.py
input : data/full_run/majority_labels.tsv (v2.3), data/full_run/recheck_v24/{pass_x,pass_y}/input_NN.json
output: data/full_run/labels_v24.tsv  = majority_labels.tsv + columns labels_v23, recheck_x, recheck_y, v24_change
A candidate sentence gets the new label set only when both passes propose the same set and it differs from v2.3;
otherwise it keeps its v2.3 labels (v24_change = 'split' when the passes disagree, for a later look).
"""
import csv, glob, json, os
import pandas as pd

CODES = 'AUT CLS OTW REC NAR COO QUO THM EVA FEE IDI CTX RCP REF SCX STR STY MET'.split()
fmt = lambda s: ','.join(sorted(s, key=CODES.index))
D = 'data/full_run/recheck_v24'
p = {}
for k in ('pass_x', 'pass_y'):
    p[k] = {}
    for f in sorted(glob.glob(f'{D}/{k}/input_*.json')):
        p[k].update({o['sent_id']: o for o in json.load(open(f))})
cand = pd.read_csv(f'{D}/candidates.tsv', sep='\t', keep_default_na=False)
missing = [s for s in cand.sent_id if s not in p['pass_x'] or s not in p['pass_y']]
bad = [s for k in p for s, o in p[k].items() if not set(o['labels']) <= set(CODES)]
print(f'{len(cand)} candidates; missing in a pass: {len(missing)}; invalid codes: {len(bad)}')

m = pd.read_csv('data/full_run/majority_labels.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False)
m['labels_v23'] = m['labels']
m['recheck_x'] = m.sent_id.map(lambda s: fmt(set(p['pass_x'][s]['labels'])) if s in p['pass_x'] else '')
m['recheck_y'] = m.sent_id.map(lambda s: fmt(set(p['pass_y'][s]['labels'])) if s in p['pass_y'] else '')
m['v24_change'] = ''
c = set(cand.sent_id) - set(missing)
for i, r in m[m.sent_id.isin(c)].iterrows():
    if r.recheck_x == r.recheck_y:
        if r.recheck_x != r.labels_v23:
            m.at[i, 'labels'] = r.recheck_x
            m.at[i, 'v24_change'] = 'changed'
        else:
            m.at[i, 'v24_change'] = 'kept'
    else:
        m.at[i, 'v24_change'] = 'split'
m.to_csv('data/full_run/labels_v24.tsv', sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
print(m.v24_change.value_counts().to_dict())
rule = {s: (p['pass_x'][s].get('rule') or p['pass_y'][s].get('rule')) for s in m.sent_id[m.v24_change == 'changed']}
print('changes by rule:', pd.Series(rule).value_counts().to_dict())
old = m.labels_v23.str.split(',').explode().value_counts(); new = m.labels.str.split(',').explode().value_counts()
print('label counts v2.3 -> v2.4:', {c: f'{old.get(c, 0)}->{new.get(c, 0)}' for c in CODES if old.get(c, 0) != new.get(c, 0)})
