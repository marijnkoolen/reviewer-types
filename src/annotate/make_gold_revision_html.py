"""Revision page for batch 1-3 gold sentences that differ from the v2.3 full-run labels (UA-013 a).

usage: python src/annotate/make_gold_revision_html.py
input : data/adjudication/batch{1,2,3}_final.tsv (your gold), data/full_run/majority_labels.tsv,
        data/full_run/{opus_a,opus_b,opus_c}/chunk_NNN.json (for the 'why' notes), data/full_run/chunks/
output: data/adjudication/gold_revision.html  -> export gold_revision_adjudicated.tsv
Rows shown per sentence: your gold | v2.3 final (2-of-3 Opus vote) | opus_c (tie-breaker, empty if a and b agreed).
The hint line shows opus_a and opus_b with their notes. Your labels start as your old gold; nothing is marked done.
"""
import csv, glob, json, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_meta import review_headers

CODES = 'AUT CLS OTW REC NAR COO QUO THM EVA FEE IDI CTX RCP REF SCX STR STY MET'.split()
srt = lambda s: sorted(s, key=lambda c: CODES.index(c) if c in CODES else 99)
labels = [("AUT","Author"),("CLS","Classification"),("OTW","Other works"),("REC","Recommendations"),("NAR","Narrative"),
 ("COO","Content: other"),("QUO","Quote"),("THM","Theme"),("EVA","Evaluation"),("FEE","Feelings"),("IDI","Identification & immersion"),
 ("CTX","Reading context"),("RCP","Reception"),("REF","Reflection"),("SCX","Style: context"),("STR","Style: structure"),
 ("STY","Stylistic features"),("MET","Metadata")]

m = pd.read_csv('data/full_run/majority_labels.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False)
final = {r.sent_id: set(filter(None, r.labels.split(','))) for r in m.itertuples()}

runs = {k: {} for k in ('opus_a', 'opus_b', 'opus_c')}
for k in runs:
    for f in glob.glob(f'data/full_run/{k}/chunk_*.json'):
        if '.part' not in f:
            runs[k].update({o['sent_id']: o for o in json.load(open(f))})

order, sents = [], {}          # review_id -> list of (sent_id, text) in order
for f in sorted(glob.glob('data/full_run/chunks/chunk_*.json')):
    for rid, r in json.load(open(f)).items():
        sents[rid] = [(s['sent_id'], s['text']) for s in r['sentences']]

gold = []
for b in (1, 2, 3):
    d = pd.read_csv(f'data/adjudication/batch{b}_final.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\',
                    keep_default_na=False, dtype=str)
    for r in d.itertuples():
        g = set(re.split(r'[\s,;]+', r.adjudicated.strip())) - {''}
        if r.sent_id in final and g != final[r.sent_id]:
            gold.append((b, r.review_id.replace(' (blind)', ''), r.sent_id, g, r.note))

meta = review_headers({g[1] for g in gold})
rows = []
for b, rid, sid, g, note in sorted(gold, key=lambda x: (x[0], x[1], x[2])):
    ss = sents[rid]; i = [s for s, _ in ss].index(sid)
    ab = ' | '.join(f"{k}: {' '.join(srt(runs[k][sid]['labels'])) or '-'}" + (f" ({runs[k][sid]['why']})" if runs[k][sid].get('why') else '')
                    for k in ('opus_a', 'opus_b'))
    cnote = runs['opus_c'].get(sid, {}).get('why', '')
    rows.append(dict(meta=f'Batch {b} · ' + meta.get(rid, ''), review=rid, sid=sid, text=ss[i][1],
                     prev=ss[i-1][1] if i > 0 else '', next=ss[i+1][1] if i + 1 < len(ss) else '',
                     hint=ab + (f' | opus_c note: {cnote}' if cnote else ''),
                     a=[srt(g), srt(final[sid]), srt(runs['opus_c'][sid]['labels']) if sid in runs['opus_c'] else []],
                     gold=srt(g), note=note, done=False))

html = open('src/annotate/adjudication_template.html').read()
html = (html.replace('__NAMES__', json.dumps(['your gold', 'v2.3 final', 'opus_c'])).replace('__DATA__', json.dumps(rows, ensure_ascii=False))
        .replace('__LABELS__', json.dumps(labels)).replace('__BATCH__', 'gold_revision'))
out = 'data/adjudication/gold_revision.html'
open(out, 'w').write(html)
print(out, len(rows), 'sentences in', len({r['review'] for r in rows}), 'reviews')
