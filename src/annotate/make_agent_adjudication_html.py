"""Adjudication page over agent annotations (no human labels shown).

usage: python src/annotate/make_agent_adjudication_html.py BATCH RUN1 RUN2 RUN3 [--blind N]

Your labels start as the agents' majority vote; sentences where the agents disagree are tinted.
--blind N: the last N reviews show no agent labels (and no prefill), so the effect of seeing
agent labels on your judgements can be measured.
--meta-chunks DIR: take the review headers from the chunk JSONs in DIR (Phase 4 languages, whose reviews are
not in metadata/review_metadata.tsv.gz).
"""
import json, sys, argparse
import pandas as pd, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from review_meta import review_headers

ap = argparse.ArgumentParser()
ap.add_argument('batch'); ap.add_argument('runs', nargs='+'); ap.add_argument('--blind', type=int, default=0)
ap.add_argument('--meta-chunks', default=None)
a = ap.parse_args()
base = 'data/adjudication/'
df = pd.read_csv(f'{base}{a.batch}_sheet.tsv', sep='\t').fillna('')
if a.meta_chunks:
    meta = {rid: v['meta'] for f in sorted(_os.listdir(a.meta_chunks)) if f.endswith('.json')
            for rid, v in json.load(open(f'{a.meta_chunks}/{f}')).items()}
else:
    meta = review_headers(df.review_id)
runs = {r: {x['sent_id']: x for x in json.load(open(f'{base}agent_runs/{a.batch}_{r}.json'))} for r in a.runs}
labels = [("AUT","Author"),("CLS","Classification"),("OTW","Other works"),("REC","Recommendations"),("NAR","Narrative"),
 ("COO","Content: other"),("QUO","Quote"),("THM","Theme"),("EVA","Evaluation"),("FEE","Feelings"),("IDI","Identification & immersion"),
 ("CTX","Reading context"),("RCP","Reception"),("REF","Reflection"),("SCX","Style: context"),("STR","Style: structure"),
 ("STY","Stylistic features"),("MET","Metadata")]
reviews = list(dict.fromkeys(df.review_id))
blind = set(reviews[len(reviews) - a.blind:]) if a.blind else set()
rows, sents = [], list(df.itertuples())
for i, r in enumerate(sents):
    labs = [sorted(runs[n][r.sent_id]['labels']) for n in a.runs]
    whys = [f"{n}: {runs[n][r.sent_id]['why']}" for n in a.runs if runs[n][r.sent_id].get('why')]
    maj = sorted(c for c, _ in labels if sum(c in l for l in labs) * 2 > len(labs))
    b = r.review_id in blind
    prev = sents[i-1].sent_text if i > 0 and sents[i-1].review_id == r.review_id else ''
    nxt = sents[i+1].sent_text if i+1 < len(sents) and sents[i+1].review_id == r.review_id else ''
    rows.append(dict(meta=meta.get(r.review_id, ''), review=r.review_id + (' (blind)' if b else ''), sid=r.sent_id, text=r.sent_text, prev=prev, next=nxt,
                     hint='' if b else ' · '.join(whys), a=[[] for _ in a.runs] if b else labs,
                     gold=[] if b else maj, note='', done=False))
# note: blind reviews carry ' (blind)' in review_id in the export; strip it when reading back
html = open('src/annotate/adjudication_template.html').read()
html = (html.replace('__NAMES__', json.dumps(a.runs)).replace('__DATA__', json.dumps(rows, ensure_ascii=False))
        .replace('__LABELS__', json.dumps(labels)).replace('__BATCH__', a.batch + '_agents'))
out = f'{base}{a.batch}_agents_adjudicate.html'
open(out, 'w').write(html)
print(out, len(rows), 'sentences;', len(blind), 'blind reviews')
