"""Revision page: sentences where adjudicated labels and an agent disagree on selected labels."""
import json, sys, os as _os
import pandas as pd
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from review_meta import review_headers
sys.path.insert(0,'.')
base='data/adjudication/'
g=pd.read_csv(base+'batch1_adjudicated.tsv',sep='\t').fillna('')
headers = review_headers(g.review_id)
agents={n:{x['sent_id']:x for x in json.load(open(f'{base}agent_runs/batch1_{n}.json'))} for n in ['opus','sonnet']}
focus={'REF','FEE','IDI','SCX','CTX','MET'}
labels=[("AUT","Author"),("CLS","Classification"),("OTW","Other works"),("REC","Recommendations"),("NAR","Narrative"),
 ("COO","Content: other"),("QUO","Quote"),("THM","Theme"),("EVA","Evaluation"),("FEE","Feelings"),("IDI","Identification & immersion"),
 ("CTX","Reading context"),("RCP","Reception"),("REF","Reflection"),("SCX","Style: context"),("STR","Style: structure"),
 ("STY","Stylistic features"),("MET","Metadata (new)")]
hints={'impfic-review-2438-sent-011':'Metadata candidate (edition/print info).'}
rows=[]; prev_r=None
sents=list(g.itertuples())
for i,r in enumerate(sents):
    gold=set(r.adjudicated.split()); op=set(agents['opus'][r.sent_id]['labels']); so=set(agents['sonnet'][r.sent_id]['labels'])
    diff=(gold^op)&focus
    anyscx='SCX' in gold|op|so or 'SCX' in set(r.anno1.split())|set(r.anno2.split())|set(r.anno3.split())
    meta = r.sent_id in hints or 'vertaal' in r.sent_text.lower()
    if not(diff or anyscx or meta): continue
    prev=sents[i-1].sent_text if i>0 and sents[i-1].review_id==r.review_id else ''
    nxt=sents[i+1].sent_text if i+1<len(sents) and sents[i+1].review_id==r.review_id else ''
    why=agents['opus'][r.sent_id]['why']
    hint=f"Differs on: {', '.join(sorted(diff)) or '—'}" + (f" · Opus: {why}" if why else '')+(f" · {hints[r.sent_id]}" if r.sent_id in hints else '')
    rows.append(dict(meta=headers.get(r.review_id, ''), review=r.review_id,sid=r.sent_id,text=r.sent_text,prev=prev,next=nxt,hint=hint,
        a=[sorted(gold),sorted(op),sorted(so)],gold=sorted(gold),note=r.note,done=False))
names=['previous','opus','sonnet']
html=open('src/annotate/adjudication_template.html').read()
html=(html.replace('__NAMES__',json.dumps(names)).replace('__DATA__',json.dumps(rows,ensure_ascii=False))
      .replace('__LABELS__',json.dumps(labels)).replace('__BATCH__','batch1_revision'))
open(base+'batch1_revision.html','w').write(html)
print(len(rows),'sentences')
