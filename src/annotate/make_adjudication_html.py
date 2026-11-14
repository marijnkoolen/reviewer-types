"""Build a standalone interactive HTML adjudication page from a batch sheet TSV."""
import json, sys, os as _os
import pandas as pd
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from review_meta import review_headers
batch = sys.argv[1] if len(sys.argv)>1 else 'batch1'
df = pd.read_csv(f'data/adjudication/{batch}_sheet.tsv', sep='\t').fillna('')
meta = review_headers(df.review_id)
labels = [("AUT","Author"),("CLS","Classification"),("OTW","Other works"),("REC","Recommendations"),
 ("NAR","Narrative"),("COO","Content: other"),("QUO","Quote"),("THM","Theme"),
 ("EVA","Evaluation"),("FEE","Feelings"),("IDI","Identification & immersion"),("CTX","Reading context"),
 ("RCP","Reception"),("REF","Reflection"),("SCX","Style: context"),("STR","Style: structure"),("STY","Stylistic features"),("MET","Metadata")]
data = [dict(meta=meta.get(r.review_id, ''), review=r.review_id, sid=r.sent_id, text=r.sent_text, a=[r.anno1.split(), r.anno2.split(), r.anno3.split()],
             gold=r.adjudicated.split(), note=r.note) for r in df.itertuples()]
html = open('src/annotate/adjudication_template.html').read()
html = html.replace('__NAMES__', '["anno1","anno2","anno3"]').replace('__DATA__', json.dumps(data, ensure_ascii=False)).replace('__LABELS__', json.dumps(labels)).replace('__BATCH__', batch)
open(f'data/adjudication/{batch}_adjudicate.html','w').write(html)
print('ok', len(data))
