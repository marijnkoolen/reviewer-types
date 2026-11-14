"""Write the blind agent input for a batch: reviews with metadata header and sentences, no labels.

usage: python src/annotate/make_agent_input.py BATCH
output: data/adjudication/agent_runs/BATCH_input.json
  {review_id: {"meta": "Genre: ... · Rating: ...", "sentences": [{"sent_id", "text"}, ...]}}
"""
import json, sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from review_meta import review_headers

batch = sys.argv[1]
df = pd.read_csv(f'data/adjudication/{batch}_sheet.tsv', sep='\t').fillna('')
meta = review_headers(df.review_id)
out = {r: {'meta': meta.get(r, ''), 'sentences': [{'sent_id': x.sent_id, 'text': x.sent_text} for x in g.itertuples()]}
       for r, g in df.groupby('review_id', sort=False)}
path = f'data/adjudication/agent_runs/{batch}_input.json'
json.dump(out, open(path, 'w'), ensure_ascii=False, indent=1)
print(path, len(out), 'reviews')
