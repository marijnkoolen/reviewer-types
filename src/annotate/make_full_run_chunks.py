"""Split all annotated reviews into fixed chunks for the full agent run.

usage: python src/annotate/make_full_run_chunks.py [CHUNK_SIZE]
output: data/full_run/chunks/chunk_NNN.json  ({review_id: {"meta", "sentences": [{sent_id, text}]}})
        data/full_run/chunks/index.tsv       (review_id, chunk, n_sentences)

Reviews are shuffled with a fixed seed so every chunk is a random sample (the calibration
chunk is representative). The order never changes once written; rerunning refuses to overwrite.
"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_meta import review_headers

size = int(sys.argv[1]) if len(sys.argv) > 1 else 25
out = 'data/full_run/chunks'
if os.path.exists(f'{out}/index.tsv'):
    sys.exit(f'{out}/index.tsv exists; chunks are fixed. Delete the folder deliberately to rebuild.')
os.makedirs(out, exist_ok=True)
d = pd.read_csv('data/raw_annotations.tsv.gz', sep='\t', usecols=['review_id', 'sent_id', 'sent_num', 'sent_text'])
s = d.drop_duplicates('sent_id').sort_values(['review_id', 'sent_num'])
ids = pd.Series(s.review_id.unique()).sample(frac=1, random_state=20261114).tolist()
meta = review_headers(ids)
rows = []
for c in range(0, len(ids), size):
    chunk = ids[c:c + size]; n = c // size
    data = {}
    for r in chunk:
        sub = s[s.review_id == r]
        data[r] = {'meta': meta.get(r, 'Genre: unknown · Rating: unknown · Review year: unknown'),
                   'sentences': [{'sent_id': x.sent_id, 'text': x.sent_text} for x in sub.itertuples()]}
        rows.append((r, n, len(sub)))
    json.dump(data, open(f'{out}/chunk_{n:03d}.json', 'w'), ensure_ascii=False, indent=1)
idx = pd.DataFrame(rows, columns=['review_id', 'chunk', 'n_sentences'])
idx.to_csv(f'{out}/index.tsv', sep='\t', index=False)
print(len(ids), 'reviews,', idx.chunk.nunique(), 'chunks,', idx.n_sentences.sum(), 'sentences')
print(idx.groupby('chunk').n_sentences.sum().describe().round(0).to_dict())
