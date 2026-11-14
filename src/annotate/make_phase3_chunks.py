"""Phase 3: build annotation chunks for the drawn sample once the review texts are staged.

usage: python src/annotate/make_phase3_chunks.py [TEXTS] [CHUNK_SIZE=25]
TEXTS (default data/phase3/texts.tsv.gz): tab-separated, header review_id, sent_num, sent_text; one row per
sentence, split the same way as the RobBERT predictions (sent_num 1..n). Extra columns are ignored.
output: data/phase3/chunks/chunk_NNN.json (same format as data/full_run/chunks) and index.tsv
(review_id, chunk, n_sentences, stratum). Reviews are shuffled with a fixed seed so strata are mixed in
every chunk. Refuses to overwrite; reports sample reviews missing from TEXTS and sentence-count mismatches.
"""
import csv, json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_meta import review_headers

texts = sys.argv[1] if len(sys.argv) > 1 else 'data/phase3/texts.tsv.gz'
size = int(sys.argv[2]) if len(sys.argv) > 2 else 25
out = 'data/phase3/chunks'
if os.path.exists(f'{out}/index.tsv'):
    sys.exit(f'{out}/index.tsv exists; chunks are fixed. Delete the folder deliberately to rebuild.')

smp = pd.read_csv('data/phase3/sample.tsv', sep='\t').set_index('review_id')
t = pd.read_csv(texts, sep='\t', quoting=csv.QUOTE_NONE, dtype={'review_id': str}, keep_default_na=False)
t = t[t.review_id.isin(smp.index)].copy()
t['sent_num'] = t.sent_num.astype(int)
t = t.sort_values(['review_id', 'sent_num'])
missing = smp.index.difference(t.review_id.unique())
got = t.groupby('review_id').size()
mismatch = (got - smp.n_sentences.reindex(got.index)).loc[lambda s: s != 0]
print(f'{len(got)} of {len(smp)} sample reviews have text; missing {len(missing)}; '
      f'sentence-count mismatches vs RobBERT split: {len(mismatch)}')
if len(missing):
    print('  first missing:', list(missing[:5]))

ids = pd.Series(got.index).sample(frac=1, random_state=20261115).tolist()
meta = review_headers(ids)
os.makedirs(out, exist_ok=True)
rows = []
for c in range(0, len(ids), size):
    n = c // size
    data = {}
    for r in ids[c:c + size]:
        sub = t[t.review_id == r]
        data[r] = {'meta': meta.get(r, 'Genre: unknown · Rating: unknown · Review year: unknown'),
                   'sentences': [{'sent_id': f'{r}-sent-{x.sent_num:03d}', 'text': x.sent_text} for x in sub.itertuples()]}
        rows.append((r, n, len(sub), smp.stratum[r]))
    json.dump(data, open(f'{out}/chunk_{n:03d}.json', 'w'), ensure_ascii=False, indent=1)
idx = pd.DataFrame(rows, columns=['review_id', 'chunk', 'n_sentences', 'stratum'])
idx.to_csv(f'{out}/index.tsv', sep='\t', index=False)
print(len(ids), 'reviews,', idx.chunk.nunique(), 'chunks,', idx.n_sentences.sum(), 'sentences')
