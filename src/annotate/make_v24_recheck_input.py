"""Build the v2.4 re-check input: full-run sentences that codebook v2.4 rules R1-R7 (UA-016) may change.

usage: python src/annotate/make_v24_recheck_input.py [REVIEWS_PER_FILE=51]
output: data/full_run/recheck_v24/input_NN.json  {review_id: {meta, sentences, check: [{sent_id, labels, rules}]}}
        data/full_run/recheck_v24/candidates.tsv
Candidates are found by label combinations and simple Dutch text patterns (see RULES below); the re-check
agents then apply only the v2.4 corrections. Refuses to overwrite.
"""
import csv, glob, json, os, sys
import pandas as pd

OUT = 'data/full_run/recheck_v24'
if os.path.exists(f'{OUT}/candidates.tsv'):
    sys.exit(f'{OUT} exists; delete it deliberately to rebuild.')
size = int(sys.argv[1]) if len(sys.argv) > 1 else 51
m = pd.read_csv('data/full_run/majority_labels.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False)
L = m.labels.str.split(',').map(lambda x: set(filter(None, x)))
T = m.sent_text.str.lower()
has = lambda c: L.map(lambda s: c in s)
RULES = {
    'R1': has('OTW') & (has('NAR') | has('THM') | has('MET') | has('RCP')),
    'R2': has('RCP') & (has('COO') | T.str.contains('achterflap|flaptekst|achterkant|achterzijde|omslag')),
    'R3': has('IDI') & T.str.contains(r'zou ik|ik zou|\bzelf\b'),
    'R4': has('IDI') & (has('FEE') | T.str.contains('erger|irrit|frustr|toespreken|door elkaar schud|kruipen|mens,')),
    'R5': has('EVA') & (has('COO') | T.str.contains('nieuwsgierig|aan het denken')),
    'R6': ~has('STY') & T.str.contains(r'neer\s*gezet|neer\s*zet|neer te zetten|neerzet|lee?st .*\bweg\b|las .*\bweg\b|leest (?:vlot|lekker|fijn|makkelijk)'),
    'R7': has('REC') & T.str.contains(r'\bals (?:je|u)\b.*\bdan\b'),
}
R = pd.DataFrame(RULES)
m['rules'] = R.apply(lambda row: ' '.join(k for k, v in row.items() if v), axis=1)
c = m[m.rules != '']
os.makedirs(OUT, exist_ok=True)
c[['review_id', 'sent_id', 'labels', 'rules']].to_csv(f'{OUT}/candidates.tsv', sep='\t', index=False)
chunks = {}
for f in glob.glob('data/full_run/chunks/chunk_*.json'):
    chunks.update(json.load(open(f)))
reviews = list(dict.fromkeys(c.review_id))
for i in range(0, len(reviews), size):
    data = {}
    for rid in reviews[i:i + size]:
        sub = c[c.review_id == rid]
        data[rid] = {**chunks[rid], 'check': [{'sent_id': r.sent_id, 'labels': [x for x in r.labels.split(',') if x], 'rules': r.rules.split()}
                                               for r in sub.itertuples()]}
    json.dump(data, open(f'{OUT}/input_{i // size:02d}.json', 'w'), ensure_ascii=False, indent=1)
print(len(c), 'sentences in', len(reviews), 'reviews,', -(-len(reviews) // size), 'files;', {k: int(v.sum()) for k, v in RULES.items()})
