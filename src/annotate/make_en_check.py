"""Phase 4 (English): pick the reviews for your check of the English pilot labels (plan step 13).

usage: python src/annotate/make_en_check.py [PER_PLATFORM=5; batch en_batch1 used 7] [MAX_SENT=20]
Random pilot reviews (fixed seed), PER_PLATFORM per platform, at most MAX_SENT sentences each, so ~120 sentences.
Reviews are ordered so the blind ones (en_batch1: last 6 on the page, --blind 6) mix platforms.
output: data/adjudication/en_batch1_sheet.tsv (review_id, sent_id, sent_text) and
        data/adjudication/agent_runs/en_batch1_{opus_a,opus_b,final}.json for make_agent_adjudication_html.py.
'final' is the a/b/c majority (opus_c only labelled the disputed sentences); a majority over a, b and final always
equals final, so the page prefill is the pilot label. Refuses to overwrite.
"""
import csv, json, os, sys
import pandas as pd

D, A = 'data/other_languages/en', 'data/adjudication'
per = int(sys.argv[1]) if len(sys.argv) > 1 else 5
max_sent = int(sys.argv[2]) if len(sys.argv) > 2 else 20
sheet = f'{A}/en_batch1_sheet.tsv'
if os.path.exists(sheet):
    sys.exit(f'{sheet} exists; the batch is fixed.')

m = pd.read_csv(f'{D}/majority_labels.tsv', sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False)
idx = pd.read_csv(f'{D}/chunks/index.tsv', sep='\t')
pick = idx[idx.n_sentences <= max_sent].groupby('platform').sample(per, random_state=20261114)
pick['k'] = pick.groupby('platform').cumcount()
ids = pick.sort_values(['k', 'platform']).review_id.tolist()
s = m[m.review_id.isin(ids)].copy()
s['o'] = s.review_id.map({r: i for i, r in enumerate(ids)})
s = s.sort_values(['o', 'sent_id'])
s[['review_id', 'sent_id', 'sent_text']].to_csv(sheet, sep='\t', index=False, quoting=csv.QUOTE_NONE)
for run in ['opus_a', 'opus_b', 'final']:
    col = 'labels' if run == 'final' else run
    json.dump([{'sent_id': r.sent_id, 'labels': [c for c in getattr(r, col).split(',') if c], 'why': ''}
               for r in s.itertuples()], open(f'{A}/agent_runs/en_batch1_{run}.json', 'w'), indent=0)
print(len(ids), 'reviews,', len(s), 'sentences;', (s.status == 'majority').sum(), 'disputed;',
      pick.platform.value_counts().to_dict())
