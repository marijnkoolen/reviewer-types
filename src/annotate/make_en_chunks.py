"""Phase 4: sentence-split a language pool (English, Italian, ...) and build annotation chunks for the pilot.

usage: [LANG_CODE=en] [BASE=data/other_languages] python3.11 src/annotate/make_en_chunks.py [CHUNK_SIZE=25] [SUBSET=pilot]
LANG_CODE picks data/other_languages/<lang> and the spaCy model (MODELS below); default en.
Splitting: line breaks first (paragraphs, list items), then the spaCy parser per line. A fragment
without letters or digits ('...', ':)', '***') is merged into the sentence before it. The Dutch reviews were split
with Trankit; matching that splitter is not required for other languages.
input:  data/other_languages/<lang>/sample.tsv, texts.tsv.gz (from make_en_sample.py / make_gr_lang_sample.py)
output: data/other_languages/<lang>/sentences.tsv.gz (review_id, sent_num, sent_text) for the whole pool (split once,
        then fixed); chunks/chunk_NNN.json for SUBSET (same format as data/phase3/chunks) and chunks/index.tsv
        (review_id, chunk, n_sentences, platform). Reviews are shuffled with a fixed seed so platforms mix.
Header: review_meta.format_header with the GR-10 genre, rating (unknown when unrated) and review year.
Refuses to overwrite chunks.
"""
import csv, json, os, re, sys
import pandas as pd
import spacy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_meta import format_header

MODELS = {'en': 'en_core_web_sm', 'it': 'it_core_news_lg', 'ru': 'ru_core_news_sm', 'nl': 'nl_core_news_lg'}
LANG = os.environ.get('LANG_CODE', 'en')
BASE = os.environ.get('BASE', 'data/other_languages')   # data/matched for the matched-books sample
D = f'{BASE}/{LANG}'
size = int(sys.argv[1]) if len(sys.argv) > 1 else 25
subset = sys.argv[2] if len(sys.argv) > 2 else 'pilot'
out = f'{D}/chunks'
if os.path.exists(f'{out}/index.tsv'):
    sys.exit(f'{out}/index.tsv exists; chunks are fixed. Delete the folder deliberately to rebuild.')

smp = pd.read_csv(f'{D}/sample.tsv', sep='\t', dtype={'review_year': str}).set_index('review_id')
tx = pd.read_csv(f'{D}/texts.tsv.gz', sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False)
tx['text'] = tx.text.map(lambda t: re.sub(r'\\(.)', lambda m: '\n' if m.group(1) == 'n' else m.group(1), t))

sents_path = f'{D}/sentences.tsv.gz'
if os.path.exists(sents_path):
    ss = pd.read_csv(sents_path, sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False)
else:
    nlp = spacy.load(MODELS[LANG], disable=['ner', 'lemmatizer'])
    rows = []
    for r in tx.itertuples():
        lines = [l.strip() for l in r.text.split('\n') if l.strip()]
        sents = []
        for doc in nlp.pipe(lines):
            for s in doc.sents:
                t = s.text.strip()
                if not t:
                    continue
                if sents and not re.search(r'\w', t):
                    sents[-1] += ' ' + t
                else:
                    sents.append(t)
        rows += [(r.review_id, i + 1, s) for i, s in enumerate(sents)]
    ss = pd.DataFrame(rows, columns=['review_id', 'sent_num', 'sent_text'])
    ss.to_csv(sents_path, sep='\t', index=False, quoting=csv.QUOTE_NONE)
    n = ss.groupby('review_id').size()
    print(f'split {len(n)} reviews into {len(ss)} sentences; per review median {n.median():g}, '
          f'p90 {n.quantile(.9):g}, max {n.max()}')

ids = smp.index[smp.pilot] if subset == 'pilot' else smp.index
ids = pd.Series(ids).sample(frac=1, random_state=20261116).tolist()
os.makedirs(out, exist_ok=True)
rows = []
for c in range(0, len(ids), size):
    n = c // size
    data = {}
    for r in ids[c:c + size]:
        m = smp.loc[r]
        sub = ss[ss.review_id == r].sort_values('sent_num')
        data[r] = {'meta': format_header(m.genre, m.rating, m.review_year),
                   'sentences': [{'sent_id': f'{r}-sent-{x.sent_num:03d}', 'text': x.sent_text}
                                 for x in sub.itertuples()]}
        rows.append((r, n, len(sub), m.platform))
    json.dump(data, open(f'{out}/chunk_{n:03d}.json', 'w'), ensure_ascii=False, indent=1)
idx = pd.DataFrame(rows, columns=['review_id', 'chunk', 'n_sentences', 'platform'])
idx.to_csv(f'{out}/index.tsv', sep='\t', index=False)
print(len(ids), 'reviews,', idx.chunk.nunique(), 'chunks,', idx.n_sentences.sum(), 'sentences')
print(idx.groupby('chunk').n_sentences.sum().describe().round(0).to_dict())
print(idx.groupby('platform').n_sentences.describe().round(1).to_string())
