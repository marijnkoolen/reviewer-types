"""Phase 4 (other languages): draw a review pool for one language from the parsed Goodreads language pages.

usage: python3.11 src/annotate/make_gr_lang_sample.py LANG [POOL=1800] [PILOT=300]      e.g. it
input:  data/other_languages/<lang>/gr_reviews.tsv.gz (from parse_gr_lang_pages.py)
Same filters as make_en_sample.py, for one platform: review text in LANG (langid, normalised probability >= 0.9;
the pages list reviews by Goodreads' language filter, which is ~98% right) -> genre known (GR-10 categories from
gr_book_genre in the reader_profiles DuckDB, by Goodreads book id) -> 5..2000 words -> deduplicated text -> at most
MAX_PER_BOOK reviews per book (a page lists up to 30 reviews of one book; the English pool drew random reviews from
15M and has almost no repeats). Then 85% fiction categories / 15% non-fiction, shuffled with a fixed seed;
`order` numbers the pool, the pilot is order < PILOT.
output: data/other_languages/<lang>/sample.tsv, texts.tsv.gz, source_keys.tsv (review_id, book_id, review_url) in
        the formats of the English pool; platform is 'goodreads'. Labellers never see source keys. Refuses to
        overwrite.
"""
import csv, os, random, re, sys
import duckdb
import pandas as pd
from langid.langid import LanguageIdentifier, model

DB = '/Volumes/T7_Data/Data/claude-projects-external/reader_profiles/db/reader_profiles.duckdb'
lang = sys.argv[1]
pool = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
pilot = int(sys.argv[3]) if len(sys.argv) > 3 else 300
NONFIC_SHARE, MAX_PER_BOOK, SEED = 0.15, 2, 20261114
OUT = f'data/other_languages/{lang}'
if os.path.exists(f'{OUT}/sample.tsv'):
    sys.exit(f'{OUT}/sample.tsv exists; the pool is fixed. Delete it deliberately to redraw.')

d = pd.read_csv(f'{OUT}/gr_reviews.tsv.gz', sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False,
                dtype={'rating': str, 'review_date': str})
n_all = len(d)
d['raw'] = d.text
d['text'] = d.text.map(lambda t: re.sub(r'\\(.)', lambda m: '\n' if m.group(1) == 'n' else m.group(1), t))
d['n_words'] = d.text.str.split().str.len()
d = d[(d.n_words >= 5) & (d.n_words <= 2000)].drop_duplicates('text')
n_len = len(d)

con = duckdb.connect(DB, read_only=True)
con.execute('create or replace temp table ids as select unnest(?::VARCHAR[]) book_id',
            [sorted({str(b) for b in d.book_id})])
g = con.execute('select book_id, dominant_genre genre from ids join gr_book_genre using (book_id)').df()
d = d.merge(g.assign(book_id=g.book_id.astype(int)), on='book_id')
n_genre = len(d)

ident = LanguageIdentifier.from_modelstring(model, norm_probs=True)
lid = d.text.map(lambda t: ident.classify(t.replace('\n', ' ')))
d = d[[l == lang and p >= 0.9 for l, p in lid]]
n_lang = len(d)

d = d.sample(frac=1, random_state=SEED).groupby('book_id').head(MAX_PER_BOOK)
k_nf = round(pool * NONFIC_SHARE)
nonfic = d.genre.eq('non-fiction')
nf, fic = d[nonfic], d[~nonfic]
if len(nf) < k_nf or len(fic) < pool - k_nf:
    sys.exit(f'{lang}: too few candidates (fiction {len(fic)}, non-fiction {len(nf)}) for a pool of {pool}')
a = pd.concat([fic.sample(pool - k_nf, random_state=SEED), nf.sample(k_nf, random_state=SEED)])
a = a.sample(frac=1, random_state=SEED).reset_index(drop=True)
a['platform'], a['order'] = 'goodreads', range(len(a))
a['pilot'] = a.order < pilot
a['rating'] = pd.to_numeric(a.rating, errors='coerce')
a['review_year'] = a.review_date.str[:4].replace('', pd.NA)
a['review_id'] = [f'{lang}-review-{i + 1:05d}' for i in range(len(a))]
print(f'{lang}: {n_all} reviews, {n_len} after length/dedup, {n_genre} with known genre, {n_lang} in {lang} '
      f'(p >= 0.9), {len(d)} after max {MAX_PER_BOOK} per book; kept {len(a)} from {a.book_id.nunique()} books')

a[['review_id', 'platform', 'order', 'pilot', 'genre', 'rating', 'review_year', 'n_words']].to_csv(
    f'{OUT}/sample.tsv', sep='\t', index=False)
a[['review_id', 'book_id', 'review_url']].to_csv(f'{OUT}/source_keys.tsv', sep='\t', index=False)
a[['review_id', 'raw']].rename(columns={'raw': 'text'}).to_csv(
    f'{OUT}/texts.tsv.gz', sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar=None)
print(len(a), 'reviews;', int(a.pilot.sum()), 'in pilot; unrated', int(a.rating.isna().sum()))
print(a.n_words.describe().round(0).to_dict())
print(a.groupby('genre').size().to_string())
