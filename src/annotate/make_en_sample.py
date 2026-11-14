"""Phase 4 (English): draw the English review pool from the reader_profiles DuckDB (UA-006, UA-018).

usage: python3.11 src/annotate/make_en_sample.py [PER_PLATFORM=600] [PILOT_PER_PLATFORM=100]
Sources: Goodreads (gr_reviews), Amazon (az_reviews), LibraryThing (lt_anon_reviews), opened read-only.
Per platform: random rowids -> English (langid is_english, confidence >= 0.9) -> genre known (GR-10 categories:
gr_book_genre, az_book_genre_xwalk, lt_book_genre) -> 5..2000 words -> deduplicated text. Then 85% fiction
categories / 15% non-fiction (the Dutch corpus has 16% non-fiction; Amazon alone is 53%). The pool is shuffled with
a fixed seed and `order` numbers it per platform: the pilot is order < PILOT_PER_PLATFORM, later batches take the next.
Ratings: 1..5 on all platforms (LibraryThing 1..10 is halved); 0 = unrated becomes empty (header 'unknown').
output: data/other_languages/en/sample.tsv (review_id, platform, order, pilot, genre, rating, review_year, n_words)
        data/other_languages/en/texts.tsv.gz (review_id, text; backslash and newline escaped as \\\\ and \\n)
        data/other_languages/en/source_keys.tsv (review_id, platform, user_id, item_id) for traceability only.
Labellers never see platform or source keys. Refuses to overwrite.
"""
import csv, html, os, random, re, sys
import duckdb
import pandas as pd

DB = '/Volumes/T7_Data/Data/claude-projects-external/reader_profiles/db/reader_profiles.duckdb'
OUT = 'data/other_languages/en'
per = int(sys.argv[1]) if len(sys.argv) > 1 else 600
pilot = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NONFIC_SHARE = 0.15
DRAW = 15000  # random rowids per platform before filtering
SEED = 20261114
if os.path.exists(f'{OUT}/sample.tsv'):
    sys.exit(f'{OUT}/sample.tsv exists; the pool is fixed. Delete it deliberately to redraw.')

# per platform: query on the drawn rowids (table `ids`), the review table, and the "no rating" value
PLATFORMS = {
    'goodreads': dict(
        sql="""select r.user_id, r.book_id item_id, r.review_text as text, r.rating::DOUBLE rating,
                      right(r.date_added, 4) review_year, g.dominant_genre genre
               from gr_reviews r
               join gr_review_language l on l.user_id = r.user_id and l.item_id = r.book_id
               join gr_book_genre g on g.book_id = r.book_id
               where r.rowid in (select rid from ids) and l.is_english and l.confidence >= 0.9""",
        table='gr_reviews', no_rating=0),
    'amazon': dict(
        sql="""select r.user_id, r.parent_asin item_id, r.text as text, r.rating,
                      strftime(to_timestamp(r.timestamp / 1000), '%Y') review_year, g.gr_genre genre
               from az_reviews r
               join az_review_language l on l.user_id = r.user_id and l.item_id = r.asin
               join az_book_genre_xwalk g on g.parent_asin = r.parent_asin
               where r.rowid in (select rid from ids) and l.is_english and l.confidence >= 0.9
                 and g.gr_genre is not null""",
        table='az_reviews', no_rating=None),
    'librarything': dict(
        sql="""select r.user_id, r.work_id item_id, r.review_text as text, r.rating,
                      strftime(r.review_date, '%Y') review_year, g.dominant_genre genre
               from lt_anon_reviews r
               join lt_review_language l on l.user_id = r.user_id and l.item_id = r.work_id
               join lt_book_genre g on g.thing_id = r.work_id
               where r.rowid in (select rid from ids) and l.is_english and l.confidence >= 0.9""",
        table='lt_anon_reviews', no_rating=0, scale=0.5),  # LT stores half stars as 1..10
}


def clean(t):
    t = re.sub(r'<br\s*/?>', '\n', t or '', flags=re.I)
    t = html.unescape(re.sub(r'</?(?:i|b|em|strong|p|a[^>]*)>', '', t, flags=re.I))
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r' *\n *', '\n', t)
    return re.sub(r'\n{3,}', '\n\n', t).strip()


con = duckdb.connect(DB, read_only=True)
con.execute('set threads = 4')
rng = random.Random(SEED)
frames = []
for plat, p in PLATFORMS.items():
    n = con.execute(f'select count(*) from {p["table"]}').fetchone()[0]
    con.execute('create or replace temp table ids as select unnest(?::BIGINT[]) rid', [rng.sample(range(n), DRAW)])
    d = con.execute(p['sql']).df()
    n_lang = len(d)
    d['text'] = d.text.map(clean)
    d['n_words'] = d.text.str.split().str.len()
    d = d[(d.n_words >= 5) & (d.n_words <= 2000)].drop_duplicates('text')
    if p['no_rating'] is not None:
        d.loc[d.rating == p['no_rating'], 'rating'] = float('nan')
    d['rating'] = d.rating * p.get('scale', 1)
    k_nf = round(per * NONFIC_SHARE)
    nonfic = d.genre.eq('non-fiction')
    nf, fic = d[nonfic], d[~nonfic]
    if len(nf) < k_nf or len(fic) < per - k_nf:
        sys.exit(f'{plat}: too few candidates (fiction {len(fic)}, non-fiction {len(nf)}); raise DRAW')
    s = pd.concat([fic.sample(per - k_nf, random_state=SEED), nf.sample(k_nf, random_state=SEED)])
    s = s.sample(frac=1, random_state=SEED).reset_index(drop=True)
    s['platform'], s['order'] = plat, range(len(s))
    print(f'{plat}: {n:,} reviews, {DRAW} drawn, {n_lang} English with known genre, '
          f'{len(d)} after length/dedup filter, kept {len(s)}')
    frames.append(s)

a = pd.concat(frames, ignore_index=True)
a['pilot'] = a.order < pilot
a = a.sort_values(['order', 'platform']).reset_index(drop=True)
a['review_id'] = [f'en-review-{i + 1:05d}' for i in range(len(a))]
os.makedirs(OUT, exist_ok=True)
a[['review_id', 'platform', 'order', 'pilot', 'genre', 'rating', 'review_year', 'n_words']].to_csv(
    f'{OUT}/sample.tsv', sep='\t', index=False)
a[['review_id', 'platform', 'user_id', 'item_id']].to_csv(f'{OUT}/source_keys.tsv', sep='\t', index=False)
t = a[['review_id', 'text']].copy()
t['text'] = (t.text.str.replace('\\', '\\\\', regex=False).str.replace('\n', '\\n', regex=False)
             .str.replace('\t', ' ', regex=False))
t.to_csv(f'{OUT}/texts.tsv.gz', sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar=None)
print(len(a), 'reviews;', int(a.pilot.sum()), 'in pilot')
print(a.groupby('platform').n_words.describe().round(0).to_string())
print(a.groupby(['platform', 'genre']).size().unstack(0).fillna(0).astype(int).to_string())
