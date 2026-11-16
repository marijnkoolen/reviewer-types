"""Matched-books sample: the same books reviewed on Goodreads in Dutch, English, Italian and Russian, plus the
Dutch corpus (UA-031 a, UA-032 1a/2a).

usage: python3.11 src/annotate/make_matched_sample.py [N_BOOKS=152]
inputs: data/other_languages/book_links.tsv (link_dutch_goodreads.py); data/other_languages/{nl,it,ru}/gr_reviews.tsv.gz
        (parse_gr_lang_pages.py, with likes and page position); the scraped English Goodreads pages of the candidate
        books (T7 drive, parsed here); the Dutch corpus (data/review_sentences.tsv.gz, metadata/review_metadata.tsv.gz);
        the reader_profiles DuckDB (GR-10 genre; earliest publication year over the work's editions).
Arms: nl, en, it, ru = Goodreads language pages (one platform, one ranking, same years); nlc = Dutch corpus reviews of
the same work (other platforms, not ranked).
Likes ranking (UA-032 2a): Goodreads sorts a language page by likes; a page with 30 reviews shows the 30 most liked.
Per book and Goodreads arm, one review from the top band (the first min(5, ceil(n/3)) positions) and one from the
bottom band (positions >= floor(2n/3), n = reviews on the page), so every arm has the same mix of high- and
low-ranked reviews; position, likes and band are kept. nlc: 2 random corpus reviews, reviews with v2.4 labels first.
Usable review: in the arm's language by langid p >= 0.9, 5..2000 words, deduplicated text.
Books: linked works with a usable review in both bands of every Goodreads arm and >= 2 usable corpus reviews; one
Goodreads work per Dutch work. Strata: 4 genre groups x 4 eras of first publication; N_BOOKS split equally over the
genre groups and as evenly over the eras as the books allow.
Header: Goodreads arms GR-10 genre; nlc the NUR genre (as in the ground truth).
output: data/matched/books.tsv; data/matched/sample.tsv (review_id, arm, gr_work_id, source_review_id, band, position,
        n_on_page, likes, labelled, genre, rating, review_year, n_words); per arm data/matched/<arm>/texts.tsv.gz
        (review_id, text) and sample.tsv (the make_en_chunks.py input), and for nlc sentences.tsv.gz (the corpus
        split, so labels stay comparable with the ground truth). Ids mt-<arm>-NNNNN. Refuses to overwrite.
"""
import csv
import math
import os
import re
import sys

import duckdb
import pandas as pd
from langid.langid import LanguageIdentifier, model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_gr_lang_pages import esc, parse_page  # noqa: E402

N_BOOKS = int(sys.argv[1]) if len(sys.argv) > 1 else 152
SEED = 20261116
DB = '/Volumes/T7_Data/Data/claude-projects-external/reader_profiles/db/reader_profiles.duckdb'
EN_PAGES = '/Volumes/T7_Data/Code/impact-of-fiction/impfic-multilingual-reviews/data/Book_language_pages/en'
OUT = 'data/matched'
GR_ARMS = ['nl', 'en', 'it', 'ru']
GROUP = {'fiction': 'general/literary fiction', 'mystery, thriller, crime': 'crime/thriller', 'non-fiction': 'non-fiction'}
ERAS = [(0, 1950, '<1950'), (1950, 2000, '1950-1999'), (2000, 2010, '2000-2009'), (2010, 9999, '>=2010')]
if os.path.exists(f'{OUT}/sample.tsv'):
    sys.exit(f'{OUT}/sample.tsv exists; the matched sample is fixed. Delete it deliberately to redraw.')
ident = LanguageIdentifier.from_modelstring(model, norm_probs=True)


def unesc(t):
    return re.sub(r'\\(.)', lambda m: '\n' if m.group(1) == 'n' else m.group(1), t)


def usable(d, lang):
    d = d.assign(n_words=d.text.str.split().str.len())
    d = d[(d.n_words >= 5) & (d.n_words <= 2000)].drop_duplicates('text')
    lid = d.text.map(lambda t: ident.classify(t.replace('\n', ' ')))
    return d[[l == lang and p >= 0.9 for l, p in lid]]


def band(pos, n):
    if pos < min(5, math.ceil(n / 3)):
        return 'top'
    return 'bottom' if pos >= (2 * n) // 3 else 'middle'


L = pd.read_csv('data/other_languages/book_links.tsv', sep='\t', dtype={'gr_book_id': str, 'gr_work_id': str})
L = L[(L.n_nl >= 2) & (L.n_it >= 2) & (L.n_ru >= 2) & L.gr_book_id.notna()]
L = (L.sort_values(['n_it', 'n_ru', 'n_nl'], ascending=False).drop_duplicates('gr_work_id')
     .drop_duplicates('impfic_work_id'))
pages = {re.match(r'(\d+)', f).group(1): f for f in os.listdir(EN_PAGES) if re.match(r'\d+', f)}
L = L[L.gr_book_id.isin(pages)]

con = duckdb.connect(DB, read_only=True)
con.register('w', L[['gr_work_id', 'gr_book_id']])
meta = con.execute("""
    select w.gr_work_id, any_value(g.dominant_genre) genre,
           min(case when try_cast(b.publication_year as int) between 1000 and 2030
                    then try_cast(b.publication_year as int) end) first_year
    from w join gr_books b on b.work_id = w.gr_work_id
    left join gr_book_genre g on g.book_id = w.gr_book_id
    group by 1""").df()
L = L.merge(meta, on='gr_work_id').dropna(subset=['genre', 'first_year'])
L['genre_group'] = L.genre.map(GROUP).fillna('other genre fiction')
L['era'] = [next(e for lo, hi, e in ERAS if lo <= y < hi) for y in L.first_year]
books = set(L.gr_book_id)
print(f'{len(L)} candidate works (linked, >= 2 reviews in nl/it/ru on Goodreads, English page, genre and year known)')

gr = {}
for arm in ['nl', 'it', 'ru']:
    d = pd.read_csv(f'data/other_languages/{arm}/gr_reviews.tsv.gz', sep='\t', quoting=csv.QUOTE_NONE,
                    keep_default_na=False, dtype={'book_id': str, 'rating': str})
    d = d[d.book_id.isin(books)].copy()
    d['text'] = d.text.map(unesc)
    gr[arm] = d
rows = []
for b in L.gr_book_id:
    revs = list(parse_page(f'{EN_PAGES}/{pages[b]}'))
    rows += [dict(book_id=b, page_file=pages[b], position=k, n_on_page=len(revs), **r) for k, r in enumerate(revs)]
gr['en'] = pd.DataFrame(rows).dropna(subset=['text'])
for arm in GR_ARMS:
    d = usable(gr[arm], arm)
    d['band'] = [band(int(p), int(n)) for p, n in zip(d.position, d.n_on_page)]
    gr[arm] = d[d.band != 'middle']

nl_rev = pd.read_csv('metadata/review_metadata.tsv.gz', sep='\t', dtype={'review_date': str},
                     usecols=['review_id', 'work_id', 'rating', 'review_date', 'nur_genre']).drop_duplicates('review_id')
nl_rev = nl_rev[nl_rev.work_id.isin(set(L.impfic_work_id))]
gt = pd.read_csv('data/gt_v2/labels.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False,
                 usecols=['review_id'])
nl_rev['labelled'] = nl_rev.review_id.isin(set(gt.review_id))
sents = pd.read_csv('data/review_sentences.tsv.gz', sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False,
                    usecols=['review_id', 'sent_num', 'sent_text'])
sents = sents[sents.review_id.isin(set(nl_rev.review_id))]
nl_txt = sents.groupby('review_id', sort=False).sent_text.apply(' '.join).rename('text').reset_index()
nlc = usable(nl_rev.merge(nl_txt, on='review_id'), 'nl')

ok = L.impfic_work_id.map(nlc.work_id.value_counts()).fillna(0) >= 2
for arm in GR_ARMS:
    have = gr[arm].groupby('book_id').band.agg(lambda s: {'top', 'bottom'} <= set(s))
    ok &= L.gr_book_id.map(have).fillna(False).astype(bool)
L = L[ok]
print(f'{len(L)} works with a usable top and bottom review in every Goodreads arm and >= 2 usable corpus reviews')
print(pd.crosstab(L.genre_group, L.era, margins=True).to_string())

chosen, per_group = [], N_BOOKS // L.genre_group.nunique()
for g, grp in L.groupby('genre_group'):
    pools = {e: list(grp[grp.era == e].sample(frac=1, random_state=SEED).index) for _, _, e in ERAS}
    take = []
    while len(take) < per_group and any(pools.values()):
        for e in [e for _, _, e in ERAS]:
            if pools[e] and len(take) < per_group:
                take.append(pools[e].pop())
    chosen += take
B = L.loc[chosen].reset_index(drop=True)
print(f'chosen {len(B)} books'); print(pd.crosstab(B.genre_group, B.era, margins=True).to_string())

out = []
for b in B.itertuples():
    for arm in GR_ARMS:
        d = gr[arm][gr[arm].book_id == b.gr_book_id]
        for bd in ['top', 'bottom']:
            r = d[d.band == bd].sample(1, random_state=SEED).iloc[0]
            out.append({'arm': arm, 'gr_work_id': b.gr_work_id, 'source_review_id': r.review_url, 'band': bd,
                        'position': int(r.position), 'n_on_page': int(r.n_on_page), 'likes': int(r.likes),
                        'labelled': False, 'genre': b.genre, 'rating': r.rating,
                        'review_year': str(r.review_date)[:4], 'n_words': r.n_words, 'text': r.text})
    d = (nlc[nlc.work_id == b.impfic_work_id].sample(frac=1, random_state=SEED)
         .sort_values('labelled', ascending=False, kind='stable').head(2))
    for r in d.itertuples():
        out.append({'arm': 'nlc', 'gr_work_id': b.gr_work_id, 'source_review_id': r.review_id, 'band': '',
                    'position': None, 'n_on_page': None, 'likes': None, 'labelled': bool(r.labelled),
                    'genre': r.nur_genre, 'rating': r.rating, 'review_year': str(r.review_date)[:4],
                    'n_words': r.n_words, 'text': r.text})
S = pd.DataFrame(out)
S['review_id'] = [f'mt-{a}-{i + 1:05d}' for a, i in zip(S.arm, S.groupby('arm').cumcount())]
os.makedirs(OUT, exist_ok=True)
B[['gr_work_id', 'impfic_work_id', 'gr_book_id', 'genre', 'genre_group', 'first_year', 'era']].to_csv(
    f'{OUT}/books.tsv', sep='\t', index=False)
cols = ['review_id', 'arm', 'gr_work_id', 'source_review_id', 'band', 'position', 'n_on_page', 'likes', 'labelled',
        'genre', 'rating', 'review_year', 'n_words']
S[cols].to_csv(f'{OUT}/sample.tsv', sep='\t', index=False)
for arm, g in S.groupby('arm'):
    os.makedirs(f'{OUT}/{arm}', exist_ok=True)
    g.assign(platform='dutch_corpus' if arm == 'nlc' else 'goodreads', pilot=True, order=range(len(g)))[
        ['review_id', 'platform', 'order', 'pilot', 'genre', 'rating', 'review_year', 'n_words']].to_csv(
        f'{OUT}/{arm}/sample.tsv', sep='\t', index=False)
    t = g[['review_id', 'text']].copy()
    t['text'] = t.text.map(esc)
    t.to_csv(f'{OUT}/{arm}/texts.tsv.gz', sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar=None)
src = dict(zip(S[S.arm == 'nlc'].source_review_id, S[S.arm == 'nlc'].review_id))
ss = sents[sents.review_id.isin(src)].assign(review_id=lambda x: x.review_id.map(src))
ss.sort_values(['review_id', 'sent_num'])[['review_id', 'sent_num', 'sent_text']].to_csv(
    f'{OUT}/nlc/sentences.tsv.gz', sep='\t', index=False, quoting=csv.QUOTE_NONE)
print(S.groupby(['arm', 'band'], dropna=False).agg(reviews=('review_id', 'size'), median_words=('n_words', 'median'),
                                                   median_likes=('likes', 'median')).to_string())
