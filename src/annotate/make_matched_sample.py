"""Matched-books sample: the same books reviewed in Dutch, English, Italian and Russian (UA-031 a).

usage: python3.11 src/annotate/make_matched_sample.py [N_BOOKS=152] [PER_LANG=2]
inputs: data/other_languages/book_links.tsv (link_dutch_goodreads.py), the Dutch corpus (data/review_sentences.tsv.gz,
        metadata/review_metadata.tsv.gz), data/other_languages/{it,ru}/gr_reviews.tsv.gz, the scraped English Goodreads
        pages of the linked books (T7 drive, parsed here with parse_gr_lang_pages.parse_page), and the reader_profiles
        DuckDB (GR-10 genre and the earliest publication year over the work's editions).
Books: linked works with >= PER_LANG usable reviews in every language (in the language by langid p >= 0.9, 5..2000
words, deduplicated text; Dutch: the corpus reviews of the work). Strata: four genre groups (general/literary
fiction; crime/thriller; other genre fiction: fantasy, romance, young adult, children, comics, historical, poetry;
non-fiction) x era of first publication (< 1950, 1950-1999, 2000-2009, >= 2010). N_BOOKS split equally over the genre
groups, and within a group as evenly over the eras as the books allow. Reviews: PER_LANG per language per book,
random; for Dutch, reviews that already have v2.4 labels (data/gt_v2/labels.tsv) are taken first.
Header: each language keeps its usual review_meta header (Dutch NUR genre, others GR-10), as the model will see it.
output: data/matched/books.tsv (gr_work_id, impfic_work_id, gr_book_id, genre, genre_group, first_year, era)
        data/matched/sample.tsv (review_id, lang, gr_work_id, source_review_id, labelled, genre, rating,
        review_year, n_words) and data/matched/<lang>/texts.tsv.gz (review_id, text) for en/it/ru;
        Dutch texts stay in the corpus (source_review_id). New ids: mt-<lang>-NNNNN. Refuses to overwrite.
"""
import csv
import os
import re
import sys

import duckdb
import numpy as np
import pandas as pd
from langid.langid import LanguageIdentifier, model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_gr_lang_pages import esc, parse_page  # noqa: E402

N_BOOKS = int(sys.argv[1]) if len(sys.argv) > 1 else 152
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 2
SEED = 20261116
DB = '/Volumes/T7_Data/Data/claude-projects-external/reader_profiles/db/reader_profiles.duckdb'
EN_PAGES = '/Volumes/T7_Data/Code/impact-of-fiction/impfic-multilingual-reviews/data/Book_language_pages/en'
OUT = 'data/matched'
GROUP = {'fiction': 'general/literary fiction', 'mystery, thriller, crime': 'crime/thriller', 'non-fiction': 'non-fiction'}
ERAS = [(0, 1950, '<1950'), (1950, 2000, '1950-1999'), (2000, 2010, '2000-2009'), (2010, 9999, '>=2010')]
if os.path.exists(f'{OUT}/sample.tsv'):
    sys.exit(f'{OUT}/sample.tsv exists; the matched sample is fixed. Delete it deliberately to redraw.')
ident = LanguageIdentifier.from_modelstring(model, norm_probs=True)
rng = np.random.default_rng(SEED)


def unesc(t):
    return re.sub(r'\\(.)', lambda m: '\n' if m.group(1) == 'n' else m.group(1), t)


def usable(d, lang):
    """Rows of d (text column) in `lang`, 5..2000 words, deduplicated."""
    d = d.assign(n_words=d.text.str.split().str.len())
    d = d[(d.n_words >= 5) & (d.n_words <= 2000)].drop_duplicates('text')
    lid = d.text.map(lambda t: ident.classify(t.replace('\n', ' ')))
    return d[[l == lang and p >= 0.9 for l, p in lid]]


L = pd.read_csv('data/other_languages/book_links.tsv', sep='\t', dtype={'gr_book_id': str, 'gr_work_id': str})
L = L[(L.n_nl >= PER) & (L.n_it >= PER) & (L.n_ru >= PER) & L.gr_book_id.notna()]
L = (L.sort_values(['n_it', 'n_ru', 'n_nl'], ascending=False).drop_duplicates('gr_work_id')
     .drop_duplicates('impfic_work_id'))   # one Goodreads work per Dutch work and vice versa
pages = {re.match(r'(\d+)', f).group(1): f for f in os.listdir(EN_PAGES) if re.match(r'\d+', f)}
L = L[L.gr_book_id.isin(pages)]
print(f'{len(L)} candidate works with >= {PER} reviews in nl/it/ru and an English page')

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

# reviews per language for the candidates
it_ru = {}
for lang in ['it', 'ru']:
    d = pd.read_csv(f'data/other_languages/{lang}/gr_reviews.tsv.gz', sep='\t', quoting=csv.QUOTE_NONE,
                    keep_default_na=False, dtype={'book_id': str, 'rating': str})
    d = d[d.book_id.isin(set(L.gr_book_id))]
    d['text'] = d.text.map(unesc)
    it_ru[lang] = usable(d, lang)
en_rows = []
for b in L.gr_book_id:
    en_rows += [dict(book_id=b, **r) for r in parse_page(f'{EN_PAGES}/{pages[b]}')]
en = usable(pd.DataFrame(en_rows).dropna(subset=['text']), 'en')
nl_rev = pd.read_csv('metadata/review_metadata.tsv.gz', sep='\t', dtype={'review_date': str},
                     usecols=['review_id', 'work_id', 'rating', 'review_date', 'nur_genre']).drop_duplicates('review_id')
nl_rev = nl_rev[nl_rev.work_id.isin(set(L.impfic_work_id))]
gt = pd.read_csv('data/gt_v2/labels.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False,
                 usecols=['review_id'])
nl_rev['labelled'] = nl_rev.review_id.isin(set(gt.review_id))
sents = pd.read_csv('data/review_sentences.tsv.gz', sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False,
                    usecols=['review_id', 'sent_text'])
sents = sents[sents.review_id.isin(set(nl_rev.review_id))]
nl_txt = sents.groupby('review_id', sort=False).sent_text.apply(' '.join).rename('text').reset_index()
nl = usable(nl_rev.merge(nl_txt, on='review_id'), 'nl')

count = pd.DataFrame({
    'nl': L.impfic_work_id.map(nl.work_id.value_counts()).fillna(0).to_numpy(),
    'en': L.gr_book_id.map(en.book_id.value_counts()).fillna(0).to_numpy(),
    'it': L.gr_book_id.map(it_ru['it'].book_id.value_counts()).fillna(0).to_numpy(),
    'ru': L.gr_book_id.map(it_ru['ru'].book_id.value_counts()).fillna(0).to_numpy()}, index=L.index)
L = L[(count >= PER).all(axis=1)]
print(f'{len(L)} works with >= {PER} usable reviews in all four languages')
print(pd.crosstab(L.genre_group, L.era, margins=True).to_string())

# equal books per genre group; within a group spread over eras (round robin over the eras that have books left)
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

rows, texts = [], {l: [] for l in ['en', 'it', 'ru']}
for b in B.itertuples():
    picks = {'nl': nl[nl.work_id == b.impfic_work_id].sample(frac=1, random_state=SEED)
                   .sort_values('labelled', ascending=False, kind='stable').head(PER),
             'en': en[en.book_id == b.gr_book_id].sample(PER, random_state=SEED),
             'it': it_ru['it'][it_ru['it'].book_id == b.gr_book_id].sample(PER, random_state=SEED),
             'ru': it_ru['ru'][it_ru['ru'].book_id == b.gr_book_id].sample(PER, random_state=SEED)}
    for lang, p in picks.items():
        for r in p.itertuples():
            rows.append({'lang': lang, 'gr_work_id': b.gr_work_id,
                         'source_review_id': r.review_id if lang == 'nl' else r.review_url,
                         'labelled': bool(getattr(r, 'labelled', False)),
                         'genre': r.nur_genre if lang == 'nl' else b.genre,
                         'rating': r.rating, 'review_year': str(r.review_date)[:4], 'n_words': r.n_words,
                         'text': r.text})
S = pd.DataFrame(rows)
S['review_id'] = [f'mt-{l}-{i + 1:05d}' for l, i in zip(S.lang, S.groupby('lang').cumcount())]
os.makedirs(OUT, exist_ok=True)
B[['gr_work_id', 'impfic_work_id', 'gr_book_id', 'genre', 'genre_group', 'first_year', 'era']].to_csv(
    f'{OUT}/books.tsv', sep='\t', index=False)
S[['review_id', 'lang', 'gr_work_id', 'source_review_id', 'labelled', 'genre', 'rating', 'review_year',
   'n_words']].to_csv(f'{OUT}/sample.tsv', sep='\t', index=False)
for lang in ['en', 'it', 'ru']:
    t = S[S.lang == lang][['review_id', 'text']].copy()
    t['text'] = t.text.map(esc)
    os.makedirs(f'{OUT}/{lang}', exist_ok=True)
    t.to_csv(f'{OUT}/{lang}/texts.tsv.gz', sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar=None)
print(S.groupby('lang').agg(reviews=('review_id', 'size'), median_words=('n_words', 'median'),
                            labelled=('labelled', 'sum')).to_string())
