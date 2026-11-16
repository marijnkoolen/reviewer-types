"""Link the Dutch corpus works to Goodreads works, for a matched-books sample across languages (UA-031 a).

usage: python3.11 src/annotate/link_dutch_goodreads.py
inputs: /Users/marijnkoolen/Code/impact-of-fiction/data/book_metadata/work_isbn_title_genre.tsv.gz (impfic work_id ->
        ISBNs, title, author, year, NUR), metadata/review_metadata.tsv.gz (Dutch reviews per work), the reader_profiles
        DuckDB on the T7 drive (gr_books: Goodreads editions with isbn/isbn13/work_id; LibraryThing thingISBN tables),
        data/other_languages/{it,ru}/gr_reviews.tsv.gz (reviews per scraped Goodreads book id).
Route 1: Dutch ISBN -> Goodreads edition (gr_books.isbn13 / isbn) -> Goodreads work_id.
Route 2 (when route 1 finds nothing): Dutch ISBN -> LibraryThing work (thingISBN groups translations) -> the other
        ISBNs of that work -> Goodreads edition -> work_id.
A Goodreads work is then matched to the scraped book ids: a scraped id is a Goodreads edition id, so every edition of the
work is looked up and the one with it/ru reviews is kept.
output: data/other_languages/book_links.tsv: impfic_work_id, gr_work_id, gr_book_id (scraped id, empty if none), route,
        n_nl (Dutch reviews), n_it, n_ru (parsed Goodreads reviews), n_en_2017 (English Goodreads reviews in
        reader_profiles), nur, work_year. Titles and authors stay out of this file (labellers never see them).
"""
import csv
import os

import duckdb
import pandas as pd

MAP = '/Users/marijnkoolen/Code/impact-of-fiction/data/book_metadata/work_isbn_title_genre.tsv.gz'
DB = '/Volumes/T7_Data/Data/claude-projects-external/reader_profiles/db/reader_profiles.duckdb'
OUT = 'data/other_languages/book_links.tsv'


def isbn13(s):
    s = ''.join(ch for ch in str(s) if ch.isdigit() or ch in 'Xx')
    if len(s) == 13:
        return s
    if len(s) == 10:
        core = '978' + s[:9]
        chk = (10 - sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(core)) % 10) % 10
        return core + str(chk)
    return None


def isbn10(s13):
    """978-prefixed ISBN-13 -> ISBN-10 (LibraryThing's thingISBN tables use ISBN-10)."""
    if not s13 or not s13.startswith('978'):
        return None
    core = s13[3:12]
    chk = (11 - sum(int(c) * (10 - i) for i, c in enumerate(core)) % 11) % 11
    return core + ('X' if chk == 10 else str(chk))


m = pd.read_csv(MAP, sep='\t', dtype=str)
nl_isbn = m[m.record_id_type == 'isbn'].assign(isbn13=lambda x: x.record_id.map(isbn13)).dropna(subset=['isbn13'])
nl_isbn = nl_isbn[['work_id', 'isbn13']].drop_duplicates()
nl_isbn['isbn10'] = nl_isbn.isbn13.map(isbn10)
works = m.drop_duplicates('work_id').set_index('work_id')
rev = pd.read_csv('metadata/review_metadata.tsv.gz', sep='\t', usecols=['review_id', 'work_id']).drop_duplicates('review_id')
n_nl = rev.work_id.value_counts()
print(f'{m.work_id.nunique()} Dutch works, {nl_isbn.work_id.nunique()} with an ISBN; '
      f'{n_nl.size} works with reviews ({n_nl.sum()} reviews)')

con = duckdb.connect(DB, read_only=True)
con.register('nl', nl_isbn)
r1 = con.execute("""
    select distinct nl.work_id, b.work_id gr_work_id
    from nl join gr_books b on b.isbn13 = nl.isbn13 where b.work_id is not null and b.work_id <> ''""").df()
left = nl_isbn[~nl_isbn.work_id.isin(r1.work_id)]
con.register('nl2', left)
r2 = con.execute("""
    with lt as (select distinct nl2.work_id, t.lt_work_id from nl2 join lt_thingisbn_nonmeta t on t.isbn = nl2.isbn10),
         other as (select distinct lt.work_id, t2.isbn from lt join lt_thingisbn_nonmeta t2 on t2.lt_work_id = lt.lt_work_id)
    select distinct other.work_id, b.work_id gr_work_id
    from other join gr_books b on b.isbn13 = other.isbn or b.isbn = other.isbn
    where b.work_id is not null and b.work_id <> ''""").df()
links = pd.concat([r1.assign(route='gr_isbn'), r2.assign(route='lt_work')], ignore_index=True)
print(f'linked works: route 1 {r1.work_id.nunique()}, route 2 {r2.work_id.nunique()}; '
      f'{links.work_id.nunique()} Dutch works -> {links.gr_work_id.nunique()} Goodreads works')

scr = {}
for lang in ['it', 'ru']:
    d = pd.read_csv(f'data/other_languages/{lang}/gr_reviews.tsv.gz', sep='\t', quoting=csv.QUOTE_NONE,
                    keep_default_na=False, usecols=['book_id'])
    scr[lang] = d.book_id.astype(str).value_counts()
scraped = pd.DataFrame({'gr_book_id': sorted(set(scr['it'].index) | set(scr['ru'].index))})
con.register('scraped', scraped)
con.register('gw', links[['gr_work_id']].drop_duplicates())
ed = con.execute("""
    select b.work_id gr_work_id, b.book_id gr_book_id from gr_books b
    join gw on gw.gr_work_id = b.work_id join scraped s on s.gr_book_id = b.book_id""").df()
en = con.execute("""
    select b.work_id gr_work_id, count(*) n_en_2017 from gr_reviews r join gr_books b on b.book_id = r.book_id
    join gw on gw.gr_work_id = b.work_id group by 1""").df()
out = links.merge(ed, on='gr_work_id', how='left').merge(en, on='gr_work_id', how='left')
out['n_it'] = out.gr_book_id.map(scr['it']).fillna(0).astype(int)
out['n_ru'] = out.gr_book_id.map(scr['ru']).fillna(0).astype(int)
out['n_en_2017'] = out.n_en_2017.fillna(0).astype(int)
out['n_nl'] = out.work_id.map(n_nl).fillna(0).astype(int)
out['nur'] = out.work_id.map(works.nur)
out['work_year'] = out.work_id.map(works.work_year)
out = out.rename(columns={'work_id': 'impfic_work_id'})
out = (out.sort_values(['impfic_work_id', 'n_it', 'n_ru'], ascending=[True, False, False])
       .drop_duplicates(['impfic_work_id', 'gr_work_id']))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out[['impfic_work_id', 'gr_work_id', 'gr_book_id', 'route', 'n_nl', 'n_it', 'n_ru', 'n_en_2017', 'nur',
     'work_year']].to_csv(OUT, sep='\t', index=False)

w = out.groupby('impfic_work_id').agg(n_nl=('n_nl', 'max'), n_it=('n_it', 'max'), n_ru=('n_ru', 'max'),
                                      n_en=('n_en_2017', 'max'))
for k in (1, 2, 5):
    ok = (w.n_nl >= k) & (w.n_it >= k) & (w.n_ru >= k) & (w.n_en >= k)
    print(f'Dutch works with >= {k} reviews in nl, it, ru and en (2017): {ok.sum()}')
print(f'with >= 2 in nl/it/ru (en ignored): {((w.n_nl >= 2) & (w.n_it >= 2) & (w.n_ru >= 2)).sum()}')
