"""Phase 4 (other languages): extract the reviews from the scraped Goodreads book pages per language.

usage: python3.11 src/annotate/parse_gr_lang_pages.py LANG [LANG ...]      e.g. it ru
Source: impfic-multilingual-reviews/data/Book_language_pages/<lang>/*.html.gz on the T7 drive (read-only), one page
per book (Goodreads' language version of the book page, which lists up to 30 reviews in that language, sorted by
likes: a page with 30 reviews shows the 30 most-liked ones, not all).
Adapted from impfic-multilingual-reviews/scripts/parse_html.py (extract_review, extract_review_rating), with two
changes: <br> becomes a line break (the original joined all strings with spaces, which loses the paragraphs that
the sentence splitter uses), and reviewer names are not kept or hashed (the review URL is the source key).
output: data/other_languages/<lang>/gr_reviews.tsv.gz (book_id, page_file, review_url, rating, review_date, likes, comments,
        position (0-based rank on the page), n_on_page, text;
        backslash, newline and tab escaped as in texts.tsv.gz), one row per review, deduplicated by review URL.
Refuses to overwrite.
"""
import csv, gzip, os, re, sys
import pandas as pd
from bs4 import BeautifulSoup
from dateutil.parser import parse as date_parse

PAGES = '/Volumes/T7_Data/Code/impact-of-fiction/impfic-multilingual-reviews/data/Book_language_pages'


def review_text(span):
    for br in span.find_all('br'):
        br.replace_with('\n')
    t = span.get_text()
    t = re.sub(r'[ \t\xa0]+', ' ', t)
    t = re.sub(r' *\n *', '\n', t)
    return re.sub(r'\n{3,}', '\n\n', t).strip()


def rating(content):
    stars = content.find('span', class_='RatingStars')
    if stars is None:
        return None
    m = re.match(r'Rating (\d) out of 5', stars.attrs.get('aria-label', ''))
    return int(m.group(1)) if m else None


def parse_page(path):
    with gzip.open(path, 'rt') if path.endswith('.gz') else open(path) as fh:
        soup = BeautifulSoup(fh, 'lxml')
    for card in soup.find_all('article', class_='ReviewCard'):
        content = card.find('section', class_='ReviewCard__content')
        span = content.find('span', class_='Formatted') if content else None
        if span is None:
            continue
        row = content.find('section', class_='ReviewCard__row')
        link = row.find('a') if row else None
        date = row.find('span', class_='Text Text__body3') if row else None
        try:
            date = date_parse(date.text).date().isoformat() if date else None
        except (ValueError, OverflowError):
            date = None
        stats = card.find(class_=re.compile('SocialFooter__statsContainer'))
        st = stats.get_text(' ', strip=True) if stats else ''
        likes = re.search(r'([\d,.]+)\s*likes?', st)
        comments = re.search(r'([\d,.]+)\s*comments?', st)
        yield {'review_url': link.attrs.get('href') if link else None, 'rating': rating(content),
               'review_date': date, 'likes': int(re.sub(r'\D', '', likes.group(1))) if likes else 0,
               'comments': int(re.sub(r'\D', '', comments.group(1))) if comments else 0, 'text': review_text(span)}


def esc(t):
    return t.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', ' ')


for lang in sys.argv[1:]:
    out = f'data/other_languages/{lang}/gr_reviews.tsv.gz'
    if os.path.exists(out):
        print(f'{out} exists; skipping {lang}')
        continue
    files = sorted(f for f in os.listdir(f'{PAGES}/{lang}') if re.match(r'\d+', f))
    rows, empty = [], 0
    for i, f in enumerate(files):
        book_id = int(re.match(r'(\d+)', f).group(1))
        revs = list(parse_page(f'{PAGES}/{lang}/{f}'))
        empty += not revs
        rows += [dict(book_id=book_id, page_file=f, position=k, n_on_page=len(revs), **r) for k, r in enumerate(revs)]
        if (i + 1) % 2000 == 0:
            print(f'{lang}: {i + 1}/{len(files)} pages, {len(rows)} reviews', flush=True)
    d = pd.DataFrame(rows)
    n = len(d)
    d = pd.concat([d[d.review_url.isna()], d[d.review_url.notna()].drop_duplicates('review_url')])
    d['rating'] = d.rating.astype('Int64')
    d['text'] = d.text.map(esc)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar=None)
    print(f'{lang}: {len(files)} pages ({empty} without reviews), {n} review cards, {len(d)} after dedup by URL, '
          f'{d.book_id.nunique()} books; no URL {d.review_url.isna().sum()}, no rating {d.rating.isna().sum()}, '
          f'no date {d.review_date.isna().sum()}')
