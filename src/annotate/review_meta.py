"""Review metadata header shown above each review.

Principle: every labeller (human adjudicator, annotation agent) and the model that is later trained
on the ground truth and applied to the full corpus must get the SAME context. So the header holds only
fields that exist in metadata/review_metadata.tsv.gz for the whole corpus (genre, rating, review year).
No book titles, authors or other outside knowledge. Missing values are shown as 'unknown' so the
format is identical for every review. Platform ids are anonymised, so they are left out.
"""
import pandas as pd

_META = 'metadata/review_metadata.tsv.gz'


def format_header(nur_genre, rating, review_date):
    genre = nur_genre.replace('_', ' ') if isinstance(nur_genre, str) else 'unknown'
    rate = f'{rating:g}' if pd.notna(rating) else 'unknown'
    year = review_date[:4] if isinstance(review_date, str) else 'unknown'
    return f'Genre: {genre} · Rating: {rate} · Review year: {year}'


def review_headers(review_ids):
    """Return {review_id: header string} for the given review ids."""
    ids = set(review_ids)
    m = pd.read_csv(_META, sep='\t', usecols=['review_id', 'nur_genre', 'rating', 'review_date'])
    m = m[m.review_id.isin(ids)]
    return {r.review_id: format_header(r.nur_genre, r.rating, r.review_date) for r in m.itertuples()}
