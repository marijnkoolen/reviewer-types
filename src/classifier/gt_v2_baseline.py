"""Baseline: TF-IDF + one-vs-rest logistic regression on the v2.4 ground truth.

Features (only the labellers' context): the sentence (word 1-2 grams + char 2-5 grams), the previous and
next sentence of the review (word n-grams, down-weighted), the review header (genre, rating, review year)
and the sentence position in the review. Thresholds per label tuned on dev.

Run: python src/classifier/gt_v2_baseline.py   (after gt_v2_data.py; ~2 min on an M1 Max)
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder

from gt_v2_data import LABELS, load, to_matrix
from gt_v2_eval import evaluate_all, tune_thresholds

CTX_WEIGHT = 0.5
C = 4.0


def features(d, fit_mask):
    prev = d.groupby('review_id').sent_text.shift(1).fillna('')
    nxt = d.groupby('review_id').sent_text.shift(-1).fillna('')
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)
    wc = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    w.fit(d.sent_text[fit_mask]); c.fit(d.sent_text[fit_mask])
    wc.fit(np.concatenate([prev[fit_mask], nxt[fit_mask]]))
    hdr = d.header.str.extract(r'Genre: (.*) · Rating: (.*) · Review year: (.*)')
    oh = OneHotEncoder(handle_unknown='ignore').fit(hdr[fit_mask])
    rel = (d.pos / d.n_sent.clip(lower=1)).to_numpy()
    num = np.c_[rel, d.pos == 0, d.pos == d.n_sent - 1, np.log1p(d.n_sent) / 5].astype(float)
    return sp.hstack([w.transform(d.sent_text), c.transform(d.sent_text),
                      CTX_WEIGHT * wc.transform(prev), CTX_WEIGHT * wc.transform(nxt),
                      oh.transform(hdr), sp.csr_matrix(num)]).tocsr()


def main():
    d = load()
    tr = (d.split == 'train').to_numpy()
    dev = (d.split == 'dev').to_numpy()
    X = features(d, tr)
    Y = to_matrix(d.labels)
    clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000, solver='liblinear'), n_jobs=-1)
    clf.fit(X[tr], Y[tr])
    P = clf.predict_proba(X)
    th = tune_thresholds(Y[dev], P[dev])
    summary, pl = evaluate_all('tfidf_lr', d, P, th)
    for k, s in summary.items():
        print(f"{k:24s} n={s['n_sentences']:5d} micro-F1={s['micro_f1']:.3f} macro-F1={s['macro_f1']:.3f}")
    print(pl[pl.eval_set.isin(['test', 'gold_human'])].pivot(index='label', columns='eval_set', values='f1')
          .reindex(LABELS).round(3))


if __name__ == '__main__':
    main()
