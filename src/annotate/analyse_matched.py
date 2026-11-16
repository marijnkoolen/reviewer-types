"""Matched-books sample (UA-031 a, UA-032): compare label shares across the five arms (nl, en, it, ru Goodreads; nlc
Dutch corpus) on the same 152 books.

usage: venv/bin/python src/annotate/analyse_matched.py [ARM ...]      (default: all arms with majority_labels.tsv)
Input: data/matched/sample.tsv, data/matched/books.tsv, data/matched/<arm>/majority_labels.tsv,
       data/gt_v2/labels.tsv (for the nlc reviews that already have ground-truth labels).
Output (printed, and written to results/matched/):
  shares_by_arm.tsv          per arm: % of sentences with each label, and % of reviews with it (review-weighted:
                             mean over reviews of the share of its sentences, so long reviews do not dominate)
  shares_by_arm_band.tsv     the same per arm x band (top = top-5 by likes, bottom = bottom third; nlc has no band)
  shares_by_arm_genre.tsv    the same per arm x genre group
  diff_vs_nl.tsv            per arm vs Dutch Goodreads (nl): difference in review-weighted share, paired by book, with
                             a 95% bootstrap interval over the 152 books (2,000 resamples)
  adj_vs_nl.tsv              per label: OLS of the review's sentence share on arm (reference nl) + log(words) + book
                             fixed effects, SE clustered by book: arm effects with length and book held equal
  shares_by_arm_length.tsv   review-weighted shares per arm x review-length bin
  nlc_vs_gt.tsv              the 24 nlc reviews already in gt_v2: exact agreement and per-label counts
"""
import csv, os, sys
import pandas as pd

BASE = 'data/matched'
OUT = 'results/matched'
ARMS = sys.argv[1:] or [a for a in ['nl', 'en', 'it', 'ru', 'nlc'] if os.path.exists(f'{BASE}/{a}/majority_labels.tsv')]
CODES = ['NAR', 'EVA', 'FEE', 'STY', 'STR', 'THM', 'CTX', 'COO', 'RCP', 'IDI', 'REF', 'SCX', 'MET', 'QUO', 'REC', 'AUT',
         'CLS', 'OTW']


def read(p):
    return pd.read_csv(p, sep='\t', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


S = read(f'{BASE}/sample.tsv')
B = read(f'{BASE}/books.tsv')[['gr_work_id', 'genre_group', 'era']]
S = S.merge(B, on='gr_work_id', how='left')
S['band'] = S.band.replace('', 'corpus')

M = pd.concat([read(f'{BASE}/{a}/majority_labels.tsv') for a in ARMS])
M['labset'] = M.labels.map(lambda s: set(x for x in s.split(',') if x))
seen = sorted(set().union(*M.labset))
codes = [c for c in CODES if c in seen] + [c for c in seen if c not in CODES]
for c in codes:
    M[c] = M.labset.map(lambda s, c=c: c in s).astype(float)
M['none'] = M.labset.map(lambda s: not s).astype(float)
M['n_labels'] = M.labset.map(len).astype(float)
M = M.merge(S[['review_id', 'arm', 'band', 'genre_group', 'era', 'gr_work_id']], on='review_id', how='left')
assert M.arm.notna().all()
cols = codes + ['none']


def shares(g):
    sent = g[cols].mean() * 100
    rev = g.groupby('review_id')[cols].mean().mean() * 100
    r = pd.concat([sent.add_prefix('s_'), rev.add_prefix('r_')])
    r['sentences'] = len(g)
    r['reviews'] = g.review_id.nunique()
    r['labels_per_sent'] = g.n_labels.mean()
    return r.round(1)


os.makedirs(OUT, exist_ok=True)
pd.set_option('display.width', 250, 'display.max_columns', 40)
by_arm = M.groupby('arm').apply(shares).reindex(ARMS)
by_arm.to_csv(f'{OUT}/shares_by_arm.tsv', sep='\t')
by_band = M.groupby(['arm', 'band']).apply(shares)
by_band.to_csv(f'{OUT}/shares_by_arm_band.tsv', sep='\t')
by_genre = M.groupby(['genre_group', 'arm']).apply(shares)
by_genre.to_csv(f'{OUT}/shares_by_arm_genre.tsv', sep='\t')

show = lambda d, p: d[[f'{p}_{c}' for c in cols]].rename(columns=lambda c: c[2:])
print('% of sentences with label, per arm')
print(show(by_arm, 's').join(by_arm[['sentences', 'labels_per_sent']]).to_string())
print('\n% review-weighted (mean over reviews of the share of its sentences), per arm')
print(show(by_arm, 'r').join(by_arm[['reviews']]).to_string())
print('\n% review-weighted, per arm x band')
print(show(by_band, 'r').join(by_band[['reviews']]).to_string())
print('\n% review-weighted, per genre group x arm')
print(show(by_genre, 'r').join(by_genre[['reviews']]).to_string())

# paired by book: per book and arm the mean of its reviews' sentence shares; bootstrap over books
if 'nl' in ARMS and len(ARMS) > 1:
    import numpy as np
    rng = np.random.default_rng(1)
    R = M.groupby(['arm', 'gr_work_id', 'review_id'])[cols].mean().groupby(['arm', 'gr_work_id']).mean() * 100
    base = R.loc['nl']
    idx = rng.integers(0, len(base), size=(2000, len(base)))
    rows = []
    for a in [x for x in ARMS if x != 'nl']:
        d = (R.loc[a] - base.reindex(R.loc[a].index)).dropna()
        v = d.to_numpy()
        boot = v[idx % len(v)].mean(axis=1)
        for j, c in enumerate(cols):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            rows.append(dict(arm=a, label=c, diff=v[:, j].mean(), lo=lo, hi=hi, books=len(v)))
    D = pd.DataFrame(rows).round(1)
    D['sig'] = np.where((D.lo > 0) | (D.hi < 0), '*', '')
    D.to_csv(f'{OUT}/diff_vs_nl.tsv', sep='\t', index=False)
    print('\nreview-weighted share minus Dutch Goodreads (nl), paired by book; * = 95% bootstrap interval excludes 0')
    P = D.assign(v=D['diff'].map('{:+.1f}'.format) + D.sig).pivot(index='arm', columns='label', values='v')
    print(P[cols].reindex([x for x in ARMS if x != 'nl']).to_string())

# length-adjusted: longer reviews have more NAR/THM and a smaller EVA share, and arms differ a lot in length
if 'nl' in ARMS and len(ARMS) > 1:
    import numpy as np, statsmodels.formula.api as smf
    V = M.groupby(['review_id', 'arm', 'gr_work_id'])[cols].mean().mul(100).reset_index()
    V = V.merge(S[['review_id', 'n_words']], on='review_id')
    V['log_words'] = np.log(V.n_words.astype(float).clip(lower=1))
    V['len_bin'] = pd.cut(V.n_words.astype(float), [0, 50, 100, 200, 400, 1e9],
                          labels=['<50', '50-99', '100-199', '200-399', '400+'], right=False)
    L = V.groupby(['arm', 'len_bin'], observed=True)[cols].mean().round(1)
    L['reviews'] = V.groupby(['arm', 'len_bin'], observed=True).size()
    L.to_csv(f'{OUT}/shares_by_arm_length.tsv', sep='\t')
    print('\n% review-weighted, per arm x review length (words)')
    print(L[['NAR', 'EVA', 'FEE', 'STY', 'THM', 'QUO', 'REF', 'none', 'reviews']].to_string())
    rows = []
    for c in cols:
        V['y'] = V[c]
        f = smf.ols("y ~ C(arm, Treatment('nl')) + log_words + C(gr_work_id)", data=V).fit(
            cov_type='cluster', cov_kwds={'groups': V.gr_work_id.astype('category').cat.codes})
        for a in [x for x in ARMS if x != 'nl']:
            k = f"C(arm, Treatment('nl'))[T.{a}]"
            rows.append(dict(label=c, arm=a, coef=f.params[k], p=f.pvalues[k], log_words=f.params['log_words']))
    A = pd.DataFrame(rows)
    A.to_csv(f'{OUT}/adj_vs_nl.tsv', sep='\t', index=False)
    print('\nlength- and book-adjusted difference from nl (percentage points; * p < 0.05, clustered by book); '
          'last row: effect of doubling review length')
    P = A.assign(v=A.coef.map('{:+.1f}'.format) + np.where(A.p < 0.05, '*', '')).pivot(
        index='arm', columns='label', values='v')[cols].reindex([x for x in ARMS if x != 'nl'])
    P.loc['x2 words'] = (A.groupby('label').log_words.first() * np.log(2)).map('{:+.1f}'.format)[cols]
    print(P.to_string())

# nlc reviews already in the ground truth: same sentence split, so compare by sentence number
if 'nlc' in ARMS:
    G = read('data/gt_v2/labels.tsv')
    lab = S[(S.arm == 'nlc') & (S.labelled == 'True')][['review_id', 'source_review_id']]
    N = M[M.review_id.isin(lab.review_id)].merge(lab, on='review_id')
    N['sent_num'] = N.sent_id.str.rsplit('-', n=1).str[-1].astype(int)
    G = G[G.review_id.isin(lab.source_review_id)].copy()
    G['sent_num'] = G.sent_id.str.rsplit('-', n=1).str[-1].astype(int)
    J = N.merge(G[['review_id', 'sent_num', 'labels', 'status', 'set']].rename(
        columns={'review_id': 'source_review_id', 'labels': 'gt_labels', 'status': 'gt_status'}),
        on=['source_review_id', 'sent_num'], how='left')
    J['gt_set'] = J.gt_labels.fillna('').map(lambda s: set(x for x in s.split(',') if x))
    assert J.gt_labels.notna().all(), 'sentence split differs from gt_v2'
    exact = (J.labset == J.gt_set).mean() * 100
    rows = []
    for c in codes:
        a, g = J.labset.map(lambda s: c in s), J.gt_set.map(lambda s: c in s)
        rows.append(dict(label=c, n_matched=a.sum(), n_gt=g.sum(), n_both=(a & g).sum()))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/nlc_vs_gt.tsv', sep='\t', index=False)
    print(f'\nnlc vs gt_v2: {J.review_id.nunique()} reviews, {len(J)} sentences, exact agreement {exact:.1f}% '
          f'(gt sets: {J.set.value_counts().to_dict()})')
    print(R[(R.n_matched > 0) | (R.n_gt > 0)].to_string(index=False))
