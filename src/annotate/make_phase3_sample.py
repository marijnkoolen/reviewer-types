"""Phase 3: draw extra reviews enriched for the rarest v2.3 labels, plus a random control stratum.

usage: python src/annotate/make_phase3_sample.py
input : data/predctions-robbert-sentences-all.tsv.gz (old RobBERT sentence predictions, whole pool)
        data/full_run/majority_labels.tsv (v2.3 labels for the 1,250 annotated reviews)
        data/ground_truth-test_set.tsv (the 815 annotated reviews RobBERT did NOT train on)
        metadata/review_metadata.tsv.gz
output: data/phase3/sample.tsv          one row per drawn review: review_id, stratum, inclusion probability, scores
        data/phase3/selection_model.tsv per label: cross-validated AUC and precision of the selection model
Refuses to overwrite data/phase3/sample.tsv.

Method
- Selection model per target label: logistic regression predicting "review has >= 1 sentence with the label" from
  RobBERT's predicted label counts/fractions per review, review length, rating and genre. Trained on the 815 RobBERT
  test reviews only (honest features); 5-fold CV estimates are written out.
- The model is unweighted, so its probabilities are roughly calibrated (CV check in selection_model.tsv).
- Ranking: expected label sentences per annotated sentence = p(review has label) * k / n_sentences, with k the mean
  number of label sentences in a positive annotated review. Long reviews are not denser in these labels, and cost
  scales with sentences, so this maximises positives per unit of annotation cost.
- Pools (minus the 1,250 annotated): random stratum 3-60 sentences; enriched strata 3-30 sentences.
- Strata (drawn in this order, without replacement across strata):
    random  N_RANDOM reviews uniformly from its pool              -> prevalence estimates for all labels
    RCP/REF/SCX/MET  N reviews uniformly from the top TOP_FRAC of the enriched pool by that label's density
  Uniform draws inside a fixed top set keep the inclusion probability simple (N / size of the set), so prevalence
  in the pool can be estimated with weights 1/p.
"""
import csv, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 20261114
N_RANDOM = 500
TARGETS = {'RCP': 300, 'REF': 400, 'SCX': 1000, 'MET': 400}  # reviews per enriched stratum (~300 extra label sentences each)
TOP_FRAC = 0.02                                              # top 2% of the enriched pool (~10k reviews)
CALIB = {'SCX': 0.75}                                        # CV shows SCX probabilities ~25% too high
OUT = 'data/phase3'
if os.path.exists(f'{OUT}/sample.tsv'):
    sys.exit(f'{OUT}/sample.tsv exists; the sample is fixed. Delete it deliberately to redraw.')
os.makedirs(OUT, exist_ok=True)

P = {'AUT': 'aut', 'CLS': 'cla', 'OTW': 'oth', 'REC': 'rec', 'NAR': 'con_nar', 'COO': 'con_oth', 'QUO': 'con_quo',
     'THM': 'con_the', 'EVA': 'rea_eva', 'FEE': 'rea_fee', 'IDI': 'rea_ide', 'CTX': 'rea_rea', 'RCP': 'rea_rec',
     'REF': 'rea_ref', 'SCX': 'sty_con', 'STR': 'sty_str', 'STY': 'sty_sty'}

p = pd.read_csv('data/predctions-robbert-sentences-all.tsv.gz', sep='\t')
rp = p.groupby('review_id')[list(P.values())].sum()
rp['nsent'] = p.groupby('review_id').size()
meta = (pd.read_csv('metadata/review_metadata.tsv.gz', sep='\t', usecols=['review_id', 'nur_genre', 'rating', 'review_date'])
        .drop_duplicates('review_id').set_index('review_id'))

X = pd.DataFrame(index=rp.index)
for c in P.values():
    X[c + '_any'] = (rp[c] > 0).astype(float)
    X[c + '_frac'] = rp[c] / rp['nsent']
X['logn'] = np.log1p(rp['nsent'])
mm = meta.reindex(rp.index)
X['rating'] = mm.rating.fillna(3.0)
X['rating_na'] = mm.rating.isna().astype(float)
X = X.join(pd.get_dummies(mm.nur_genre.fillna('unknown'), prefix='g').astype(float))

m = pd.read_csv('data/full_run/majority_labels.tsv', sep='\t', quoting=csv.QUOTE_NONE, escapechar='\\', keep_default_na=False)
annotated = set(m.review_id)
test = set(pd.read_csv('data/ground_truth-test_set.tsv', sep='\t').review_id)
cnt = {lab: m.groupby('review_id').labels.apply(lambda s: s.str.contains(rf'(?:^|,){lab}(?:,|$)').sum()) for lab in TARGETS}
train = sorted(test & annotated & set(X.index))

scores, report, exp_pos = {}, [], {}
for lab in TARGETS:
    y = (cnt[lab].reindex(train) > 0).astype(int).values
    k = cnt[lab][cnt[lab] > 0].mean()
    clf = LogisticRegression(C=0.5, max_iter=3000)
    cv = cross_val_predict(clf, X.loc[train].values, y, method='predict_proba',
                           cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
    o = np.argsort(-cv)
    report.append(dict(label=lab, base_rate=y.mean(), cv_auc=roc_auc_score(y, cv), k=k,
                       prec_top5=y[o[:len(y) // 20]].mean(), prec_top10=y[o[:len(y) // 10]].mean(),
                       pred_top10=cv[o[:len(y) // 10]].mean()))
    clf.fit(X.loc[train].values, y)
    pr = pd.Series(clf.predict_proba(X.values)[:, 1], index=X.index) * CALIB.get(lab, 1.0)
    scores[lab] = pr
    exp_pos[lab] = pr * k
pd.DataFrame(report).round(3).to_csv(f'{OUT}/selection_model.tsv', sep='\t', index=False)

pool = rp.index[(~rp.index.isin(annotated)) & rp['nsent'].between(3, 60)]
epool = rp.index[(~rp.index.isin(annotated)) & rp['nsent'].between(3, 30)]
rng = np.random.default_rng(SEED)
taken, rows = set(), []
cand = pool.to_numpy()
pick = rng.choice(cand, N_RANDOM, replace=False)
for r in pick:
    taken.add(r); rows.append((r, 'random', N_RANDOM / len(cand)))
for lab, n in TARGETS.items():
    s = exp_pos[lab].loc[epool] / rp['nsent'].loc[epool]
    top = s[s >= s.quantile(1 - TOP_FRAC)].index.difference(list(taken)).to_numpy()
    pick = rng.choice(top, n, replace=False)
    for r in pick:
        taken.add(r); rows.append((r, lab, n / len(top)))

out = pd.DataFrame(rows, columns=['review_id', 'stratum', 'inclusion_prob']).set_index('review_id')
for lab in TARGETS:
    out['p_' + lab] = scores[lab].reindex(out.index).round(4)
    out['exp_' + lab] = exp_pos[lab].reindex(out.index).round(3)
out['n_sentences'] = rp['nsent'].reindex(out.index)
out = out.join(mm[['nur_genre', 'rating', 'review_date']])
out.reset_index().to_csv(f'{OUT}/sample.tsv', sep='\t', index=False)

print(pd.DataFrame(report).round(2).to_string(index=False))
print(len(out), 'reviews,', int(out.n_sentences.sum()), 'sentences;', out.stratum.value_counts().to_dict())
print('sentences per stratum:', out.groupby('stratum').n_sentences.sum().to_dict())
print('expected extra label sentences (model estimate):', {l: int(out['exp_' + l].sum()) for l in TARGETS})
