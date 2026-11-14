"""Validate the KSS implementation of src/28c_variance_decomposition.py on simulated outcomes with
known variance shares, on the REAL reviewer-work-platform-genre-author structure (two samples:
base connected set; Recommendations-onset connected set). y = platform + reviewer effect (with
sorting: the reviewer effect contains c * standardised mean work effect of the reviewer's books)
+ work effect (genre + author + work parts) + heteroskedastic noise (sd multiplied by a lognormal
reviewer factor). 'Truth' = the realised sample variance of each component at review level
(finite-population estimand). Spec A design (year dummies). Output: validation_kss.csv"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

M = importlib.import_module("28c_variance_decomposition")
OUT = M.OUT
var0 = lambda v: ((v - v.mean()) ** 2).mean()

SETTINGS = {  # target shares of var(y): reviewer total, work total, (genre, author, within) split of work, platform part of reviewer
    "dense": dict(a=0.35, w=0.06, plat=0.17, gen=0.08, auth=0.5, c=0.6),
    "sparse": dict(a=0.12, w=0.12, plat=0.08, gen=0.08, auth=0.5, c=0.6),
}


def gmean_s(g, v):
    return pd.Series(v).groupby(g.to_numpy()).transform("mean").to_numpy()


def simulate(d, st, rng):
    N = len(d)
    u = pd.factorize(d.user_id)[0]; w = pd.factorize(d.work_id)[0]
    pl = pd.factorize(d.source_id)[0]; ge = pd.factorize(d.genre_modal)[0]; au = pd.factorize(d.author_id)[0]
    # work effects: genre + author + within-author (work-level); standardised then scaled
    def lvl(codes_work, n):
        return rng.normal(size=n)
    wk_df = d[["work_id", "genre_modal", "author_id"]].drop_duplicates("work_id")
    gz = pd.Series(rng.normal(size=wk_df.genre_modal.nunique()), index=wk_df.genre_modal.unique())
    az = pd.Series(rng.normal(size=wk_df.author_id.nunique()), index=wk_df.author_id.unique())
    wz = pd.Series(rng.normal(size=len(wk_df)), index=wk_df.work_id)
    sg, sa = np.sqrt(st["gen"]), np.sqrt((1 - st["gen"]) * st["auth"])
    sw = np.sqrt((1 - st["gen"]) * (1 - st["auth"]))
    wi = sg * d.genre_modal.map(gz).to_numpy() + sa * d.author_id.map(az).to_numpy() + sw * d.work_id.map(wz).to_numpy()
    wi = wi - wi.mean(); wi *= np.sqrt(st["w"] / var0(wi))
    # reviewer effects: platform + within + sorting
    pz = pd.Series(rng.normal(size=d.source_id.nunique()), index=d.source_id.unique())
    rz = pd.Series(rng.normal(size=d.user_id.nunique()), index=d.user_id.unique())
    zbar = pd.Series(wi).groupby(d.user_id.to_numpy()).transform("mean").to_numpy()
    zbar = (zbar - zbar.mean()) / zbar.std()
    ai = (np.sqrt(st["plat"]) * d.source_id.map(pz).to_numpy() / 1.0
          + np.sqrt(1 - st["plat"]) * d.user_id.map(rz).to_numpy())
    ai = ai / ai.std() + st["c"] * zbar
    ai = ai - ai.mean(); ai *= np.sqrt(st["a"] / var0(ai))
    # noise, heteroskedastic by reviewer (and level differs)
    mult = np.exp(0.7 * d.user_id.map(pd.Series(rng.normal(size=d.user_id.nunique()), index=d.user_id.unique())).to_numpy())
    e = rng.normal(size=N) * mult
    e *= np.sqrt((1 - st["a"] - st["w"]) / var0(e))
    y = ai + wi + e
    pairs = {"a": ai, "aP": gmean_s(d.source_id, ai), "w": wi, "wG": gmean_s(d.genre_modal, wi),
             "wA": gmean_s(d.author_id, wi), "c": np.zeros(N)}
    tv = {k: var0(v) for k, v in pairs.items()}
    for p, q in M.PAIRS:
        tv[p + "+" + q] = var0(pairs[p] + pairs[q])
    tv["resid"] = var0(e)
    return y, M.shares(var0(y), tv, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    df = pd.read_parquet("/private/tmp/claude-504/scr/base.parquet") if False else M.build_base()[0]
    samples = {"base": df, "sparse_recs_onset": df[df["onset__Recommendations"].notna()]}
    rows = []
    for sname, d0 in samples.items():
        d = d0[M.loo_connected(d0)].reset_index(drop=True)
        print(sname, len(d), flush=True)
        m = M.Model(d, False)
        Pii, B = M.leverage_and_B(m, M.COMPS_A, a.draws, 5)
        for stn, st in SETTINGS.items():
            for rep in range(a.reps):
                rng = np.random.default_rng(1000 + rep)
                y, truth = simulate(d, st, rng)
                plug, corr = M.estimate(m, y, M.COMPS_A, Pii, B)
                vy = var0(y)
                ps, cs = M.shares(vy, plug, False), M.shares(vy, corr, True)
                for k in truth:
                    rows.append(dict(sample=sname, setting=stn, rep=rep, component=k, truth=truth[k],
                                     plugin=ps[k], corrected=cs[k]))
                print(sname, stn, rep, "done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "validation_kss.csv", index=False)


if __name__ == "__main__":
    main()
