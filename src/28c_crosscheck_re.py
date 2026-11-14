"""P1 cross-check: crossed random effects  y ~ year FE + (1|reviewer) + (1|work)  fitted with NUTS
(PyMC; NOT ADVI) on a random subsample of the leave-one-out connected set, for a few outcomes.
Non-centred parametrisation; HalfNormal priors on the 3 SDs; 4 chains x 1000 draws after 1000 tuning;
fixed seeds. Reports posterior mean (95% HDI) of sigma_r^2, sigma_w^2, sigma_e^2 as shares of
(a) var(y) in the subsample and (b) the sum of the three variance components, plus convergence
(max R-hat, min ESS, divergences). Requires results/28_variance_decomposition/connected_set_loglen.parquet
(written by 28c_variance_decomposition.py: the base connected set).
Usage: python src/28c_crosscheck_re.py [--n 25000] [--only a,b]
"""
import argparse
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/28_variance_decomposition"
OUTS = {"prop__Content": "prop__Content", "prop__Reader_response": "prop__Reader_response",
        "loglen": "loglen", "contig_excess": "excess_over_chance"}


def fit(d, col, seed):
    y = d[col].to_numpy(float)
    yc = (y - y.mean()) / y.std()
    r, _ = pd.factorize(d.user_id)
    w, _ = pd.factorize(d.work_id)
    yr = pd.get_dummies(d.year).to_numpy(float)[:, 1:]
    with pm.Model():
        sr = pm.HalfNormal("sigma_r", 1.0)
        sw = pm.HalfNormal("sigma_w", 1.0)
        se = pm.HalfNormal("sigma_e", 1.0)
        zr = pm.Normal("zr", 0, 1, shape=int(r.max() + 1))
        zw = pm.Normal("zw", 0, 1, shape=int(w.max() + 1))
        b0 = pm.Normal("b0", 0, 1)
        by = pm.Normal("by", 0, 1, shape=int(yr.shape[1]))
        mu = b0 + pm.math.dot(yr, by) + sr * zr[r] + sw * zw[w]
        pm.Normal("obs", mu, se, observed=yc)
        idata = pm.sample(1000, tune=1000, chains=4, cores=1, random_seed=seed, target_accept=0.9,
                          progressbar=False, idata_kwargs={"log_likelihood": False})
        # year fixed-effect variance per draw
    post = idata.posterior
    s = {k: post[k].values.reshape(-1) ** 2 for k in ["sigma_r", "sigma_w", "sigma_e"]}
    byv = post["by"].values.reshape(-1, yr.shape[1])
    vyear = ((yr @ byv.T) - (yr @ byv.T).mean(0)).var(0)
    tot = s["sigma_r"] + s["sigma_w"] + s["sigma_e"]
    out = {}
    for nm, v in [("reviewer", s["sigma_r"]), ("work", s["sigma_w"]), ("resid", s["sigma_e"])]:
        for lab, den in [("share_of_var_y", 1.0), ("share_of_RE_total", tot)]:
            x = v / den
            lo, hi = np.percentile(x, [2.5, 97.5])
            out[f"{nm}_{lab}"] = x.mean(); out[f"{nm}_{lab}_lo"] = lo; out[f"{nm}_{lab}_hi"] = hi
    out["year_share_of_var_y"] = vyear.mean()
    summ = az.summary(idata, var_names=["sigma_r", "sigma_w", "sigma_e"])
    out["max_rhat"] = float(summ["r_hat"].max()); out["min_ess_bulk"] = float(summ["ess_bulk"].min())
    out["divergences"] = int(idata.sample_stats["diverging"].sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cs = pd.read_parquet(OUT / "connected_set_loglen.parquet")
    rows = []
    for name, col in OUTS.items():
        if a.only and name not in a.only.split(","):
            continue
        d0 = cs[cs[col].notna()]
        d = d0.sample(a.n, random_state=a.seed).reset_index(drop=True)
        t0 = time.time()
        o = fit(d, col, a.seed)
        o.update(outcome=name, n=len(d), n_reviewers=d.user_id.nunique(), n_works=d.work_id.nunique(),
                 n_reviewers_ge2=int((d.user_id.value_counts() >= 2).sum()),
                 n_works_ge2=int((d.work_id.value_counts() >= 2).sum()), secs=time.time() - t0)
        rows.append(o)
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in o.items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"crosscheck_random_effects{'_' + a.only.replace(',', '_') if a.only else ''}.csv", index=False)


if __name__ == "__main__":
    main()
