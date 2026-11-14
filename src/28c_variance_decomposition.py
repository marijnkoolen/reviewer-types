"""P1: reviewer vs book vs genre vs platform variance decomposition (silver).

Two-way fixed effects (reviewer FE + work FE, the AKM worker-firm model; reviewer ~ worker,
work ~ firm) with the Kline-Saggio-Solvsten (KSS) LEAVE-OUT bias correction, implemented
here directly (pytwoway/bipartitepandas could not be used: they pin pandas 2.0.x, which is
binary-incompatible with the project's numpy 2.x; installing them broke the venv, which was
restored -- see FINDINGS.md).

Model, per outcome y (one value per review i):
    y_i = alpha_r(i) + psi_w(i) + x_i' gamma + e_i
x_i = year dummies (spec A) or year dummies + log(num_sentences) (spec B). gamma, alpha and
psi are estimated JOINTLY by sparse least squares on the full design (reviewer dummies +
work dummies + covariates), solved by a sparse LU of the FE block + a Schur complement for
the (dense, few) covariate columns. No separate residualising step.

Variance components are quadratic forms theta = beta' A beta, A = Z'Z, Z = centred,
N^-1/2-scaled review-level fitted part (e.g. alpha_r(i), or the platform-mean of alpha_r(i)).
Plug-in: ||Z beta_hat||^2 (biased upward by sampling noise in the FE). KSS leave-out:
    theta_LO = ||Z b||^2 - sum_i y_i e_i / (1-P_ii) * B_ii ,  B_ii = ||Z S^-1 x_i||^2,
S = X'X, P_ii = x_i' S^-1 x_i (leverage). Unbiased under arbitrary heteroskedasticity if the
sample is leave-one-observation-out connected (bridges are removed iteratively).
P_ii and B_ii are computed with Johnson-Lindenstrauss random projections (Rademacher), as in
KSS; fixed seeds. Components: reviewer; platform-mean of reviewer effect; work; genre-mean /
author-mean of work effect; covariates (and log length alone, spec B); plus sums
(alpha+psi, alpha+c, psi+c) so that the covariances (sorting) are bias-corrected too.
Between-group parts are corrected with the same machinery (group mean = linear operator);
within = corrected total - between. Residual share = 1 - all other components.

Usage: python src/28c_variance_decomposition.py [--draws 300] [--only name,...] [--jobs 6]
Writes results/28_variance_decomposition/*.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spl
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "28_variance_decomposition"
MAIN7 = ["Author", "Classification", "Content", "Other_works", "Reader_response",
         "Recommendations", "Style"]


# ------------------------------------------------------------------ data
def build_base():
    f = pd.read_parquet(RES / "00_data_prep/silver_review_features.parquet")
    f = f[f.source_id != "platform_1"].copy()
    f["year"] = pd.to_numeric(f.review_date.str[:4], errors="coerce")
    f = f[f.year >= 1999].copy()                      # drops undated and the 1970 placeholders
    f["year"] = f.year.astype(int)
    f["loglen"] = np.log(f.num_sentences)
    c = pd.read_csv(RES / "22_review_contiguity_scaled/silver_review_contiguity_scaled.csv",
                    usecols=["review_id", "excess_over_chance", "boundary_excess", "eligible_a1"])
    c.loc[~c.eligible_a1, ["excess_over_chance", "boundary_excess"]] = np.nan
    d = pd.read_csv(RES / "09_length_dimension/silver_density_specificity_by_review.csv",
                    usecols=["review_id", "mean_density", "specificity"])
    f = f.merge(c[["review_id", "excess_over_chance", "boundary_excess"]], on="review_id", how="left")
    f = f.merge(d, on="review_id", how="left")
    s = pd.read_parquet(RES / "00_data_prep/silver_sentence_sequences.parquet",
                        columns=["review_id", "num_sentences", "rel_position"] + MAIN7)
    s = s[s.num_sentences > 1]                        # same definition as 04c onset_analysis
    on = {f"onset__{cat}": s.loc[s[cat] == 1].groupby("review_id").rel_position.min() for cat in MAIN7}
    f = f.merge(pd.DataFrame(on), left_on="review_id", right_index=True, how="left")
    # author modal genre (work-weighted; ties -> more reviews -> alphabetical)
    wg = f.groupby(["author_id", "work_id", "nur_genre"]).size().rename("n").reset_index()
    ag = wg.groupby(["author_id", "nur_genre"]).agg(nw=("work_id", "size"), n=("n", "sum")).reset_index()
    ag = ag.sort_values(["author_id", "nw", "n", "nur_genre"], ascending=[True, False, False, True])
    modal = ag.drop_duplicates("author_id").set_index("author_id").nur_genre
    f["genre_modal"] = f.author_id.map(modal)
    ginfo = {
        "authors": int(f.author_id.nunique()),
        "authors_multi_genre": int((wg.groupby("author_id").nur_genre.nunique() > 1).sum()),
        "works": int(f.work_id.nunique()),
        "works_not_in_author_modal_genre": int((f.drop_duplicates("work_id").nur_genre
                                                != f.drop_duplicates("work_id").genre_modal).sum()),
        "reviews": len(f),
        "reviews_not_in_author_modal_genre": int((f.nur_genre != f.genre_modal).sum()),
    }
    keep = ["review_id", "user_id", "work_id", "author_id", "source_id", "nur_genre", "genre_modal",
            "year", "loglen", "num_sentences", "rating", "excess_over_chance", "boundary_excess",
            "mean_density", "specificity"] + [f"prop__{c}" for c in MAIN7] + [f"onset__{c}" for c in MAIN7]
    return f[keep].reset_index(drop=True), ginfo


OUTCOMES = (
    [(f"prop__{c}", f"prop__{c}") for c in MAIN7]
    + [("loglen", "loglen"), ("contig_excess", "excess_over_chance"),
       ("boundary_excess", "boundary_excess")]
    + [(f"onset__{c}", f"onset__{c}") for c in MAIN7]
    + [("rating", "rating"), ("density", "mean_density"), ("specificity", "specificity")]
)


# ------------------------------------------------------------------ connected set
def loo_connected(df):
    """Iterate until stable: reviewer/work count >= 2; largest connected component; remove
    bridges (reviews whose deletion disconnects the reviewer-work graph, multiplicity 1).
    Returns a boolean mask over df: the leave-one-observation-out connected set."""
    keep = np.ones(len(df), bool)
    u_all = pd.factorize(df.user_id)[0]
    w_all = pd.factorize(df.work_id)[0]
    nu = u_all.max() + 1
    while True:
        idx = np.flatnonzero(keep)
        u, w = u_all[idx], w_all[idx] + nu
        cnt = np.bincount(np.concatenate([u, w]), minlength=nu + w_all.max() + 1)
        ok = (cnt[u] >= 2) & (cnt[w] >= 2)
        if not ok.all():
            keep[idx[~ok]] = False
            continue
        A = sp.coo_matrix((np.ones(len(u)), (u, w)), shape=(nu + w_all.max() + 1,) * 2)
        _, lab = csg.connected_components(A, directed=False)
        ok = lab[u] == np.bincount(lab[u]).argmax()
        if not ok.all():
            keep[idx[~ok]] = False
            continue
        key = u.astype(np.int64) * (w.max() + 1) + w
        _, inv, mc = np.unique(key, return_inverse=True, return_counts=True)
        mult = mc[inv]
        G = nx.Graph()
        G.add_edges_from(zip(u.tolist(), w.tolist()))
        br = {(min(a, b), max(a, b)) for a, b in nx.bridges(G)}
        isb = np.fromiter(((min(a, b), max(a, b)) in br for a, b in zip(u.tolist(), w.tolist())),
                          bool, len(u)) & (mult == 1)
        if not isb.any():
            return keep
        keep[idx[isb]] = False


# ------------------------------------------------------------------ estimation
class Model:
    def __init__(self, d, use_len, base=None):
        self.N = N = len(d)
        ur, ru = pd.factorize(d.user_id)
        wk, wu = pd.factorize(d.work_id)
        self.nR, self.nW = len(ru), len(wu)
        yrs = pd.factorize(d.year, sort=True)[0]
        ny = yrs.max() + 1
        ones = np.ones(N)
        ar = np.arange(N)
        self.XR = sp.csr_matrix((ones, (ar, ur)), shape=(N, self.nR))
        self.XW = sp.csr_matrix((ones, (ar, wk)), shape=(N, self.nW))[:, 1:]   # ground work 0
        cols = [sp.csr_matrix((ones, (ar, yrs)), shape=(N, ny))[:, 1:]]
        if use_len:
            ll = d.loglen.to_numpy()
            cols.append(sp.csr_matrix((ll - ll.mean())[:, None]))
        self.XC = XC = sp.hstack(cols).tocsr()
        self.nF = self.nR + self.nW - 1
        self.XF = sp.hstack([self.XR, self.XW]).tocsr()
        self.X = sp.hstack([self.XF, XC]).tocsr()
        if base is None:      # the FE-block factorisation is shared across specs and outcomes
            F = (self.XF.T @ self.XF).tocsc()
            self.lu = spl.splu(F, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0)
        else:
            self.lu = base.lu
        self.C = (self.XF.T @ XC).toarray()
        self.FiC = self.lu.solve(self.C)
        self.Sc = (XC.T @ XC).toarray() - self.C.T @ self.FiC
        self.use_len = use_len
        self.G = {}
        for nm, g in [("plat", d.source_id), ("genre", d.genre_modal), ("auth", d.author_id)]:
            g = pd.factorize(g)[0]
            D_ = sp.csr_matrix((ones, (ar, g)), shape=(N, g.max() + 1))
            self.G[nm] = (D_, np.bincount(g).astype(float))

    def solve(self, B):
        """S^-1 B for B (nF+k, m)."""
        B = np.asarray(B)
        bF, bC = B[:self.nF], B[self.nF:]
        t = self.lu.solve(np.ascontiguousarray(bF))
        vc = np.linalg.solve(self.Sc, bC - self.C.T @ t)
        return np.vstack([t - self.FiC @ vc, vc])

    def gmean(self, nm, R):
        D_, cnt = self.G[nm]
        return D_ @ ((D_.T @ R) / cnt[:, None])

    def fit(self, y):
        self.y = y.astype(float)
        self.beta = self.solve((self.X.T @ self.y)[:, None])[:, 0]
        self.e = self.y - self.X @ self.beta

    def parts(self, V, comps):
        """Review-level (N x m) parts for param matrix V (q x m)."""
        nF, nR = self.nF, self.nR
        a = self.XR @ V[:nR]
        w = self.XW @ V[nR:nF]
        out = {}
        for c in comps:
            out[c] = {"a": lambda: a, "aP": lambda: self.gmean("plat", a), "w": lambda: w,
                      "wG": lambda: self.gmean("genre", w), "wA": lambda: self.gmean("auth", w),
                      "c": lambda: self.XC @ V[nF:],
                      "cl": lambda: self.XC[:, -1] @ V[-1:]}[c]()
        return out

    def adjoint(self, comp, R):
        """Z_c' R (q x m): Z_c = centred review-level part / sqrt(N)."""
        R = R - R.mean(0, keepdims=True)
        nF, nR = self.nF, self.nR
        out = np.zeros((nF + self.XC.shape[1], R.shape[1]))
        if comp in ("a", "aP"):
            out[:nR] = self.XR.T @ (self.gmean("plat", R) if comp == "aP" else R)
        elif comp in ("w", "wG", "wA"):
            Rg = R if comp == "w" else self.gmean("genre" if comp == "wG" else "auth", R)
            out[nR:nF] = self.XW.T @ Rg
        elif comp == "c":
            out[nF:] = self.XC.T @ R
        elif comp == "cl":
            out[-1:] = self.XC[:, -1].T @ R
        return out / np.sqrt(self.N)


COMPS_A = ["a", "aP", "w", "wG", "wA", "c"]
PAIRS = [("a", "w"), ("a", "c"), ("w", "c")]


def leverage_and_B(m, comps, draws, seed, chunk=30):
    """P_ii and B_ii^(c) (per component and per pair-sum) by JL projections. Depends only on the
    design, so computed once per (sample, spec) and reused for every outcome on that sample."""
    N = m.N
    rng = np.random.default_rng(seed)
    Pii = np.zeros(N)
    for s in range(0, draws, chunk):
        k = min(chunk, draws - s)
        R = rng.choice([-1.0, 1.0], size=(N, k))
        Pii += ((m.X @ m.solve(m.X.T @ R)) ** 2).sum(1)
    Pii /= draws
    B = {c: np.zeros(N) for c in comps}
    B.update({p + "+" + q: np.zeros(N) for p, q in PAIRS})
    for s in range(0, draws, chunk):
        k = min(chunk, draws - s)
        R = rng.choice([-1.0, 1.0], size=(N, k))
        W = {c: m.X @ m.solve(m.adjoint(c, R)) for c in comps}
        for c in comps:
            B[c] += (W[c] ** 2).sum(1)
        for p, q in PAIRS:
            B[p + "+" + q] += ((W[p] + W[q]) ** 2).sum(1)
    for c in B:
        B[c] /= draws
    return Pii, B


def estimate(m, y, comps, Pii, B):
    m.fit(y)
    fp = m.parts(m.beta[:, None], comps)
    var0 = lambda v: ((v - v.mean()) ** 2).mean()
    plug = {c: var0(fp[c][:, 0]) for c in comps}
    for p, q in PAIRS:
        plug[p + "+" + q] = var0(fp[p][:, 0] + fp[q][:, 0])
    plug["resid"] = var0(m.e)
    lo = m.y * m.e / (1 - np.clip(Pii, 0, 0.995))     # y_i times leave-out residual
    corr = {c: plug[c] - (lo * B[c]).sum() for c in B}
    return plug, corr


def shares(vy, v, corrected):
    cov_aw = (v["a+w"] - v["a"] - v["w"]) / 2
    cov_ac = (v["a+c"] - v["a"] - v["c"]) / 2
    cov_wc = (v["w+c"] - v["w"] - v["c"]) / 2
    out = {
        "cov_total": v["c"], "cov_len": v.get("cl", np.nan),
        "reviewer_total": v["a"], "reviewer_between_platform": v["aP"],
        "reviewer_within_platform": v["a"] - v["aP"],
        "work_total": v["w"], "work_between_genre": v["wG"],
        "work_between_author_within_genre": v["wA"] - v["wG"],
        "work_within_author": v["w"] - v["wA"],
        "sorting_2cov_reviewer_work": 2 * cov_aw,
        "cross_cov_x_FE": 2 * cov_ac + 2 * cov_wc,
    }
    tot = sum(out[k] for k in ["cov_total", "reviewer_total", "work_total",
                               "sorting_2cov_reviewer_work", "cross_cov_x_FE"])
    out["residual"] = vy - tot if corrected else v["resid"]
    return {k: x / vy for k, x in out.items()}


def run_group(args):
    """One connected-set sample shared by one or more outcomes."""
    outs, df, draws, seed, minn = args
    t0 = time.time()
    col0 = outs[0][1]
    d0 = df[df[col0].notna()].copy()
    keep = loo_connected(d0)
    d = d0[keep].reset_index(drop=True)
    infos, rows = [], []
    for name, col in outs:
        infos.append(dict(outcome=name, n_available=int(df[col].notna().sum()), n_reviews=len(d),
                          n_reviewers=d.user_id.nunique(), n_works=d.work_id.nunique(),
                          n_authors=d.author_id.nunique(), n_genres=d.genre_modal.nunique(),
                          n_platforms=d.source_id.nunique(), var_y=d[col].var(ddof=0)))
    if len(d) < minn:
        return infos, rows
    if "loglen" in [n for n, _ in outs]:
        d.to_parquet(OUT / "connected_set_loglen.parquet")
    base = None
    for spec in ["A", "B"]:
        m = Model(d, spec == "B", base)
        base = base or m
        comps = COMPS_A + (["cl"] if spec == "B" else [])
        Pii, B = leverage_and_B(m, comps, draws, seed)
        print(f"  group[{outs[0][0]}..] N={m.N} spec={spec} trace(P)={Pii.sum():.0f}/rank {m.X.shape[1]} "
              f"maxP={Pii.max():.3f} P>0.995: {(Pii > 0.995).sum()} ({time.time()-t0:.0f}s)", flush=True)
        for name, col in outs:
            if spec == "B" and name == "loglen":
                continue
            y = d[col].to_numpy(float)
            plug, corr = estimate(m, y, comps, Pii, B)
            meta = dict(N=m.N, var_y=y.var(), trace_P=Pii.sum(), rank=m.X.shape[1], maxP=Pii.max(),
                        meanP=Pii.mean(), n_clipped=int((Pii > 0.995).sum()))
            for which, comp in [("corrected", corr), ("plugin", plug)]:
                rows.append(dict(outcome=name, spec=spec, estimator=which,
                                 **shares(meta["var_y"], comp, which == "corrected"), **meta))
    return infos, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=28)
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-n", type=int, default=3000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df, ginfo = build_base()
    print("base", len(df), ginfo, flush=True)
    pd.Series(ginfo).to_csv(OUT / "genre_assignment_info.csv", header=["value"])
    outs = [o for o in OUTCOMES if not a.only or o[0] in a.only.split(",")]
    # outcomes with an identical availability mask share one connected set
    groups = {}
    for n, c in outs:
        groups.setdefault(df[c].notna().to_numpy().tobytes(), []).append((n, c))
    groups = sorted(groups.values(), key=lambda g: -df[g[0][1]].notna().sum())
    print("sample groups:", [[n for n, _ in g] for g in groups], flush=True)
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    infos, rows = [], []
    jobs = [(g, df, a.draws, a.seed, a.min_n) for g in groups]
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, r in ex.map(run_group, jobs):
            infos += i
            rows += r
            print(f"done {[x['outcome'] for x in i]} n={i[0]['n_reviews']} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"decomposition_long{a.tag}.csv", index=False)
            pd.DataFrame(infos).to_csv(OUT / f"sample_sizes{a.tag}.csv", index=False)
    print("total", time.time() - t0)


if __name__ == "__main__":
    main()
