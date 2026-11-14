"""Cross-check follow-up: refit the KSS fixed-effects decomposition on exactly the 25K-review random
subsample used for the NUTS random-effects fit (same seed/rows), for the 4 cross-check outcomes.
The FE needs reviewers/works with >=2 reviews and leave-one-out connectivity, so it keeps only the
connected part of the subsample (size reported). Also reports the RE-style quantity on that same part
is NOT refit; compare with crosscheck_random_effects.csv. Writes crosscheck_fe_on_subsample.csv"""
import importlib
import pandas as pd

M = importlib.import_module("28c_variance_decomposition")
cs = pd.read_parquet(M.OUT / "connected_set_loglen.parquet")
cols = {"prop__Content": "prop__Content", "prop__Reader_response": "prop__Reader_response",
        "loglen": "loglen", "contig_excess": "excess_over_chance"}
rows = []
for name, col in cols.items():
    d0 = cs[cs[col].notna()].sample(25000, random_state=7).reset_index(drop=True)
    d = d0[M.loo_connected(d0)].reset_index(drop=True)
    m = M.Model(d, False)
    Pii, B = M.leverage_and_B(m, M.COMPS_A, 600, 3)
    y = d[col].to_numpy(float)
    plug, corr = M.estimate(m, y, M.COMPS_A, Pii, B)
    sh = M.shares(y.var(), corr, True)
    shp = M.shares(y.var(), plug, False)
    rows.append(dict(outcome=name, n_subsample=len(d0), n_connected=len(d), reviewers=d.user_id.nunique(),
                     works=d.work_id.nunique(), fe_reviewer=sh["reviewer_total"], fe_work=sh["work_total"],
                     fe_sorting=sh["sorting_2cov_reviewer_work"], fe_year=sh["cov_total"],
                     fe_residual=sh["residual"], plug_reviewer=shp["reviewer_total"], plug_work=shp["work_total"]))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(M.OUT / "crosscheck_fe_on_subsample.csv", index=False)
