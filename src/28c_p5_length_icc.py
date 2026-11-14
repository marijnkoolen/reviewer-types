"""P5: how much of the reviewer ICC of review length (0.61 in report section 11c) is platform?
One-way ICC(1) (same estimator as src/24_contiguity_as_trait.py, imported) of num_sentences and
log(num_sentences): raw, after removing platform means, after platform + genre. Two samples:
(i) the section-11c sample (eligible reviews, >=10 per reviewer, platform_1 excluded);
(ii) all non-platform_1 reviews of reviewers with >=2 reviews. Writes results/28_variance_decomposition/p5_length_icc.csv"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("m24", ROOT / "src/24_contiguity_as_trait.py")
m24 = importlib.util.module_from_spec(spec)
import sys
sys.argv = [sys.argv[0], "silver"]
spec.loader.exec_module(m24)

rf = pd.read_parquet(ROOT / "results/00_data_prep/silver_review_features.parquet")
rf = rf[rf.source_id != "platform_1"].copy()
rf["loglen"] = np.log(rf.num_sentences)
d11 = m24.load().rename(columns={"n_sentences": "num_sentences"})   # section 11c sample
d2 = rf[rf.groupby("user_id").user_id.transform("size") >= 2]
rows = []
for sname, d in [("11c sample (eligible, >=10/reviewer)", d11), (">=2 reviews/reviewer, all", d2)]:
    d = d[["user_id", "num_sentences", "nur_genre", "source_id"]].copy()
    d["loglen"] = np.log(d.num_sentences)
    for col in ["num_sentences", "loglen"]:
        for lab, rhs in [("raw", None), ("minus platform", "C(source_id)"),
                         ("minus platform+genre", "C(source_id) + C(nur_genre)")]:
            if rhs is None:
                v = d[col]
            else:
                v = pd.Series(smf.ols(f"{col} ~ {rhs}", d).fit().resid.to_numpy(), index=d.index)
            rows.append(dict(sample=sname, outcome=col, adjustment=lab, n_reviews=len(d),
                             n_reviewers=d.user_id.nunique(),
                             icc=round(m24.icc1(v.to_numpy(), d.user_id.to_numpy()), 4)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/28_variance_decomposition/p5_length_icc.csv", index=False)
print(out.to_string(index=False))
