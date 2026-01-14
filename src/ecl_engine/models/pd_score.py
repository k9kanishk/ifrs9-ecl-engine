from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ecl_engine.models.pd_train import _month_end, _pick_col


def score_asof(
    asof: str | None = None,
    model_path: str = "models/pd_logit.joblib",
    schema_path: str = "models/pd_feature_schema.json",
    accounts_path: str = "data/curated/accounts.parquet",
    perf_path: str = "data/curated/performance_monthly.parquet",
    staging_path: str = "data/curated/staging_output.parquet",
    outdir: str = "data/curated",
) -> Path:
    pipe = joblib.load(model_path)
    schema = json.loads(Path(schema_path).read_text())
    cat_cols = schema["cat_cols"]
    num_cols = schema["num_cols"]
    citl_delta = float(schema.get("citl_delta", 0.0))

    acc = pd.read_parquet(accounts_path)
    perf = pd.read_parquet(perf_path)
    stg = pd.read_parquet(staging_path)

    perf["snapshot_date"] = _month_end(perf["snapshot_date"])
    stg["snapshot_date"] = _month_end(stg["snapshot_date"])

    asof_dt = pd.to_datetime(asof) if asof else stg["snapshot_date"].max()

    # Detect columns
    seg_acc = _pick_col(acc.columns.tolist(), ["segment", "product_segment", "portfolio", "seg"])
    bal_col = _pick_col(perf.columns.tolist(), ["balance", "outstanding_balance", "bal"])
    dpd_col = _pick_col(perf.columns.tolist(), ["dpd", "days_past_due", "dpd_days", "delinquency_days"])
    limit_col = _pick_col(acc.columns.tolist(), ["limit_amount", "credit_limit", "limit"])
    ttc_col = _pick_col(acc.columns.tolist(), ["ttc_pd_annual", "ttc_pd", "pd_ttc"])

    if seg_acc is None or bal_col is None:
        raise ValueError("Missing required columns for scoring (segment in accounts, balance in perf).")

    # Build ASOF feature frame
    p = perf[perf["snapshot_date"] == asof_dt].copy()
    s = stg[stg["snapshot_date"] == asof_dt][["account_id", "stage"]].copy()

    x = p[["account_id", bal_col]].rename(columns={bal_col: "balance"}).merge(
        s, on="account_id", how="left"
    )
    x = x.merge(
        acc[["account_id", seg_acc]].rename(columns={seg_acc: "segment"}),
        on="account_id",
        how="left",
    )

    if dpd_col and dpd_col in p.columns:
        x = x.merge(
            p[["account_id", dpd_col]].rename(columns={dpd_col: "dpd"}),
            on="account_id",
            how="left",
        )
    else:
        x["dpd"] = 0.0

    if limit_col:
        x = x.merge(
            acc[["account_id", limit_col]].rename(columns={limit_col: "limit_amount"}),
            on="account_id",
            how="left",
        )
    else:
        x["limit_amount"] = np.nan

    if ttc_col:
        x = x.merge(
            acc[["account_id", ttc_col]].rename(columns={ttc_col: "ttc_pd_annual"}),
            on="account_id",
            how="left",
        )
    else:
        x["ttc_pd_annual"] = np.nan

    # Remove stage 3 from scoring universe (optional)
    x = x[x["stage"].fillna(1).astype(int) != 3].copy()

    # Feature engineering consistent with training
    x["utilization"] = np.where(
        x["limit_amount"].notna() & (x["limit_amount"] > 0),
        x["balance"] / x["limit_amount"],
        np.nan,
    )
    x["log_balance"] = np.log1p(np.maximum(x["balance"], 0.0))
    x["dpd_bucket"] = pd.cut(
        x["dpd"].fillna(0.0),
        bins=[-0.1, 0.1, 30, 60, 90, 10_000],
        labels=["0", "1_30", "31_60", "61_90", "90p"],
        include_lowest=True,
    ).astype(str)

    # Build model input
    X = x[cat_cols + num_cols].copy()
    for c in num_cols:
        X[c] = X[c].astype(float)
    # Fill missing numerics with 0 (safe at scoring time)
    X[num_cols] = X[num_cols].fillna(0.0)

    pd_hat = pipe.predict_proba(X)[:, 1]

    def logit(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    pd_hat = sigmoid(logit(pd_hat) + citl_delta)
    out = x[["account_id", "segment"]].copy()
    out["asof_date"] = asof_dt
    out["pd_12m_hat"] = pd_hat.astype(float)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"pd_scores_asof_{asof_dt.date().isoformat()}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} ({out.shape[0]:,} rows)")
    return out_path


if __name__ == "__main__":
    score_asof()
