from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from ecl_engine.ecl import prepare_macro_z_all_scenarios
from ecl_engine.utils.dates import month_end_index
from ecl_engine.utils.math import logit, sigmoid
from ecl_engine.utils.io import load_yml


def _macro_block_ffill(mz_scen: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return macro z columns aligned to dates, ffill/bfill beyond available history."""
    dates = pd.to_datetime(dates)
    idx = mz_scen.index.union(dates)
    df2 = mz_scen.reindex(idx).sort_index().ffill().bfill()
    return df2.loc[dates, ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]].copy()


def compute_scenario_qc(
    asof: str | None = None,
    horizon_months: int = 12,
    macro_path: str = "data/curated/macro_scenarios_monthly.parquet",
    staging_path: str = "data/curated/staging_output.parquet",
    accounts_path: str = "data/curated/accounts.parquet",
    pd_scores_path: str | None = None,
    policy_path: str = "configs/policy.yml",
    outdir: str = "data/curated",
) -> tuple[pd.Path, pd.Path]:
    """
    Produces:
      1) scenario_severity_asof_<date>.csv  : macro z severity + implied PIT PD summaries by scenario
      2) scenario_pd_summary_asof_<date>.csv: PIT PD (m1, avg12, cum12) distributions by scenario (Stage 1/2 only)
    """

    policy = load_yml(policy_path)

    # ASOF determination: prefer staging max date
    stg = pd.read_parquet(staging_path, columns=["snapshot_date", "account_id", "stage"])
    stg["snapshot_date"] = pd.to_datetime(stg["snapshot_date"]) + pd.offsets.MonthEnd(0)

    asof_dt = (pd.to_datetime(asof) + pd.offsets.MonthEnd(0)) if asof else stg["snapshot_date"].max()

    # future months
    future_dates = month_end_index((asof_dt + pd.offsets.MonthEnd(0)).normalize(), horizon_months)

    # macro (z-scored inside helper)
    macro = pd.read_parquet(macro_path)
    macro["date"] = pd.to_datetime(macro["date"]) + pd.offsets.MonthEnd(0)

    mz = prepare_macro_z_all_scenarios(macro).set_index(["scenario", "date"]).sort_index()

    # betas + macro scaling (Phase 3.2)
    betas = policy["default"]["pd_logit"]
    macro_scale = float(policy["default"].get("macro_scale", 1.0))
    pd_floor = float(policy["default"]["pd_floor"])
    pd_cap = float(policy["default"]["pd_cap"])

    # accounts + (optional) pd_scores + (optional) anchor shift
    acc = pd.read_parquet(accounts_path)
    if "ttc_pd_annual" not in acc.columns:
        raise ValueError("accounts.parquet must include ttc_pd_annual")

    # bring stage at asof for filtering
    stg_asof = stg[stg["snapshot_date"] == asof_dt].copy()
    stg_asof["stage"] = stg_asof["stage"].astype(int)

    a = acc.merge(stg_asof[["account_id", "stage"]], on="account_id", how="left")
    a["stage"] = a["stage"].fillna(1).astype(int)

    # bring fitted pd_12m_hat if exists
    if pd_scores_path is None:
        pd_scores_path = f"data/curated/pd_scores_asof_{asof_dt.date().isoformat()}.parquet"
    pd_scores_path = Path(pd_scores_path)

    if pd_scores_path.exists():
        s = pd.read_parquet(pd_scores_path)
        if "pd_12m_hat" in s.columns:
            a = a.merge(s[["account_id", "pd_12m_hat"]], on="account_id", how="left")

    # bring anchor shift if exists
    shift_path = Path(f"data/curated/pd_anchor_shift_asof_{asof_dt.date().isoformat()}.parquet")
    if shift_path.exists():
        sh = pd.read_parquet(shift_path)
        a = a.merge(sh[["account_id", "pd_anchor_shift"]], on="account_id", how="left")
    else:
        a["pd_anchor_shift"] = 0.0

    # filter scoring universe: Stage 1+2 only (avoid defaulted)
    a = a[a["stage"].isin([1, 2])].copy()

    # base logits
    ttc = np.clip(a["ttc_pd_annual"].astype(float).to_numpy(), 1e-9, 1 - 1e-9)
    logit_ttc = logit(ttc).astype(np.float64)
    logit_ttc = logit_ttc + a["pd_anchor_shift"].fillna(0.0).to_numpy(dtype=np.float64)

    def pit_pd_summary_for_scenario(scen: str) -> pd.DataFrame:
        mz_s = mz.xs(scen, level=0).sort_index()
        m = _macro_block_ffill(mz_s, future_dates)

        z = (
            float(betas["intercept"])
            + float(betas["unemployment_z"]) * m["unemployment_z"].to_numpy(dtype=np.float64)
            + float(betas["gdp_yoy_z"]) * m["gdp_yoy_z"].to_numpy(dtype=np.float64)
            + float(betas["policy_rate_z"]) * m["policy_rate_z"].to_numpy(dtype=np.float64)
        )
        z = macro_scale * z  # <-- key governance knob

        # (N, H) annual PDs
        pd_annual = np.clip(sigmoid(logit_ttc[:, None] + z[None, :]), pd_floor, pd_cap).astype(np.float64)

        # monthly hazard and 12m cumulative PD
        h = 1.0 - np.power(1.0 - pd_annual, 1.0 / 12.0)
        cum12 = 1.0 - np.prod(1.0 - h, axis=1)

        out = pd.DataFrame(
            {
                "scenario": scen,
                "pd_m1": pd_annual[:, 0],
                "pd_avg12": pd_annual.mean(axis=1),
                "pd_cum12": cum12,
            }
        )
        return out

    # 1) Macro severity summary by scenario
    sev_rows = []
    for scen in ["Base", "Upside", "Downside"]:
        mz_s = mz.xs(scen, level=0).sort_index()
        m = _macro_block_ffill(mz_s, future_dates)

        sev_rows.append(
            {
                "asof_date": asof_dt.date().isoformat(),
                "scenario": scen,
                "macro_scale": macro_scale,
                "unemp_z_mean": float(m["unemployment_z"].mean()),
                "unemp_z_min": float(m["unemployment_z"].min()),
                "unemp_z_max": float(m["unemployment_z"].max()),
                "gdp_z_mean": float(m["gdp_yoy_z"].mean()),
                "gdp_z_min": float(m["gdp_yoy_z"].min()),
                "gdp_z_max": float(m["gdp_yoy_z"].max()),
                "rate_z_mean": float(m["policy_rate_z"].mean()),
                "rate_z_min": float(m["policy_rate_z"].min()),
                "rate_z_max": float(m["policy_rate_z"].max()),
            }
        )

    sev = pd.DataFrame(sev_rows)

    # 2) PIT PD distributions by scenario
    pd_all = pd.concat([pit_pd_summary_for_scenario(s) for s in ["Base", "Upside", "Downside"]], axis=0, ignore_index=True)

    # add percentiles for reporting
    def pct_summary(df: pd.DataFrame, col: str) -> dict:
        return {
            f"{col}_mean": float(df[col].mean()),
            f"{col}_p50": float(df[col].quantile(0.50)),
            f"{col}_p90": float(df[col].quantile(0.90)),
            f"{col}_p99": float(df[col].quantile(0.99)),
            f"{col}_max": float(df[col].max()),
        }

    rows = []
    for scen in ["Base", "Upside", "Downside"]:
        d = pd_all[pd_all["scenario"] == scen].copy()
        r = {"asof_date": asof_dt.date().isoformat(), "scenario": scen, "n_accounts": int(len(d))}
        r.update(pct_summary(d, "pd_m1"))
        r.update(pct_summary(d, "pd_avg12"))
        r.update(pct_summary(d, "pd_cum12"))
        rows.append(r)

    pd_sum = pd.DataFrame(rows)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sev_path = outdir / f"scenario_severity_asof_{asof_dt.date().isoformat()}.csv"
    pd_path = outdir / f"scenario_pd_summary_asof_{asof_dt.date().isoformat()}.csv"

    sev.to_csv(sev_path, index=False)
    pd_sum.to_csv(pd_path, index=False)

    print(f"Wrote: {sev_path}")
    print(f"Wrote: {pd_path}")
    return sev_path, pd_path


if __name__ == "__main__":
    compute_scenario_qc()
