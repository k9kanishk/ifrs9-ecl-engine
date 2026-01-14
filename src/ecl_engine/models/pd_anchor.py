from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecl_engine.ecl import (
    _logit,
    _sigmoid,
    load_yml,
    month_end_index,
    prepare_macro_z_all_scenarios,
)


def _macro_block_ffill(mz_scen: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    dates = pd.to_datetime(dates)
    idx = mz_scen.index.union(dates)
    df2 = mz_scen.reindex(idx).sort_index().ffill().bfill()
    return df2.loc[dates, ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]].to_numpy(
        dtype=np.float64
    )


def cum_pd12_from_shift(
    logit_ttc: np.ndarray,  # (N,)
    shift: np.ndarray,  # (N,)
    macro_linear: np.ndarray,  # (12,)
    pd_floor: float,
    pd_cap: float,
) -> np.ndarray:
    """
    Build annual PD_t (t=1..12) = sigmoid(logit_ttc + shift + macro_linear[t])
    Convert to monthly hazard and compute 12m cumulative default probability.
    """
    z = (logit_ttc[:, None] + shift[:, None] + macro_linear[None, :]).astype(
        np.float64
    )  # (N,12)
    pd_annual = np.clip(_sigmoid(z), pd_floor, pd_cap)  # (N,12)
    h = 1.0 - np.power(1.0 - pd_annual, 1.0 / 12.0)  # monthly hazard
    cum = 1.0 - np.prod(1.0 - h, axis=1)
    return cum


def compute_anchor_shift(
    asof: str | None = None,
    accounts_path: str = "data/curated/accounts.parquet",
    macro_path: str = "data/curated/macro_scenarios_monthly.parquet",
    pd_scores_path: str | None = None,
    policy_path: str = "configs/policy.yml",
    outdir: str = "data/curated",
    iters: int = 30,
    lo: float = -10.0,
    hi: float = 10.0,
) -> Path:
    acc = pd.read_parquet(accounts_path)
    policy = load_yml(policy_path)

    if "ttc_pd_annual" not in acc.columns:
        raise ValueError("accounts.parquet must contain ttc_pd_annual for anchor calibration.")

    macro = pd.read_parquet(macro_path)
    macro["date"] = pd.to_datetime(macro["date"])
    mz = prepare_macro_z_all_scenarios(macro).set_index(["scenario", "date"]).sort_index()

    if asof:
        asof_dt = pd.to_datetime(asof) + pd.offsets.MonthEnd(0)
    else:
        ecl_path = Path("data/curated/ecl_with_overlays.parquet")
        if ecl_path.exists():
            ecl = pd.read_parquet(ecl_path)
            ecl["asof_date"] = pd.to_datetime(ecl["asof_date"])
            asof_dt = ecl["asof_date"].max()
        else:
            asof_dt = macro["date"].max()

    if pd_scores_path is None:
        pd_scores_path = f"data/curated/pd_scores_asof_{asof_dt.date().isoformat()}.parquet"
    pd_scores_path = str(pd_scores_path)

    scores = pd.read_parquet(pd_scores_path)
    if "pd_12m_hat" not in scores.columns:
        raise ValueError("pd_scores file must contain pd_12m_hat.")

    a = acc.merge(scores[["account_id", "pd_12m_hat"]], on="account_id", how="left")
    a = a[a["pd_12m_hat"].notna()].copy()

    target = a["pd_12m_hat"].astype(float).to_numpy()
    target = np.clip(target, 1e-6, 0.999)

    ttc = np.clip(a["ttc_pd_annual"].astype(float).to_numpy(), 1e-6, 0.999)
    logit_ttc = _logit(ttc).astype(np.float64)

    future_dates = month_end_index((asof_dt + pd.offsets.MonthEnd(0)).normalize(), 12)
    mz_base = mz.xs("Base", level=0).sort_index()
    m_base = _macro_block_ffill(mz_base, future_dates)  # (12,3)

    betas = policy["default"]["pd_logit"]
    pd_floor = float(policy["default"]["pd_floor"])
    pd_cap = float(policy["default"]["pd_cap"])

    macro_linear = (
        float(betas["intercept"])
        + float(betas["unemployment_z"]) * m_base[:, 0]
        + float(betas["gdp_yoy_z"]) * m_base[:, 1]
        + float(betas["policy_rate_z"]) * m_base[:, 2]
    ).astype(np.float64)  # (12,)

    low = np.full_like(target, lo, dtype=np.float64)
    high = np.full_like(target, hi, dtype=np.float64)

    for _ in range(iters):
        mid = 0.5 * (low + high)
        cum_mid = cum_pd12_from_shift(logit_ttc, mid, macro_linear, pd_floor, pd_cap)
        low = np.where(cum_mid < target, mid, low)
        high = np.where(cum_mid >= target, mid, high)

    shift = 0.5 * (low + high)

    out = pd.DataFrame(
        {
            "account_id": a["account_id"].values,
            "asof_date": asof_dt,
            "pd_12m_hat": target,
            "pd_anchor_shift": shift,
        }
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"pd_anchor_shift_asof_{asof_dt.date().isoformat()}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} ({out.shape[0]:,} rows)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", type=str, default=None)
    args = ap.parse_args()
    compute_anchor_shift(asof=args.asof)


if __name__ == "__main__":
    main()
