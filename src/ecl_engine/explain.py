from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Reuse ECL helpers where possible
from ecl_engine.ecl import load_yml, month_end_index, prepare_macro_z_all_scenarios, _logit, _sigmoid


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _macro_block_ffill(mz_scen: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    mz_scen: DataFrame indexed by date with columns unemployment_z, gdp_yoy_z, policy_rate_z
    Returns (H,3) macro z block aligned to dates, forward-filled beyond last known macro date.
    """
    dates = pd.to_datetime(dates)
    idx = mz_scen.index.union(dates)
    df2 = mz_scen.reindex(idx).sort_index().ffill().bfill()
    return df2.loc[dates, ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]].to_numpy(dtype=np.float32)


def months_to_maturity(asof: pd.Timestamp, maturity: pd.Series) -> np.ndarray:
    asof_me = (asof + pd.offsets.MonthEnd(0)).normalize()
    mat_me = (pd.to_datetime(maturity) + pd.offsets.MonthEnd(0)).dt.normalize()
    m = (mat_me.dt.year - asof_me.year) * 12 + (mat_me.dt.month - asof_me.month)
    m = m.astype(int).to_numpy()
    return np.maximum(m, 1)


def stage_reason_text(stage: int, dpd: float | None, default_dpd: int = 90, sicr_dpd: int = 30) -> str:
    if stage == 3:
        if dpd is not None and dpd >= default_dpd:
            return f"Stage 3: Default / credit-impaired (DPD ≥ {default_dpd})"
        return "Stage 3: Default / credit-impaired (trigger not shown in drilldown)"
    if stage == 2:
        if dpd is not None and dpd >= sicr_dpd:
            return f"Stage 2: SICR (DPD backstop ≥ {sicr_dpd})"
        return "Stage 2: SICR (PD deterioration trigger / qualitative SICR)"
    return "Stage 1: Performing (no SICR / default trigger)"


def build_account_explain(
    asof: str | None = None,
    ecl_with_overlays_path: str = "data/curated/ecl_with_overlays.parquet",
    accounts_path: str = "data/curated/accounts.parquet",
    staging_path: str = "data/curated/staging_output.parquet",
    perf_path: str = "data/curated/performance_monthly.parquet",
    macro_path: str = "data/curated/macro_scenarios_monthly.parquet",
    policy_path: str = "configs/policy.yml",
    weights_path: str = "configs/scenario_weights.yml",
    params_path: str = "configs/portfolio_params.yml",
    outdir: str = "data/curated",
) -> Path:
    ecl = pd.read_parquet(ecl_with_overlays_path)
    ecl["asof_date"] = pd.to_datetime(ecl["asof_date"])
    asof_dt = pd.to_datetime(asof) if asof else ecl["asof_date"].max()

    # Filter to ASOF
    e = ecl[ecl["asof_date"] == asof_dt].copy()

    acc = pd.read_parquet(accounts_path)
    stg = pd.read_parquet(staging_path)
    stg["snapshot_date"] = pd.to_datetime(stg["snapshot_date"])
    stg_asof = stg[stg["snapshot_date"] == asof_dt][["account_id", "segment", "stage"]].copy()

    # Join account attributes we care about
    acc_cols = ["account_id", "segment", "eir", "maturity_date", "limit_amount", "ttc_pd_annual"]
    acc2 = acc[acc_cols].copy()
    acc2["maturity_date"] = pd.to_datetime(acc2["maturity_date"])

    # Pull perf metrics at ASOF (to explain staging triggers)
    perf = pd.read_parquet(perf_path)
    perf["snapshot_date"] = pd.to_datetime(perf["snapshot_date"])
    p = perf[perf["snapshot_date"] == asof_dt].copy()

    dpd_col = _pick_col(p, ["dpd", "days_past_due", "delinquency_days", "dpd_days"])
    bal_col = _pick_col(p, ["balance", "outstanding_balance", "bal"])

    # Base join key is account_id + segment
    base = (
        e.merge(stg_asof, on=["account_id", "segment"], how="left", suffixes=("", "_stg"))
        .merge(acc2, on=["account_id", "segment"], how="left")
    )

    # Ensure stage is present
    if "stage" not in base.columns or base["stage"].isna().any():
        # fallback: use ecl stage column if exists
        if "stage" in e.columns:
            base["stage"] = base["stage"].fillna(e["stage"])
    base["stage"] = base["stage"].astype(int)

    # Bring dpd into explain if available
    if dpd_col:
        base = base.merge(
            p[["account_id", "segment", dpd_col]].rename(columns={dpd_col: "dpd"}),
            on=["account_id", "segment"],
            how="left",
        )
    else:
        base["dpd"] = np.nan

    # Ensure balance in explain (from ECL output)
    # Your ECL output already has "balance". If not, fallback to perf balance.
    if "balance" not in base.columns or base["balance"].isna().all():
        if bal_col:
            base = base.merge(
                p[["account_id", "segment", bal_col]].rename(columns={bal_col: "balance"}),
                on=["account_id", "segment"],
                how="left",
            )

    # Load configs
    policy = load_yml(policy_path)
    weights = load_yml(weights_path)
    params = load_yml(params_path)

    # Staging thresholds (fallback defaults if not present)
    default_dpd = int(policy.get("staging", {}).get("default_dpd", 90))
    sicr_dpd = int(policy.get("staging", {}).get("sicr_dpd", 30))

    # Months to maturity
    base["months_to_maturity"] = months_to_maturity(asof_dt, base["maturity_date"]).astype(int)

    # EAD explanation (simple but transparent; mirrors engine)
    seg = base["segment"].astype(str).to_numpy()
    bal0 = base["balance"].astype(float).to_numpy()
    limit = base["limit_amount"].fillna(0.0).astype(float).to_numpy()
    is_rev = seg == "Revolving"

    amort_map = params.get("amort_rate_monthly", {})
    amort = np.array([float(amort_map.get(s, 0.01)) for s in seg], dtype=np.float64)

    ccf = float(params.get("ccf_base", {}).get("Revolving", 0.75))
    ead0 = bal0.copy()
    ead0[is_rev] = bal0[is_rev] + ccf * np.maximum(limit[is_rev] - bal0[is_rev], 0.0)

    base["ead0"] = ead0
    base["ead_rule"] = np.where(is_rev, f"Revolving: balance + {ccf:.2f}×undrawn", "Amortising: projected balance")

    # Approx avg EAD over next 12m (amortising); revolving constant
    m2m = base["months_to_maturity"].to_numpy()
    H = np.minimum(12, m2m)
    ead12_avg = []
    for b, a, h, rev in zip(bal0, amort, H, is_rev):
        if rev:
            ead12_avg.append(float(b + ccf * max(0.0, 0.0)))  # not used; keep simple
        else:
            # mean of balance path over months 1..h
            ts = np.arange(1, int(h) + 1)
            path = b * (1.0 - a) ** ts
            ead12_avg.append(float(path.mean()) if len(path) else float(b))
    base["ead12_avg"] = np.array(ead12_avg, dtype=np.float64)

    # LGD explanation
    lgd_base_map = params.get("lgd_base", {})
    lgd_base = np.array([float(lgd_base_map.get(s, 0.5)) for s in seg], dtype=np.float64)
    base["lgd_base"] = np.clip(lgd_base, 0, 1)

    mults = params.get("lgd_scenario_multiplier", {"Base": 1.0, "Upside": 1.0, "Downside": 1.0})
    base["lgd_base_scen"] = base["lgd_base"] * float(mults.get("Base", 1.0))
    base["lgd_upside_scen"] = base["lgd_base"] * float(mults.get("Upside", 1.0))
    base["lgd_downside_scen"] = base["lgd_base"] * float(mults.get("Downside", 1.0))

    # PD explanation (PIT term structure summary for next 12 months)
    macro = pd.read_parquet(macro_path)
    macro["date"] = pd.to_datetime(macro["date"])
    mz = prepare_macro_z_all_scenarios(macro)

    betas = policy["default"]["pd_logit"]
    pd_floor = float(policy["default"]["pd_floor"])
    pd_cap = float(policy["default"]["pd_cap"])

    future_dates = month_end_index((asof_dt + pd.offsets.MonthEnd(0)).normalize(), 12)

    mz = mz.set_index(["scenario", "date"]).sort_index()
    # Build per-scenario date-indexed blocks with ffill
    mz_base = mz.xs("Base", level=0).sort_index()
    mz_up = mz.xs("Upside", level=0).sort_index()
    mz_dn = mz.xs("Downside", level=0).sort_index()

    m_base = _macro_block_ffill(mz_base, future_dates)  # (12,3)
    m_up = _macro_block_ffill(mz_up, future_dates)
    m_dn = _macro_block_ffill(mz_dn, future_dates)

    def pit_pd_metrics(m: np.ndarray, ttc_pd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = (
            float(betas["intercept"])
            + float(betas["unemployment_z"]) * m[:, 0]
            + float(betas["gdp_yoy_z"]) * m[:, 1]
            + float(betas["policy_rate_z"]) * m[:, 2]
        ).astype(np.float64)[None, :]  # (1,12)

        logit_ttc = _logit(np.clip(ttc_pd, 1e-12, 1 - 1e-12)).astype(np.float64)[:, None]
        pd_annual = np.clip(_sigmoid(logit_ttc + x), pd_floor, pd_cap).astype(np.float64)  # (n,12)

        # Convert annual PD to monthly hazard
        h = 1.0 - np.power((1.0 - pd_annual), 1.0 / 12.0)
        cum12 = 1.0 - np.prod(1.0 - h, axis=1)  # approx 12m cumulative default prob

        m1 = pd_annual[:, 0]
        avg12 = pd_annual.mean(axis=1)
        return m1, avg12, cum12

    ttc = base["ttc_pd_annual"].astype(float).to_numpy()

    base_m1, base_avg, base_cum = pit_pd_metrics(m_base, ttc)
    up_m1, up_avg, up_cum = pit_pd_metrics(m_up, ttc)
    dn_m1, dn_avg, dn_cum = pit_pd_metrics(m_dn, ttc)

    base["ttc_pd_annual"] = ttc
    base["pit_pd_m1_base"] = base_m1
    base["pit_pd_avg12_base"] = base_avg
    base["pit_cum_pd12_base"] = base_cum

    base["pit_pd_m1_upside"] = up_m1
    base["pit_pd_avg12_upside"] = up_avg
    base["pit_cum_pd12_upside"] = up_cum

    base["pit_pd_m1_downside"] = dn_m1
    base["pit_pd_avg12_downside"] = dn_avg
    base["pit_cum_pd12_downside"] = dn_cum

    # Stage reason (simple interpretation)
    dpd_vals = base["dpd"].where(~base["dpd"].isna(), np.nan).to_numpy()
    reasons = []
    for stg_i, dpd_i in zip(base["stage"].to_numpy(), dpd_vals):
        dpd_val = None if np.isnan(dpd_i) else float(dpd_i)
        reasons.append(stage_reason_text(int(stg_i), dpd_val, default_dpd=default_dpd, sicr_dpd=sicr_dpd))
    base["stage_reason"] = reasons

    # Add final reporting columns (these should exist from ecl_with_overlays)
    keep = [
        "asof_date",
        "account_id",
        "segment",
        "stage",
        "stage_reason",
        "dpd",
        "balance",
        "months_to_maturity",
        "eir",
        "ead_rule",
        "ead0",
        "ead12_avg",
        "ttc_pd_annual",
        "pit_pd_m1_base",
        "pit_pd_avg12_base",
        "pit_cum_pd12_base",
        "pit_pd_m1_upside",
        "pit_pd_avg12_upside",
        "pit_cum_pd12_upside",
        "pit_pd_m1_downside",
        "pit_pd_avg12_downside",
        "pit_cum_pd12_downside",
        "lgd_base",
        "lgd_base_scen",
        "lgd_upside_scen",
        "lgd_downside_scen",
        # reporting
        "ecl_pre_overlay",
        "overlay_amount",
        "ecl_post_overlay",
        "overlay_audit",
        "ecl12_weighted",
        "ecllt_weighted",
        "ecl_selected",
        # scenario ECL already in output
        "ecl12_base",
        "ecl12_upside",
        "ecl12_downside",
        "ecllt_base",
        "ecllt_upside",
        "ecllt_downside",
    ]
    keep = [c for c in keep if c in base.columns]
    out = base[keep].copy()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"account_explain_asof_{asof_dt.date().isoformat()}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} ({out.shape[0]:,} rows)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", type=str, default=None)
    args = ap.parse_args()
    build_account_explain(asof=args.asof)


if __name__ == "__main__":
    main()
